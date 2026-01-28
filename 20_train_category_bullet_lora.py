"""
20_train_category_bullet_lora.py
역할:
  - 19_prepare_category_bullet_sft.py 로 만든 JSONL(SFT)을 이용해
    Qwen2.5 Instruct 계열 CausalLM을 LoRA(PEFT)로 미세튜닝한다.

필요 패키지:
  - transformers / datasets / accelerate (이미 requirements.txt에 있음)
  - peft  (추가 설치 필요)

설치 예:
  pip install peft

실행 예:
  python 19_prepare_category_bullet_sft.py --input_csv products_all.csv --out_jsonl training_category_bullets_sft.jsonl
  python 20_train_category_bullet_lora.py --train_jsonl training_category_bullets_sft.jsonl --out_dir results_bullets/qwen2.5-3b-lora
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# 프록시 설정 제거 (Hugging Face 다운로드 문제 해결)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Hugging Face 오프라인 모드 활성화 (로컬 캐시 우선 사용)
os.environ["HF_HUB_OFFLINE"] = "0"  # 오프라인 모드는 비활성화 (캐시는 자동 사용)

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model  # noqa: F401
        return
    except Exception as e:
        raise RuntimeError(
            "peft 패키지가 필요합니다. 설치 후 다시 실행하세요: pip install peft\n"
            f"원인: {e}"
        )


def _build_train_example(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """
    example 형식:
      {"messages":[... system, user, assistant ...]}

    학습 형태:
      prompt(assistant 응답 유도) + answer(JSON)
    라벨은 prompt 구간은 -100 처리하여 손실 계산 제외.
    """
    msgs: List[Dict[str, str]] = example["messages"]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        raise ValueError("messages 포맷이 올바르지 않습니다(assistant 응답 누락).")

    prompt_msgs = msgs[:-1]
    answer = msgs[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + answer

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False,
    )

    labels = full["input_ids"].copy()
    # prompt 구간 마스킹
    prompt_len = min(len(prompt_ids), max_length)
    for i in range(prompt_len):
        labels[i] = -100
    # padding 마스킹
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels = [(-100 if t == pad_id else t) for t in labels]

    full["labels"] = labels
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_jsonl", default="training_category_bullets_sft.jsonl")
    ap.add_argument("--out_dir", default="results_bullets/qwen2.5-3b-lora-category-bullets")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    args = ap.parse_args()

    _require_peft()
    from peft import LoraConfig, get_peft_model  # noqa: E402

    print("=" * 60)
    print("카테고리 요약 불릿 LoRA 학습")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Train JSONL: {args.train_jsonl}")
    print(f"Out dir: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\n토크나이저 로딩 중...")
    # 이전 학습에서 저장된 토크나이저가 있으면 사용 (프록시 문제 회피)
    prev_model_dir = "results_bullets/qwen2.5-3b-lora-category-bullets-v3"
    tokenizer_path = os.path.join(prev_model_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path) or os.path.exists(os.path.join(prev_model_dir, "tokenizer_config.json")):
        print(f"이전 학습에서 저장된 토크나이저 사용: {prev_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(prev_model_dir, use_fast=True)
    else:
        print(f"베이스 모델에서 토크나이저 로드: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # 일부 모델은 pad_token이 없을 수 있음
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n모델 로딩 중... (시간이 걸릴 수 있습니다)")
    print("참고: 모델이 로컬에 없으면 Hugging Face에서 다운로드합니다.")
    print("      Windows 환경에서는 페이징 파일 부족으로 실패할 수 있습니다.")
    
    try:
        # Windows 페이징 파일 문제 대응: device_map을 명시적으로 설정
        if torch.cuda.is_available():
            # GPU가 있으면 GPU로 로드
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,  # torch_dtype="auto" 대신 명시적 설정
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            # CPU만 있으면 CPU로 로드
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        print("모델 로딩 완료")
    except OSError as e:
        error_msg = str(e)
        if "1455" in error_msg or "paging file" in error_msg.lower():
            print(f"\n[오류] Windows 페이징 파일 부족: {e}")
            print("\n해결 방법:")
            print("1. Windows 가상 메모리(페이징 파일) 크기 증가:")
            print("   - 제어판 > 시스템 > 고급 시스템 설정 > 성능 설정 > 고급 > 가상 메모리")
            print("   - 자동 관리 해제 후 사용자 지정 크기로 설정 (최소 16GB 권장)")
            print("2. 또는 더 작은 모델 사용: --base_model Qwen/Qwen2.5-1.5B-Instruct")
            print("3. 또는 템플릿 모드 사용: --template_only 옵션으로 모델 없이 실행")
        else:
            print(f"\n[오류] 모델 로딩 실패: {e}")
        raise
    except Exception as e:
        print(f"\n[오류] 모델 로딩 실패: {e}")
        print("\n가능한 해결 방법:")
        print("1. 인터넷 연결 확인 (모델 다운로드 필요)")
        print("2. Hugging Face 토큰 설정: huggingface-cli login")
        print("3. 메모리 부족 시: --base_model을 더 작은 모델로 변경")
        raise
    
    # gradient checkpointing 사용 시 cache 비활성화 권장
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora)

    # 중요: gradient checkpointing + PEFT 조합에서 입력에 grad가 없으면
    # loss가 grad_fn 없이 나와 backward가 실패할 수 있음.
    # (현재 오류: element 0 of tensors does not require grad ...)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds = ds.map(
        lambda ex: _build_train_example(ex, tokenizer, args.max_length),
        remove_columns=ds.column_names,
    )

    collator = default_data_collator

    # bf16 지원이면 bf16 우선, 아니면 fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # TrainingArguments에서 gradient_checkpointing을 켰더라도,
    # 일부 모델/조합에서는 명시적으로 enable 해주는 것이 안전함.
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    # LoRA 어댑터 + 토크나이저 저장
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    meta = {
        "base_model": args.base_model,
        "train_jsonl": args.train_jsonl,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("학습 완료")
    print("=" * 60)
    print(f"Saved to: {args.out_dir}")
    print("다음 단계: 21_generate_category_bullets.py 로 추론 테스트")


if __name__ == "__main__":
    main()

