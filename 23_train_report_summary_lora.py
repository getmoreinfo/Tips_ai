"""
23_train_report_summary_lora.py
역할:
  - 22_prepare_report_summary_sft.py 로 만든 JSONL(SFT)을 이용해
    Qwen2.5 Instruct 계열 CausalLM을 LoRA(PEFT)로 미세튜닝한다.
  - 커머스 시장 리포트 요약 생성 모델 학습

필요 패키지:
  - transformers / datasets / accelerate
  - peft

실행 예:
  python 22_prepare_report_summary_sft.py --input_csv products_all.csv --reviews_csv reviews_all.csv --out_jsonl training_report_summary_sft.jsonl
  python 23_train_report_summary_lora.py --train_jsonl training_report_summary_sft.jsonl --out_dir results_report/qwen2.5-3b-lora-report-summary
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# 프록시 설정 제거
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_HUB_OFFLINE"] = "0"

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
    ap = argparse.ArgumentParser(
        description="Qwen2.5 LoRA 학습. A100 등 대용량 GPU 기준 기본값(학습량 많음)."
    )
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="베이스 모델 (7B 권장)")
    ap.add_argument("--train_jsonl", default="training_report_summary_sft.jsonl")
    ap.add_argument("--out_dir", default="results_report/qwen2.5-7b-lora-report-summary")
    # A100(40GB+) 기준: 긴 시퀀스 + 대배치
    ap.add_argument("--max_length", type=int, default=4096, help="최대 시퀀스 길이 (A100 권장 4096)")
    ap.add_argument("--epochs", type=int, default=10, help="학습 에폭 (A100에서 많이 돌리기)")
    ap.add_argument("--lr", type=float, default=1e-4, help="학습률")
    ap.add_argument("--batch_size", type=int, default=4, help="디바이스당 배치 (A100에서 4~8)")
    ap.add_argument("--grad_accum", type=int, default=8, help="그래디언트 누적 (effective batch = batch_size * grad_accum)")
    ap.add_argument("--lora_r", type=int, default=32, help="LoRA rank (A100에서 16~64)")
    ap.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (보통 2*r)")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=3, help="유지할 체크포인트 수")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="워밍업 비율")
    args = ap.parse_args()

    _require_peft()
    from peft import LoraConfig, get_peft_model  # noqa: E402

    print("=" * 60)
    print("커머스 시장 리포트 요약 LoRA 학습")
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
    # 리포트 요약 학습은 입력 포맷이 길고 민감해서, 토크나이저는 base_model과 반드시 일치시키는 것이 안전함
    print(f"베이스 모델에서 토크나이저 로드: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n모델 로딩 중... (시간이 걸릴 수 있습니다)")
    try:
        if torch.cuda.is_available():
            # device_map="auto"는 일부 환경에서 CPU offload + 페이징/드라이버 이슈를 유발할 수 있어
            # 8GB 환경에서는 일단 단순하게 CUDA로 올리는 방식이 더 안정적임.
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model.to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        print("모델 로딩 완료")
    except Exception as e:
        print(f"\n[오류] 모델 로딩 실패: {e}")
        raise

    model.config.use_cache = False

    lora = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds = ds.map(
        lambda ex: _build_train_example(ex, tokenizer, args.max_length),
        remove_columns=ds.column_names,
    )

    collator = default_data_collator

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
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

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
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("학습 완료")
    print("=" * 60)
    print(f"Saved to: {args.out_dir}")
    print("다음 단계: 24_generate_report_summary.py 로 추론 테스트")


if __name__ == "__main__":
    main()
