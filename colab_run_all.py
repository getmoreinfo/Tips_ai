"""
colab_run_all.py

Colab에서 2→3→4→4-1 단계를 한 번에 실행.
- 전제: 1번(Drive 마운트) 이미 실행됨. /content/drive 에 마운트되어 있어야 함.
- 사용: Colab 셀에서
    from google.colab import drive
    drive.mount('/content/drive')
    %run /content/drive/MyDrive/tips_ai_colab/colab_run_all.py
  또는, 이 스크립트를 Drive tips_ai_colab에 넣은 뒤 마운트 셀 실행하고
  %run /content/drive/MyDrive/tips_ai_colab/colab_run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys

DRIVE_BASE = "/content/drive/MyDrive/tips_ai_colab"
WORK_DIR = "/content/tips_ai"
RESULTS_DIR = f"{WORK_DIR}/results/qwen2.5-7b-lora-report-summary"
TRAIN_JSONL = "training_report_summary_sft_500.jsonl"  # 22 쓴 경우 training_report_summary_sft.jsonl 로 변경


def run(cmd: str | list, check: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd if isinstance(cmd, list) else ["sh", "-c", cmd],
        check=check,
        cwd=cwd,
    )


def main() -> None:
    print("=" * 60)
    print("colab_run_all: 2→3→4→4-1 자동 실행")
    print("=" * 60)

    if not os.path.isdir("/content/drive"):
        print("[ERROR] /content/drive 없음. 1번 Drive 마운트 셀을 먼저 실행하세요.")
        sys.exit(1)

    if not os.path.isdir(DRIVE_BASE):
        print(f"[ERROR] {DRIVE_BASE} 없음. Drive에 tips_ai_colab 폴더 생성 후 필수 파일 업로드하세요.")
        sys.exit(1)

    # 2. 작업 디렉터리 + 파일 복사
    print("\n[2] 작업 디렉터리 + 파일 복사")
    os.makedirs(WORK_DIR, exist_ok=True)
    run(f"cp -r {DRIVE_BASE}/* {WORK_DIR}/")
    run(["ls", "-lh"], cwd=WORK_DIR)

    # 3. 패키지 설치 + GPU 확인
    print("\n[3] 패키지 설치 + GPU 확인")
    run(["pip", "install", "-q", "transformers", "peft", "datasets", "accelerate", "torch", "bitsandbytes", "pandas"])
    run(["python", "-c", "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"])

    # 4. LoRA 학습
    train_jsonl = os.path.join(WORK_DIR, TRAIN_JSONL)
    if not os.path.isfile(train_jsonl):
        print(f"[ERROR] {train_jsonl} 없음. tips_ai_colab 에 JSONL 업로드 후 다시 실행.")
        sys.exit(1)
    print("\n[4] LoRA 학습 (Qwen 7B)")
    run(["python", "23_train_report_summary_lora.py",
         "--base_model", "Qwen/Qwen2.5-7B-Instruct",
         "--train_jsonl", TRAIN_JSONL,
         "--out_dir", RESULTS_DIR,
         "--epochs", "5", "--lr", "1e-4", "--max_length", "2048",
         "--lora_r", "8", "--lora_alpha", "16", "--batch_size", "1", "--grad_accum", "8",
         "--save_steps", "100", "--logging_steps", "10"],
        cwd=WORK_DIR)

    # 4-1. 학습 결과 → Drive 복사
    print("\n[4-1] 학습 결과 → Drive 복사")
    run(["ls", "-la", RESULTS_DIR])
    run(["mkdir", "-p", f"{DRIVE_BASE}/results"])
    run(f"cp -r {RESULTS_DIR} {DRIVE_BASE}/results/", check=True)
    run(["ls", "-la", f"{DRIVE_BASE}/results/qwen2.5-7b-lora-report-summary"])

    print("\n" + "=" * 60)
    print("✅ 자동 실행 완료. Drive → tips_ai_colab → results → qwen2.5-7b-lora-report-summary")
    print("=" * 60)


if __name__ == "__main__":
    main()
