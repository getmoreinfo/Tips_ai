"""
Google Colab에서 Qwen 7B 학습을 위한 간단한 실행 스크립트
이 파일을 Colab 노트북에서 실행하세요.
"""

# 필수 패키지 설치
!pip install -q transformers peft datasets accelerate torch bitsandbytes

# GPU 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 학습 실행
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary \
  --epochs 5 \
  --lr 1e-4 \
  --max_length 2048 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --grad_accum 8 \
  --save_steps 100 \
  --logging_steps 10

print("\n학습 완료!")
print("결과는 Google Drive에 저장되었습니다:")
print("/content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary/")
