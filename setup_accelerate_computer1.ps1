# 컴퓨터 1 Accelerate 설정 스크립트 (메인 노드, rank=0)

Write-Host "=========================================="
Write-Host "컴퓨터 1 Accelerate 설정 (메인 노드, rank=0)"
Write-Host "=========================================="
Write-Host ""

# IP 주소 정보
$mainNodeIP = "210.93.16.37"  # 컴퓨터 1 (메인) - 현재 컴퓨터
$computer2IP = "210.93.16.36"
$computer3IP = "210.93.16.35"
$port = "29500"

Write-Host "IP 주소 정보:"
Write-Host "- 컴퓨터 1 (메인): $mainNodeIP"
Write-Host "- 컴퓨터 2: $computer2IP"
Write-Host "- 컴퓨터 3: $computer3IP"
Write-Host "- 포트: $port"
Write-Host ""

# 설정 디렉토리 생성
$configDir = "$env:USERPROFILE\.cache\huggingface\accelerate"
if (-not (Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    Write-Host "[OK] 설정 디렉토리 생성: $configDir"
}

# Accelerate 설정 파일 생성
$configContent = @"
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 3
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_ip: $mainNodeIP
main_process_port: $port
"@

$configFile = "$configDir\default_config.yaml"
$configContent | Out-File -FilePath $configFile -Encoding utf8 -Force

Write-Host "[OK] Accelerate 설정 파일 생성 완료!"
Write-Host "경로: $configFile"
Write-Host ""

# 설정 확인
Write-Host "생성된 설정 파일:"
Get-Content $configFile
Write-Host ""

Write-Host "=========================================="
Write-Host "설정 완료!"
Write-Host "=========================================="
Write-Host ""
Write-Host "컴퓨터 1 (메인 노드) 설정이 완료되었습니다."
Write-Host ""
Write-Host "분산 학습 실행:"
Write-Host "py -3.11 -m accelerate.commands.launch 02_finetune_local.py"
Write-Host ""
Write-Host "중요: 컴퓨터 1이 먼저 실행되어야 합니다!"
Write-Host ""
