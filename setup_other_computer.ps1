# 다른 컴퓨터 설정 자동화 스크립트
# 사용법: PowerShell에서 .\setup_other_computer.ps1 실행

Write-Host "=========================================="
Write-Host "다른 컴퓨터 설정 자동화 스크립트"
Write-Host "=========================================="
Write-Host ""

# Python 확인
Write-Host "[1/6] Python 버전 확인..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] $pythonVersion"
} else {
    Write-Host "[ERROR] Python이 설치되지 않았습니다!"
    Write-Host "       Python을 먼저 설치해주세요: https://www.python.org/downloads/"
    exit 1
}
Write-Host ""

# GPU 확인
Write-Host "[2/6] GPU 드라이버 확인..."
$nvidiaSmi = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] GPU 드라이버가 설치되어 있습니다."
    Write-Host $nvidiaSmi | Select-String "CUDA Version"
} else {
    Write-Host "[WARNING] GPU 드라이버가 설치되지 않았거나 nvidia-smi를 찾을 수 없습니다."
    Write-Host "          GPU를 사용하려면 NVIDIA 드라이버를 설치하세요."
}
Write-Host ""

# pip 업그레이드
Write-Host "[3/6] pip 업그레이드..."
python -m pip install --upgrade pip
Write-Host ""

# PyTorch 설치
Write-Host "[4/6] PyTorch 설치 (CUDA 지원)..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Write-Host ""

# 필수 패키지 설치
Write-Host "[5/6] 필수 패키지 설치..."
$packages = @(
    "transformers",
    "datasets",
    "scikit-learn",
    "pandas",
    "psycopg2-binary",
    "python-dotenv",
    "accelerate"
)

foreach ($package in $packages) {
    Write-Host "  - $package 설치 중..."
    python -m pip install $package
}
Write-Host ""

# 설치 확인
Write-Host "[6/6] 설치 확인..."
Write-Host ""
Write-Host "PyTorch CUDA 지원:"
python -c "import torch; print('  CUDA 사용 가능:', torch.cuda.is_available()); print('  GPU 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "설치된 패키지:"
pip list | Select-String "torch|transformers|accelerate" | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "=========================================="
Write-Host "설치 완료!"
Write-Host "=========================================="
Write-Host ""
Write-Host "다음 단계:"
Write-Host "1. 프로젝트 폴더를 이 컴퓨터에 복사하세요"
Write-Host "2. training_data_10000.csv 파일이 있는지 확인하세요"
Write-Host "3. IP 주소 확인: ipconfig"
Write-Host "4. Accelerate 설정: accelerate config"
Write-Host "5. 실행 준비 완료!"
Write-Host ""
