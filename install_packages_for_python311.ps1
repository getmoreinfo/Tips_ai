# Python 3.11.7용 패키지 설치 스크립트
# 사용법: PowerShell에서 .\install_packages_for_python311.ps1 실행

Write-Host "=========================================="
Write-Host "Python 3.11.7용 패키지 설치 스크립트"
Write-Host "=========================================="
Write-Host ""

# Python 버전 확인
Write-Host "[1/4] Python 버전 확인..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] $pythonVersion"
    
    # Python 3.11인지 확인
    if ($pythonVersion -match "Python 3\.11") {
        Write-Host "[OK] Python 3.11 버전 확인됨"
    } else {
        Write-Host "[WARNING] Python 3.11이 아닙니다. 현재 버전: $pythonVersion"
        Write-Host "          Python 3.11.7 설치를 권장합니다."
        Write-Host ""
        $continue = Read-Host "계속하시겠습니까? (y/n)"
        if ($continue -ne "y") {
            exit 1
        }
    }
} else {
    Write-Host "[ERROR] Python이 설치되지 않았습니다!"
    Write-Host "       Python 3.11.7을 먼저 설치해주세요."
    exit 1
}
Write-Host ""

# pip 업그레이드
Write-Host "[2/4] pip 업그레이드..."
python -m pip install --upgrade pip
Write-Host ""

# PyTorch 설치 (CUDA 11.8)
Write-Host "[3/4] PyTorch 설치 (CUDA 11.8)..."
Write-Host "      이 과정은 시간이 걸릴 수 있습니다..."
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
Write-Host ""

# 나머지 패키지 설치
Write-Host "[4/4] 필수 패키지 설치..."
$packages = @(
    "transformers==4.57.6",
    "datasets==4.5.0",
    "scikit-learn==1.8.0",
    "pandas==2.3.3",
    "numpy==2.3.5",
    "psycopg2-binary==2.9.11",
    "python-dotenv==1.2.1",
    "accelerate==1.12.0"
)

foreach ($package in $packages) {
    Write-Host "  - $package 설치 중..."
    python -m pip install $package
}
Write-Host ""

# 설치 확인
Write-Host "=========================================="
Write-Host "설치 확인"
Write-Host "=========================================="
Write-Host ""

Write-Host "Python 버전:"
python --version
Write-Host ""

Write-Host "PyTorch CUDA 지원:"
python -c "import torch; print('  CUDA 사용 가능:', torch.cuda.is_available()); print('  GPU 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
Write-Host ""

Write-Host "설치된 패키지:"
pip list | Select-String "torch|transformers|accelerate|pandas|numpy|scikit|datasets" | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "=========================================="
Write-Host "설치 완료!"
Write-Host "=========================================="
Write-Host ""
Write-Host "다음 단계:"
Write-Host "1. 테스트 실행: python test_training_quick.py"
Write-Host "2. 정상 작동 확인 후 분산 학습 시작"
Write-Host ""
