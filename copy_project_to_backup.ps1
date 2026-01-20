# 프로젝트 폴더 및 Cursor 설정 백업 스크립트
# 사용법: PowerShell에서 .\copy_project_to_backup.ps1 실행

param(
    [string]$BackupPath = "C:\Backup\tips_ai_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [switch]$IncludeCursorHistory = $false
)

Write-Host "=========================================="
Write-Host "프로젝트 백업 스크립트"
Write-Host "=========================================="
Write-Host ""

# 현재 프로젝트 폴더 경로
$ProjectPath = $PSScriptRoot
$ProjectName = Split-Path -Leaf $ProjectPath

Write-Host "프로젝트 경로: $ProjectPath"
Write-Host "백업 경로: $BackupPath"
Write-Host ""

# 백업 폴더 생성
New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null
$ProjectBackupPath = Join-Path $BackupPath $ProjectName
New-Item -ItemType Directory -Force -Path $ProjectBackupPath | Out-Null

# 1. 프로젝트 파일 복사
Write-Host "[1/3] 프로젝트 파일 복사 중..."
$excludeItems = @('__pycache__', '.git', 'venv', 'results', 'logs', 'finetuned_e5_large', '*.pyc')

$items = Get-ChildItem -Path $ProjectPath -Exclude $excludeItems
foreach ($item in $items) {
    $destination = Join-Path $ProjectBackupPath $item.Name
    Copy-Item -Path $item.FullName -Destination $destination -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "[OK] 프로젝트 파일 복사 완료"
Write-Host ""

# 2. .env 파일 처리 (민감 정보)
Write-Host "[2/3] .env 파일 확인..."
$envFile = Join-Path $ProjectPath ".env"
if (Test-Path $envFile) {
    $envBackup = Join-Path $ProjectBackupPath ".env.example"
    Copy-Item -Path $envFile -Destination $envBackup -Force
    Write-Host "[OK] .env 파일 복사됨 (민감 정보 주의!)"
    Write-Host "[INFO] 실제 .env 파일은 다른 컴퓨터에서 다시 생성하세요."
} else {
    Write-Host "[INFO] .env 파일이 없습니다."
}
Write-Host ""

# 3. Cursor 설정 복사 (선택사항)
if ($IncludeCursorHistory) {
    Write-Host "[3/3] Cursor 설정 복사 중..."
    $cursorAppData = "$env:APPDATA\Cursor"
    $cursorBackupPath = Join-Path $BackupPath "Cursor_Settings"
    
    if (Test-Path $cursorAppData) {
        New-Item -ItemType Directory -Force -Path $cursorBackupPath | Out-Null
        
        # 설정 파일만 복사 (전체 폴더는 너무 클 수 있음)
        $cursorSettings = Join-Path $cursorAppData "User\settings.json"
        if (Test-Path $cursorSettings) {
            $settingsBackup = Join-Path $cursorBackupPath "settings.json"
            Copy-Item -Path $cursorSettings -Destination $settingsBackup -Force
            Write-Host "[OK] Cursor 설정 파일 복사됨"
        }
        
        # workspaceStorage 복사 (대화 히스토리가 있다면)
        $workspaceStorage = Join-Path $cursorAppData "User\workspaceStorage"
        if (Test-Path $workspaceStorage) {
            $wsBackup = Join-Path $cursorBackupPath "workspaceStorage"
            Copy-Item -Path $workspaceStorage -Destination $wsBackup -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "[OK] Cursor 워크스페이스 스토리지 복사됨 (대화 내용 포함 가능)"
        }
    } else {
        Write-Host "[INFO] Cursor 설정 폴더를 찾을 수 없습니다."
    }
} else {
    Write-Host "[3/3] Cursor 설정 복사 건너뜀 (IncludeCursorHistory 플래그가 없음)"
    Write-Host "[INFO] Cursor 설정을 포함하려면: .\copy_project_to_backup.ps1 -IncludeCursorHistory"
}
Write-Host ""

# 4. 프로젝트 정보 파일 생성
Write-Host "[INFO] 프로젝트 정보 파일 생성 중..."
$infoFile = Join-Path $BackupPath "BACKUP_INFO.txt"
$infoContent = @"
프로젝트 백업 정보
==================
백업 일시: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
프로젝트 경로: $ProjectPath
백업 경로: $BackupPath

포함된 내용:
- 모든 Python 파일 (.py)
- 데이터 파일 (.csv)
- 설정 파일 (.env.example)
- 문서 파일 (.md)
- PowerShell 스크립트 (.ps1)
$(
    if ($IncludeCursorHistory) {
        "- Cursor 설정 및 워크스페이스 스토리지"
    }
)

제외된 내용:
- __pycache__/
- .git/ (Git 저장소가 있다면)
- venv/ (가상 환경)
- results/
- logs/
- finetuned_e5_large/

다른 컴퓨터에서 복원:
1. 백업 폴더를 다른 컴퓨터로 복사
2. 프로젝트 폴더를 원하는 위치에 붙여넣기
3. .env 파일 생성 (create_env_file.py 실행)
4. Cursor 설정 복사 (선택사항): Cursor_Settings 폴더의 내용을 %APPDATA%\Cursor 에 복사
"@

$infoContent | Out-File -FilePath $infoFile -Encoding UTF8
Write-Host "[OK] 백업 정보 파일 생성됨: $infoFile"
Write-Host ""

# 완료
Write-Host "=========================================="
Write-Host "백업 완료!"
Write-Host "=========================================="
Write-Host ""
Write-Host "백업 위치: $BackupPath"
Write-Host ""
Write-Host "다음 단계:"
Write-Host "1. 백업 폴더를 USB/네트워크/클라우드로 복사"
Write-Host "2. 다른 컴퓨터로 옮기기"
Write-Host "3. 프로젝트 폴더를 원하는 위치에 붙여넣기"
Write-Host ""
if ($IncludeCursorHistory) {
    Write-Host "4. Cursor 설정 복사 (선택사항):"
    Write-Host "   $cursorBackupPath → %APPDATA%\Cursor"
}
Write-Host ""
