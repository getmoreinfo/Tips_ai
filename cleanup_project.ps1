# 프로젝트 정리 스크립트
# 어제 오늘 만든 파일 + 필수 파일만 유지하고 나머지 삭제

# 유지할 파일 목록
$keepFiles = @(
    # 어제(1/28) 만든 파일들
    "22_prepare_report_summary_sft.py",
    "23_train_report_summary_lora.py",
    "24_generate_report_summary.py",
    "25_generate_category_report_from_csv.py",
    "report_summary_lib.py",
    "report_viewer.html",
    "serve_report_viewer.py",
    "training_report_summary_sft.jsonl",
    "training_report_summary_sft_200.jsonl",
    "report_stroller.json",
    "report_storybook.json",
    "report_category_4.json",
    "일일보고_연구일지_김민수_2026_01_28.md",
    
    # 오늘(1/29) 만든 파일들
    "26_evaluate_report_summary.py",
    "training_report_summary_sft_500.jsonl",
    "COLAB_SETUP_GUIDE.md",
    "COLAB_QUICK_START.md",
    "COLAB_FILES_CHECKLIST.md",
    "COLAB_VSCODE_INTEGRATION.md",
    "colab_train_qwen7b.py",
    "CLOUD_GPU_SETUP_GUIDE.md",
    "requirements_lora.txt",
    
    # 필수 의존 파일들
    "ai_report_bullets_lib.py",
    "db_category_loader.py",
    "README.md",
    
    # Git 관련
    ".git",
    ".gitignore",
    
    # 이 스크립트 자체
    "cleanup_project.ps1"
)

Write-Host "프로젝트 정리 시작..." -ForegroundColor Cyan
Write-Host "유지할 파일: $($keepFiles.Count)개" -ForegroundColor Green

# 모든 파일 가져오기
$allFiles = Get-ChildItem -File -Recurse | Where-Object { 
    $_.FullName -notmatch "\\\.git\\" -and 
    $_.FullName -notmatch "\\reports\\" -and
    $_.FullName -notmatch "\\results" 
}

$deletedCount = 0
$keptCount = 0

foreach ($file in $allFiles) {
    $fileName = $file.Name
    $shouldKeep = $false
    
    foreach ($keepFile in $keepFiles) {
        if ($fileName -eq $keepFile) {
            $shouldKeep = $true
            break
        }
    }
    
    if (-not $shouldKeep) {
        Write-Host "삭제: $fileName" -ForegroundColor Yellow
        Remove-Item $file.FullName -Force
        $deletedCount++
    } else {
        $keptCount++
    }
}

Write-Host "`n정리 완료!" -ForegroundColor Green
Write-Host "유지된 파일: $keptCount개" -ForegroundColor Green
Write-Host "삭제된 파일: $deletedCount개" -ForegroundColor Yellow
