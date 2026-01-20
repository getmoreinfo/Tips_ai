# 새 Git 저장소 설정 자동화 스크립트
# 사용법: PowerShell에서 .\setup_new_repository.ps1 실행

Write-Host "=========================================="
Write-Host "새 Git 저장소 설정"
Write-Host "=========================================="
Write-Host ""

# 현재 원격 저장소 확인
Write-Host "현재 원격 저장소:"
git remote -v
Write-Host ""

$choice = Read-Host "1. GitHub 새 저장소 연결
2. GitLab 새 저장소 연결
3. 기존 저장소 그대로 사용 (푸시만)
4. 원격 저장소 제거 (로컬만 사용)
선택 (1-4): "

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "GitHub 저장소 설정"
        Write-Host "=================="
        Write-Host ""
        Write-Host "1. GitHub에서 새 Private 저장소를 만드세요:"
        Write-Host "   https://github.com/new"
        Write-Host ""
        $repoUrl = Read-Host "2. 새 저장소 URL을 입력하세요 (예: https://github.com/username/tips_ai.git): "
        
        # 기존 원격 저장소 제거
        $currentRemote = git remote | Select-Object -First 1
        if ($currentRemote) {
            Write-Host "기존 원격 저장소 제거 중..."
            git remote remove origin
        }
        
        # 새 원격 저장소 추가
        Write-Host "새 GitHub 저장소 연결 중..."
        git remote add origin $repoUrl
        Write-Host "[OK] 원격 저장소 연결 완료"
        Write-Host ""
        
        # 파일 추가 및 커밋
        Write-Host "파일 추가 중..."
        git add .
        Write-Host ""
        
        $commitMessage = Read-Host "커밋 메시지를 입력하세요 (기본: Initial commit): "
        if ([string]::IsNullOrWhiteSpace($commitMessage)) {
            $commitMessage = "Initial commit"
        }
        
        Write-Host "커밋 중..."
        git commit -m $commitMessage
        Write-Host ""
        
        Write-Host "푸시할까요? (y/N): "
        $push = Read-Host
        if ($push -eq "y" -or $push -eq "Y") {
            Write-Host "GitHub에 푸시 중..."
            git push -u origin main
            Write-Host "[OK] 푸시 완료!"
        } else {
            Write-Host "[INFO] 나중에 'git push -u origin main' 명령으로 푸시하세요."
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "GitLab 저장소 설정"
        Write-Host "=================="
        Write-Host ""
        Write-Host "1. GitLab에서 새 Private 저장소를 만드세요:"
        Write-Host "   https://gitlab.com/projects/new"
        Write-Host ""
        $repoUrl = Read-Host "2. 새 저장소 URL을 입력하세요 (예: https://gitlab.com/username/tips_ai.git): "
        
        # 기존 원격 저장소 제거
        $currentRemote = git remote | Select-Object -First 1
        if ($currentRemote) {
            Write-Host "기존 원격 저장소 제거 중..."
            git remote remove origin
        }
        
        # 새 원격 저장소 추가
        Write-Host "새 GitLab 저장소 연결 중..."
        git remote add origin $repoUrl
        Write-Host "[OK] 원격 저장소 연결 완료"
        Write-Host ""
        
        # 파일 추가 및 커밋
        Write-Host "파일 추가 중..."
        git add .
        Write-Host ""
        
        $commitMessage = Read-Host "커밋 메시지를 입력하세요 (기본: Initial commit): "
        if ([string]::IsNullOrWhiteSpace($commitMessage)) {
            $commitMessage = "Initial commit"
        }
        
        Write-Host "커밋 중..."
        git commit -m $commitMessage
        Write-Host ""
        
        Write-Host "푸시할까요? (y/N): "
        $push = Read-Host
        if ($push -eq "y" -or $push -eq "Y") {
            Write-Host "GitLab에 푸시 중..."
            git push -u origin main
            Write-Host "[OK] 푸시 완료!"
        } else {
            Write-Host "[INFO] 나중에 'git push -u origin main' 명령으로 푸시하세요."
        }
    }
    
    "3" {
        Write-Host ""
        Write-Host "기존 저장소에 푸시"
        Write-Host "=================="
        Write-Host ""
        
        # 파일 추가 및 커밋
        Write-Host "파일 추가 중..."
        git add .
        Write-Host ""
        
        $commitMessage = Read-Host "커밋 메시지를 입력하세요 (기본: Update project files): "
        if ([string]::IsNullOrWhiteSpace($commitMessage)) {
            $commitMessage = "Update project files"
        }
        
        Write-Host "커밋 중..."
        git commit -m $commitMessage
        Write-Host ""
        
        Write-Host "푸시 중..."
        git push origin main
        Write-Host "[OK] 푸시 완료!"
    }
    
    "4" {
        Write-Host ""
        Write-Host "원격 저장소 제거 (로컬만 사용)"
        Write-Host "=============================="
        Write-Host ""
        
        $confirm = Read-Host "원격 저장소를 제거하시겠습니까? (y/N): "
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            git remote remove origin
            Write-Host "[OK] 원격 저장소 제거 완료"
            Write-Host ""
            Write-Host "로컬에서만 작업합니다."
            Write-Host "파일을 추가하려면: git add . && git commit -m 'message'"
        } else {
            Write-Host "[취소] 원격 저장소 제거가 취소되었습니다."
        }
    }
    
    default {
        Write-Host "[ERROR] 잘못된 선택입니다."
        exit 1
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "설정 완료!"
Write-Host "=========================================="
Write-Host ""
Write-Host "현재 원격 저장소:"
git remote -v
Write-Host ""
