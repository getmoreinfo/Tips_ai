# setup_hf_token.py
# 역할: Hugging Face 토큰을 환경 변수로 설정하는 헬퍼 스크립트

import os
from huggingface_hub import login

def setup_token_from_env():
    """
    환경 변수에서 HF_TOKEN을 읽어서 로그인
    """
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("=" * 60)
        print("환경 변수 설정 방법")
        print("=" * 60)
        print()
        print("PowerShell에서 다음 명령어 실행:")
        print('$env:HF_TOKEN="여기에_토큰_붙여넣기"')
        print()
        print("또는 영구적으로 설정하려면:")
        print('[System.Environment]::SetEnvironmentVariable("HF_TOKEN", "여기에_토큰_붙여넣기", "User")')
        print()
        print("그 다음 이 스크립트를 다시 실행하세요.")
        return False
    
    try:
        login(token=token)
        print("[OK] Hugging Face 로그인 성공!")
        print()
        print("이제 TranslateGemma 모델을 사용할 수 있습니다.")
        return True
    except Exception as e:
        print(f"[오류] 로그인 실패: {e}")
        print()
        print("토큰을 확인하세요:")
        print("1. https://huggingface.co/settings/tokens 에서 토큰 확인")
        print("2. 토큰이 'hf_'로 시작하는지 확인")
        print("3. 토큰에 공백이 없는지 확인")
        return False


def setup_token_direct(token):
    """
    토큰을 직접 전달하여 로그인
    
    Args:
        token: Hugging Face 토큰 문자열
    """
    try:
        login(token=token)
        print("[OK] Hugging Face 로그인 성공!")
        return True
    except Exception as e:
        print(f"[오류] 로그인 실패: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face 토큰 설정")
    print("=" * 60)
    print()
    
    # 방법 1: 환경 변수에서 읽기
    if setup_token_from_env():
        print("로그인 완료!")
    else:
        print()
        print("=" * 60)
        print("대안: 토큰을 직접 입력")
        print("=" * 60)
        print()
        print("아래 코드를 수정하여 사용하세요:")
        print()
        print("from setup_hf_token import setup_token_direct")
        print("setup_token_direct('여기에_토큰_붙여넣기')")
        print()
