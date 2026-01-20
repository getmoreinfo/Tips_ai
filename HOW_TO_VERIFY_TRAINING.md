# 학습 정상 작동 확인 방법

## 방법 1: 빠른 테스트 실행 (가장 쉬움! ⭐)

### 10스텝만 실행하여 확인:

```bash
python test_training_quick.py
```

**이 스크립트는:**
- ✅ 전체 데이터 대신 1000개만 사용
- ✅ 10 스텝만 실행 (약 1-2분)
- ✅ GPU 사용 확인
- ✅ 정상 작동 여부 확인

**정상 작동 시 출력:**
```
✅ 테스트 성공!
학습이 정상적으로 작동합니다!
```

---

## 방법 2: GPU 사용률 확인

### 별도 PowerShell 창에서:

```bash
nvidia-smi
```

**정상 작동 중이면:**
- GPU 사용률: 50-100%
- GPU 메모리 사용 중
- 프로세스 이름: python.exe

**실시간 모니터링:**
```bash
# Windows에서는 watch 명령이 없으므로
# 1초마다 실행하려면 PowerShell에서:
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

---

## 방법 3: 학습 로그 확인

### 학습이 실행 중이면:

**로그 파일 확인:**
```bash
dir logs
```

**최신 로그 확인:**
```powershell
Get-Content logs\*.log -Tail 20
```

**정상 작동 시 로그 예시:**
```
{'loss': 3.45, 'learning_rate': 0.00002, 'epoch': 0.1}
{'loss': 3.12, 'learning_rate': 0.00002, 'epoch': 0.2}
{'eval_loss': 2.89, 'eval_accuracy': 0.45, 'eval_f1': 0.42, 'epoch': 0.1}
```

---

## 방법 4: 일부만 실행하고 중단

### 전체 학습 실행 후 일부 확인:

```bash
python 02_finetune_local.py
```

**정상 작동 확인 방법:**
1. 학습 시작 메시지 확인:
   ```
   Fine-tuning 시작
   총 Epoch: 3
   ```

2. 첫 몇 스텝 진행 확인:
   ```
   {'loss': 3.45, 'learning_rate': 0.00002, 'epoch': 0.1}
   {'loss': 3.12, 'learning_rate': 0.00002, 'epoch': 0.2}
   ```

3. 정상 작동 확인되면 `Ctrl + C`로 중단

**중단해도 되나요?**
- ✅ 네, 정상 작동 확인만 하려면 중단해도 됩니다
- ✅ 나중에 전체 학습을 다시 실행하면 됩니다
- ✅ 모델은 저장되므로 중간에 중단해도 문제없습니다

---

## 방법 5: 체크리스트로 확인

### 학습 시작 후 확인:

**정상 작동 체크리스트:**

- [ ] GPU 확인 메시지 출력됨
- [ ] 데이터 로드 완료 메시지 출력됨
- [ ] 모델 로드 완료 메시지 출력됨
- [ ] 토크나이징 완료 메시지 출력됨
- [ ] "Fine-tuning 시작" 메시지 출력됨
- [ ] 로그에서 loss 값이 출력됨
- [ ] GPU 사용률이 증가함 (`nvidia-smi` 확인)
- [ ] 에러 메시지가 없음

**모두 체크되면 정상 작동!** ✅

---

## 추천 순서

### 1단계: 빠른 테스트 (2분)
```bash
python test_training_quick.py
```

### 2단계: 정상 작동 확인되면
- Git 커밋 및 푸시
- 나머지 컴퓨터에서 pull
- 전체 학습 시작

### 3단계: 전체 학습 (선택사항)
```bash
python 02_finetune_local.py
```

---

## 빠른 확인 명령어

**모두 한 번에 실행:**

```powershell
# 1. 빠른 테스트 실행
python test_training_quick.py

# 2. 성공하면 Git 커밋
git add .
git commit -m "Fix NaN handling and verify training works"
git push origin main
```

---

## 정상 작동 확인 후 다음 단계

1. ✅ **Git 커밋 및 푸시**
   ```bash
   git add .
   git commit -m "Complete setup: verified training works"
   git push origin main
   ```

2. ✅ **나머지 두 컴퓨터에서 Pull**
   ```bash
   git pull origin main
   ```

3. ✅ **멀티 노드 설정 및 실행**
   - 각 컴퓨터에서 Accelerate 설정
   - 3대 동시에 실행

---

**가장 빠른 확인 방법: `python test_training_quick.py` 실행!** ⚡
