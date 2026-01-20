# Accelerate 설정 가이드 (각 컴퓨터별)

## 확인된 IP 주소

- **컴퓨터 1 (메인)**: `210.93.16.37` (rank=0)
- **컴퓨터 2**: `210.93.16.36` (rank=1)
- **컴퓨터 3**: `210.93.16.35` (rank=2)

---

## 컴퓨터 1 (메인, rank=0) 설정

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 0

What is the IP address of the machine that will host the main process?
> 210.93.16.37

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 210.93.16.36,210.93.16.35

What is the IP address of this machine?
> 210.93.16.37

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

## 컴퓨터 2 (rank=1) 설정

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 1

What is the IP address of the machine that will host the main process?
> 210.93.16.37

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 210.93.16.36,210.93.16.35

What is the IP address of this machine?
> 210.93.16.36

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

## 컴퓨터 3 (현재, rank=2) 설정

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 2

What is the IP address of the machine that will host the main process?
> 210.93.16.37

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 210.93.16.36,210.93.16.35

What is the IP address of this machine?
> 210.93.16.35

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

## 설정 확인

각 컴퓨터에서 설정 확인:

```powershell
py -3.11 -m accelerate.env
```

설정이 올바르게 되었는지 확인됩니다.

---

## 다음 단계

1. ✅ 모든 컴퓨터에서 Accelerate 설정 완료
2. ✅ 방화벽 설정 (포트 29500)
3. ✅ 데이터 파일 확인
4. ✅ 분산 학습 시작!
