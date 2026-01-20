# 다른 컴퓨터 설정 가이드

**중요:** Cursor를 설치할 필요는 없습니다! Cursor는 코드 편집용 IDE일 뿐이며, 학습 실행에는 필요하지 않습니다.

## 필요한 것들

### 1. Python 환경 (필수)

#### Python 설치 확인
```powershell
python --version
```
Python 3.8 이상이 필요합니다.

없다면:
- [Python 공식 사이트](https://www.python.org/downloads/)에서 다운로드
- 또는 Microsoft Store에서 설치

### 2. GPU 드라이버 및 CUDA (필수)

#### GPU 드라이버 확인
```powershell
nvidia-smi
```

다음이 출력되어야 합니다:
- GPU 이름 (RTX 4060 Ti)
- CUDA Version (예: 12.1)

#### CUDA가 없다면:
1. [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
2. GPU 모델 선택 후 다운로드
3. 설치 후 재부팅

### 3. 프로젝트 폴더 복사 (필수)

#### 방법 1: USB/네트워크 공유로 복사
```
다음 파일들을 다른 컴퓨터에 복사:
- 모든 .py 파일 (00_*.py, 01_*.py, 02_*.py 등)
- training_data_10000.csv
- .env 파일 (선택사항, DB 연결이 필요한 경우)
- README 파일들
```

#### 방법 2: Git 사용 (권장)
```bash
# 다른 컴퓨터에서
git clone <your-repository-url>
# 또는 프로젝트 폴더 전체를 복사
```

### 4. 패키지 설치 (필수)

#### 다른 컴퓨터에서 프로젝트 폴더로 이동:
```powershell
cd C:\Users\<사용자명>\dev\tips_ai
```

#### 필수 패키지 설치:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas psycopg2-binary python-dotenv accelerate
```

또는 한 번에:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 transformers datasets scikit-learn pandas psycopg2-binary python-dotenv accelerate
```

### 5. 네트워크 설정 (필수)

#### 모든 컴퓨터가 같은 네트워크에 연결되어 있어야 합니다.

#### IP 주소 확인 (각 컴퓨터에서):
```powershell
ipconfig
```

**이더넷 어댑터** 또는 **무선 LAN 어댑터**에서:
- IPv4 주소 확인 (예: 192.168.1.100)
- 서브넷 마스크 확인 (예: 255.255.255.0)

#### 방화벽 설정 (Windows)
1. **Windows 보안** → **방화벽 및 네트워크 보호**
2. **고급 설정**
3. **인바운드 규칙** → **새 규칙**
4. **포트** 선택 → **다음**
5. **TCP** 선택, **특정 로컬 포트**: `29500` → **다음**
6. **연결 허용** → **다음**
7. **모든 프로필** 체크 → **다음**
8. 이름: "PyTorch Distributed Training" → **완료**

### 6. Accelerate 설정 (멀티 노드 사용 시)

#### Accelerate 설치 (위에서 이미 설치했으면 생략)
```bash
pip install accelerate
```

#### 설정 파일 생성:
```bash
accelerate config
```

**질문에 답변:**

**노드 1 (메인 노드)에서:**
```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine?
> 0

What is the IP address of the machine that will host the main process?
> 192.168.1.100 (메인 노드 IP)

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 192.168.1.101,192.168.1.102 (다른 노드 IP들)

What is the IP address of this machine?
> 192.168.1.100

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision?
> fp16 (GPU가 지원하는 경우)
```

**노드 2에서:**
- 위와 동일하지만 rank는 **1**, IP는 **192.168.1.101**

**노드 3에서:**
- 위와 동일하지만 rank는 **2**, IP는 **192.168.1.102**

## 설정 확인

### 각 컴퓨터에서 테스트:

#### 1. GPU 확인:
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

#### 2. 패키지 확인:
```bash
pip list | findstr "torch transformers accelerate"
```

#### 3. 파일 확인:
```powershell
ls *.py
ls training_data_10000.csv
```

## 실행 방법

### 모든 컴퓨터에서 동시에 실행:

**방법 1: Accelerate 사용 (추천)**
```bash
accelerate launch 02_finetune_local.py
```

**방법 2: Distributed 직접 사용**
```bash
# 노드 1에서
.\run_distributed_node1.ps1

# 노드 2에서
.\run_distributed_node2.ps1

# 노드 3에서
.\run_distributed_node3.ps1
```

**중요:** 3개를 거의 동시에(5초 이내) 실행해야 합니다!

## 요약 체크리스트

각 컴퓨터에서 확인:

- [ ] Python 3.8+ 설치됨
- [ ] GPU 드라이버 설치됨 (`nvidia-smi` 동작)
- [ ] 프로젝트 폴더 복사됨
- [ ] 필요한 패키지 설치됨
- [ ] `training_data_10000.csv` 파일 존재
- [ ] IP 주소 확인됨
- [ ] 같은 네트워크에 연결됨
- [ ] 방화벽 설정 완료 (포트 29500)
- [ ] Accelerate 설정 완료 (`accelerate config`)

## 문제 해결

### GPU가 인식되지 않을 때:
1. `nvidia-smi` 실행 확인
2. PyTorch CUDA 버전 확인: `python -c "import torch; print(torch.version.cuda)"`
3. GPU 드라이버 재설치

### 연결 오류가 날 때:
1. 모든 컴퓨터가 같은 네트워크에 있는지 확인
2. 방화벽 설정 확인
3. IP 주소 확인
4. 메인 노드의 IP가 올바른지 확인

### 패키지 오류:
1. 모든 컴퓨터에서 같은 버전 설치 확인
2. Python 버전 확인
3. pip 업데이트: `python -m pip install --upgrade pip`

## 결론

**필요한 것:**
1. ✅ Python
2. ✅ GPU 드라이버
3. ✅ 프로젝트 파일
4. ✅ 패키지 설치
5. ✅ 네트워크 연결

**불필요한 것:**
- ❌ Cursor IDE (필수 아님, 텍스트 에디터면 충분)
- ❌ 복잡한 개발 환경
- ❌ Git (파일 복사만으로도 가능)

간단하게 **PowerShell** 또는 **명령 프롬프트**로 실행하면 됩니다!
