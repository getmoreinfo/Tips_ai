# 맥북 환경 설정 (Python · DB 추출)

맥에서는 `python` 대신 **`python3`**를 쓰고, 가상환경에서 패키지를 설치하는 것을 권장합니다.

---

## 1. Python 확인

```bash
python3 --version   # 3.9+ 권장
```

없으면 **Homebrew**로 설치:

```bash
brew install python@3.11
```

---

## 2. 가상환경 생성 및 활성화

```bash
cd /path/to/ai-tr
python3 -m venv venv
source venv/bin/activate
```

프롬프트에 `(venv)`가 붙으면 활성화된 상태입니다.

---

## 3. DB 추출용 패키지 설치 (21 스크립트)

```bash
pip install -r requirements_db.txt
```

---

## 4. .env 설정

```bash
cp env.example .env
```

`.env`를 열어 `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD` 등을 실제 DB 값으로 수정합니다.

---

## 5. DB → CSV 추출 실행

```bash
python 21_export_products_reviews_from_db.py
```

가상환경이 활성화된 상태에서는 `python`이 `python3`를 가리키므로 위처럼 실행하면 됩니다.

---

## 6. Colab 학습용 패키지 (로컬에서 22·23 등 실행 시)

```bash
pip install -r requirements_lora.txt
```

---

## 요약

| 단계 | 명령 |
|------|------|
| 1 | `python3 -m venv venv` |
| 2 | `source venv/bin/activate` |
| 3 | `pip install -r requirements_db.txt` |
| 4 | `cp env.example .env` → 편집 |
| 5 | `python 21_export_products_reviews_from_db.py` |
