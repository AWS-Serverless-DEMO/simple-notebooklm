# 📚 Simple NotebookLM

AWS S3 Vectors와 Bedrock Claude를 활용한 문서 기반 질의응답(RAG) 시스템

## 🎯 프로젝트 개요

Simple NotebookLM은 PDF, DOCX, TXT 문서를 업로드하고 AI에게 문서 내용에 대해 질문할 수 있는 웹 애플리케이션입니다. AWS의 최신 벡터 스토리지 기술인 S3 Vectors와 Claude Sonnet 4를 활용하여 정확한 답변과 출처를 제공합니다.

## 🏗️ 아키텍처

```
[1단계] 문서 업로드 & 전처리
  ↓ Streamlit 파일 업로드 (PDF/DOCX/TXT)
  ↓ PyPDF2로 텍스트 추출 (페이지별)
  ↓ LangChain RecursiveCharacterTextSplitter
  ↓ chunk_size: 500 tokens, overlap: 50 tokens

[2단계] 벡터화 & S3 Vectors 저장
  ↓ Bedrock Titan Text Embeddings V2
  ↓ 1024차원 벡터 생성
  ↓ S3 Vector Bucket에 PutVectors API로 저장
  ↓ 강한 일관성 보장 (즉시 검색 가능)

[3단계] 질의응답 (RAG)
  ↓ 사용자 질문 입력
  ↓ Titan Embeddings로 질문 벡터화
  ↓ S3 Vectors QueryVectors API 호출 (top_k=3)
  ↓ 유사도 임계값 체크 (>0.7)
  ↓ 시스템 프롬프트 + 검색된 청크 → Bedrock Claude
  ↓ 답변 + 출처(문서명, 페이지) 반환
```

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **텍스트 추출**: PyPDF2, python-docx
- **청크 분할**: LangChain RecursiveCharacterTextSplitter
- **임베딩**: AWS Bedrock Titan Text Embeddings V2 (1024차원)
- **벡터 저장**: AWS S3 Vectors (Preview)
- **LLM**: AWS Bedrock Claude Sonnet 4

## 📁 프로젝트 구조

```
simple-notebooklm/
├── app.py                      # Streamlit 메인 애플리케이션
├── cleanup.py                  # 리소스 정리 스크립트 (NEW!)
├── config.py                   # 설정 관리
├── requirements.txt            # Python 패키지 의존성
├── .env.example               # 환경변수 템플릿
├── README.md                  # 프로젝트 문서
└── utils/
    ├── __init__.py
    ├── document_processor.py  # PDF/DOCX/TXT 텍스트 추출
    ├── text_splitter.py       # LangChain 청크 분할
    ├── embeddings.py          # Bedrock Titan Embeddings
    ├── s3_vectors.py          # S3 Vectors 관리 (PutVectors/QueryVectors/DeleteVectors)
    └── rag_engine.py          # Claude RAG 엔진
```

## 🚀 시작하기

### 1. 사전 요구사항

- Python 3.8 이상
- AWS 계정 및 자격증명
- AWS Bedrock 액세스 권한
- AWS S3 Vectors 액세스 권한 (Preview 신청 필요)

### 2. AWS 설정

#### 2.1 AWS Bedrock 모델 액세스 활성화

1. AWS Console → Bedrock → Model access 이동
2. 다음 모델 액세스 요청:
   - Amazon Titan Text Embeddings V2
   - Anthropic Claude Sonnet 4

#### 2.2 S3 Vectors 설정

1. S3 Console에서 Vector Bucket 생성
2. Vector Index 생성
3. Bucket 이름과 Index 이름 기록

#### 2.3 IAM 권한 설정

IAM 사용자/역할에 다음 권한 추가:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockAccess",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
                "arn:aws:bedrock:*::foundation-model/anthropic.*"
            ]
        },
        {
            "Sid": "S3VectorsReadWrite",
            "Effect": "Allow",
            "Action": [
                "s3vectors:PutVectors",
                "s3vectors:QueryVectors",
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:ListVectors",
                "s3vectors:CreateIndex",
                "s3vectors:CreateVectorBucket",
                "s3vectors:ListVectorBuckets",
                "s3vectors:ListIndexes",
                "s3vectors:GetVectorBucket",
                "s3vectors:GetIndex"
            ],
            "Resource": "arn:aws:s3vectors:*:*:bucket/*"
        }
    ]
}
```

### 3. 설치

```bash
# 1. 저장소 클론 또는 디렉토리 이동
cd simple-notebooklm

# 2. Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

### 4. 환경 설정

```bash
# 1. .env 파일 생성
cp .env.example .env

# 2. .env 파일 편집
nano .env  # 또는 원하는 텍스트 에디터 사용
```

`.env` 파일 내용 예시:

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# S3 Vectors Configuration
S3_VECTOR_BUCKET_NAME=my-vector-bucket
S3_VECTOR_INDEX_NAME=my-vector-index

# Bedrock Model IDs
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL_ID=global.anthropic.claude-sonnet-4-20250514-v1:0

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=15
```

### 5. 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 애플리케이션에 접근할 수 있습니다.

## 📖 사용 방법

1. **문서 업로드**
   - 왼쪽 사이드바에서 "📤 문서 업로드" 섹션 찾기
   - PDF, DOCX, TXT 파일 선택
   - "문서 처리 시작" 버튼 클릭

2. **문서 처리 진행**
   - 텍스트 추출 (페이지별)
   - 청크 분할 (500자 단위, 50자 중복)
   - 임베딩 생성 (1024차원 벡터)
   - S3 Vectors 저장

3. **질문하기**
   - 문서 처리 완료 후 질문 입력창 표시
   - 질문 입력 후 "🔍 질문하기" 버튼 클릭
   - AI가 문서 내용 기반 답변 생성

4. **답변 확인**
   - 답변 내용 표시
   - 출처 정보 (문서명, 페이지, 유사도) 제공
   - 관련 청크 미리보기 제공

5. **문서 관리**
   - 사이드바 "🗂️ 문서 관리" 섹션에서 저장된 문서 확인
   - 개별 문서 삭제 또는 전체 삭제 가능
   - 목록 새로고침으로 최신 상태 유지

## 🧹 리소스 정리 (Resource Cleanup)

**중요**: 프로젝트 발표 후 AWS 비용을 절감하려면 반드시 리소스를 정리하세요!

### 방법 1: Streamlit UI에서 삭제 (간편)

**개별 문서 삭제**
1. 사이드바 "🗂️ 문서 관리" 섹션 확인
2. 삭제할 문서의 expandable 클릭
3. "🗑️ 삭제" 버튼 클릭

**모든 문서 삭제**
1. "🗑️ 모든 문서 삭제" 버튼 클릭
2. 확인 경고 표시 → 한 번 더 클릭하여 확정

### 방법 2: cleanup.py 스크립트 사용 (권장)

**인터랙티브 모드** (가장 안전)
```bash
python cleanup.py
```
대화형 메뉴를 통해:
- 저장된 문서 목록 확인
- 개별 문서 삭제
- 모든 벡터 삭제
- 각 단계마다 확인 절차 포함

**명령줄 옵션**

```bash
# 저장된 문서 목록만 확인
python cleanup.py --list

# 특정 문서 삭제 (확인 포함)
python cleanup.py --delete "document.pdf"

# 특정 문서 즉시 삭제 (확인 없이)
python cleanup.py --delete "document.pdf" --force

# 모든 벡터 삭제 (확인 포함)
python cleanup.py --delete-all

# 모든 벡터 즉시 삭제 (확인 없이)
python cleanup.py --delete-all --force
```

### 정리 범위

✅ **삭제되는 항목**
- S3 Vectors에 저장된 모든 벡터 데이터
- 문서 메타데이터 (문서명, 페이지, 청크 정보)

❌ **삭제되지 않는 항목** (수동 삭제 필요)
- S3 Vector Bucket 자체
- S3 Vector Index 자체
- 원본 업로드 파일 (로컬에만 존재)

### 비용 최적화 팁

1. **프로젝트 발표 직후**: `python cleanup.py --delete-all --force` 실행
2. **테스트 중**: UI에서 불필요한 문서만 개별 삭제
3. **완전 종료 시**: AWS Console에서 Vector Bucket/Index도 삭제
4. **정기 점검**: `python cleanup.py --list`로 저장된 벡터 확인

### AWS Console에서 완전 삭제 (선택)

Vector 데이터 삭제 후에도 Bucket/Index는 남아있어 최소 비용이 발생할 수 있습니다.

```bash
# AWS CLI로 Index 삭제
aws s3vectors delete-vector-index \
  --vector-bucket-name YOUR_BUCKET_NAME \
  --index-name YOUR_INDEX_NAME

# Bucket 삭제 (Index가 모두 삭제된 후)
aws s3vectors delete-vector-bucket \
  --vector-bucket-name YOUR_BUCKET_NAME
```

또는 AWS Console → S3 → Vector Buckets에서 수동 삭제

## 🔧 설정 옵션

`config.py` 또는 `.env` 파일에서 다음 설정을 조정할 수 있습니다:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `CHUNK_SIZE` | 1000 | 청크 최대 크기 (문자 수) |
| `CHUNK_OVERLAP` | 100 | 청크 간 중복 문자 수 |
| `SIMILARITY_THRESHOLD` | 0.3 | 유사도 임계값 (0.0 ~ 1.0, NotebookLM 스타일: 낮은 값으로 LLM이 관련성 판단) |
| `TOP_K_RESULTS` | 15 | 검색할 최대 청크 수 (더 많은 후보를 LLM에게 제공) |

## 📊 주요 특징

### 1. 다양한 문서 형식 지원
- PDF: 페이지별 텍스트 추출
- DOCX: 문단 단위 텍스트 추출
- TXT: 전체 텍스트 읽기

### 2. 고급 청크 분할
- RecursiveCharacterTextSplitter 사용
- 의미 단위 유지 (문단 → 문장 → 단어 순)
- 청크 간 컨텍스트 중복으로 연속성 보장

### 3. 최신 벡터 스토리지
- AWS S3 Vectors (클라우드 최초 네이티브 벡터 스토리지)
- 강한 일관성 보장 (즉시 검색 가능)
- 최대 90% 비용 절감
- 서브초(sub-second) 쿼리 성능

### 4. NotebookLM 스타일 지능형 RAG
- **낮은 임계값 (0.15)**: 벡터 검색은 후보만 찾고, LLM이 최종 관련성 판단
- **LLM 기반 필터링**: Claude가 검색된 청크 중 실제 관련 있는 내용만 사용
- **의미적 이해**: 표현이 다르더라도 의미가 같으면 연결 (예: "만점 받으려면?" → "채점 기준" 활용)
- **스마트 컨텍스트 활용**: 15개 후보 청크를 LLM에게 제공하여 최적의 답변 생성

### 5. 정확한 출처 제공
- 답변과 함께 문서명, 페이지 번호 제공
- 유사도 점수로 신뢰도 표시
- 관련 청크 미리보기 제공

## 🐛 트러블슈팅

### 문제: "Configuration Error: Missing required configuration"
**해결**: `.env` 파일이 올바르게 설정되었는지 확인하세요.

```bash
# .env 파일 존재 확인
ls -la .env

# .env 파일 내용 확인
cat .env
```

### 문제: "Failed to generate embedding"
**해결**:
1. AWS 자격증명이 올바른지 확인
2. Bedrock 모델 액세스가 활성화되었는지 확인
3. AWS 리전이 Bedrock을 지원하는지 확인 (us-east-1 권장)

### 문제: "Failed to put vectors"
**해결**:
1. S3 Vectors가 Preview로 활성화되었는지 확인
2. Vector Bucket과 Index가 생성되었는지 확인
3. IAM 권한에 `s3vectors:PutVectors`가 포함되었는지 확인

### 문제: 텍스트 추출 실패
**해결**:
- PDF: PyPDF2로 읽을 수 없는 스캔된 PDF일 수 있습니다 (OCR 필요)
- DOCX: 파일이 손상되지 않았는지 확인
- 파일 크기가 너무 큰 경우 처리 시간이 오래 걸릴 수 있습니다

## 📝 참고 자료

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Titan Text Embeddings V2](https://aws.amazon.com/blogs/aws/amazon-titan-text-v2-now-available-in-amazon-bedrock-optimized-for-improving-rag/)
- [Amazon S3 Vectors](https://aws.amazon.com/blogs/aws/introducing-amazon-s3-vectors-first-cloud-storage-with-native-vector-support-at-scale/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 기여

이슈 리포트와 풀 리퀘스트를 환영합니다!

## 📄 라이선스

MIT License

## ⚠️ 주의사항

1. **AWS 비용**: Bedrock과 S3 Vectors 사용 시 비용이 발생합니다.
2. **S3 Vectors Preview**: S3 Vectors는 현재 Preview 단계이므로 프로덕션 사용 전 충분한 테스트가 필요합니다.
3. **보안**: `.env` 파일에 AWS 자격증명이 포함되므로 절대 공개 저장소에 커밋하지 마세요.
4. **데이터 프라이버시**: 업로드된 문서는 AWS 서비스를 통해 처리됩니다. 민감한 데이터 처리 시 주의하세요.

## 🆘 지원

문제가 발생하거나 질문이 있으시면 GitHub Issues를 통해 문의해주세요.
