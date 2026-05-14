# lawkeeper
# ⚖️ Lawkeeper – AI 법률 상담 에이전트

<img src="https://raw.githubusercontent.com/tjehdgus/lawkeeper_rag_openAI/main/images/cover.png" alt="Lawkeeper 발표자료" width="500">\

---
## 🗂 발표 자료
- [📂 발표자료 PDF](./docs/법률AI에이전트.pdf)
- 프로젝트의 자세한 내용은 발표자료에서 확인할 수 있습니다:  
# 📖 프로젝트 개요

**Lawkeeper**는 법학 문서, 판례, 약관 데이터를 결합하여  
전문 변호사 수준의 신뢰할 수 있는 **AI 법률 상담 서비스**를 구축한 프로젝트입니다.  

- **목표**: 누구나 쉽게 법률 상담을 받을 수 있는 AI 상담 에이전트 제공  
- **핵심 아이디어**: LLM + Vector DB 기반 RAG(Retrieval-Augmented Generation) 활용  
- **결과물**: 웹 기반 상담 시스템 (PC/모바일 대응), BigQuery 연동 로그, 사용자 친화적 UI  

---

## 📂 프로젝트 주요 내용

### 1️⃣ 프로젝트 목적
- LLM을 기반으로 **특정 도메인(법률)**에 특화된 AI 상담 시스템 구현  
- 법률 문제가 생기면 **판례/법령/약관 근거를 제공**하여 신뢰성 있는 상담 지원  

---

### 2️⃣ 전체 프로세스
1. 문서 데이터 수집 (법전, 판례, 약관, Q&A 크롤링 데이터)  
2. 데이터 전처리 및 JSON 변환  
3. ChromaDB에 임베딩 저장 (총 5개 컬렉션)  
4. 사용자가 질문 입력 → 코사인 유사도 검색 → 문서 검색  
5. 검색 결과 + 프롬프트 → LLM 전달 → 최종 답변 생성  

---

### 3️⃣ 데이터 수집 & 전처리
- **데이터 형식**: JSON, CSV, PDF  
- **처리 방식**:  
  - 본문(질문/조문/판례) vs 메타데이터 분리  
  - PDF → 텍스트 추출 → JSON 변환  
  - 판례/약관 단위로 별도 JSON 저장  

---

### 4️⃣ 임베딩 전략 & RAG 검색
- **임베딩**: Sentence-BERT 기반 (Ko-SBERT 모델)  
- **ChromaDB 구성**:  
  - `legal_cases` (판례)  
  - `terms_clauses` (약관 조항)  
  - `legal_qa_rag_docs` (법률 Q&A)  
  - `legal_rag_docs` (법령 본문)  
  - 기타 보조 데이터  

- **검색 방식**:  
  - 병렬 검색 후 유사도 높은 문서 수집  
  - 후속 질문에서는 상담 히스토리 반영  

---

### 5️⃣ 프롬프트 구성
- **초기 자문**: 정해진 답변 구조 + 신뢰성 있는 상담 톤 유지  
- **후속 자문**: 이전 상담 내용 + 새로운 정보 반영, 유연한 대화 진행  
- **히스토리 요약**: 상담의 충분성 판단 및 요약 기능 포함  

---

### 6️⃣ 홈페이지 기능
- **주요 기능**:  
  - 개인정보 동의서  
  - 초기 상담 화면 & 리셋 버튼  
  - 결과 신뢰도(관련 문서 개수 표기)  
  - 사용자 맞춤 테마/글씨 크기 조절  
  - 상담 책임 관련 고지  

- **핵심 클래스 구조**:  
  - `DirectBigQueryLogger`: BigQuery에 상담 내용/성능 로그 기록  
  - `ResponseCache & EmbeddingCache`: 캐시 기반 성능 최적화  
  - `ChromaDB Service`: 병렬 검색 및 최적화  
  - `Chat Service`: 상담 흐름 관리 (설명 충분성 판단 & 요약 포함)  

---

### 7️⃣ 배포 및 활용
- **배포 환경**: Google Cloud Run  
- **로그 분석**: BigQuery + Dashboard  
- **내부 활용 예시**: 주소 데이터 검증, 법률 자문 기록 관리  

---

# 🛠️ 기술 스택

### Programming Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Frameworks & DB
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge)  
![ChromaDB](https://img.shields.io/badge/ChromaDB-7D40FF?style=for-the-badge)  
![BigQuery](https://img.shields.io/badge/GoogleBigQuery-669DF6?style=for-the-badge&logo=googlecloud&logoColor=white)

### AI/ML
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
![Sentence-BERT](https://img.shields.io/badge/SBERT-NLP-blue?style=for-the-badge)

---

# 📊 주요 성과
- 법률 도메인 특화 AI 상담 시스템 구축  
- RAG 기반 검색 & 프롬프트 최적화로 상담 품질 향상  
- PC/모바일 웹 배포 및 실시간 로그 분석 지원  
- 실제 상담에 활용 가능한 신뢰성·확장성 확보  
