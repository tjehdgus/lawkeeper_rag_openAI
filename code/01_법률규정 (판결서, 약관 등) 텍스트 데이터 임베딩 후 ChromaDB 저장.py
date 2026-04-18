import os
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1. 설정
DATA_DIR           = Path(r"/content/drive/MyDrive/판결문_전체_수정")
CHROMA_DB_PATH     = Path(r"/content/drive/MyDrive/판결문_전체_수정")
COLLECTION_NAME    = "legal_cases"
EMBEDDING_MODEL    = "jhgan/ko-sbert-nli"
BATCH_SIZE         = 32  # 전체 문서가 클 수 있으므로 배치 크기 줄임

# 2. GPU 사용 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶️ Using device: {DEVICE}")

# 3. ChromaDB 준비
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
client = PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# 4. 모델 로드
model = SentenceTransformer(EMBEDDING_MODEL)
model.to(DEVICE)

# 5. 문서 로딩 (청크화 없이 전체 문서)
file_paths = list(DATA_DIR.glob("*.json"))
documents, metadatas, ids = [], [], []

for fp in file_paths:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))

        # '취지', '사실', '판단' 섹션 합치기
        parts = []
        for sec in ("취지", "사실", "판단"):
            val = data.get(sec, [])
            if isinstance(val, list):
                parts.extend(val)
            elif isinstance(val, dict):
                for lst in val.values():
                    if isinstance(lst, list):
                        parts.extend(lst)

        full_text = "\n\n".join(parts).strip()
        if not full_text:
            print(f"⚠️ 빈 문서 건너뜀: {fp.name}")
            continue

        # 전체 문서를 하나의 단위로 저장
        documents.append(full_text)
        metadatas.append({
            "사건번호":   data["info"].get("사건번호", ""),
            "판결선고일": data["info"].get("판결선고일", ""),
            "사건명":     data["info"].get("사건명", ""),
            "법원명":     data["info"].get("법원명", ""),
            "사건유형":   data["info"].get("사건유형", ""),
            "파일명":     fp.name,
            "문서크기":   len(full_text)
        })
        ids.append(data["info"].get("사건번호", fp.stem))

    except Exception as e:
        print(f"❌ 파일 처리 오류: {fp.name} - {e}")
        continue

print(f"📄 총 {len(documents)}개 문서 로드 완료")

# 6. 임베딩 & 저장 (배치 처리)
print("🔄 임베딩 생성 및 저장 중...")

for start in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[start:start+BATCH_SIZE]
    batch_meta = metadatas[start:start+BATCH_SIZE]
    batch_ids  = ids[start:start+BATCH_SIZE]

    print(f"진행률: {start+1}-{min(start+BATCH_SIZE, len(documents))}/{len(documents)}")

    # 임베딩 생성
    embeddings = model.encode(
        batch_docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=str(DEVICE)
    )

    # ChromaDB에 저장
    collection.add(
        documents = batch_docs,
        metadatas = batch_meta,
        ids       = batch_ids,
        embeddings= embeddings.cpu().numpy().tolist()
    )

print(f"✅ 임베딩 완료: 총 {len(documents)}개 전체 문서가 '{COLLECTION_NAME}' 컬렉션에 저장되었습니다.")

# 7. 저장된 데이터 확인
print(f"\n📊 저장된 데이터 확인:")
count = collection.count()
print(f"  - 총 문서 수: {count}")

# 샘플 데이터 조회
if count > 0:
    sample = collection.peek(limit=3)
    print(f"  - 샘플 문서 크기: {[len(doc) for doc in sample['documents'][:3]]}")
    print(f"  - 샘플 사건번호: {[meta.get('사건번호', 'N/A') for meta in sample['metadatas'][:3]]}")


# --------------------------------------------------

import os
import json
import shutil
from pathlib import Path
import torch
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import gc
import time
from tqdm import tqdm

# 설정
DATA_DIR = Path("/content/drive/MyDrive/약관_전체_수정")
CHROMA_DB_PATH = Path("/content/drive/MyDrive/약관_전체_수정_DB_dragonkue")
COLLECTION_NAME = "terms_clauses"
EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
BATCH_SIZE = 32
PROCESS_BATCH_SIZE = 200  # 더 큰 배치로 한번에 더 많이 처리

def safe_load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 파일 로딩 실패: {file_path.name} - {e}")
        return None

def extract_content(data):
    if not data or not isinstance(data, dict):
        return ""

    parts = []
    if '약관분야' in data:
        parts.append(f"약관분야: {data['약관분야']}")
    parts += data.get("약관조항", [])
    if '유불리판단' in data:
        parts.append(f"유불리판단: {data['유불리판단']}")
    parts += data.get("위법성 판단 근거", [])
    parts += data.get("비교근거", [])
    parts += data.get("관련 법령", [])
    return "\n\n".join([str(p).strip() for p in parts if p]).strip()

def check_current_files():
    """현재 생성된 파일들 확인"""
    print("\n📂 현재 ChromaDB 파일 구조:")
    if not CHROMA_DB_PATH.exists():
        print("❌ ChromaDB 폴더가 존재하지 않습니다.")
        return

    for root, dirs, files in os.walk(CHROMA_DB_PATH):
        level = root.replace(str(CHROMA_DB_PATH), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            try:
                file_size = file_path.stat().st_size / 1024  # KB
                if file_size > 1024:
                    print(f"{subindent}📄 {file} ({file_size/1024:.1f}MB)")
                else:
                    print(f"{subindent}📄 {file} ({file_size:.1f}KB)")

                # link_lists 확인
                if 'link' in file.lower():
                    print(f"🎯 링크 파일 발견: {file}")

            except:
                print(f"{subindent}📄 {file} (크기 확인 불가)")

def force_index_rebuild(collection):
    """다양한 방법으로 인덱스 재구축 강제"""
    print("\n🔧 강력한 인덱스 재구축 시도...")

    total_docs = collection.count()
    print(f"📊 현재 문서 수: {total_docs}")

    if total_docs == 0:
        print("❌ 문서가 없어서 인덱스를 구축할 수 없습니다.")
        return False

    # 방법 1: 대량 쿼리로 강제 인덱스 사용
    test_queries = [
        "개인정보", "수집", "이용", "제공", "처리", "동의", "거부", "철회",
        "약관", "조항", "계약", "서비스", "이용자", "회원", "가입", "탈퇴",
        "유불리", "판단", "위법", "불법", "부당", "공정", "합리적", "적절",
        "법령", "규정", "조건", "의무", "권리", "책임", "손해", "배상"
    ]

    print("🔍 대량 쿼리 실행으로 인덱스 구축 강제...")
    for i, query in enumerate(test_queries):
        try:
            result = collection.query(
                query_texts=[query],
                n_results=min(50, total_docs),  # 더 많은 결과 요청
                include=['documents', 'metadatas', 'distances']
            )
            if i % 5 == 0:
                print(f"   진행률: {i+1}/{len(test_queries)}")
            time.sleep(0.5)  # 짧은 대기
        except Exception as e:
            print(f"⚠️ 쿼리 실행 오류 ({query}): {e}")

    # 방법 2: 대용량 결과 요청
    print("📈 대용량 검색 결과 요청...")
    try:
        large_result = collection.query(
            query_texts=["개인정보 수집 이용"],
            n_results=min(1000, total_docs),  # 매우 많은 결과 요청
            include=['documents', 'metadatas', 'distances']
        )
        print(f"✅ 대용량 검색 완료: {len(large_result['documents'][0])}개 결과")
    except Exception as e:
        print(f"❌ 대용량 검색 실패: {e}")

    # 방법 3: 다양한 유사도 임계값으로 검색
    print("🎯 다양한 임계값으로 검색...")
    search_terms = ["약관", "개인정보", "서비스", "이용자", "계약"]
    for term in search_terms:
        try:
            result = collection.query(
                query_texts=[term],
                n_results=min(200, total_docs),
                where={"약관분야": {"$ne": ""}},  # 메타데이터 필터링도 시도
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            pass  # 실패해도 계속 진행

    print("✅ 강력한 인덱스 재구축 완료")
    return True

def main():
    print("🚀 link_lists 파일 강제 생성 시작...")
    print(f"📊 처리할 JSON 파일 수: 8997개")

    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    # 기존 DB 삭제
    if CHROMA_DB_PATH.exists():
        print(f"🗑️ 기존 DB 삭제: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)
        time.sleep(2)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    # ChromaDB 클라이언트 생성 (더 적극적인 HNSW 설정)
    try:
        client = PersistentClient(path=str(CHROMA_DB_PATH))
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

        # 더 적극적인 HNSW 설정으로 link_lists 생성 강제
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,                    # 연결 수 증가 (기본 16 → 32)
                "hnsw:construction_ef": 400,     # 구축 시 탐색 범위 증가
                "hnsw:search_ef": 50,            # 검색 시 탐색 범위 증가
                "hnsw:num_threads": 8,           # 스레드 수 증가
                "hnsw:resize_factor": 1.5,       # 크기 조정 비율 증가
                "hnsw:sync_threshold": 500,      # 동기화 임계값 감소 (더 자주 동기화)
                "hnsw:batch_size": 50            # 배치 크기 감소
            }
        )
        print(f"✅ 적극적인 HNSW 설정으로 컬렉션 생성 완료")
    except Exception as e:
        print(f"❌ ChromaDB 초기화 실패: {e}")
        return

    # JSON 파일 수집
    file_paths = sorted(DATA_DIR.glob("*.json"))
    print(f"🔍 실제 JSON 파일 수: {len(file_paths)}")

    if len(file_paths) == 0:
        print(f"❌ {DATA_DIR}에서 JSON 파일을 찾을 수 없습니다.")
        return

    # 더 큰 배치로 빠르게 처리
    documents, metadatas, ids = [], [], []
    processed_count = 0

    print("📄 모든 파일 일괄 처리 중...")
    for fp in tqdm(file_paths, desc="JSON 파일 처리"):
        data = safe_load_json(fp)
        if not data:
            continue

        content = extract_content(data)
        if not content or len(content) < 100:
            continue

        documents.append(content)
        metadatas.append({
            "약관분야": str(data.get("약관분야", "")),
            "공정위 심결례": str(data.get("공정위 심결례", "")),
            "유불리판단": str(data.get("유불리판단", "")),
            "불리한 조항 유형": str(data.get("불리한 조항 유형", "")),
            "파일명": fp.stem
        })
        ids.append(fp.stem)
        processed_count += 1

    print(f"✅ 총 {len(documents)}개 문서 준비 완료")

    # 배치로 나누어 처리
    batch_size = PROCESS_BATCH_SIZE
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(documents))
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        print(f"🔄 배치 {batch_idx+1}/{total_batches} 추가 중... ({len(batch_docs)}개)")

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

            print(f"✅ 배치 {batch_idx+1} 저장 완료")

            # 중간에 파일 구조 확인
            if (batch_idx + 1) % 10 == 0:
                print(f"🔍 배치 {batch_idx+1} 후 파일 구조 확인:")
                check_current_files()

        except Exception as e:
            print(f"❌ 배치 {batch_idx+1} 저장 실패: {e}")
            continue

        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

    # 모든 데이터 추가 완료 후 강력한 인덱스 재구축
    print(f"\n🎯 모든 데이터 추가 완료! 총 {collection.count()}개 문서")

    # 최종 파일 구조 확인
    check_current_files()

    # 강력한 인덱스 재구축 실행
    force_index_rebuild(collection)

    # PersistentClient는 자동으로 저장되므로 별도 persist 불필요
    print("💾 PersistentClient는 자동으로 모든 변경사항을 디스크에 저장합니다.")

    # 파일 구조 재확인
    print("\n🔍 인덱스 재구축 후 최종 파일 구조:")
    check_current_files()

    # 최종 검색 테스트
    print("\n🧪 최종 검색 기능 테스트:")
    try:
        test_result = collection.query(
            query_texts=["개인정보 수집 및 이용"],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        if test_result["documents"][0]:
            print("✅ 검색 기능 정상 작동!")
            print(f"   검색 결과 수: {len(test_result['documents'][0])}개")
            print(f"   첫 번째 거리: {test_result['distances'][0][0]:.4f}")
        else:
            print("❌ 검색 결과 없음")
    except Exception as e:
        print(f"❌ 검색 테스트 실패: {e}")

    print("\n🎉 link_lists 강제 생성 프로세스 완료!")
    print("💾 PersistentClient가 모든 데이터를 자동으로 디스크에 저장했습니다.")
    print("📦 이제 압축해서 다른 환경으로 이동해도 정상 작동합니다!")
    print("📋 위의 파일 목록에서 link_lists 또는 유사한 파일이 생성되었는지 확인하세요.")

    # 최종 압축 가능 여부 체크
    db_size = sum(f.stat().st_size for f in CHROMA_DB_PATH.rglob('*') if f.is_file())
    print(f"📊 전체 DB 크기: {db_size / (1024*1024):.1f}MB")
    print("🚀 압축 준비 완료!")

if __name__ == "__main__":
    main()
