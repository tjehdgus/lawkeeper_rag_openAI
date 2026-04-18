from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jhgan/ko-sbert-nli")
model.save("./ko-sbert-nli")  # 로컬 저장


# --------------------------------------------------

import os
import json
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- 설정 ----------
RAG_DOCS_DIR    = r"C:\\law\\qa_rag_문서변환_한글메데"
PERSIST_PATH    = r"C:\\law\\chroma_qa"
COLLECTION_NAME = "legal_qa_rag_docs"
HF_MODEL_NAME   = "c:\\law\\ko-sbert-nli"

CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 100
# ------------------------------------

def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)

def flatten_metadata(meta: dict) -> dict:
    flat = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, (list, dict)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat

def main():
    # 1) PersistentClient 생성
    client = PersistentClient(path=PERSIST_PATH)

    # 2) 기존 컬렉션이 있으면 삭제 (임베딩 함수 충돌 방지)
    if COLLECTION_NAME in client.list_collections():
        client.delete_collection(name=COLLECTION_NAME)

    # 3) Sentence-Transformers 임베딩 함수
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=HF_MODEL_NAME
    )

    # 4) 컬렉션 생성
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # 5) 청킹 도구 설정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # 6) RAG 문서 파일 목록
    rag_files = [f for f in os.listdir(RAG_DOCS_DIR) if f.endswith("_rag.json")]
    print(f"📦 총 {len(rag_files)}개의 문서를 저장합니다...")

    # 7) 배치 처리
    for fname in tqdm(rag_files):
        doc = load_json(os.path.join(RAG_DOCS_DIR, fname))
        content = doc["content"]
        metadata = flatten_metadata(doc["metadata"])
        chunks = splitter.split_text(content)

        ids = [f"{os.path.splitext(fname)[0]}_chunk{i}" for i in range(len(chunks))]
        docs = chunks
        metas = [metadata for _ in chunks]  # 동일 메타정보 복제

        collection.add(ids=ids, documents=docs, metadatas=metas)

    # 8) 완료 메시지
    total = len(collection.get()["ids"])
    print(f"✅ '{COLLECTION_NAME}' 컬렉션에 {total}개 문서를 저장했습니다.")
    print(f"📁 저장 경로: {PERSIST_PATH}")

if __name__ == "__main__":
    main()
