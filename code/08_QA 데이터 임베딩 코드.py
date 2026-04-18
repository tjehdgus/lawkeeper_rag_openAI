from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jhgan/ko-sbert-nli")
model.save("./ko-sbert-nli")  # 로컬 저장


# --------------------------------------------------

import os
import json
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ---------- 설정 ----------
RAG_DOCS_DIR    = r"F:\data\법률\qa_rag_문서변환_한글메데"
PERSIST_PATH    = r"./chroma_qa_rag2"
COLLECTION_NAME = "legal_qa_rag_docs"
HF_MODEL_NAME = "./ko-sbert-nli"
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

    # 4) 컬렉션 생성 (새로 생성되므로 임베딩 함수 이름이 일치)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # 5) RAG 문서 파일 목록
    rag_files = [f for f in os.listdir(RAG_DOCS_DIR) if f.endswith("_rag.json")]
    print(f"📦 총 {len(rag_files)}개의 문서를 저장합니다...")

    # 6) 배치 크기 설정
    batch_size = 500

    # 7) 배치 단위로 추가
    for i in tqdm(range(0, len(rag_files), batch_size)):
        batch = rag_files[i : i + batch_size]
        ids, docs, metas = [], [], []
        for fname in batch:
            doc = load_json(os.path.join(RAG_DOCS_DIR, fname))
            unique_id = os.path.splitext(fname)[0]
            ids.append(unique_id)
            docs.append(doc["content"])
            metas.append(flatten_metadata(doc["metadata"]))
        collection.add(ids=ids, documents=docs, metadatas=metas)

    # 8) 완료 메시지
    total = len(collection.get()["ids"])
    print(f"✅ '{COLLECTION_NAME}' 컬렉션에 {total}개 문서를 저장했습니다.")
    print(f"📁 저장 경로: {PERSIST_PATH}")

if __name__ == "__main__":
    main()
