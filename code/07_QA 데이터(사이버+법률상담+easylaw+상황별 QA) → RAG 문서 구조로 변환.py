import os
import json
import csv

# ========== 설정 ==========
# 원본 데이터 경로
QA_JSON_DIR   = r"F:\data\법률\상황압축해제\Other\QA데이터"
LAWQA_DIR     = r"F:\data\lawqa"

# 출력 경로
OUTPUT_DIR    = r"F:\data\법률\qa_rag_문서변환"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================

def load_json(path: str):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="utf-8") as f:
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

def transform_qa_file(path: str):
    """
    JSON 파일을 읽어 RAG 문서 튜플 리스트로 반환.
    - 리스트 형식: [{instruction,input,output,...}, ...]
    - dict 형식: {id,title,question,answer,commentary,...}
    """
    obj = load_json(path)
    base = os.path.splitext(os.path.basename(path))[0]
    docs = []

    if isinstance(obj, list):
        # 리스트 형식
        for idx, qa in enumerate(obj):
            doc_id = f"{base}_{idx}"
            inst = qa.get("instruction", "").strip()
            inp  = qa.get("input", "").strip()
            out  = qa.get("output", "").strip()
            parts = [f"### Instruction:\n{inst}"] \
                  + ([f"### Input:\n{inp}"] if inp else []) \
                  + [f"### Output:\n{out}"]
            content = "\n\n".join(parts)
            meta = {"source_file": os.path.basename(path)}
            meta.update(flatten_metadata(qa))
            docs.append((doc_id, content, meta))

    elif isinstance(obj, dict):
        # 단일 객체 형식
        if "instruction" in obj:
            # instruction-based
            doc_id = base
            inst = obj.get("instruction", "").strip()
            inp  = obj.get("input", "").strip()
            out  = obj.get("output", "").strip()
            parts = [f"### Instruction:\n{inst}"] \
                  + ([f"### Input:\n{inp}"] if inp else []) \
                  + [f"### Output:\n{out}"]
            content = "\n\n".join(parts)
            meta = {"source_file": os.path.basename(path)}
            meta.update(flatten_metadata(obj))
            docs.append((doc_id, content, meta))

        elif "question" in obj and "answer" in obj:
            # QA_00002.json 스타일
            doc_id = obj.get("id", base)
            title = obj.get("title", "").strip()
            q     = obj.get("question", "").strip()
            a     = obj.get("answer", "").strip()
            comm  = obj.get("commentary", "").strip()
            parts = ([f"Title: {title}"] if title else []) \
                  + [f"Q: {q}", f"A: {a}"] \
                  + ([f"Commentary:\n{comm}"] if comm else [])
            content = "\n\n".join(parts)
            meta = {k: v for k, v in obj.items() if k not in ("id","title","question","answer","commentary")}
            meta["source_file"] = os.path.basename(path)
            meta = flatten_metadata(meta)
            docs.append((doc_id, content, meta))

    return docs

def transform_csv_file(path: str):
    """
    CSV 파일을 읽어 각 행을 RAG 문서 튜플 리스트로 반환.
    컬럼명 '질문'/'답변' 또는 'question'/'answer' 사용.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    docs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            doc_id = f"{base}_{idx}"
            q = row.get("질문", row.get("question", "")).strip()
            a = row.get("답변", row.get("answer", "")).strip()
            content = f"Q: {q}\n\nA: {a}"
            meta = {k: v for k, v in row.items() if k not in ("질문","답변","question","answer")}
            meta["source_file"] = os.path.basename(path)
            docs.append((doc_id, content, flatten_metadata(meta)))
    return docs

def save_doc(doc_id: str, content: str, metadata: dict):
    out = {"id": doc_id, "content": content, "metadata": metadata}
    out_path = os.path.join(OUTPUT_DIR, f"{doc_id}_rag.json")
    with open(out_path, "w", encoding="utf-8-sig") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    # 1) QA JSON 디렉터리 처리
    for fname in os.listdir(QA_JSON_DIR):
        if not fname.lower().endswith(".json"):
            continue
        full = os.path.join(QA_JSON_DIR, fname)
        for doc_id, content, meta in transform_qa_file(full):
            save_doc(doc_id, content, meta)

    # 2) lawqa 디렉터리 파일 처리
    for fname in os.listdir(LAWQA_DIR):
        full = os.path.join(LAWQA_DIR, fname)
        if fname.lower().endswith(".json"):
            # easylaw_kr.json 등
            for doc_id, content, meta in transform_qa_file(full):
                save_doc(doc_id, content, meta)
        elif fname.lower().endswith(".csv"):
            # 법률상담.csv, 사이버상담.csv
            for doc_id, content, meta in transform_csv_file(full):
                save_doc(doc_id, content, meta)

    print(f"✅ 모든 RAG 문서가 '{OUTPUT_DIR}'에 생성되었습니다. 총 파일 수: {len(os.listdir(OUTPUT_DIR))}")

if __name__ == "__main__":
    main()