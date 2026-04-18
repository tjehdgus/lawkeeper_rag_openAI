import os
import json
from datetime import datetime

def load_json(path):
    """
    Load JSON with BOM-safe utf-8-sig, fallback to utf-8.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

def strip_text(txt):
    """Trim and collapse whitespace/newlines."""
    if isinstance(txt, str):
        return txt.strip().replace("\n", " ").replace("  ", " ")
    return ""

def process_file(path):
    """
    Convert a single JSON law-case file (of any supported structure)
    into a RAG-ready document dict, preserving all structured info.
    """
    data = load_json(path)
    fname = os.path.basename(path)

    # --- 1) Collect top‐level metadata ---
    metadata = {
        "case_id": data.get("case_id") or data.get("사건번호"),
        "title":   data.get("title")   or data.get("사건명"),
        "court":   data.get("court")   or data.get("법원명") or data.get("재결청"),
        "judgment_date": data.get("judgment_date")
                         or data.get("선고일자")
                         or data.get("의결일자"),
        "judgment_type": data.get("judgment_type") or data.get("court_type"),
        "keywords":     [],
        "issues":       [],
        "legal_basis":  [],
        "precedents":   [],
        "source_files": [fname]
    }

    # --- 2) Extract all "source_*" sections, or fall back to raw file ---
    sections = {k: v for k, v in data.items() if k.startswith("source_")}
    if not sections:
        # no merged sections: treat entire top-level data as one section
        sections["raw"] = data

    content_parts = []

    # --- 3) For each section, pull every field we care about ---
    for sec in sections.values():
        if not isinstance(sec, dict):
            continue

        # Common labels for 판례/심결례
        for label in (
            "판시사항", "판결요지", "판례내용",  # 판례
            "재결요지", "주문", "청구취지", "이유"  # 심결례
        ):
            if label in sec and sec[label]:
                content_parts.append(f"[{label}]\n{strip_text(sec[label])}")

        # Summary array (요약)
        if "Summary" in sec:
            for e in sec["Summary"]:
                txt = e.get("summ_pass") or e.get("summ_contxt")
                if txt:
                    content_parts.append(f"[요약]\n{strip_text(txt)}")

        # 추가 판결요지 (jdgmn)
        if sec.get("jdgmn"):
            content_parts.append(f"[추가 판결요지]\n{strip_text(sec['jdgmn'])}")

        # Q&A entries
        if "jdgmnInfo" in sec:
            for qa in sec["jdgmnInfo"]:
                q = strip_text(qa.get("question", ""))
                a = strip_text(qa.get("answer", ""))
                if q or a:
                    content_parts.append(f"[질문]\n{q}\n[답변]\n{a}")
            metadata["issues"].extend(sec["jdgmnInfo"])

        # Keywords
        if "keyword_tagg" in sec:
            metadata["keywords"].extend(
                [kw["keyword"] for kw in sec["keyword_tagg"] if kw.get("keyword")]
            )

        # 법조문·판례 참조
        if "Reference_info" in sec:
            ri = sec["Reference_info"]
            if ri.get("reference_rules"):
                metadata["legal_basis"].extend(
                    [r.strip() for r in ri["reference_rules"].split(",")]
                )
            if ri.get("reference_court_case"):
                metadata["precedents"].extend(
                    [p.strip() for p in ri["reference_court_case"].split(",")]
                )
        else:
            # fallback raw keys
            if sec.get("참조조문"):
                metadata["legal_basis"].extend(
                    [r.strip() for r in sec["참조조문"].split(",")]
                )
            if sec.get("참조판례"):
                metadata["precedents"].extend(
                    [p.strip() for p in sec["참조판례"].split(",")]
                )

    # --- 4) Deduplicate lists ---
    metadata["keywords"]    = list(dict.fromkeys(metadata["keywords"]))
    metadata["legal_basis"] = list(dict.fromkeys(metadata["legal_basis"]))
    metadata["precedents"]  = list(dict.fromkeys(metadata["precedents"]))
    # issues may be left as-is (they're structured dicts)

    # --- 5) Build final RAG document ---
    rag_doc = {
        "id":      metadata["case_id"] or f"doc_{datetime.now().timestamp()}",
        "content": "\n\n".join(content_parts),
        "metadata": metadata
    }
    return rag_doc

def batch_convert(input_dir, output_dir):
    """
    Process every .json in input_dir → convert to RAG docs → save under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue
        in_path  = os.path.join(input_dir, fname)
        out_name = fname.replace(".json", "_rag.json")
        out_path = os.path.join(output_dir, out_name)
        try:
            rag = process_file(in_path)
            with open(out_path, "w", encoding="utf-8-sig") as f:
                json.dump(rag, f, ensure_ascii=False, indent=2)
            print(f"✅ {fname} → {out_name}")
        except Exception as e:
            print(f"⚠️ 변환 실패: {fname} ({e})")

if __name__ == "__main__":
    batch_convert(
        input_dir=r"F:\data\법률\병합결과_판례",
        output_dir=r"F:\data\법률\rag_문서변환결과"
    )


# --------------------------------------------------

import os
import json
from datetime import datetime

def load_json(path):
    """
    Load JSON with BOM-safe utf-8-sig, fallback to utf-8.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

def strip_text(txt):
    """Trim and collapse whitespace/newlines."""
    if isinstance(txt, str):
        return txt.strip().replace("\n", " ").replace("  ", " ")
    return ""

def process_file(path):
    """
    Convert a single JSON law-case file into a RAG-ready document dict,
    with metadata keys in Korean.
    """
    data = load_json(path)
    fname = os.path.basename(path)

    # --- 1) Collect top‐level fields into Korean‐keyed metadata ---
    md = {}
    md["대표사건번호"]   = data.get("case_id") or data.get("사건번호")
    md["사건명"]       = data.get("title")   or data.get("사건명")
    md["법원명"]       = data.get("court")   or data.get("법원명") or data.get("재결청")
    md["판결선고일"]   = data.get("judgment_date") \
                      or data.get("선고일자") \
                      or data.get("의결일자")
    md["판결유형"]     = data.get("judgment_type") or data.get("court_type")
    md["키워드태그"]   = []
    md["Q&A"]         = []
    md["참조법령태깅"]  = []
    md["참조판례태깅"]  = []
    md["출처파일"]     = [fname]

    # --- 2) Determine sections to process ---
    sections = {k: v for k, v in data.items() if k.startswith("source_")}
    if not sections:
        sections["raw"] = data

    content_parts = []

    # --- 3) Extract from each section ---
    for sec in sections.values():
        if not isinstance(sec, dict):
            continue

        # 판시사항 / 판결요지 / 판례내용
        for lbl in ("판시사항", "판결요지", "판례내용"):
            if sec.get(lbl):
                content_parts.append(f"[{lbl}]\n{strip_text(sec[lbl])}")

        # 심결례 fields
        for lbl in ("재결요지", "주문", "청구취지", "이유"):
            if sec.get(lbl):
                content_parts.append(f"[{lbl}]\n{strip_text(sec[lbl])}")

        # Summary 요약
        if sec.get("Summary"):
            for e in sec["Summary"]:
                txt = e.get("summ_pass") or e.get("summ_contxt")
                if txt:
                    content_parts.append(f"[요약]\n{strip_text(txt)}")

        # 추가 판결요지 (jdgmn)
        if sec.get("jdgmn"):
            content_parts.append(f"[추가 판결요지]\n{strip_text(sec['jdgmn'])}")

        # Q&A (jdgmnInfo)
        if sec.get("jdgmnInfo"):
            for qa in sec["jdgmnInfo"]:
                q = strip_text(qa.get("question", ""))
                a = strip_text(qa.get("answer", ""))
                content_parts.append(f"[질문]\n{q}\n[답변]\n{a}")
            md["Q&A"].extend(sec["jdgmnInfo"])

        # 키워드태그
        if sec.get("keyword_tagg"):
            md["키워드태그"].extend(
                [kw.get("keyword") for kw in sec["keyword_tagg"] if kw.get("keyword")]
            )

        # 참조 정보
        ri = sec.get("Reference_info", {})
        if ri:
            if ri.get("reference_rules"):
                md["참조법령태깅"].extend(
                    [r.strip() for r in ri["reference_rules"].split(",")]
                )
            if ri.get("reference_court_case"):
                md["참조판례태깅"].extend(
                    [p.strip() for p in ri["reference_court_case"].split(",")]
                )
        else:
            # fallback to Korean keys
            if sec.get("참조조문"):
                md["참조법령태깅"].extend(
                    [r.strip() for r in sec["참조조문"].split(",")]
                )
            if sec.get("참조판례"):
                md["참조판례태깅"].extend(
                    [p.strip() for p in sec["참조판례"].split(",")]
                )

    # --- 4) Deduplicate lists ---
    md["키워드태그"]   = list(dict.fromkeys(md["키워드태그"]))
    md["참조법령태깅"] = list(dict.fromkeys(md["참조법령태깅"]))
    md["참조판례태깅"] = list(dict.fromkeys(md["참조판례태깅"]))

    # --- 5) Assemble RAG document ---
    rag_doc = {
        "id":      md.get("대표사건번호") or f"doc_{datetime.now().timestamp()}",
        "content": "\n\n".join(content_parts),
        "metadata": md
    }
    return rag_doc

def batch_convert(input_dir, output_dir):
    """
    Process every .json in input_dir into RAG docs with Korean metadata keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname.replace(".json", "_rag.json"))
        try:
            rag = process_file(src)
            with open(dst, "w", encoding="utf-8-sig") as f:
                json.dump(rag, f, ensure_ascii=False, indent=2)
            print(f"✅ {fname} → {os.path.basename(dst)}")
        except Exception as e:
            print(f"⚠️ 변환 실패: {fname} ({e})")

if __name__ == "__main__":
    batch_convert(
        input_dir=r"F:\data\법률\병합결과_판례",
        output_dir=r"F:\data\법률\rag_문서변환결과_한글메데"
    )
