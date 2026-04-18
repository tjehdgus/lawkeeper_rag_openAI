import os
import re
import json
import fitz  # PyMuPDF

# === 설정 ===
pdf_path = "한글 6법전(헌법, 민법, 형법, 상법, 민사소송법, 형사소송법).pdf"
output_dir = "./law_json_output"
os.makedirs(output_dir, exist_ok=True)

law_sections = {
    "헌법": (5, 19),
    "민법": (21, 132),
    "형법": (133, 170),
    "상법": (171, 340),
    "민사소송법": (341, 398),
    "형사소송법": (399, 486)
}

def extract_articles_precise_with_multiline_title(law_name, start_page, end_page):
    doc = fitz.open(pdf_path)
    articles = []
    buffer = ""
    current_title = None
    pending_title = ""

    for i in range(start_page - 1, end_page):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip().replace("\n", " ")
                    font = span["font"]
                    size = span["size"]
                    flags = span["flags"]

                    is_title_candidate = (
                        text.startswith("제") and
                        "조" in text and
                        font == "H2gtrE" and
                        abs(size - 7.79) < 0.05 and
                        flags == 4
                    )

                    # 조문 제목 시작
                    if is_title_candidate:
                        if current_title and buffer.strip():
                            articles.append({
                                "id": f"{law_name}_{current_title}",
                                "title": f"{law_name} {current_title}",
                                "text": buffer.strip().replace("\n", " "),
                                "metadata": {
                                    "법령명": law_name,
                                    "조문": current_title
                                }
                            })

                        pending_title = text
                        if "(" in pending_title and not ")" in pending_title:
                            continue  # 다음 span에서 이어 붙임
                        else:
                            current_title = pending_title
                            buffer = ""
                            pending_title = ""

                    # 제목 이어붙이기
                    elif pending_title:
                        pending_title += text
                        if ")" in pending_title:
                            current_title = pending_title
                            buffer = ""
                            pending_title = ""
                    else:
                        # 📌 장/절 제목이면 본문에서 제외
                        if re.match(r"^제\d+장\s*\S+", text) or re.match(r"^제\d+절\s*\S+", text):
                            continue
                    
                        # 📌 일반 본문은 누적
                        buffer += text + " "

    # 마지막 조문 저장
    if current_title and buffer.strip():
        articles.append({
            "id": f"{law_name}_{current_title}",
            "title": f"{law_name} {current_title}",
            "text": buffer.strip().replace("\n", " "),
            "metadata": {
                "법령명": law_name,
                "조문": current_title
            }
        })

    return articles

# === 저장 ===
for law_name, (start, end) in law_sections.items():
    print(f"⏳ {law_name} 처리 중...")
    articles = extract_articles_precise_with_multiline_title(law_name, start, end)
    output_path = os.path.join(output_dir, f"{law_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ {law_name} 저장 완료 → {output_path}")

print("\n🎉 전체 작업 완료!")