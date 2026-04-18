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

circled_number_map = {
    '①': '1항 ', '②': '2항 ', '③': '3항 ', '④': '4항 ', '⑤': '5항 ',
    '⑥': '6항 ', '⑦': '7항 ', '⑧': '8항 ', '⑨': '9항 ', '⑩': '10항 ',
    '⑪': '11항 ', '⑫': '12항 ', '⑬': '13항 ', '⑭': '14항 ', '⑮': '15항 ',
    '⑯': '16항 ', '⑰': '17항 ', '⑱': '18항 ', '⑲': '19항 ', '⑳': '20항 ',
    '㉑': '21항 ', '㉒': '22항 ', '㉓': '23항 ', '㉔': '24항 ', '㉕': '25항 ',
    '㉖': '26항 ', '㉗': '27항 ', '㉘': '28항 ', '㉙': '29항 ', '㉚': '30항 ',
    '㉛': '31항 ', '㉜': '32항 ', '㉝': '33항 ', '㉞': '34항 ', '㉟': '35항 ',
    '㊱': '36항 ', '㊲': '37항 ', '㊳': '38항 ', '㊴': '39항 ', '㊵': '40항 ',
    '㊶': '41항 ', '㊷': '42항 ', '㊸': '43항 ', '㊹': '44항 ', '㊺': '45항 ',
    '㊻': '46항 ', '㊼': '47항 ', '㊽': '48항 ', '㊾': '49항 ', '㊿': '50항 '
}

def clean_text(text):
    # 괄호 안에 한자 제거
    text = re.sub(r"[（(][^()\n\r]{0,20}[\u3400-\u9FFF]+[^()\n\r]{0,20}[)）]", "", text)
    # 방송통신대 푸터 제거
    text = re.sub(r"방송통신대학교.*법학과", "", text)
    # 페이지 번호 제거 (- 301 - 같은)
    text = re.sub(r"-\s*\d+\s*-", "", text)
    # 여분 공백 제거
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_articles_precise_with_multiline_title(law_name, start_page, end_page):
    doc = fitz.open(pdf_path)
    articles = []
    buffer = ""
    current_title = None
    pending_title = ""
    is_after_supplement = False

    for i in range(start_page - 1, end_page):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    raw_text = span["text"].strip().replace("\n", " ")
                    font = span["font"]
                    size = span["size"]
                    flags = span["flags"]

                    if "부칙" in raw_text:
                        is_after_supplement = True

                    if is_after_supplement:
                        continue  # 부칙 이후는 무시

                    is_title_candidate = (
                        raw_text.startswith("제") and
                        "조" in raw_text and
                        font == "H2gtrE" and
                        abs(size - 7.79) < 0.05 and
                        flags == 4
                    )

                    if is_title_candidate:
                        if current_title and buffer.strip():
                            articles.append({
                                "id": f"{law_name}_{current_title}",
                                "title": f"{law_name} {current_title}",
                                "text": buffer.strip(),
                                "metadata": {
                                    "법령명": law_name,
                                    "조문": current_title
                                }
                            })
                        pending_title = raw_text
                        if "(" in pending_title and not ")" in pending_title:
                            continue
                        else:
                            current_title = pending_title
                            buffer = ""
                            pending_title = ""
                    elif pending_title:
                        pending_title += raw_text
                        if ")" in pending_title:
                            current_title = pending_title
                            buffer = ""
                            pending_title = ""
                    else:
                        if re.match(r"^제\d+장\s*\S+", raw_text) or re.match(r"^제\d+절\s*\S+", raw_text):
                            continue
                        buffer += raw_text + " "

    if current_title and buffer.strip():
        articles.append({
            "id": f"{law_name}_{current_title}",
            "title": f"{law_name} {current_title}",
            "text": buffer.strip(),
            "metadata": {
                "법령명": law_name,
                "조문": current_title
            }
        })

    # 후처리
    for article in articles:
        for k, v in circled_number_map.items():
            article["text"] = article["text"].replace(k, v)

        article["id"] = clean_text(article["id"])
        article["title"] = clean_text(article["title"])
        article["text"] = clean_text(article["text"])
        article["metadata"]["조문"] = clean_text(article["metadata"]["조문"])

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