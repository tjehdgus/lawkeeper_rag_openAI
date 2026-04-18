import os
import json
from pykospacing import Spacing

# === 설정 ===
input_dir = r"C:\Users\sdh\Desktop\ai_agent_project\law_json_output"
output_dir = os.path.join(input_dir, "spaced_output")
os.makedirs(output_dir, exist_ok=True)

spacing = Spacing()

# === ①~㊿ 특수항 기호 치환용 딕셔너리 자동 생성 ===
article_number_map = {}

# ①~⑳: U+2460 ~ U+2473
for i in range(1, 21):
    article_number_map[chr(0x2460 + i - 1)] = f"{i}항"

# ㉑~㉟: U+3251 ~ U+325F
for i in range(21, 36):
    article_number_map[chr(0x3251 + i - 21)] = f"{i}항"

# ㊱~㊿: U+32B1 ~ U+32BF
for i in range(36, 51):
    article_number_map[chr(0x32B1 + i - 36)] = f"{i}항"

def replace_article_numbers(text):
    for symbol, replacement in article_number_map.items():
        text = text.replace(symbol, replacement)
    return text

def process_file(filepath, output_path):
    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        original_text = article.get("text", "")
        if original_text.strip():
            try:
                # 항목 치환
                replaced_text = replace_article_numbers(original_text)

                # 띄어쓰기 교정
                corrected = spacing(replaced_text)

                article["text"] = corrected
            except Exception as e:
                print(f"❌ 오류 발생 - {article.get('id', '알 수 없음')}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료 → {os.path.basename(output_path)}")

# === 폴더 내 전체 JSON 처리 ===
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"📝 처리 중: {filename}")
        process_file(input_path, output_path)

print("\n🎉 ①~㊿ 항목 기호 치환 + 띄어쓰기 교정 완료! 결과는 spaced_output 폴더에 저장되었습니다.")