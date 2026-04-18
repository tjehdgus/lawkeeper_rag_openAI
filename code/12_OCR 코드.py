import requests
import time
import os
import mimetypes
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO

# Azure OCR 설정
endpoint = "https://1231.cognitiveservices.azure.com/"
subscription_key = os.getenv("AZURE_OCR_KEY")
ocr_url = endpoint + "vision/v3.2/read/analyze"

# 분석할 파일 경로
file_path = r"D:\AI\다운로드.jpg"

# 전체 결과 저장용 리스트
all_text = []

def prepare_ocr_input(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        print(f"📄 총 {total_pages}페이지의 PDF 분석을 시작합니다...\n")
        for page_num in range(total_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            buffer = BytesIO()
            writer.write(buffer)
            buffer.seek(0)
            yield buffer.getvalue(), "application/pdf", page_num + 1
    else:
        mime_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as f:
            data = f.read()
        print(f"🖼 이미지 파일 분석을 시작합니다...\n")
        yield data, mime_type or "application/octet-stream", 1

# 분석 시작
for data, content_type, page_num in prepare_ocr_input(file_path):
    print(f"🔍 페이지 {page_num} 분석 요청 중...")

    for attempt in range(3):
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": content_type
        }
        response = requests.post(ocr_url, headers=headers, data=data)

        if response.status_code == 202:
            break
        elif response.status_code == 429:
            print(f"[페이지 {page_num}] ❌ 요청 제한(429). 60초 대기 후 재시도 ({attempt + 1}/3)...")
            time.sleep(60)
        else:
            print(f"[페이지 {page_num}] ❌ 요청 실패: {response.status_code} - {response.text}")
            break
    else:
        print(f"[페이지 {page_num}] ❌ 요청 재시도 실패. 건너뜁니다.")
        continue

    operation_url = response.headers["Operation-Location"]
    retry_count = 0
    while retry_count < 3:
        try:
            result = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": subscription_key})
            if result.status_code == 429:
                retry_count += 1
                print(f"[페이지 {page_num}] ❌ 결과 확인 제한(429). 60초 대기 후 재시도 ({retry_count}/3)...")
                time.sleep(60)
                continue
            result_json = result.json()
        except Exception as e:
            print(f"[페이지 {page_num}] ❌ JSON 파싱 실패: {e}")
            break

        status = result_json.get("status")
        if status is None:
            print(f"[페이지 {page_num}] ❌ 잘못된 응답 구조:\n{result_json}")
            break
        elif status == "succeeded":
            break
        elif status == "failed":
            print(f"[페이지 {page_num}] ❌ 분석 실패")
            break

        time.sleep(1)
    else:
        print(f"[페이지 {page_num}] ❌ 결과 확인 재시도 실패. 페이지 건너뜁니다.")
        continue

    # 콘솔에 바로 출력
    print(f"\n📄 [페이지 {page_num}]\n" + "-" * 40)
    for read_result in result_json.get("analyzeResult", {}).get("readResults", []):
        for line in read_result.get("lines", []):
            print(line["text"])

print("\n✅ 전체 OCR 처리가 완료되었습니다.")