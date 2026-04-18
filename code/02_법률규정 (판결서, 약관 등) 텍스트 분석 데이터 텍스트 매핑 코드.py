import os
import json
from pathlib import Path

# 원본 및 저장 경로
src_folder = Path(r"C:/Users/sdh/Desktop/판결문데이터/라벨링 데이터/판결문/전체")
dst_folder = Path(r"C:/Users/sdh/Desktop/판결문데이터/라벨링 데이터/판결문/전체_수정")
dst_folder.mkdir(parents=True, exist_ok=True)

# 필드별 숫자→텍스트 매핑 테이블
field_maps = {
    "caseField": {
        "1": "민사", "2": "형사", "3": "행정"
    },
    "detailField": {
        "1": "민사", "2": "신청", "3": "가사", "4": "특허", "5": "행정", "6": "형사"
    },
    "trailField": {
        "1": "1심", "2": "2심"
    },
    "acusr": {
        "1": "자연인", "2": "법인", "3": "국가", "4": "검사", "5": "기타"
    },
    "dedat": {
        "1": "자연인", "2": "법인", "3": "국가", "4": "검사", "5": "기타"
    },
    "disposalform": {
        "1": "손해배상금", "2": "손실보상금", "3": "산불출액", "4": "위자료",
        "5": "양육비", "6": "직역", "7": "금고", "8": "집행유예", "9": "벌금", "10": "취소"
    }
}

# JSON 변환 함수
def map_json_values(data):
    info = data.get("info", {})
    for field in ["caseField", "detailField", "trailField"]:
        if field in info and field in field_maps:
            info[field] = field_maps[field].get(info[field], info[field])

    concerned = data.get("concerned", {})
    for field in ["acusr", "dedat"]:
        if field in concerned and field in field_maps:
            concerned[field] = field_maps[field].get(concerned[field], concerned[field])

    if "disposal" in data and "disposalform" in data["disposal"]:
        value = data["disposal"]["disposalform"]
        data["disposal"]["disposalform"] = field_maps["disposalform"].get(value, value)

    return data

# 폴더 내 모든 JSON 처리
for file in src_folder.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = map_json_values(data)

        dst_file = dst_folder / file.name
        with open(dst_file, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)

        print(f"✅ 변환 완료: {file.name}")
    except Exception as e:
        print(f"⚠️ 에러 발생 ({file.name}): {e}")


# --------------------------------------------------

import os
import json
from pathlib import Path

# 경로 설정
src_folder = Path(r"C:\Users\sdh\Desktop\판결문데이터\라벨링 데이터\약관")
dst_folder = Path(r"C:\Users\sdh\Desktop\판결문데이터\라벨링 데이터\약관_전체_수정")
dst_folder.mkdir(parents=True, exist_ok=True)

# 키 이름 매핑
key_name_map = {
    "clauseField": "약관분야",
    "ftcCnclsns": "공정위 심결례",
    "clauseArticle": "약관조항",
    "dvAntageous": "유불리판단",
    "comProvision": "비교근거",
    "illdcssBasiss": "위법성 판단 근거",
    "relateLaword": "관련 법령",
    "unfavorableProvision": "불리한 조항 유형"
}

# 숫자 값 매핑 예시
# 위 이미지 데이터를 기반으로 직접 매핑 테이블 생성

mapping_dict_fixed = {
    "clauseField": {
        "1": "가맹계약", "2": "공급계약", "3": "분양계약", "4": "신탁계약", "5": "임대차계약", "6": "입소, 입주, 입점계약)",
        "7": "신용카드", "8": "은행여신", "9": "은행전자금융서비스", "10": "전자결제수단", "11": "전자금융거래",
        "12": "상해보험", "13": "손해보험", "14": "질병보험", "15": "연금보험", "16": "자동차보험", "17": "책임보험",
        "18": "화재보험", "19": "증권사1", "20": "증권사2", "21": "증권사3", "22": "여객운송", "23": "화물운송",
        "24": "개인정보취급방침", "25": "게임", "26": "국내·외 여행", "27": "결혼정보서비스", "28": "렌트(자동차 이외)",
        "29": "마일리지/포인트", "30": "보증 -245", "31": "사이버몰", "32": "산후조리원", "33": "상조서비스",
        "34": "상품권", "35": "생명보험", "36": "예식업", "37": "온라인서비스", "38": "자동차 리스 및 렌트",
        "39": "체육시설", "40": "택배)", "41": "통신, 방송서비스", "42": "교육", "43": "매매계약"
    },
    "ftcCnclsns": {
        "1": "해당", "2": "비해당"
    },
    "clauseArticle": {
        "1": "유리", "2": "불리"
    },
    "dvAntageous": {
        "1": "유리", "2": "불리"
    },
    "unfavorableProvision": {
        "1": "신의성실의 원칙 위반", "2": "개별금지 조항의 위반"
    }
}

mapping_dict_fixed["clauseField"]["5"]  # 예시 확인: "임대차계약" 출력 예상


# 변환 함수
def transform_json(data, key_map, value_map):
    new_data = {}
    for key, value in data.items():
        new_key = key_map.get(key, key)

        if isinstance(value, str) and key in value_map and value in value_map[key]:
            new_data[new_key] = value_map[key][value]
        else:
            new_data[new_key] = value
    return new_data

# 변환 실행
for file in src_folder.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = transform_json(data, key_name_map, mapping_dict_fixed)

        dst_file = dst_folder / file.name
        with open(dst_file, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)

        print(f"✅ 변환 완료: {file.name}")
    except Exception as e:
        print(f"⚠️ 에러 발생 ({file.name}): {e}")
