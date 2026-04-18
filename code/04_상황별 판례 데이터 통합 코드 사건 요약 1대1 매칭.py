import os
import json
from datetime import datetime

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fully_merge_legal_json(data1, data2, label1="source_tl", label2="source_ts"):
    return {
        "case_id": data2.get("사건번호") or data1.get("info", {}).get("caseNo"),
        "title": data2.get("사건명") or data1.get("info", {}).get("caseNm"),
        "court": data2.get("법원명") or data1.get("info", {}).get("courtNm"),
        "judgment_date": data2.get("선고일자") or data1.get("info", {}).get("judmnAdjuDe"),
        "court_type": data1.get("info", {}).get("courtType"),
        "judgment_type": data2.get("판결유형"),
        label1: data1,
        label2: data2,
        "meta": {
            "source_merged": [label1, label2],
            "merge_time": datetime.now().isoformat()
        }
    }

def extract_suffix(folder_name):
    parts = folder_name.split("_", 2)
    return parts[-1] if len(parts) == 3 else None

def merge_json_by_custom_folder_mapping(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    folders = os.listdir(root_dir)
    ts_folders = [f for f in folders if f.startswith("TS_")]
    vs_folders = [f for f in folders if f.startswith("VS_")]

    for ts_folder in ts_folders:
        suffix = extract_suffix(ts_folder)
        if suffix is None:
            continue
        tl_folder = f"TL_{suffix}"

        ts_path = os.path.join(root_dir, ts_folder)
        tl_path = os.path.join(root_dir, tl_folder)

        if not os.path.exists(tl_path):
            print(f"❌ 대응 TL 폴더 없음: {tl_path}")
            continue

        ts_files = {os.path.splitext(f)[0]: os.path.join(ts_path, f)
                    for f in os.listdir(ts_path) if f.endswith(".json")}
        tl_files = {os.path.splitext(f)[0]: os.path.join(tl_path, f)
                    for f in os.listdir(tl_path) if f.endswith(".json")}

        common_keys = set(ts_files.keys()) & set(tl_files.keys())
        for key in common_keys:
            try:
                data1 = load_json(tl_files[key])
                data2 = load_json(ts_files[key])
                merged = fully_merge_legal_json(data1, data2, "source_tl", "source_ts")
                save_path = os.path.join(output_dir, f"{key}_merged.json")
                with open(save_path, "w", encoding="utf-8-sig") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 병합 실패 - {key}: {e}")

    for vs_folder in vs_folders:
        suffix = extract_suffix(vs_folder)
        if suffix is None:
            continue
        vl_folder = f"VL_{suffix}"

        vs_path = os.path.join(root_dir, vs_folder)
        vl_path = os.path.join(root_dir, vl_folder)

        if not os.path.exists(vl_path):
            print(f"❌ 대응 VL 폴더 없음: {vl_path}")
            continue

        vs_files = {os.path.splitext(f)[0]: os.path.join(vs_path, f)
                    for f in os.listdir(vs_path) if f.endswith(".json")}
        vl_files = {os.path.splitext(f)[0]: os.path.join(vl_path, f)
                    for f in os.listdir(vl_path) if f.endswith(".json")}

        common_keys = set(vs_files.keys()) & set(vl_files.keys())
        for key in common_keys:
            try:
                data1 = load_json(vl_files[key])
                data2 = load_json(vs_files[key])
                merged = fully_merge_legal_json(data1, data2, "source_vl", "source_vs")
                save_path = os.path.join(output_dir, f"{key}_merged.json")
                with open(save_path, "w", encoding="utf-8-sig") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 병합 실패 - {key}: {e}")

    print(f"\n✅ 병합 완료! 결과 저장 폴더: {output_dir}")

# 사용 예시
merge_json_by_custom_folder_mapping(
    root_dir=r"F:\data\법률\상황압축해제",
    output_dir=r"F:\data\법률\병합결과_판례"
)


# --------------------------------------------------

import os
import shutil

def move_all_decision_case_files(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    folders = [f for f in os.listdir(root_dir) if "심결례" in f]
    total_moved = 0

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                src_path = os.path.join(folder_path, file)
                dst_path = os.path.join(output_dir, file)

                # 이름 충돌 방지: 폴더 이름 prefix로 추가
                if os.path.exists(dst_path):
                    filename, ext = os.path.splitext(file)
                    dst_path = os.path.join(output_dir, f"{folder}_{filename}{ext}")

                try:
                    shutil.copy2(src_path, dst_path)
                    total_moved += 1
                except Exception as e:
                    print(f"❌ 복사 실패: {file} - {e}")

    print(f"\n✅ 심결례 파일 {total_moved}개 복사 완료 → {output_dir}")

# 사용 예시
move_all_decision_case_files(
    root_dir=r"F:\data\법률\상황압축해제",
    output_dir=r"F:\data\법률\병합결과_판례"
)
