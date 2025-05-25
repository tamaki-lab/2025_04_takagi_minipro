import pandas as pd
import json
import re

# CSVファイルのパス
csv_path = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/batch_SSv2_result/1000beam5_focus_movement.csv"

# CSVを読み込む
df = pd.read_csv(csv_path)

# エラー判定関数（行数が6行以上 または 内容が不正ならエラー）


def is_valid_result(text):
    if pd.isna(text):
        return False  # NaNはエラーとみなす

    lines = str(text).strip().splitlines()
    if len(lines) > 5:
        return False  # 行数が多すぎる場合はエラー

    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            return False
        if not all(k in data for k in ["label", "template", "placeholders"]):
            return False

        # ① labelに [1] や [something] が含まれていないこと
        if re.search(r"\[\s*\d+\s*\]|\[something\]", data["label"]):
            return False

        # ② placeholders に [1], [2] 等の記号が含まれていないこと
        if any(re.fullmatch(r"\[\s*\d+\s*\]", p) or "something" in p for p in data["placeholders"]):
            return False

        # ③ template が [1] -> [2] -> ... 形式など記号だけで構成されていないこと
        if re.fullmatch(r"(\[\d+\]\s*->\s*)+\[\d+\]", data["template"]):
            return False

        return True  # すべての条件を満たした場合のみ有効
    except Exception:
        return False  # JSONとして壊れていたらエラー


# 判定列を追加（True = 正常, False = エラー）
df["is_valid_json"] = df["result"].apply(is_valid_result)

# 統計の算出
total = len(df)
invalid = (~df["is_valid_json"]).sum()
error_rate = invalid / total

# 結果出力
print(f"全データ数: {total}")
print(f"不正な出力件数: {invalid}")
print(f"エラー率: {error_rate:.2%}")
