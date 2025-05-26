import pandas as pd
import json
import re
import hydra
from omegaconf import DictConfig


def is_valid_result(text):
    """
    JSON構造・内容に関するチェック関数。
    templateはエラー判定から除外。
    """
    if pd.isna(text):
        return False

    lines = str(text).strip().splitlines()
    if len(lines) > 5:
        return False

    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            return False

        # "label"と"placeholders"は必須（templateは除外）
        if not all(k in data for k in ["label", "placeholders"]):
            return False

        # labelに [1] や [something] が含まれていないか
        if re.search(r"\[\s*\d+\s*\]|\[something\]", data["label"]):
            return False

        # placeholders に [1], [2], or "something" が含まれていないか
        if any(re.fullmatch(r"\[\s*\d+\s*\]", p) or "something" in p for p in data["placeholders"]):
            return False

        # placeholdersが空でないことを確認
        if not data["placeholders"]:
            return False

        # labelの長さチェック
        if len(data["label"].split()) < 3:
            return False

        # labelとplaceholdersの重複を確認
        if not any(p in data["label"] for p in data["placeholders"]):
            return False

        return True
    except Exception:
        return False


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path

    df = pd.read_csv(csv_path)

    # 判定列を追加
    df["is_valid_json"] = df["result"].apply(is_valid_result)

    # 統計
    total = len(df)
    invalid = (~df["is_valid_json"]).sum()
    error_rate = invalid / total

    # 出力
    print(f"全データ数: {total}")
    print(f"不正な出力件数: {invalid}")
    print(f"エラー率: {error_rate:.2%}")


if __name__ == "__main__":
    main()
