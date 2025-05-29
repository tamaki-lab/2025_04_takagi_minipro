import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import hydra
from omegaconf import DictConfig
import os


def extract_placeholders(json_text):
    """
    JSON文字列からplaceholdersの単語を抽出
    """
    try:
        data = json.loads(json_text)
        placeholders = data.get("placeholders", [])
        words = []
        for item in placeholders:
            item_words = re.findall(r'\b\w+\b', str(item))
            words.extend(item_words)
        return words
    except Exception:
        return []


def plot_top_words(word_counts, title, save_path, top_k=30):
    """
    単語カウントを受け取り、ヒストグラムを保存
    """
    most_common = word_counts.most_common(top_k)
    if not most_common:
        print(f"No data to plot for: {title}")
        return
    words, counts = zip(*most_common)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


@hydra.main(config_path="../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    csv_path = cfg.csv_path
    save_dir = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/analyze_text"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # result側
    result_words = []
    for text in df["result"]:
        result_words.extend(extract_placeholders(text))
    result_counts = Counter(result_words)
    plot_top_words(
        result_counts,
        "Top 30 Words in Predicted Placeholders",
        save_path=os.path.join(save_dir, "predicted_placeholders.png")
    )

    # ground truth側
    gt_words = []
    for text in df["ground_truth_json"]:
        gt_words.extend(extract_placeholders(text))
    gt_counts = Counter(gt_words)
    plot_top_words(
        gt_counts,
        "Top 30 Words in Ground Truth Placeholders",
        save_path=os.path.join(save_dir, "ground_truth_placeholders.png")
    )


if __name__ == "__main__":
    main()
