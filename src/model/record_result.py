import os
import csv
import html
from omegaconf import DictConfig


def record_result(cfg: DictConfig,
                  model_name: str,
                  max_new_tokens: int,
                  prompt: str,
                  result_text: str,
                  image_path: str,
                  generation_kwargs: dict,
                  generation_mode: str):
    result_text = extract_assistant_only(result_text)

    if cfg.output_format == "csv":
        record_topk_csv(cfg, model_name, prompt, result_text, generation_kwargs, generation_mode)
    else:
        record_to_text(cfg, model_name, max_new_tokens, prompt,
                       result_text, image_path, generation_kwargs, generation_mode)


def extract_assistant_only(text: str) -> str:
    """
    llava出力からASSISTANT:以降のみを抽出
    """
    if "ASSISTANT:" in text:
        return text.split("ASSISTANT:", 1)[1].strip()
    return text.strip()


def record_to_text(
        cfg,
        model_name,
        max_new_tokens,
        prompt,
        result_text,
        image_path,
        generation_kwargs,
        generation_mode):
    folder_path = f"{cfg.save_dir.root + cfg.model_name}"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{model_name}_{max_new_tokens}.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("【Prompt】\n" + prompt + "\n")
        f.write("【Result】\n" + result_text + "\n")
        f.write("【Image Path】\n" + image_path + "\n")
        f.write("【Generation Mode】\n" + generation_mode + "\n")
        f.write("【Generation kwargs】\n")
        for key, value in generation_kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n\n")

    print(f"Text result saved to: {file_path}")


# for escaping in case kwargs have special chars like < >
def record_topk_csv(cfg, model_name, prompt, result_text, generation_kwargs, generation_mode):

    folder_path = os.path.join(cfg.save_dir.root, model_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{model_name}_topk.csv")

    # 初期列構成
    default_fields = ["prompt", "generation_mode", "generation_kwargs"]
    top_k_fields = [f"top_k={k}" for k in range(1, 21)]
    fieldnames = default_fields + top_k_fields

    # ファイル読み込み
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # 念のため、足りない列があれば補う（途中中断などの安全対策）
            for field in fieldnames:
                if field not in reader.fieldnames:
                    fieldnames.append(field)
    else:
        rows = []

    top_k = generation_kwargs.get("top_k", "")
    column_name = f"top_k={top_k}"

    # kwargsをテキスト化（HTMLエスケープで安全にCSV化）
    kwargs_text = "\n".join(f"{k}: {v}" for k, v in generation_kwargs.items())
    kwargs_text = html.escape(kwargs_text)

    # prompt + generation_mode で行を特定
    existing_row = next((row for row in rows
                         if row["prompt"] == prompt and row["generation_mode"] == generation_mode), None)

    if existing_row:
        existing_row["generation_kwargs"] = kwargs_text
        if column_name in existing_row and existing_row[column_name]:
            existing_row[column_name] += "\n\n" + result_text
        else:
            existing_row[column_name] = result_text
    else:
        new_row = {field: "" for field in fieldnames}
        new_row["prompt"] = prompt
        new_row["generation_mode"] = generation_mode
        new_row["generation_kwargs"] = kwargs_text
        new_row[column_name] = result_text
        rows.append(new_row)

    # 保存
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Top-k CSV result saved to: {file_path}")
