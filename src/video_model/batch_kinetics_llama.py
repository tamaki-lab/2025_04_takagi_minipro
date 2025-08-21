import os
import torch
import ffmpeg
import tempfile
import csv
import json
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForCausalLM, AutoProcessor
from omegaconf import DictConfig
import hydra
from typing import List, Tuple, Iterable


def process_video(video_path, question, resize_size, crop_size, model, processor, device, cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%03d.jpg")
        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=1)
            .output(frame_pattern, vframes=16)
            .run(quiet=True, overwrite_output=True)
        )

        resize = T.Resize(resize_size)
        crop = T.CenterCrop(crop_size)

        for f in sorted(os.listdir(tmpdir)):
            if f.endswith(".jpg"):
                img_path = os.path.join(tmpdir, f)
                img = Image.open(img_path).convert("RGB")
                img = crop(resize(img))
                img.save(img_path)

        processed_video_path = os.path.join(tmpdir, "resized_video.mp4")
        (
            ffmpeg
            .input(os.path.join(tmpdir, "frame_%03d.jpg"), framerate=1)
            .output(processed_video_path, vcodec="libx264", pix_fmt="yuv420p")
            .run(quiet=True, overwrite_output=True)
        )

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": processed_video_path, "fps": 1, "max_frames": 16}},
                    {"type": "text", "text": question},
                ]
            },
        ]

        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # 生成パラメータはYAMLに合わせて渡す（beam時はtop_k/top_p/temperatureは無視されます）
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
        )
        # サンプリング系がある場合もそのまま渡す（モデル側が無視/警告）
        if hasattr(cfg, "temperature"):
            gen_kwargs["temperature"] = cfg.temperature
        if hasattr(cfg, "top_k"):
            gen_kwargs["top_k"] = cfg.top_k
        if hasattr(cfg, "top_p"):
            gen_kwargs["top_p"] = cfg.top_p

        output_ids = model.generate(**inputs, **gen_kwargs)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

# ------------------------
# Kinetics-400: ルート解決 & 再帰列挙
# ------------------------


def resolve_scan_root(video_path: str) -> str:
    """
    - video_path がファイル: その親の親（= subset 直下: train/val/test）をルートにする
    - video_path がディレクトリ: そのディレクトリをルートにする
    """
    if os.path.isdir(video_path):
        return video_path
    if os.path.isfile(video_path):
        # .../val/<class>/<video> -> .../val
        return os.path.dirname(os.path.dirname(video_path))
    # 存在しない場合もそのまま返す（エラーは後段で）
    return video_path


def list_videos_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    found = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                found.append(os.path.join(r, fn))
    found.sort()
    return found


def iter_k400_videos(scan_root: str, start_index: int, num_videos: int,
                     exts: Tuple[str, ...] = (".mp4", ".mkv", ".webm", ".mov", ".avi")) -> Iterable[Tuple[str, str]]:
    """
    Kinetics-400 の再帰列挙: (video_path, class_label) を返す
    class_label は親ディレクトリ名を採用
    """
    all_files = list_videos_recursive(scan_root, exts)
    if not all_files:
        print(f"[WARN] No videos found under: {scan_root}")
        return
    end = start_index + num_videos if num_videos and num_videos > 0 else None
    for vp in all_files[start_index:end]:
        class_label = os.path.basename(os.path.dirname(vp))
        yield vp, class_label


@hydra.main(config_path="/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/config",
            config_name="video_llama", version_base=None)
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_visible
    device = torch.device("cuda:0")

    model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # スキャンルート決定
    scan_root = resolve_scan_root(cfg.video_path)
    if not os.path.exists(scan_root):
        raise FileNotFoundError(f"scan_root not found: {scan_root}")

    start_index = int(getattr(cfg, "start_video_id", 0))
    num_videos = int(getattr(cfg, "num_videos", 0))

    # 出力CSV準備
    csv_path = cfg.csv_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["video_path", "result_text", "ground_truth", "question"])
        first_write = (os.stat(csv_path).st_size == 0)
        if first_write:
            writer.writeheader()

        first_row = True
        for video_path, class_label in iter_k400_videos(scan_root, start_index, num_videos):
            print(f"Processing {video_path}")
            try:
                result_text = process_video(
                    video_path,
                    cfg.question,
                    cfg.resize,
                    cfg.crop,
                    model,
                    processor,
                    device,
                    cfg
                )
            except Exception as e:
                print(f"[ERROR] {video_path}: {e}")
                result_text = f"[ERROR] {e}"

            gt_info = {"label": class_label, "placeholders": []}
            gt_label_json = json.dumps(gt_info, ensure_ascii=False)

            writer.writerow({
                "video_path": video_path,
                "result_text": result_text,
                "ground_truth": gt_label_json,
                "question": cfg.question if first_row else ""
            })
            first_row = False

    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
