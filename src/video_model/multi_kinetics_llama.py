import os
import json
import csv
import ffmpeg
import torch
import tempfile
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForCausalLM, AutoProcessor
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import List, Tuple, Iterable, Dict, Any
import multiprocessing as mp
import time

# ========= 既存のユーティリティ群 =========


def process_video(video_path, question, resize_size, crop_size, model, processor, device, cfg_dict):
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

        gen_kwargs = dict(
            max_new_tokens=cfg_dict.get("max_new_tokens", 256),
            num_beams=cfg_dict.get("num_beams", 1),
            do_sample=cfg_dict.get("do_sample", False),
        )
        # サンプリング系（beamでも渡してOK。モデル側が無視/警告）
        for k in ("temperature", "top_k", "top_p"):
            if cfg_dict.get(k) is not None:
                gen_kwargs[k] = cfg_dict[k]

        output_ids = model.generate(**inputs, **gen_kwargs)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response


def resolve_scan_root(video_path: str) -> str:
    if os.path.isdir(video_path):
        return video_path
    if os.path.isfile(video_path):
        return os.path.dirname(os.path.dirname(video_path))
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
    all_files = list_videos_recursive(scan_root, exts)
    if not all_files:
        print(f"[WARN] No videos found under: {scan_root}")
        return
    end = start_index + num_videos if num_videos and num_videos > 0 else None
    for vp in all_files[start_index:end]:
        class_label = os.path.basename(os.path.dirname(vp))
        yield vp, class_label


# ========= 並列：ワーカー / ライター =========

def worker_proc(gpu_id: int, task_q: mp.Queue, result_q: mp.Queue, cfg_dict: Dict[str, Any], model_name: str):
    # 各ワーカーで1GPUを占有。0番として扱うため可視GPUを限定
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルは各ワーカーで一度だけロード（再利用）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": 0},  # このプロセスからはGPU0=実GPU{gpu_id}
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    question = cfg_dict["question"]
    resize_size = cfg_dict["resize"]
    crop_size = cfg_dict["crop"]

    while True:
        item = task_q.get()
        if item is None:  # 終了シグナル
            break
        video_path, class_label = item
        try:
            result_text = process_video(
                video_path, question, resize_size, crop_size, model, processor, device, cfg_dict
            )
        except Exception as e:
            print(f"[ERROR][GPU{gpu_id}] {video_path}: {e}")
            result_text = f"[ERROR] {e}"

        gt_info = {"label": class_label, "placeholders": []}
        result_q.put({
            "video_path": video_path,
            "result_text": result_text,
            "ground_truth": json.dumps(gt_info, ensure_ascii=False),
        })


def writer_proc(csv_path: str, result_q: mp.Queue, total_rows: int, question: str, write_question_once: bool = True):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    wrote_header = os.path.exists(csv_path) and os.stat(csv_path).st_size > 0
    wrote_question = False

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "result_text", "ground_truth", "question"])
        if not wrote_header:
            writer.writeheader()

        for _ in range(total_rows):
            row = result_q.get()
            if write_question_once:
                row["question"] = question if not wrote_question else ""
                wrote_question = True
            else:
                row["question"] = question
            writer.writerow(row)
            f.flush()


# ========= メイン =========

@hydra.main(config_path="/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/config",
            config_name="video_llama", version_base=None)
def main(cfg: DictConfig):
    t0 = time.perf_counter()

    # cfg から必要なものだけシリアライズ（mp経由で渡すため）
    cfg_dict = {
        "question": cfg.question,
        "resize": cfg.resize,
        "crop": cfg.crop,
        "max_new_tokens": getattr(cfg, "max_new_tokens", 256),
        "num_beams": getattr(cfg, "num_beams", 1),
        "do_sample": getattr(cfg, "do_sample", False),
        "temperature": getattr(cfg, "temperature", None),
        "top_k": getattr(cfg, "top_k", None),
        "top_p": getattr(cfg, "top_p", None),
    }

    model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"

    # スキャン対象一覧をまず作る（件数が分かればCSVライターが終端を判断できる）
    scan_root = resolve_scan_root(cfg.video_path)
    if not os.path.exists(scan_root):
        raise FileNotFoundError(f"scan_root not found: {scan_root}")
    start_index = int(getattr(cfg, "start_video_id", 0))
    num_videos = int(getattr(cfg, "num_videos", 0))
    tasks = list(iter_k400_videos(scan_root, start_index, num_videos))
    if not tasks:
        print("[WARN] no tasks")
        return

    # GPUリスト（例: "0,1,2" / "0"）
    devs = str(getattr(cfg, "device_visible", "0")).split(",")
    gpu_ids = [int(d.strip()) for d in devs if d.strip() != ""]

    # 並列の足回り
    ctx = mp.get_context("spawn")  # どの環境でも安定
    task_q: mp.Queue = ctx.Queue(maxsize=len(tasks))
    result_q: mp.Queue = ctx.Queue(maxsize=len(tasks))

    # タスク投入
    for t in tasks:
        task_q.put(t)

    # ワーカー起動（GPUごと1プロセス；GPUメモリ節約＆モデル再利用）
    workers = []
    for gid in gpu_ids:
        p = ctx.Process(target=worker_proc, args=(gid, task_q, result_q, cfg_dict, model_name))
        p.start()
        workers.append(p)

    # ライター起動（1つのCSVに直書き）
    csv_path = cfg.csv_path
    writer = ctx.Process(target=writer_proc, args=(csv_path, result_q, len(tasks), cfg.question, True))
    writer.start()

    # 終了シグナルをワーカーへ
    for _ in workers:
        task_q.put(None)

    # join
    for p in workers:
        p.join()
    writer.join()

    elapsed = time.perf_counter() - t0
    print(f"Results saved to: {csv_path}")
    print(f"Total runtime: {elapsed:.1f} s ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
