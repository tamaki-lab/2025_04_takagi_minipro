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

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
        )
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response


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

    start_id = cfg.start_video_id
    num_videos = cfg.num_videos
    base_video_dir = "/mnt/NAS-TVS872XT/dataset/something-something-v2/video"
    annotation_path = "/mnt/NAS-TVS872XT/dataset/something-something-v2/anno/something-something-v2-train.formatted.json"

    # GTラベルとプレースホルダの読み込み
    with open(annotation_path, "r") as f:
        gt_data = json.load(f)

    gt_dict = {
        str(entry["id"]): {
            "label": entry.get("label", ""),
            "placeholders": entry.get("placeholders", [])
        }
        for entry in gt_data
    }

    output_dir = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/video_llama"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "video_llama50000.csv")

    with open(csv_path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["video_path", "result_text", "ground_truth", "question"])

        first_write = os.stat(csv_path).st_size == 0
        if first_write:
            writer.writeheader()

        first_row = True
        for i in range(start_id, start_id + num_videos):
            video_path = os.path.join(base_video_dir, f"{i}.webm")
            if not os.path.exists(video_path):
                print(f"Skipping {video_path} (not found)")
                continue

            print(f"Processing {video_path}")
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

            gt_info = gt_dict.get(str(i), {"label": "", "placeholders": []})
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
