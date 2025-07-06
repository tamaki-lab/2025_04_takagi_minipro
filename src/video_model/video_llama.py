import os
import torch
import tempfile
import ffmpeg
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForCausalLM, AutoProcessor

import hydra
from omegaconf import DictConfig


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

    original_video_path = cfg.video_path
    question = cfg.question
    resize_size = cfg.resize
    crop_size = cfg.crop

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%03d.jpg")
        (
            ffmpeg
            .input(original_video_path)
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
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(response)

        # 結果の保存
        output_dir = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/video_llama"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "result.txt")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("【Video Path】\n")
            f.write(original_video_path + "\n")
            f.write("【Result】\n")
            f.write(response + "\n")
            f.write("=" * 50 + "\n\n")

        print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    main()
