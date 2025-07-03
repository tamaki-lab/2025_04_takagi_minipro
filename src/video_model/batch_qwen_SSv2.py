from model.image_utils import concat_images_2x2_from_base
import os
import json
import csv
import torch
import uuid
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../src")))

torch.manual_seed(1234)


def init_qwen(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_device)
    print(f'Using CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.get('torch_dtype', 'bfloat16'))
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    return tokenizer, model


def run_qwen(tokenizer, model, image_input, question: str) -> str:
    query = tokenizer.from_list_format([
        {'image': image_input},
        {'text': question}
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


@hydra.main(
    config_path='/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/config',
    config_name='video_qwen',
    version_base=None
)
def main(cfg: DictConfig):
    # Load ground truth annotations
    annotation_path = cfg.get('gt_json', '')
    gt_dict = {}
    if annotation_path and os.path.exists(annotation_path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        gt_dict = {
            str(entry['id']): {
                'label': entry.get('label', ''),
                'placeholders': entry.get('placeholders', [])
            }
            for entry in gt_data
        }

    # Initialize model and tokenizer
    tokenizer, model = init_qwen(cfg)

    # Prepare CSV output (filename configurable via YAML)
    csv_path = cfg.summary_csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'video_path', 'result_text', 'ground_truth', 'question'
        ])
        # Write header if file is empty
        if os.stat(csv_path).st_size == 0:
            writer.writeheader()

        base_path = os.path.dirname(os.path.dirname(cfg.image_path))
        start = cfg.get('start_folder', 1)
        num = cfg.get('num_samples', 10)
        first_row = True

        tmp_dir = cfg.get('tmp_dir', '/tmp/qwen_tmp_images')
        os.makedirs(tmp_dir, exist_ok=True)

        for idx in range(start, start + num):
            folder = os.path.join(base_path, str(idx))
            if not os.path.isdir(folder):
                print(f'Skipping {idx}: not a directory')
                continue
            jpgs = sorted([fn for fn in os.listdir(folder) if fn.endswith('.jpg')])
            if len(jpgs) < 4:
                print(f'Skipping {idx}: not enough images (<4)')
                continue

            # Create 2x2 concatenated image
            sample_path = os.path.join(folder, jpgs[0])
            concat_img, selected = concat_images_2x2_from_base(sample_path)

            # Save to temporary file
            temp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.jpg")
            concat_img.save(temp_path)

            # Run inference on saved image path
            response = run_qwen(tokenizer, model, temp_path, cfg.question)

            # Prepare ground truth JSON
            gt_info = gt_dict.get(str(idx), {'label': '', 'placeholders': []})
            gt_label_json = json.dumps(gt_info, ensure_ascii=False)

            # Write row; only first row includes question
            writer.writerow({
                'video_path': sample_path,
                'result_text': response,
                'ground_truth': gt_label_json,
                'question': cfg.question if first_row else ''
            })
            first_row = False

    print('Batch processing complete.')


if __name__ == '__main__':
    main()
