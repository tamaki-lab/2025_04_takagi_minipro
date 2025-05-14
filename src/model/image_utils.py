from PIL import Image
import os


def concat_images_2x2_from_base(image_path: str) -> Image.Image:
    base_dir = os.path.dirname(image_path)
    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".jpg")])

    if len(all_files) < 4:
        raise ValueError(f"Not enough images in {base_dir} to create a 2x2 grid.")

    total = len(all_files)
    indices = [int(i * total / 5) for i in range(1, 5)]  # 20%, 40%, 60%, 80%
    selected_files = [all_files[i] for i in indices]

    # 選ばれた画像のファイル名を表示
    print("Selected images for 2x2 grid:")
    for fname in selected_files:
        print(f"- {fname}")

    images = [Image.open(os.path.join(base_dir, fname)) for fname in selected_files]
    img_width, img_height = images[0].size
    resized_images = [img.resize((img_width, img_height)) for img in images]

    new_img = Image.new("RGB", (2 * img_width, 2 * img_height))
    new_img.paste(resized_images[0], (0, 0))
    new_img.paste(resized_images[1], (img_width, 0))
    new_img.paste(resized_images[2], (0, img_height))
    new_img.paste(resized_images[3], (img_width, img_height))

    return new_img, selected_files
