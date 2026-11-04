from concurrent.futures import ProcessPoolExecutor

import yaml
from PIL import Image
from tqdm import tqdm

from src.data.nih import NIH_DIR, PROJECT_ROOT, find_images

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "images_resized"

#Resize a single image, converts to greyscale and save
def resize_one(args):

    src_path, dst_path, size = args

    with Image.open(src_path) as img:
        img = img.convert("L")

        img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
        img_resized.save(dst_path)

def main():
    #Load parameters from yaml
    params_path = PROJECT_ROOT / "params.yaml"
    print(f"Loading parameters from {params_path}")

    params = yaml.safe_load(params_path.read_text())["resize"]
    size = params["size"]
    print(f"Config loaded: target size: {size}")

    #Finding raw images
    print(f"Scanning raw images in {NIH_DIR}")
    image_paths = find_images(NIH_DIR)
    image_count = len(image_paths)
    print(f"Found {image_count} raw images")

    #checking
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    #task tuple: (src_path, dst_path, size)
    tasks = [(src_path, OUT_DIR / name, size) for name, src_path in image_paths.items()]

    print(f"Resizing {image_count} images in parallel to {OUT_DIR}...")

    # As cpu has multiple core, so instead of opening and resizing images one by one(single thread), ProcessPoolExecutor
    #fires up multiple worker processes to resize images and simultaneously across all CPU cores.
    with ProcessPoolExecutor() as executor:
        #chunk size 256 optimal for multiprocessing overhead balance
        #so instead of 1 chunk size which is default of executor map; here we are sending 256 images
        list(
            # tqd, wraps around the parellel execution pool, as each batch finishes, tqdm updates the progress bar on screen
            tqdm(
                #this distributes the list of image-resizing tasks (tasks) across worker processes in batches (chunksize)
                executor.map(resize_one, tasks, chunksize=256),
                total=len(tasks),
                desc="Resizing Images",
            )
        )

    #fail check
    output_files = list(OUT_DIR.glob("*.png"))
    output_count = len(output_files)

    assert output_count == image_count


if __name__ == "__main__":
    main()
