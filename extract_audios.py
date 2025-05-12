from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import ffmpeg

input_dir = 'MELD/MELD.Raw/train/train_splits'  # original .mp4 root
output_root = 'data'  # output root containing actor subfolders

def extract_audio(full_path):
    rel_path = os.path.relpath(full_path, input_dir)  # relative path like Actor_01/clip.mp4
    base_name = os.path.splitext(rel_path)[0] + '.wav'  # change extension
    output_path = os.path.join(output_root, base_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        ffmpeg.input(full_path).output(output_path, acodec='pcm_s16le').run(quiet=True, overwrite_output=True)
        return f"✓ Done: {rel_path}"
    except Exception as e:
        return f"✗ Failed: {rel_path} ({e})"

def find_mp4_files(root):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root)
        for filename in filenames if filename.endswith('.mp4')
    ]

if __name__ == "__main__":
    all_mp4_files = find_mp4_files(input_dir)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_audio, path): path for path in all_mp4_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Audio"):
            print(future.result())
