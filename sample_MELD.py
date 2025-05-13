import os
import random
import shutil
from collections import defaultdict

# === PATHS ===
input_file = "MELD/data/test_list.txt"
base_audio_dir = "//MELD/datatrain_splits_audio_organized"
output_list_1 = "MELD/datadata/test_half1.txt"
output_list_2 = "MELD/datadata/test_half2.txt"
output_dir_1 = "MELD/datatest_half1"
output_dir_2 = "MELD/datatest_half2"

# === READ AND PARSE FILE ===
with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

data = []
for line in lines:
    path, emotion = line.rsplit("\t", 1)
    actor = path.split("/")[3]
    data.append((actor, int(emotion), line))

# === STRATIFIED SPLIT ===
strata = defaultdict(list)
for actor, emotion, line in data:
    strata[(actor, emotion)].append(line)

half1, half2 = [], []
random.seed(42)

for group, items in strata.items():
    items_copy = items[:]
    random.shuffle(items_copy)
    mid = len(items_copy) // 2
    half1.extend(items_copy[:mid])
    half2.extend(items_copy[mid:])

# === SAVE TEXT FILES ===
with open(output_list_1, "w") as f1:
    for line in half1:
        f1.write(f"{line}\n")
with open(output_list_2, "w") as f2:
    for line in half2:
        f2.write(f"{line}\n")

# === MOVE AUDIO FILES ===
def copy_files(file_list, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for line in file_list:
        rel_path = line.split("\t")[0]  # Extract just the relative audio path
        src_path = os.path.join(base_audio_dir, rel_path)  # Full source path
        dst_path = os.path.join(dest_dir, rel_path)         # Full destination path

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy if exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Missing file: {src_path}")


copy_files(half1, output_dir_1)
copy_files(half2, output_dir_2)
