#!/usr/bin/env python3
# rebuild_mapping_by_index.py
import json, argparse

def build(npy_list, wav_list, out):
    with open(npy_list) as f:
        npy_rows = [line.rstrip('\n').split('\t') for line in f]
    with open(wav_list) as f:
        wav_rows = [line.rstrip('\n').split('\t') for line in f]

    assert len(npy_rows) == len(wav_rows), \
        "Feature list and WAV list have different lengths!"

    mapping = {}
    for (npy_path, npy_lbl), (wav_path, wav_lbl) in zip(npy_rows, wav_rows):
        # Sanity-check: labels should match (after Calm-removal they’re 0-6)
        if int(npy_lbl) != int(wav_lbl):
            print(f"⚠️ label mismatch  {npy_path}  vs  {wav_path}")
        mapping[npy_path] = wav_path          # 1-to-1 row mapping

    with open(out, 'w') as fp:
        json.dump(mapping, fp)
    print(f'✓ {len(mapping)} pairs written → {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy_list', default='dataset/test_list_features.txt')
    ap.add_argument('--wav_list', default='dataset/test_list.txt')
    ap.add_argument('--out',      default='dataset/npy_to_wav.json')
    build(**vars(ap.parse_args()))
