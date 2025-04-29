import os
import itertools
import re

# Hyperparameters
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64]
model_types = ['BiLSTM', 'StackedLSTM', 'LSTMwithSDPA', 'StackedLSTMwithSDPA']
weight_decays = [1e-6, 1e-5, 1e-4]

# Paths
os.makedirs('tuning_logs', exist_ok=True)
base_command = "CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/bi_lstm.yml"

# === Real-time Parsing Function ===
def parse_log_file(filepath):
    best_acc = -1
    best_acc_epoch = -1
    lowest_loss = float('inf')
    lowest_loss_epoch = -1
    pattern = re.compile(r"Test epoch: (\d+),.*?loss: ([\d\.]+), accuracy: ([\d\.]+)")

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))

                if acc > best_acc:
                    best_acc = acc
                    best_acc_epoch = epoch
                if loss < lowest_loss:
                    lowest_loss = loss
                    lowest_loss_epoch = epoch

    return best_acc, best_acc_epoch, lowest_loss, lowest_loss_epoch

# === Run and parse each job one at a time ===
combinations = list(itertools.product(learning_rates, batch_sizes, model_types, weight_decays))

for lr, bs, model, wd in combinations:
    overwrite_str = (
        f"optimizer_conf.optimizer_args.lr={lr},"
        f"dataset_conf.dataLoader.batch_size={bs},"
        f"model_conf.model={model},"
        f"optimizer_conf.optimizer_args.weight_decay={wd}"
    )
    log_file = f"tuning_logs/{model}_lr{lr}_bs{bs}_wd{wd}.log"
    command = f"{base_command} --overwrites \"{overwrite_str}\" > {log_file} 2>&1"

    print(f"\n🚀 Launching: {command}")
    import subprocess

    try:
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user. Exiting...")
        break
      # <-- Waits until training finishes

    # 🔍 Immediately parse the log after training ends
    if os.path.exists(log_file):
        acc, acc_ep, loss, loss_ep = parse_log_file(log_file)
        print(f"✅ Finished: {os.path.basename(log_file)}")
        print(f"   🔵 Best Accuracy: {acc:.5f} at epoch {acc_ep}")
        print(f"   🟠 Lowest Loss  : {loss:.5f} at epoch {loss_ep}")
    else:
        print(f"⚠️ Log file not found: {log_file}")
