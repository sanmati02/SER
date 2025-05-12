import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_emotion_lines(file_path, name):
    labels = ["Neutral", "Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]
    with open(file_path, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        path, emotion = line.rsplit("\t", 1)
        actor = path.split("/")[0]  # Adjusted to match your folder structure
        data.append((actor, int(emotion), line))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["actor", "emotion", "line"])

    # Plot 1: Overall Emotion Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="emotion", data=df, palette="muted")
    plt.title(f"Overall Emotion Distribution – {name}")
    plt.xlabel("Emotion Label")
    plt.ylabel("Count")
    plt.legend(title="Emotion", labels = labels)
    plt.tight_layout()
    plt.savefig(f"MELD/data_balanced/visualizations_reformated/overall_emotion_distribution_{name}.png")  # ✅ Fixed
    plt.close()

    # Plot 2: Emotion Distribution Per Actor
    plt.figure(figsize=(12, 6))
    sns.countplot(x="actor", hue="emotion", data=df, palette="tab10")
    plt.title(f"Emotion Distribution Per Actor – {name}")
    plt.xlabel("Actor")
    plt.ylabel("Count")
    plt.legend(title="Emotion", labels = labels)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"MELD/data_balanced/visualizations_reformated/emotion_distribution_per_actor_{name}.png")  # ✅ Fixed
    plt.close()


def check(train_file, test_file):
    with open(train_file, "r") as f:
        train_lines = f.readlines()
    with open(test_file, "r") as f:
        test_lines = f.readlines()
    print("TRAIN")
    train_data = set()
    for line in train_lines:
        line = line.strip()
        if not line:
            continue
        path, _ = line.rsplit("\t", 1)
        if path in train_data:
            print(path)
        train_data.add(path)
    print("TEST")
    test_data = set()
    for line in test_lines:
        line = line.strip()
        if not line:
            continue
        path, _ = line.rsplit("\t", 1)
        if path in test_data:
            print(path)
        test_data.add(path)
    print(len(train_data), len(test_data))
    return train_data.intersection(test_data)

if __name__ == "__main__":
    print(check("MELD/data_balanced/train_reformatted.txt", "MELD/data_balanced/test_reformatted.txt"))
    # parse_emotion_lines("MELD/data_balanced/test_reformatted.txt", 'Test Balanced')
    # parse_emotion_lines("MELD/data_balanced/train_reformatted.txt", 'Train Balanced')



