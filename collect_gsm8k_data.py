import os
from datasets import load_dataset


NUM_SAMPLES = 20
DATA_DIR = "test_templates"


def main() -> None:
    print("Loading GSM8K test data from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    print("Data loaded")

    print(f"Writing first {NUM_SAMPLES} samples to text files...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for i, row in enumerate(ds.take(NUM_SAMPLES)):
        question_file = f"{DATA_DIR}/template_{i}_question.txt"
        answer_file = f"{DATA_DIR}/template_{i}_answer.txt"
        if not os.path.exists(question_file):
            with open(question_file, "w") as f:
                f.write(row["question"])
        if not os.path.exists(answer_file):
            with open(answer_file, "w") as f:
                f.write(row["answer"])
    print(f"Samples written to {DATA_DIR}/")


if __name__ == "__main__":
    main()
