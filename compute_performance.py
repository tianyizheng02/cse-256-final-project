import json
import natsort
import os
import statistics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


EVAL_DIR = "evaluation"


sns.set_theme(style="whitegrid", context="paper", rc={"figure.dpi": 500})


def compute_file_accuracy(file: str) -> float:
    with open(file, "r") as f:
        question_list = json.load(f)
    assert isinstance(question_list, list)
    assert isinstance(question_list[0], dict)
    filtered_questions = [  # Ignore malformed questions (non-int answers)
        question for question in question_list if question["final_answer"].isdigit()
    ]
    accuracy = statistics.mean(
        question["final_answer"] in question["model_output"]
        or f"{int(question["final_answer"]):,}" in question["model_output"]
        for question in filtered_questions
    )
    return 100 * accuracy


def draw_model_accuracy_histogram(model_dir: str, response_dirs: list[str]) -> None:
    for response_dir in response_dirs:
        response_files = natsort.natsorted(
            os.listdir(f"{EVAL_DIR}/{model_dir}/{response_dir}")
        )
        accuracies = [
            compute_file_accuracy(
                f"{EVAL_DIR}/{model_dir}/{response_dir}/{response_file}"
            )
            for response_file in response_files
        ]
        print(
            f"{model_dir:<15}{response_dir:<16}"
            f"mean accuracy %: {f"{statistics.mean(accuracies):.4f}":<11}"
            f"std dev: {statistics.stdev(accuracies):.4f}"
        )
        sns.histplot(
            data=accuracies,
            kde=True,
            binwidth=3,
            label=response_dir,
        )
    histogram_file = f"{model_dir}_accuracies.png"
    plt.title(model_dir)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Frequency")
    plt.xticks(np.linspace(50, 100, 6))
    plt.yticks(np.arange(0, 21, 2))
    plt.legend()
    plt.savefig(histogram_file, bbox_inches="tight")
    print(f"Accuracy histogram for {model_dir} saved to {histogram_file}")
    plt.clf()


def main() -> None:
    model_dirs = natsort.natsorted(os.listdir(EVAL_DIR))
    for model_dir in model_dirs:
        response_dirs = natsort.natsorted(os.listdir(f"{EVAL_DIR}/{model_dir}"))
        draw_model_accuracy_histogram(model_dir, response_dirs)


if __name__ == "__main__":
    main()
