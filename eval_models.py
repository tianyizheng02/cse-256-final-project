import json
import os

import dotenv
import natsort

from openai import OpenAI


EVAL_DIR = "evaluation"
SYNTHETIC_DIR = "synthetic_data"
SYNTHETIC_NO_OP_DIR = "synthetic_data_no_op"


class MathProblem:
    question: str
    solution: str
    model_output: str | None
    final_answer: str

    def __init__(self, problem_dict: dict[str, str]) -> None:
        self.question = problem_dict["question"]
        self.solution = problem_dict["solution"]
        self.model_output = problem_dict.get("model_output")
        self.final_answer = problem_dict["final_answer"]


def collect_prompt_and_dataset(dataset_dir: str) -> tuple[str, list[str]]:
    dataset_files = os.listdir(dataset_dir)
    prompting_shots_files = [file for file in dataset_files if "prompting" in file]
    assert len(prompting_shots_files) == 1
    prompting_shots_file = prompting_shots_files[0]
    dataset_files = natsort.natsorted(
        set(dataset_files).difference(prompting_shots_files)
    )
    return prompting_shots_file, dataset_files


def eval_model(
    model_name: str,
    dataset_dir: str,
    prompting_shots_file: str,
    dataset_files: list[str],
    output_dir: str,
    richer_prompting: bool = True,
) -> None:
    print("Preparing prompting shots...")
    with open(f"{dataset_dir}/{prompting_shots_file}", "r") as f:
        prompting_shots_json = json.load(f)
    assert isinstance(prompting_shots_json, list)
    prompting_shots = [MathProblem(shot) for shot in prompting_shots_json]

    model_msgs = [
        {
            "role": "system",
            "content": "As an expert problem solver, solve step by step the following mathematical questions.",
        }
    ]
    for shot in prompting_shots:
        if not richer_prompting and "\n\n" in shot.solution:
            explain_nums, original_solution, *_ = shot.solution.split("\n\n")
            # print(f"{explain_nums = }")
            # print(f"{original_solution = }")
            # exit()
            shot.solution = original_solution.strip()
        model_msgs.extend(
            [
                {"role": "user", "content": shot.question},
                {
                    "role": "assistant",
                    "content": f"Let's think step by step.  {shot.solution}{"" if shot.solution[-1] == "." else "."}  The final answer is {shot.final_answer}.",
                },
            ]
        )
    print("Prompting shots prepared\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, dataset_file in enumerate(dataset_files):
        print(f"Evaluating {dataset_file}...")
        with open(f"{dataset_dir}/{dataset_file}", "r") as f:
            dataset_json = json.load(f)
        assert isinstance(dataset_json, list)
        dataset = [MathProblem(shot) for shot in dataset_json]

        for j, sample in enumerate(dataset):
            completion = client.chat.completions.create(
                model=model_name,
                messages=model_msgs + [{"role": "user", "content": sample.question}],
            )
            dataset[j].model_output = completion.choices[0].message.content.strip()

        output_file = f"{output_dir}/model_output_{i}.json"
        with open(output_file, "w") as f:
            json.dump([sample.__dict__ for sample in dataset], f, indent=2)
        print(f"Model output written to {output_file}")


def main() -> None:
    print("Loading and sorting dataset directories...")
    synthetic_prompting_shots_file, synthetic_dataset_files = (
        collect_prompt_and_dataset(SYNTHETIC_DIR)
    )
    no_op_prompting_shots_file, no_op_dataset_files = collect_prompt_and_dataset(
        SYNTHETIC_NO_OP_DIR
    )
    print("Dataset directories loaded and sorted\n")

    models = ["gpt-4o-mini", "gpt-4o"]
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
    for model in models:
        print(f"Evaluating {model}...\n")
        eval_model(
            model,
            dataset_dir=SYNTHETIC_DIR,
            prompting_shots_file=synthetic_prompting_shots_file,
            dataset_files=synthetic_dataset_files,
            output_dir=f"{EVAL_DIR}/{model}/synthetic",
        )
        print()
        eval_model(
            model,
            dataset_dir=SYNTHETIC_NO_OP_DIR,
            prompting_shots_file=no_op_prompting_shots_file,
            dataset_files=no_op_dataset_files,
            output_dir=f"{EVAL_DIR}/{model}/no_op",
            richer_prompting=False,
        )
        print()
        eval_model(
            model,
            dataset_dir=SYNTHETIC_NO_OP_DIR,
            prompting_shots_file=no_op_prompting_shots_file,
            dataset_files=no_op_dataset_files,
            output_dir=f"{EVAL_DIR}/{model}/no_op_richer",
            richer_prompting=True,
        )
        print(f"\n{model} evaluation complete!\n")


if __name__ == "__main__":
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    main()
