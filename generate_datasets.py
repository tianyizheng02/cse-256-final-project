import json
import os
import random
import re

import dotenv
import natsort

from openai import OpenAI
from pydantic import BaseModel


TEMPLATE_DIR = "test_templates"
TEMPLATE_NO_OP_DIR = "test_templates_no_op"
SYNTHETIC_DIR = "synthetic_data"
SYNTHETIC_NO_OP_DIR = "synthetic_data_no_op"
NUM_SHOTS = 8
NUM_SYNTHETIC_DATASETS = 50
QUESTIONS_PER_TEMPLATE = 1


class FilledTemplates(BaseModel):
    question: str
    answer: str


def fill_templates(gpt_query: str) -> dict[str, str] | None:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",  # gpt-4o-mini for development, gpt-4o for final dataset
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. The user will give you a question template with a question statement, variable definitions, and conditions. The user will also give you an answer template with an answer statement using the same variables as the question statement. Initialize values for the defined variables, and ensure that the variable values satisfy the conditions. Populate both the question statement and the answer statement with those variable values. Your numbers do not have to be realistic or nice to work with, as long as the conditions are satisfied. Preserve all formatting, including line breaks.",
            },
            {"role": "user", "content": gpt_query},
        ],
        response_format=FilledTemplates,
    )
    message = completion.choices[0].message
    if message.parsed:
        question_str = message.parsed.question.strip()
        final_answer_index = message.parsed.answer.index("####")  # "#### <answer>"
        solution_str = message.parsed.answer[:final_answer_index].strip()
        final_answer_str = message.parsed.answer[final_answer_index + 5 :].strip()
        return {
            "question": question_str,
            "solution": solution_str,
            "final_answer": final_answer_str,
        }
    print(message.refusal)
    return None


def generate_synthetic_questions_from_template(
    template_dir: str,
    question_template_file: str,
    answer_template_file: str,
    repetitions: int = 1,
) -> list[dict[str, str] | None]:
    with open(f"{template_dir}/{question_template_file}", "r") as f:
        question_template = f.read()

    with open(f"{template_dir}/{answer_template_file}", "r") as f:
        answer_template = f.read()

    gpt_query = f"""Question template:
```
{question_template}
```

Answer template:
```
{answer_template}
```
"""
    return [
        filled_template
        for _ in range(repetitions)
        if (filled_template := fill_templates(gpt_query)) is not None
    ]


def generate_synthetic_dataset(
    template_dir: str, template_files: list[str], output_file: str
) -> None:
    synthetic_questions_all: list[dict[str, str]] = []
    for i in range(0, len(template_files), 2):
        answer_template_file, question_template_file, *_ = template_files[i : i + 2]
        assert "question" in question_template_file and "answer" in answer_template_file

        # Check for matching question/answer nums
        question_nums = re.findall(r"\d+", question_template_file)
        answer_nums = re.findall(r"\d+", answer_template_file)
        assert len(question_nums) == len(answer_nums) == 1
        assert question_nums[0] == answer_nums[0]
        question_num = int(question_nums[0])

        print(f"Generating synthetic questions for template {question_num}...")
        synthetic_questions = generate_synthetic_questions_from_template(
            template_dir,
            question_template_file,
            answer_template_file,
            repetitions=QUESTIONS_PER_TEMPLATE,
        )
        if synthetic_questions is not None:
            synthetic_questions_all.extend(synthetic_questions)

    with open(output_file, "w") as f:
        json.dump(synthetic_questions_all, f, indent=2)
    print(f"Synthetic questions written to {output_file}")


def generate_prompting_shots(
    template_dir: str,
    template_files: list[str],
    output_file: str,
    shot_indices: list[int],
) -> None:
    synthetic_prompting_shots_all: list[dict[str, str]] = []
    shot_indices.sort()
    curr_shot = 0
    for i in range(0, len(template_files), 2):
        if curr_shot >= len(shot_indices):
            break

        answer_template_file, question_template_file, *_ = template_files[i : i + 2]
        assert "question" in question_template_file and "answer" in answer_template_file

        # Check for matching question/answer nums
        question_nums = re.findall(r"\d+", question_template_file)
        answer_nums = re.findall(r"\d+", answer_template_file)
        assert len(question_nums) == len(answer_nums) == 1
        assert question_nums[0] == answer_nums[0]
        question_num = int(question_nums[0])

        if question_num == shot_indices[curr_shot]:
            print(f"Generating synthetic prompting shot for template {question_num}...")
            synthetic_questions = generate_synthetic_questions_from_template(
                template_dir,
                question_template_file,
                answer_template_file,
            )
            if synthetic_questions is not None:
                synthetic_prompting_shots_all.extend(synthetic_questions)
            curr_shot += 1

    assert len(synthetic_prompting_shots_all) == len(shot_indices)
    with open(output_file, "w") as f:
        json.dump(synthetic_prompting_shots_all, f, indent=2)
    print(f"Synthetic prompting shots written to {output_file}")


def main() -> None:
    # Sort file names in "natural" order, with numbers in increasing order,
    # rather than lexicographical order
    print("Loading and sorting template directories...")
    template_files = natsort.natsorted(os.listdir(TEMPLATE_DIR))
    template_no_op_files = natsort.natsorted(os.listdir(TEMPLATE_NO_OP_DIR))
    print("Template directories loaded and sorted\n")
    assert len(template_files) == len(template_no_op_files)

    if not os.path.exists(SYNTHETIC_DIR):
        os.makedirs(SYNTHETIC_DIR)
    if not os.path.exists(SYNTHETIC_NO_OP_DIR):
        os.makedirs(SYNTHETIC_NO_OP_DIR)

    prompting_shot_indices = random.sample(range(len(template_files) // 2), k=NUM_SHOTS)

    print(f"Generating synthetic data for {NUM_SHOTS}-shot prompting...")
    generate_prompting_shots(
        TEMPLATE_DIR,
        template_files,
        output_file=f"{SYNTHETIC_DIR}/synthetic_prompting_shots.json",
        shot_indices=prompting_shot_indices,
    )
    print()

    print(f"Generating synthetic no-op data for {NUM_SHOTS}-shot prompting...")
    generate_prompting_shots(
        TEMPLATE_NO_OP_DIR,
        template_no_op_files,
        output_file=f"{SYNTHETIC_NO_OP_DIR}/synthetic_prompting_shots_no_op.json",
        shot_indices=prompting_shot_indices,
    )
    print()

    for i in range(NUM_SYNTHETIC_DATASETS):
        print(f"Generating synthetic dataset {i}...")
        generate_synthetic_dataset(
            TEMPLATE_DIR,
            template_files,
            output_file=f"{SYNTHETIC_DIR}/synthetic_dataset_{i}.json",
        )
        print(f"Generating synthetic no-op dataset {i}...")
        generate_synthetic_dataset(
            TEMPLATE_NO_OP_DIR,
            template_no_op_files,
            output_file=f"{SYNTHETIC_NO_OP_DIR}/synthetic_dataset_no_op_{i}.json",
        )
        print()
    print("Synthetic dataset generation complete!")


if __name__ == "__main__":
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    random.seed(256)

    main()
