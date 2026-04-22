import argparse

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset

from scripts.paths import prepare_output_path, resolve_path


def flip_answer(answer: str) -> str:
    if answer == "choice_0.png":
        return "choice_1.png"
    if answer == "choice_1.png":
        return "choice_0.png"
    raise ValueError(f"Unexpected answer label: {answer}")


def build_swapped_rows(ds: Dataset) -> list[dict]:
    rows = []

    for sample in ds:
        rows.append(
            {
                "question": sample["question"].strip(),
                "choices": "choice_0.png,choice_1.png",
                "answer": flip_answer(sample["answer"]),
                "query_image": sample["query_image"],
                "choice_image_0": sample["choice_image_1"],
                "choice_image_1": sample["choice_image_0"],
                "source": "iconqa_swap",
                "orig_question_id": sample.get("question_id", ""),
                "orig_answer": sample["answer"],
            }
        )
    return rows


def write_arrow_dataset(ds: Dataset, output_path: str) -> None:
    table = ds.data.table
    with pa.OSFile(output_path, "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/icon-qa-train.arrow")
    parser.add_argument("--output", default="custom_data/custom_flipped.arrow")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = prepare_output_path(args.output)

    base_ds = Dataset.from_file(str(input_path))
    rows = build_swapped_rows(base_ds)[:1000]
    custom_ds = Dataset.from_list(rows)
    write_arrow_dataset(custom_ds, str(output_path))

    # verify the written dataset
    reloaded = Dataset.from_file(str(output_path))
    print(f"Wrote {len(reloaded)} rows to {output_path}")
    print("Features:", reloaded.features)
    print("First sample answer:", reloaded[0]["answer"])


if __name__ == "__main__":
    main()
