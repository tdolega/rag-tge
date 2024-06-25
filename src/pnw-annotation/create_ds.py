import pandas as pd
import re
import datasets


def parse_file_to_dataframe(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split content by the delimiters
        entries = content.strip().split("= = = = = = =")

        data = {"id": [], "pytanie": [], "odpowiedz": [], "ok": []}

        for entry in entries:
            if not entry.strip():
                continue

            try:
                id_match = re.search(r"> ID:(\d+)", entry)
                question_match = re.search(r"> PYTANIE:\n(.*?)\n> ODPOWIEDŹ:", entry, re.DOTALL)
                answer_match = re.search(r"> ODPOWIEDŹ:\n(.*?)\n> OK:", entry, re.DOTALL)
                ok_match = re.search(r"> OK:(\d+)", entry)

                if id_match and question_match and answer_match and ok_match:
                    data["id"].append(id_match.group(1))
                    data["pytanie"].append(question_match.group(1).strip())
                    data["odpowiedz"].append(answer_match.group(1).strip())
                    data["ok"].append(ok_match.group(1))
                else:
                    raise ValueError(f"Unexpected structure in entry: {entry}")

            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue

        df = pd.DataFrame(data)
        return df

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")


review1 = parse_file_to_dataframe("reviewed/review1.txt")
review2 = parse_file_to_dataframe("reviewed/review2.txt")
review = pd.concat([review1, review2], ignore_index=True)
review["id"] = review["id"].astype(int)

print(f"Length of the dataframe: {len(review)}")
review = review[review["ok"] == "1"]
print(f"Length of the dataframe after filtering: {len(review)}")
print("review dataframe:", review)

# open "answers_all.jsonl" into dataframe
answers_all = pd.read_json("answers_all.jsonl", lines=True)
print("answers_all dataframe:", answers_all)

# left merge using id
merged = pd.merge(review, answers_all, left_on="id", right_on="id", how="left")
print("merged dataframe:", merged)
print("columns:", merged.columns)


# map into new dataset
def converter(row):
    prompt = row["last_prompt"][0]
    human_prompt_start_index = prompt.find("\nHuman: ")
    if human_prompt_start_index == -1:
        raise ValueError(f"Human prompt not found in: {prompt}")
    prompt = prompt[human_prompt_start_index + len("\nHuman: ") :]
    human_prompt_question_index = prompt.rfind("Pytanie: ")
    if human_prompt_question_index == -1:
        raise ValueError(f"Question not found in: {prompt}")
    prompt = prompt[: human_prompt_question_index + len("Pytanie: ")]
    prompt += row["pytanie"]

    return {
        "prompt": prompt,
        "answer": row["odpowiedz"],
    }


ds = merged.apply(converter, axis=1, result_type="expand")
print(ds)
print("first prompt:", ds.iloc[0]["prompt"])
print("first answer:", ds.iloc[0]["answer"])

dataset = datasets.Dataset.from_pandas(ds)
dataset = dataset.shuffle(seed=50)
dataset = dataset.train_test_split(test_size=10, seed=50)
print(dataset)
dataset.save_to_disk("rag-tge_trl-pwr-dataset")
dataset.push_to_hub("rag-tge_trl-pwr-dataset", private=True)
