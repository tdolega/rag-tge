import json
import os
import random
import re

all_lines = []
for filename in os.listdir("original"):
    if not filename.endswith(".jsonl"):
        continue
    with open("original/" + filename, "r") as f:
        data = f.readlines()
        all_lines.extend(data)

random.seed(50)
random.shuffle(all_lines)

print("Total lines:", len(all_lines))
# print(json.dumps(json.loads(all_lines[0]), indent=4, ensure_ascii=False))

all_answers = set()


def line_ok(line):
    jline = json.loads(line)
    if "error" in jline:
        return False
    if not jline["evaluation"]:
        return False
    ct = jline["evaluation"]["citations"]
    if ct["ais_precision"] < 1 or ct["ais_recall"] < 1 or ct["n_overcitations"] > 0 or ct["n_total_citations"] == 0:
        return False
    question = jline["question"].strip()
    answer = jline["response"]["answer"].strip()
    pair = (question, answer)
    if pair in all_answers:
        return False
    all_answers.add(pair)
    return True


lines_kept = [line for line in all_lines if line_ok(line)]
print("Lines kept:", len(lines_kept))

lines_kept2 = []
for i, line in enumerate(lines_kept):
    jline = json.loads(line)
    jline["id"] = i
    line = json.dumps(jline, ensure_ascii=False)
    lines_kept2.append(line)
lines_kept = lines_kept2


with open("answers_all.jsonl", "w") as f:
    for line in lines_kept:
        f.write(line + "\n")


def fix_citations(text):
    pattern = r"\.\s*\[(\d+)\]"
    replacement = r" [\1]."
    text = re.sub(pattern, replacement, text)

    pattern_duplicate = r"\[(\d+)\]\[\1\]"
    replacement_duplicate = r"[\1]"
    text = re.sub(pattern_duplicate, replacement_duplicate, text)

    return text


DIVIDER = "\n= = = = = = =\n\n"


def save(path, lines):
    with open(path, "w") as f:
        for line in lines:
            input_line = json.loads(line)

            pid = input_line["id"]
            question = input_line["question"].strip()
            answer = input_line["response"]["answer"].strip()

            answer = fix_citations(answer)

            assert DIVIDER not in question
            assert DIVIDER not in answer

            f.write(f"> ID:{pid}\n")
            f.write(f"> PYTANIE:\n{question}\n")
            f.write(f"> ODPOWIEDŹ:\n{answer}\n")
            f.write(f"> OK:1\n")
            f.write(DIVIDER)


# divide lines_kept into 2
half_len = len(lines_kept) // 2
lines1 = lines_kept[:half_len]
lines2 = lines_kept[half_len:]
save("original/review1.txt", lines1)
save("original/review2.txt", lines2)

print("Done")
