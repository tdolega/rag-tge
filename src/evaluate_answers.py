import argparse
import json
from tqdm.auto import tqdm
import nltk
import os

from common.utils import (
    get_start_idx,
    get_dataset,
    remove_citations,
    merge_sentences,
    clean_sentence,
    get_refs,
    filename_to_obj,
    obj_to_filename,
)
from common.nlis import get_nli, add_nli_args
from common.llms import get_llm, add_llm_args
from common.sims import get_sim, add_sim_args

nltk.download("punkt", quiet=True)

dataset_by_id = {item["id"]: item for item in get_dataset()}


def evaluate_citations(passages, answer, nli, language="english"):
    sentences = nltk.sent_tokenize(answer, language=language)
    n_sentences = len(sentences)
    # * sentences that are correctly cited by at least one passage
    supported = [0 for _ in range(n_sentences)]
    # * are all citations correct
    correct_citations = [[] for _ in range(n_sentences)]
    # * all citations
    citations = [[] for _ in range(n_sentences)]  # list of citations for each sentence
    # * out of range citations
    out_of_range = [0 for _ in range(n_sentences)]

    for sentence_idx, sentence in enumerate(sentences):
        decited_sentence = remove_citations(sentence)

        refs = get_refs(sentence)
        n_out_of_range = len([ref for ref in refs if ref >= len(passages)])
        if n_out_of_range > 0:  # * citation out of range
            out_of_range[sentence_idx] = n_out_of_range
            refs = [ref for ref in refs if ref < len(passages)]  # * remove out of range citations
        if len(refs) == 0:  # * no citations
            continue
        citations[sentence_idx] = refs

        # * calculate the recall score
        joint_passage = "\n".join([passages[ref] for ref in refs])
        joint_entail = nli.evaluate(joint_passage, decited_sentence)
        supported[sentence_idx] = joint_entail

        # * calculate the precision score if applicable
        if joint_entail and len(refs) > 1:
            # * precision check: did the model cite any unnecessary documents?
            for passage_idx in refs:
                # * does sentence entail the passage
                passage = passages[passage_idx]
                single_entail = nli.evaluate(passage, decited_sentence)

                if single_entail:
                    correct_citations[sentence_idx].append(True)
                else:
                    # * overcite check: does rest of the joint passage entail the sentence
                    rest_refs = [ref for ref in refs if ref != passage_idx]
                    passage = "\n".join([passage[passage_idx] for passage_idx in rest_refs])
                    rest_entail = nli.evaluate(passage, decited_sentence)
                    correct_citations[sentence_idx].append(rest_entail == 0)
        else:
            # * only one citation
            correct_citations[sentence_idx].append(joint_entail)

    n_total_citations = sum([len(refs) for refs in citations])
    n_correct_citations = sum([sum(refs) for refs in correct_citations])
    ais_recall = sum(supported) / n_sentences if n_sentences > 0 else 0
    ais_precision = n_correct_citations / n_total_citations if n_total_citations > 0 else 0
    n_correctly_multicited_sentences = sum([all(correct_citations[sentence_idx]) for sentence_idx in range(n_sentences) if len(citations[sentence_idx]) > 1])
    n_overcitations = sum([len(refs) - sum(correct_citations[sentence_idx]) for sentence_idx, refs in enumerate(citations) if len(refs) > 1])

    return {
        "ais_recall": ais_recall,  # * percent of correctly cited sentences out of all sentences
        "ais_precision": ais_precision,  # * percent of correctly cited documents out of all citations
        "n_sentences": n_sentences,
        "n_total_citations": n_total_citations,
        "n_correct_citations": n_correct_citations,
        "n_correctly_multicited_sentences": n_correctly_multicited_sentences,
        "n_overcitations": n_overcitations,
        "sentences": sentences,
        "supported": supported,
        "citations": citations,
        "correct_citations": correct_citations,
        "out_of_range": out_of_range,
    }


def evaluate_correctness(dataset_row, answer, nli):
    # * calculate word overlap
    clean_answer = clean_sentence(answer)
    gt_answer = dataset_row["answer"]
    clean_gt = clean_sentence(gt_answer)
    answer_words = set(clean_answer.split())
    gt_words = set(clean_gt.split())
    answer_overlap = len(answer_words.intersection(gt_words)) / len(gt_words) if len(gt_words) > 0 else 0

    # * calculate answer entailment
    question = dataset_row["question"]
    answer_entail = nli.evaluate(f"{question} {answer}", f"{question} {gt_answer}")

    # * calculate citations overlap with ground truth
    refs = set(get_refs(answer))
    gt_refs = set()
    for supporting_fact_title in dataset_row["supporting_facts"]["title"]:
        citation_idx = dataset_row["context"]["title"].index(supporting_fact_title)
        gt_refs.add(citation_idx)
    citations_recall = len(refs.intersection(gt_refs)) / len(gt_refs) if len(gt_refs) > 0 else 0
    citations_precision = len(refs.intersection(gt_refs)) / len(refs) if len(refs) > 0 else 0

    return {
        "answer_overlap": answer_overlap,
        "answer_entail": answer_entail,
        "citations_recall": citations_recall,
        "citations_precision": citations_precision,
    }


def evaluate_quality(question, answer, llm, sim, language="english"):
    system_prompt = "Generate a question that results in the following answer. Respond with question and nothing else."
    if language != "english":
        system_prompt += f" You must generate question in {language} language."
    _, new_question = llm.generate(system_prompt=system_prompt, user_prompt=f"Answer: {answer}")
    if new_question.lower().startswith("question: "):
        new_question = new_question[len("question: ") :]
    new_question = new_question.split("\n")[0]
    _, answer_relevance = sim.calculate(question, new_question)

    return {
        "answer_relevance": answer_relevance,
        "new_question": new_question,
    }


def evaluate_answer(dataset_row, answer, nli, llm, sim):
    question = dataset_row["question"]
    answer = answer[:1024]  # * truncate the answer to speed up the evaluation, if answer is larger that this, it's never valid anyway
    passages = [merge_sentences(passage_sentences) for passage_sentences in dataset_row["context"]["sentences"]]
    return {
        "citations": evaluate_citations(passages, answer, nli),
        "correctness": evaluate_correctness(dataset_row, answer, nli),
        "quality": evaluate_quality(question, answer, llm, sim),
    }


def evaluate_answers(input_file, nli, llm, sim):
    with open(input_file, "r") as handle:
        answers = [json.loads(line) for line in handle]
    params = filename_to_obj(input_file)
    params["nli"] = nli.short_model_name
    params["ellm"] = llm.short_model_name
    params["sim"] = sim.short_model_name
    output_file = obj_to_filename(params)
    output_file = f"../data/results/{output_file}"
    start_idx = get_start_idx(output_file)
    output_handle = open(output_file, "a")
    for question_idx, answers_row in tqdm(enumerate(answers), total=len(answers)):
        if question_idx < start_idx:
            continue
        question_id, answers = answers_row["question_id"], answers_row["answers"]
        dataset_row = dataset_by_id[question_id]

        evaluations = [evaluate_answer(dataset_row, answer, nli, llm, sim) for answer in answers]

        output_json = {"question_id": question_id, "evaluations": evaluations}
        output_handle.write(json.dumps(output_json) + "\n")
        output_handle.flush()
    output_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate_answers",
        description="Evaluate the answers generated by the model.",
    )
    parser.add_argument(
        "--answers",
        type=str,
        help='path to the answers file, e.g. "../data/answers/xyz.jsonl"',
    )
    add_nli_args(parser)
    add_llm_args(parser)
    add_sim_args(parser)
    args = parser.parse_args()
    print("args:", args)

    nli = get_nli(args)
    llm = get_llm(args)
    sim = get_sim(args)

    if args.answers:
        evaluate_answers(args.answers, nli, llm, sim)
    else:
        answers_path = "../data/answers"
        for filename in os.listdir(answers_path):
            if not filename.endswith(".jsonl"):
                print(f"skipping {filename}")
                continue
            path = os.path.join(answers_path, filename)
            print(f"evaluating {path}")
            evaluate_answers(path, nli, llm, sim)
