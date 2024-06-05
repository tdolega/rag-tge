import argparse
from queue import Queue
from threading import Thread
import logging
import traceback
import signal
from dotenv import load_dotenv
import os
from functools import cache

from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from common.nlis import get_nli, add_nli_args
from common.llms import get_llm, add_llm_args
from common.sims import get_sim, add_sim_args
from evaluate_answers import evaluate_citations, evaluate_quality, evaluate_similarity

load_dotenv()

assert len(os.getenv("RAGTGE_USER", "").strip()) > 0, "RAGTGE_USER must be set"
assert len(os.getenv("RAGTGE_PASSWORD", "").strip()) > 0, "RAGTGE_PASSWORD must be set"

users = {
    os.getenv("RAGTGE_USER"): generate_password_hash(os.getenv("RAGTGE_PASSWORD")),
}

app = Flask(__name__)
auth = HTTPBasicAuth()
request_queue = Queue()
logging.basicConfig(level=logging.INFO)

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


def process_requests(evaluator):
    while True:
        job = request_queue.get()
        if job is None:
            break
        try:
            response = evaluator.evaluate(**job["params"])
        except Exception as e:
            logging.error(f"error during evaluation: {e}")
            logging.info(traceback.format_exc())
            response = {"error": str(e)}
        job["response"].append(response)
        request_queue.task_done()


@app.route("/evaluate", methods=["GET"])
@auth.login_required
def enqueue_request():
    passages = request.args.getlist("passages")
    question = request.args.get("question", None)
    answer = request.args.get("answer", None)
    check_quality = request.args.get("check_quality", None)
    check_similarity = request.args.get("check_similarity", None)
    language = request.args.get("language", "polish")

    def check_string(name, value):
        if type(value) != str:
            return f"Parameter '{name}' must be a string."
        if len(value) == 0:
            return f"Parameter '{name}' must be a non-empty string."
        return None

    def check_bool(name, value):
        if type(value) == bool:
            return None
        if type(value) != str:
            return f"Parameter '{name}' must be a string or a boolean."
        if value.lower() not in ["true", "false"]:
            return f"Parameter '{name}' must be a string 'true' or 'false' or a boolean."
        return None

    def check_stringlist(name, value):
        if not type(value) == list:
            return f"Parameter '{name}' must be a list."
        if not all(type(p) == str for p in value):
            return f"Parameter '{name}' must be a list of strings."
        if len(value) == 0:
            return f"Parameter '{name}' must be a non-empty list."
        return None

    for name, value in [
        ("question", question),
        ("answer", answer),
        ("language", language),
    ]:
        error = check_string(name, value)
        if error:
            logging.error(error)
            return jsonify({"error": error})

    for name, value in [
        ("check_quality", check_quality),
        ("check_similarity", check_similarity),
    ]:
        error = check_bool(name, value)
        if error:
            logging.error(error)
            return jsonify({"error": error})

    check_quality = check_quality.lower() == "true" if type(check_quality) == str else check_quality
    check_similarity = check_similarity.lower() == "true" if type(check_similarity) == str else check_similarity

    for name, value in [
        ("passages", passages),
    ]:
        error = check_stringlist(name, value)
        if error:
            logging.error(error)
            return jsonify({"error": error})

    passages = tuple(passages)  # allow caching

    job = {
        "params": {
            "passages": passages,
            "question": question,
            "answer": answer,
            "language": language,
            "check_quality": check_quality,
            "check_similarity": check_similarity,
        },
        "response": [],
    }
    request_queue.put(job)
    request_queue.join()  # todo: don't block
    assert len(job["response"]) == 1
    return jsonify(job["response"][0])


class Evaluator:
    def __init__(self, args):
        self.nli = get_nli(args)
        self.llm = get_llm(args)
        self.sim = get_sim(args)

    @cache
    def evaluate(self, passages, question, answer, language, check_quality, check_similarity):
        citations = evaluate_citations(passages, answer, self.nli, language)

        if check_quality:
            quality = evaluate_quality(question, answer, self.llm, self.sim, language)
        else:
            quality = None

        if check_similarity:
            similarity = evaluate_similarity(passages, citations, self.sim, language)
        else:
            similarity = None

        return {
            "citations": citations,
            "quality": quality,
            "similarity": similarity,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=52888,
    )
    add_nli_args(parser)
    add_llm_args(parser)
    add_sim_args(parser)
    args = parser.parse_args()
    logging.info(f"args: {args}")

    evaluator = Evaluator(args)

    thread = Thread(target=process_requests, args=(evaluator,))
    thread.start()

    def signal_handler(sig, frame):
        logging.info("exiting...")
        request_queue.put(None)
        # thread.join()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    app.run(host="0.0.0.0", port=args.port)
