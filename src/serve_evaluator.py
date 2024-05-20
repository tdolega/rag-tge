import argparse
from queue import Queue
from threading import Thread
import logging
import traceback
import signal

from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from common.nlis import get_nli, add_nli_args
from common.llms import get_llm, add_llm_args
from common.sims import get_sim, add_sim_args
from evaluate_answers import evaluate_citations, evaluate_quality


users = {
    "Tymek": generate_password_hash("pnw2024"),
}

app = Flask(__name__)
auth = HTTPBasicAuth()
request_queue = Queue()
logging.basicConfig(level=logging.INFO)


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


def process_requests(evaluator):
    while True:
        logging.info("ready!")
        # pobierz zapytanie z kolejki, blokując jeśli kolejka jest pusta
        (request, context) = request_queue.get()
        if request is None:
            break
        logging.info("processing...")
        with app.test_request_context(context["path"], environ_base=context["environ"]):
            try:
                response = evaluate(evaluator)
                context["response"].append(response.get_json())
            except Exception as e:
                logging.error(f"error processing request: {e}")
                context["response"].append({"error": str(e)})
            request_queue.task_done()


@app.route("/evaluate", methods=["GET"])
@auth.login_required
def enqueue_request():
    context = {"path": request.path, "environ": request.environ.copy(), "response": []}
    request_queue.put((request, context))
    request_queue.join()
    return jsonify(context["response"][0])


class Evaluator:
    def __init__(self, args):
        self.nli = get_nli(args)
        self.llm = get_llm(args)
        self.sim = get_sim(args)

    def evaluate(self, passages, question, answer, language):
        return {
            "citations": evaluate_citations(passages, answer, self.nli, language),
            "quality": evaluate_quality(question, answer, self.llm, self.sim, language),
        }


def evaluate(evaluator):
    passages = request.args.getlist("passages", None)
    question = request.args.get("question", None)
    answer = request.args.get("answer", None)
    language = request.args.get("language", "polish")

    if not type(question) == str or not type(answer) == str or not type(language) == str or len(question) == 0 or len(answer) == 0 or len(language) == 0:
        return jsonify({"error": "Parameters 'question', 'answer' and 'language' must be non-empty strings."})

    if not type(passages) == list or len(passages) == 0 or not all(type(p) == str for p in passages):
        return jsonify({"error": "Parameter 'passages' must be a non-empty list of strings."})

    try:
        response = evaluator.evaluate(passages, question, answer, language)
        return jsonify(response)
    except Exception as e:
        logging.error(f"error during evaluation: {e}")
        logging.debug(traceback.format_exc())
        return jsonify({"error": str(e)})


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
        request_queue.put((None, {}))
        thread.join()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    app.run(host="0.0.0.0", port=args.port)
