from common.nlis.t5 import NLI_T5
from common.nlis.llm import NLI_LLM
from common.nlis.interceptor import NLI_INTERCEPTOR


def get_nli(args):
    model_name = args.nli

    if model_name == "t5":
        model = NLI_T5()
    elif model_name == "interceptor":
        model = NLI_INTERCEPTOR()
    else:
        model = NLI_LLM(model_name)

    return model


def add_nli_args(parser):
    parser.add_argument(
        "--nli",
        type=str,
        default="t5",
        help='evaluator model name like: "t5" for T5 evaluator or model name for LLM evaluator',
    )
