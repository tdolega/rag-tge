from common.llms.llamacpp import LLM_LLAMACPP
from common.llms.gpt import LLM_GPT
from common.llms.hf import LLM_HF
from common.llms.awanllm import LLM_AWANLLM
from common.llms.hfspaces import LLM_HFSPACES


def get_llm(args):
    model_name = args.llm
    temperature = args.temperature

    if model_name == "llama.cpp":
        model = LLM_LLAMACPP()
    elif "gpt" in model_name:
        model = LLM_GPT(model_name)
    elif model_name.startswith("awanllm_"):
        model = LLM_AWANLLM(model_name)
    elif model_name.startswith("hfspaces_"):
        model = LLM_HFSPACES(model_name)
    elif "/" in model_name:
        model = LLM_HF(model_name)
    else:
        raise ValueError(f"unknown model name: {model_name}")

    model.temperature = temperature
    return model


def add_llm_args(parser):
    parser.add_argument(
        "--llm",
        type=str,
        default="llama.cpp",
        help='model name like: "llama.cpp", "meta-llama/Llama-2-13b-chat-hf" or "gpt-4"',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for sampling",
    )
