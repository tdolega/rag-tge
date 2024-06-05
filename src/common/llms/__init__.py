from common.llms.llamacpp import LLM_LLAMACPP
from common.llms.gpt import LLM_GPT
from common.llms.transformers import LLM_TRANSFORMERS
from common.llms.awanllm import LLM_AWANLLM
from common.llms.hfspaces import LLM_HFSPACES
from common.llms.hfia import LLM_HFIA
from common.llms.groq import LLM_GROQ
from common.llms.clarin import LLM_CLARIN


def get_llm(args):
    model_name = args.llm

    if model_name == "llama.cpp":
        model = LLM_LLAMACPP()
    elif model_name.startswith("openai_"):
        model_name = model_name[len("openai_") :]
        model = LLM_GPT()
    elif model_name.startswith("awanllm_"):
        model_name = model_name[len("awanllm_") :]
        model = LLM_AWANLLM()
    elif model_name.startswith("hfspaces_"):
        model_name = model_name[len("hfspaces_") :]
        model = LLM_HFSPACES(model_name)
    elif model_name.startswith("transformers_"):
        model_name = model_name[len("transformers_") :]
        model = LLM_TRANSFORMERS(model_name, device=args.llm_device)
    elif model_name.startswith("hfia_"):
        model_name = model_name[len("hfia_") :]
        model = LLM_HFIA(model_name)
    elif model_name.startswith("groq_"):
        model_name = model_name[len("groq_") :]
        model = LLM_GROQ()
    elif model_name.startswith("clarin_"):
        model_name = model_name[len("clarin_") :]
        model = LLM_CLARIN()
    else:
        raise ValueError(f"Unknown model: {model_name}. Did you forget to add prefix?")

    if model.model_name is None:
        model.model_name = model_name
    if model.short_model_name is None:
        model.short_model_name = model.model_name
        if model.short_model_name.endswith("/"):
            model.short_model_name = model.short_model_name[:-1]
        model.short_model_name = model.short_model_name.split("/")[-1]

    model.temperature = args.temperature

    return model


def add_llm_args(parser):
    parser.add_argument(
        "--llm",
        type=str,
        default="hfia_mistralai/Mistral-7B-Instruct-v0.2",
        help='model name like: "llama.cpp", "hfia_meta-llama/Llama-2-13b-chat-hf" or "openai_gpt-4"',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--llm_device",
        type=str,
        default="cuda",
        help="[only for 'transformers' models] device to run LLM model on, for example: 'auto', 'cuda', 'cpu'",
    )
