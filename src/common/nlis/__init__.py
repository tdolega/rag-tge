from common.nlis.true import NLI_TRUE
from common.nlis.llm import NLI_LLM


def get_nli(args):
    model_name = args.nli

    if model_name.startswith("true_"):
        model_name = model_name[len("true_") :]
        model = NLI_TRUE(model_name, device=args.nli_device)
    elif model_name.startswith("llm_"):
        model_name = model_name[len("llm_") :]
        model = NLI_LLM(model_name)
    else:
        raise ValueError(f"Unknown NLI model: {model_name}, did you forget to add prefix?")

    if model.model_name is None:
        model.model_name = model_name
    if model.short_model_name is None:
        model.short_model_name = model.model_name
        if model.short_model_name.endswith("/"):
            model.short_model_name = model.short_model_name[:-1]
        model.short_model_name = model.short_model_name.split("/")[-1]
        if model.short_model_name.endswith("-bf16"):
            model.short_model_name = model.short_model_name[: -len("-bf16")]

    return model


def add_nli_args(parser):
    parser.add_argument(
        "--nli",
        type=str,
        default="true_tdolega/t5_xxl_true_nli_mixture-bf16",
        help='evaluator model name like: "true_google/t5_xxl_true_nli_mixture" for TRUE T5 evaluator',
    )
    parser.add_argument(
        "--nli_device",
        type=str,
        default="cuda",
        help="device to run NLI model on, for example: 'auto', 'cuda', 'cpu'",
    )
