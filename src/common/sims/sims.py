from common.sims.hfia import SIMS_HFIA


def get_sim(args):
    model_name = args.sim

    if model_name.startswith("hfia_"):
        model_name = model_name[len("hfia_") :]
        model = SIMS_HFIA(model_name)
    else:
        raise ValueError(f"Unknown sentence similarity model: {model_name}, did you forget to add prefix?")

    if model.model_name is None:
        model.model_name = model_name
    if model.short_model_name is None:
        model.short_model_name = model.model_name
        if model.short_model_name.endswith("/"):
            model.short_model_name = model.short_model_name[:-1]
        model.short_model_name = model_name.split("/")[-1]

    return model


def add_sim_args(parser):
    parser.add_argument(
        "--sim",
        type=str,
        default="hfia_sentence-transformers/all-MiniLM-L6-v2",
        help='sentence similarity model name like: "hfia_sentence-transformers/all-MiniLM-L6-v2"',
    )
