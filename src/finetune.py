import argparse

from common.consts import DS_UPLOAD_PATH, MODELS_DIR
from common.prompts import get_system_prompt
from common.llm_finetuning import LLM_TRAINER_ASSISTANT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_prompt_id", type=int, default=4)
    parser.add_argument("--language", type=str, default="en")
    trainer_assistant = LLM_TRAINER_ASSISTANT(
        parser=parser,
        force_args={
            # "dataset_name": DS_UPLOAD_PATH,
            "output_dir": "/".join(MODELS_DIR.split("/")[-2:]),
        },
    )
    args = trainer_assistant.args

    def row_to_messages(row):
        messages = [
            {"role": "system", "content": get_system_prompt(args.system_prompt_id)},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        if args.language == "en":
            messages.append({"role": "user", "content": "Thank you!"})
            messages.append({"role": "assistant", "content": "You're welcome!"})
        elif args.language == "pl":
            messages.append({"role": "user", "content": "Dziękuję!"})
            messages.append({"role": "assistant", "content": "Proszę!"})
        else:
            raise ValueError(f"unsupported language: {args.language}")
        return messages

    trainer_assistant.row_to_messages = row_to_messages
    trainer_assistant()


if __name__ == "__main__":
    main()
