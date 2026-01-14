import unsloth
from unsloth import FastLanguageModel

import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()


def push_model(
    model_path: str,
    hub_name: str,
):
    """Push a trained model to Hugging Face Hub."""

    # Login to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment. Please set it in .env file.")
    login(token=hf_token)

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=False,
    )

    # Use standard push_to_hub for already-merged models
    print(f"Pushing model to {hub_name}")
    model.push_to_hub(hub_name, token=hf_token)
    tokenizer.push_to_hub(hub_name, token=hf_token)
    print(f"Successfully pushed model to https://huggingface.co/{hub_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a trained model to Hugging Face Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model checkpoint"
    )
    parser.add_argument(
        "--hub_name",
        type=str,
        default=None,
        help="Name for the model on HF Hub (e.g., 'Alienpenguin10/model-name'). "
             "Defaults to 'Alienpenguin10/<model_path_basename>'"
    )
    args = parser.parse_args()

    # Default hub name based on model path
    if args.hub_name is None:
        basename = os.path.basename(args.model_path.rstrip("/"))
        args.hub_name = f"Alienpenguin10/{basename}"

    push_model(
        model_path=args.model_path,
        hub_name=args.hub_name,
    )
