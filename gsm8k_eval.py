import os
import sys
import re
import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

# Fix shadowing of installed packages by local directories
sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))

from unsloth import FastLanguageModel
from datasets import load_dataset

# Configuration
MODEL_PATH = "./M3PO"
max_seq_length = 2048
max_new_tokens = 1024
batch_size = 8  # Adjust based on GPU memory
temperature = 0.0  # Use greedy decoding for eval
device = "cuda" if torch.cuda.is_available() else "cpu"

ANSWER_START = "####"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)


def extract_hash_answer(text: str) -> str | None:
    """Extract the ground truth answer from GSM8K format."""
    if "####" not in text:
        return ''
    return text.split("####")[1].strip()


def extract_from_response(text: str) -> str:
    """Extract the answer from model response."""
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def delete_extra_zero(n):
    """Normalize numeric strings by removing trailing zeros."""
    try:
        n = float(n)
    except:
        try:
            n = eval(n)
        except:
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n = str(n)
        return n


def process_gsm8k_answer(pred: str) -> str:
    """Process and normalize an answer string for comparison."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", ""))
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def format_prompt(question: str, tokenizer) -> str:
    """Format a question into the chat format used during training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_path: str, accelerator: Accelerator = None):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")

    # When using accelerate, load on the specific device for this process
    if accelerator is not None:
        device_map = {"": accelerator.local_process_index}
    else:
        device_map = "auto"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map=device_map,
    )

    # Enable fast inference mode
    FastLanguageModel.for_inference(model)

    # Verify adapters are loaded
    print(f"Active adapters: {getattr(model, 'active_adapters', 'Not found')}")
    if hasattr(model, "peft_config"):
        print(f"PEFT config keys: {list(model.peft_config.keys())}")
    else:
        print("WARNING: No PEFT config found on model!")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (exclude the prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def generate_responses_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    """Generate responses for a batch of prompts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_seq_length - max_new_tokens).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i, output in enumerate(outputs):
        # Get the length of the input for this example
        input_len = (inputs["input_ids"][i] != tokenizer.pad_token_id).sum().item()
        generated_tokens = output[input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate(model, tokenizer, dataset, use_batch: bool = True, verbose: bool = False):
    """Evaluate the model on the dataset."""
    correct = 0
    total = 0
    results = []

    if use_batch:
        # Batch evaluation
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i + batch_size]
            questions = batch["question"]
            answers = batch["answer"]

            prompts = [format_prompt(q, tokenizer) for q in questions]
            responses = generate_responses_batch(model, tokenizer, prompts)

            for j, (question, answer, response) in enumerate(zip(questions, answers, responses)):
                ground_truth = extract_hash_answer(answer)
                ground_truth_processed = process_gsm8k_answer(ground_truth) if ground_truth else ""

                extracted = extract_from_response(response)
                prediction = process_gsm8k_answer(extracted)

                is_correct = prediction == ground_truth_processed
                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response,
                    "extracted": extracted,
                    "prediction": prediction,
                    "correct": is_correct
                })

                if verbose and not is_correct:
                    print(f"\n{'='*60}")
                    print(f"Question: {question[:200]}...")
                    print(f"Ground truth: {ground_truth_processed}")
                    print(f"Prediction: {prediction}")
                    print(f"Response: {response[:500]}...")
    else:
        # Single example evaluation (slower but more memory efficient)
        for example in tqdm(dataset, desc="Evaluating"):
            question = example["question"]
            answer = example["answer"]
            ground_truth = extract_hash_answer(answer)
            ground_truth_processed = process_gsm8k_answer(ground_truth) if ground_truth else ""

            prompt = format_prompt(question, tokenizer)
            response = generate_response(model, tokenizer, prompt)

            extracted = extract_from_response(response)
            prediction = process_gsm8k_answer(extracted)

            is_correct = prediction == ground_truth_processed
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "response": response,
                "extracted": extracted,
                "prediction": prediction,
                "correct": is_correct
            })

            if verbose and not is_correct:
                print(f"\n{'='*60}")
                print(f"Question: {question[:200]}...")
                print(f"Ground truth: {ground_truth_processed}")
                print(f"Prediction: {prediction}")
                print(f"Response: {response[:500]}...")

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    global batch_size
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GRPO-trained model on GSM8K test set")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print incorrect predictions")
    parser.add_argument("--no_batch", action="store_true",
                        help="Disable batch processing (slower but more memory efficient)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save detailed results as JSON")
    args = parser.parse_args()
    batch_size = args.batch_size

    # Initialize accelerator
    accelerator = Accelerator()

    # Load model (each process loads on its own GPU)
    model, tokenizer = load_model(args.model_path, accelerator)

    # Load test dataset
    if accelerator.is_main_process:
        print("Loading GSM8K test dataset...")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    if args.max_samples:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        if accelerator.is_main_process:
            print(f"Evaluating on {len(test_dataset)} samples (limited)")
    else:
        if accelerator.is_main_process:
            print(f"Evaluating on {len(test_dataset)} samples")

    # Split dataset across processes
    with accelerator.split_between_processes(list(range(len(test_dataset)))) as indices:
        local_dataset = test_dataset.select(indices)
        if accelerator.is_main_process:
            print(f"\nStarting evaluation...")

        # Evaluate on local subset
        _, local_results = evaluate(
            model, tokenizer, local_dataset,
            use_batch=not args.no_batch,
            verbose=args.verbose
        )

    # Gather results from all processes
    all_results = gather_object(local_results)

    # Only main process prints and saves
    if accelerator.is_main_process:
        correct = sum(1 for r in all_results if r['correct'])
        total = len(all_results)
        accuracy = correct / total if total > 0 else 0

        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"Dataset: GSM8K test set")
        print(f"Total samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"{'='*60}")

        # Save detailed results if requested
        if args.save_results:
            import json
            with open(args.save_results, "w") as f:
                json.dump({
                    "model_path": args.model_path,
                    "accuracy": accuracy,
                    "total": total,
                    "correct": correct,
                    "results": all_results
                }, f, indent=2)
            print(f"Detailed results saved to {args.save_results}")

        # Print a few example predictions
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS (first 3)")
        print("="*60)
        for i, r in enumerate(all_results[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {r['question'][:150]}...")
            print(f"Ground truth: {r['ground_truth']}")
            print(f"Prediction: {r['prediction']}")
            print(f"Correct: {'✓' if r['correct'] else '✗'}")


if __name__ == "__main__":
    main()
