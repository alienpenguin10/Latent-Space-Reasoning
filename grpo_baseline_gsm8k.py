import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import argparse
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
# from patch import patch_trainer_optimizer # HRPO specific
from utils import *
from dotenv import load_dotenv


load_dotenv()

os.environ["WANDB_PROJECT"] = "latent-space-reasoning"

def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    # Modified experiment name for baseline
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-gsm8k-baseline-group{args.group_size}"
                f"-lora{args.lora_rank}-temp{args.temperature}")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = True,  # Required for vLLM support
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Removed modules_to_save (HRPO specific)
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    # Removed reset_lambda_parameters (HRPO specific)

    training_args = GRPOConfig(
        use_vllm = True,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_strategy = "no",
        report_to = "wandb",
        output_dir = exp_name,
        dataloader_num_workers = 4,  # Parallel data loading
        dataloader_prefetch_factor = 2,  # Prefetch batches
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),
        ],
        args = training_args,
        train_dataset = dataset,
    )
    # Removed patch_trainer_optimizer (HRPO specific)
    trainer.train()

    
    # Save the final merged model locally
    print(f"Saving final merged model to {exp_name}")
    model.save_pretrained_merged(exp_name, tokenizer, save_method="merged_16bit")
    
    model.push_to_hub_merged(f"Alienpenguin10/{exp_name}", tokenizer, save_method = "merged_16bit")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)


    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    # Removed HRPO arguments: residual_r_min, residual_r_max, lr_residual_gate, lr_residual_Lambda
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set keys from env if present (overrides if passed/set before)
    if not os.environ.get("WANDB_API_KEY"):
         print("Warning: WANDB_API_KEY not found in environment.")
    if not os.environ.get("HF_TOKEN"):
         print("Warning: HF_TOKEN not found in environment.")

    main(args)
