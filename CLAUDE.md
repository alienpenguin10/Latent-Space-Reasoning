# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

H3PO (also referenced as M3PO) is a research implementation of **Multi-Path Collaborative Reasoning via Reinforcement Learning**. This project implements the M3PO algorithm described in the included research paper (M3PO.pdf), which enhances LLM reasoning through cross-path collaboration during GRPO (Group Relative Policy Optimization) training.

The codebase integrates three modified library repositories:
- **transformers/** - Modified HuggingFace Transformers library
- **trl/** - Modified TRL (Transformer Reinforcement Learning) library with M3PO-enabled GRPO trainer
- **unsloth/** - Modified Unsloth library for optimized training with M3PO module

## Repository Structure

```
H3PO/
├── gsm8k_grpo.py          # Main training script for GSM8k dataset
├── M3PO.pdf               # Research paper describing the algorithm
├── transformers/          # Modified HuggingFace Transformers (submodule)
├── trl/                   # Modified TRL with M3PO GRPO trainer (submodule)
├── unsloth/               # Modified Unsloth with M3PO module (submodule)
├── logs/                  # TensorBoard logs
├── outputs/               # Saved model checkpoints
└── unsloth_compiled_cache/ # Compiled kernel cache
```

## Key Architecture Components

### M3PO Module (`unsloth/unsloth/models/m3po_module.py`)

The core M3PO implementation provides cross-path collaborative reasoning:

- **`apply_cross_path_interaction()`** - Single-batch cross-path blending
  - Computes cosine similarity between parallel reasoning paths
  - Uses temperature-scaled softmax for attention weights
  - Blends embeddings: `h_i = (1 - λ) * e_i + λ * c_i`

- **`apply_cross_path_interaction_batch()`** - Batched version for multiple questions
  - Groups sequences by question (batch_size * N paths)
  - Applies cross-path interaction independently per question

- **Configuration Parameters:**
  - `lambda_blend` (default: 0.1) - Controls blending between original and contextual embeddings
  - `temperature` (default: 0.1) - Controls attention sharpness (lower = more focused)
  - `num_generations` - Number of parallel paths per prompt
  - `thinking_mask` - Tracks which paths are still in reasoning mode

### GRPO Trainer with M3PO (`trl/trl/trainer/grpo_trainer.py`)

Modified GRPO trainer that integrates M3PO cross-path collaboration:

- Extends standard GRPO with M3PO parameters:
  - `enable_cross_path` - Toggle M3PO collaboration
  - `cross_path_lambda` - Blending coefficient
  - `cross_path_temp` - Temperature for cross-path attention

### Training Script (`gsm8k_grpo.py`)

Main training script demonstrating M3PO usage:

**Key Environment Variables:**
- `M3PO_DEBUG` - Set to "1" to enable debug printing
- `UNSLOTH_WARN_UNINITIALIZED` - Controls Unsloth warnings
- `WORLD_SIZE`, `LOCAL_RANK`, `RANK` - Distributed training configuration

**Important Configuration:**
- Uses `sys.path.insert(0, ...)` to prioritize local library versions over installed packages
- Patches `DistributedDataParallel.config` property for DDP compatibility
- Implements custom reward function based on answer format: `{reasoning} #### {answer}`

## Common Commands

### Training

**Single GPU:**
```bash
python gsm8k_grpo.py
```

**Multi-GPU (Distributed):**
```bash
torchrun --nproc_per_node=NUM_GPUS gsm8k_grpo.py
```

### Monitoring Training

TensorBoard is auto-launched in the script:
```bash
# View logs at http://localhost:6007
# Or manually launch:
tensorboard --logdir=./logs --port=6007 --bind_all
```

### Environment Setup

The script uses local library versions via path manipulation:
```python
sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))
```

This ensures modified versions are used instead of pip-installed packages.

## M3PO Algorithm Details

### Cross-Path Collaboration Mechanism

1. **Parallel Rollouts**: Generate N completions per prompt during GRPO training
2. **Thinking Phase**: During chain-of-thought reasoning (before "####" delimiter):
   - Compute pairwise cosine similarity between path embeddings
   - Apply temperature-scaled softmax to get attention weights
   - Blend each path's embedding with weighted context from other paths
3. **Answer Phase**: Paths generate answers independently (no cross-path interaction)
4. **Policy Update**: Use group-relative advantages with hybrid embeddings

### Critical Implementation Notes

- **Thinking Delimiter**: The `####` token separates reasoning from answers
  - Reasoning phase: Cross-path collaboration active
  - Answer phase: Independent generation (preserves diversity)

- **DDP Patch Required**: The script patches `DistributedDataParallel` to expose `.config`:
  ```python
  @property
  def ddp_config_patch(self):
      return getattr(self.module, "config", None)
  DistributedDataParallel.config = ddp_config_patch
  ```

- **Reward Function**: Assigns zero reward to malformed outputs:
  - Multiple "####" delimiters (prevents early exit)
  - Missing "####" delimiter
  - Incorrect answer format

## Hyperparameters

### M3PO-Specific:
- `enable_m3po`: True/False toggle
- `m3po_lambda`: 0.1 (blending coefficient)
- `m3po_temperature`: 0.1 (attention temperature)
- `num_generations`: 8 (parallel paths per prompt)

### GRPO Training:
- `beta`: 0.005 (KL penalty)
- `temperature`: 0.5 (sampling temperature)
- `max_prompt_length`: 1024
- `max_completion_length`: 1024
- `learning_rate`: 5e-6
- `lora_rank`: 32, `lora_alpha`: 64

## Debugging

Set debug mode to see M3PO module internals:
```python
os.environ["M3PO_DEBUG"] = "1"
```

This prints:
- Batch sizes and dimensions
- Device placement
- Cross-path interaction details

## File Modifications

Modified files in the submodules (tracked by git):
- `transformers/src/transformers/utils/generic.py`
- `trl/trl/trainer/grpo_config.py`
- `trl/trl/trainer/grpo_trainer.py`
- `unsloth/unsloth/models/llama.py`
- `unsloth/unsloth/models/m3po_module.py` (new file)

## Dataset Format

Expected dataset structure (e.g., GSM8k):
```python
{
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "answer": "numerical_answer"  # Extracted from after ####
}
```

The system prompt instructs the model to use the format:
```
{reasoning process} #### {answer}
```

## Important Notes

1. **Path Modification**: The script modifies `sys.path` to use local library versions. This is intentional and critical for using the M3PO-modified code.

2. **LoRA Configuration**: When using M3PO, avoid training MoE-specific layers:
   ```python
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
   ```

3. **Quantization**: Currently uses `load_in_4bit=False` due to M3PO requirements. Standard GRPO would use 4-bit quantization.

4. **Answer Delimiter**: The `####` delimiter is hardcoded in multiple places:
   - Training script: `ANSWER_START = "####"`
   - Model attribute: `model.answer_start = ANSWER_START`
   - Reward function: Checks for exactly one occurrence

5. **Distributed Training**: The script handles multi-GPU training with proper device mapping and uses `IS_MAIN_PROCESS` guard for logging and checkpoint saving.

6. **Thinking Mask**: During generation, M3PO tracks which paths have exited "thinking mode" (generated the #### delimiter) to stop applying cross-path interaction to those paths.
