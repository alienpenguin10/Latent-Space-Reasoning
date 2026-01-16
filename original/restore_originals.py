import shutil
import os

# Base directory
BASE_DIR = "/homes/vk545/Neuralese/C3PO"

# Mapping: Source relative to BASE_DIR -> Destination relative to BASE_DIR
# Note: Source filenames in 'original/' have an 'original_' prefix which we remove or map to the correct destination filename.
files_to_restore = [
    (
        "original/original_utils.py",
        "transformers/src/transformers/generation/utils.py"
    ),
    (
        "original/original_modeling_qwen2.py",
        "transformers/src/transformers/models/qwen2/modeling_qwen2.py"
    ),
    (
        "original/original_modeling_llama.py",
        "transformers/src/transformers/models/llama/modeling_llama.py"
    ),
    (
        "original/original_grpo_trainer.py",
        "trl/trl/trainer/grpo_trainer.py"
    ),
    (
        "original/original_llama.py",
        "unsloth/unsloth/models/llama.py"
    )
]

def restore_files():
    print("Starting restoration of original files...")
    
    for src_rel, dst_rel in files_to_restore:
        src_path = os.path.join(BASE_DIR, src_rel)
        dst_path = os.path.join(BASE_DIR, dst_rel)
        
        if not os.path.exists(src_path):
            print(f"Error: Source file not found: {src_path}")
            continue
            
        # Ensure destination directory exists (it should, but good practice)
        os.path.dirname(dst_path)
            
        try:
            print(f"Restoring {src_rel} -> {dst_rel}...")
            shutil.copy2(src_path, dst_path)
            print("  Success -> Overwrote destination.")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\nRestoration complete.")

if __name__ == "__main__":
    restore_files()
