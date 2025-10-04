# scripts/evaluation/reasoning_evaluation_patch_worker.py
import os
import sys
import pickle
import argparse
import json
from tqdm import tqdm

# Import the actual evaluation function from your original script
from reasoning_evaluation import evaluate_reasoning_item

# Add project root to path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator

def patch_file(args):
    """Loads a corrupted file, evaluates the missing prefix, and saves the corrected full file."""
    # Load original input data and corrupted evaluation data
    with open(args.input_path, 'rb') as f:
        full_input_data = pickle.load(f)
    with open(args.output_path, 'rb') as f:
        corrupted_data = pickle.load(f)
    with open(args.raw_data_path, 'rb') as f:
        raw_dataset = pickle.load(f)
    
    num_missing = len(full_input_data) - len(corrupted_data)
    if num_missing <= 0:
        print("File is already complete. Nothing to patch.")
        return

    print(f"File is missing {num_missing} items from the beginning. Starting patch...")

    # Set up the API and evaluator model
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        api_key = api_config.get("api_key")

    chat_response_generator = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=api_key)
    
    newly_patched_data = []
    # Use a temporary file for checkpointing the patch process itself
    patch_checkpoint_file = args.output_path + ".patching"
    
    start_patch_index = 0
    if os.path.exists(patch_checkpoint_file):
        with open(patch_checkpoint_file, 'rb') as f:
            newly_patched_data = pickle.load(f)
        start_patch_index = len(newly_patched_data)
        print(f"Resuming patch from item {start_patch_index}...")

    items_to_patch = full_input_data[start_patch_index:num_missing]

    with tqdm(total=num_missing, initial=start_patch_index, desc="Patching missing items") as pbar:
        for i, item_to_patch in enumerate(items_to_patch):
            original_index = start_patch_index + i
            raw_item = raw_dataset[original_index]
            
            # Add necessary metadata back for the evaluator function
            item_to_patch["other_metadata"] = raw_item.get("other_metadata", {})
            if args.task_name == "code":
                for k_idx, k in enumerate(item_to_patch["required_knowledge"]):
                    k['answer'] = raw_item.get("probe_questions", [])[k_idx].get("answer", "")
            
            # Run the same evaluation logic
            processed_item, _ = evaluate_reasoning_item(item_to_patch, args, chat_response_generator)
            newly_patched_data.append(processed_item)

            # Checkpoint the patch progress
            with open(patch_checkpoint_file, 'wb') as f:
                pickle.dump(newly_patched_data, f)
            pbar.update(1)

    # Stitch the file back together: [newly_patched_data] + [old_corrupted_data]
    final_complete_data = newly_patched_data + corrupted_data
    
    # Overwrite the original file with the fully corrected data
    with open(args.output_path, 'wb') as f:
        pickle.dump(final_complete_data, f)
        
    # Clean up the checkpoint file
    if os.path.exists(patch_checkpoint_file):
        os.remove(patch_checkpoint_file)

    print(f"\n✅ Successfully patched {args.output_path}. Final length: {len(final_complete_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker script to patch a single corrupted evaluation file.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the full input data from the 'injection' folder.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the corrupted output file to patch.")
    parser.add_argument('--raw_data_path', type=str, required=True, help="Path to the original raw dataset (e.g., test_500.pkl).")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json")
    # We need to pass through some original args for the evaluator function
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--evaluate_model_name', type=str, required=True)
    args = parser.parse_args()
    
    patch_file(args)