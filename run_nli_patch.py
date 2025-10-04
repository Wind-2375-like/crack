# run_nli_patch.py
import os
import sys
import pickle
import argparse
import json
import itertools
from tqdm import tqdm
import re

# Add project root to path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator
from scripts.evaluation.reasoning_evaluation import PROMPT_TEMPLATES_NLI # Re-use prompts

# This helper function contains just the NLI logic
def run_nli_evaluation_for_item(item, task_name, chat_response_generator):
    """Performs only the NLI part of the evaluation and updates the item in-place."""
    
    # Extract code block if it's a code task, default to "N/A"
    model_final_answer_candidate = "N/A"
    if task_name == "code" and item.get("model_response"):
        match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
        if match:
            model_final_answer_candidate = match.group(1).strip()
            
    model_full_response_context = item.get("model_response", "")

    chat_response_generator.update_chat_history([
        ("system", PROMPT_TEMPLATES_NLI.get(task_name, PROMPT_TEMPLATES_NLI["default"])),
    ])

    for knowledge_item in item["required_knowledge"]:
        knowledge_text = knowledge_item["knowledge"]
        if task_name == "code":
            llm_input_prompt_nli = (
                f"Code:\n```python\n{model_final_answer_candidate}\n```\n\n"
                f"Function:\n{knowledge_item['answer']}\n\n"
                f"NLI:\n"
            )
        else: # For 'grow' and 'math'
             llm_input_prompt_nli = (
                f"Context:\n{model_full_response_context}\n\n"
                f"Statement:\n{knowledge_text}\n\n"
                f"NLI:\n"
            )

        raw_nli_response = chat_response_generator.generate_response(
            llm_input_prompt_nli, temperature=0, top_p=1, n=1, max_tokens=4096
        )[0]
        
        # Simplified parser for this script
        response_text_lower = raw_nli_response.replace("NLI:", "").strip().lower()
        if "entailment" in response_text_lower: nli_class = "entailment"
        elif "contradiction" in response_text_lower: nli_class = "contradiction"
        else: nli_class = "neutral"
        
        knowledge_item["nli_class"] = nli_class
        knowledge_item["nli_explanation"] = raw_nli_response.strip()

def find_and_patch_files(args):
    """Scans all relevant files and patches items missing NLI keys."""
    
    with open(args.api_config_file, 'r') as f:
        api_key = json.load(f).get("api_key")
    chat_gen = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=api_key)

    # Generate all file paths to check
    all_params = []
    # Simplified logic to generate all combinations
    methods = args.method_names + (['all'] if not args.no_method_all_eval else [])
    
    for task in args.task_names:
        for model in args.model_names:
            if not args.no_base_eval: all_params.append({'task': task, 'model': model, 'method': 'base', 'scope': 1, 'inject': False})
            for method in methods:
                for scope in args.knowledge_aggregation_scopes:
                     if not args.no_inject_eval: all_params.append({'task': task, 'model': model, 'method': method, 'scope': scope, 'inject': True})

    print(f"Scanning {len(all_params)} potential result files...")

    for params in tqdm(all_params, desc="Scanning Files"):
        task, model, method, scope, inject = params['task'], params['model'], params['method'], params['scope'], params['inject']
        
        output_dir = f'data/eval_results/{task}/injection_evaluated/'
        if inject: filename = f"{method}_{args.data_size}_{model}_{scope}.pkl"
        else: filename = f"original_{args.data_size}_{model}_1.pkl"
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(output_path): continue

        try:
            with open(output_path, 'rb') as f:
                evaluated_data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            print(f"\n[bold red]Warning: Could not read corrupted file {output_path}. Skipping.[/bold red]")
            continue
            
        items_to_patch_indices = []
        for i, item in enumerate(evaluated_data):
            if item.get('required_knowledge') and 'nli_class' not in item['required_knowledge'][0]:
                items_to_patch_indices.append(i)

        if not items_to_patch_indices: continue

        print(f"\n[bold yellow]Found {len(items_to_patch_indices)} items to patch in {output_path}[/bold yellow]")
        file_was_modified = False

        for item_idx in tqdm(items_to_patch_indices, desc=f"Patching {os.path.basename(output_path)}", leave=False):
            item = evaluated_data[item_idx]
            run_nli_evaluation_for_item(item, task, chat_gen)
            file_was_modified = True
            # Save after every single change for safety
            with open(output_path, 'wb') as f:
                pickle.dump(evaluated_data, f)
        
        if file_was_modified:
            print(f"[bold green]Successfully patched and saved {output_path}[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch evaluation files missing NLI keys.")
    # Copy arguments from your main evaluation script
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names to check.')
    parser.add_argument('--task_names', nargs='+', required=True, help='List of task names to check.')
    parser.add_argument('--method_names', nargs='+', default=["base", "mello"], help='List of method names to check.')
    parser.add_argument('--data_size', type=int, default=500, help='Data size for testing.')
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-5-mini-2025-08-07", help="Model name for the evaluation judge.")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json")
    parser.add_argument('--no-base-eval', action='store_true')
    parser.add_argument('--no-inject-eval', action='store_true')
    parser.add_argument('--no-method-all-eval', action='store_true')
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100, 500])
    args = parser.parse_args()
    
    find_and_patch_files(args)
    print("\n\nAll patchable files have been processed.")