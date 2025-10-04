# run_nli_patch.py (Parallel Version)
import os
import sys
import pickle
import argparse
import json
import itertools
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console

# Add project root to path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator
from scripts.evaluation.reasoning_evaluation import PROMPT_TEMPLATES_NLI # Re-use prompts

console = Console()

# This helper function remains the same. Each thread will call it.
def run_nli_evaluation_for_item(item, task_name, chat_response_generator):
    """Performs only the NLI part of the evaluation and updates the item in-place."""
    model_final_answer_candidate = "N/A"
    if task_name == "code" and item.get("model_response"):
        match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
        if match: model_final_answer_candidate = match.group(1).strip()
            
    model_full_response_context = item.get("model_response", "")
    chat_response_generator.update_chat_history([("system", PROMPT_TEMPLATES_NLI.get(task_name, PROMPT_TEMPLATES_NLI["default"]))])

    for knowledge_item in item["required_knowledge"]:
        knowledge_text = knowledge_item["knowledge"]
        if task_name == "code":
            llm_input_prompt_nli = f"Code:\n```python\n{model_final_answer_candidate}\n```\n\nFunction:\n{knowledge_item.get('answer', '')}\n\nNLI:\n"
        else:
             llm_input_prompt_nli = f"Context:\n{model_full_response_context}\n\nStatement:\n{knowledge_text}\n\nNLI:\n"
        
        raw_nli_response = chat_response_generator.generate_response(llm_input_prompt_nli, temperature=0, top_p=1, n=1, max_tokens=4096)[0]
        
        response_text_lower = raw_nli_response.replace("NLI:", "").strip().lower()
        if "entailment" in response_text_lower: nli_class = "entailment"
        elif "contradiction" in response_text_lower: nli_class = "contradiction"
        else: nli_class = "neutral"
        
        knowledge_item["nli_class"] = nli_class
        knowledge_item["nli_explanation"] = raw_nli_response.strip()

# This is the new worker function that each thread will run
def patch_single_file_worker(params, args):
    """
    Worker function to process a single file. It finds items to patch,
    evaluates them, and saves the file once at the end.
    """
    task, model, method, scope, inject = params['task'], params['model'], params['method'], params['scope'], params['inject']
    
    # Construct the file path
    output_dir = f'data/eval_results/{task}/injection_evaluated/'
    if inject: filename = f"{method}_{args.data_size}_{model}_{scope}.pkl"
    else: filename = f"original_{args.data_size}_{model}_1.pkl"
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(output_path):
        return f"[dim]Skipped (Not Found): {os.path.basename(output_path)}[/dim]"

    try:
        with open(output_path, 'rb') as f:
            evaluated_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        return f"[bold red]Error (Corrupted): Could not read {os.path.basename(output_path)}[/bold red]"
        
    items_to_patch_indices = [
        i for i, item in enumerate(evaluated_data)
        if item.get('required_knowledge') and 'nli_class' not in item['required_knowledge'][0]
    ]

    if not items_to_patch_indices:
        return f"[green]Healthy: {os.path.basename(output_path)}[/green]"

    # Create a dedicated API client for this thread
    with open(args.api_config_file, 'r') as f:
        api_key = json.load(f).get("api_key")
    chat_gen = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=api_key)

    for item_idx in items_to_patch_indices:
        item = evaluated_data[item_idx]
        run_nli_evaluation_for_item(item, task, chat_gen)
    
    # Save the fully patched data ONCE at the end
    with open(output_path, 'wb') as f:
        pickle.dump(evaluated_data, f)
        
    return f"[bold yellow]Patched {len(items_to_patch_indices)} items in {os.path.basename(output_path)}[/bold yellow]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch evaluation files missing NLI keys in parallel.")
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--task_names', nargs='+', required=True)
    parser.add_argument('--method_names', nargs='+', default=["base", "mello"])
    parser.add_argument('--data_size', type=int, default=500)
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json")
    parser.add_argument('--no-base-eval', action='store_true')
    parser.add_argument('--no-inject-eval', action='store_true')
    parser.add_argument('--no-method-all-eval', action='store_true')
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100, 500])
    parser.add_argument('--max-workers', type=int, default=16, help="Maximum number of parallel workers.")
    args = parser.parse_args()

    # Generate all file combinations to check
    all_params = []
    methods_to_check = args.method_names
    if not args.no_method_all_eval: methods_to_check.append('all')
    
    for task in args.task_names:
        for model in args.model_names:
            if not args.no_base_eval: all_params.append({'task': task, 'model': model, 'method': 'base', 'scope': 1, 'inject': False})
            if not args.no_inject_eval:
                for method in methods_to_check:
                    for scope in args.knowledge_aggregation_scopes:
                        all_params.append({'task': task, 'model': model, 'method': method, 'scope': scope, 'inject': True})

    console.print(f"[bold]Scanning and patching {len(all_params)} potential files using up to {args.max_workers} workers...[/bold]")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs to the thread pool
        futures = [executor.submit(patch_single_file_worker, params, args) for params in all_params]
        
        # Use tqdm to show overall progress as files complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                result = future.result()
                console.print(result)
            except Exception as e:
                console.print(f"[bold red]A worker failed with an unexpected error: {e}[/bold red]")

    console.print("\n\n[bold green]All files have been processed.[/bold green]")