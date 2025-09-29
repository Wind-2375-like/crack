import os
import sys
import pandas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import pickle
import argparse
import importlib
from tqdm import tqdm
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers import translate_model_name
from utils.dataset.reasoning_dataset import ReasoningEvalDataset

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=500, help="Number of triples to process")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b", help="Model name for the API")
    parser.add_argument('--task_name', type=str, default="grow", help="Task name")
    parser.add_argument('--inject_knowledge', action='store_true', help="Whether to inject knowledge into the input")
    parser.add_argument('--knowledge_aggregation_scope', type=int, default=1, help="Scope for aggregating 'unknown' knowledge.")
    parser.add_argument('--method', type=str, default="base", help="Method to use for complex reasoning.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for the model")
    parser.add_argument('--top_p', type=float, default=0.7, help="Top-p sampling for the model")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum tokens for the model")
    parser.add_argument('--num_responses', type=int, default=1, help="Number of responses to generate")
    return parser.parse_args()

# --- MODIFIED: These functions now only take one argument ---
def extract_required_unknown_knowledge(items_list):
    """Extracts unknown knowledge from a list of unified data items from ReasoningEvalDataset."""
    seen_knowledge = set()
    unknown_knowledge_list = []
    for an_item in items_list:
        probe_questions = an_item.get('probe_questions', [])
        for i, k_entry in enumerate(an_item.get('required_knowledge', [])):
            knowledge_str = k_entry.get('knowledge')
            if not k_entry.get('knowledgable', True) and knowledge_str and knowledge_str not in seen_knowledge:
                seen_knowledge.add(knowledge_str)
                if i < len(probe_questions):
                    probe_info = probe_questions[i]
                    unknown_knowledge_list.append({
                        'knowledge': k_entry.get('knowledge'),
                        'knowledgable': k_entry.get('knowledgable'),
                        'knowledge_confidence': k_entry.get('knowledge_confidence'),
                        'probe_question': probe_info.get('question'),
                        'probe_answer': probe_info.get('answer'),
                    })
    return unknown_knowledge_list
    
def extract_all_required_knowledge(items_list):
    """Extracts all knowledge from a list of unified data items."""
    seen_knowledge = set()
    all_knowledge_list = []
    for an_item in items_list:
        probe_questions = an_item.get('probe_questions', [])
        for i, k_entry in enumerate(an_item.get('required_knowledge', [])):
            knowledge_str = k_entry.get('knowledge')
            if knowledge_str and knowledge_str not in seen_knowledge:
                seen_knowledge.add(knowledge_str)
                if i < len(probe_questions):
                    probe_info = probe_questions[i]
                    all_knowledge_list.append({
                        'knowledge': k_entry.get('knowledge'),
                        'knowledgable': k_entry.get('knowledgable'),
                        'knowledge_confidence': k_entry.get('knowledge_confidence'),
                        'probe_question': probe_info.get('question'),
                        'probe_answer': probe_info.get('answer'),
                    })
    return all_knowledge_list

if __name__ == "__main__":
    args = parse_args()
    assert args.knowledge_aggregation_scope >= 1 , "knowledge_aggregation_scope must be a positive integer."

    # This dataset class is our single source of truth.
    eval_dataset = ReasoningEvalDataset(
        raw_path=f'data/{args.task_name}/test_{args.data_size}.pkl',
        probe_path=f'data/eval_results/{args.task_name}/probe_evaluated/test_{args.data_size}_{args.model_name}.pkl',
    )
    
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", {}).get("user_agent", None)
        if args.api_key is None: raise ValueError("API key not found.")
        if args.ua is None: raise ValueError("User agent not found.")
        
    output_dir = f'data/eval_results/{args.task_name}/injection/'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl")

    processed_data = []
    start_index = 0
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'rb') as f:
                processed_data = pickle.load(f)
            start_index = len(processed_data)
            print(f"--- Resuming from checkpoint: {output_file_path} ---")
            print(f"--- Found {start_index} items already processed. ---")
        except (pickle.UnpicklingError, EOFError):
            print(f"--- Checkpoint file corrupted. Starting from scratch. ---")
            processed_data = []; start_index = 0
            
    effective_model_name = args.model_name
    method_module_name = 'base' if args.method == 'append_t' else args.method
    if args.method == 'append_t':
        if 'qwen-3' not in args.model_name:
            print("❌ Error: 'append_t' is only for 'qwen-3' models."); sys.exit(1)
        effective_model_name += "-thinking"

    token_counts = {'prompt': 0, 'completion': 0, 'total': 0}
    chat_response_generator = ChatResponseGenerator(model_name=translate_model_name(effective_model_name), api_key=args.api_key)
    all_items_list = list(eval_dataset)
    dataset_size = len(all_items_list)
    
    try:
        module_path = f"utils.methods.{method_module_name}"
        method_module = importlib.import_module(module_path)
        MethodClass = getattr(method_module, 'Method')
        method_instance = MethodClass(args, chat_response_generator)
    except (ImportError, AttributeError) as e:
        print(f"❌ Error loading method '{args.method}': {e}"); sys.exit(1)

    if start_index >= dataset_size:
        print("--- All items have been processed. Exiting. ---"); sys.exit(0)

    with tqdm(total=dataset_size, initial=start_index, desc="Processing items") as pbar:
        if not args.inject_knowledge:
            for item in all_items_list[start_index:]:
                processed_item, usage = method_instance.run(item, [])
                processed_data.append(processed_item)
                with open(output_file_path, 'wb') as f: pickle.dump(processed_data, f)
                
                t_model = translate_model_name(effective_model_name)
                if usage and t_model in usage:
                    token_counts['prompt'] += usage[t_model].get("prompt_tokens", 0)
                    token_counts['completion'] += usage[t_model].get("completion_tokens", 0)
                    token_counts['total'] += usage[t_model].get("total_tokens", 0)
                
                pbar.set_postfix_str(f"Prompt: {token_counts['prompt']}, Completion: {token_counts['completion']}, Total: {token_counts['total']}")
                pbar.update(1)
        else:
            scope = args.knowledge_aggregation_scope
            start_batch_idx = (start_index // scope) * scope
            
            for group_start_idx in range(start_batch_idx, dataset_size, scope):
                group_end_idx = min(group_start_idx + scope, dataset_size)
                items_in_batch = all_items_list[group_start_idx:group_end_idx]
                
                local_start_index = max(0, start_index - group_start_idx)
                items_to_process_in_batch = items_in_batch[local_start_index:]
                        
                if not items_to_process_in_batch:
                    pbar.update(len(items_in_batch)); continue
                        
                if args.method != "all":
                    current_knowledge_to_inject = extract_required_unknown_knowledge(items_in_batch)
                else:
                    current_knowledge_to_inject = extract_all_required_knowledge(items_in_batch)
                
                method_instance.edit(current_knowledge_to_inject)
                
                for item_to_process in items_to_process_in_batch:
                    processed_item, _ = method_instance.run(item_to_process, current_knowledge_to_inject)
                    processed_data.append(processed_item)
                    with open(output_file_path, 'wb') as f: pickle.dump(processed_data, f)
                    pbar.update(1)
                    
                batch_usage = method_instance.restore()
                
                if batch_usage:
                    t_model = translate_model_name(effective_model_name)
                    token_counts['prompt'] += batch_usage[t_model].get("prompt_tokens", 0)
                    token_counts['completion'] += batch_usage[t_model].get("completion_tokens", 0)
                    token_counts['total'] += batch_usage[t_model].get("total_tokens", 0)
                
                pbar.set_postfix_str(f"Prompt: {token_counts['prompt']}, Completion: {token_counts['completion']}, Total: {token_counts['total']}")