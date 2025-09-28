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
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=500, help="Number of triples to process")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b", help="Model name for the API")
    parser.add_argument('--task_name', type=str, default="grow", help="Task name")
    parser.add_argument('--inject_knowledge', action='store_true', help="Whether to inject knowledge into the input")
    parser.add_argument('--knowledge_aggregation_scope', type=int, default=1, help="Scope for aggregating 'unknown' knowledge. Must be >= 1. 1: item-specific. N (e.g., 10, 100): group of N items.")
    parser.add_argument('--method', type=str, default="base", help="Method to use for complex reasoning amid conflicting knowledge")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for the model")
    parser.add_argument('--top_p', type=float, default=0.7, help="Top-p sampling for the model")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum tokens for the model")
    parser.add_argument('--num_responses', type=int, default=1, help="Number of responses to generate")
    return parser.parse_args()

def extract_required_unknown_knowledge(items_list, raw_items):
    """
    Helper to extract unique knowledge strings from a list of items,
    where 'knowledgable' is False.
    """
    seen_knowledge = set()
    unknown_knowledge_list = []
    for i, an_item in enumerate(items_list):
        for j, k_entry in enumerate(an_item.get('required_knowledge', [])):
            knowledge_str = k_entry.get('knowledge')
            if not k_entry.get('knowledgable', True) and knowledge_str and knowledge_str not in seen_knowledge:
                seen_knowledge.add(knowledge_str)
                unknown_knowledge_list.append({
                    'knowledge': k_entry['knowledge'],
                    'knowledgable': k_entry['knowledgable'],
                    'knowledge_confidence': k_entry['knowledge_confidence'],
                    'probe_question': raw_items[i]['probe_questions'][j]['question'],
                    'probe_answer': raw_items[i]['probe_questions'][j]['answer'],
                })
    return unknown_knowledge_list
    
def extract_all_required_knowledge(items_list, raw_items):
    """
    Helper to extract unique knowledge strings from a list of items,
    where 'knowledgable' is False.
    """
    seen_knowledge = set()
    all_knowledge_list = []
    for i, an_item in enumerate(items_list):
        for j, k_entry in enumerate(an_item.get('required_knowledge', [])):
            knowledge_str = k_entry.get('knowledge')
            if knowledge_str and knowledge_str not in seen_knowledge:
                all_knowledge_list.append({
                    'knowledge': k_entry['knowledge'],
                    'knowledgable': k_entry['knowledgable'],
                    'knowledge_confidence': k_entry['knowledge_confidence'],
                    'probe_question': raw_items[i]['probe_questions'][j]['question'],
                    'probe_answer': raw_items[i]['probe_questions'][j]['answer'],
                })
    return all_knowledge_list

def update_pbar(processed_item, usage, processed_data_list, token_counts_dict, pbar_instance, model_name_str): # Renamed from _process_and_update_results
    """Helper to update token counts, append data, and refresh progress bar."""
    translated_model_name = translate_model_name(model_name_str)
    token_counts_dict['prompt'] = usage[translated_model_name]["prompt_tokens"]
    token_counts_dict['completion'] = usage[translated_model_name]["completion_tokens"]
    token_counts_dict['total'] = usage[translated_model_name]["total_tokens"]
    processed_data_list.append(processed_item)
    pbar_instance.set_postfix_str(f"Prompt: {token_counts_dict['prompt']}, Completion: {token_counts_dict['completion']}, Total: {token_counts_dict['total']}")
    pbar_instance.update(1)


if __name__ == "__main__":
    args = parse_args()
    assert args.knowledge_aggregation_scope >= 1 , "knowledge_aggregation_scope must be a positive integer (>= 1)."

    eval_dataset = ReasoningEvalDataset(
        raw_path=f'data/{args.task_name}/test_{args.data_size}.pkl',
        probe_path=f'data/eval_results/{args.task_name}/probe_evaluated/test_{args.data_size}_{args.model_name}.pkl',
    )
    
    with open(f'data/{args.task_name}/test_{args.data_size}.pkl', "rb") as f:
        raw_dataset = pickle.load(f)
    
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", {}).get("user_agent", None)
        if args.api_key is None:
            raise ValueError("API key not found in the configuration file.")
        if args.ua is None:
            raise ValueError("User agent not found in the configuration file.")
        
    # Setup for Resume Functionality
    output_dir = f'data/eval_results/{args.task_name}/injection/'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl")

    processed_data = []
    start_index = 0
    if os.path.exists(output_file_path):
        print(f"--- Resuming from checkpoint: {output_file_path} ---")
        with open(output_file_path, 'rb') as f:
            processed_data = pickle.load(f)
        start_index = len(processed_data)
        print(f"--- Found {start_index} items already processed. Resuming... ---")
        
    effective_model_name = args.model_name
    method_module_name = args.method

    if args.method == 'append_t':
        if 'qwen-3' not in args.model_name:
            print("❌ Error: The 'append_t' method is only compatible with 'qwen-3' models.")
            sys.exit(1)
        # Use the 'thinking' version of the model
        effective_model_name += "-thinking"
        # The underlying implementation is 'base' (or 'append_t' if you have the file)
        method_module_name = 'base'

    token_counts = {'prompt': 0, 'completion': 0, 'total': 0}
    chat_response_generator = ChatResponseGenerator(
        model_name=translate_model_name(effective_model_name),
        api_key=args.api_key
    )

    all_items_list = list(eval_dataset) # Convert dataset to list for consistent indexing and sizing
    dataset_size = len(all_items_list)

    # Create output directory if it doesn't exist
    output_dir = f'data/eval_results/{args.task_name}/injection/'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if method_module_name == "all":
            method_module = importlib.import_module(f"utils.methods.base")
        else:
            method_module = importlib.import_module(f"utils.methods.{method_module_name}")
        MethodClass = getattr(method_module, 'Method')
        method_instance = MethodClass(args, chat_response_generator)
    except (ImportError, AttributeError) as e:
        print(f"❌ Error: Could not load the method '{args.method}'.")
        print(f"Ensure 'methods/{args.method}.py' exists and contains a class named 'Method' that inherits from RootExperimentMethod.")
        sys.exit(1)
        
    # Modify the main data list to only include unprocessed items
    items_to_process = all_items_list[start_index:]

    with tqdm(total=dataset_size, initial=start_index, desc="Processing items") as pbar:
        if not args.inject_knowledge:
            for item in items_to_process:
                processed_item, usage = method_instance.run(item, [])
                update_pbar(processed_item, usage, processed_data, token_counts, pbar, args.model_name)
                # Incremental save
                with open(output_file_path, 'wb') as f:
                    pickle.dump(processed_data, f)
        else:
            scope = args.knowledge_aggregation_scope
            
            # Outer loop iterates by groups, defined by the 'scope'
            # If scope is 1, each group contains one item (item-specific).
            # If scope > 1, groups contain 'scope' items (or fewer for the last group).
            for group_start_idx in range(0, dataset_size, scope):
                group_end_idx = min(group_start_idx + scope, dataset_size)
                
                # Items from which knowledge is extracted for the current conceptual group
                items_for_knowledge_extraction = all_items_list[group_start_idx:group_end_idx]
                raw_items = raw_dataset[group_start_idx:group_end_idx]
                
                # Determine which items in *this specific batch* still need processing
                current_batch_items_to_process = []
                local_start_index = max(0, start_index - group_start_idx)

                # Directly slice the list to get only the items that still need processing.
                current_batch_items_to_process = items_for_knowledge_extraction[local_start_index:]
                        
                if not current_batch_items_to_process:
                    # This batch is already fully processed, but we might need to update pbar
                    pbar.update(len(items_for_knowledge_extraction))
                    continue
                        
                if args.method != "all":
                    current_knowledge_to_inject = extract_required_unknown_knowledge(items_for_knowledge_extraction, raw_items)
                else:
                    current_knowledge_to_inject = extract_all_required_knowledge(items_for_knowledge_extraction, raw_items)
                
                # Edit the model ONCE for the entire group
                method_instance.edit(current_knowledge_to_inject)
                
                # Process each item within this conceptual group using the extracted knowledge
                # The items_for_knowledge_extraction list itself contains the items to process for this group
                for item_to_process in current_batch_items_to_process:
                    processed_item, _ = method_instance.run(item_to_process, current_knowledge_to_inject)
                    processed_data.append(processed_item)
                    
                    # Incremental save after each item
                    with open(output_file_path, 'wb') as f:
                        pickle.dump(processed_data, f)
                    
                    pbar.update(1)
                    
                batch_usage = method_instance.restore()
                
                translated_model_name = translate_model_name(args.model_name)
                batch_prompt = batch_usage[translated_model_name]["prompt_tokens"]
                batch_completion = batch_usage[translated_model_name]["completion_tokens"]
                batch_total = batch_usage[translated_model_name]["total_tokens"]

                token_counts['prompt'] += batch_prompt
                token_counts['completion'] += batch_completion
                token_counts['total'] += batch_total
                
                pbar.set_postfix_str(f"Prompt: {token_counts['prompt']}, Completion: {token_counts['completion']}, Total: {token_counts['total'] }")
            
    # Use the exact output filename format you requested
    output_file_path = os.path.join(output_dir, f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_data, f)
