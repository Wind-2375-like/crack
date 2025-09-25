import os
import sys
import json
import pickle
import argparse
import importlib
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers import translate_model_name

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run knowledge probing with injected knowledge.")
    parser.add_argument('--data_size', type=int, default=500, help="Size of the dataset to process.")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to API configuration.")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b", help="Model name for the API.")
    parser.add_argument('--task_name', type=str, required=True, help="Task name (e.g., 'grow', 'code', 'math').")
    parser.add_argument('--method', type=str, default="base", help="Knowledge injection method to use.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.7, help="Sampling top-p.")
    parser.add_argument('--max_tokens', type=int, default=100, help="Max tokens for generation.")
    parser.add_argument('--num_responses', type=int, default=10, help="Number of responses to generate per probe.")
    return parser.parse_args()


def probe_with_injection(item, all_probes, method_instance, args):
    """
    Injects knowledge for an efficacy probe and then tests both efficacy and locality probes.
    """
    # 1. Identify the efficacy and locality probes from the full dataset
    efficacy_probe = all_probes[item['efficacy_id']]
    locality_probe = all_probes[item['locality_id']]

    # 2. Define the single piece of knowledge to inject based on the efficacy probe
    # The 'run' methods in ft_ck, rome, etc., expect a list of knowledge dictionaries
    knowledge_to_inject = [{
        'knowledge': efficacy_probe['knowledge'],
        'probe_question': efficacy_probe['question'],
        'probe_answer': efficacy_probe['answer']
    }]

    # 3. Use the method's `run` function to inject knowledge and get the EFFICACY response
    processed_efficacy_item, efficacy_usage = method_instance.run(efficacy_probe, knowledge_to_inject, probe=True)

    # 4. Now, probe the LOCALITY question using the *same* knowledge-edited model state
    processed_locality_item, locality_usage = method_instance.run(locality_probe, knowledge_to_inject, probe=True)

    # 5. Store the results clearly
    item['post_injection_efficacy_answers'] = processed_efficacy_item.get('probe_answers')
    item['post_injection_locality_answers'] = processed_locality_item.get('probe_answers')

    # Aggregate usage (this is an approximation, but useful for tracking)
    total_usage = {
        'prompt_tokens': efficacy_usage.get('prompt_tokens', 0) + locality_usage.get('prompt_tokens', 0),
        'completion_tokens': efficacy_usage.get('completion_tokens', 0) + locality_usage.get('completion_tokens', 0),
        'total_tokens': efficacy_usage.get('total_tokens', 0) + locality_usage.get('total_tokens', 0)
    }
    
    return item, total_usage


if __name__ == "__main__":
    args = parse_args()

    # --- Load Data and Configs ---
    input_path = f'data/{args.task_name}/test_{args.data_size}_probe.pkl'
    print(f"Loading data from: {input_path}")
    with open(input_path, 'rb') as f:
        probe_items_with_ids = pickle.load(f)

    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        if args.api_key is None:
            raise ValueError("API key not found in the configuration file.")
        
    effective_model_name = args.model_name
    method_module_name = args.method
    
    if args.method == 'append_t':
        if 'qwen-3' not in args.model_name:
            print("❌ Error: The 'append_t' method is only compatible with 'qwen-3' models.")
            sys.exit(1)
        # Use the 'thinking' version of the model
        effective_model_name += "-thinking"
        # The underlying implementation is 'base' (or 'append_t' if you have the file)
        method_module_name = 'append_t'

    # --- Initialize Model and Method ---
    chat_response_generator = ChatResponseGenerator(
        model_name=translate_model_name(effective_model_name), # Use effective model name
        api_key=args.api_key
    )

    try:
        # Dynamically import the specified method's module
        method_module = importlib.import_module(f"utils.methods.{method_module_name}")
        MethodClass = getattr(method_module, 'Method')
        # This custom probe_item function will be used inside the method's run call
        method_instance = MethodClass(args, chat_response_generator)
    except (ImportError, AttributeError):
        print(f"Error: Could not load method '{args.method}'.")
        sys.exit(1)

    processed_data = []
    token_counts = {'prompt': 0, 'completion': 0, 'total': 0}
    translated_model_name = translate_model_name(args.model_name)

    with tqdm(total=len(probe_items_with_ids), desc="Probing with Injection") as pbar:
        for item in probe_items_with_ids:
            # 1. Identify probes and knowledge
            efficacy_probe = probe_items_with_ids[item['efficacy_id']]
            locality_probe = probe_items_with_ids[item['locality_id']]
            knowledge_to_inject = [{'knowledge': efficacy_probe['knowledge'], 'probe_question': efficacy_probe['question'], 'probe_answer': efficacy_probe['answer']}]

            # 2. Edit the model ONCE for this efficacy/locality pair
            method_instance.edit(
                knowledge_to_inject=knowledge_to_inject,
            )
            
            # 3. Probe efficacy and locality on the edited model
            processed_efficacy, _ = method_instance.run(efficacy_probe, probe=True)
            processed_locality, _ = method_instance.run(locality_probe, probe=True)
            
            # 4. Restore the model and get usage
            usage = method_instance.restore()
            batch_prompt = usage[translated_model_name]["prompt_tokens"]
            batch_completion = usage[translated_model_name]["completion_tokens"]
            batch_total = usage[translated_model_name]["total_tokens"]

            token_counts['prompt'] += batch_prompt
            token_counts['completion'] += batch_completion
            token_counts['total'] += batch_total

            # 5. Store results and update progress
            item['post_injection_efficacy_answers'] = processed_efficacy.get('probe_answers')
            item['post_injection_locality_answers'] = processed_locality.get('probe_answers')
            processed_data.append(item)
            pbar.set_postfix_str(f"Prompt: {token_counts['prompt']}, Completion: {token_counts['completion']}, Total: {token_counts['total'] }")
            pbar.update(1)

    # --- Save Results ---
    output_dir = f'data/eval_results/{args.task_name}/injection_probe/'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.method}_{args.data_size}_{args.model_name}.pkl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print("Experiment complete.")