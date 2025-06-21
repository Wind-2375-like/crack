import os
import sys
import time
import json
import pickle
import argparse
from tqdm import tqdm
from datasets import load_dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers.code import process_item, build_cache_and_generate_knowledge

def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=100, help="Number of triples to process")
    parser.add_argument('--depth', type=int, default=4, help="Depth of the chain")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="gpt-4.1-mini", help="Model name for the API")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load the API key from the configuration file
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)

    # Process all chains
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    processed_data = []
    
    chat_response_generator = ChatResponseGenerator(
        model_name=args.model_name,
        api_key=args.api_key,
        local=False
    )
    
    bigcodebench = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    global_cache = {}
    library_call_knowledge = build_cache_and_generate_knowledge(
        bigcodebench,
        global_cache,
        force_rebuild_cache_for_all=True
    )

    with tqdm(total=args.data_size, desc="Processing codes", unit="code") as pbar:
        i = 0
        count = 0
        while count < args.data_size:
            try:
                item = bigcodebench[i]
                facts = library_call_knowledge[i]['knowledge']
                i += 1
                # Process each item
                processed_item, usage = process_item(item, args, chat_response_generator, facts)
                # Update the total token counts
                prompt_tokens = usage[args.model_name]["prompt_tokens"]
                completion_tokens = usage[args.model_name]["completion_tokens"]
                total_tokens = usage[args.model_name]["total_tokens"]
                # Append the processed item to the list
                processed_data.append(processed_item)
                # Update the progress bar with the number of tokens used
                pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                pbar.update(1)
                count += 1
            except Exception as e:
                time.sleep(1)  # Sleep to avoid hitting API limits or causing too many errors
                continue
            
    # Save the processed data to a new pickle file
    with open(f'data/code/test_{args.data_size}_depth_{args.depth}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)