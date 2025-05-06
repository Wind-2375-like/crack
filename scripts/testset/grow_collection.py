import os
import sys
import time
import json
import pickle
import argparse
from tqdm import tqdm
from wikibaseintegrator import wbi_login, WikibaseIntegrator
from wikibaseintegrator.wbi_config import config as wbi_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers.grow import sample_chain_exact, process_chain

def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=1, help="Number of triples to process")
    parser.add_argument('--depth', type=int, default=4, help="Depth of the chain")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="Model name for the API")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load the API key from the configuration file
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", None).get("user_agent", None)
        args.consumer_key = api_config.get("wikimedia", None).get("client_application_key", None)
        args.consumer_secret = api_config.get("wikimedia", None).get("client_application_secret", None)
        if args.api_key is None:
            raise ValueError("API key not found in the configuration file.")
        if args.ua is None:
            raise ValueError("User agent not found in the configuration file.")

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
    UA = args.ua
    wbi_config['USER_AGENT'] = UA
    try:
        # Ensure your consumer token/secret are correctly set up for OAuth2
        login = wbi_login.OAuth2(consumer_token=args.consumer_key,
                                  consumer_secret=args.consumer_secret)
        wbi = WikibaseIntegrator(login=login)
        print("WikibaseIntegrator initialized with OAuth2 login.")
    except Exception as e:
        print(f"Warning: Failed to initialize WikibaseIntegrator with OAuth2 login: {e}")
        print("Falling back to anonymous WikibaseIntegrator.")
        # Fallback to anonymous access if OAuth fails or is not needed for reads
        wbi = WikibaseIntegrator()
    
    chains = []
    with tqdm(total=args.data_size, desc="Processing chains", unit="chain") as pbar:
        for i in range(args.data_size):
            while True:
                try:
                    chain = sample_chain_exact(args, wbi, UA)
                    chains.append(chain)
                    # Process each chain
                    processed_chain, usage = process_chain(chain, args, chat_response_generator)
                    # Update the total token counts
                    prompt_tokens += usage[args.model_name]["prompt_tokens"]
                    completion_tokens += usage[args.model_name]["completion_tokens"]
                    total_tokens += usage[args.model_name]["total_tokens"]
                    # Append the processed chain to the list
                    processed_data.append(processed_chain)
                    # Update the progress bar with the number of tokens used
                    pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                    pbar.update(1)
                    break
                except Exception as e:
                    time.sleep(1)  # Sleep to avoid hitting API limits or causing too many errors
                    continue
            
    # Save the processed data to a new pickle file
    with open(f'data/grow/test_{args.data_size}_depth_{args.depth}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)