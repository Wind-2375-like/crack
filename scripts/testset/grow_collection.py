import os
import sys
import time
import json
import pickle
import requests
import argparse
from tqdm import tqdm
from wikibaseintegrator import WikibaseIntegrator
from wikibaseintegrator.wbi_config import config as wbi_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers.grow import sample_chain_exact, process_chain, postprocess_chain

def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=20, help="Number of triples to process")
    parser.add_argument('--depths', type=list, default=[2], help="Depth of the chain")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="gpt-5-mini-2025-08-07", help="Model name for the API")
    return parser.parse_args()

class BearerTokenLogin:
    def __init__(self, token, user_agent_string):
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'User-Agent': user_agent_string
        })
    def get_session(self):
        return self.session

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load the API key from the configuration file
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", None).get("user_agent", None)
        args.access_token = api_config.get("wikimedia", None).get("access_token", None)
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
    CACHE_FILE = 'data/grow/non_factual_triples_cache.pkl'  # Initialize and load the non-factual triples cache
    try:
        with open(CACHE_FILE, 'rb') as f:
            non_factual_cache = pickle.load(f)
        print(f"Loaded {len(non_factual_cache)} non-factual triples from cache.")
    except FileNotFoundError:
        non_factual_cache = set()
        print("No cache file found. Initializing an empty cache for non-factual triples.")
        
    try:
        # Ensure your consumer token/secret are correctly set up for OAuth2
        login_with_token = BearerTokenLogin(args.access_token, UA)
        wbi = WikibaseIntegrator(login=login_with_token)
        print("WikibaseIntegrator initialized with Bearer Token login.")
    except Exception as e:
        print(f"Warning: Failed to initialize WikibaseIntegrator with OAuth2 login: {e}")
        print("Falling back to anonymous WikibaseIntegrator.")
        # Fallback to anonymous access if OAuth fails or is not needed for reads
        wbi = WikibaseIntegrator()
    
    with tqdm(total=args.data_size, desc="Processing chains", unit="chain") as pbar:
        for i in range(args.data_size):
            while True:
                try:
                    depth = args.depths[int(i/(args.data_size/len(args.depths)))]
                    chain = sample_chain_exact(depth, wbi, args, non_factual_cache)
                    # Process each chain
                    processed_chain, usage = process_chain(chain, args, chat_response_generator)
                    # # Post Processing and filter bad quality questions
                    # keep_chain, usage = postprocess_chain(processed_chain, args, chat_response_generator, non_factual_cache)
                    # Compute usage
                    prompt_tokens = usage[args.model_name]["prompt_tokens"]
                    completion_tokens = usage[args.model_name]["completion_tokens"]
                    total_tokens = usage[args.model_name]["total_tokens"]
                    # Update the progress bar with the number of tokens used
                    pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                    # Append the processed chain to the list
                    # if not keep_chain:
                    #     continue
                    # else:
                    processed_data.append(processed_chain)
                    pbar.update(1)
                    break
                except Exception as e:
                    time.sleep(1)  # Sleep to avoid hitting API limits or causing too many errors
                    continue
            # Save the processed data to a new pickle file
            with open(f'data/grow/test_{args.data_size}.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
            # Also save the cache file
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(non_factual_cache, f)
            
    # Save the processed data to a new pickle file
    with open(f'data/grow/test_{args.data_size}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(non_factual_cache, f)