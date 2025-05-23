import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import pickle
import argparse
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
    parser.add_argument('--data_size', type=int, default=100, help="Number of triples to process")
    parser.add_argument('--depth', type=int, default=4, help="Depth of the chain")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b", help="Model name for the API")
    parser.add_argument('--task_name', type=str, default="grow", help="Task name")
    parser.add_argument('--inject_knowledge', action='store_true', help="Whether to inject knowledge into the input")
    parser.add_argument('--knowledge_aggregation_scope', type=int, default=1, help="Scope for aggregating 'unknown' knowledge. Must be >= 1. 1: item-specific. N (e.g., 10, 100): group of N items.")
    parser.add_argument('--method', type=str, default="base", help="Method to use for complex reasoning amid conflicting knowledge")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for the model")
    parser.add_argument('--top_p', type=float, default=0.7, help="Top-p sampling for the model")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k sampling for the model")
    parser.add_argument('--max_tokens', type=int, default=512, help="Maximum tokens for the model")
    parser.add_argument('--num_responses', type=int, default=1, help="Number of responses to generate")
    return parser.parse_args()


def prepare_input(item, args, knowledge_to_inject_str=""):
    """
    Prepares the input for the model.
    Args:
        item (dict): A dictionary containing:
            - "id": index of the question
            - "question": the question text
            - "answer": the answer text
            - "required_knowledge": the list of required knowledge
        args (Namespace): Command line arguments.
        knowledge_to_inject_str (str, optional): Pre-formatted string of knowledge to inject. Defaults to "".
    Returns:
        tuple: A tuple containing:
            - prepared_user_prompt (str): The user prompt for the model.
            - prepared_system_prompt (str): The system prompt for the model.
    """
    
    unified_system_prompt = (
        "You are given a question. To answer the question, you should think step by step. "
        "Use line breaks between steps, but do not use line breaks within each step. "
        "You should number each step. The final answer to the question should start with "
        "\"The answer is ...\", and should be placed at the final step. "
        "Users may also provide a set of facts. If they conflict with your knowledge, you should update your "
        "knowledge and use the facts to answer the question.\n\n"
        "[Here are two demonstrations]\n\n"
        "User:\nWhat is the capital of the country where Plainfield Town Hall is located?\n\n"
        "Assistant:\n"
        "1. Plainfield Town Hall is one of two town halls in Plainfield, New Hampshire.\n"
        "2. New Hampshire is a state in the New England region of the Northeastern United States.\n"
        "3. Thus, Plainfield Town Hall is located in the country of the United States of America. \n"
        "4. The capital of United States is Washington, D.C.\n"
        "5. The answer is Washington, D.C.\n\n"
        "User:\nWho is married to the British Prime Minister?\nPlease update your knowledge with the following facts:\n"
        "The name of the current head of the British government is Keir Starmer.\n\n"
        "Assistant:\n"
        "1. As of my knowledge cutoff in December 2023, the current Prime Minister of the United Kingdom is Rishi Sunak, and Keir Starmer's spouse is Victoria Starmer.\n"
        "2. The user provided that Keir Starmer is the current head of government. As of my knowledge, he is the leader of the Opposition and the leader of the Labour Party in the UK. So it is possible that he is the Prime Minister now.\n"
        "3. Therefore, the current British Prime Minister is Keir Starmer.\n"
        "4. Victoria Starmer is married to British Prime Minister.\n"
        "5. The answer is Victoria Starmer."
    )
    
    prepared_system_prompt = unified_system_prompt

    if args.task_name == "grow":
        if args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
            prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
        else: # No knowledge injection or no "unknown" knowledge found for this scope
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
    else:
        raise NotImplementedError(f"Task {args.task_name} is not implemented.")

    return prepared_user_prompt, prepared_system_prompt


def experiment(item, args, chat_response_generator, knowledge_to_inject_str=""):
    """
    Probes a chain of triples using the specified model.
    Args:
        item (dict): A dictionary containing question and other details.
        args (Namespace): Command line arguments.
        chat_response_generator (ChatResponseGenerator): An instance of the ChatResponseGenerator class.
        knowledge_to_inject_str (str, optional): Pre-formatted string of knowledge to inject. Defaults to "".
    Returns:
        item (dict): The updated item with model_response.
        usage (dict): A dictionary containing the token usage information for the model.
    """
    
    prepared_user_prompt, prepared_system_prompt = prepare_input(item, args, knowledge_to_inject_str)
    
    chat_response_generator.update_chat_history([
        ("system", prepared_system_prompt),
    ])
    
    model_response = chat_response_generator.generate_response(
        prepared_user_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.num_responses,
        max_tokens=args.max_tokens,
    )[0].replace("Assistant:", "").strip()
        
    item["model_response"] = model_response

    return item, chat_response_generator.get_usage()

def extract_required_unknown_knowledge(items_list):
    """
    Helper to extract unique knowledge strings from a list of items,
    where 'knowledgable' is False.
    """
    unknown_knowledge_set = set()
    for an_item in items_list:
        for k_entry in an_item.get('required_knowledge', []):
            # Check if 'knowledgable' is False and 'knowledge' string exists and is not empty
            if not k_entry.get('knowledgable', True) and k_entry.get('knowledge'):
                unknown_knowledge_set.add(k_entry['knowledge'])
    return " ".join(sorted(list(unknown_knowledge_set)))


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
    # Corrected assertion: raise error if scope is not positive
    assert args.knowledge_aggregation_scope >= 1 , "knowledge_aggregation_scope must be a positive integer (>= 1)."

    eval_dataset = ReasoningEvalDataset(
        raw_path=f'data/{args.task_name}/test_{args.data_size}_depth_{args.depth}.pkl',
        probe_path=f'data/eval_results/{args.task_name}/probe/eval_{args.data_size}_depth_{args.depth}_{args.model_name}.pkl',
    )
    
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", {}).get("user_agent", None)
        if args.api_key is None:
            raise ValueError("API key not found in the configuration file.")
        if args.ua is None:
            raise ValueError("User agent not found in the configuration file.")

    token_counts = {'prompt': 0, 'completion': 0, 'total': 0}
    processed_data = []
    chat_response_generator = ChatResponseGenerator(model_name=translate_model_name(args.model_name), api_key=args.api_key)

    all_items_list = list(eval_dataset) # Convert dataset to list for consistent indexing and sizing
    dataset_size = len(all_items_list)

    # Create output directory if it doesn't exist
    output_dir = f'data/eval_results/{args.task_name}/injection/'
    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=dataset_size, desc="Processing items") as pbar: # Use dataset_size for tqdm
        if not args.inject_knowledge:
            for i, item in enumerate(all_items_list): # Iterate over all_items_list
                processed_item, usage = experiment(item, args, chat_response_generator, knowledge_to_inject_str="")
                update_pbar(processed_item, usage, processed_data, token_counts, pbar, args.model_name)
        else:
            scope = args.knowledge_aggregation_scope
            
            # Outer loop iterates by groups, defined by the 'scope'
            # If scope is 1, each group contains one item (item-specific).
            # If scope > 1, groups contain 'scope' items (or fewer for the last group).
            for group_start_idx in range(0, dataset_size, scope):
                group_end_idx = min(group_start_idx + scope, dataset_size)
                
                # Items from which knowledge is extracted for the current conceptual group
                items_for_knowledge_extraction = all_items_list[group_start_idx:group_end_idx]
                current_knowledge_to_inject = extract_required_unknown_knowledge(items_for_knowledge_extraction)
                
                # Process each item within this conceptual group using the extracted knowledge
                # The items_for_knowledge_extraction list itself contains the items to process for this group
                for item_to_process in items_for_knowledge_extraction:
                    processed_item, usage = experiment(item_to_process, args, chat_response_generator, current_knowledge_to_inject)
                    update_pbar(processed_item, usage, processed_data, token_counts, pbar, args.model_name)
            
    # Use the exact output filename format you requested
    output_file_path = os.path.join(output_dir, f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_depth_{args.depth}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_data, f)
