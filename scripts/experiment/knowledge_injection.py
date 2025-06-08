import os
import sys
import pandas
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

    if args.task_name == "grow":
        system_prompt_without_injection = (
            "You are given a question. To answer the question, you should think step by step. "
            "Use line breaks between steps, but do not use line breaks within each step. "
            "The final answer to the question should start with "
            "\"The answer is ...\", and should be placed at the final step. "
            "Please make an educated guess and always return an entity.\n\n"
            "[Here is one demonstration]\n\n"
            "User:\nWhat is the capital of the country where Plainfield Town Hall was created?\n\n"
            "Assistant:\n"
            "1. Given my knowledge, Plainfield Town Hall was created in the United States of America.\n"
            "2. Given my knowledge, the capital of United States is Washington, D.C.\n"
            "3. Therefore, the capital of the country where Plainfield Town Hall was created is Washington, D.C.\n"
            "4. The answer is Washington, D.C.\n\n"
        )
        
        system_prompt_after_injection = (
            "You are given a question. To answer the question, you should think step by step. "
            "Use line breaks between steps, but do not use line breaks within each step. "
            "The final answer to the question should start with "
            "\"The answer is ...\", and should be placed at the final step. "
            "Please make an educated guess and always return an entity. "
            "Users may provide a set of facts or not. If they provide facts that conflict with your knowledge, you should update your "
            "knowledge and use the facts to answer the question.\n\n"
            "[Here is one demonstration]\n\n"
            "User:\nWho is the person who is the current head of government of British married to?\nPlease update your knowledge with the following facts:\n"
            "The name of the current head of the British government is Keir Starmer.\n\n"
            "Assistant:\n"
            "1. The user provided that Keir Starmer is the current head of government of the British government.\n"
            "2. I will update my knowledge with the provided fact: the current head of government of the British government is Keir Starmer.\n"
            "3. Given my knowledge, Keir Starmer is married to Victoria Starmer.\n"
            "4. Therefore, the person who is the current head of government of British married to is Victoria Starmer.\n"
            "5. The answer is Victoria Starmer."
        )
        if args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
            prepared_system_prompt = system_prompt_after_injection
            prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
        else: # No knowledge injection or no "unknown" knowledge found for this scope
            prepared_system_prompt = system_prompt_without_injection
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
    elif args.task_name == "code":
        prompt_solution = """```python
import pandas as pd

def task_func(dealer_sales_data):
    # Step 1: Create DataFrame & Step 2: Handle Empty Input (if dealer_sales_data is empty)
    df = pd.DataFrame(dealer_sales_data)
    
    if not dealer_sales_data:
        return []

    # Ensure 'id' and 'num_sold' columns exist, otherwise it's malformed input
    if 'id' not in df.columns or 'num_sold' not in df.columns:
        return []

    # Step 3: Find Max Sales
    max_sold = df['num_sold'].max()

    # Step 4: Identify Top Sellers
    top_selling_cars = df[df['num_sold'] == max_sold]

    # Step 5: Extract and Sort IDs
    top_selling_ids = top_selling_cars['id'].tolist()
    sorted_ids = sorted(top_selling_ids)

    # Step 6: Return Result
    return sorted_ids
```"""
    
        system_prompt_without_injection = (
            "You are given a coding question. To answer the question, you should think step by step. "
            "You should also write self-contained code starting with\n"
            "```python\n"
            "import ...\n"
            "def task_func(...):\n"
            "\t...\n"
            "```\n"
            "[Here is one demonstration]\n\n"
            "User:\nCompute and return the IDs of the best-selling cars from `dealer_sales_data`, a list of dictionaries, with each composed of an 'id' key (a string identifier) and a 'num_sold' key (an integer). The function should output with:\nids: A list of string ids for the car(s) that sell the best. If multiple car ids achieve the maximum sales, all are returned. The list should be sorted alphabetically for consistent output.\nYou should write self-contained code starting with:\n```python\nimport pandas as pd\ndef task_func(dealer_sales_data):\n```\n\n"
            "Assistant:\n"
            "Okay, I'll outline the steps to solve this problem and then provide the Python code using pandas.\n"
            "Here's the plan:\n"
            "1.  **Create DataFrame**: Convert the list of car records into a pandas DataFrame as we are required to use the pandas library.\n"
            "2.  **Handle Empty Input**: If the initial data is empty (resulting in an empty DataFrame), there's no data to process, so we should return an empty list early.\n"
            "3.  **Find Max Sales**: Find the maximum value in their 'num_sold' column. This gives us the sales figure of the best-selling car(s).\n"
            "4.  **Identify Top Sellers**: Filter the DataFrame again to get only those cars whose 'num_sold' is equal to the maximum sales figure found in the previous step.\n"
            "5.  **Extract and Sort IDs**: From these top-selling cars, extract their 'id' values into a list. Then, sort this list of IDs alphabetically.\n"
            "6.  **Return Result**: The sorted list of IDs is the final answer.\n\n"
            "Now, let's implement this solution.\n\n"
        ) + prompt_solution
        
        system_prompt_after_injection = (
            "You are given a coding question. To answer the question, you should think step by step. "
            "You should also write self-contained code starting with\n"
            "```python\n"
            "import ...\n"
            "def task_func(...):\n"
            "\t...\n"
            "```\n"
            "Users may provide a set of functions and docstrings as facts. If they provide facts that conflict with your knowledge, you should update your "
            "knowledge and use the facts to answer the question.\n\n"
            "[Here is one demonstration]\n\n"
            "User:\nCompute and return the IDs of the best-selling cars from `dealer_sales_data`, a list of dictionaries, with each composed of an 'id' key (a string identifier) and a 'num_sold' key (an integer). The function should output with:\nids: A list of string ids for the car(s) that sell the best. If multiple car ids achieve the maximum sales, all are returned. The list should be sorted alphabetically for consistent output.\nYou should write self-contained code starting with:\n```python\nimport pandas as pd\ndef task_func(dealer_sales_data):\n```\n"
            "Please update your knowledge with following facts:\n"
            f"Function: pandas.DataFrame.max()\n\nDocstring: {pandas.DataFrame.max.__doc__.strip()}\n\n"
            "Assistant:\n"
            "The user provided the docstring of the max function for pandas DataFrame. I'll update my knowledge with user-provided facts, outline the steps to solve this problem, and then provide the Python code using pandas.\n"
            "Here's the plan:\n"
            "1.  **Create DataFrame**: Convert the list of car records into a pandas DataFrame as we are required to use the pandas library.\n"
            "2.  **Handle Empty Input**: If the initial data is empty (resulting in an empty DataFrame), there's no data to process, so we should return an empty list early.\n"
            "3.  **Find Max Sales**: Find the maximum value in their 'num_sold' column. We need to use the max function for pandas DataFrame. This gives us the sales figure of the best-selling car(s).\n"
            "4.  **Identify Top Sellers**: Filter the DataFrame again to get only those cars whose 'num_sold' is equal to the maximum sales figure found in the previous step.\n"
            "5.  **Extract and Sort IDs**: From these top-selling cars, extract their 'id' values into a list. Then, sort this list of IDs alphabetically.\n"
            "6.  **Return Result**: The sorted list of IDs is the final answer.\n\n"
            "Now, let's implement this solution.\n\n"
        ) + prompt_solution
        if args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
            prepared_system_prompt = system_prompt_after_injection
            prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
        else: # No knowledge injection or no "unknown" knowledge found for this scope
            prepared_system_prompt = system_prompt_without_injection
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
        n=args.num_responses,
        max_tokens=args.max_tokens,
    )[0].replace("Assistant:", "").strip()
        
    item["model_response"] = model_response

    return item, chat_response_generator.get_usage()

def extract_required_unknown_knowledge(items_list, task_name):
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
    if task_name == "grow":
        return " ".join(sorted(list(unknown_knowledge_set)))
    else:
        return "\n\n".join(sorted(list(unknown_knowledge_set)))
    
def extract_all_required_knowledge(items_list, task_name):
    """
    Helper to extract unique knowledge strings from a list of items,
    where 'knowledgable' is False.
    """
    all_knowledge_set = set()
    for an_item in items_list:
        for k_entry in an_item.get('required_knowledge', []):
            # Check if 'knowledgable' is False and 'knowledge' string exists and is not empty
            if k_entry.get('knowledge'):
                all_knowledge_set.add(k_entry['knowledge'])
    if task_name == "grow":
        return " ".join(sorted(list(all_knowledge_set)))
    else:
        return "\n\n".join(sorted(list(all_knowledge_set)))

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
        probe_path=f'data/eval_results/{args.task_name}/probe_evaluated/test_{args.data_size}_depth_{args.depth}_{args.model_name}.pkl',
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
                if args.method != "all":
                    current_knowledge_to_inject = extract_required_unknown_knowledge(items_for_knowledge_extraction, args.task_name)
                else:
                    current_knowledge_to_inject = extract_all_required_knowledge(items_for_knowledge_extraction, args.task_name)
                
                # Process each item within this conceptual group using the extracted knowledge
                # The items_for_knowledge_extraction list itself contains the items to process for this group
                for item_to_process in items_for_knowledge_extraction:
                    processed_item, usage = experiment(item_to_process, args, chat_response_generator, current_knowledge_to_inject)
                    update_pbar(processed_item, usage, processed_data, token_counts, pbar, args.model_name)
            
    # Use the exact output filename format you requested
    output_file_path = os.path.join(output_dir, f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_depth_{args.depth}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_data, f)
