import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import pickle
import argparse
from tqdm import tqdm
from utils.generator.chat_response_generator import ChatResponseGenerator
from scripts.evaluation.knowledge_evaluation import PROMPT_TEMPLATES as PROMPT_TEMPLATES_EQUIVALENCE
from utils.helpers import translate_model_name
from collections import Counter


PROMPT_TEMPLATES_NLI = {
    "grow": """You are an expert in natural language inference and commonsense reasoning. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.

Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Context mentions some knowledge is "unknown", it should be treated as "N/A" and contradictory to the Statement.

Examples:

Context:
1. As of my knowledge cutoff in December 2023, the current Prime Minister of the United Kingdom is Rishi Sunak, and Keir Starmer's spouse is Victoria Starmer.
2. The user provided that Keir Starmer is the current head of government. As of my knowledge, he is the leader of the Opposition and the leader of the Labour Party in the UK. So it is possible that he is the Prime Minister now.
3. Therefore, the current British Prime Minister is Keir Starmer.
4. Victoria Starmer is married to British Prime Minister.
5. The answer is Victoria Starmer.
Statement:
The name of the current head of the British government is Keir Starmer.
NLI:
Entailment. The context initially stated that Rishi Sunak is the current Prime Minister, but later it corrected that Keir Starmer is the current head of government.

Context:
1. As of my knowledge cutoff in December 2023, the current Prime Minister of the United Kingdom is Rishi Sunak, and Keir Starmer's spouse is Victoria Starmer.
2. The user provided that Keir Starmer is the current head of government. As of my knowledge, he is the leader of the Opposition and the leader of the Labour Party in the UK. So it is possible that he is the Prime Minister now.
3. Therefore, the current British Prime Minister is Keir Starmer.
4. Victoria Starmer is married to British Prime Minister.
5. The answer is Victoria Starmer.
Statement:
The name of the current head of the British government is Rishi Sunak.
NLI:
Contradiction. The context initially stated that Rishi Sunak is the current Prime Minister, but later it corrected that Keir Starmer is the current head of government.

Context:
1. As of my knowledge cutoff in December 2023, the current Prime Minister of the United Kingdom is Rishi Sunak, and Keir Starmer's spouse is Victoria Starmer.
2. The user provided that Keir Starmer is the current head of government. As of my knowledge, he is the leader of the Opposition and the leader of the Labour Party in the UK. So it is possible that he is the Prime Minister now.
3. Therefore, the current British Prime Minister is Keir Starmer.
4. Victoria Starmer is married to British Prime Minister.
5. The answer is Victoria Starmer.
Statement:
The name of the current head of the US government is Donald Trump.
NLI:
Neutral. The context does not provide any information about the current head of the US government.
""",
    "default": """You are an expert in natural language inference and commonsense reasoning. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.
    
Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Context mentions some knowledge is "unknown", it should be treated as "N/A" and contradictory to the Statement.
"""
}

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
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-4o-mini", help="Model name for the evaluation")
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


def evaluate_reasoning_item(item, args, chat_response_generator):
    """
    Evaluates LLM's probed knowledge.
    Args:
        item (dict): A dictionary containing:
            - "id": index of the question
            - "question": the question text
            - "answer": the answer text (ground truth)
            - "required_knowledge": a list of dictionaries containing:
                - "knowledge": the knowledge text
                - "knowledgable": a boolean indicating if the knowledge is knowledgable
                - "knowledge_confidence": a float indicating the confidence in the knowledge
            - "model_response": the response generated by the model
        args (Namespace): Command line arguments. (args.task_name)
        chat_response_generator (ChatResponseGenerator): An instance of the ChatResponseGenerator class for evaluation.
    Returns:
        item (dict): The updated item with additional keys:
            - "required_knowledge": a list of dictionaries containing:
                - "nli_class": entailment/contradiction/neutral
                - "nli_explanation": explanation of the NLI class
            - "final_answer_correct": a boolean indicating if the final answer is correct
            - "final_answer_explanation": explanation of the final answer
        usage (dict): A dictionary containing the token usage information for the model.
    """

    def _parse_llm_equivalence_response(response_text):
        """
        Parses the LLM's response for equivalence check.
        Returns a tuple: (is_equivalent_bool, explanation_str)
        """
        response_text_stripped = response_text.replace("Equivalence:", "").strip()
        is_equivalent = "yes" in response_text_stripped.lower()
        explanation = response_text_stripped
            
        return is_equivalent, explanation

    def _parse_nli_response(response_text):
        """
        Parses the LLM's NLI response.
        Returns a tuple: (nli_class_str, explanation_str)
        """
        response_text_lower = response_text.replace("NLI:", "").strip()
        if "entailment" in response_text_lower.lower():
            nli_class = "entailment"
        elif "contradiction" in response_text_lower.lower():
            nli_class = "contradiction"
        else:
            nli_class = "neutral"

        return nli_class, response_text.strip()

    # 1. Evaluate the final answer by comparing the final line with the answer
    ground_truth_final_answer = item["answer"]
    
    # Extract the model's final answer from its response.
    # This assumes the model's response also puts its final answer on the last non-empty line.
    # If item["model_response"] can be None or empty, add checks.
    model_final_answer_candidate = ""
    if item.get("model_response") and item["model_response"].strip():
        model_response_lines = [line.strip() for line in item["model_response"].split("\n") if line.strip()]
        if model_response_lines:
            model_final_answer_candidate = model_response_lines[-1]
    
    system_prompt_key_equivalence = args.task_name if args.task_name in PROMPT_TEMPLATES_EQUIVALENCE else "default"
    system_prompt_equivalence = PROMPT_TEMPLATES_EQUIVALENCE[system_prompt_key_equivalence]
    
    llm_input_prompt_equivalence = (
        f"Question:\n{item['question']}\n"
        f"Response:\n{model_final_answer_candidate}\n" # Use extracted model's final answer line
        f"Answer:\n{ground_truth_final_answer}\n"
        f"Equivalence:\n"
    )
    chat_response_generator.update_chat_history([
        ("system", system_prompt_equivalence),
    ])
    raw_equivalence_response = chat_response_generator.generate_response(
        llm_input_prompt_equivalence,
        temperature=0, top_p=1, top_k=1, n=1, max_tokens=100
    )[0]
    
    final_answer_correct, final_answer_explanation = _parse_llm_equivalence_response(raw_equivalence_response)
    item["final_answer_correct"] = final_answer_correct
    item["final_answer_explanation"] = final_answer_explanation

    # 2. Evaluate whether each knowledge in required_knowledge is entailment/contradiction/neutral with the model response
    chat_response_generator.update_chat_history([
        ("system", PROMPT_TEMPLATES_NLI[args.task_name]),
    ])
    model_full_response_context = item.get("model_response", "") # Use the full model response as context for NLI

    for knowledge_item in item["required_knowledge"]:
        knowledge_text = knowledge_item["knowledge"]
        llm_input_prompt_nli = (
            f"Context:\n{model_full_response_context}\n\n"
            f"Statement:\n{knowledge_text}\n\n"
            f"NLI:\n"
        )
        
        raw_nli_response = chat_response_generator.generate_response(
            llm_input_prompt_nli,
            temperature=0, top_p=1, top_k=1, n=1, max_tokens=100
        )[0]
        
        nli_class, nli_explanation = _parse_nli_response(raw_nli_response)
        knowledge_item["nli_class"] = nli_class
        knowledge_item["nli_explanation"] = nli_explanation

    return item, chat_response_generator.get_usage()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load the API key from the configuration file
    with open(args.api_config_file, 'r') as f:
        api_config = json.load(f)
        args.api_key = api_config.get("api_key", None)
        args.ua = api_config.get("wikimedia", None).get("user_agent", None)
        if args.api_key is None:
            raise ValueError("API key not found in the configuration file.")
        if args.ua is None:
            raise ValueError("User agent not found in the configuration file.")
        
    # Define input and output paths
    input_file_dir = f'data/eval_results/{args.task_name}/injection/'
    input_file_name = f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_depth_{args.depth}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl"
    input_file_path = os.path.join(input_file_dir, input_file_name)

    output_dir_base = f'data/eval_results/{args.task_name}/injection_evaluated/'
    os.makedirs(output_dir_base, exist_ok=True)
    output_file_name = f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_depth_{args.depth}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl"
    output_file_path = os.path.join(output_dir_base, output_file_name)

    # Load the chains from a pickle file
    with open(input_file_path, 'rb') as f:
        eval_dataset = pickle.load(f)
        
    # Process all chains
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    processed_data = []
    chat_response_generator = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=args.api_key)
    
    # Use tqdm to show progress
    with tqdm(total=len(eval_dataset), desc="Processing chains for evaluation") as pbar:
        for item_idx, item in enumerate(eval_dataset):
            # Process each chain
            processed_item, usage = evaluate_reasoning_item(item, args, chat_response_generator)
            # Update the total token counts
            if args.evaluate_model_name in usage:
                prompt_tokens = usage[args.evaluate_model_name].get("prompt_tokens", 0)
                completion_tokens = usage[args.evaluate_model_name].get("completion_tokens", 0)
                total_tokens = usage[args.evaluate_model_name].get("total_tokens", 0)
            # Append the processed chain to the list
            processed_data.append(processed_item)
            pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            pbar.update(1)

    # Save the processed data to a new pickle file
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nProcessing complete. Evaluated data saved to: {output_file_path}")
    print(f"Total tokens used for evaluation ({args.evaluate_model_name}):")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")