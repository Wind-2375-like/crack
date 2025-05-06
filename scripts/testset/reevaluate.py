import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import argparse
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers import translate_model_name
from utils.evaluator import CotEvaluator, knowledge_confidence_evaluator


def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b-turbo", help="Model name for the API")
    return parser.parse_args()


def probe_chain(chain, args):
    """
    Probes a chain of triples using the specified model.
    Args:
        chain (list): A list of dictionaries representing the chain. It includes the following keys:
            - "triples: A list of dictionaries, each containing:
                - "triple": A tuple of (subject_id, predicate_id, object_id)
                - "triple_label": A tuple of (subject_label, predicate_label, object_label)
            - "probe_questions": A list of dictionaries to probe each triple. Each dictionary contains:
                - "question": The question format for probing the triple.
                - "cloze": The cloze format for the triple.
                - "answer": A string of ground truth answer for the question.
            - "multihop_questions": A list of questions (same meaning) for the chain.
            - "multihop_answer": A string of possible answer for the rephrased questions.
        args (Namespace): Command line arguments.
    Returns:
        chain (list): The processed chain with the following additional keys:
            - "model_answers": A list of answers for the chain.
            - "model_answers_correct": A list of tuple (boolean values, str explanations) indicating whether the model's answer is correct.
            - "model_probe_answers": A list of lists of answers for each probe question.
            - "model_probe_answers_correct": A list of lists of tuple (boolean values, str explanations) indicating whether the model's answer is correct.
            - "model_probe_knowledge_confidence": A list of str indicating the model's confidence in its knowledge.
        usage (dict): A dictionary containing the token usage information for the model.
    """
    # Initialize the response generator
    chat_response_generator = ChatResponseGenerator(model_name=translate_model_name(args.model_name), api_key=args.api_key, local=False)
    cot_evaluator = CotEvaluator(args)
    
    # 1. Get the model answers for each multihop question
    chat_response_generator.update_chat_history([
        ("system", "You are given a question. To answer the question, you should think step by step. Use line breaks between steps, but do not use line breaks within each step. You should number each step. The final answer to the question should start with \\\"The answer is ...\\\", and should be placed at the final step.\n\n[Here is one demonstration]\n\nUser:\nWhat is the capital of the country where Plainfield Town Hall is located?\n\nAssistant:\n1. Plainfield Town Hall is one of two town halls in Plainfield, New Hampshire.\n2. New Hampshire is a state in the New England region of the Northeastern United States.\n3. Thus, Plainfield Town Hall is located in the country of the United States of America. \n4. The capital of United States is Washington, D.C.\n5. The answer is Washington, D.C."),
    ])
    
    model_answers = []
    model_answers_correct = []
    for question in chain["multihop_questions"]:
        responses = chat_response_generator.generate_response(
            f"User:\n{question}\nAssistant:\n",
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            n=1,
        )
        responses = [response.replace("Assistant:", "").strip() for response in responses]
        responses_correct = [cot_evaluator.answer_evaluator_cot(question, response, chain["multihop_answer"]) for response in responses]
        model_answers.extend(responses)
        model_answers_correct.extend(responses_correct)
    
    # 2. Get the model answers for each probe question
    model_probe_answers_list = []
    model_probe_answers_correct_list = []
    model_probe_knowledge_confidence = []
    for probe in chain["probe_questions"]:
        model_probe_answers = []
        model_probe_answers_correct = []
        # 2.1 Send the probe question to the model
        chat_response_generator.update_chat_history([
            ("system", "Answer the question with the name of an entity. Provide only the name of the entity as your answer. If you are unsure, please make an educated guess.\n\n[Here is one demonstration]\n\nUser:\nWho is the developer of Telegram?\n\nAssistant:\nTelegram FZ-LLC"),
        ])
        probe_question = probe["question"]
        response_question_no_randomness = chat_response_generator.generate_response(
            f"User:\n{probe_question}\nAssistant:\n",
            temperature=0,
            top_p=1,
            top_k=1,
            n=1,
            max_tokens=100,
        )[0].replace("Assistant:", "").strip()
        responses = chat_response_generator.generate_response(
            f"User:\n{probe_question}\nAssistant:\n",
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            n=5,
            max_tokens=100,
        )
        responses = [response.replace("Assistant:", "").strip() for response in responses]
        model_probe_answers.extend(responses)
        # 2.2 Send the cloze question to the model
        chat_response_generator.update_chat_history([
            ("system", "Fill in the blank with the name of an entity. Provide only the name of the entity as your answer. If you are unsure, please make an educated guess.\n\n[Here is one demonstration]\n\nUser:\nTelegram was developed by ___\n\nAssistant:\nTelegram FZ-LLC"),
        ])
        cloze_question = probe["cloze"]
        response_cloze_no_randomness = chat_response_generator.generate_response(
            f"User:\n{cloze_question}\nAssistant:\n",
            temperature=0,
            top_p=1,
            top_k=1,
            n=1,
            max_tokens=100,
        )[0].replace("Assistant:", "").strip()
        responses = chat_response_generator.generate_response(
            f"User:\n{cloze_question}\nAssistant:\n",
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            n=5,
            max_tokens=100,
        )
        responses = [response.replace("Assistant:", "").strip() for response in responses]
        model_probe_answers.extend(responses)
        model_probe_answers = [response_question_no_randomness, response_cloze_no_randomness] + model_probe_answers
        # 2.3 Evaluate the model's answers
        model_probe_answers_correct = [cot_evaluator.answer_evaluator_cot(probe_question, response, probe["answer"]) for response in model_probe_answers]
        knowledge_confidence = knowledge_confidence_evaluator(model_probe_answers_correct)
        # 2.4 Update the model probe answers and knowledge confidence
        model_probe_answers_list.append(model_probe_answers)
        model_probe_answers_correct_list.append(model_probe_answers_correct)
        model_probe_knowledge_confidence.append(knowledge_confidence)
        
    # 3. Update the chain with the model's answers
    chain["model_answers"] = model_answers
    chain["model_answers_correct"] = model_answers_correct
    chain["model_probe_answers"] = model_probe_answers_list
    chain["model_probe_answers_correct"] = model_probe_answers_correct_list
    chain["model_probe_knowledge_confidence"] = model_probe_knowledge_confidence
    
    return chain, chat_response_generator.get_usage()


if __name__ == "__main__":
    import pickle
    from tqdm import tqdm
    # Load the chains from a pickle file
    with open('data/grow/chains_processed.pkl', 'rb') as f:
        chains = pickle.load(f)
        
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
        
    # Process all chains
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    processed_data = []
    
    # Use tqdm to show progress
    with tqdm(total=len(chains), desc="Processing chains") as pbar:
        for chain in chains:
            # Process each chain
            processed_chain, usage = probe_chain(chain, args)
            # Update the total token counts
            translated_model_name = translate_model_name(args.model_name)
            prompt_tokens += usage[translated_model_name]["prompt_tokens"]
            completion_tokens += usage[translated_model_name]["completion_tokens"]
            total_tokens += usage[translated_model_name]["total_tokens"]
            # Append the processed chain to the list
            processed_data.append(processed_chain)
            # Update the progress bar with the number of tokens used
            pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            pbar.update(1)

    # Save the processed data to a new pickle file
    with open(f'data/grow/chains_{args.model_name}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
