import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import argparse
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers.grow import translate_model_name
from utils.evaluator import answer_evaluator_cot, answer_evaluator, knowledge_confidence_evaluator

import requests
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wikidata API endpoint
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

# # 1. Get the model answers for each multihop question
# chat_response_generator.update_chat_history([
#     ("system", "You are given a question. To answer the question, you should think step by step. Use line breaks between steps, but do not use line breaks within each step. You should number each step. The final answer to the question should start with \\\"The answer is ...\\\", and should be placed at the final step.\n\n[Here is one demonstration]\n\nUser:\nWhat is the capital of the country where Plainfield Town Hall is located?\n\nAssistant:\n1. Plainfield Town Hall is one of two town halls in Plainfield, New Hampshire.\n2. New Hampshire is a state in the New England region of the Northeastern United States.\n3. Thus, Plainfield Town Hall is located in the country of the United States of America. \n4. The capital of United States is Washington, D.C.\n5. The answer is Washington, D.C."),
# ])

# --- Helper Functions ---

def get_wikidata_label(entity_id: str, lang: str = "en") -> str:
    """
    Fetches the label for a Wikidata entity (e.g. 'Q42' or 'P31') in the specified language.

    :param entity_id: The Wikidata ID (e.g. 'Q42' for Douglas Adams, 'P31' for instance of).
    :param lang:       The language code for the label (default 'en').
    :return:           The label string, or an empty string if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels",
        "languages": lang,
        "format": "json"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Navigate the JSON to extract the label
    entities = data.get("entities", {})
    entity = entities.get(entity_id, {})
    labels = entity.get("labels", {})
    label_info = labels.get(lang, {})

    return label_info.get("value", "")

def make_api_request(params):
    """Makes a request to the Wikidata API and returns the JSON response."""
    try:
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers={'User-Agent': 'CoolTool/0.0 (https://example.org/cool-tool/; cool-tool@example.org) Generic Bot'})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        # logging.error(f"API request failed: {e}")
        return None
    except ValueError as e:
        # logging.error(f"Failed to decode JSON response: {e}")
        return None

def get_instance_of_qids(entity_qid):
    """
    Fetches the QID(s) of the 'instance of' (P31) property for a given entity QID.

    Args:
        entity_qid (str): The QID of the entity (e.g., "Q30" for USA).

    Returns:
        list: A list of QIDs representing the 'instance of' values, or an empty list if not found or error occurs.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_qid,
        "format": "json",
        "props": "claims",
        "languages": "en" # Optional: helps debugging but not essential for logic
    }

    data = make_api_request(params)

    if not data or "entities" not in data or entity_qid not in data["entities"]:
        # logging.warning(f"Could not retrieve entity data for {entity_qid}")
        return []

    entity_data = data["entities"][entity_qid]
    if "claims" not in entity_data or "P31" not in entity_data["claims"]:
        # logging.warning(f"Entity {entity_qid} has no 'instance of' (P31) claims.")
        return []

    instance_of_qids = []
    for claim in entity_data["claims"]["P31"]:
        if (claim.get("mainsnak", {}).get("snaktype") == "value" and
            claim["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"):
            instance_of_qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])

    # if not instance_of_qids:
    #     logging.warning(f"Could not extract valid QIDs from P31 claims for {entity_qid}.")

    return instance_of_qids

# --- Main Function ---

def find_similar_entity(input_qid, max_results_to_consider=50):
    """
    Finds another Wikidata entity that shares the same 'instance of' (P31) property.

    Args:
        input_qid (str): The QID of the entity to find a similar one for (e.g., "Q142" for France).
        max_results_to_consider (int): The maximum number of search results to fetch and consider.

    Returns:
        str: The QID of a similar entity, or None if none could be found or an error occurred.
    """
    # logging.info(f"Attempting to find an entity similar to {input_qid}")

    # 1. Get the 'instance of' QID(s) for the input entity
    type_qids = get_instance_of_qids(input_qid)
    if not type_qids:
        # logging.error(f"Could not determine 'instance of' type for {input_qid}.")
        return None

    # For simplicity, we'll primarily use the first type found.
    # A more complex implementation could try multiple types if the first yields no results.
    target_type_qid = type_qids[0]
    # logging.info(f"Entity {input_qid} is an instance of {target_type_qid} (and possibly others: {type_qids})")

    # 2. Search for other entities with the same 'instance of' property (P31 = target_type_qid)
    # We use the 'haswbstatement' search feature which is efficient for this.
    # Randomize offset slightly to potentially get different results on subsequent calls
    random_offset = random.randint(0, 200) # Adjust range as needed

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": f"haswbstatement:P31={target_type_qid}",
        "srnamespace": 0,  # Search only in the main (Item) namespace
        "format": "json",
        "srlimit": max_results_to_consider,
        "sroffset": random_offset,
        "srinfo": "", # Don't need extra info like snippet
        "srprop": ""  # Don't need extra properties like timestamp
    }

    search_data = make_api_request(search_params)

    if not search_data or "query" not in search_data or "search" not in search_data["query"]:
        # logging.error(f"Search query failed for type {target_type_qid}.")
        return None

    search_results = search_data["query"]["search"]

    if not search_results:
        # logging.warning(f"No other entities found with P31={target_type_qid} in the sampled search results.")
        return None

    # 3. Filter out the original QID and select a random one
    potential_matches = [result["title"] for result in search_results if result["title"] != input_qid]

    if not potential_matches:
        # logging.warning(f"Search results only contained the input QID {input_qid} or were empty after filtering.")
        # Optional: Could retry search with a different offset or larger limit here
        return None

    # 4. Return a random match
    similar_qid = random.choice(potential_matches)
    # logging.info(f"Found similar entity for {input_qid}: {similar_qid} (also instance of {target_type_qid})")
    return similar_qid


def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b-turbo", help="Model name for the API")
    parser.add_argument('--factuality', type=str, default="f", help="Factuality of injected knowledge. Can be 'f' or 'f+nf'.")
    return parser.parse_args()


def experiment(chain, args, factuality="f"):
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
                - "answers": A list of possible answers for the question.
            - "multihop_questions": A list of questions (same meaning) for the chain.
            - "multihop_answers": A list of possible answers for the rephrased questions.
            - "model_answers": A list of answers for the chain.
            - "model_answers_correct": A list of boolean values indicating whether the model's answer is correct.
            - "model_probe_answers": A list of lists of answers for each probe question.
            - "model_probe_answers_correct": A list of lists of boolean values indicating whether the model's answer is correct.
            - "model_probe_knowledge_confidence": A list of str indicating the model's confidence in its knowledge.
        args (Namespace): Command line arguments.
        factuality (str): The factuality of injected knowledge. It can be "f" for factual or "f+nf" for both factual and non-factual.
    Returns:
        chain (list): The processed chain with the following additional keys:
            - "model_answers_after_injection": A list of answers for the chain.
            - "model_answers_correct_after_injection": A list of boolean values indicating whether the model's answer is correct.
        usage (dict): A dictionary containing the token usage information for the model.
    """
    # Initialize the response generator
    chat_response_generator = ChatResponseGenerator(model_name=translate_model_name(args.model_name), api_key=args.api_key, local=False)
    
    # 1. Get the model answers for each multihop question
    chat_response_generator.update_chat_history([
        ("system", "You are given a question. To answer the question, you should think step by step. Use line breaks between steps, but do not use line breaks within each step. You should number each step. The final answer to the question should start with \"The answer is ...\", and should be placed at the final step. You may also be given a set of facts. Evaluate the provided facts for both correctness and relevance to the question. If you determine the facts are accurate and pertinent, utilize them in your step-by-step reasoning to answer the question. Otherwise disregard them. \n\n[Here are two demonstrations]\n\nUser:\nWhat is the capital of the country where Plainfield Town Hall is located?\n\nAssistant:\n1. Plainfield Town Hall is one of two town halls in Plainfield, New Hampshire.\n2. New Hampshire is a state in the New England region of the Northeastern United States.\n3. Thus, Plainfield Town Hall is located in the country of the United States of America. \n4. The capital of United States is Washington, D.C.\n5. The answer is Washington, D.C.\n\nUser:\nWho is married to the British Prime Minister? The name of the current head of the British government is Keir Starmer. Keir Starmer\'s spouse is Ivanka Trump.\n\nAssistant:\n1. As of my knowledge cutoff in December 2023, the current Prime Minister of the United Kingdom is Rishi Sunak, and Keir Starmer\'s spouse is Victoria Starmer.\n2. The user provided that Keir Starmer is the current head of government. As of my knowledge, he is the leader of the Opposition and the leader of the Labour Party in the UK. So it is possible that he is the Prime Minister now.\n3. The user also provided that Keir Starmer\'s spouse is Ivanka Trump. This is unlikely possible, since Ivanka Trump is an American businesswoman and the daughter of Donald Trump. They have different nationalities, professions, and public profiles.\n4. Therefore, the current British Prime Minister is Keir Starmer, and Victoria Starmer is married to British Prime Minister.\n5. The answer is Victoria Starmer."),
    ])
    
    model_answers_after_injection = []
    model_answers_correct_after_injection = []
    for question in chain["multihop_questions"]:
        responses = chat_response_generator.generate_response(
            inject_knowledge_to_input(question, chain, factuality),
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            n=1,
        )
        responses = [response.replace("Assistant:", "").strip() for response in responses]
        responses_correct = [answer_evaluator_cot(response, chain["multihop_answers"]) for response in responses]
        model_answers_after_injection.extend(responses)
        model_answers_correct_after_injection.extend(responses_correct)
        
    # 3. Update the chain with the model's answers
    chain["model_answers_after_injection"] = model_answers_after_injection
    chain["model_answers_correct_after_injection"] = model_answers_correct_after_injection
    
    return chain, chat_response_generator.get_usage()


def inject_knowledge_to_input(question, chain, factuality="f"):
    """
    Injects knowledge into the input for the model.
    Args:
        question (str): The input question to be injected with knowledge.
        chain (list): A list of dictionaries representing the chain. Same as in the experiment function.
        factuality (str): The factuality of injected knowledge. It can be "f" for factual or "f+nf" for both factual and non-factual.
    Returns:
        str: The input string with injected knowledge.
    """
    if factuality == "f":
        # 1. Get the knowledge from the chain with the "Unknown" confidence
        unknown_knowledge = [confidence == "Unknown" for confidence in chain["model_probe_knowledge_confidence"]]
        injected_knowledge = []
        for i, probe in enumerate(chain["probe_questions"]):
            if unknown_knowledge[i]:
                injected_knowledge.append(probe["cloze"].replace("___", chain["triples"][i]["triple_label"][2]))
        # 2. Inject the knowledge into the input
        return f"User:\n{question} " + ". ".join(injected_knowledge) + ".\nAssistant:\n"
    elif factuality == "f+nf":
        # 1. Get the knowledge from the chain with the "Unknown" confidence
        unknown_knowledge = [confidence == "Unknown" for confidence in chain["model_probe_knowledge_confidence"]]
        injected_knowledge = []
        for i, probe in enumerate(chain["probe_questions"]):
            if unknown_knowledge[i]:
                injected_knowledge.append(probe["cloze"].replace("___", chain["triples"][i]["triple_label"][2]))
            else:
                # Synthesize non-factual knowledge
                try:
                    similar_entity = get_wikidata_label(find_similar_entity(chain["triples"][i]["triple"][2]))
                    if similar_entity:
                        injected_knowledge.append(probe["cloze"].replace("___", similar_entity))
                except Exception as e:
                    continue
                
        # 2. Inject the knowledge into the input
        return f"User:\n{question} " + ". ".join(injected_knowledge) + ".\nAssistant:\n"
    else:
        raise ValueError("Invalid factuality value. It should be either 'f' or 'f+nf'.")


if __name__ == "__main__":
    import pickle
    from tqdm import tqdm
        
    # Parse command line arguments
    args = parse_args()
    
    # Load the chains from a pickle file
    with open(f'data/grow/chains_{args.model_name}.pkl', 'rb') as f:
        chains = pickle.load(f)
    
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
            processed_chain, usage = experiment(chain, args, factuality=args.factuality)
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
    with open(f'data/grow/chains_{args.model_name}_exp_{args.factuality}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
        
    # Evaluation
    original_correctness = []
    for i, chain in enumerate(chains):
        original_correctness.extend(chain["model_answers_correct"])
    after_injection_correctness = []
    for i, chain in enumerate(processed_data):
        after_injection_correctness.extend(chain["model_answers_correct_after_injection"])
    print(f"Original correctness: {sum(original_correctness) / len(original_correctness)}")
    print(f"After injection correctness: {sum(after_injection_correctness) / len(after_injection_correctness)}")