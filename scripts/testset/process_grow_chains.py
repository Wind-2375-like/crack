import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
from utils.generator.chat_response_generator import ChatResponseGenerator


# Store the provided templates in a dictionary for easy lookup
PROBE_TEMPLATES = {
    'P30': {'question': 'Which continent is [S] located in?', 'cloze': '[S] is located in the continent of ___'},
    'P36': {'question': 'What is the capital of [S]?', 'cloze': 'The capital of [S] is ___'},
    'P35': {'question': 'What is the name of the current head of state in [S]?', 'cloze': 'The name of the current head of state in [S] is ___'},
    'P6': {'question': 'What is the name of the current head of the [S] government?', 'cloze': 'The name of the current head of the [S] government is ___'},
    'P20': {'question': 'Which city did [S] die in?', 'cloze': '[S] died in the city of ___'},
    'P26': {'question': 'Who is [S] married to?', 'cloze': '[S] is married to ___'},
    'P140': {'question': 'Which religion is [S] affiliated with?', 'cloze': '[S] is affiliated with the religion of ___'},
    'P1412': {'question': 'What language does [S] speak?', 'cloze': '[S] speaks the language of ___'},
    'P19': {'question': 'Which city was [S] born in?', 'cloze': '[S] was born in the city of ___'},
    'P69': {'question': 'Which university was [S] educated at?', 'cloze': 'The univeristy where [S] was educated is ___'},
    'P40': {'question': 'Who is [S]’s child?', 'cloze': '[S]’s child is ___'},
    'P27': {'question': 'What is the country of citizenship of [S]?', 'cloze': '[S] is a citizen of ___'},
    'P175': {'question': 'Who performed [S]?', 'cloze': '[S] was performed by ___'},
    'P108': {'question': 'Who is the employer of [S]?', 'cloze': '[S] is employed by ___'},
    'P112': {'question': 'Who founded [S]?', 'cloze': '[S] was founded by ___'},
    'P50': {'question': 'Who is the author of [S]?', 'cloze': 'The author of [S] is ___'},
    'P170': {'question': 'Who was [S] created by?', 'cloze': '[S] was created by ___'},
    'P407': {'question': 'Which language was [S] written in?', 'cloze': '[S] was written in the language of ___'},
    'P37': {'question': 'What is the official language of [S]?', 'cloze': 'The official language of [S] is ___'},
    'P740': {'question': 'Where was [S] founded?', 'cloze': '[S] was founded in the city of ___'},
    'P495': {'question': 'Which country was [S] created in?', 'cloze': '[S] was created in the country of ___'},
    'P106': {'question': 'What kind of work does [S] do?', 'cloze': '[S] works in the field of ___'},
    'P136': {'question': 'What type of music does [S] play?', 'cloze': 'The type of music that [S] plays is ___'},
    'P364': {'question': 'What is the original language of [S]?', 'cloze': 'The original language of [S] is ___'},
    'P937': {'question': 'Which city did [S] work in?', 'cloze': '[S] worked in the city of ___'},
    'P800': {'question': 'What is [S] famous for?', 'cloze': '[S] is famous for ___'},
    'P641': {'question': 'Which sport is [S] associated with?', 'cloze': '[S] is associated with the sport of ___'},
    'P413': {'question': 'What position does [S] play?', 'cloze': '[S] plays the position of ___'},
    'P286': {'question': 'Who is the head coach of [S]?', 'cloze': 'The head coach of [S] is ___'},
    'P159': {'question': 'Which city is the headquarter of [S] located in?', 'cloze': 'The headquarters of [S] is located in the city of ___'},
    'P178': {'question': 'Who is the developer of [S]?', 'cloze': '[S] was developed by ___'},
    'P488': {'question': 'Who is the chairperson of [S]?', 'cloze': 'The chairperson of [S] is ___'},
    'P169': {'question': 'Who is the chief executive officer of [S]?', 'cloze': 'The chief executive officer of [S] is ___'},
    'P449': {'question': 'Who is the original broadcaster of [S]?', 'cloze': 'The origianl broadcaster of [S] is ___'},
    'P176': {'question': 'Which company is [S] produced by?', 'cloze': 'The company that produced [S] is ___'},
    'P1037': {'question': 'Who is the director of [S]?', 'cloze': 'The director of [S] is ___'},
    'P1308': {'question': 'Who is the [S]?', 'cloze': 'The [S] is ___'}
}


# Function to simulate the generation of multi-hop questions
def generate_multihop_questions_via_api(prompt_system, prompt_user, args):
    """
    Simulates a call to an LLM to generate multi-hop questions.

    Args:
        prompt (str): The formatted prompt for the LLM.

    Returns:
        list: A list of 3 generated question strings (dummy implementation).
    """
    chat_response_generator = ChatResponseGenerator(
        model_name=args.model_name,
        chat_history=[
            ("system", prompt_system),
        ],
        api_key=args.openai_api_key,
        local=False
    )
    response = chat_response_generator.generate_response(
        query=prompt_user,
        n=1,
        temperature=0.0,
        top_p=1.0,
    )[0]
    # Calculate the number of tokens used
    usage = chat_response_generator.get_usage()
    
    questions = response.replace("Assistant:", "").split("\n")
    questions = [q.strip() for q in questions if q.strip()]
    return questions[:3], usage


# Function to format the chain into the required prompt format
def format_chain_for_multihop_prompt(chain):
    """
    Formats a chain of triples into the specified prompt format for the LLM.

    Args:
        chain (list): A list of triple dictionaries for a single chain.

    Returns:
        str: The formatted prompt string.
    """
    if not chain:
        return "Cannot generate prompt for empty chain."

    # Extract labels and structure the chain representation
    head_label = chain[0]['triple_label'][0]
    prop_labels = [triple['triple_label'][1] for triple in chain]

    # Construct the final prompt string
    prompt_system = "System:\nYou are a powerful multi-hop question generator. Users will provide a question,\nand you will rephrase it to make it more natural, which has the exactly same meaning as the user-provided question. Return 3 rephrased questions, each in one separate line.\n\n[Here are 3 demonstrations]\n\nUser:\nRephrase the question \"What is Xbox Live\'s developer\'s chief executive officer\'s place of birth\'s continent?\" to three questions in natural English. These questions should have exactly the same meaning as the original question.\n\nAssistant:\nWhich continent is home to the birthplace of the CEO of Xbox Live developer?\nWhere was the CEO of the developer of Xbox Live born in which continent?\nIn what continent was the CEO of Xbox Live’s developer born?\n\nUser:\nRephrase the question \"What is Winnie the Pooh\'s creator\'s child\'s country of citizenship\'s official language?\" to three questions in natural English. These questions should have exactly the same meaning as the original question.\n\nAssistant:\nWhat is the official language of the country where the child of Winnie the Pooh’s creator holds citizenship?\nWhich language is officially spoken in the country where the child of the creator of Winnie the Pooh is a citizen?\nWhat is the officiated language of the country where the child of Winnie the Pooh’s creator is a citizen of?\n\nUser:\nRephrase the question \"What is watchOS\'s developer\'s chief executive officer\'s country of citizenship\'s capital?\" to three questions in natural English. These questions should have exactly the same meaning as the original question.\n\nAssistant:\nWhat is the capital of the country where the CEO of the developer of watchOS holds citizenship?\nIn which city does the CEO of the company that developed watchOS hold citizenship?\nWhich city is the capital of the home country of the CEO of the developer of watchOS?"

    prompt_user = f"User:\nRephrase the question \"What is {head_label}'s " + "'s ".join(prop_labels) + "?\" to three questions in natural English, each in one line. These questions should have exactly the same meaning as the original question.\n\nAssistant:\n"

    return prompt_system, prompt_user


# Get aliases from Wikidata using SPARQL
def get_alias_from_wikidata(entity_id, args):
    """
    Fetches the aliases of a given entity from Wikidata using SPARQL.
    Args:
        entity_id (str): The Wikidata entity ID (e.g., Q42).
        ua (str): User agent string for the SPARQL request.
    Returns:
        list: A list of aliases for the entity.
    """
    
    query = f"""
    SELECT ?altLabel
    {{
    VALUES (?wd) {{(wd:{entity_id})}}
    ?wd skos:altLabel ?altLabel .
    FILTER (lang(?altLabel) = "en")
    }}
    """
    try:
        endpoint_url = "https://query.wikidata.org/sparql"

        user_agent = args.ua
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        aliases = []
        for result in results["results"]["bindings"]:
            if "altLabel" in result:
                aliases.append(result["altLabel"]["value"])
        return aliases
    except Exception as e:
        return []


# Function to process a single chain of triples
def process_chain(chain, args):
    """
    Processes a single chain of triples to generate the desired output format.

    Args:
        chain (list): A list of triple dictionaries for a single chain.
        ua (str): User agent string for the SPARQL request.

    Returns:
        dict: 
            A dictionary containing triples, probe questions, multi-hop questions,
            and multi-hop answer for the chain.
    """
    if not chain:
        return {
            "triples": [],
            "probe_questions": [],
            "multihop_questions": [],
            "multihop_answer": ""
        }

    processed_triples = []
    probe_questions = []

    # Generate probe questions for each triple
    for triple_dict in chain:
        s_id, p_id, o_id = triple_dict['triple']
        s_label, p_label, o_label = triple_dict['triple_label']

        # Add the original triple structure to the output
        processed_triples.append({
            "triple": (s_id, p_id, o_id),
            "triple_label": (s_label, p_label, o_label)
        })

        # Look up templates for the current property ID
        templates = PROBE_TEMPLATES.get(p_id)
        if templates:
            question = templates['question'].replace("[S]", s_label)
            cloze = templates['cloze'].replace("[S]", s_label)
        else:
            # Handle cases where the property ID is not in our templates
            question = f"Generated probe: What is the {p_label} ({p_id}) of {s_label}?"
            cloze = f"{s_label}'s {p_label} ({p_id}) is ___."

        # Format the answer string including the object ID
        answer = o_label

        probe_questions.append({
            "question": question,
            "cloze": cloze,
            "answer": answer,
        })

    # Generate multi-hop questions (using the simulated API call)
    multihop_prompt_system, multihop_prompt_user = format_chain_for_multihop_prompt(chain)
    multihop_questions, usage = generate_multihop_questions_via_api(multihop_prompt_system, multihop_prompt_user, args) # Simulates API call

    # Determine multi-hop answer
    tail_entity_label = chain[-1]['triple_label'][2]
    multihop_answer = tail_entity_label

    return {
        "triples": processed_triples,
        "probe_questions": probe_questions,
        "multihop_questions": multihop_questions,
        "multihop_answer": multihop_answer
    }, usage
    
    
def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="Model name for the API")
    return parser.parse_args()


if __name__ == "__main__":
    import pickle
    from tqdm import tqdm
    # Load the chains from a pickle file
    with open('data/grow/chains_raw.pkl', 'rb') as f:
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
            processed_chain, usage = process_chain(chain, args)
            # Update the total token counts
            prompt_tokens += usage[args.model_name]["prompt_tokens"]
            completion_tokens += usage[args.model_name]["completion_tokens"]
            total_tokens += usage[args.model_name]["total_tokens"]
            # Append the processed chain to the list
            processed_data.append(processed_chain)
            # Update the progress bar with the number of tokens used
            pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            pbar.update(1)

    # Save the processed data to a new pickle file
    with open('data/grow/chains_processed.pkl', 'wb') as f:
        pickle.dump(processed_data, f)