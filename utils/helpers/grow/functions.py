import random
import requests
import calendar
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Tuple, Optional

# Wikidata API endpoint
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

ALLOWED_PROPS = {
    "P39", # position held
    "P36", "P35", "P6", "P20", "P26", "P140", "P1412",
    "P19", "P69", "P40", "P27", "P175", "P108", "P112", "P50",
    "P170", "P407", "P37", "P740", "P495", "P106", "P136", "P364",
    "P937", "P800", "P641", "P413", "P286", "P159", "P178", "P488",
    "P169", "P449", "P176", "P1037"
    # Add more properties relevant to potential changes if needed
}

def subject_should_be(prop: str) -> Optional[str]:
    """
    Returns what the subject of a property should be.
    """
    if prop in {"P36", "P35"}:
        return "country"
    elif prop in {"P6"}:
        return "city"
    elif prop in {"P159"}:
        return "organization"
    else:
        return None
    
def check_type(entity_id: str, expected_type: str) -> bool:
    """
    Check if the entity is of the expected type.
    
    Args:
        entity_id (str): The
        expected_type (str): The expected type ('city', 'country', 'organization', 'person').
        wbi (WikibaseIntegrator): The WikibaseIntegrator instance.
        
    Returns:
        bool: True if the entity is of the expected type, False otherwise.
    """
    if expected_type == "city":
        query = f"""
        ASK WHERE {{
            wd:{entity_id} wdt:P31/wdt:P279* wd:Q515.
        }}
        """
    elif expected_type == "country":
        query = f"""
        ASK WHERE {{
            wd:{entity_id} wdt:P31/wdt:P279* wd:Q6256.
        }}
        """
    elif expected_type == "organization":
        query = f"""
        ASK WHERE {{
            wd:{entity_id} wdt:P31/wdt:P279* wd:Q43229.
        }}
        """
    elif expected_type == "person":
        query = f"""
        ASK WHERE {{
            wd:{entity_id} wdt:P31/wdt:P279* wd:Q5.
        }}
        """
    else:
        raise ValueError(f"Unknown expected type: {expected_type}")

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    try:
        assert isinstance(results["boolean"], bool)
        return results["boolean"]
    except:
        return False
    
def object_should_be(prop: str) -> Optional[str]:
    if prop in {"P20", "P19", "P740", "P937", "P159"}:
        return "city"
    elif prop in {"P27", "P495"}:
        return "country"
    elif prop in {"P108", "P176"}:
        return "organization"
    else:
        return None

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

def get_random_qid() -> str:
    """
    Uses the MediaWiki API to pick a random main-namespace page
    and returns its Q‑ID (always starts with 'Q'). Loops until
    a valid Q‑ID is found.
    """
    session = requests.Session()
    URL = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'random',
        'rnnamespace': 0,
        'rnlimit': 1,
        'format': 'json'
    }

    while True:
        resp = session.get(URL, params=params).json()
        title = resp['query']['random'][0]['title']
        # Only accept titles that look like valid Q‑IDs
        if title.startswith("Q") and title[1:].isdigit():
            return title

def build_chain(start_qid: str, max_depth: int = 5, wbi = None, UA = None) -> List[dict]:
    """
    Returns a list of (e_i, p_i, e_{i+1}) up to max_depth,
    never revisiting the same entity.
    """
    chain = []
    visited = {start_qid}
    visited_props = set()
    current = start_qid

    for i in range(max_depth):
        # 1) Fetch the item
        item = wbi.item.get(entity_id=current)
        claims = item.get_json().get('claims', {})

        # 2) Scan for the first outgoing Item link that doesn't revisit
        found = False
        for prop, claim_list in claims.items():
            # Skip if this property is not in the allowed set
            if prop not in ALLOWED_PROPS:
                continue

            # Skip if not a single claim
            if len(claim_list) != 1:
                continue
            
            claim = claim_list[0]
            dv = claim.get('mainsnak', {}).get('datavalue', {})
            val = dv.get('value', {})

            # check that it’s an item link
            if isinstance(val, dict) and 'id' in val:
                nxt = val['id']
                if nxt in visited or prop in visited_props:
                    # would form a cycle
                    continue
                
                current_label = get_wikidata_label(current)
                nxt_label = get_wikidata_label(nxt)
                prop_label = get_wikidata_label(prop)
                
                if current_label == "" or nxt_label == "" or prop_label == "" or len(current_label.split()) > 5 or len(nxt_label.split()) > 5 or len(current_label) > 50 or len(nxt_label) > 50:
                    # skip if any label is empty or too long
                    continue
                
                # If the subject or object of the property is a specific type, check if it matches
                subject_type = subject_should_be(prop)
                object_type = object_should_be(prop)
                if subject_type and not check_type(current, subject_type):
                    continue
                if object_type and not check_type(nxt, object_type):
                    continue

                # record the triple and advance the chain
                chain.append({
                    "triple": (current, prop, nxt),
                    "triple_label": (get_wikidata_label(current), get_wikidata_label(prop), get_wikidata_label(nxt)),
                })
                visited.add(nxt)
                visited_props.add(prop)
                current = nxt
                found = True
                break

        if not found:
            # no further step possible
            break

    return chain

def sample_chain_exact(args, wbi, UA) -> List[Tuple[str, str, str]]:
    """
    Sample a chain with exactly length==depth such that:
    1. No entity is revisited (i.e. no cycles)
    2. The first triple is updated between 2023-12 and 2025-04
    3. The properties are in the allowed set
    4. There is only one claim for each property
    5. The labels for the entities and properties are not empty
    """
    tries = 0
    depth = args.depth
        
    while True:
        tries += 1
        start = get_random_qid()
        if not start:
            continue
        chain = build_chain(start, max_depth=depth, wbi=wbi, UA=UA)
        if len(chain) == depth:
            return chain
        
# Store the provided templates in a dictionary for easy lookup
PROBE_TEMPLATES = {
    'P30': {'question': 'Which continent is [S] located in?', 'cloze': '[S] is located in the continent of ___.', 'attributive': 'the continent where [S] is located'},
    'P36': {'question': 'What is the capital of [S]?', 'cloze': 'The capital of [S] is ___.', 'attributive': 'the capital of [S]'},
    'P35': {'question': 'Who is the current head of state in [S]?', 'cloze': 'The current head of state in [S] is ___.', 'attributive': 'the person who is the current head of state in [S]'},
    'P6': {'question': 'Who is the current head of the [S] government?', 'cloze': 'The current head of the [S] government is ___.', 'attributive': 'the person who is the current head of government of [S]'},
    'P20': {'question': 'Which city did [S] die in?', 'cloze': '[S] died in the city of ___.', 'attributive': 'the city where [S] died'},
    'P26': {'question': 'Who is [S] married to?', 'cloze': '[S] is married to ___.', 'attributive': 'the person who is married to [S]'},
    'P140': {'question': 'Which religion is [S] affiliated with?', 'cloze': '[S] is affiliated with the religion of ___.', 'attributive': 'the religion that [S] is affiliated with'},
    'P1412': {'question': 'What language does [S] speak?', 'cloze': '[S] speaks the language of ___.', 'attributive': 'the language that [S] speaks'},
    'P19': {'question': 'Which city was [S] born in?', 'cloze': '[S] was born in the city of ___.', 'attributive': 'the city where [S] was born'},
    'P69': {'question': 'Which university was [S] educated at?', 'cloze': 'The univerisity where [S] was educated is ___.', 'attributive': 'the university where [S] was educated'},
    'P40': {'question': 'Who is [S]’s child?', 'cloze': '[S]’s child is ___.', 'attributive': 'the child of [S]'},
    'P27': {'question': 'What is the country of citizenship of [S]?', 'cloze': '[S] is a citizen of ___.', 'attributive': 'the country of citizenship of [S]'},
    'P175': {'question': 'Who performed [S]?', 'cloze': '[S] was performed by ___.', 'attributive': 'the performer of [S]'},
    'P108': {'question': 'Which organization is the employer of [S]?', 'cloze': '[S] is employed by the organization ___.', 'attributive': 'the organization that employs [S]'},
    'P112': {'question': 'Who founded [S]?', 'cloze': '[S] was founded by ___.', 'attributive': 'the founder of [S]'},
    'P50': {'question': 'Who is the author of [S]?', 'cloze': 'The author of [S] is ___.', 'attributive': 'the author of [S]'},
    'P170': {'question': 'Who was [S] created by?', 'cloze': '[S] was created by ___.', 'attributive': 'the creator of [S]'},
    'P407': {'question': 'Which language was [S] written in?', 'cloze': '[S] was written in the language of ___.', 'attributive': 'the language that [S] was written in'},
    'P37': {'question': 'What is the official language of [S]?', 'cloze': 'The official language of [S] is ___.', 'attributive': 'the official language of [S]'},
    'P740': {'question': 'Which city was [S] founded?', 'cloze': '[S] was founded in the city of ___.', 'attributive': 'the city where [S] was founded'},
    'P495': {'question': 'Which country was [S] created in?', 'cloze': '[S] was created in the country of ___.', 'attributive': 'the country where [S] was created'},
    'P106': {'question': 'What kind of work does [S] do?', 'cloze': '[S] works in the field of ___.', 'attributive': 'the field of work of [S]'},
    'P136': {'question': 'What type of music does [S] play?', 'cloze': 'The type of music that [S] plays is ___.', 'attributive': 'the type of music that [S] plays'},
    'P364': {'question': 'What is the original language of [S]?', 'cloze': 'The original language of [S] is ___.', 'attributive': 'the original language of [S]'},
    'P937': {'question': 'Which city did [S] work in?', 'cloze': '[S] worked in the city of ___.', 'attributive': 'the city where [S] worked'},
    'P800': {'question': 'What is [S] famous for?', 'cloze': '[S] is famous for ___.', 'attributive': 'the thing that [S] is famous for'},
    'P641': {'question': 'Which sport is [S] associated with?', 'cloze': '[S] is associated with the sport of ___.', 'attributive': 'the sport that [S] is associated with'},
    'P413': {'question': 'What position does [S] play?', 'cloze': '[S] plays the position of ___.', 'attributive': 'the position that [S] plays'},
    'P286': {'question': 'Who is the head coach of [S]?', 'cloze': 'The head coach of [S] is ___.', 'attributive': 'the head coach of [S]'},
    'P159': {'question': 'Which city is the headquarter of [S] located in?', 'cloze': 'The headquarters of [S] is located in the city of ___.', 'attributive': 'the city where the headquarters of [S] is located'},
    'P178': {'question': 'Who is the developer of [S]?', 'cloze': '[S] was developed by ___.', 'attributive': 'the developer of [S]'},
    'P488': {'question': 'Who is the chairperson of [S]?', 'cloze': 'The chairperson of [S] is ___.', 'attributive': 'the chairperson of [S]'},
    'P169': {'question': 'Who is the chief executive officer of [S]?', 'cloze': 'The chief executive officer of [S] is ___.', 'attributive': 'the chief executive officer of [S]'},
    'P449': {'question': 'Who is the original broadcaster of [S]?', 'cloze': 'The original broadcaster of [S] is ___.', 'attributive': 'the original broadcaster of [S]'},
    'P176': {'question': 'Which company is [S] produced by?', 'cloze': 'The company that produced [S] is ___.', 'attributive': 'the company that produced [S]'},
    'P1037': {'question': 'Who is the director of [S]?', 'cloze': 'The director of [S] is ___.', 'attributive': 'the director of [S]'},
    'P39': {'question': 'What position is held by [S]?', 'cloze': 'The position held by [S] is ___.', 'attributive': 'the position held by [S]'}
}


# Function to format the knowledge string for the 'grow' task
def format_grow_knowledge(triple_item):
    """
    Formats a knowledge string for the 'grow' task using a triple and templates.
    Args:
        triple_item (dict): A dictionary like {'triple': ('Q_S', 'P_ID', 'Q_O'), 'triple_label': ('S_label', 'P_label', 'O_label')}.
    Returns:
        str: The formatted knowledge string (cloze statement).
    """
    s_label, p_label, o_label = triple_item['triple_label']
    p_id = triple_item['triple'][1] # Property ID
    assert p_id in ALLOWED_PROPS, f"Property ID {p_id} is not allowed."

    cloze_template = PROBE_TEMPLATES[p_id]['cloze']
    # Replace subject placeholder [S] and object placeholder ___
    knowledge_str = cloze_template.replace('[S]', s_label).replace('___', o_label)
    return knowledge_str


# Function to simulate the generation of multi-hop questions
def generate_multihop_questions_via_api(prompt_system, prompt_user, args, chat_response_generator):
    """
    Simulates a call to an LLM to generate multi-hop questions.

    Args:
        prompt (str): The formatted prompt for the LLM.

    Returns:
        str: The generated multi-hop question.
    """
    chat_response_generator.chat_history = [
        ("system", prompt_system),
    ]
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

# Function to process a single chain of triples
def process_chain(chain, args, chat_response_generator):
    """
    Processes a single chain of triples to generate the desired output format.

    Args:
        chain (list): A list of triple dictionaries for a single chain.
        args (Namespace): Command line arguments.
        chat_response_generator (ChatResponseGenerator): Instance of the ChatResponseGenerator class.

    Returns:
        dict: 
            A dictionary containing triples, probe questions, multi-hop questions,
            and multi-hop answer for the chain.
    """
    if not chain:
        return {
            "triples": [],
            "probe_questions": [],
            "multihop_question": "",
            "multihop_answer": ""
        }

    processed_triples = []
    probe_questions = []

    # Generate probe questions for each triple
    multihop_question = chain[0]['triple_label'][0]
    for i, triple_dict in enumerate(chain):
        s_id, p_id, o_id = triple_dict['triple']
        s_label, p_label, o_label = triple_dict['triple_label']

        # Add the original triple structure to the output
        processed_triples.append({
            "triple": (s_id, p_id, o_id),
            "triple_label": (s_label, p_label, o_label)
        })

        # Look up templates for the current property ID
        templates = PROBE_TEMPLATES.get(p_id)
        question = templates['question'].replace("[S]", s_label)
        if i < len(chain) - 1:
            multihop_question = templates['attributive'].replace("[S]", multihop_question)
        else:
            multihop_question = templates['question'].replace("[S]", multihop_question)

        # Format the answer string including the object ID
        answer = o_label

        probe_questions.append({
            "question": question,
            "answer": answer,
            "knowledge": format_grow_knowledge(triple_dict),
        })

    # # Generate multi-hop questions (using the simulated API call)
    # multihop_prompt_system, multihop_prompt_user = format_chain_for_multihop_prompt(chain)
    # multihop_questions, usage = generate_multihop_questions_via_api(multihop_prompt_system, multihop_prompt_user, args) # Simulates API call

    # Determine multi-hop answer
    tail_entity_label = chain[-1]['triple_label'][2]
    multihop_answer = tail_entity_label

    return {
        "triples": processed_triples,
        "probe_questions": probe_questions,
        "multihop_question": multihop_question,
        "multihop_answer": multihop_answer
    }, {args.model_name: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    
def make_api_request(params):
    """Makes a request to the Wikidata API and returns the JSON response."""
    try:
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers={'User-Agent': 'CoolTool/0.0 (https://example.org/cool-tool/; cool-tool@example.org) Generic Bot'})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return None
    except ValueError as e:
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
        return []

    entity_data = data["entities"][entity_qid]
    if "claims" not in entity_data or "P31" not in entity_data["claims"]:
        return []

    instance_of_qids = []
    for claim in entity_data["claims"]["P31"]:
        if (claim.get("mainsnak", {}).get("snaktype") == "value" and
            claim["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"):
            instance_of_qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])

    if not instance_of_qids:
        return []

    return instance_of_qids
    
def find_similar_entity(input_qid, max_results_to_consider=50):
    """
    Finds another Wikidata entity that shares the same 'instance of' (P31) property.
    May not used...
    Args:
        input_qid (str): The QID of the entity to find a similar one for (e.g., "Q142" for France).
        max_results_to_consider (int): The maximum number of search results to fetch and consider.

    Returns:
        str: The QID of a similar entity, or None if none could be found or an error occurred.
    """

    # 1. Get the 'instance of' QID(s) for the input entity
    type_qids = get_instance_of_qids(input_qid)
    if not type_qids:
        return None

    # For simplicity, we'll primarily use the first type found.
    # A more complex implementation could try multiple types if the first yields no results.
    target_type_qid = type_qids[0]

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
        return None

    search_results = search_data["query"]["search"]

    if not search_results:
        return None

    # 3. Filter out the original QID and select a random one
    potential_matches = [result["title"] for result in search_results if result["title"] != input_qid]

    if not potential_matches:
        # Optional: Could retry search with a different offset or larger limit here
        return None

    # 4. Return a random match
    similar_qid = random.choice(potential_matches)
    return similar_qid