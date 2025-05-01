import requests
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wikidata API endpoint
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

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
        logging.error(f"API request failed: {e}")
        return None
    except ValueError as e:
        logging.error(f"Failed to decode JSON response: {e}")
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
        logging.warning(f"Could not retrieve entity data for {entity_qid}")
        return []

    entity_data = data["entities"][entity_qid]
    if "claims" not in entity_data or "P31" not in entity_data["claims"]:
        logging.warning(f"Entity {entity_qid} has no 'instance of' (P31) claims.")
        return []

    instance_of_qids = []
    for claim in entity_data["claims"]["P31"]:
        if (claim.get("mainsnak", {}).get("snaktype") == "value" and
            claim["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"):
            instance_of_qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])

    if not instance_of_qids:
        logging.warning(f"Could not extract valid QIDs from P31 claims for {entity_qid}.")

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
    logging.info(f"Attempting to find an entity similar to {input_qid}")

    # 1. Get the 'instance of' QID(s) for the input entity
    type_qids = get_instance_of_qids(input_qid)
    if not type_qids:
        logging.error(f"Could not determine 'instance of' type for {input_qid}.")
        return None

    # For simplicity, we'll primarily use the first type found.
    # A more complex implementation could try multiple types if the first yields no results.
    target_type_qid = type_qids[0]
    logging.info(f"Entity {input_qid} is an instance of {target_type_qid} (and possibly others: {type_qids})")

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
        logging.error(f"Search query failed for type {target_type_qid}.")
        return None

    search_results = search_data["query"]["search"]

    if not search_results:
        logging.warning(f"No other entities found with P31={target_type_qid} in the sampled search results.")
        return None

    # 3. Filter out the original QID and select a random one
    potential_matches = [result["title"] for result in search_results if result["title"] != input_qid]

    if not potential_matches:
        logging.warning(f"Search results only contained the input QID {input_qid} or were empty after filtering.")
        # Optional: Could retry search with a different offset or larger limit here
        return None

    # 4. Return a random match
    similar_qid = random.choice(potential_matches)
    logging.info(f"Found similar entity for {input_qid}: {similar_qid} (also instance of {target_type_qid})")
    return similar_qid

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Find another country (France is Q142, an instance of country Q6256)
    country_qid = "Q142" # France
    similar_country = find_similar_entity(country_qid)
    if similar_country:
        print(f"Input: {country_qid} (France)")
        print(f"Found similar entity (another country): {similar_country} ({get_wikidata_label(similar_country)})")
    else:
        print(f"Could not find a similar entity for {country_qid}")

    print("-" * 20)

    # Example 2: Find another university (Harvard University is Q13371, an instance of university Q3918)
    university_qid = "Q13371" # Harvard University
    similar_university = find_similar_entity(university_qid)
    if similar_university:
        print(f"Input: {university_qid} (Harvard University)")
        print(f"Found similar entity (another university): {similar_university} ({get_wikidata_label(similar_university)})")
    else:
        print(f"Could not find a similar entity for {university_qid}")

    print("-" * 20)

    # Example 3: Find another human (Marie Curie is Q7186, an instance of human Q5)
    human_qid = "Q7186" # Marie Curie
    similar_human = find_similar_entity(human_qid)
    if similar_human:
        print(f"Input: {human_qid} (Marie Curie)")
        print(f"Found similar entity (another human): {similar_human} ({get_wikidata_label(similar_human)})")
    else:
        print(f"Could not find a similar entity for {human_qid}")

    print("-" * 20)

    # Example 4: Entity with no P31 (or an error case)
    # Let's use a property ID like P31 itself, which isn't an 'Item' and won't have P31
    invalid_qid = "P31"
    similar_invalid = find_similar_entity(invalid_qid)
    if similar_invalid:
         print(f"Input: {invalid_qid}")
         print(f"Found similar entity: {similar_invalid}") # Should not happen
    else:
        print(f"Correctly could not find a similar entity for invalid input {invalid_qid}")