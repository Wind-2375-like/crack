import os
import re
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import pickle
import argparse
from tqdm import tqdm
from utils.generator.chat_response_generator import ChatResponseGenerator
from utils.helpers import translate_model_name
from collections import Counter


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
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-4.1-mini", help="Model name for the evaluation")
    parser.add_argument('--task_name', type=str, default="grow", help="Task name")
    return parser.parse_args()


PROMPT_TEMPLATES = {
    "grow": """You are given a question, a response, and a ground truth answer. The ground truth answer is an entity, and each response might contain an entity. Your task is to use commonsense knowledge to evaluate whether the response most probably refers the same entity as the ground truth.

If they are equivalent, answer 'Yes' and provide an explanation. Otherwise, answer 'No' and provide an explanation.

Note that if the response does not contain an entity, it should be treated as 'N/A' and not equivalent to the answer.

Examples:

Question:
Who is the current US president?
Response:
Therefore, the answer is Donald Trump.
Answer:
Donald J. Trump
Equivalence:
Yes, Donald Trump is the same person as Donald J. Trump.

Question:
Who is the current US president?
Response:
Therefore, the answer is Donald Trump.
Answer:
Joe Biden
Equivalence:
No, Donald Trump is a different person from Joe Biden. They belong to different political parties.

Question:
Who is Albuquerque’s head of government?
Response:
Based on my cutoff knowledge, Albuquerque’s head of government is Tim Keller.
Answer:
Timothy M. Keller
Equivalence:
Yes, 'Tim' is a common short form of 'Timothy'.

Question:
Who is Albuquerque’s head of government?
Response:
I cannot provide an exact answer based on my cutoff knowledge.
Answer:
Timothy M. Keller
Equivalence:
No, the response fails to provide an entity (N/A), while the answer provides the entity 'Timothy M. Keller'.
""",
    "default": """You are given a question, a response, and a ground truth answer. The ground truth answer is an entity, and each response might contain an entity. Your task is to use commonsense knowledge to evaluate whether the response most probably refers the same entity as the ground truth.

If they are equivalent, answer 'Yes' and provide an explanation. Otherwise, answer 'No' and provide an explanation.

Note that if the response does not contain an entity, it should be treated as 'N/A' and not equivalent to the answer.
""",
"code": """You are given a question, a canonical function from a library for the question, a corresponding docstring, and the model's response. Each response might also contain a function call from a library. Your task is to use basic Python coding knowledge to evaluate whether the model's response is most probably correct.

If the answer is correct, answer 'Yes' and provide an explanation. Otherwise, answer 'No' and provide an explanation.

Note that if the response does not contain a function call, it should be treated as 'N/A' and not correct.

--- Examples 1 ---

Question: 
Given the library pandas, how can we create a DataFrame by explicitly passing the input data (such as an ndarray, Iterable, dict, or DataFrame) using the `data` parameter?

Function: pandas.DataFrame(data)

Docstring: Two-dimensional, size-mutable, potentially heterogeneous tabular data.

Data structure also contains labeled axes (rows and columns).
Arithmetic operations align on both row and column labels. Can be
thought of as a dict-like container for Series objects. The primary
pandas data structure.

Parameters
----------
data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
    Dict can contain Series, arrays, constants, dataclass or list-like objects. If
    data is a dict, column order follows insertion-order. If a dict contains Series
    which have an index defined, it is aligned by its index. This alignment also
    occurs if data is a Series or a DataFrame itself. Alignment is done on
    Series/DataFrame inputs.

    If data is a list of dicts, column order follows insertion-order.

index : Index or array-like
    Index to use for resulting frame. Will default to RangeIndex if
    no indexing information part of input data and no index provided.
columns : Index or array-like
    Column labels to use for resulting frame when data does not have them,
    defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
    will perform column selection instead.
dtype : dtype, default None
    Data type to force. Only a single dtype is allowed. If None, infer.
copy : bool or None, default None
    Copy data from inputs.
    For dict data, the default of None behaves like ``copy=True``.  For DataFrame
    or 2d ndarray input, the default of None behaves like ``copy=False``.
    If data is a dict containing one or more Series (possibly of different dtypes),
    ``copy=False`` will ensure that these inputs are not copied.
...

Response:
```python
pandas.DataFrame(arr)
```

Correct:
Yes, the response contains the same function call as the ground truth function call. The function call `pandas.DataFrame(arr)` is equivalent to the ground truth function call, which creates a DataFrame from the provided data.

--- Example 2 ---

Question: 
... (the same function as above)

Function: pandas.DataFrame(data)

Docstring: ... (the same docstring as above)

Response:
```python
pandas.DataFrame({"id": [0, 1, 2, 3, 4], "val": [100, 200, -2, 34, 45.2]}, dtype=None)
```

Correct:
Yes, the response contains the same function call as the ground truth function call. The function call `pandas.DataFrame({"id": [0, 1, 2, 3, 4], "val": [100, 200, -2, 34, 45.2]}, dtype=None)` is equivalent to the ground truth function call, which creates a DataFrame from the provided data. The `dtype` parameter is optional and defaults to None, so it does not change the equivalence.

--- Example 3 ---

Question: 
... (the same function as above)

Function: pandas.DataFrame(data)

Docstring: ... (the same docstring as above)

Response:
```python
pandas.DataFrame(data, dtype="float")
```

Correct:
No, the response contains a different function call than the ground truth function call. The function call `pandas.DataFrame(data, dtype="float")` specifies a dtype of "float", which is not equivalent to the ground truth function call that does not specify a dtype. The ground truth function call creates a DataFrame from the provided data without any specific dtype.
"""
}

# Internal key for representing the group of answers equivalent to ground truth in merged_frequencies
_GROUND_TRUTH_MERGED_KEY = "##_INTERNAL_GROUND_TRUTH_EQUIVALENT_##"

def evaluate_probe_item(item, args, chat_response_generator):
    """
    Evaluates LLM's probed knowledge.
    Args:
        item (dict): A dictionary containing:
            - "id": index of the question
            - "question": the question text
            - "answer": the answer text (ground truth)
            - "complex_question_id": the id of the original list
            - "probe_answers": a list of answers generated by the model.
        args (Namespace): Command line arguments. (args.task_name)
        chat_response_generator (ChatResponseGenerator): An instance of the ChatResponseGenerator class for evaluation.
    Returns:
        item (dict): The updated item with additional keys:
            - "knowledgable": a boolean indicating if the model is knowledgable.
            - "knowledge_confidence": a float indicating the model's confidence in its knowledge.
            - "eval_probe_results": a dictionary containing the full evaluation results:
                - "probe_answers": a list of answers generated by the model.
                - "probe_answers_correct": a list of explanations indicating if the model's answers are correct.
                - "knowledge_confidence": a float indicating the model's confidence in its knowledge.
                - "majority_answer": the most common answer from the model.
                - "majority_answer_correct": a boolean indicating if the majority answer is correct.
                - "majority_answer_confidence": a float indicating the confidence of the majority answer.
        usage (dict): A dictionary containing the token usage information for the model.
    """

    def _parse_llm_equivalence_response(response_text):
        """
        Parses the LLM's response for equivalence check.
        Returns a tuple: (is_equivalent_bool, explanation_str)
        """
        response_text_stripped = response_text.replace("Equivalence:", "").replace("Correct:", "").strip()
        response_text_lower = response_text_stripped.lower()
        explanation = response_text_stripped

        if response_text_lower.startswith("yes"):
            is_equivalent = True
        else:
            is_equivalent = False
            
        return is_equivalent, explanation

    # Select the appropriate system prompt based on args.task_name
    system_prompt_key = args.task_name if args.task_name in PROMPT_TEMPLATES else "default"
    system_prompt = PROMPT_TEMPLATES[system_prompt_key]
    chat_response_generator.update_chat_history([
        ("system", system_prompt),
    ])

    if args.task_name == "grow":
        probe_answers_from_item = item["probe_answers"]
    elif args.task_name == "code":
        probe_answers_from_item = []
        for i in item["probe_answers"]:
            match = re.search(r"```python\s*([\s\S]*?)\s*```", i)
            if match:
                probe_answers_from_item.append(match.group(1).strip())
            else:
                probe_answers_from_item.append(i.strip())
        
    ground_truth_answer_text = item["answer"]
    question_text = item["question"]
    knowledge_text = item["knowledge"]
    total_votes = len(probe_answers_from_item)

    # Step 1: Group original probe answers by frequency
    original_frequencies = Counter(probe_answers_from_item)

    # Step 2: Perform semantic equivalence check for each unique probe answer
    # against the ground truth, and prepare explanations for all probe answers.
    unique_answer_eval_cache = {}  # Cache: {unique_probe_answer: (is_equivalent_bool, explanation_str)}
    probe_answers_correct_explanations = [] # List of "Yes/No: explanation" for each original probe answer

    for p_answer_instance in probe_answers_from_item:
        if p_answer_instance not in unique_answer_eval_cache:
            if args.task_name == "grow":
                llm_input_prompt = (
                    f"Question:\n{question_text}\n"
                    f"Response:\n{p_answer_instance}\n"
                    f"Answer:\n{ground_truth_answer_text}\n"
                    f"Equivalence:\n"
                )
            elif args.task_name == "code":
                llm_input_prompt = (
                    f"Question:\n{question_text}\n\n"
                    f"{knowledge_text}\n\n"
                    f"Response:\n```python\n{p_answer_instance}\n```\n\n"
                    f"Correct:\n"
                )
            else:
                raise NotImplementedError
            llm_responses = chat_response_generator.generate_response(
                llm_input_prompt,
                temperature=0, top_p=1, n=1, max_tokens=100
            )
            raw_llm_response = llm_responses[0]
            is_equivalent, explanation = _parse_llm_equivalence_response(raw_llm_response)
            unique_answer_eval_cache[p_answer_instance] = (is_equivalent, explanation)

        # Retrieve cached evaluation for the current probe answer instance
        _, expl_cached = unique_answer_eval_cache[p_answer_instance]
        probe_answers_correct_explanations.append(expl_cached)
        
    # Step 3: Merge frequencies based on equivalence to ground truth
    merged_frequencies = Counter()
    for unique_ans_text, freq in original_frequencies.items():
        is_equivalent_to_gt, _ = unique_answer_eval_cache[unique_ans_text]
        if is_equivalent_to_gt:
            merged_frequencies[_GROUND_TRUTH_MERGED_KEY] += freq
        else:
            merged_frequencies[unique_ans_text] += freq

    # Step 4: Majority Voting and calculation of final metrics
    # Calculate the actual number of votes that were equivalent to the ground truth
    ground_truth_equivalent_votes = merged_frequencies.get(_GROUND_TRUTH_MERGED_KEY, 0)
    # This is the core "knowledge_confidence" reflecting the proportion of correct answers
    knowledge_confidence_value = ground_truth_equivalent_votes / total_votes if total_votes > 0 else 0.0

    item_knowledgable: bool
    majority_answer_display_text: str | None
    is_majority_answer_correct_val: bool
    majority_answer_confidence_val: float

    if not merged_frequencies: # Should only occur if total_votes was 0, already handled
        majority_answer_display_text = None
        is_majority_answer_correct_val = False
        majority_answer_confidence_val = 0.0
        item_knowledgable = False
    else:
        # Determine the highest frequency in the merged set
        max_freq = 0
        if merged_frequencies.values(): # Ensure there are values to check
            max_freq = max(merged_frequencies.values())

        # Identify all candidates (keys in merged_frequencies) that have this max frequency
        top_candidates_keys = [key for key, freq in merged_frequencies.items() if freq == max_freq]

        # Decide the winner: Ground truth wins if it's among the top candidates (handles ties)
        if _GROUND_TRUTH_MERGED_KEY in top_candidates_keys:
            chosen_winner_internal_key = _GROUND_TRUTH_MERGED_KEY
            item_knowledgable = True
            majority_answer_display_text = ground_truth_answer_text # Use actual GT text for display
            is_majority_answer_correct_val = True
        else:
            # Ground truth is not the winner (or not tied for winning with the highest frequency)
            item_knowledgable = False
            # Sort other top candidates alphabetically for deterministic tie-breaking if GT is not involved
            top_candidates_keys.sort()
            chosen_winner_internal_key = top_candidates_keys[0] # Pick the "first" non-GT winner
            majority_answer_display_text = chosen_winner_internal_key # This is the actual text of the non-GT winner
            is_majority_answer_correct_val = False

        # Calculate confidence of the declared majority answer
        winner_frequency = merged_frequencies[chosen_winner_internal_key]
        majority_answer_confidence_val = winner_frequency / total_votes if total_votes > 0 else 0.0

    # Populate the item with all calculated results
    item["knowledgable"] = item_knowledgable
    item["knowledge_confidence"] = knowledge_confidence_value # Overall correctness proportion

    item["eval_probe_results"] = {
        "probe_answers": probe_answers_from_item, # Original list of model's probe answers
        "probe_answers_correct": probe_answers_correct_explanations, # List of "Yes/No: explanation"
        "knowledge_confidence": knowledge_confidence_value, # Same as item-level
        "majority_answer": majority_answer_display_text, # Text of the winning answer
        "majority_answer_correct": is_majority_answer_correct_val, # Was the majority answer the GT?
        "majority_answer_confidence": majority_answer_confidence_val, # Confidence of the winning answer
    }

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
        
    # Load the chains from a pickle file
    with open(f'data/eval_results/{args.task_name}/probe/test_{args.data_size}_depth_{args.depth}_{args.model_name}.pkl', 'rb') as f:
        probe_dataset = pickle.load(f)
        
    # Process all chains
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    processed_data = []
    chat_response_generator = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=args.api_key)
    
    # Use tqdm to show progress
    with tqdm(total=len(probe_dataset), desc="Processing chains") as pbar:
        for item in probe_dataset:
            # Process each chain
            processed_item, usage = evaluate_probe_item(item, args, chat_response_generator)
            # Update the total token counts
            prompt_tokens = usage[args.evaluate_model_name]["prompt_tokens"]
            completion_tokens = usage[args.evaluate_model_name]["completion_tokens"]
            total_tokens = usage[args.evaluate_model_name]["total_tokens"]
            # Append the processed chain to the list
            processed_data.append(processed_item)
            # Update the progress bar with the number of tokens used
            pbar.set_postfix_str(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            pbar.update(1)
            
            # Save the processed data to a new pickle file
            with open(f'data/eval_results/{args.task_name}/probe_evaluated/test_{args.data_size}_depth_{args.depth}_{args.model_name}.pkl', 'wb') as f:
                pickle.dump(processed_data, f)

    # Save the processed data to a new pickle file
    with open(f'data/eval_results/{args.task_name}/probe_evaluated/test_{args.data_size}_depth_{args.depth}_{args.model_name}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
