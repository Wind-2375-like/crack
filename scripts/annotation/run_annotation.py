import sys
import json
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
import os

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from utils.generator.chat_response_generator import ChatResponseGenerator

CONFIG = {
    "models_to_run": [
        "gpt-5-mini-2025-08-07",
        "gemini-2.5-pro"
    ],
    "domains": {
        "code": "data/code/test_500.pkl",
        "math": "data/math/test_500.pkl",
        "grow": "data/grow/test_500.pkl"
    },
    "num_samples": 50,
    "random_seed": 42, # for reproducible sampling
    "api_config_file": "api_key/config.json",
    "results_output_file": "annotation_results.json"
}

PROMPT_TEMPLATES = {
    "code": {
        "factuality": """You are given a coding question which requires a function call from Python external libraries as the ground truth answer. Is the answer factually correct? Please provide a short sentence as explanation and then answer Yes if the answer is factually correct or No if it is not.

Question: {probe_question}

Answer: {probe_answer}""",
        "necessity": """You are given a coding question which requires implementing a Python code with multiple required functions from external libraries. Does the problem description explicitly require the use of every function listed below to be considered a valid solution? Answer Yes if all functions are mandated by the prompt's text, or No otherwise.

Here is an example where all function calls are necessary (Yes):
Question: Calculates the average of the sums of absolute differences for all permutations of a given list. Each permutation is shuffled... Your code should include these functions: `itertools.permutations`, `random.shuffle`.
Function Calls:
itertools.permutations(args_0)
random.shuffle(x)
Reason: Yes, as indicated in the question, `itertools.permutations`, `random.shuffle` must be included.

Here is an example where some facts are not necessary (No):
Question: Calculates the average... Your code should include these functions: `itertools.permutations`, `random.shuffle`.
Function Calls:
math.pow(base, exp)
math.sqrt(x)
Reason: No, they are not stated in the question.

Examples finish and response begins:

Question: {multihop_question}

Function Calls:
{function_calls}

Reason:"""
    },
    "math": {
        "factuality": """You are given a math question. Is the answer to the question factually correct? Please provide a short sentence as explanation and then answer Yes if the answer is factually correct or No if it is not.

Question: {probe_question}

Answer: {probe_answer}""",
        "necessity": """You are given a math question and a required solution path. You are also given a list of knowledge steps. Your task is to determine if every piece of knowledge is essential for solving the problem **by following the required path exactly**.

Are all the listed knowledge steps required to construct the solution **as demanded by the problem's constraints**? An alternative mathematical method is irrelevant if it deviates from the specified path. Provide a short sentence as explanation. Then answer **Yes** if all knowledge is essential to the required method, or **No** if some knowledge is irrelevant or deviates from the method.

---
**Example where the knowledge is required (Yes):**
Question: Find the vector $\\mathbf{{v}}$ with the smallest magnitude... **Your solution must focus on solving the linear system from the cross-product and minimizing the norm by completing the square.**
Knowledge:
- The cross product yields the linear system...
- Solving for b and c in terms of a gives...
- The squared magnitude is...
- Completing the square on the magnitude expression gives...
- The minimum occurs at a = -7...
Reason: **Yes, every knowledge step is a critical part of the required method of setting up a linear system and minimizing the magnitude by completing the square.**
---
**Examples complete. Your response begins now:**

Question: {multihop_question}

Knowledge:
{knowledge_steps}

Reason:"""
    },
    "grow": {
        "factuality": """You are given a question and an answer. Is the answer to the question factually correct? Please provide a short sentence as explanation and then answer Yes if the answer is factually correct or No if it is not.

Question: {probe_question}

Answer: {probe_answer}""",
        "necessity": """You are given a multi-hop reasoning question. You are also given a list of knowledge steps. Your task is to determine if every piece of knowledge is essential for solving the problem. A step is essential if the final answer cannot be reached without it.

Question: {multihop_question}

Knowledge:
{knowledge_steps}

Reason:"""
    }
}


def parse_response_to_binary(response_text):
    """Parses model response text into a binary label (1 for Yes, 0 for No)."""
    return 1 if "yes" in response_text.lower() else 0


def annotate_factuality(generator, data, domain, num_samples):
    """Runs the factuality annotation task for probing questions."""
    print(f"\nAnnotating factuality for {num_samples} probing questions in '{domain}' domain...")
    
    all_probe_questions = [pq for item in data for pq in item["probe_questions"]]
    sampled_questions = all_probe_questions[:num_samples]
    
    labels = []
    for chain in tqdm(sampled_questions, desc=f"Factuality ({domain})"):
        prompt = PROMPT_TEMPLATES[domain]["factuality"].format(
            probe_question=chain["question"],
            probe_answer=chain["answer"]
        )
        try:
            response = generator.generate_response(query=prompt, max_tokens=4096)[0]
            labels.append(parse_response_to_binary(response))
        except Exception as e:
            print(f"Error processing item: {e}")
            labels.append(-1) # Use -1 to denote an error
    return labels

def annotate_necessity(generator, data, domain, num_samples):
    """Runs the necessity annotation task for multihop questions."""
    print(f"\nAnnotating necessity for {num_samples} multihop questions in '{domain}' domain...")

    sampled_items = list(data)[:num_samples]
    
    labels = []
    for item in tqdm(sampled_items, desc=f"Necessity ({domain})"):
        if domain == "code":
            # Code domain uses 'function_calls' format
            knowledge_or_calls = '\n'.join([p['answer'] for p in item["probe_questions"]])
            prompt = PROMPT_TEMPLATES[domain]["necessity"].format(
                multihop_question=item["multihop_question"],
                function_calls=knowledge_or_calls
            )
        else:
            # Math and Grow domains use 'knowledge_steps' format
            knowledge_or_calls = '\n'.join([p['knowledge'] for p in item["probe_questions"]])
            prompt = PROMPT_TEMPLATES[domain]["necessity"].format(
                multihop_question=item["multihop_question"],
                knowledge_steps=knowledge_or_calls
            )
        
        try:
            response = generator.generate_response(query=prompt, max_tokens=4096)[0]
            labels.append(parse_response_to_binary(response))
        except Exception as e:
            print(f"Error processing item: {e}")
            labels.append(-1) # Use -1 to denote an error

    return labels

def calculate_statistics(results):
    """Calculates and prints average agreement and Cohen's Kappa."""
    print("\n" + "="*50)
    print("      ANNOTATION RESULTS & STATISTICS")
    print("="*50)

    models = list(results.keys())
    if len(models) < 2:
        print("\nNOTE: Only one model's results found. Cohen's Kappa requires two sets of annotations.")
    
    for domain in CONFIG["domains"]:
        print(f"\n--- Domain: {domain.upper()} ---")
        
        # --- Factuality Stats ---
        print("\n  [Probing Question Factuality]")
        factuality_labels = {}
        for model in models:
            labels = [l if l != -1 else 0 for l in results[model][domain]["factuality"]]
            if not labels: continue
            agreement_rate = sum(labels) / len(labels)
            factuality_labels[model] = labels
            print(f"    - {model}: Agreement Rate (Yes %) = {agreement_rate:.2%}")
        
        if len(factuality_labels) == 2:
            kappa = cohen_kappa_score(factuality_labels[models[0]], factuality_labels[models[1]])
            print(f"    - Cohen's Kappa ({models[0]} vs {models[1]}): {kappa:.4f}")

        # --- Necessity Stats ---
        print("\n  [Multihop Question Necessity]")
        necessity_labels = {}
        for model in models:
            labels = [l if l != -1 else 0 for l in results[model][domain]["necessity"]]
            if not labels: continue
            agreement_rate = sum(labels) / len(labels)
            necessity_labels[model] = labels
            print(f"    - {model}: Agreement Rate (Yes %) = {agreement_rate:.2%}")
        
        if len(necessity_labels) == 2:
            kappa = cohen_kappa_score(necessity_labels[models[0]], necessity_labels[models[1]])
            print(f"    - Cohen's Kappa ({models[0]} vs {models[1]}): {kappa:.4f}")

def main():
    """Main script execution function."""
    random.seed(CONFIG["random_seed"])
    
    # Load API key once
    try:
        with open(CONFIG["api_config_file"], 'r') as f:
            api_key = json.load(f).get("api_key")
        if not api_key:
            raise ValueError("API key not found in config file.")
    except FileNotFoundError:
        print(f"Error: API config file not found at {CONFIG['api_config_file']}")
        return

    all_results = defaultdict(lambda: defaultdict(dict))

    for model_name in CONFIG["models_to_run"]:
        generator = ChatResponseGenerator(model_name=model_name, api_key=api_key)
        
        for domain, filepath in CONFIG["domains"].items():
            print(f"\n{'='*20} Processing Domain: {domain.upper()} with Model: {model_name} {'='*20}")
            try:
                processed_data = pd.read_pickle(filepath)
            except FileNotFoundError:
                print(f"Warning: Data file not found for domain '{domain}' at {filepath}. Skipping.")
                continue

            # Task 1: Probing Question Factuality
            factuality_labels = annotate_factuality(generator, processed_data, domain, CONFIG["num_samples"])
            all_results[model_name][domain]["factuality"] = factuality_labels

            # Task 2: Multihop Question Necessity
            necessity_labels = annotate_necessity(generator, processed_data, domain, CONFIG["num_samples"])
            all_results[model_name][domain]["necessity"] = necessity_labels

    # Save results to a JSON file
    with open(CONFIG["results_output_file"], 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll annotation results saved to '{CONFIG['results_output_file']}'.")

    # Calculate and display summary statistics
    calculate_statistics(all_results)


if __name__ == "__main__":
    main()