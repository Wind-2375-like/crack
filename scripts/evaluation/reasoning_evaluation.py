import os
import re
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import json
import pickle
import argparse
import multiprocessing
from tqdm import tqdm
from utils.generator.chat_response_generator import ChatResponseGenerator
from scripts.evaluation.knowledge_evaluation import PROMPT_TEMPLATES as PROMPT_TEMPLATES_EQUIVALENCE
from utils.helpers.code import unsafe_execute_worker, EXECUTION_TIMEOUT

INVALID_TASK_IDS = {
    "BigCodeBench/39", "BigCodeBench/80", "BigCodeBench/81", "BigCodeBench/82",
    "BigCodeBench/83", "BigCodeBench/90", "BigCodeBench/91", "BigCodeBench/101",
    "BigCodeBench/111", "BigCodeBench/115", "BigCodeBench/129", "BigCodeBench/147",
    "BigCodeBench/157", "BigCodeBench/177", "BigCodeCodeBench/202", "BigCodeBench/205",
    "BigCodeBench/221", "BigCodeBench/237", "BigCodeBench/245", "BigCodeBench/272",
    "BigCodeBench/274", "BigCodeBench/276", "BigCodeBench/289", "BigCodeBench/334",
    "BigCodeBench/352", "BigCodeBench/363", "BigCodeBench/372", "BigCodeBench/383",
    "BigCodeBench/416", "BigCodeBench/417", "BigCodeBench/418", "BigCodeBench/419",
    "BigCodeBench/461", "BigCodeBench/495"
}

PROMPT_TEMPLATES_EQUIVALENCE_MATH = {
    "math": """You are given a question, a final step of response, and a ground truth answer. The final step may contain a step number and "the answer is ...". PLEASE IGNORE THE STEP NUMBER. Your task is to use math knowledge to evaluate whether the response is mathematically equivalent to the ground truth answer.

If they are equivalent, answer 'Yes' and provide an explanation. Otherwise, answer 'No' and provide an explanation.

Examples:

Question:
Evaluate $(1+2i)6-3i$.
Response:
7. The answer is $9i+6$.
Ground Truth:
6+9i
Correctness:
Yes, the answer from response $9i+6$ is equivalent to the ground truth $6+9i$.

Question:
Evaluate $(1+2i)6-3i$.
Response:
7. The answer is $9+6i$.
Ground Truth:
6+9i
Correctness:
No, the answer from response is different from ground truth. The real part and the imaginary part are reversed.

Question:
The lengths of two opposite sides of a square are decreased by $40\\%$ while the lengths of the other two sides are increased by $50\\%$ to form a rectangle. By what percent does the square's area decrease?
Response:
Therefore, the square's area decreases by 10%.
Ground Truth:
10
Correctness:
Yes, the response "10%" is equivalent to the ground truth "10" because the question asks for a percentage."""
}

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
... (the same context as above)
Statement:
The name of the current head of the British government is Rishi Sunak.
NLI:
Contradiction. The context initially stated that Rishi Sunak is the current Prime Minister, but later it corrected that Keir Starmer is the current head of government.

Context:
... (the same context as above)
Statement:
The name of the current head of the US government is Donald Trump.
NLI:
Neutral. The context does not provide any information about the current head of the US government.

Context:
1. I don't know the current head of the British government.
2. I don't have the information about the answer.
Statement:
The name of the current head of the British government is Keir Starmer.
NLI:
Contradiction. The context states that it does not know the current head of the British government, which contradicts the statement that Keir Starmer is the current head of the British government.

Context:
... (the same context as above)
Statement:
The name of the current head of the US government is Donald Trump.
NLI:
Neutral. The context does not provide any information about the current head of the US government.
""",
    "default": """You are an expert in natural language inference and commonsense reasoning. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.
    
Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Context mentions some knowledge is "unknown", it should be treated as "N/A" and contradictory to the Statement.
""",
    "code": """You are an expert in Python programming and natural language inference. You will be given a 'Code' snippet and a 'Function'. Your task is to determine if the Code's usage of the function **Entails**, **Contradicts**, or is **Neutral** with respect to the correct usage of the function.

**Reasoning Framework:**

1.  **Check for Usage:** First, determine if the `Code` calls the specified function. If it does not, the answer is **Neutral**.

2.  **Validate the Call:** If the function is called, analyze the arguments used in the `Code` against the `Function Signature`.
    * **Entailment:** The usage is valid. This means:
        * All required arguments are provided.
        * Any keyword arguments used are valid (i.e., they exist in the function signature).
        * **Crucially, omitting optional arguments (e.g., those with default values like `r=None`) is a valid use case.**
    * **Contradiction:** The usage is invalid. This means:
        * A required argument is missing.
        * An incorrect or non-existent keyword argument is used (e.g., `datatype=` when it should be `dtype=`).

--- Example 1 ---

Code:
```python
import pandas as pd

def task_func(dealer_sales_data):
    # Step 1: Create DataFrame & Step 2: Handle Empty Input (if dealer_sales_data is empty)
    df = pd.DataFrame(dealer_sales_data, dtype=None)
    
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
```

Function: pandas.DataFrame(data)

NLI:
Entailment. The code contains the function call `pd.DataFrame(dealer_sales_data)`. The usage `pd.DataFrame(dealer_sales_data)` is entailed in the provided function. Only one positional parameter and the optional parameter `dtype` is set to `None`, which is by default.

--- Example 2 ---

Code:
```python
import pandas as pd

def task_func(dealer_sales_data):
    # Step 1: Create DataFrame & Step 2: Handle Empty Input (if dealer_sales_data is empty)
    df = pd.DataFrame(dealer_sales_data, datatype="float")
    
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
```

Function: pandas.DataFrame(data)

NLI:
Contradiction. The code contains a related function call `pd.DataFrame(dealer_sales_data, datatype="float")`, but it uses a different keyword argument `datatype` which does not exist. The usage of the function contradicts with the correct usage of of the provided function.

--- Example 3 ---

Code:
... (the same code as above)

Function: sklearn.linear_model.LinearRegression()

NLI:
Neutral. The code does not contain any function call related to the provided function.

--- Example 4 ---

Code:
N/A

Function: ... (any function)

NLI:
Neutral. The code does not contain any function call related to the provided function.
""",
    "default": """You are an expert in natural language inference and commonsense reasoning. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.
    
Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Context mentions some knowledge is "unknown", it should be treated as "N/A" and contradictory to the Statement.
""",
"math": """You are an expert in natural language inference and math reasoning. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.

Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Statement can be indirectly implied from the Context with math reasoning, it should also be treated as entailment.

Examples:

Context:
Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?
1. Let's call the price of a pencil x and the price of a jumbo eraser y. Then we can write two equations.
2. We have $3x+y=1.24$ and $5x+y=1.82$.
3. To solve this system, let's subtract the first equation from the second equation. This will eliminate y.
4. This simplifies to $x=0.29$.
5. That means a pencil costs 29 cents.
6. The answer is 29 cents.
Statement:
After subtracting $3p+e=1.24$ from $5p+e=1.82$, we will have $2p = 0.58$, which solves to $p = 0.29$.
NLI:
Entailment. The statement is a direct summary of the mathematical reasoning presented in steps 2, 3, and 4 in the context. It describes the exact same process and reaches the identical conclusion. The use of different variables (p and e instead of x and y) is a superficial change that doesn't affect the logic, and missing $2p = 0.58$ does not matter because $x = 0.29$ already implies it.

Context:
Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?
1. Let's call the price of a pencil x and the price of a jumbo eraser y. Then we can write two equations.
2. We have $3x+y=1.24$ and $5x+y=1.82$.
3. To solve this system, let's add the first equation to the second equation. This will eliminate y.
4. This simplifies to $2x=0.58$. So $x=0.29$.
5. That means a pencil costs 29 cents.
6. The answer is 29 cents.
Statement:
After subtracting $3p+e=1.24$ from $5p+e=1.82$, we will have $2p = 0.58$, which solves to $p = 0.29$.
NLI:
Contradiction. The context states in step 3 that the two equations should be added. The statement, however, describes the process using subtraction. Since adding and subtracting are opposite operations, the statement directly contradicts the method described in the context.

Context:
Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?
1. Let's call the price of a pencil x and the price of a jumbo eraser y. Then we can write two equations.
2. We have $3x+y=1.24$ and $5x+y=1.82$.
3. To solve this system, let's add the first equation to the second equation. This will eliminate y.
4. This simplifies to $2x=0.58$. So $x=0.29$.
5. That means a pencil costs 29 cents.
6. The answer is 29 cents.
Statement:
The commutative rule, also known as the commutative property, states that the order of numbers in addition and multiplication doesn't change the result.
NLI:
Neutral. The context does not provide any information about the commutative rules. The statement is completely irrelevant to the context."""
}

def parse_args():
    """
    Parses command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process a chain of triples.")
    parser.add_argument('--data_size', type=int, default=500, help="Number of triples to process")
    parser.add_argument('--api_config_file', type=str, default="./api_key/config.json", help="Path to the API configuration file")
    parser.add_argument('--model_name', type=str, default="llama-3.2-3b", help="Model name for the API")
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-5-mini-2025-08-07", help="Model name for the evaluation")
    parser.add_argument('--task_name', type=str, default="grow", help="Task name")
    parser.add_argument('--inject_knowledge', action='store_true', help="Whether to inject knowledge into the input")
    parser.add_argument('--knowledge_aggregation_scope', type=int, default=1, help="Scope for aggregating 'unknown' knowledge. Must be >= 1. 1: item-specific. N (e.g., 10, 100): group of N items.")
    parser.add_argument('--method', type=str, default="base", help="Method to use for complex reasoning amid conflicting knowledge")
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
    
    def _execute_llm_generated_code_and_test(model_final_answer_candidate: str, unit_test_str: str):
        """
        Execute the LLM's generated code against a set of unit tests in a
        safe, isolated, and time-limited process.

        Args:
            model_final_answer_candidate: A string containing the Python code
                                        generated by the LLM (expected to define 'task_func').
            unit_test_str: A string containing the Python unittest code
                        (expected to define a 'TestCases' class that uses 'task_func').

        Returns:
            A tuple (model_pass: bool, explanation: str).
            model_pass is True if all tests pass, False otherwise.
            explanation provides details of the test run or errors.
        """
        if model_final_answer_candidate == "N/A":
            return False, "The model failed to provide a valid code block."

        # Use a manager to create a shared dictionary for inter-process communication
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict()
            
            # Create the child process to run the untrusted code safely
            process = multiprocessing.Process(
                target=unsafe_execute_worker,
                args=(model_final_answer_candidate, unit_test_str, result_dict)
            )
            
            process.start()
            process.join(timeout=EXECUTION_TIMEOUT)

            # Check if the process is still running after the timeout
            if process.is_alive():
                process.terminate()  # Forcefully stop the process
                process.join()
                return False, f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
            
            # Check if the process exited with an error code
            if process.exitcode != 0:
                # Check if the worker populated the result dict before crashing
                if 'explanation' in result_dict:
                    return result_dict.get('model_pass', False), result_dict['explanation']
                return False, f"Execution process crashed with exit code {process.exitcode}."

            # Return the results captured by the worker process
            return result_dict.get('model_pass', False), result_dict.get('explanation', "Unknown error: Result dictionary was not populated.")     

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
    if args.task_name == "grow":
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
            temperature=0, top_p=1, n=1, max_tokens=4096
        )[0]
        
        final_answer_correct, final_answer_explanation = _parse_llm_equivalence_response(raw_equivalence_response)
        item["final_answer_correct"] = final_answer_correct
        item["final_answer_explanation"] = final_answer_explanation
    elif args.task_name == "code":
        unit_test = item.get("other_metadata", {}).get("test", "")
        model_final_answer_candidate = "N/A"
        if item.get("model_response") and item["model_response"].strip():
            # Extract coding block between ```python and ```
            match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
            if match:
                model_final_answer_candidate = match.group(1).strip()
        final_answer_correct, final_answer_explanation = _execute_llm_generated_code_and_test(model_final_answer_candidate, unit_test)
        item["final_answer_correct"] = final_answer_correct
        item["final_answer_explanation"] = final_answer_explanation
    elif args.task_name == "math":
        model_final_answer_candidate = ""
        if item.get("model_response") and item["model_response"].strip():
            model_response_lines = [line.strip() for line in item["model_response"].split("\n") if line.strip()]
            if model_response_lines:
                model_final_answer_candidate = model_response_lines[-1]
    
        system_prompt_key_equivalence = args.task_name if args.task_name in PROMPT_TEMPLATES_EQUIVALENCE_MATH else "default"
        system_prompt_equivalence = PROMPT_TEMPLATES_EQUIVALENCE_MATH[system_prompt_key_equivalence]
        
        llm_input_prompt_equivalence = (
            f"Question:\n{item['question']}\n"
            f"Response:\n{model_final_answer_candidate}\n" # Use extracted model's final answer line
            f"Ground Truth:\n{ground_truth_final_answer}\n"
            f"Equivalence:\n"
        )
        chat_response_generator.update_chat_history([
            ("system", system_prompt_equivalence),
        ])
        raw_equivalence_response = chat_response_generator.generate_response(
            llm_input_prompt_equivalence,
            temperature=0, top_p=1, n=1, max_tokens=4096
        )[0]
        
        final_answer_correct, final_answer_explanation = _parse_llm_equivalence_response(raw_equivalence_response)
        item["final_answer_correct"] = final_answer_correct
        item["final_answer_explanation"] = final_answer_explanation
    else:
        return NotImplementedError(f"Task {args.task_name} is not implemented!")

    # 2. Evaluate whether each knowledge in required_knowledge is entailment/contradiction/neutral with the model response
    chat_response_generator.update_chat_history([
        ("system", PROMPT_TEMPLATES_NLI[args.task_name]),
    ])
    model_full_response_context = item.get("model_response", "") # Use the full model response as context for NLI

    for knowledge_item in item["required_knowledge"]:
        knowledge_text = knowledge_item["knowledge"]
        if args.task_name == "grow":
            llm_input_prompt_nli = (
                f"Context:\n{model_full_response_context}\n\n"
                f"Statement:\n{knowledge_text}\n\n"
                f"NLI:\n"
            )
        elif args.task_name == "code":
            answer_text = knowledge_item["answer"]
            llm_input_prompt_nli = (
                f"Code:\n```python\n{model_final_answer_candidate}\n```\n\n"
                f"Function:\n{answer_text}\n\n"
                f"NLI:\n"
            )
        elif args.task_name == "math":
            llm_input_prompt_nli = (
                f"Context:\n{model_full_response_context}\n\n"
                f"Statement:\n{knowledge_text}\n\n"
                f"NLI:\n"
            )
        else:
            raise NotImplementedError(f"Task {args.task_name} is not implemented!")
        
        raw_nli_response = chat_response_generator.generate_response(
            llm_input_prompt_nli,
            temperature=0, top_p=1, n=1, max_tokens=4096
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
    input_file_name = f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl"
    input_file_path = os.path.join(input_file_dir, input_file_name)

    output_dir_base = f'data/eval_results/{args.task_name}/injection_evaluated/'
    os.makedirs(output_dir_base, exist_ok=True)
    output_file_name = f"{'original' if not args.inject_knowledge else args.method}_{args.data_size}_{args.model_name}_{args.knowledge_aggregation_scope}.pkl"
    output_file_path = os.path.join(output_dir_base, output_file_name)

    # Load the chains from a pickle file
    with open(input_file_path, 'rb') as f:
        eval_dataset = pickle.load(f)
        
    with open(f'data/{args.task_name}/test_{args.data_size}.pkl', "rb") as f:
        raw_dataset = pickle.load(f)
        
    processed_data = []
    start_index = 0
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'rb') as f:
                # Load whatever is already done
                processed_data = pickle.load(f)
            # Set the starting point to the number of items already processed
            start_index = len(processed_data)
            if start_index > 0:
                print(f"--- Resuming from checkpoint: {start_index} items already evaluated. ---")
        except (pickle.UnpicklingError, EOFError, IndexError):
            print(f"--- Warning: Checkpoint file corrupted. Starting from scratch. ---")
            processed_data = []
            start_index = 0

    if start_index >= len(eval_dataset):
        print("--- All items have already been processed. Exiting. ---")
        sys.exit(0)
        
    # Process all chains
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    processed_data = []
    chat_response_generator = ChatResponseGenerator(model_name=args.evaluate_model_name, api_key=args.api_key)
    
    # Use tqdm to show progress
    with tqdm(total=len(eval_dataset), initial=start_index, desc="Evaluating items") as pbar:
        for item_idx, item in enumerate(eval_dataset):
            if item_idx < start_index:
                pbar.update(1)
                continue
            raw_item = raw_dataset[item_idx]
            item["other_metadata"] = raw_item.get("other_metadata", {})
            if args.task_name == "code":
                for idx, k in enumerate(item["required_knowledge"]):
                    k['answer'] = raw_item.get("probe_questions", [])[idx].get("answer", "")
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

    # Save the processed data to a new pickle file
    with open(output_file_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nProcessing complete. Evaluated data saved to: {output_file_path}")
    print(f"Total tokens used for evaluation ({args.evaluate_model_name}):")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")