import os
import re
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import io
import json
import pickle
import argparse
import unittest
import traceback
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
    "code": """You are an expert in natural language inference and coding. You will be given a "Context" (the model's reasoning response) and a "Statement" (a piece of knowledge). Your task is to determine if the Context finally entails, contradicts, or is neutral with respect to the Statement.

Answer "Entailment", "Contradiction", or "Neutral" and provide a brief explanation of your reasoning.

Note that if the Context mentions some knowledge is "unknown", it should be treated as "N/A" and contradictory to the Statement.

--- Example 1 ---

Code:
```python
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
```

Function: pandas.DataFrame(data)

Docstring: Two-dimensional, size-mutable, potentially heterogeneous tabular data...

Parameters
----------
data : ndarray (structured or homogeneous), ...
...
dtype : dtype, default None
    Data type to force. Only a single dtype is allowed. If None, infer.
...

NLI:
Entailment. The code contains the function call `pd.DataFrame(dealer_sales_data)`. The usage `pd.DataFrame(dealer_sales_data)` is entailed in the docstring. Only one positional parameter and no optional parameters.

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

Docstring: ... (the same as above)

NLI:
Contradiction. The code contains a related function call `pd.DataFrame(dealer_sales_data, datatype="float")`, but it uses a different keyword argument `datatype` which does not exist in the docstring. The usage of the function contradicts with the docstring.

--- Example 3 ---

Code:
... (the same code as above)

Function: sklearn.linear_model.LinearRegression()

Docstring: Ordinary least squares Linear Regression...

NLI:
Neutral. The code does not contain any function call related to the docstring.

--- Example 4 ---

Code:
N/A

Function: ... (any function)

Docstring: ... (any docstring)

NLI:
Neutral. The code does not contain any function call related to the docstring.
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
        Execute the LLM's generated code against a set of unit tests.

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
        if model_final_answer_candidate is None:
            model_final_answer_candidate = "" # Ensure it's a string

        # Combine the LLM's code (defining task_func) with the unit tests
        # task_func should be defined before TestCases uses it.
        full_code = model_final_answer_candidate + "\n\n" + unit_test_str

        # Prepare a global environment for the execution.
        # Setting __name__ to __main__ can be important if the script has checks like
        # if __name__ == "__main__":, though not typical for just function and class defs.
        execution_globals = {"__name__": "__main__"}

        try:
            # First, try to compile the combined code to catch syntax errors early.
            compiled_code = compile(full_code, '<string>', 'exec')
        except SyntaxError as e:
            model_pass = False
            # Format a detailed syntax error message
            error_line_text = e.text.rstrip() if e.text else "N/A"
            caret_line = ""
            if e.text and e.offset is not None:
                caret_line = "\n" + " " * (e.offset - 1) + "^" if e.offset > 0 else "\n^"
            
            explanation = (
                f"Compilation Error (SyntaxError):\n"
                f"  Message: {e.msg}\n"
                f"  Line number: {e.lineno}\n"
                f"  Offset: {e.offset}\n"
                f"  Problematic line: {error_line_text}{caret_line}\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
            return model_pass, explanation
        except Exception as e: # Catch other potential compilation errors
            model_pass = False
            explanation = (
                f"Compilation Error ({type(e).__name__}): {e}\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
            return model_pass, explanation

        # Prepare to capture the output of the test runner
        log_stream = io.StringIO()
        # verbosity=2 provides detailed output for each test
        runner = unittest.TextTestRunner(stream=log_stream, verbosity=2)
        suite = unittest.TestSuite()
        
        # The unit test string is expected to define a class named 'TestCases'
        test_class_name = "TestCases"

        try:
            # Execute the compiled code. This will define `task_func` (from LLM code)
            # and `TestCases` (from unit_test_str) in `execution_globals`.
            exec(compiled_code, execution_globals)

            # Load tests from the dynamically defined 'TestCases' class
            if test_class_name in execution_globals:
                test_class = execution_globals[test_class_name]
                loader = unittest.TestLoader()
                suite.addTest(loader.loadTestsFromTestCase(test_class))
            else:
                # This would happen if 'TestCases' is not defined in unit_test_str
                model_pass = False
                explanation = (
                    f"Execution Error: The test class '{test_class_name}' was not found "
                    "after executing the combined code. Please ensure the unit_test string "
                    f"defines a class named '{test_class_name}'."
                )
                return model_pass, explanation

            # Run the tests
            result = runner.run(suite)

            if result.wasSuccessful():
                model_pass = True
                explanation = "All testing points passed."
            else:
                model_pass = False
                test_output = log_stream.getvalue()
                explanation = f"Some tests failed or errored. Full test output:\n{test_output}"

        except NameError as e:
            # This typically occurs if 'task_func' is not defined by the LLM's code,
            # or if the LLM's code (or test code) refers to an undefined name.
            model_pass = False
            explanation = (
                f"Execution Error (NameError): {e}.\n"
                f"This often means 'task_func' was not defined correctly by the model's code, "
                f"or an undefined name was used.\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
        except Exception as e:
            # Catch any other unexpected errors during execution or test running
            model_pass = False
            explanation = (
                f"An unexpected error occurred during execution or testing ({type(e).__name__}): {e}\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
        finally:
            log_stream.close()

        return model_pass, explanation     

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
                model_final_answer_candidate = model_response_lines[-1]\
    
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
            temperature=0, top_p=1, n=1, max_tokens=100
        )[0]
        
        final_answer_correct, final_answer_explanation = _parse_llm_equivalence_response(raw_equivalence_response)
        item["final_answer_correct"] = final_answer_correct
        item["final_answer_explanation"] = final_answer_explanation
    elif args.task_name == "code":
        unit_test = item.get("other_metadata", {}).get("test", "")
        model_final_answer_candidate = ""
        if item.get("model_response") and item["model_response"].strip():
            # Extract coding block between ```python and ```
            match = re.search(r"```python\s*([\s\S]*?)\s*```", item["model_response"])
            if match:
                model_final_answer_candidate = match.group(-1)
        final_answer_correct, final_answer_explanation = _execute_llm_generated_code_and_test(model_final_answer_candidate, unit_test)
        item["final_answer_correct"] = final_answer_correct
        item["final_answer_explanation"] = final_answer_explanation

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
            llm_input_prompt_nli = (
                f"Code:\n```python\n{model_final_answer_candidate}\n```\n\n"
                f"{knowledge_text}"
                f"NLI:\n"
            )
        
        raw_nli_response = chat_response_generator.generate_response(
            llm_input_prompt_nli,
            temperature=0, top_p=1, n=1, max_tokens=256
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