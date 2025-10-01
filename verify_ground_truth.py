import pickle
import re
import multiprocessing
import os
from tqdm import tqdm
import sys

# Make sure the script can find your helper functions
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the NOW-CORRECTED worker function
from utils.helpers.code.functions import unsafe_execute_worker, EXECUTION_TIMEOUT

def verify_item(item):
    """
    Runs the canonical_solution against the test for a single item.
    Returns a tuple: (is_correct: bool, explanation: str, item_id: str)
    """
    item_id = item.get("other_metadata", {}).get("task_id", "N/A")
    
    # The ground truth code is the combination of the prompt and the canonical solution
    model_code = item.get('other_metadata', {}).get('code_prompt', '') + \
                 item.get('other_metadata', {}).get('canonical_solution', '')
                 
    unit_test_str = item.get("other_metadata", {}).get("test", "")

    with multiprocessing.Manager() as manager:
        result_dict = manager.dict()
        process = multiprocessing.Process(
            target=unsafe_execute_worker,
            args=(model_code, unit_test_str, result_dict)
        )
        process.start()
        process.join(timeout=EXECUTION_TIMEOUT)
        
        is_correct, explanation = False, "Verification failed: Unknown error."
        if process.is_alive():
            process.terminate()
            process.join()
            explanation = f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
        else:
            is_correct = result_dict.get('model_pass', False)
            explanation = result_dict.get('explanation', f"Crashed or no explanation. Exit code: {process.exitcode}")

    return is_correct, explanation, item_id

def verify_dataset(input_path):
    """Loads a dataset and verifies each item's ground truth."""
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'rb') as f:
        data_to_verify = pickle.load(f)

    print(f"\nVerifying ground truth for: {input_path}")
    
    failures = []
    
    for item in tqdm(data_to_verify, desc="Verifying items"):
        is_correct, explanation, item_id = verify_item(item)
        if not is_correct:
            failures.append((item_id, explanation))

    print("\n--- Verification Complete ---")
    if not failures:
        print(f"✅ All {len(data_to_verify)} items passed their unit tests!")
    else:
        print(f"❌ Found {len(failures)} failures out of {len(data_to_verify)} items.")
        print("--- Failure Details ---")
        for item_id, explanation in failures:
            print(f"\nTask ID: {item_id}")
            print(f"Reason: {explanation}\n")
            print("-" * 20)

if __name__ == "__main__":
    # --- CONFIGURE THE FILE YOU WANT TO TEST HERE ---
    # This should be your original data collection output, not an evaluated file.
    file_to_verify = 'data/code/test_500.pkl'

    verify_dataset(file_to_verify)