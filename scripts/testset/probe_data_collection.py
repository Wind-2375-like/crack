import os
import sys
import pickle
import random
import argparse
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from utils.dataset.probe_dataset import ProbeDataset

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process a flattened probe dataset to add locality pairs."
    )
    parser.add_argument('--data_size', type=int, default=500, help="The size of the dataset to process.")
    parser.add_argument('--task_name', type=str, required=True, help="The name of the task (e.g., 'math', 'code').")
    return parser.parse_args()

def add_locality_pairs(task_name: str, data_size: int):
    """
    Loads a probe dataset, adds efficacy and locality IDs to each probe,
    and saves the result to a new file.
    """
    input_path = f'data/{task_name}/test_{data_size}.pkl'
    
    base_name, extension = os.path.splitext(input_path)
    output_path = f"{base_name}_probe{extension}"

    print(f"Loading flattened probes from: {input_path}")
    try:
        all_probes = ProbeDataset(input_path)
        all_probes_list = list(all_probes)
        if not all_probes_list:
            print("Warning: The loaded dataset is empty.")
            return
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    # Use a fixed seed for reproducible random pairing
    random.seed(42)
    
    total_probes = len(all_probes_list)
    print(f"Processing {total_probes} probes to find irrelevant questions...")

    # Iterate through each probe to assign its efficacy and locality IDs
    for i, efficacy_probe in enumerate(tqdm(all_probes_list, desc="Pairing Probes")):
        while True:
            locality_id_candidate = random.randrange(total_probes)
            
            if locality_id_candidate == i:
                continue

            locality_probe_candidate = all_probes_list[locality_id_candidate]
            
            is_irrelevant_context = (locality_probe_candidate['complex_question_id'] != efficacy_probe['complex_question_id'])
            is_different_question = (locality_probe_candidate['question'] != efficacy_probe['question'])

            if is_irrelevant_context and is_different_question:
                efficacy_probe['efficacy_id'] = i
                efficacy_probe['locality_id'] = locality_id_candidate
                break

    # Save the modified list to the new output file
    print(f"Saving processed data with locality pairs to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_probes_list, f)
        
    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    add_locality_pairs(args.task_name, args.data_size)