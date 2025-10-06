import json
from collections import defaultdict
import os

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

RESULTS_FILE = "scripts/annotation/annotation_results.json"
MODELS = ["gpt-5-mini-2025-08-07", "gemini-2.5-pro"]

def analyze_results(results_file):
    """
    Loads annotation results and calculates single-model and joint agreement rates.
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{results_file}'")
        return

    model1, model2 = MODELS[0], MODELS[1]
    domains = list(results[model1].keys())
    
    # Dictionaries to hold aggregate counts across all domains
    total_samples = defaultdict(int)
    model1_positive_counts = defaultdict(int)
    model2_positive_counts = defaultdict(int)
    joint_positive_counts = defaultdict(int)

    print("\n" + "="*50)
    print("      DETAILED AGREEMENT ANALYSIS")
    print("="*50)

    for domain in domains:
        print(f"\n--- Domain: {domain.upper()} ---")
        for task in ["factuality", "necessity"]:
            print(f"  Task: {task.capitalize()}")

            # Get labels for both models for the current domain and task
            labels1 = results[model1][domain][task]
            labels2 = results[model2][domain][task]

            # Replace error marker -1 with 0 (No) for calculation
            labels1 = [0 if l == -1 else l for l in labels1]
            labels2 = [0 if l == -1 else l for l in labels2]
            
            num_samples = len(labels1)
            total_samples[task] += num_samples

            # Calculate single-model positives
            m1_positives = sum(labels1)
            m2_positives = sum(labels2)
            model1_positive_counts[task] += m1_positives
            model2_positive_counts[task] += m2_positives

            # Calculate joint positives (both models said "Yes")
            joint_positives = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == 1 and l2 == 1)
            joint_positive_counts[task] += joint_positives

            # Print domain-specific results
            print(f"    - Samples: {num_samples}")
            print(f"    - {model1} Positives: {m1_positives} ({m1_positives/num_samples:.2%})")
            print(f"    - {model2} Positives: {m2_positives} ({m2_positives/num_samples:.2%})")
            print(f"    - Joint Positives (Both Yes): {joint_positives} ({joint_positives/num_samples:.2%})")

    print("\n" + "="*50)
    print("      AGGREGATE RESULTS FOR PAPER")
    print("="*50)
    
    # Calculate and print the final aggregate percentages for the paper
    for task in ["factuality", "necessity"]:
        total = total_samples[task]
        m1_total_positives = model1_positive_counts[task]
        m2_total_positives = model2_positive_counts[task]
        joint_total_positives = joint_positive_counts[task]

        print(f"\n--- Overall {task.capitalize()} ---")
        print(f"Total Samples Analyzed: {total}")
        print(f"  - {model1} Positive Rate: {m1_total_positives/total:.2%} ({m1_total_positives}/{total})")
        print(f"  - {model2} Positive Rate: {m2_total_positives/total:.2%} ({m2_total_positives}/{total})")
        print(f"  - Joint Positive Rate: {joint_total_positives/total:.2%} ({joint_total_positives}/{total})")

if __name__ == "__main__":
    analyze_results(RESULTS_FILE)