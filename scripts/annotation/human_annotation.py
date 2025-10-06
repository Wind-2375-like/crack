import json
import pandas as pd
import random
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from collections import defaultdict
import os
import sys

CONFIG = {
    "domains": {
        "code": "data/code/test_500.pkl",
        "math": "data/math/test_500.pkl",
        "grow": "data/grow/test_500.pkl"
    },
    "llm_results_file": "scripts/annotation/annotation_results.json",
    "human_annotations_output_file": "scripts/annotation/human_annotations.json",
    "models": ["gpt-5-mini-2025-08-07", "gemini-2.5-pro"],
    "num_samples_per_task": 50,
    "random_seed": 42
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_data():
    try:
        with open(CONFIG["llm_results_file"], 'r') as f:
            llm_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: LLM results file not found at '{CONFIG['llm_results_file']}'")
        sys.exit(1)
    domain_data = {}
    for domain, filepath in CONFIG["domains"].items():
        try:
            domain_data[domain] = pd.read_pickle(filepath)
        except FileNotFoundError:
            print(f"⚠️ Warning: Data file not found for domain '{domain}' at {filepath}. Skipping.")
    if not domain_data:
        print("❌ Error: No domain data could be loaded. Exiting.")
        sys.exit(1)
    return llm_results, domain_data

def create_annotation_pools(llm_results, domain_data):
    factuality_pool = []
    necessity_pool = []
    for model in CONFIG["models"]:
        for domain, tasks in llm_results[model].items():
            if domain not in domain_data: continue
            if "factuality" in tasks:
                all_probes = [(item_idx, pq) for item_idx, item in enumerate(domain_data[domain]) for pq in item["probe_questions"]]
                for i, label in enumerate(tasks["factuality"]):
                    item_idx, probe_info = all_probes[i]
                    factuality_pool.append({
                        "model": model, "domain": domain, "task": "factuality", "index": i,
                        "llm_label": label if label != -1 else 0,
                        "question": probe_info['question'],
                        "context": f"Answer: {probe_info['answer']}"
                    })
            if "necessity" in tasks:
                for i, label in enumerate(tasks["necessity"]):
                    item = domain_data[domain][i]
                    context_str = ""
                    if domain == 'code':
                        knowledge = '\n'.join([p['answer'] for p in item["probe_questions"]])
                        context_str = f"Function Calls:\n{knowledge}"
                    else:
                        knowledge = '\n'.join([p['knowledge'] for p in item["probe_questions"]])
                        context_str = f"Knowledge:\n{knowledge}"
                    necessity_pool.append({
                        "model": model, "domain": domain, "task": "necessity", "index": i,
                        "llm_label": label if label != -1 else 0,
                        "question": item['multihop_question'],
                        "context": context_str
                    })
    return factuality_pool, necessity_pool

def run_annotation_session(samples_to_annotate, task_name):
    human_annotations = []
    total_items = len(samples_to_annotate)
    try:
        for i, item in enumerate(samples_to_annotate):
            clear_screen()
            print("="*80)
            print(f"Human Annotation: {task_name.upper()} ({i+1}/{total_items})")
            print("="*80)
            print(f"Domain: {item['domain'].upper()} | LLM Judge: {item['model']}")
            print("-" * 30)
            print(f"Question: {item['question']}")
            print(f"\n{item['context']}")
            print("-" * 30)
            print(f"The LLM judged this as: {'Yes' if item['llm_label'] == 1 else 'No'}")
            while True:
                user_input = input("\nYour judgment (y/n) or 'quit': ").lower()
                if user_input in ['y', 'yes']:
                    item_with_human_label = item.copy()
                    item_with_human_label["human_label"] = 1
                    human_annotations.append(item_with_human_label)
                    break
                elif user_input in ['n', 'no']:
                    item_with_human_label = item.copy()
                    item_with_human_label["human_label"] = 0
                    human_annotations.append(item_with_human_label)
                    break
                elif user_input == 'quit':
                    print("\nAborting session.")
                    return human_annotations, True
                else:
                    print("❌ Invalid input. Please enter 'y' for yes or 'n' for no.")
        return human_annotations, False
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user (Ctrl+C).")
        return human_annotations, True

def calculate_and_display_stats(annotated_items):
    """
    Calculates and displays statistics, aggregating all models for each task.
    """
    results_by_task = {
        "factuality": {'llm_labels': [], 'human_labels': []},
        "necessity": {'llm_labels': [], 'human_labels': []}
    }
    
    if not annotated_items:
        print("\nNo annotations to analyze.")
        return

    for item in annotated_items:
        task = item['task']
        if task in results_by_task:
            results_by_task[task]['llm_labels'].append(item['llm_label'])
            results_by_task[task]['human_labels'].append(item['human_label'])

    print("\n" + "="*80)
    print(" " * 18 + "HUMAN vs. LLM ANNOTATION ANALYSIS")
    print("="*80)

    for task_name, task_data in results_by_task.items():
        print(f"\n{'---'*10} TASK: {task_name.upper()} {'---'*10}")
        
        llm_labels = task_data['llm_labels']
        human_labels = task_data['human_labels']
        num_items = len(llm_labels)

        if num_items == 0:
            print(f"No data for task: {task_name}")
            continue

        f1 = f1_score(human_labels, llm_labels, zero_division=0)
        precision = precision_score(human_labels, llm_labels, zero_division=0)
        recall = recall_score(human_labels, llm_labels, zero_division=0)

        print(f"\n  --- Overall Task Results ({num_items} total samples) ---")
        print("    Classification Metrics (Human as ground truth):")
        print(f"      - Precision: {precision:.4f}")
        print(f"      - Recall:    {recall:.4f}")
        print(f"      - F1-Score:  {f1:.4f}")
            
    print("\n" + "="*80)

def start_new_annotation_workflow():
    """Handles the entire process of starting a new annotation session."""
    llm_results, domain_data = load_data()
    factuality_pool, necessity_pool = create_annotation_pools(llm_results, domain_data)
    
    random.seed(CONFIG["random_seed"])
    factuality_samples = random.sample(factuality_pool, CONFIG["num_samples_per_task"])
    necessity_samples = random.sample(necessity_pool, CONFIG["num_samples_per_task"])
    
    all_completed_annotations = []
    
    # --- Factuality Session ---
    clear_screen()
    print("You will now annotate 50 samples for the FACTUALITY task.")
    input("Press Enter to begin...")
    
    factuality_annotations, was_quit = run_annotation_session(factuality_samples, task_name="Factuality")
    all_completed_annotations.extend(factuality_annotations)
    
    # --- Necessity Session ---
    if not was_quit:
        clear_screen()
        print(f"✅ Factuality task complete ({len(factuality_annotations)} items annotated).")
        print("\nYou will now annotate 50 samples for the NECESSITY task.")
        input("Press Enter to begin...")
        
        necessity_annotations, was_quit = run_annotation_session(necessity_samples, task_name="Necessity")
        all_completed_annotations.extend(necessity_annotations)

    # --- Save and Analyze ---
    if all_completed_annotations:
        with open(CONFIG["human_annotations_output_file"], 'w') as f:
            json.dump(all_completed_annotations, f, indent=4)
        print(f"\n✅ Successfully saved {len(all_completed_annotations)} total annotations to '{CONFIG['human_annotations_output_file']}'.")
        
        if was_quit:
            print("\nAnalysis is based on the partial session.")
        else:
            print("\nAnnotation complete! Now running final analysis...")
        calculate_and_display_stats(all_completed_annotations)
    else:
        print("\nNo annotations were made. Nothing to save or analyze.")

if __name__ == "__main__":
    # Check if an existing annotation file is present
    if os.path.exists(CONFIG["human_annotations_output_file"]):
        clear_screen()
        choice = input(
            "An existing 'human_annotations.json' file was found.\n\n"
            "  [A] Analyze the existing file\n"
            "  [N] Start a new annotation session (will overwrite the old file)\n\n"
            "Your choice (A/N): "
        ).lower()

        if choice == 'a':
            print("\nAnalyzing existing annotation file...")
            try:
                with open(CONFIG["human_annotations_output_file"], 'r') as f:
                    completed_annotations = json.load(f)
                calculate_and_display_stats(completed_annotations)
            except (json.JSONDecodeError, FileNotFoundError):
                print("❌ Error reading the annotation file. It might be corrupted or empty.")
                sys.exit(1)
        
        elif choice == 'n':
            print("\nStarting a new annotation session...")
            start_new_annotation_workflow()
        
        else:
            print("\nInvalid choice. Exiting.")
            sys.exit(0)
            
    else:
        # If no file exists, start the annotation process directly
        print("No existing annotation file found.")
        start_new_annotation_workflow()