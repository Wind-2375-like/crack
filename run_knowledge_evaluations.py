import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from utils.helpers import run_command

def generate_commands(args):
    """Generates all the evaluation commands based on the provided arguments."""
    commands = []
    base_script = ["python", "scripts/evaluation/knowledge_evaluation.py"]

    print("Generating knowledge probe evaluation commands...")
    for task, model in itertools.product(args.task_names, args.model_names):
        commands.append(base_script + ["--task_name", task, "--model_name", model])

    return commands

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning evaluation scripts in parallel.")
    
    # --- Model and Task Arguments ---
    parser.add_argument('--model_names', nargs='+', default=["llama-3.2-1b", "llama-3.2-3b", "qwen-2.5-1.5b", "qwen-2.5-3b"], help='List of model names to evaluate.')
    parser.add_argument('--task_names', nargs='+', default=["code", "math", "grow"], help='List of task names to evaluate.')

    # --- Parallelism Control ---
    parser.add_argument('--max-workers', type=int, default=32, help='Maximum number of parallel processes.')

    args = parser.parse_args()

    all_commands = generate_commands(args)
    
    if not all_commands:
        print("No commands were generated based on the provided arguments. Exiting.")
        exit()
        
    print(f"\nGenerated {len(all_commands)} commands to run with up to {args.max_workers} parallel workers.")
    
    # --- Execute Commands in Parallel ---
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in all_commands]
        
        for future in as_completed(futures):
            print(future.result())

    print("\n🎉 All evaluation commands have been executed.")