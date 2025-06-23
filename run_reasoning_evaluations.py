import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from utils.helpers import run_command
import os

def get_output_file_path(task_name, model_name, inject_knowledge, method, scope, data_size=100):
    """
    Constructs the output file path based on evaluation parameters.
    """
    output_dir = f'data/eval_results/{task_name}/injection_evaluated/'
    default_scope_for_original = '1' 
    if inject_knowledge:
        filename = f"{method}_{data_size}_depth_4_{model_name}_{scope}.pkl"
    else:
        filename = f"original_{data_size}_depth_4_{model_name}_{default_scope_for_original}.pkl"
    return os.path.join(output_dir, filename)

def generate_commands(args):
    """Generates all the evaluation commands based on the provided arguments."""
    commands = []
    base_script = ["python", "scripts/evaluation/reasoning_evaluation.py"]

    # --- 1. Base evaluations (no knowledge injection) ---
    if args.run_base_eval:
        print("Generating base evaluation commands...")
        for task, model in itertools.product(args.task_names, args.model_names):
            output_path = get_output_file_path(task, model, inject_knowledge=False, method='base', scope=1)
            if not args.overwrite and os.path.exists(output_path):
                print(f"SKIPPING (result exists): {output_path}")
                continue
            commands.append(base_script + ["--task_name", task, "--model_name", model])

    # --- 2. Evaluations with knowledge injection ---
    if args.run_inject_eval:
        print("Generating knowledge injection evaluation commands...")
        for task, model, method, scope in itertools.product(args.task_names, args.model_names, args.method_names, args.knowledge_aggregation_scopes):
            output_path = get_output_file_path(task, model, inject_knowledge=True, method=method, scope=scope)
            if not args.overwrite and os.path.exists(output_path):
                print(f"SKIPPING (result exists): {output_path}")
                continue
            commands.append(
                base_script + [
                    "--inject_knowledge",
                    "--knowledge_aggregation_scope", str(scope),
                    "--task_name", task,
                    "--model_name", model
                ]
            )
    
    # --- 3. Evaluations with --method all ---
    if args.run_method_all_eval:
        print("Generating '--method all' evaluation commands...")
        # This specifically targets scope=1 as per your requirement
        for task, model in itertools.product(args.task_names, args.model_names):
            output_path = get_output_file_path(task, model, inject_knowledge=True, method='all', scope=1)
            if not args.overwrite and os.path.exists(output_path):
                print(f"SKIPPING (result exists): {output_path}")
                continue
            commands.append(
                base_script + [
                    "--inject_knowledge",
                    "--knowledge_aggregation_scope", "1",
                    "--method", "all",
                    "--task_name", task,
                    "--model_name", model
                ]
            )

    return commands

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning evaluation scripts in parallel.")
    
    # --- Model and Task Arguments ---
    parser.add_argument('--model_names', nargs='+', default=["llama-3.2-1b", "llama-3.2-3b", "qwen-2.5-1.5b", "qwen-2.5-3b"], help='List of model names to evaluate.')
    parser.add_argument('--task_names', nargs='+', default=["code", "math", "grow"], help='List of task names to evaluate.')
    parser.add_argument('--method_names', nargs='+', default=["base", "mello"], help='List of method names to evaluate.')
    
    # --- Evaluation Type Control ---
    parser.add_argument('--no-base-eval', action='store_false', dest='run_base_eval', help='Do not run base evaluations.')
    parser.add_argument('--no-inject-eval', action='store_false', dest='run_inject_eval', help='Do not run knowledge injection evaluations.')
    parser.add_argument('--no-method-all-eval', action='store_false', dest='run_method_all_eval', help='Do not run --method all evaluations.')
    parser.add_argument('--overwrite', action='store_true', help='Force run all commands, overwriting existing results.')
    
    # --- Knowledge Injection Specifics ---
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100], help='List of knowledge aggregation scopes.')

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