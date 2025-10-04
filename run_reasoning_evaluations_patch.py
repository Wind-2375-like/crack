# run_reasoning_evaluations_patch.py
import argparse
import itertools
import subprocess
import time
import os
import pickle
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.console import Console

console = Console()

# --- Re-used helper functions from your main script ---
def get_pickle_file_length(file_path: str) -> int:
    if not os.path.exists(file_path): return 0
    try:
        with open(file_path, 'rb') as f: data = pickle.load(f)
        return len(data) if isinstance(data, (list, tuple)) else 0
    except (pickle.UnpicklingError, EOFError, IndexError): return 0

def get_output_file_path(task_name, model_name, inject_knowledge, method, scope, data_size=500):
    output_dir = f'data/eval_results/{task_name}/injection_evaluated/'
    if inject_knowledge: filename = f"{method}_{data_size}_{model_name}_{scope}.pkl"
    else: filename = f"original_{data_size}_{model_name}_1.pkl"
    return os.path.join(output_dir, filename)

def generate_patch_jobs(args):
    """Identifies only the jobs that need patching and creates commands for the patch worker."""
    jobs = []
    base_script = ["python", "scripts/evaluation/reasoning_evaluation_patch_worker.py"]
    os.makedirs("logs/patch_logs", exist_ok=True)
    console.print("[bold yellow]Scanning for corrupted evaluation files to patch...[/bold yellow]")

    # Combine all parameter iterations
    all_params = []
    if args.run_base_eval:
        for task, model in itertools.product(args.task_names, args.model_names):
            all_params.append({'task': task, 'model': model, 'method': 'base', 'scope': 1, 'inject': False})
    if args.run_inject_eval:
        for task, model, method, scope in itertools.product(args.task_names, args.model_names, args.method_names, args.knowledge_aggregation_scopes):
            all_params.append({'task': task, 'model': model, 'method': method, 'scope': scope, 'inject': True})
    if args.run_method_all_eval:
        for task, model in itertools.product(args.task_names, args.model_names):
            all_params.append({'task': task, 'model': model, 'method': 'all', 'scope': 1, 'inject': True})

    for params in all_params:
        task, model, method, scope, inject = params['task'], params['model'], params['method'], params['scope'], params['inject']
        
        # Determine input path from the 'injection' folder
        input_dir = f'data/eval_results/{task}/injection/'
        if not inject: input_filename = f"original_{args.data_size}_{model}_1.pkl"
        else: input_filename = f"{method}_{args.data_size}_{model}_{scope}.pkl"
        input_path = os.path.join(input_dir, input_filename)

        # Determine output path from the 'injection_evaluated' folder
        output_path = get_output_file_path(task, model, inject, method, scope, args.data_size)

        input_len = get_pickle_file_length(input_path)
        output_len = get_pickle_file_length(output_path)

        status = "✅ Healthy"
        if input_len == 0:
            status = "⚠️ No Input"
        elif 0 < output_len < input_len:
            status = "💔 Corrupted (Needs Patch)"
        
        if status == "💔 Corrupted (Needs Patch)":
            cmd_list = base_script + [
                "--input_path", input_path,
                "--output_path", output_path,
                "--raw_data_path", f'data/{task}/test_{args.data_size}.pkl',
                # Pass necessary args to the worker
                "--task_name", task,
                "--model_name", model, 
                "--evaluate_model_name", args.evaluate_model_name,
            ]
            log_path = f"logs/patch_logs/patch_{task}_{model}_{method}_{scope}.log"
            jobs.append({"cmd": cmd_list, "display": f"{task}/{model}/{method}/{scope}", "status": "⏳ Pending Patch", "process": None, "log_path": log_path})

    return deque(jobs)

def generate_dashboard_table(jobs) -> Table:
    table = Table(title="Reasoning Evaluation Patch Dashboard", expand=True)
    table.add_column("ID", justify="right")
    table.add_column("Job", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Log File", style="blue")
    for i, job in enumerate(jobs):
        status_text = job['status']
        if "Running" in status_text: status_text = f"[green]{status_text}[/green]"
        elif "Failed" in status_text: status_text = f"[red]{status_text}[/red]"
        elif "Success" in status_text: status_text = f"[bright_green]{status_text}[/bright_green]"
        table.add_row(str(i + 1), job['display'], status_text, job['log_path'])
    return table

def run_patcher(args):
    """Scheduler to run only the patching jobs."""
    all_jobs = list(generate_patch_jobs(args))
    if not all_jobs:
        console.print("[bold green]✅ No corrupted files found to patch. All evaluation files are healthy![/bold green]")
        return

    pending_jobs = deque(all_jobs)
    running_jobs = []
    console.print(f"\n[bold]Found {len(pending_jobs)} files to patch. Max workers: {args.max_workers}[/bold]")
    time.sleep(2)

    with Live(generate_dashboard_table(all_jobs), refresh_per_second=2, console=console) as live:
        while pending_jobs or running_jobs:
            for job in running_jobs[:]:
                if job['process'].poll() is not None:
                    running_jobs.remove(job)
                    job['status'] = "✅ Success" if job['process'].returncode == 0 else f"❌ Failed (Code: {job['process'].returncode})"
            
            while pending_jobs and len(running_jobs) < args.max_workers:
                job_to_run = pending_jobs.popleft()
                job_to_run['status'] = "🚀 Running"
                with open(job_to_run['log_path'], 'w') as log_file:
                    p = subprocess.Popen(job_to_run['cmd'], stdout=log_file, stderr=subprocess.STDOUT)
                job_to_run['process'] = p
                running_jobs.append(job_to_run)
            
            live.update(generate_dashboard_table(all_jobs))
            time.sleep(1)
    
    console.print("\n🎉 Patching process complete.")
    console.print(generate_dashboard_table(all_jobs))

if __name__ == "__main__":
    # Uses almost the same arguments as the original script for consistency
    parser = argparse.ArgumentParser(description="Patch corrupted reasoning evaluation files.")
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names to check.')
    parser.add_argument('--task_names', nargs='+', required=True, help='List of task names to check.')
    parser.add_argument('--method_names', nargs='+', default=["base", "mello"], help='List of method names to check.')
    parser.add_argument('--data_size', type=int, default=500, help='Data size for testing.')
    parser.add_argument('--evaluate_model_name', type=str, default="gpt-5-mini-2025-08-07", help="Model name for the evaluation judge.")
    parser.add_argument('--no-base-eval', action='store_false', dest='run_base_eval')
    parser.add_argument('--no-inject-eval', action='store_false', dest='run_inject_eval')
    parser.add_argument('--no-method-all-eval', action='store_false', dest='run_method_all_eval')
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100, 500])
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel patch processes.')
    args = parser.parse_args()
    run_patcher(args)