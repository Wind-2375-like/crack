import argparse
import itertools
import subprocess
import time
import os
from collections import deque
from threading import Thread

# Third-party libraries
from rich.live import Live
from rich.table import Table
from rich.console import Console

console = Console()

def get_output_file_path(task_name, model_name, inject_knowledge, method, scope, data_size=500):
    """
    Constructs the final output file path for a given evaluation job.
    This is the file we check to see if a job is already complete.
    """
    output_dir = f'data/eval_results/{task_name}/injection_evaluated/'
    if inject_knowledge:
        filename = f"{method}_{data_size}_{model_name}_{scope}.pkl"
    else:
        # Base 'original' runs are treated as method='base' and scope=1 for consistency
        filename = f"original_{data_size}_{model_name}_1.pkl"
    return os.path.join(output_dir, filename)

def generate_commands(args):
    """
    Generates evaluation commands as structured job dictionaries,
    checking for existing results to enable resuming.
    """
    jobs = []
    base_script = ["python", "scripts/evaluation/reasoning_evaluation.py"]
    os.makedirs("logs", exist_ok=True) # Ensure log directory exists

    console.print("Generating evaluation commands and checking for completed jobs...")

    # --- 1. Base evaluations (no knowledge injection) ---
    if args.run_base_eval:
        console.print("[bold cyan]--> Generating BASE evaluation jobs...[/bold cyan]")
        for task, model in itertools.product(args.task_names, args.model_names):
            output_path = get_output_file_path(task, model, inject_knowledge=False, method='base', scope=1, data_size=args.data_size)
            cmd_list = base_script + ["--task_name", task, "--model_name", model, "--data_size", str(args.data_size)]
            
            status = "⏳ Pending"
            if os.path.exists(output_path) and not args.overwrite:
                status = "✅ Skipped (Complete)"

            log_path = f"logs/eval_{task}_{model}_original.log"
            jobs.append({
                "cmd": cmd_list, "task": task, "model": model, "method": "original", "scope": "1",
                "status": status, "process": None, "retries": 0, "log_path": log_path
            })

    # --- 2. Evaluations with knowledge injection ---
    if args.run_inject_eval:
        console.print("[bold cyan]--> Generating INJECTION evaluation jobs...[/bold cyan]")
        for task, model, method, scope in itertools.product(args.task_names, args.model_names, args.method_names, args.knowledge_aggregation_scopes):
            output_path = get_output_file_path(task, model, inject_knowledge=True, method=method, scope=scope, data_size=args.data_size)
            cmd_list = base_script + [
                "--inject_knowledge", "--knowledge_aggregation_scope", str(scope),
                "--method", method, "--task_name", task, "--model_name", model,
                "--data_size", str(args.data_size)
            ]

            status = "⏳ Pending"
            if os.path.exists(output_path) and not args.overwrite:
                status = "✅ Skipped (Complete)"
            
            log_path = f"logs/eval_{task}_{model}_{method}_{scope}_injected.log"
            jobs.append({
                "cmd": cmd_list, "task": task, "model": model, "method": method, "scope": str(scope),
                "status": status, "process": None, "retries": 0, "log_path": log_path
            })
    
    # --- 3. Evaluations with --method all ---
    if args.run_method_all_eval:
        console.print("[bold cyan]--> Generating '--method all' evaluation jobs...[/bold cyan]")
        for task, model in itertools.product(args.task_names, args.model_names):
            # As per your original script, this specifically targets scope=1
            scope = 1
            method = 'all'
            output_path = get_output_file_path(task, model, inject_knowledge=True, method=method, scope=scope, data_size=args.data_size)
            cmd_list = base_script + [
                "--inject_knowledge", "--knowledge_aggregation_scope", str(scope),
                "--method", method, "--task_name", task, "--model_name", model,
                "--data_size", str(args.data_size)
            ]

            status = "⏳ Pending"
            if os.path.exists(output_path) and not args.overwrite:
                status = "✅ Skipped (Complete)"

            log_path = f"logs/eval_{task}_{model}_{method}_{scope}_injected.log"
            jobs.append({
                "cmd": cmd_list, "task": task, "model": model, "method": method, "scope": str(scope),
                "status": status, "process": None, "retries": 0, "log_path": log_path
            })

    return deque(jobs)

def generate_dashboard_table(jobs) -> Table:
    """Creates the Rich table for the live dashboard."""
    table = Table(title="Reasoning Evaluation Dashboard", expand=True)
    table.add_column("ID", justify="right")
    table.add_column("Task", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("Method", style="green")
    table.add_column("Scope", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Log File", style="blue")

    for i, job in enumerate(jobs):
        status_style, status_text = "", job['status']
        if "Running" in status_text: status_style = "green"
        elif "Failed" in status_text or "Retrying" in status_text: status_style = "red"
        elif "Success" in status_text: status_style = "bright_green"
        if status_style: status_text = f"[{status_style}]{status_text}[/{status_style}]"
        
        table.add_row(
            str(i + 1), job['task'], job['model'], job['method'], str(job['scope']),
            status_text, job['log_path']
        )
    return table

def run_scheduler(args):
    """The main scheduler loop to manage and run evaluation jobs."""
    all_jobs_list = list(generate_commands(args))
    # The actual queue only contains jobs that are not already skipped
    pending_jobs = deque([job for job in all_jobs_list if "Skipped" not in job['status']])
    
    if not pending_jobs:
        console.print("[bold green]✅ All evaluation results already exist. Nothing to do.[/bold green]")
        return

    running_jobs, failed_jobs_waiting = [], []
    console.print(f"\n[bold]Starting scheduler for {len(pending_jobs)} evaluation jobs. Max workers: {args.max_workers}[/bold]")
    time.sleep(2)

    try:
        with Live(generate_dashboard_table(all_jobs_list), refresh_per_second=2, console=console) as live:
            while pending_jobs or running_jobs or failed_jobs_waiting:
                # 1. Check for completed jobs
                for job in running_jobs[:]:
                    if job['process'].poll() is not None:
                        running_jobs.remove(job)
                        if job['process'].returncode == 0:
                            job['status'] = "✅ Success"
                        else:
                            if job['retries'] < args.max_retries:
                                job['retries'] += 1
                                job['status'] = f"🔁 Failed, waiting {args.retry_delay}s to retry ({job['retries']}/{args.max_retries})"
                                failed_jobs_waiting.append((time.time() + args.retry_delay, job))
                            else:
                                job['status'] = f"❌ Failed Permanently (Code: {job['process'].returncode})"
                
                # 2. Re-queue jobs that are ready for a retry
                for finish_time, job in failed_jobs_waiting[:]:
                    if time.time() >= finish_time:
                        failed_jobs_waiting.remove((finish_time, job))
                        job['status'] = "⏳ Pending Retry"
                        pending_jobs.append(job) # Add to the end for fairness
                
                # 3. Launch new jobs if workers are available
                while pending_jobs and len(running_jobs) < args.max_workers:
                    job_to_run = pending_jobs.popleft()
                    job_to_run['status'] = "🚀 Running"
                    
                    # Open log file and start the subprocess
                    with open(job_to_run['log_path'], 'w') as log_file:
                        p = subprocess.Popen(job_to_run['cmd'], stdout=log_file, stderr=subprocess.STDOUT)
                    
                    job_to_run['process'] = p
                    running_jobs.append(job_to_run)
                
                live.update(generate_dashboard_table(all_jobs_list))
                time.sleep(1)
    finally:
        # Final update to the table to show the last statuses
        console.print(generate_dashboard_table(all_jobs_list))
        console.print("\n🎉 All evaluation jobs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning evaluation scripts in parallel with a live dashboard.")
    
    # --- Model and Task Arguments ---
    parser.add_argument('--model_names', nargs='+', default=["llama-3.2-1b", "llama-3.2-3b", "qwen-2.5-1.5b", "qwen-2.5-3b"], help='List of model names to evaluate.')
    parser.add_argument('--task_names', nargs='+', default=["code", "math", "grow"], help='List of task names to evaluate.')
    parser.add_argument('--method_names', nargs='+', default=["base", "mello"], help='List of method names to evaluate.')
    parser.add_argument('--data_size', type=int, default=500, help='Data size for testing.')
    
    # --- Evaluation Type Control ---
    parser.add_argument('--no-base-eval', action='store_false', dest='run_base_eval', help='Do not run base evaluations.')
    parser.add_argument('--no-inject-eval', action='store_false', dest='run_inject_eval', help='Do not run knowledge injection evaluations.')
    parser.add_argument('--no-method-all-eval', action='store_false', dest='run_method_all_eval', help='Do not run --method all evaluations.')
    parser.add_argument('--overwrite', action='store_true', help='Force run all commands, overwriting existing results.')
    
    # --- Knowledge Injection Specifics ---
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100, 500], help='List of knowledge aggregation scopes.')

    # --- Scheduler Arguments ---
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel processes.')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries for a failed process.')
    parser.add_argument('--retry-delay', type=int, default=10, help='Seconds to wait before re-queueing a failed job.')

    args = parser.parse_args()

    # The new main execution function
    run_scheduler(args)