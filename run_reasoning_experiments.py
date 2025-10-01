import argparse
import itertools
import subprocess
import time
import re
import os
from collections import deque
import pickle
from threading import Thread

# Third-party libraries - install with: pip install rich pynvml
from rich.live import Live
from rich.table import Table
from rich.console import Console

# IPython imports are now optional
try:
    from IPython.display import display, clear_output
    IS_IPYTHON = True
except ImportError:
    IS_IPYTHON = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# --- Configuration (can be the same as the other script) ---
MODEL_MEMORY_OVERRIDES = {
    "7b": 1,
    "11b": 1,
    "default": 1,
}
GPU_HEADROOM_GB = 0
# --- End Configuration ---

console = Console()

# --- All helper classes and functions are reused without changes ---
# (GpuMonitor, get_required_memory)
class GpuMonitor:
    """A thread-safe class to monitor GPU stats."""
    def __init__(self):
        self.is_active = False; self.free_gb = 0; self.total_gb = 0; self._stop = False
        if not HAS_PYNVML: return
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.device_count == 0: pynvml.nvmlShutdown(); return
            handle = pynvml.nvmlDeviceGetHandleByIndex(0); info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.total_gb = info.total / (1024**3); self.is_active = True
            self._thread = Thread(target=self._run, daemon=True); self._thread.start()
        except pynvml.NVMLError: pass
    def _run(self):
        while not self._stop:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0); info = pynvml.nvmlDeviceGetMemoryInfo(handle); self.free_gb = info.free / (1024**3)
            except pynvml.NVMLError: self.free_gb = 0
            time.sleep(2)
    def stop(self):
        if self.is_active and not self._stop: self._stop = True; pynvml.nvmlShutdown()

def get_required_memory(model_name: str) -> float:
    size_key_match = re.search(r'(\d+(\.\d+)?b)', model_name.lower()); numerical_match = re.search(r'(\d+(\.\d+)?)b', model_name.lower())
    if size_key_match:
        size_key = size_key_match.group(1)
        if size_key in MODEL_MEMORY_OVERRIDES: return MODEL_MEMORY_OVERRIDES[size_key]
    if numerical_match:
        numerical_size = float(numerical_match.group(1)); return numerical_size ** 0
    return MODEL_MEMORY_OVERRIDES.get("default", 0)

def generate_commands(args):
    """Generates and sorts reasoning experiment commands with corrected logic."""
    commands = []
    base_script = ["python", "scripts/experiment/knowledge_injection.py"]
    cpu_models_set = set(args.cpu_only_models or [])
    has_gpu_jobs = False

    console.print("Generating and classifying reasoning commands...")

    if args.inject_knowledge:
        methods_to_run = args.methods
        scopes_to_run = args.knowledge_aggregation_scopes
    else:
        console.print("[yellow]Knowledge injection is OFF. Forcing method='base' and scope=1 for baseline runs.[/yellow]")
        methods_to_run = ['base']
        scopes_to_run = [1]
        
    try:
        sample_raw_path = f'data/{args.task_names[0]}/test_{args.data_size}.pkl'
        with open(sample_raw_path, 'rb') as f:
            total_data_size = len(pickle.load(f))
    except (FileNotFoundError, IndexError):
        console.print(f"[bold red]Warning: Could not find sample data file to determine total size. Using data_size arg.[/bold red]")
        total_data_size = args.data_size

    for task, model, method, scope in itertools.product(
        args.task_names, args.model_names, methods_to_run, scopes_to_run
    ):
        is_cpu = model in cpu_models_set
        required_mem = 0 if is_cpu else get_required_memory(model)
        if not is_cpu:
            has_gpu_jobs = True

        cmd_list = base_script + [
            "--data_size", str(args.data_size), # Pass data_size to worker
            "--task_name", task,
            "--model_name", model,
            "--method", method,
            "--knowledge_aggregation_scope", str(scope)
        ]
        
        if args.inject_knowledge:
            cmd_list.append("--inject_knowledge")
        
        log_suffix = "injected" if args.inject_knowledge else "original"
        log_path = f"logs/reasoning_{task}_{model}_{method}_{scope}_{log_suffix}.log"
        
        output_dir = f'data/eval_results/{task}/injection/'
        output_filename = f"{'original' if not args.inject_knowledge else method}_{args.data_size}_{model}_{scope}.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        status = "⏳ Pending"
        if os.path.exists(output_path):
            try:
                with open(output_path, 'rb') as f:
                    existing_data = pickle.load(f)
                if len(existing_data) >= total_data_size:
                    status = "✅ Skipped (Complete)"
            except (pickle.UnpicklingError, EOFError):
                 pass
        
        commands.append({
            "cmd": cmd_list, "task": task, "model": model, "method": method, "scope": scope,
            "required_mem": required_mem, "is_cpu": is_cpu,
            "status": status, "process": None, "retries": 0,
            "log_path": log_path
        })
        
        job_type = "[bold blue]CPU[/bold blue]" if is_cpu else f"[bold cyan]{required_mem:.1f} GB[/bold cyan]"
        console.print(f"  - [magenta]{task}[/magenta]/[yellow]{model}[/yellow]/[green]{method}[/green]/[bold]scope={scope}[/bold] -> {job_type}")

    commands.sort(key=lambda c: c['required_mem'])
    return deque(commands), has_gpu_jobs

# --- MODIFIED: `generate_dashboard_table` with new columns ---
def generate_dashboard_table(jobs, gpu_monitor: GpuMonitor) -> Table:
    """Creates the Rich table for the live dashboard."""
    title = "Reasoning Experiment Dashboard"
    if gpu_monitor and gpu_monitor.is_active:
        title += f" (GPU Free: {gpu_monitor.free_gb:.2f} / {gpu_monitor.total_gb:.2f} GB)"

    table = Table(title=title, expand=True)
    table.add_column("ID", justify="right")
    table.add_column("Task", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("Method", style="green")
    table.add_column("Scope", style="white")
    table.add_column("Type", justify="right", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Log File", style="blue")

    for i, job in enumerate(jobs):
        status_style, status_text = "", job['status']
        if "Running" in status_text: status_style = "green"
        elif "Failed" in status_text or "Retrying" in status_text: status_style = "red"
        elif "Success" in status_text: status_style = "bright_green"
        if status_style: status_text = f"[{status_style}]{status_text}[/{status_style}]"
        
        job_type = "[blue]CPU[/blue]" if job['is_cpu'] else f"{job['required_mem']:.1f} GB"

        table.add_row(
            str(i + 1), job['task'], job['model'], job['method'], str(job['scope']), job_type,
            status_text, job['log_path']
        )
    return table

# --- The `run_scheduler` function is reused without changes ---
def run_scheduler(args, is_notebook_run=False):
    # This function is long, but its internal logic is identical to the other script.
    os.makedirs("logs", exist_ok=True)
    pending_jobs_all, has_gpu_jobs = generate_commands(args)
    all_jobs = list(pending_jobs_all) # Keep all for dashboard view
    # The actual queue only contains jobs that are not already complete
    pending_jobs = deque([job for job in all_jobs if job['status'] != "✅ Skipped (Complete)"])
    running_jobs, failed_jobs_waiting = [], []
    if not pending_jobs: console.print("[yellow]No commands to run. Exiting.[/yellow]"); return
    gpu_monitor = GpuMonitor() if has_gpu_jobs else None
    console.print(f"\n[bold]Starting scheduler for {len(pending_jobs)} jobs. Max workers: {args.max_workers}[/bold]")
    if gpu_monitor and gpu_monitor.is_active:
        console.print(f"GPU monitoring is ACTIVE. Total Memory: {gpu_monitor.total_gb:.2f} GB | Safety Headroom: {GPU_HEADROOM_GB} GB")
    else:
        console.print("GPU monitoring is INACTIVE. No GPU jobs requested or GPU not found.")
    time.sleep(2)
    def main_loop_logic():
        for job in running_jobs[:]:
            if job['process'].poll() is not None:
                running_jobs.remove(job)
                if job['process'].returncode == 0: job['status'] = "✅ Success"
                else:
                    if job['retries'] < args.max_retries:
                        job['retries'] += 1; job['status'] = f"🔁 Failed, waiting {args.retry_delay}s to retry ({job['retries']}/{args.max_retries})"
                        failed_jobs_waiting.append((time.time() + args.retry_delay, job))
                    else: job['status'] = f"❌ Failed Permanently (Code: {job['process'].returncode})"
        for finish_time, job in failed_jobs_waiting[:]:
            if time.time() >= finish_time:
                failed_jobs_waiting.remove((finish_time, job)); job['status'] = f"⏳ Pending Retry"; pending_jobs.append(job)
        if pending_jobs and len(running_jobs) < args.max_workers:
            next_job = pending_jobs[0]; can_launch = False
            if next_job['is_cpu']: can_launch = True
            elif gpu_monitor and gpu_monitor.is_active and gpu_monitor.free_gb > (next_job["required_mem"] + GPU_HEADROOM_GB): can_launch = True
            if can_launch:
                job_to_run = pending_jobs.popleft(); job_to_run['status'] = "🚀 Running"
                log_dir = os.path.dirname(job_to_run['log_path'])
                if log_dir: os.makedirs(log_dir, exist_ok=True)
                with open(job_to_run['log_path'], 'w') as log_file:
                    p = subprocess.Popen(job_to_run['cmd'], stdout=log_file, stderr=subprocess.STDOUT)
                job_to_run['process'] = p; running_jobs.append(job_to_run)
                if not job_to_run['is_cpu'] and job_to_run['required_mem'] > 15: time.sleep(15)
        return not (pending_jobs or running_jobs or failed_jobs_waiting)
    if is_notebook_run:
        if not IS_IPYTHON: console.print("[bold red]Error: Notebook mode selected, but IPython is not available.[/bold red]"); return
        try:
            while True:
                if main_loop_logic(): break
                clear_output(wait=True); display(generate_dashboard_table(all_jobs, gpu_monitor)); time.sleep(2)
        finally:
            if gpu_monitor: gpu_monitor.stop()
            clear_output(wait=True); display(generate_dashboard_table(all_jobs, gpu_monitor)); console.print("\n🎉 All experiments complete.")
    else: # Terminal mode
        try:
            with Live(generate_dashboard_table(all_jobs, gpu_monitor), refresh_per_second=2, console=console) as live:
                while True:
                    if main_loop_logic(): break
                    live.update(generate_dashboard_table(all_jobs, gpu_monitor)); time.sleep(1)
        finally:
            if gpu_monitor: gpu_monitor.stop()
            console.print("\n🎉 All experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning experiments intelligently for CPU and GPU.")
    
    # --- Experiment Arguments ---
    parser.add_argument('--task_names', nargs='+', default=["grow", "code", "math"], help='List of task names.')
    parser.add_argument('--model_names', nargs='+', required=True, help='List of ALL model names to run.')
    parser.add_argument('--methods', nargs='+', default=["base"], help='List of methods to test (e.g., base, rome, append_t).')
    parser.add_argument('--knowledge_aggregation_scopes', nargs='+', type=int, default=[1, 10, 100, 500], help='List of aggregation scopes to test.')
    parser.add_argument('--data_size', type=int, default=500, help="Size of the dataset to process in each experiment.")
    parser.add_argument('--inject-knowledge', action='store_true', help='Flag to run with knowledge injection. If not set, runs original baselines.')
    parser.add_argument('--cpu-only-models', nargs='+', help='List of model names that are CPU-only (e.g., OpenAI models).')
    
    # --- Scheduler Arguments ---
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel processes.')
    parser.add_argument('--max-retries', type=int, default=10000, help='Max retries for ANY failed process.')
    parser.add_argument('--retry-delay', type=int, default=60, help='Seconds to wait before re-queueing a failed job.')
    
    args = parser.parse_args()
    run_scheduler(args)