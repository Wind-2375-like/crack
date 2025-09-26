import argparse
import itertools
import subprocess
import time
import re
import os
from collections import deque
from threading import Thread
from rich.console import Console
from IPython.display import clear_output


# Third-party libraries - install with: pip install rich pynvml
from rich.live import Live
from rich.table import Table
from rich.console import Console
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# --- Configuration ---
MODEL_MEMORY_OVERRIDES = {
    "7b": 30,
    "11b": 30,
    "default": 12,
}
GPU_HEADROOM_GB = 4
# --- End Configuration ---

console = Console()

class GpuMonitor:
    """A thread-safe class to monitor GPU stats."""
    def __init__(self):
        self.is_active = False
        self.free_gb = 0
        self.total_gb = 0
        self._stop = False

        if not HAS_PYNVML:
            console.print("[yellow]Warning: `pynvml` not found. Running in CPU-only mode.[/yellow]")
            return

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.device_count == 0:
                console.print("[yellow]Warning: No NVIDIA GPU found. Running in CPU-only mode.[/yellow]")
                pynvml.nvmlShutdown()
                return
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.total_gb = info.total / (1024**3)
            self.is_active = True
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()
        except pynvml.NVMLError as error:
            console.print(f"[bold red]Failed to initialize NVML: {error}. Running in CPU-only mode.[/bold red]")

    def _run(self):
        while not self._stop:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.free_gb = info.free / (1024**3)
            except pynvml.NVMLError:
                self.free_gb = 0
            time.sleep(2)

    def stop(self):
        if self.is_active and not self._stop:
            self._stop = True
            pynvml.nvmlShutdown()

def get_required_memory(model_name: str) -> float:
    """
    Estimates VRAM in GB for a model with the new logic.
    1. Checks for a manual override in MODEL_MEMORY_OVERRIDES.
    2. If not found, calculates memory as 4 * size_in_billions_of_params.
    3. Falls back to a default value if size cannot be parsed.
    """
    # Try to parse size key (e.g., "3b") and numerical value (e.g., 3.0)
    size_key_match = re.search(r'(\d+(\.\d+)?b)', model_name.lower())
    numerical_match = re.search(r'(\d+(\.\d+)?)b', model_name.lower())

    if size_key_match:
        size_key = size_key_match.group(1)
        # 1. Check for manual override first
        if size_key in MODEL_MEMORY_OVERRIDES:
            return MODEL_MEMORY_OVERRIDES[size_key]

    if numerical_match:
        # 2. Apply default 4x calculation if no override was found
        numerical_size = float(numerical_match.group(1))
        return 4.0 * numerical_size

    # 3. Fallback to default if size cannot be parsed from the name
    return MODEL_MEMORY_OVERRIDES.get("default", 12)

def generate_commands(args):
    """Generates and sorts experiment commands from smallest to largest model."""
    commands = []
    base_script = ["python", "scripts/experiment/knowledge_probe.py"]

    console.print("Generating and estimating memory for commands...")
    for task, model in itertools.product(args.task_names, args.model_names):
        required_mem = get_required_memory(model)
        cmd_list = base_script + ["--task_name", task, "--model_name", model]
        commands.append({
            "cmd": cmd_list,
            "task": task,
            "model": model,
            "required_mem": required_mem,
            "status": "⏳ Pending",
            "process": None,
            "retries": 0,
            "log_path": f"logs/probe_{task}_{model}.log"
        })
        console.print(f"  - [magenta]{task}[/magenta] / [yellow]{model}[/yellow] -> Estimated [bold cyan]{required_mem:.1f} GB[/bold cyan]")

    # Sort commands by their calculated memory requirement
    commands.sort(key=lambda c: c['required_mem'])
    return deque(commands)

def generate_commands(args):
    """Generates and sorts experiment commands, identifying CPU vs GPU jobs."""
    commands = []
    base_script = ["python", "scripts/experiment/knowledge_probe.py"]
    cpu_models_set = set(args.cpu_only_models or [])
    has_gpu_jobs = False

    console.print("Generating and classifying commands...")
    for task, model in itertools.product(args.task_names, args.model_names):
        is_cpu = model in cpu_models_set
        required_mem = 0 if is_cpu else get_required_memory(model)
        if not is_cpu:
            has_gpu_jobs = True
            
        cmd_list = base_script + ["--task_name", task, "--model_name", model]
        commands.append({
            "cmd": cmd_list, "task": task, "model": model,
            "required_mem": required_mem, "is_cpu": is_cpu,
            "status": "⏳ Pending", "process": None, "retries": 0,
            "log_path": f"logs/probe_{task}_{model}.log"
        })
        
        job_type = "[bold blue]CPU[/bold blue]" if is_cpu else f"[bold cyan]{required_mem:.1f} GB[/bold cyan]"
        console.print(f"  - [magenta]{task}[/magenta] / [yellow]{model}[/yellow] -> Type: {job_type}")

    commands.sort(key=lambda c: c['required_mem'])
    return deque(commands), has_gpu_jobs

def generate_dashboard_table(jobs, gpu_monitor: GpuMonitor) -> Table:
    """Creates the Rich table for the live dashboard."""
    title = "Knowledge Experiment Dashboard"
    if gpu_monitor and gpu_monitor.is_active:
        title += f" (GPU Free: {gpu_monitor.free_gb:.2f} / {gpu_monitor.total_gb:.2f} GB)"

    table = Table(title=title, expand=True)
    table.add_column("ID", justify="right")
    table.add_column("Task", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("Type", justify="right", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("PID", style="dim")
    table.add_column("Log File", style="blue")

    for i, job in enumerate(jobs):
        status_style, status_text = "", job['status']
        if "Running" in status_text: status_style = "green"
        elif "Failed" in status_text: status_style = "red"
        elif "Success" in status_text: status_style = "bright_green"
        if status_style: status_text = f"[{status_style}]{status_text}[/{status_style}]"
        
        job_type = "[blue]CPU[/blue]" if job['is_cpu'] else f"{job['required_mem']:.1f} GB"

        # --- THIS IS THE FIX ---
        # Conditionally format the status text only if a style is set.
        status_text = job['status']
        if status_style:
            status_text = f"[{status_style}]{status_text}[/{status_style}]"
        # --- END OF FIX ---

        table.add_row(
            str(i + 1), job['task'], job['model'], job_type,
            status_text, str(job['process'].pid) if job['process'] else "-",
            job['log_path']
        )
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run knowledge probe scripts intelligently for CPU and GPU.")
    parser.add_argument('--model_names', nargs='+', required=True, help='List of ALL model names to run.')
    parser.add_argument('--cpu-only-models', nargs='+', help='List of model names that are CPU-only (e.g., OpenAI models).')
    parser.add_argument('--task_names', nargs='+', default=["code", "math", "grow"], help='List of task names.')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel processes.')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries for ANY failed process.')
    parser.add_argument('--retry-delay', type=int, default=60, help='Seconds to wait before re-queueing a failed job.')

    args = parser.parse_args()
    os.makedirs("logs", exist_ok=True)

    pending_jobs, has_gpu_jobs = generate_commands(args)
    all_jobs = list(pending_jobs)
    running_jobs = []
    failed_jobs_waiting = []

    if not pending_jobs:
        console.print("[yellow]No commands to run. Exiting.[/yellow]"); exit()

    gpu_monitor = GpuMonitor() if has_gpu_jobs else None

    console.print(f"\n[bold]Starting scheduler for {len(pending_jobs)} jobs. Max workers: {args.max_workers}[/bold]")
    if gpu_monitor and gpu_monitor.is_active:
        console.print(f"GPU monitoring is ACTIVE. Total Memory: {gpu_monitor.total_gb:.2f} GB | Safety Headroom: {GPU_HEADROOM_GB} GB")
    else:
        console.print("GPU monitoring is INACTIVE. No GPU jobs requested or GPU not found.")
    time.sleep(2)

    # --- MODIFIED BLOCK FOR COLAB ---
    # Replace the `with Live(...)` context manager with a `try/finally` block
    try:
        while pending_jobs or running_jobs or failed_jobs_waiting:
            # 1. Check for finished jobs
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

            # 2. Check waiting jobs
            for finish_time, job in failed_jobs_waiting[:]:
                if time.time() >= finish_time:
                    failed_jobs_waiting.remove((finish_time, job))
                    job['status'] = f"⏳ Pending Retry"
                    pending_jobs.append(job)

            # 3. Try to launch a new job
            if pending_jobs and len(running_jobs) < args.max_workers:
                next_job = pending_jobs[0]
                can_launch = False
                if next_job['is_cpu']:
                    can_launch = True
                elif gpu_monitor and gpu_monitor.is_active:
                    if gpu_monitor.free_gb > (next_job["required_mem"] + GPU_HEADROOM_GB):
                        can_launch = True
                
                if can_launch:
                    job_to_run = pending_jobs.popleft()
                    job_to_run['status'] = "🚀 Running"
                    log_dir = os.path.dirname(job_to_run['log_path'])
                    if log_dir: os.makedirs(log_dir, exist_ok=True)
                    with open(job_to_run['log_path'], 'w') as log_file:
                        p = subprocess.Popen(
                            job_to_run['cmd'], stdout=log_file, stderr=subprocess.STDOUT
                        )
                    job_to_run['process'] = p
                    running_jobs.append(job_to_run)
                    if not job_to_run['is_cpu'] and job_to_run['required_mem'] > 15:
                        time.sleep(15)

            # 4. Clear the output and print the updated dashboard
            clear_output(wait=True)
            console.print(generate_dashboard_table(all_jobs, gpu_monitor))
            time.sleep(2) # Refresh every 2 seconds

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. Terminating running processes...[/bold yellow]")
        for job in running_jobs:
            if job['process']:
                job['process'].terminate()
    finally:
        if gpu_monitor: gpu_monitor.stop()
        console.print("\n🎉 All experiments complete.")
    # --- END OF MODIFIED BLOCK ---
