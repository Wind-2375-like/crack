# smart_runner.py
import argparse
import itertools
import subprocess
import time
import re
import os
from collections import deque
from threading import Thread

# Third-party libraries - install with: pip install rich pynvml
from rich.live import Live
from rich.table import Table
from rich.console import Console
import pynvml

# --- Configuration: TUNE THESE VALUES FOR YOUR SETUP ---

# 1. Manual VRAM overrides (in GB).
#    Use this to set a SPECIFIC memory value for a model size if the
#    default 4x calculation is inaccurate.
#    The key should be the model size (e.g., "7b", "1.5b").
MODEL_MEMORY_OVERRIDES = {
    "7b": 30,      # Example: Override for a specific 7b model that needs 30GB
    "default": 12, # Fallback for models where size cannot be parsed (e.g., "bert-base")
}

# 2. Safety margin. Always leave this much VRAM free (in GB).
GPU_HEADROOM_GB = 4

# --- End Configuration ---

console = Console()

class GpuMonitor:
    # ... (This class is unchanged) ...
    """A thread-safe class to monitor GPU stats."""
    def __init__(self):
        self.free_gb = 0
        self.total_gb = 0
        self._stop = False
        self._thread = Thread(target=self._run, daemon=True)
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.device_count == 0:
                console.print("[bold red]Error: No NVIDIA GPU found.[/bold red]")
                exit(1)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming single GPU
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.total_gb = info.total / (1024**3)
            self._thread.start()
        except pynvml.NVMLError as error:
            console.print(f"[bold red]Failed to initialize NVML: {error}[/bold red]")
            exit(1)

    def _run(self):
        """Continuously update GPU stats in the background."""
        while not self._stop:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.free_gb = info.free / (1024**3)
            except pynvml.NVMLError:
                self.free_gb = 0 # Assume no memory if query fails
            time.sleep(2) # Update every 2 seconds

    def stop(self):
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

def generate_dashboard_table(jobs, gpu_monitor: GpuMonitor) -> Table:
    """Creates the Rich table for the live dashboard."""
    title = f"Knowledge Experiment Dashboard (GPU Free: {gpu_monitor.free_gb:.2f} / {gpu_monitor.total_gb:.2f} GB)"
    table = Table(title=title, expand=True)
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("VRAM (GB)", justify="right", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("PID", style="dim")
    table.add_column("Retries", style="dim")
    table.add_column("Log File", style="blue")

    for i, job in enumerate(jobs):
        status_style = ""
        if "Running" in job['status']: status_style = "green"
        elif "Failed" in job['status']: status_style = "red"
        elif "Success" in job['status']: status_style = "bright_green"

        # --- THIS IS THE FIX ---
        # Conditionally format the status text only if a style is set.
        status_text = job['status']
        if status_style:
            status_text = f"[{status_style}]{status_text}[/{status_style}]"
        # --- END OF FIX ---

        table.add_row(
            str(i + 1),
            job['task'],
            job['model'],
            f"{job['required_mem']:.1f}",
            status_text, # Use the corrected variable here
            str(job['process'].pid) if job['process'] else "-",
            str(job['retries']),
            job['log_path']
        )
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run knowledge probe scripts intelligently with GPU monitoring.")
    parser.add_argument('--model_names', nargs='+', default=["llama-3.2-1b", "llama-3.2-3b", "qwen-2.5-1.5b", "qwen-2.5-3b"], help='List of model names.')
    parser.add_argument('--task_names', nargs='+', default=["code", "math", "grow"], help='List of task names.')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel processes.')
    parser.add_argument('--max-retries', type=int, default=50, help='Max retries for ANY failed process.')
    
    args = parser.parse_args()

    # --- Setup ---
    # We no longer need to create the 'logs' dir here, it will be handled in the loop.
    pending_jobs = generate_commands(args)
    all_jobs = list(pending_jobs) # A persistent list for the dashboard
    running_jobs = []

    if not pending_jobs:
        console.print("[yellow]No commands to run. Exiting.[/yellow]")
        exit()

    gpu_monitor = GpuMonitor()
    
    console.print(f"\n[bold]Starting scheduler for {len(pending_jobs)} jobs. Max workers: {args.max_workers}[/bold]")
    console.print(f"Total GPU Memory: {gpu_monitor.total_gb:.2f} GB | Safety Headroom: {GPU_HEADROOM_GB} GB")
    # You can add this line for your own debugging to be 100% sure
    console.print(f"Script CWD: [dim]{os.getcwd()}[/dim]")
    time.sleep(2)

    try:
        with Live(generate_dashboard_table(all_jobs, gpu_monitor), refresh_per_second=2, console=console) as live:
            while pending_jobs or running_jobs:
                # 1. Check for finished jobs
                for job in running_jobs[:]: # Iterate on a copy
                    if job['process'].poll() is not None: # Process has finished
                        running_jobs.remove(job)
                        if job['process'].returncode == 0:
                            job['status'] = "✅ Success"
                        else:
                            if job['retries'] < args.max_retries:
                                job['retries'] += 1
                                job['status'] = f"🔁 Retrying ({job['retries']}/{args.max_retries})"
                                pending_jobs.appendleft(job) # Add to front of queue
                            else:
                                job['status'] = f"❌ Failed (Code: {job['process'].returncode})"

                # 2. Try to launch a new job
                free_slots = args.max_workers - len(running_jobs)
                if pending_jobs and free_slots > 0:
                    next_job = pending_jobs[0] # Peek at the next job
                    required_mem = next_job["required_mem"]
                    
                    if gpu_monitor.free_gb > (required_mem + GPU_HEADROOM_GB):
                        job_to_run = pending_jobs.popleft()
                        job_to_run['status'] = "🚀 Running"
                        
                        # --- THIS IS THE FIX ---
                        # Ensure the specific directory for this log file exists right before writing.
                        log_directory = os.path.dirname(job_to_run['log_path'])
                        if log_directory:
                            os.makedirs(log_directory, exist_ok=True)
                        # --- END OF FIX ---
                        
                        with open(job_to_run['log_path'], 'w') as log_file:
                            p = subprocess.Popen(
                                job_to_run['cmd'],
                                stdout=log_file,
                                stderr=subprocess.STDOUT
                            )
                        job_to_run['process'] = p
                        running_jobs.append(job_to_run)
                        
                        if required_mem > 15:
                            time.sleep(15)

                # 3. Update dashboard and sleep
                live.update(generate_dashboard_table(all_jobs, gpu_monitor))
                time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. Terminating running processes...[/bold yellow]")
        for job in running_jobs:
            if job['process']:
                job['process'].terminate()
    finally:
        gpu_monitor.stop()
        console.print("\n🎉 All experiments complete.")