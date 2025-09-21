import subprocess
import os
import time

def translate_model_name(model_name: str) -> str:
    """
    Translate a model name to a more user-friendly format.
    
    Args:
        model_name (str): The original model name.
    
    Returns:
        str: The translated model name.
    """
    # Define a mapping of model names to their user-friendly versions
    model_name_mapping = {
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-1b-turbo": "meta-llama/Llama-3.2-1B-Instruct-Turbo",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.2-3b-turbo": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.2-90b-turbo": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "qwen-2.5-1.5b": 'Qwen/Qwen2.5-1.5B-Instruct',
        "qwen-2.5-3b": 'Qwen/Qwen2.5-3B-Instruct',
        "qwen-2.5-7b": 'Qwen/Qwen2.5-7B-Instruct',
        "qwen-2.5-14b": 'Qwen/Qwen2.5-14B-Instruct',
        "qwen-3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen-3-1.7b-thinking": "Qwen/Qwen3-1.7B-Thinking",
        "qwen-3-4b": "Qwen/Qwen3-4B",
        "qwen-3-4b-thinking": "Qwen/Qwen3-4B-Thinking",
        "qwen-3-8b": "Qwen/Qwen3-8B",
        "qwen-3-8b-thinking": "Qwen/Qwen3-8B-Thinking",
        "olmo-2-1b": "allenai/OLMo-2-0425-1B-Instruct",
        "olmo-2-7b": "allenai/OLMo-2-1124-7B-Instruct",
    }
    
    # Return the translated model name or the original if not found
    return model_name_mapping.get(model_name, model_name)

def run_command(command, max_retries=10, retry_delay_minutes=10, log_dir="logs"):
    """
    Executes a command with real-time logging and a retry mechanism for CUDA OOM errors.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Identify experiment type from the script path for clearer logging
    script_path = command[1]
    experiment_type = os.path.splitext(os.path.basename(script_path))[0]
    
    # 2. Create a more descriptive and predictable log file name
    log_file_name_parts = [experiment_type]
    for flag in ['--inject_knowledge', '--task_name', '--model_name', '--method', '--knowledge_aggregation_scope']:
        try:
            index = command.index(flag)
            log_file_name_parts.append(command[index + 1])
        except (ValueError, IndexError):
            continue # Flag not in command or has no value
    log_file_name = f"{'_'.join(log_file_name_parts)}.log"
    log_path = os.path.join(log_dir, log_file_name)

    # --- Retry Loop ---
    for attempt in range(max_retries + 1):
        # 3. Print the status and log path at the very beginning of an attempt
        print(f"🚀 [{experiment_type}] Attempt {attempt + 1}/{max_retries + 1}: Starting {' '.join(command)} | Log: {log_path}")
        
        try:
            # 4. Use a file handle to stream stdout/stderr directly to the log file in real-time
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT, # Combine both streams into one file
                    text=True,
                    check=False
                )

            # 5. After the run, read the log file back to check for the OOM error
            with open(log_path, 'r') as log_file:
                oom_detected = "CUDA out of memory" in log_file.read()

            # --- Handle results ---
            if oom_detected:
                print(f"⚠️ [{experiment_type}] OOM detected for: {' '.join(command)}")
                if attempt < max_retries:
                    print(f"   -> Waiting {retry_delay_minutes} minutes before retry...")
                    time.sleep(retry_delay_minutes * 60)
                    continue  # Go to the next attempt
                else:
                    return f"❌ [{experiment_type}] Failed (OOM) after {max_retries + 1} attempts: {' '.join(command)}"

            elif result.returncode != 0:
                return f"❌ [{experiment_type}] Failed (non-OOM error): {' '.join(command)} | See log: {log_path}"

            else: # Success
                return f"✅ [{experiment_type}] Finished successfully: {' '.join(command)}"

        except Exception as e:
            return f"❌ [{experiment_type}] Critical failure to start process: {' '.join(command)} with error: {e}"

    return f"❌ [{experiment_type}] Unknown failure after all retries for: {' '.join(command)}"