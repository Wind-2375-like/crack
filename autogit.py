import subprocess
import sys
import time
from datetime import datetime

def run_command(command):
    """Executes a shell command and prints its output in real-time."""
    try:
        # The command is provided as a list of arguments for security and correctness.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # Read and print the output line by line as it comes in.
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output)
        
        # Wait for the process to terminate and get the return code.
        process.wait()
        
        # For git commit, "nothing to commit" is a success, not an error.
        full_output = "".join(output_lines)
        if "nothing to commit" in full_output and command[1] == "commit":
            return True

        if process.returncode != 0:
            print(f"\n❌ Error: Command '{' '.join(command)}' failed with return code {process.returncode}.")
            return False
            
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: The command '{command[0]}' was not found.")
        print("Please ensure Git is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def run_git_workflow():
    """Runs the sequence of Git commands once."""
    # Generate a unique commit message with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"results: {timestamp}"
    
    print(f"🚀 Starting Git automation workflow at {timestamp}...")
    
    # Step 1: Git Pull
    print("\n--- Step 1: Pulling latest changes from origin main ---")
    if not run_command(["git", "pull", "origin", "main"]):
        print("\nWorkflow step failed. Will retry in 1 hour.")
        return
        
    # Step 2: Git Add
    print("\n--- Step 2: Adding all changes to the staging area ---")
    if not run_command(["git", "add", "."]):
        print("\nWorkflow step failed. Will retry in 1 hour.")
        return

    # Step 3: Git Commit
    print(f"\n--- Step 3: Committing changes with message: '{commit_message}' ---")
    if not run_command(["git", "commit", "-m", commit_message]):
        # This check is now more robust inside run_command
        print("\nNote: Commit may have failed or there were no changes to commit.")
        
    # Step 4: Git Push
    print("\n--- Step 4: Pushing changes to origin main ---")
    if not run_command(["git", "push", "origin", "main"]):
        print("\nWorkflow step failed. Will retry in 1 hour.")
        return
        
    print("\n✅ Git workflow completed successfully!")

def main():
    """Main function to run the Git workflow in a loop."""
    while True:
        try:
            run_git_workflow()
            
            # Wait for 1 hour (3600 seconds) before the next run
            sleep_duration = 7200
            print(f"\n--- Sleeping for 15 minutes. Next run at approximately {datetime.fromtimestamp(time.time() + sleep_duration).strftime('%H:%M:%S')} ---")
            time.sleep(sleep_duration)

        except KeyboardInterrupt:
            print("\n\n🛑 Script stopped by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            print("Restarting the loop after a short delay...")
            time.sleep(60) # Wait for 1 minute before retrying after a major error

if __name__ == "__main__":
    main()

