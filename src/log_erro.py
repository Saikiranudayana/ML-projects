
import subprocess

# List of Python scripts to execute
scripts = ["logger.py", "utils.py", "exception.py"]

for script in scripts:
    try:
        print(f"Executing {script}...")
        subprocess.run(["python", script], check=True)
        print(f"Finished executing {script}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script}: {e}")
