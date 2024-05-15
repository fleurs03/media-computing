import subprocess

params = [["-t", f"{i}"] for i in range(1, 5)]

def run_script(param):
    try:
        subprocess.run(["python", "completion.py"] + param)
    except Exception as e:
        print(f"Error running script with parameters {param}: {e}")

# use multi-thread for parallel computing
# from concurrent.futures import ThreadPoolExecutor
# with ThreadPoolExecutor() as executor:
#     executor.map(run_script, params)

# use single thread
for param in params:
    run_script(param)
