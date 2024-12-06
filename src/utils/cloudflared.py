import subprocess
import re
import os
from dotenv import set_key, load_dotenv
process_list={}
def run_cloudflared(port=8765):
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True
    )
    process_list[process.pid] = (process)
    print(f"process.stdout.readline: {process.stdout.readline}")
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  
        match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if match:
            print(f"exposed url in run_cloudflared: {match}")

            return match.group(1)
    
    stderr = process.stderr.read()
    if stderr:
        print("Error:", stderr)

def set_url_in_env(url):
    env_file = '.env'
    load_dotenv(env_file)

    url =url.replace("https://","")
    print(f"final url to set in .env: {url}")
    set_key(env_file, "APPLICATION_BASE_URL", url)
    print(f"URL set in .env: {url}")
    print(f'after setup: {os.getenv("APPLICATION_BASE_URL", "")}')
    # Manually update the environment variable
    os.environ["APPLICATION_BASE_URL"] = url
