import subprocess
import os
import signal
import time
import requests
import psutil

class ProcessManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.processes = {} # {server_type: subprocess.Popen}
        os.makedirs(self.log_dir, exist_ok=True)

    def start_server(self, server_type, model_name, port, gpu_util, max_len):
        """Starts a vLLM server as a subprocess. Idempotent: returns success if already running."""
        if self.is_running(server_type):
            return True, f"{server_type} is already running."

        log_file_path = os.path.join(self.log_dir, f"{server_type}.log")
        # Overwrite with header for new start
        with open(log_file_path, "w") as f:
            f.write(f"--- Starting {server_type} at {time.ctime()} ---\n")
        
        log_file = open(log_file_path, "a")

        # Determine the correct python executable
        # Use the local vllm_engine symlink if it exists
        vllm_python = "./vllm_engine/bin/python3"
        if not os.path.exists(vllm_python):
            vllm_python = "python3"

        cmd = [
            vllm_python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(max_len),
            "--trust-remote-code"
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Create a new process group to kill children
            )
            self.processes[server_type] = process
            return True, f"Started {server_type} on port {port}"
        except Exception as e:
            return False, str(e)

    def stop_server(self, server_type):
        """Stops a server and cleans up its process group."""
        if server_type in self.processes:
            process = self.processes[server_type]
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                del self.processes[server_type]
                return True, f"Stopped {server_type}"
            except Exception as e:
                return False, str(e)
        
        # Fallback: Find and kill by port if not in our tracking
        return self._kill_by_port(server_type)

    def _kill_by_port(self, server_type):
        port = 8000 if server_type == "vlm" else 8001
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        os.kill(proc.pid, signal.SIGTERM)
                        return True, f"Killed ghost process for {server_type} on port {port}"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, f"No process found for {server_type}"

    def is_running(self, server_type):
        """Checks if the subprocess is alive or port is occupied."""
        if server_type in self.processes:
            if self.processes[server_type].poll() is None:
                return True
        
        # Port check fallback using connection matching
        port = 8000 if server_type == "vlm" else 8001
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def get_status(self, server_type):
        """Returns Health Status: STOPPED, STARTING, HEALTHY, ERROR."""
        if not self.is_running(server_type):
            return "STOPPED"
        
        port = 8000 if server_type == "vlm" else 8001
        try:
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if response.status_code == 200:
                return "HEALTHY"
        except:
            pass
            
        return "STARTING"

    def clear_logs(self, server_type):
        """Truncates the log file."""
        log_file_path = os.path.join(self.log_dir, f"{server_type}.log")
        if os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write(f"--- Log Cleared at {time.ctime()} ---\n")
            return True
        return False
