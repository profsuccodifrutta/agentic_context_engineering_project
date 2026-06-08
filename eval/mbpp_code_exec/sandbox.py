import docker
import tempfile
import os
import time
from typing import Dict, Union

class DockerSandbox:
    def __init__(
        self,
        image: str = "python:3.10-slim",
        mem_limit: str = "128m",
        cpu_quota: int = 50000,
        timeout: int = 10
    ):
        self.client = docker.from_env()
        self.image = image
        self.mem_limit = mem_limit
        self.cpu_quota = cpu_quota
        self.timeout = timeout

        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            print(f"Pulling Docker image {image}...")
            self.client.images.pull(image)

    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting from LLM outputs."""
        code = code.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        if code.startswith("python"):
            code = code[6:].strip()
            
        return code.strip()

    def execute_code(self, code_string: str) -> Dict[str, Union[str, int, None]]:
        """
        Executes Python code safely, utilizing combined log streams for maximum reliability.
        """
        code = self._clean_code(code_string)
        container = None
        container_path = "/tmp/script.py"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        os.chmod(temp_path, 0o644)

        try:
            container = self.client.containers.run(
                image=self.image,
                command=["python", "-u", container_path], # FIX: Force unbuffered execution
                environment={"PYTHONUNBUFFERED": "1"},
                volumes={
                    temp_path: {"bind": container_path, "mode": "ro"}
                },
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
                user="1000:1000",
                read_only=True,
                network_disabled=True,
                mem_limit=self.mem_limit,
                cpu_period=100000,
                cpu_quota=self.cpu_quota,
                pids_limit=64,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                detach=True
            )

            start_time = time.time()
            timed_out = False
            
            # Polling Timeout Control
            while True:
                try:
                    container.reload()
                except docker.errors.NotFound:
                    break 

                if container.status not in ["created", "running"]:
                    break 

                if time.time() - start_time > self.timeout:
                    timed_out = True
                    try:
                        container.kill()
                    except docker.errors.APIError:
                        pass
                    break
                
                time.sleep(0.1)

            # FIX: Simple, robust log capture BEFORE wait
            output = ""
            try:
                # Dropping demux=True prevents the NoneType tuple bugs entirely
                output = container.logs(stdout=True, stderr=True).decode("utf-8").strip()
            except Exception as e:
                output = f"Failed to fetch logs: {str(e)}"

            # Wait logic
            exit_code = -1
            status = "error"
            
            if timed_out:
                status = "timeout"
                exit_code = 124
                if not output:
                    output = f"Execution exceeded the {self.timeout} second timeout limit."
            else:
                try:
                    result = container.wait()
                    exit_code = result.get("StatusCode", -1)
                    status = "success" if exit_code == 0 else "error"
                except (docker.errors.APIError, docker.errors.NotFound):
                    pass

            return {
                "status": status,
                "stdout": output if status == "success" else None,
                "stderr": output if status != "success" else None,
                "exit_code": exit_code
            }

        except Exception as e:
            return {
                "status": "fatal",
                "stdout": None,
                "stderr": str(e),
                "exit_code": -1
            }

        finally:
            if container:
                try:
                    container.remove(force=True)
                except docker.errors.APIError:
                    pass
            
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass