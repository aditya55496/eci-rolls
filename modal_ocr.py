"""GE 2024 OCR on Modal CPUs, next to the runner pool (owner 2026-11-23: "use 1200 modal cores").

Each container runs PROCS copies of ocr_task.py (the same task as on the runners), each with WORKERS OCR processes. They claim
eci-rolls-up-ocr items from the hub and upload through it with the hub server token `modal` (~/setup/178; runs show repository
server/modal). Same OS and Tesseract packages as the runners (Ubuntu 24.04 apt), same family commit (COMMIT), so the readings
match the runners'. A copy stops when ocr_task finds no work twice in a row; the container ends when all its copies stopped.

  modal run modal_ocr.py::check                      one container: CPU/memory it sees (no OCR)
  modal deploy modal_ocr.py && python modal_ocr.py spawn 75   75 containers x CPU cores (1,200 cores at CPU=16), no client needed
  modal app stop eci-ocr                             stop all (claimed items are freed by the hub's stale rule and stolen)
"""
import os
import subprocess
import sys
import time

import modal

COMMIT = os.environ.get("ECI_COMMIT", "9740b69")
CPU = float(os.environ.get("ECI_MODAL_CPU", "16"))           # Modal cores (1 core = 2 vCPU)
PROCS = int(os.environ.get("ECI_MODAL_PROCS", "8"))            # ocr_task copies per container
WORKERS = int(os.environ.get("ECI_MODAL_WORKERS", "4"))        # OCR processes per copy: PROCS x WORKERS = 2 x CPU (one per vCPU)
MEM_MB = int(os.environ.get("ECI_MODAL_MEM_MB", str(PROCS * 4800)))   # measured: ~0.7 GB parent + ~0.9 GB per OCR process
TOKEN_FILE = os.path.expanduser("~/.config/eci/modal-hub.env")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("git", "ca-certificates", "tesseract-ocr", "tesseract-ocr-hin", "tesseract-ocr-eng")
    .run_commands(f"git clone -q https://github.com/s1d-labs/eci-rolls /app && git -C /app checkout -q {COMMIT}",
                  "pip install -q -r /app/requirements.txt 'scrape-hub-client @ git+https://github.com/s1d-labs/runner@sdk-stable#subdirectory=sdk'")
)
app = modal.App("eci-ocr", image=image)


def _secret():
    if not modal.is_local():
        return modal.Secret.from_dict({})
    env = dict(l.strip().split("=", 1) for l in open(TOKEN_FILE) if "=" in l)
    return modal.Secret.from_dict({k: env[k] for k in ("SCRAPE_HUB_TOKEN", "SCRAPE_HUB_URL")})


@app.function(cpu=CPU, memory=MEM_MB, timeout=24 * 3600, secrets=[_secret()], retries=modal.Retries(max_retries=3, initial_delay=10.0))
def worker(idx: int, n: int, hours: float = 20.0):
    import threading
    end = time.time() + hours * 3600
    host = os.environ.get("MODAL_TASK_ID", "modal")[-12:]
    stats = {"runs": 0, "done": 0}

    def slot(j):
        empty = 0
        while time.time() < end and empty < 2:
            env = dict(os.environ, PYTHONUNBUFFERED="1", SCRAPE_POOL_PROJECT="eci-rolls-up-ocr",
                       SCRAPE_MAX_HOURS=f"{max(0.05, min(1.0, (end - time.time()) / 3600)):.3f}",
                       SCRAPE_JOB_INDEX=str(idx * PROCS + j), SCRAPE_TOTAL_JOBS=str(n * PROCS), ECI_OCR_WORKERS=str(WORKERS),
                       SCRAPE_RUN_ID=f"modal-{host}-{j}-{int(time.time())}", HOME="/tmp")
            t = time.time()
            r = subprocess.run([sys.executable, "ocr_task.py"], cwd="/app", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            stats["runs"] += 1
            if r.returncode:
                print(f"slot {j}: ocr_task exit {r.returncode}: {r.stdout[-500:]}", flush=True)
                time.sleep(30)
            empty = empty + 1 if time.time() - t < 120 else 0      # a run that ends at once found no work
            if empty:
                time.sleep(60)

    ts = [threading.Thread(target=slot, args=(j,)) for j in range(PROCS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return {"idx": idx, **stats}


@app.function(cpu=CPU, memory=MEM_MB, timeout=600)
def probe():
    def read(p):
        try:
            return open(p).read().strip()
        except OSError:
            return None
    return {"cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)), "cpu.max": read("/sys/fs/cgroup/cpu.max"),
            "memory.max": read("/sys/fs/cgroup/memory.max"), "tesseract": subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0],
            "commit": subprocess.run(["git", "-C", "/app", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip(),
            "hin": subprocess.run(["dpkg-query", "-W", "tesseract-ocr-hin"], capture_output=True, text=True).stdout.strip()}


@app.local_entrypoint()
def main(n: int = 1, hours: float = 20.0):
    print(f"{n} containers x {CPU} cores, {PROCS} copies x {WORKERS} OCR processes, commit {COMMIT}")
    for r in worker.map(range(n), kwargs={"n": n, "hours": hours}, return_exceptions=True):
        print(r)


@app.local_entrypoint()
def check():
    print(probe.remote())


if __name__ == "__main__":
    # Deployed app + spawned calls: they run on without any local client. (`modal run --detach` with .map() stopped 3 min after
    # its local client ended: 2026-11-23 19:29.)  modal deploy modal_ocr.py && python modal_ocr.py spawn 75 [hours]
    if sys.argv[1:2] == ["spawn"]:
        n, hours = int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
        f = modal.Function.from_name("eci-ocr", "worker")
        calls = [f.spawn(i, n, hours) for i in range(n)]
        print(f"spawned {len(calls)} containers:", " ".join(c.object_id for c in calls[:3]), "...")
