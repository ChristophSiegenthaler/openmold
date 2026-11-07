"""Running CalculiX and reading its .frd results."""
import os, subprocess
from pathlib import Path
import numpy as np


def run(job, cwd, ccx="ccx", threads=1):
    """Run a CalculiX job. Keep threads at 1: see "Solver threads" in the README."""
    n = str(threads)
    env = dict(os.environ, OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, CCX_NPROC_EQUATION_SOLVER=n, CCX_NPROC_RESULTS=n)
    log = Path(cwd) / f"{job}.log"
    with open(log, "w") as out:
        code = subprocess.run([ccx, "-i", job], cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
    text = log.read_text(errors="ignore")
    if code != 0 or "*ERROR" in text or "Job finished" not in text:
        tail = "\n".join(l for l in text.splitlines()[-200:] if "ERROR" in l) or "\n".join(text.splitlines()[-8:])
        raise RuntimeError(f"CalculiX job {job} failed (exit code {code}). See {log}\n{tail}")


def read_frd(path, name="NDTEMP"):
    """Result blocks called name, as a list of (time, node ids, values). Streams the file to keep memory low."""
    out = []
    with open(path) as lines:
        for line in lines:
            if not line.startswith("  100C"):
                continue
            t = float(line[12:24])
            block = next(lines).split()[1]
            ids, vals = [], []
            for s in lines:
                if s.startswith(" -3"):
                    break
                if block == name and s.startswith(" -1"):
                    ids.append(int(s[3:13]))
                    v = s[13:].rstrip("\n")
                    vals.append([float(v[k:k + 12]) for k in range(0, len(v), 12)])
            if block == name:
                out.append((t, np.array(ids), np.array(vals)))
    return out
