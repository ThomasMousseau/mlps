"""Capture an Nsight Systems report and a tlparse report for one FA4 step.

The output directory is named ``11-11-26T15h33-<commit>-dirty`` or ``clean``.
The two files to open are ``mlps-11-11-26T15h33.nsys-rep`` and
``mlps-11-11-26T15h33.html``. tlparse also writes the FX-graph pages that
the html links to; those pages have to stay beside the html.

Run from the repo root, with Nsight Systems on ``PATH``:

    module load cuda/13.2
    uv run --extra flash-attn-4 python studies/flash-attn-4/capture_traces.py

Docs: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.observability.html
"""

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parents[1]
INNER_ENV = "MLPS_CAPTURE_INNER"
# First steps compile and capture a CUDA graph. The report starts after that.
WARMUP_STEPS = 3
PROFILED_STEPS = 3
LOCAL_TZ = ZoneInfo("America/Toronto")


def stamp_now() -> str:
    """Local time as ``11-11-26T15h33``."""
    return datetime.now(LOCAL_TZ).strftime("%m-%d-%yT%Hh%M")


def git_token() -> str:
    """``<12-char sha>-dirty`` or ``<12-char sha>-clean``."""
    sha = subprocess.run(
        ["git", "-C", str(REPO_DIR), "rev-parse", "--short=12", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_DIR), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    state = "dirty" if status.strip() else "clean"
    return f"{sha}-{state}"


def nsys_bin() -> str:
    """The ``nsys`` executable, or an error that names the module to load."""
    found = shutil.which("nsys")
    if found is None:
        msg = "nsys is not on PATH. Load it with: module load cuda/13.2"
        raise SystemExit(msg)
    return found


def tlparse_command() -> list[str]:
    """``tlparse`` on PATH, otherwise ``uvx --from tlparse``."""
    found = shutil.which("tlparse")
    if found is not None:
        return [found]
    uvx = shutil.which("uvx")
    if uvx is None:
        msg = "tlparse is not on PATH, and uvx is not available to fetch it."
        raise SystemExit(msg)
    return [uvx, "--from", "tlparse", "tlparse"]


def run_profiled() -> int:
    """Warm up one compiled attention step, then record a few forward-backward steps.

    Returns:
        Process status. ``1`` when this GPU cannot run FA4.
    """
    sys.path.insert(0, str(STUDY_DIR))
    import torch
    from examples.attn import Attention, hardware_supports_fa4

    if not hardware_supports_fa4():
        print("FA4 requires a Hopper (sm90) GPU, or newer.")
        return 1
    torch.manual_seed(0)
    batch, seqlen = 8, 1024
    model = Attention(dim=1024, heads=16)
    compiled = torch.compile(model, fullgraph=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step() -> None:
        x = torch.randn(batch, seqlen, model.dim, device="cuda", dtype=torch.bfloat16)
        torch.cuda.nvtx.range_push("attention forward")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(x)
        loss = out.float().square().mean()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("attention backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize()
    # nsys is launched with --capture-range=cudaProfilerApi, so only this
    # window is written into the report.
    torch.cuda.profiler.start()
    try:
        for _ in range(PROFILED_STEPS):
            step()
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    return 0


def publish_tlparse(trace_dir: Path, out_dir: Path, html_name: str) -> None:
    """Write the tlparse report into ``out_dir`` and rename its entry page."""
    with tempfile.TemporaryDirectory(prefix="mlps-tlparse-") as raw:
        report = Path(raw)
        subprocess.run(
            [*tlparse_command(), str(trace_dir), "--latest", "-o", str(report)],
            check=True,
        )
        index = report / "index.html"
        if not index.is_file():
            msg = f"tlparse wrote no index.html in {report}"
            raise SystemExit(msg)
        for path in report.iterdir():
            shutil.move(str(path), out_dir / path.name)
    (out_dir / "index.html").rename(out_dir / html_name)


def capture() -> int:
    """Run the profiled step under nsys and write both reports."""
    stamp = stamp_now()
    out_dir = STUDY_DIR / "traces" / f"{stamp}-{git_token()}"
    if out_dir.exists():
        msg = f"{out_dir} already exists. Wait a minute and run the capture again."
        raise SystemExit(msg)
    out_dir.mkdir(parents=True)
    html_name = f"mlps-{stamp}.html"
    report_base = out_dir / f"mlps-{stamp}"

    with tempfile.TemporaryDirectory(prefix="mlps-torch-trace-") as trace:
        env = os.environ.copy()
        env[INNER_ENV] = "1"
        env["TORCH_TRACE"] = trace
        env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            [
                nsys_bin(),
                "profile",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--cuda-graph-trace=node",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--force-overwrite=true",
                f"--output={report_base}",
                sys.executable,
                str(Path(__file__).resolve()),
            ],
            check=False,
            cwd=STUDY_DIR,
            env=env,
        )
        # nsys may also drop a sidecar next to the report. Remove it before
        # the html, which shares the same stem, is written.
        for extra in out_dir.glob(f"{report_base.name}.*"):
            if extra.suffix != ".nsys-rep":
                extra.unlink()
        publish_tlparse(Path(trace), out_dir, html_name)
    report = report_base.with_suffix(".nsys-rep")
    print(out_dir)
    print(report)
    print(out_dir / html_name)
    return completed.returncode


def main() -> int:
    """Dispatch the nsys child, or capture from a normal launch."""
    if os.environ.get(INNER_ENV) == "1":
        return run_profiled()
    return capture()


if __name__ == "__main__":
    raise SystemExit(main())
