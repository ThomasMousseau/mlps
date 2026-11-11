"""The fake-op patch must leave FA4 as one graph, forward and backward.

Each pass is counted on its own:

- ``graphs``: FX graphs Dynamo hands the backend, one call per graph.
- ``breaks``: graph breaks Dynamo reports in ``counters["graph_break"]``, one
  per time it stopped tracing mid-frame, ran the offending code eagerly, and
  resumed in a new frame. Counted separately from ``graphs`` rather than as
  ``graphs - 1``, which is how ``torch._dynamo.explain`` derives it.

The forward is ``torch.compile``'d. The backward is captured by compiled
autograd, which hands the whole autograd graph to ``torch.compile``: AOT
Autograd's ``bw_compiler`` would only see the backward of what the forward
traced, so a kernel the forward broke around would run its backward unseen.
``fullgraph`` stays off so a break is counted instead of raised.

Each implementation is measured in a fresh interpreter with FA4's persistent
CuTe cache and Inductor's caches off: a warm CuTe JIT cache takes a different
path through FA4's Python, which hides the forward breaks of the raw control.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch
from flash_attn.cute import flash_attn_func
from mlps_shared.imports import ImportPath
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import counters

with ImportPath(Path(__file__).parents[1]):
    from examples import attn

hardware_supports_fa4 = attn.hardware_supports_fa4

AttnFn = Callable[[torch.Tensor], torch.Tensor]
Impl = Literal["patched", "raw"]

B, S, H, D = 3, 250, 4, 64
FWD_OP = "fa4_attn.fa4_fwd.default"
BWD_OP = "fa4_attn.fa4_bwd.default"


def _raw_fa4(qkv: torch.Tensor) -> torch.Tensor:
    """The control: FA4's own op on unbound q, k, v, with no fake op."""
    q, k, v = qkv.unbind(2)
    out, _ = flash_attn_func(q, k, v)
    return out


IMPLS: dict[Impl, AttnFn] = {"patched": attn.fa4_attn, "raw": _raw_fa4}


@dataclass(frozen=True)
class PassCensus:
    """FX graphs compiled for one pass, the breaks that split it, and the ops
    each graph calls."""

    graphs: int
    breaks: int
    ops: tuple[frozenset[str], ...]

    def ops_anywhere(self) -> frozenset[str]:
        """Every op called by any graph of the pass."""
        return frozenset().union(*self.ops)


@dataclass(frozen=True)
class Census:
    """The forward and backward census of one implementation."""

    forward: PassCensus
    backward: PassCensus


def _recording_backend(
    ops: list[frozenset[str]],
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable[..., object]]:
    """A backend that records each graph's ops, then runs it eagerly."""

    def backend(
        gm: torch.fx.GraphModule, _example_inputs: list[torch.Tensor]
    ) -> Callable[..., object]:
        ops.append(
            frozenset(
                str(node.target)
                for node in gm.graph.nodes
                if node.op == "call_function"
            )
        )
        return gm.forward

    return backend


def _graph_breaks() -> int:
    return sum(counters["graph_break"].values())


def _census_in_this_process(fn: AttnFn) -> Census:
    """Compiles ``fn`` and counts one forward and one backward."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    qkv = torch.randn(
        B, S, 3, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dout = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    fwd_ops: list[frozenset[str]] = []
    counters.clear()
    compiled = torch.compile(fn, backend=_recording_backend(fwd_ops), dynamic=False)
    out = compiled(qkv)
    forward = PassCensus(len(fwd_ops), _graph_breaks(), tuple(fwd_ops))

    bwd_ops: list[frozenset[str]] = []
    counters.clear()
    bwd_compiler = torch.compile(backend=_recording_backend(bwd_ops), dynamic=False)
    with compiled_autograd._enable(bwd_compiler, dynamic=False):
        out.backward(dout)
    backward = PassCensus(len(bwd_ops), _graph_breaks(), tuple(bwd_ops))

    assert qkv.grad is not None, "the backward never reached qkv"
    assert qkv.grad.isfinite().all()
    return Census(forward, backward)


def _to_json(census: Census) -> str:
    return json.dumps(
        {
            name: {
                "graphs": p.graphs,
                "breaks": p.breaks,
                "ops": [sorted(ops) for ops in p.ops],
            }
            for name, p in (("forward", census.forward), ("backward", census.backward))
        }
    )


def _from_json(text: str) -> Census:
    def one(p: dict) -> PassCensus:
        return PassCensus(
            p["graphs"], p["breaks"], tuple(frozenset(ops) for ops in p["ops"])
        )

    data = json.loads(text)
    return Census(one(data["forward"]), one(data["backward"]))


@functools.cache
def _census(impl: Impl) -> Census:
    """Counts ``impl`` in a fresh interpreter, so no cache is warm."""
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "census.json"
        env = os.environ | {
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
            "TORCHINDUCTOR_CACHE_DIR": str(Path(tmp) / "inductor"),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        }
        proc = subprocess.run(  # noqa: S603  # this file, with a fixed impl
            [sys.executable, __file__, impl, str(result)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert proc.returncode == 0, f"{impl} census failed:\n{proc.stderr}"
        return _from_json(result.read_text())


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestPatched:
    """`fa4_attn` compiles to one forward graph and one backward graph."""

    def test_forward_is_one_graph(self) -> None:
        forward = _census("patched").forward
        assert (forward.graphs, forward.breaks) == (1, 0), forward
        assert FWD_OP in forward.ops[0], forward

    def test_backward_is_one_graph(self) -> None:
        backward = _census("patched").backward
        assert (backward.graphs, backward.breaks) == (1, 0), backward
        assert BWD_OP in backward.ops[0], backward


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestRawControl:
    """Unpatched FA4 breaks both passes, so the census can see a break."""

    def test_forward_breaks(self) -> None:
        forward = _census("raw").forward
        assert forward.graphs > 1, forward
        assert forward.breaks > 0, forward

    def test_backward_breaks(self) -> None:
        backward = _census("raw").backward
        assert backward.graphs > 1, backward
        assert backward.breaks > 0, backward
        assert BWD_OP not in backward.ops_anywhere(), backward


if __name__ == "__main__":
    impl, out_path = sys.argv[1], Path(sys.argv[2])
    out_path.write_text(_to_json(_census_in_this_process(IMPLS[impl])))  # pyright: ignore[reportArgumentType]
