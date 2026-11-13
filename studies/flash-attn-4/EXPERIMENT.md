# Experiment: standardized FlashAttention 4 on H100

## Goal

The goal needs some definition... here are the subgoals which at some point will need to be defined precisely:

1. Well optimized FlashAttention 4 implementation for H100 (and at some point B200)
2. On which workload can we guarantee or would recommend a user to use this/our implementation?

   2.1 Regarding the workload question, do we want to set a list of supported workloads, or should we go with a broader approach which would mean having some sort of compiler based approach such as FlexAttn for example?
3. I think it would be important to have a minimalistic regression test suite to ensure that the implementation is correct and that the performance is actually helping the user to train the model faster.

# Notes

Nov 10:

- It seems like FlashAttention 4 is not exactly built as a general purpose attention accelerator. Based on the `can_implement` method in `flash_attn/cute/flash_bwd_sm90.py`, it seems like the implementation is made for a very specific setting of input tensors and architecture (fp16, bf16, multiple of 8 head dimension, multiple of 16 kernel tile, multiple of 32 threads, ...).

Nov 11:

- The documented torch.compile problem supports the fact that FlashAttention 4 is not a general purpose attention accelerator. Indeed, given that cuteDLS is opaque Dynamo (tracer of torch compile, similar to whatever generates jaxpr for XLA in Jax) it cannot step into the kernel and determine the necessary information (tensors types and shapes) to build the computational graph.
- The root of this problem is that cuteDSL is not compiled by CPython interpreter, but by the CUTLASS Python compiler (intermediate representation (IR) to GPU binary code) stack which makes it impossible to retrieve the tensors dimensions and dtypes at compile time using PyTorch. This is why when we construct the computational graph, there is a break since without that information, we cannot generate the code for this operation. Thus, the solution was to explicitly define the input and output tensors dimensions and dtypes for both the forward and backward operations (which are the kernel optimized by cuteDLS for attention mechanism) and register them as fake ops so that at compile time, Dyanmo would know the dimensions and dtypes of the tensors and be able to generate the code for this operation and thus not force re-compilation at each foward or backward pass.
- So the conclusion to this is that cuteDSL does work, however it seems like the solution is hand crafted additional information (in the form of fake ops) so that this FA4 implementation can be used with torch.compile.

# Solution(s) to investigate

Problem in one line: cuteDSL is opaque to Dynamo which makes FA4 not compatible with non-specified workload.

Solution 1: Write explicitly the shapes and dtype for supported workload

- Limits dramatically the amount of supported workload
- Not scalable in the long term

Solution 2: Find a way to make cuteDSL transparent to Dynamo. If we are able to get the necessary information to build the computational graph at compile time, then we can generate the code for this operation and thus not force re-compilation at each forward or backward pass and still use cuteDSL for the kernel optimization.

+ This would support a broader range of workloads

- Not very knowledgeable in the CUTLASS Python compiler stack and the cuteDSL implementation, so not sure if this is possible (but I am more than willing to check this avenue!).

# WIP

- I am implementing a basic regression test to know when a fwd and bwd implementation creates breaks and more than 1 computational graph
- I am trying to generate tlpase and nsys traces so I better understand what is happening when the breaks occur. This should be very useful to understand the problem and find a solutions in the future.

## Next experiment: compare three compiled attention paths

When SDPA selects Flash Attention, `torch.compile` can capture PyTorch's
existing attention ops using their metadata and autograd rules. Activating
FA4 swaps their CUDA implementation to Dao-AILab's CuTeDSL kernels. Calling
`flash_attn_func` directly exposes FA4's Python code to Dynamo and can cause
graph breaks. The [local wrapper](examples/attn.py) keeps that code behind
custom forward and backward ops with `register_fake` metadata.

Start on one H100 with dense BF16 causal self-attention and no dropout. Use
batch size 8, sequence length 1024, 16 heads, and head dimension 64 for all
three paths:

1. `torch.compile` with default PyTorch SDPA.
2. `torch.compile` with SDPA after `activate_flash_attention_impl("FA4")`,
   forcing `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`.
3. `torch.compile` with the local `fa4_attn` wrapper calling Dao-AILab's FA4
   directly.

Use the same QKV values and upstream gradient. Check outputs and gradients,
and verify the backend that actually ran. Report steady-state training forward
and backward latency separately after warmup, with speedups relative to default
SDPA. The comparison between paths 2 and 3 measures the effect of the
integration path when both use FA4.
