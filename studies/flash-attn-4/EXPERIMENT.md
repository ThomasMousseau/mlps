# Experiment: standardized FlashAttention 4 on H100

## Goal

The goal needs some definition... here are the subgoals which at some point will need to be defined precisely:

1. Well optimized FlashAttention 4 implementation for H100 (and at some point B200)
2. On which workload can we guarantee or would recommend a user to use this/our implementation?

   2.1 Regarding the workload question, do we want to set a list of supported workloads, or should we go with a broader approach which would mean having some sort of compiler based approach such as FlexAttn for example?
3. I think it would be important to have a minimalistic regression test suite to ensure that the implementation is correct and that the performance is actually helping the user to train the model faster.

# Notes

Nov 10:

- It seems like FlashAttention 4 is not exactly built as a general purpose attention accelerator. Based on the can_implement method in flash_attn/cute/flash_bwd_sm90.py, it seems like the implementation is made for a very specific setting of input tensors and architecture (fp16, bf16, multiple of 8 heads, multiple of 16 tokens, multiple of 32 threads, ...).

Nov 11:

- The documented torch.compile problem supports the fact that FlashAttention 4 is not a general purpose attention accelerator. Indeed, given that there were no differentiable ops in some of the FA4 methods, Dynamo (which is the JIT compiler of PyTorch) would insert non-differentiable ops in the graph, which would break the computational graph. Meaning that at compute time, JIT would recompile and still act greedily.
- The root of this problem is that cuteDSL is not compiled by CPython interpreter, but by the CUTLASS Python compiler (intermediate representation (IR) to GPU binary code) stack which makes it impossible to retrieve the tensors dimensions and dtypes at compile time using PyTorch. This is why when we construct the computational graph, there is a break since without that information, we cannot generate the code for this operation. Thus, the solution was to explicitly define the input and output tensors dimensions and dtypes for both the forward and backward operations and register them as fake ops so that at compile time, JIT would know the dimensions and dtypes of the tensors and be able to generate the code for this operation.
- So the conclusion to this is that cuteDSL does work, however it seems like the solution is hand crafted additional information (in the form of fake ops) so that this FA4 implementation can be used with torch.compile.
- Now, the solution to this problem will depend on the scope of this project.
