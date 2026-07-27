# Decode kernels

Batch-1 low-bit GEMV for the Bonsai family, across CUDA, Vulkan and ARM
NEON. This is the core of Phase 2: at batch 1 the model streams every
weight exactly once per token, so decode speed is this kernel.

## The design in one page

**One weight format, three code generators.** `common/bonsai_gemv.h` is
the single source of truth. Backends differ in their dot instruction and
their vector width, not in what the bytes mean.

**Biased encoding.** Q2_0 stores codes `c` in `{0,1,2,3}` for values
`{-1,0,+1,+2}` (note: not strictly ternary -- code 3 is +2). Rather than
subtract 1 per element, dot the raw codes and correct once per chunk:

```
Q2_0:  sum w*x = d*da * ( sum c*a  -  sum a )
Q1_0:  sum w*x = d*da * ( 2*sum b*a - sum a )
```

The `sum a` table is built once per token when activations are quantized
and amortizes over every row. No sign handling, no branches, and it maps
identically onto CUDA `__dp4a`, Vulkan `dotPacked4x8AccSatEXT`, and ARM
`SDOT`. Measured worth ~+17% on CUDA on its own, because the per-element
subtract otherwise expands into PRMT+LOP3 chains (`__vsubss4` has no
single SASS instruction).

**Bit-plane interleaved repack.** The weight word is rearranged so that
`(w >> 2q) & 0x03030303` yields four *consecutive* logical weights. Then
the matching activations are one aligned load rather than a gather.
Convergent with llama.cpp's TQ2_0, microsoft/BitNet, and bonsai-turbo --
treat as settled.

**Where the permutation lands.** The repack makes lanes consecutive for a
*4-byte* word. CUDA's `__dp4a` and Vulkan's `dotPacked4x8` both consume
exactly 4 bytes, so both use the same weights *and* the same activation
layout, unpermuted. NEON's vector is 16 bytes and spans four weight
words, so its lanes are four groups of four strided by 16 -- it needs an
activation shuffle. Doing that on the activations (K bytes, once per
token) rather than on the weights (N*K/4 bytes, shared with every other
backend) is the cheap side of the trade.

## Status

| Backend | Correctness | Performance |
| :-- | :-- | :-- |
| CPU (NEON) | **bit-exact vs scalar reference** | **measured, below** |
| CUDA | reference check written, compiles `sm_110a` | **not yet run** |
| Vulkan | both shaders compile to SPIR-V | **not yet run** |

CUDA and Vulkan are unvalidated on device because the GPU is currently
unusable: after a suspend/resume cycle, 201 processes are wedged in `D`
state inside the NVIDIA UVM replayable-fault handler and `nvidia_uvm`'s
refcount is stuck at 1272, so any new CUDA or Vulkan context hangs. The
stuck processes are `rustdesk`, not ours, and `nvidia-smi` still reports
the device healthy at 0% util. **This needs a reboot to clear.** No
performance claim is made for either GPU backend until it has actually
run.

## Measured: CPU (ARM NEON), Jetson Thor, 14x Neoverse V3AE

Shape N=32768 K=4096. All variants bit-exact (max rel err 0.000e+00).
Full log in `../results/gemv-neon.txt`.

Single thread, GB/s of weight bytes:

| Kernel | Q2_0 | Q1_0 |
| :-- | --: | --: |
| scalar per-element extract (shape of the current ARM path) | 0.37 | 0.03 |
| whole-vector unpack + SDOT | **5.78** | **2.78** |

**~16x for ternary, ~90x for 1-bit** over per-element extract. Row
blocking (1/2/4/8 accumulator chains) made no difference single-threaded
-- this kernel is unpack-ALU bound, not latency bound, which matches the
core's issue rates (SHR/AND ~4/cycle, SDOT ~2/cycle).

Threaded, 8 rows per chain:

| threads | Q2_0 GB/s | Q1_0 GB/s |
| --: | --: | --: |
| 1 | 5.72 | 2.75 |
| 4 | 14.90 | 7.86 |
| 8 | 26.41 | 14.46 |
| 10 | 26.82 | **16.91** |
| **12** | **38.75** | 15.43 |
| 14 | 19.01 | 14.39 |

**14 threads collapses** -- independently reproducing the earlier
end-to-end finding that leaving cores for the OS is worth more than using
them. Optimum is 12 for Q2_0 and 10 for Q1_0.

### What that means end to end, and the honest split

Projecting weights-only bandwidth onto the real model
(ternary 7.17 GB, 1-bit 3.80 GB of weights per token):

| Variant | This kernel | llama.cpp today (t=12) | Ratio |
| :-- | --: | --: | --: |
| ternary Q2_0 | **5.40 tok/s** | 3.18 | **1.70x** |
| 1-bit Q1_0 | 4.45 tok/s | 4.85 | **0.92x** |

**Ternary is a clear win; 1-bit is a loss.** That is not a bug, it is the
predicted result: SDOT retires 16 weights per instruction regardless of
bit width, so a 1-bit kernel shaped like a 2-bit one gets the same
weights/s while moving half the bytes -- it is purely ALU-bound and gains
nothing from being smaller. llama.cpp's Q1_0 path uses a lookup table,
which retires more weights per instruction.

The fix is known and is the next task: a T-MAC style **register-resident
LUT** (`vqtbl1q`, g=4), where one TBL covers 16 lanes x 4 weights = 64
weights per instruction against SDOT's 16. Published and independently
measured at ~4.4x for 1-bit on this class of core. It is deliberately not
half-implemented here: the table entries need int16 range (4 activations
x +/-127 = +/-508) so a correct version needs a hi/lo split or a scaled
table, and shipping an unvalidated approximation would break the
bit-exactness bar every other kernel here meets.

## Building

```bash
# CPU
cc -O3 -march=armv8.2-a+dotprod -fopenmp \
   -o cpu/gemv_neon_bench cpu/gemv_neon_bench.c -lm
./cpu/gemv_neon_bench 4096 32768 3

# CUDA (needs a working GPU to run)
nvcc -O3 -arch=sm_110a --extended-lambda -o cuda/gemv_bench cuda/gemv_bench.cu
./cuda/gemv_bench 4096 65536 20

# Vulkan shaders
glslc --target-env=vulkan1.3 -O -o vulkan/gemv_q2.spv vulkan/gemv_q2.comp
glslc --target-env=vulkan1.3 -O -o vulkan/gemv_q1.spv vulkan/gemv_q1.comp
# add -DUSE_INT_DOT once the toolchain supports GL_EXT_integer_dot_product
```

The Vulkan shaders compile two ways. The default is the **portable**
path: plain integer arithmetic, any Vulkan 1.1 device. `-DUSE_INT_DOT`
selects `dotPacked4x8AccSatEXT`, which needs
`VK_KHR_shader_integer_dot_product` (~70% of devices) and a newer
shaderc than the 2023.8 installed here. Note the *packed* form is the
accelerated one: Thor reports
`integerDotProduct4x8BitPackedSignedAccelerated = true` but
`integerDotProduct8BitSignedAccelerated = false`, so the `i8vec4` form is
not the fast path anywhere it matters.

## Next, in order

1. **Q1_0 register LUT on CPU** -- the one measured regression above.
2. **Run and tune the CUDA ladder** once the GPU is back. Research
   measured 229 GB/s (84% of spec) achievable for this kernel shape on
   this device, against a reference fork sitting at ~52% of that ceiling.
3. **Run the Vulkan shaders**, then attack barrier scoping -- measured at
   ~1.4us narrow vs ~3.9us wide per dispatch, with ~2293 dispatches per
   token, which is a ~3.9 ms/token floor and the reason speculation loses
   on that backend.
4. **Wire the kernels into a decode step** (GDN recurrent layers, GQA
   attention) rather than benchmarking them standalone.
