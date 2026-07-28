# Integration notes

Item 1 (Vulkan MMVQ for `q2_0`) is **SHIPPED and measured**: +34.1% decode.
Item 2 (the CPU Q1_0 register LUT) is still blocked, and is scoped here to
the point where the actual obstacle is known, so the next attempt starts
past the reverse-engineering rather than repeating it.

Both were written up the same way for a reason: each had a specific way of
being *silently* wrong, and in item 1 both of those ways actually happened
before it worked.

---

## 1. ggml-vulkan MMVQ for `q2_0` -- SHIPPED (`q1_0` still open)

**Earlier claim, now corrected:** we said the blocker was `QUANT_K = 128`
against a framework built for 32. **That was wrong.** The framework
already handles arbitrary block sizes -- IQ1_S is 256 and Q2_K is 256,
both supported via the `DATA_A_QUANT_K` path. Block size is not the
obstacle.

### The actual contract

A type joins the integer-dot path by supplying, in
`mul_mat_vecq_funcs.glsl`:

```glsl
FLOAT_TYPE  get_dm(uint ib);                    // scale for virtual block ib
i32vec4     repack4(uint ib, uint iqs);         // 16 quants as 4 packed words
FLOAT_TYPE  mmvq_dot_product(uint ib_a, uint iqs);
```

plus the type name added to the gates at
`vulkan-shaders-gen.cpp:594` (MMQ), `:710` (MMVQ) and `:1150`.

`ib` is a **virtual block index in units of 32 quants**
(`QUANT_K_Q8_1`), not a real block index. Q2_K, being 256 quants,
converts with `ib_k = ib / 8`. Q2_0 and Q1_0 are 128 quants, so they
would use `ib_k = ib / 4` and an intra-block offset of
`(ib % 4) * 32 + iqs`.

`types.glsl` already defines `block_q1_0` and `block_q2_0` (lines ~193
and ~209) but only with `uint8_t qs[]`. Both need a `_packed32` variant
and an `A_TYPE_PACKED32` define before `data_a_packed32[...]` is usable.

### The trap: the shift trick yields STRIDED weights, not consecutive

Q2_K extracts with `(word >> shift) & 0x03030303`. Applying that to a
Q2_0 word does **not** give four consecutive weights.

Q2_0 packs weight `j` at byte `j/4`, bits `(j%4)*2`. So for a 32-bit word
covering weights `16i .. 16i+15`, the expression
`(w >> s) & 0x03030303` selects the field at shift `s` from each of the
four bytes, which is weights

```
  {16i + 0 + s/2,  16i + 4 + s/2,  16i + 8 + s/2,  16i + 12 + s/2}
```

-- **stride 4, not consecutive.** A `dotPacked4x8EXT` against
`cache_b_qs[n]` therefore pairs each code with the wrong activation
unless the activation cache is permuted to match.

This is exactly the distinction our own CUDA ladder made between v2 (GGUF
layout, gather the activations) and v3 (bit-plane repacked layout, one
aligned load). In Vulkan there is no freedom to repack -- the weights
arrive in GGUF order -- so the correct implementation is the **v2 shape**:
extract per byte and build consecutive codes,

```glsl
const uint c4 = (w >> (8*p)) & 0xFFu;
const uint codes = (c4 & 3) | (((c4 >> 2) & 3) << 8)
                 | (((c4 >> 4) & 3) << 16) | (((c4 >> 6) & 3) << 24);
```

which costs more ALU than Q2_K's single mask but keeps the activation
cache untouched.

**Why this matters:** the strided version *compiles, runs, and produces
plausible numbers*. It would be caught only by a numerical check against
a reference.

### Extraction implemented and gated -- and it is faster

The correct form is now built and validated in our own harness
(`kernels/vulkan/gemv_q2.comp`, `-DGGUF_ORDER`), against the same CPU
reference:

```
q2 int-dot           (repacked)      101.7 / 104.3 / 102.6 GB/s
q2 int-dot GGUF-ord  (as ggml gets)  106.9 / 106.1 / 110.4 GB/s
max rel err 4.783e-06 -- identical to the repacked path
```

GGUF order is **2-8% faster**, reproducibly, despite more ALU per code.
So the integration needs **no repack, no new weight layout, no on-disk
change** -- only `unpack_gguf()` plus the type added to the three gates.
Copy the function verbatim; it is tested.

### SHIPPED. Five gates, and the fifth is silent by design

The integration works and is measured: **tg128 11.75 -> 15.76 tok/s
(+34.1%)** on Ternary-Bonsai-27B-Q2_0, same binary, toggled with
`GGML_VK_DISABLE_MMVQ`. Fork changes are in
`patches/0003-vulkan-q2_0-mmvq.patch` (ggml-vulkan only, reverse-applies
cleanly). Full write-up: `results/vulkan-mmvq-q2_0.txt`.

There are **five** gates, not the four previously listed:

1. toolchain -- glslc implements `GL_EXT_integer_dot_product`
2. shader-gen -- type in the MMVQ / q8_1 lists (`~710`, `~1150`)
3. `K_PER_ITER` -- type must land in a size class in `mul_mat_vecq.comp`
4. runtime -- an explicit `ggml_vk_create_pipeline` call (`~4655`)
5. runtime -- the type must **also** be in the `b_type == GGML_TYPE_Q8_1`
   allow-list atop `ggml_vk_get_dequantize_mul_mat_vec` (`~6874`)

Gate 5 is the one to remember. Its `default:` is `return nullptr`, and the
caller treats `nullptr` as "use the float path" -- not as an error. With
gates 1-4 open and gate 5 shut, the shader compiles, the SPIR-V is in the
`.so`, the pipeline is created, generation is **correct**, and throughput
is **unchanged**. There is no error anywhere. It was found only by A/B
toggling and getting 11.58 vs 11.61 tok/s, i.e. by a null result where a
win was expected. See WIKI L8.

### The index arithmetic was right; the STRUCT was the bug

Confirmed by controlled experiment, both arms built with nothing else
changed:

| shader-side view | `test-backend-ops -o MUL_MAT` (q2_0) |
| :-- | :-- |
| `block_q2_0_packed32` (`uint32_t qs[]`) | 11 OK, 17 FAIL, NMSE 1.33-2.35 |
| `block_q2_0_packed16` (`uint16_t qs[]`) | 28 OK, 0 FAIL |

A Q2_0 block is 2 + 32 = 34 bytes, hence only 2-byte aligned; a `uint32_t`
member forces std430 alignment 4 and pads the array stride to 36,
misreading every block after the first. Q4_0 (18 bytes) uses `packed16` for
exactly this reason. The addressing `ib_a / 4`, `(ib_a % 4) * 2 + iqs` was
correct all along -- reading the caller settles it: `a_block_idx` counts
32-quant units and `b_qs_idx = tid % (32 / K_PER_ITER)` is 0 or 1.

### How to gate this, and two gates that do not work

`GGML_VK_FORCE_MMVQ=1 test-backend-ops -o MUL_MAT -b Vulkan0` -> 28 OK,
0 FAIL (10 at `n=1`). The `FORCE` is mandatory: every shape in the suite is
`k <= 1024` and `ggml_vk_should_use_mmvq` returns false for `k <= 4096` on
NVIDIA, so the tests otherwise pass without running the kernel.

Do **not** gate on perplexity (pure prefill -- gave identical `PPL =
11.6848` while measuring nothing) or on greedy text diff alone (the
integer-dot path changes summation order, so near-ties flip legitimately;
1 of 4 prompts matched 160/160 tokens exactly and another diverged at
character 83 of ~700). Coverage limit worth stating: the op tests cover
`k = 256` and `k = 1024`, while production shapes are `k = 5120` and
`17408`.

### Bias correction, already derived

Q2_0 codes are `{0,1,2,3}` for `{-1,0,1,2}`; Q1_0 bits are `{0,1}` for
`{-1,+1}`:

```
Q2_0:  mul_q8_1 = da * (q_sum*dsb.x - (1/div)*dsb.y)
Q1_0:  mul_q8_1 = da * (2*q_sum*dsb.x - (1/div)*dsb.y)
```

matching the identity used on all three of our own backends.

### Expected value

Precedent (PR #16536, same change for the k-quants) is +78% to +131% on
pp512. Our own standalone shaders are the independent evidence for this
hardware: the same Q1_0 GEMV goes **71.2 -> 99.4 GB/s (+40%)** when it
switches from the scalar fallback to `dotPacked4x8AccSatEXT`.

---

## 2. CPU Q1_0 register LUT

**The obstacle is the weight layout, not the loop order.** Our earlier
note said the loop must be transposed (activation group outer, rows
inner) so the table amortizes. That is necessary but not sufficient, and
the reason is worth stating because it is easy to get half-right.

A T-MAC `g=4` table maps a 4-bit weight pattern to the partial dot of 4
weights against 4 activations. The table therefore belongs to an
*activation group*, and there are 32 distinct groups in a 128-weight
block.

`vqtbl1q_u8` applies **one** table to 16 byte indices. Feeding it 16
consecutive nibbles of a single row means 16 *different* groups, so 16
different tables -- which the instruction cannot do.

T-MAC's answer is to make the 16 indices come from **16 different rows at
the same group position**, so one table serves all of them. That requires
the weights repacked so that the nibbles for group `g` across `R` rows
are contiguous -- a row-interleaved layout, not merely a reordered loop.

Second obstacle, independent of the first: **the table does not fit
int8.** Entries are sums of up to 4 signed int8 activations, range
+-508. Options are an int16 table with two lookups per group (roughly
doubling the lookup cost), or pre-scaling activations into a narrower
range and accepting a precision loss. The second breaks the bit-exactness
this repo gates on, so it would need its own numerical justification
rather than being slipped in.

### Where this sits against the measured numbers

Threaded, our Q1_0 kernel reaches 17.67 GB/s -- **14% of the 126.4 GB/s
CPU DRAM bandwidth**, against Q2_0's 35%. Converted end to end it is
4.65 tok/s versus llama.cpp's current 4.85, i.e. **worth nothing today**.
So the LUT is the right target, but it needs a CPU-side repack, and the
repack is the part that actually has to be built.

Note this does not violate the one-shared-format principle: the repack is
a load-time CPU-side transform of the same GGUF bytes, exactly what
llama.cpp's own `repack.cpp` does for other types, and the on-disk format
is untouched.
