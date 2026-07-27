# Integration notes: two blocked items, de-risked

Both remaining high-value items were scoped to the point where the actual
obstacle is known. Neither is shipped, because both have a specific way
of being silently wrong and neither is verified. Written so the next
attempt starts past the reverse-engineering.

---

## 1. ggml-vulkan MMVQ for `q1_0` / `q2_0`

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

### STATUS: extraction implemented and gated -- and it is faster

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

### ATTEMPTED END TO END -- and there is a THIRD gate

The integration was implemented in the fork and it does not yet work.
`patches/0003-WIP-vulkan-q2_0-mmvq.patch` holds it; the fork itself has
been reverted and re-verified correct.

**A third gate exists that none of the earlier analysis found.** Even
with the shader generated and the type in the shader-generator lists, the
runtime never creates a pipeline for it. Registration is explicit, per
type, in `ggml-vulkan.cpp` (~line 4655):

```
pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i]
```

and the set covers Q4_0..Q8_0, the k-quants, mxfp4 and iq1_s/m -- not
Q2_0. So the gate list is:

1. toolchain -- glslc implements `GL_EXT_integer_dot_product`  DONE
2. shader-gen -- type in the MMVQ/MMQ lists                    DONE
3. runtime -- an explicit `ggml_vk_create_pipeline` call        DONE
4. `K_PER_ITER` -- the type must fall in a size class           DONE

All four are now open, and the path demonstrably dispatches: adding the
registration changed the output. It changed it to **degenerate repetition
of the prompt**, so the index arithmetic is wrong.

**CORRECTION.** A follow-up "alignment fix" was applied by a script whose
`replace` calls silently matched nothing, because the tree had already
been reverted. It built, ran, and produced correct output and normal
throughput -- all of which were the BASELINE. The integration was not
present. See WIKI L7. The struct-alignment hypothesis below is therefore
**untested**, not confirmed.

**Where the error must be.** The standalone shader is validated
(4.783e-06) and its extraction is byte-for-byte the same, so
`unpack_gguf` is not the problem. What differs is the addressing:

```glsl
data_a_packed32[ib_a / 4].qs[(ib_a % 4) * 2 + iqs]
```

Reading the caller settles the index semantics and they were RIGHT:
`a_block_idx = (ibi + col)/QUANT_K_Q8_1` counts 32-quant units, and
`b_qs_idx = tid % (32 / K_PER_ITER)` is 0 or 1. So `ib_a / 4`,
`(ib_a % 4) * 2 + iqs` is correct addressing.

The remaining suspect is the **struct**, not the arithmetic. A Q2_0 block
is 2 (fp16) + 32 = **34 bytes**, only 2-byte aligned, so a `uint32_t qs[]`
member is padded to a 36-byte stride and misreads every block. Q4_0 (18
bytes) uses `packed16` for exactly this reason. The fix is a
`block_q2_0_packed16` with `uint16_t qs[16]`, assembling each word with
`pack32(u16vec2(qs[wi], qs[wi+1]))`. **Untested -- see the CORRECTION
above.**

Superseded note: the next attempt should instrument `iqs`, e.g. by writing `ib_a` and `iqs` to the
output buffer from a debug shader, which settles it in one run.

Everything else -- extraction, bias correction, gate sites, size class --
is now known good or known located. This is the last unknown.

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
