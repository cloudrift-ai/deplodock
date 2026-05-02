# profile: h200, config for 8192: TM=8, BK=32, ks=1
# === SASS analysis for tma_db @ 8192x8192x8192, batch=1 ===

# fused_matmul: 1327 SASS lines

## Our TMA kernel — opcode histogram (by family)

| Family | Count |
|---|---:|
| `FFMA*       (fused multiply-add — the actual compute)` | 1024 |
| `LDS.*       (shared-memory loads, incl. .128 vector form)` | 104 |
| `STG.*       (global stores, incl. .E.128 vector form)` | 32 |
| `LDC.*       (constant loads — kernel params, TMA descriptors)` | 2 |
| `UTMALDG.*   (TMA load commands)` | 4 |
| `CS2R/S2R    (special-register reads)` | 17 |
| `ISETP.*     (integer set-predicate, bounds + loop control)` | 2 |
| `BAR/BSYNC/BSSY (block barriers + reconvergence)` | 6 |
| `MOV/IMAD/IADD/LEA (address arithmetic, reg copies)` | 35 |
| `LOP3/PLOP3  (logic + predicate logic)` | 21 |
| `HFMA2       (FP16x2 helpers — typically address calc)` | 4 |
| `BRA/EXIT/CALL (branches)` | 11 |
| `FENCE/MEMBAR (memory ordering)` | 1 |
| `OTHER (19 mnemonics — see raw histogram)` | 64 |

## Our TMA kernel — top mnemonics (raw, top 15)

| Mnemonic | Count |
|---|---:|
| `FFMA` | 1024 |
| `LDS.128` | 96 |
| `STG.E` | 32 |
| `UIADD3` | 15 |
| `PLOP3.LUT` | 15 |
| `CS2R` | 14 |
| `UMOV` | 13 |
| `NOP` | 9 |
| `LDS` | 8 |
| `LEA` | 7 |
| `ULOP3.LUT` | 6 |
| `BRA` | 5 |
| `USEL` | 5 |
| `S2UR` | 4 |
| `HFMA2.MMA` | 4 |

## Stall counts (best-effort, control-word low 4 bits)

> ⚠️ The SASS control-word bit layout for sm_120 (Blackwell) is not
> publicly documented. The numbers below assume the same layout as
> sm_90 (bits [0:3] = stall count, bit [4] = yield), which may not
> hold on Blackwell. Treat these as suggestive, not authoritative.
> Raw 64-bit control words for the first 20 FFMA instructions are
> dumped at the end so you can re-derive the layout if you have
> better information.

| Mnemonic | N | min | median | max | yield count |
|---|---:|---:|---:|---:|---:|
| `FFMA` | 1024 | 0 | 7 | 15 | 529 |
| `LDS.128` | 96 | 0 | 0 | 0 | 0 |
| `STG.E` | 32 | 2 | 2 | 2 | 32 |
| `UIADD3` | 15 | 15 | 15 | 15 | 15 |
| `PLOP3.LUT` | 15 | 0 | 0 | 8 | 13 |
| `CS2R` | 14 | 0 | 0 | 0 | 0 |
| `UMOV` | 13 | 0 | 0 | 0 | 0 |
| `NOP` | 9 | 0 | 0 | 0 | 0 |
| `LDS` | 8 | 0 | 0 | 0 | 0 |
| `LEA` | 7 | 15 | 15 | 15 | 7 |
| `ULOP3.LUT` | 6 | 15 | 15 | 15 | 6 |
| `BRA` | 5 | 0 | 15 | 15 | 3 |
| `USEL` | 5 | 0 | 0 | 0 | 0 |
| `S2UR` | 4 | 0 | 0 | 0 | 0 |
| `HFMA2.MMA` | 4 | 15 | 15 | 15 | 4 |

### Raw control words for the first 20 FFMA instructions

```
  FFMA  ctrl=0x081fe20000000027  bin=0000100000011111111000100000000000000000000000000000000000100111
  FFMA  ctrl=0x0c0fe20000000024  bin=0000110000001111111000100000000000000000000000000000000000100100
  FFMA  ctrl=0x000fe20000000035  bin=0000000000001111111000100000000000000000000000000000000000110101
  FFMA  ctrl=0x000fe20000000034  bin=0000000000001111111000100000000000000000000000000000000000110100
  FFMA  ctrl=0x040fe20000000026  bin=0000010000001111111000100000000000000000000000000000000000100110
  FFMA  ctrl=0x080fe20000000036  bin=0000100000001111111000100000000000000000000000000000000000110110
  FFMA  ctrl=0x000fe20000000037  bin=0000000000001111111000100000000000000000000000000000000000110111
  FFMA  ctrl=0x002fe20000000006  bin=0000000000101111111000100000000000000000000000000000000000000110
  FFMA  ctrl=0x000fe2000000001c  bin=0000000000001111111000100000000000000000000000000000000000011100
  FFMA  ctrl=0x000fe20000000039  bin=0000000000001111111000100000000000000000000000000000000000111001
  FFMA  ctrl=0x080fe20000000024  bin=0000100000001111111000100000000000000000000000000000000000100100
  FFMA  ctrl=0x040fe20000000034  bin=0000010000001111111000100000000000000000000000000000000000110100
  FFMA  ctrl=0x000fe2000000001d  bin=0000000000001111111000100000000000000000000000000000000000011101
  FFMA  ctrl=0x080fe20000000036  bin=0000100000001111111000100000000000000000000000000000000000110110
  FFMA  ctrl=0x040fe20000000007  bin=0000010000001111111000100000000000000000000000000000000000000111
  FFMA  ctrl=0x000fe20000000018  bin=0000000000001111111000100000000000000000000000000000000000011000
  FFMA  ctrl=0x080fe20000000025  bin=0000100000001111111000100000000000000000000000000000000000100101
  FFMA  ctrl=0x000fe2000000001e  bin=0000000000001111111000100000000000000000000000000000000000011110
  FFMA  ctrl=0x040fe20000000026  bin=0000010000001111111000100000000000000000000000000000000000100110
  FFMA  ctrl=0x000fe20000000037  bin=0000000000001111111000100000000000000000000000000000000000110111
```

## cuBLAS — `cutlass_80_simt_sgemm_*` PTX histogram (from libcublasLt.so)

| Family | Count |
|---|---:|
| `fma.rn.f32        (FP32 multiply-add)` | 1152 |
| `cp.async          (LDGSTS cooperative load)` | 44 |
| `st.shared         (smem store)` | 256 |
| `ld.shared         (smem load)` | 310 |
| `ld.global         (global load)` | 8 |
| `st.global         (global store)` | 1 |
| `bar.sync          (__syncthreads)` | 70 |
| `setp.*            (predicate set)` | 522 |
| `mov.*             (register copies)` | 902 |
| `add.*             (integer + FP add)` | 601 |

