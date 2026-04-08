# profile: rtx_5090, config for 8192: TM=28, BK=32, ks=1
# === SASS analysis for tma_db @ 8192x8192x8192, batch=1 ===

# fused_matmul: 4541 SASS lines

## Our TMA kernel — opcode histogram (by family)

| Family | Count |
|---|---:|
| `FFMA*       (fused multiply-add — the actual compute)` | 3584 |
| `LDS.*       (shared-memory loads, incl. .128 vector form)` | 256 |
| `STG.*       (global stores, incl. .E.128 vector form)` | 112 |
| `LDC.*       (constant loads — kernel params, TMA descriptors)` | 30 |
| `UTMALDG.*   (TMA load commands)` | 4 |
| `CS2R/S2R    (special-register reads)` | 48 |
| `ISETP.*     (integer set-predicate, bounds + loop control)` | 143 |
| `BAR/BSYNC/BSSY (block barriers + reconvergence)` | 60 |
| `MOV/IMAD/IADD/LEA (address arithmetic, reg copies)` | 169 |
| `LOP3/PLOP3  (logic + predicate logic)` | 23 |
| `HFMA2       (FP16x2 helpers — typically address calc)` | 17 |
| `BRA/EXIT/CALL (branches)` | 40 |
| `FENCE/MEMBAR (memory ordering)` | 1 |
| `OTHER (16 mnemonics — see raw histogram)` | 54 |

## Our TMA kernel — top mnemonics (raw, top 15)

| Mnemonic | Count |
|---|---:|
| `FFMA` | 3584 |
| `LDS.128` | 256 |
| `ISETP.GT.AND` | 141 |
| `STG.E` | 112 |
| `LEA` | 58 |
| `CS2R` | 41 |
| `BRA` | 31 |
| `IADD` | 30 |
| `BSSY.RECONVERGENT` | 29 |
| `BSYNC.RECONVERGENT` | 29 |
| `LDC.64` | 28 |
| `IMAD.WIDE` | 28 |
| `MOV` | 20 |
| `HFMA2` | 17 |
| `UMOV` | 16 |

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
| `FFMA` | 3584 | 0 | 7 | 15 | 1797 |
| `LDS.128` | 256 | 0 | 0 | 0 | 0 |
| `ISETP.GT.AND` | 141 | 0 | 0 | 0 | 141 |
| `STG.E` | 112 | 4 | 4 | 4 | 0 |
| `LEA` | 58 | 15 | 15 | 15 | 58 |
| `CS2R` | 41 | 0 | 0 | 0 | 0 |
| `BRA` | 31 | 0 | 0 | 15 | 2 |
| `IADD` | 30 | 0 | 0 | 0 | 0 |
| `BSSY.RECONVERGENT` | 29 | 0 | 0 | 0 | 0 |
| `BSYNC.RECONVERGENT` | 29 | 0 | 0 | 0 | 0 |
| `LDC.64` | 28 | 0 | 0 | 0 | 0 |
| `IMAD.WIDE` | 28 | 4 | 4 | 4 | 0 |
| `MOV` | 20 | 0 | 0 | 0 | 0 |
| `HFMA2` | 17 | 15 | 15 | 15 | 17 |
| `UMOV` | 16 | 0 | 0 | 0 | 0 |

### Raw control words for the first 20 FFMA instructions

```
  FFMA  ctrl=0x081fe20000000009  bin=0000100000011111111000100000000000000000000000000000000000001001
  FFMA  ctrl=0x000fc40000000021  bin=0000000000001111110001000000000000000000000000000000000000100001
  FFMA  ctrl=0x000fe20000000020  bin=0000000000001111111000100000000000000000000000000000000000100000
  FFMA  ctrl=0x002fe20000000023  bin=0000000000101111111000100000000000000000000000000000000000100011
  FFMA  ctrl=0x040fe200000000b2  bin=0000010000001111111000100000000000000000000000000000000010110010
  FFMA  ctrl=0x080fe200000000b8  bin=0000100000001111111000100000000000000000000000000000000010111000
  FFMA  ctrl=0x000fe200000000ad  bin=0000000000001111111000100000000000000000000000000000000010101101
  FFMA  ctrl=0x044fe20000000021  bin=0000010001001111111000100000000000000000000000000000000000100001
  FFMA  ctrl=0x040fe20000000020  bin=0000010000001111111000100000000000000000000000000000000000100000
  FFMA  ctrl=0x000fe20000000014  bin=0000000000001111111000100000000000000000000000000000000000010100
  FFMA  ctrl=0x000fe200000000b0  bin=0000000000001111111000100000000000000000000000000000000010110000
  FFMA  ctrl=0x008fe20000000055  bin=0000000010001111111000100000000000000000000000000000000001010101
  FFMA  ctrl=0x080fe200000000b2  bin=0000100000001111111000100000000000000000000000000000000010110010
  FFMA  ctrl=0x040fe20000000016  bin=0000010000001111111000100000000000000000000000000000000000010110
  FFMA  ctrl=0x040fe200000000b8  bin=0000010000001111111000100000000000000000000000000000000010111000
  FFMA  ctrl=0x040fe200000000ad  bin=0000010000001111111000100000000000000000000000000000000010101101
  FFMA  ctrl=0x000fe20000000018  bin=0000000000001111111000100000000000000000000000000000000000011000
  FFMA  ctrl=0x000fe20000000014  bin=0000000000001111111000100000000000000000000000000000000000010100
  FFMA  ctrl=0x050fe20000000015  bin=0000010100001111111000100000000000000000000000000000000000010101
  FFMA  ctrl=0x000fe20000000017  bin=0000000000001111111000100000000000000000000000000000000000010111
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

