# profile: rtx_pro_6000, config for 8192: TM=24, BK=32, ks=1
# === SASS analysis for tma_db @ 8192x8192x8192, batch=1 ===

# fused_matmul: 3917 SASS lines

## Our TMA kernel — opcode histogram (by family)

| Family | Count |
|---|---:|
| `FFMA*       (fused multiply-add — the actual compute)` | 3072 |
| `LDS.*       (shared-memory loads, incl. .128 vector form)` | 224 |
| `STG.*       (global stores, incl. .E.128 vector form)` | 96 |
| `LDC.*       (constant loads — kernel params, TMA descriptors)` | 26 |
| `UTMALDG.*   (TMA load commands)` | 4 |
| `CS2R/S2R    (special-register reads)` | 43 |
| `ISETP.*     (integer set-predicate, bounds + loop control)` | 123 |
| `BAR/BSYNC/BSSY (block barriers + reconvergence)` | 52 |
| `MOV/IMAD/IADD/LEA (address arithmetic, reg copies)` | 150 |
| `LOP3/PLOP3  (logic + predicate logic)` | 23 |
| `HFMA2       (FP16x2 helpers — typically address calc)` | 14 |
| `BRA/EXIT/CALL (branches)` | 36 |
| `FENCE/MEMBAR (memory ordering)` | 1 |
| `OTHER (16 mnemonics — see raw histogram)` | 53 |

## Our TMA kernel — top mnemonics (raw, top 15)

| Mnemonic | Count |
|---|---:|
| `FFMA` | 3072 |
| `LDS.128` | 224 |
| `ISETP.GT.AND` | 121 |
| `STG.E` | 96 |
| `LEA` | 50 |
| `CS2R` | 36 |
| `BRA` | 27 |
| `IADD` | 26 |
| `BSSY.RECONVERGENT` | 25 |
| `BSYNC.RECONVERGENT` | 25 |
| `LDC.64` | 24 |
| `IMAD.WIDE` | 24 |
| `MOV` | 17 |
| `UMOV` | 16 |
| `HFMA2` | 14 |

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
| `FFMA` | 3072 | 0 | 7 | 15 | 1558 |
| `LDS.128` | 224 | 0 | 0 | 0 | 0 |
| `ISETP.GT.AND` | 121 | 0 | 0 | 0 | 121 |
| `STG.E` | 96 | 4 | 4 | 4 | 0 |
| `LEA` | 50 | 15 | 15 | 15 | 50 |
| `CS2R` | 36 | 0 | 0 | 0 | 0 |
| `BRA` | 27 | 0 | 0 | 15 | 2 |
| `IADD` | 26 | 0 | 0 | 0 | 0 |
| `BSSY.RECONVERGENT` | 25 | 0 | 0 | 0 | 0 |
| `BSYNC.RECONVERGENT` | 25 | 0 | 0 | 0 | 0 |
| `LDC.64` | 24 | 0 | 0 | 0 | 0 |
| `IMAD.WIDE` | 24 | 4 | 4 | 8 | 0 |
| `MOV` | 17 | 0 | 0 | 0 | 0 |
| `UMOV` | 16 | 0 | 0 | 0 | 0 |
| `HFMA2` | 14 | 15 | 15 | 15 | 14 |

### Raw control words for the first 20 FFMA instructions

```
  FFMA  ctrl=0x081fe2000000008f  bin=0000100000011111111000100000000000000000000000000000000010001111
  FFMA  ctrl=0x080fe20000000038  bin=0000100000001111111000100000000000000000000000000000000000111000
  FFMA  ctrl=0x080fe20000000069  bin=0000100000001111111000100000000000000000000000000000000001101001
  FFMA  ctrl=0x000fe2000000003a  bin=0000000000001111111000100000000000000000000000000000000000111010
  FFMA  ctrl=0x082fe20000000007  bin=0000100000101111111000100000000000000000000000000000000000000111
  FFMA  ctrl=0x080fe20000000004  bin=0000100000001111111000100000000000000000000000000000000000000100
  FFMA  ctrl=0x080fe20000000005  bin=0000100000001111111000100000000000000000000000000000000000000101
  FFMA  ctrl=0x000fe20000000006  bin=0000000000001111111000100000000000000000000000000000000000000110
  FFMA  ctrl=0x044fe2000000003e  bin=0000010001001111111000100000000000000000000000000000000000111110
  FFMA  ctrl=0x040fe20000000038  bin=0000010000001111111000100000000000000000000000000000000000111000
  FFMA  ctrl=0x040fe20000000069  bin=0000010000001111111000100000000000000000000000000000000001101001
  FFMA  ctrl=0x080fe2000000002c  bin=0000100000001111111000100000000000000000000000000000000000101100
  FFMA  ctrl=0x040fe2000000003a  bin=0000010000001111111000100000000000000000000000000000000000111010
  FFMA  ctrl=0x000fe2000000001c  bin=0000000000001111111000100000000000000000000000000000000000011100
  FFMA  ctrl=0x088fe20000000013  bin=0000100010001111111000100000000000000000000000000000000000010011
  FFMA  ctrl=0x080fe20000000010  bin=0000100000001111111000100000000000000000000000000000000000010000
  FFMA  ctrl=0x080fe20000000011  bin=0000100000001111111000100000000000000000000000000000000000010001
  FFMA  ctrl=0x000fe20000000012  bin=0000000000001111111000100000000000000000000000000000000000010010
  FFMA  ctrl=0x0c0fe20000000064  bin=0000110000001111111000100000000000000000000000000000000001100100
  FFMA  ctrl=0x000fe2000000000f  bin=0000000000001111111000100000000000000000000000000000000000001111
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

