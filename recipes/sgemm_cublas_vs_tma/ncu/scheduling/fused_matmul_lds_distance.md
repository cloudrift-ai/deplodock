# Scheduling analysis: /tmp/sass_dbg/bench kernel='fused_matmul'

## Total SASS instructions in matched kernel: 4532

## Top mnemonics

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

## LDS-to-first-FFMA-consumer distance (256 LDS instructions, 256 with consumer)

- FFMAs between LDS and first consuming FFMA: min=0, median=40, mean=44.6, max=170
- Total instructions: min=6, median=45, mean=50.9, max=171

| FFMAs between LDS and consumer | Count |
|---|---:|
| [0, 0) | 0 |
| [0, 5) | 3 |
| [5, 10) | 1 |
| [10, 20) | 13 |
| [20, 40) | 110 |
| [40, 80) | 117 |
| [80, 160) | 11 |
| [160, 320) | 1 |
| [320, ∞) | 0 |

## LDS-to-next-LDS spacing (FFMAs between consecutive LDS, 255 pairs)

- min=0, median=5, mean=13.2, max=223

| FFMAs between consecutive LDS | Count |
|---|---:|
| [0, 0) | 0 |
| [0, 1) | 14 |
| [1, 2) | 2 |
| [2, 4) | 81 |
| [4, 8) | 51 |
| [8, 16) | 48 |
| [16, 32) | 37 |
| [32, ∞) | 22 |

## Inner loop excerpt (instructions 1315..1427, FFMA density 100.0%)

```
  /*5290*/  FFMA         R162, R37, R148.reuse, R162
  /*52a0*/  FFMA         R165, R38, R148.reuse, R165
  /*52b0*/  FFMA         R164, R39, R148, R164
  /*52c0*/  FFMA         R3, R41.reuse, R152, R166
  /*52d0*/  LDS.128      R36, [R15+0x8400]
  /*52e0*/  FFMA         R168, R41.reuse, R153, R168
  /*52f0*/  FFMA         R167, R41.reuse, R154, R167
  /*5300*/  FFMA         R40, R41, R155, R40
  /*5310*/  FFMA         R5, R45.reuse, R152, R172
  /*5320*/  FFMA         R170, R45.reuse, R153, R170
  /*5330*/  FFMA         R169, R45.reuse, R154, R169
  /*5340*/  FFMA         R176, R45, R155, R176
  /*5350*/  FFMA         R9, R49.reuse, R152, R174
  /*5360*/  FFMA         R44, R49.reuse, R153, R44
  /*5370*/  FFMA         R171, R49.reuse, R154, R171
  /*5380*/  FFMA         R180, R49, R155, R180
  /*5390*/  FFMA         R17, R53.reuse, R152, R178
  /*53a0*/  FFMA         R48, R53.reuse, R153, R48
  /*53b0*/  FFMA         R173, R53.reuse, R154, R173
  /*53c0*/  FFMA         R182, R53, R155, R182
  /*53d0*/  FFMA         R21, R57.reuse, R152, R184
  /*53e0*/  FFMA         R52, R57.reuse, R153, R52
  /*53f0*/  FFMA         R175, R57.reuse, R154, R175
  /*5400*/  FFMA         R188, R57, R155, R188
  /*5410*/  FFMA         R23, R61.reuse, R152, R186
  /*5420*/  FFMA         R56, R61.reuse, R153, R56
  /*5430*/  FFMA         R177, R61.reuse, R154, R177
  /*5440*/  FFMA         R192, R61, R155, R192
  /*5450*/  FFMA         R33, R65.reuse, R152, R60
  /*5460*/  FFMA         R190, R65.reuse, R153, R190
  /*5470*/  FFMA         R179, R65.reuse, R154, R179
  /*5480*/  FFMA         R64, R65, R155, R64
  /*5490*/  FFMA         R41, R69.reuse, R152, R194
  /*54a0*/  FFMA         R196, R69.reuse, R153, R196
  /*54b0*/  FFMA         R181, R69.reuse, R154, R181
  /*54c0*/  FFMA         R200, R69, R155, R200
  /*54d0*/  FFMA         R45, R73.reuse, R152, R68
  /*54e0*/  FFMA         R198, R73.reuse, R153, R198
  /*54f0*/  FFMA         R183, R73.reuse, R154, R183
  /*5500*/  FFMA         R204, R73, R155, R204
```

### Inner-loop mnemonic counts (full body, 112 instructions)

| Mnemonic | Count |
|---|---:|
| `FFMA` | 112 |

