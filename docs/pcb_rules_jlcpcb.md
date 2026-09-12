# Mainboard PCB rules — JLCPCB 4-layer, base price

Rules for the v2.5 mainboard layout, set in `mainboard/mainboard/` (`mainboard.kicad_pro` for the
constraints and net classes, the stackup in `mainboard.kicad_pcb`, extra rules in
`mainboard.kicad_dru`). Limits checked against the JLCPCB
[capabilities](https://jlcpcb.com/capabilities/pcb-capabilities) and
[impedance](https://jlcpcb.com/impedance) pages on 2026-09-12.

If you start a new `.kicad_pcb`, use **Board Setup → Import Settings** from this project so the
stackup, constraints and net classes come with it.

## Order settings

| option | value |
|---|---|
| layers | 4 |
| size | 100 × 100 mm (the board outline, unchanged) |
| thickness | 1.0 mm |
| outer / inner copper | 1 oz / 0.5 oz |
| impedance control | yes, **JLC04101H-3313** |
| min via hole / diameter | 0.3 mm / (0.4/0.45 mm) — the no-surcharge option |
| via covering | tented |
| surface finish | HASL (with lead); ENIG only if the 0.4 mm QFN pads come out uneven |
| mask / silk | green / white |

Check in the quote that 1.0 mm and the impedance stackup cost the same as 1.6 mm with no stackup
selected. If they don't, JLC04161H-3313 (1.6 mm) has the **same 0.0994 mm outer prepreg**, so
every trace width below still applies; only the core changes.

## Stackup JLC04101H-3313

| layer | material | thickness | εr |
|---|---|---|---|
| F.Mask | LPI | 0.6 mil over copper, 1.2 mil over laminate | 3.8 |
| **F.Cu** — signal | copper 1 oz | 0.035 mm | |
| prepreg | 3313 | 0.0994 mm | 4.1 |
| **In1.Cu** — GND | copper 0.5 oz | 0.0152 mm | |
| core | FR-4 | 0.7 mm | 4.6 |
| **In2.Cu** — GND | copper 0.5 oz | 0.0152 mm | |
| prepreg | 3313 | 0.0994 mm | 4.1 |
| **B.Cu** — signal | copper 1 oz | 0.035 mm | |

Every signal is a microstrip 0.1 mm above an unbroken GND plane. Stitch the two planes with GND
vias, and do not route on In1/In2 (a DRC warning flags it).

## Trace widths

Solved with a 2D field solver: 35 µm copper etched 1 mil narrower at the top, JLCPCB's solder mask
coating. Check them once in the [JLCPCB calculator](https://jlcpcb.com/pcb-impedance-calculator)
before ordering (±10 % fab tolerance).

| target | width | gap | solved |
|---|---|---|---|
| 50 Ω single-ended | **0.16 mm** | — | 51 Ω |
| 90 Ω differential (USB) | **0.16 mm** | **0.15 mm** | 90 Ω |
| 100 Ω differential (spare) | 0.12 mm | 0.15 mm | 100 Ω |
| for reference | 0.10 / 0.127 / 0.20 / 0.25 mm | — | 63 / 57 / 46 / 40 Ω |

Nothing on the mainboard is fast enough to fail at 55–60 Ω. Necking down to 0.1 mm to escape a
QFN pad is fine; route the length at 0.16 mm.

## Constraints (Board Setup → Constraints)

| rule | set | JLCPCB minimum |
|---|---|---|
| track width | 0.1 mm | 0.09 mm |
| clearance | 0.1 mm | 0.09 mm |
| via drill / diameter | 0.3 / 0.5 mm | 0.15 / 0.25 mm (below 0.3 mm drill costs more) |
| annular ring | 0.1 mm | — |
| hole to copper | 0.2 mm; PTH pad 0.3 mm, NPTH 0.25 mm (`.kicad_dru`) | via 0.2, PTH 0.28, NPTH 0.2 |
| hole to hole | 0.25 mm; PTH to PTH 0.45 mm (`.kicad_dru`) | via 0.2, pad 0.45 |
| copper to edge | 0.3 mm; pours 0.5 mm (`.kicad_dru`) | 0.2 mm, ±0.2 mm routing tolerance |
| solder mask | 1:1 openings, 0.1 mm min web | 1:1, bridge 0.10 mm (green) |
| silkscreen | text 1.0 mm / 0.15 mm stroke, 0.15 mm to pads | same |

A 0.2 mm drill with a ≥ 0.45 mm pad is also listed without surcharge. Allow it (lower
`min_through_hole_diameter`) only if the RP2350B fan-out needs it, and check the quote.

## Net classes

| class | nets | track | clearance | via |
|---|---|---|---|---|
| Default | everything else | 0.16 | 0.15 | 0.3 / 0.6 |
| Signal_50R | QSPI (both MCUs), SDIO, SWD, I2S, `/MSPC bus/*` | 0.16 | 0.2 | 0.3 / 0.6 |
| USB_90R | USB D+/D− (MCU, hub ports, probe, connectors) — pair 0.16 / 0.15 | 0.16 | 0.2 | 0.3 / 0.6 |
| XOSC | crystal nets; 0.3 mm to other tracks (`.kicad_dru`) | 0.2 | 0.2 | 0.3 / 0.6 |
| Power_3A | `+5V_CONN_*`, `+5V_DBG`, `+5V` (VSYS) | 1.2 | 0.25 | 0.4 / 0.8 |
| Power | other `+*` rails, buck switch nodes | 0.5 | 0.2 | 0.4 / 0.8 |
| GND | GND | 0.3 | 0.15 | 0.3 / 0.6 |

Widths are for 1 oz outer copper, 10 °C rise (IPC-2221): 0.5 mm ≈ 1.4 A, 1.2 mm ≈ 2.7 A, above
the J_PWR eFuse's 2.57 A maximum limit. Use pours for the
input path where there is room, and several vias per layer change on Power_3A.

USB pairs also get a 3 mm uncoupled limit and 0.5 mm skew per pair. KiCad's diff-pair router only
pairs nets ending in `+`/`-` or `P`/`N`: rename `USB_DP`/`USB_DM` style local nets (e.g. between the
series resistors and the RP2040) before routing.

## Slot connectors

The card sockets may hang over the board edge. The SMD pads have to stay on the board with 0.3 mm
to the edge; the plastic body can overhang.
