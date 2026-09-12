# Mainboard PCB — JLCPCB 4-layer setup

Everything needed to lay out and order the v2.5 mainboard at JLCPCB's base price: board geometry,
stackup, order form, file export, trace widths, design rules and net classes.

All of it is already set in `mainboard/mainboard/`:

| file | holds |
|---|---|
| `mainboard.kicad_pcb` | stackup, board outline, Gerber plot settings |
| `mainboard.kicad_pro` | constraints, track/via/diff-pair presets, net classes and their patterns |
| `mainboard.kicad_dru` | custom DRC rules (hole clearances, pours, USB pairs, crystal, planes) |

If you start a new `.kicad_pcb` instead of clearing the old one, use **Board Setup → Import
Settings** from this project so the stackup, constraints and net classes come with it.

Limits checked against JLCPCB's [capabilities](https://jlcpcb.com/capabilities/pcb-capabilities),
[impedance](https://jlcpcb.com/impedance) and
[KiCad export](https://jlcpcb.com/help/article/how-to-generate-gerber-and-drill-files-in-kicad-8)
pages on 2026-09-12.

---

## 1. Board

The v2.5 layout is new, but the mechanics stay as on v2. Coordinates are KiCad page coordinates
in mm (the board's top-left corner is at 50, 50).

| item | value |
|---|---|
| outline | 100 × 100 mm rectangle, 50,50 → 150,150, square corners, on `Edge.Cuts` |
| layers | 4: F.Cu signal, In1.Cu GND, In2.Cu GND, B.Cu signal |
| thickness | 1.0 mm (1.03 mm computed with mask) |
| mounting holes, M3 (3.2 mm, DIN965 pad) | (55, 55), (145, 55), (55, 130), (145, 130) |
| card standoffs, M2 (2.2 mm) | (74.5, 53.5), (111, 53.5), (137.78, 53.5) |

### Card slots

Cards sit over the top half of the board. Each socket is a vertical SMD 1.27 mm pin socket, rotated
90° so the rows run along X, centred at **y = 94.3**. The card's M2 standoff is on the same X, near
the top edge.

| v2 socket | footprint | centre X | card area (F.Silkscreen outline) |
|---|---|---|---|
| J13 | `PinSocket_2x15_P1.27mm_Vertical_SMD` | 73.775 | x 50.19–97.69 (47.5 mm), y 50.19–100.19 |
| J14 | `PinSocket_2x10_P1.27mm_Vertical_SMD` | 111.5 | x 99.5–123.5 (24 mm), y 50.19–100.19 |
| J15 | `PinSocket_2x10_P1.27mm_Vertical_SMD` | 137.78 | x 125.78–149.78 (24 mm), y 50.19–100.19 |

Socket pads are 2.1 × 0.75 mm, row centres ±1.8 mm from the socket axis; courtyard ±3.35 mm across.
A 2×15 is 20.05 mm long, a 2×10 13.7 mm.

Larger sockets may overhang the card area or the board edge. The SMD pads must stay on the board
and 0.3 mm from the edge; the plastic body can hang over. Keep the socket centre line and the
standoff X positions, or move the cards' holes with them.

Update the silkscreen label (still reads "MSPC rev 2.0a … 2024").

---

## 2. Stackup — JLC04101H-3313

Set in **Board Setup → Board Stackup**; this is what JLCPCB builds when you select the stackup on the
order.

| # | layer | use | material | thickness | εr |
|---|---|---|---|---|---|
| | F.Mask | | LPI, green | 0.6 mil (15 µm) on copper, 1.2 mil on laminate | 3.8 |
| 1 | **F.Cu** | signal, components | copper 1 oz | 0.035 mm | |
| | dielectric 1 | | prepreg 3313 × 1 | **0.0994 mm** | 4.1 |
| 2 | **In1.Cu** | GND plane | copper 0.5 oz | 0.0152 mm | |
| | dielectric 2 | | core | 0.7 mm | 4.6 |
| 3 | **In2.Cu** | GND plane | copper 0.5 oz | 0.0152 mm | |
| | dielectric 3 | | prepreg 3313 × 1 | **0.0994 mm** | 4.1 |
| 4 | **B.Cu** | signal, microSD | copper 1 oz | 0.035 mm | |
| | B.Mask | | LPI, green | as F.Mask | 3.8 |

Every signal is a microstrip 0.1 mm over an unbroken GND plane:

- Fill In1 and In2 completely with GND; no tracks there (DRC warns).
- Stitch the planes with GND vias along the edges, around the switcher and next to every signal via
  that changes layer (the return current has to change plane too).
- Power goes on F.Cu/B.Cu as pours and wide tracks.

Board thickness tolerance at JLCPCB is ±0.1 mm.

---

## 3. JLCPCB order form

Field names follow JLCPCB's quote page; the form changes now and then, so match by meaning. Leave
anything not listed at its default.

| field | value | why |
|---|---|---|
| Base material | FR-4 | |
| Layers | 4 | |
| Dimensions | 100 × 100 mm | price steps up above 100 × 100 mm |
| PCB qty | 5 | minimum |
| Different design | 1 | |
| Delivery format | Single PCB | no panel |
| PCB thickness | **1.0 mm** | |
| PCB color | Green | |
| Silkscreen | White | |
| Material type | leave the default | |
| Surface finish | **HASL (with lead)** | cheapest; ENIG only if 0.4 mm QFN pads come out uneven |
| Outer copper weight | 1 oz | |
| Inner copper weight | 0.5 oz | |
| Specify stackup / impedance control | **Yes — JLC04101H-3313** | trace widths below assume it |
| Via covering | **Tented** | free; the vias are not in pads |
| Min via hole size / diameter | **0.3 mm / (0.4/0.45 mm)** | the no-surcharge option; board minimum is 0.3 / 0.5 |
| Board outline tolerance | ±0.2 mm (regular) | |
| Confirm production file | No | |
| Mark on PCB | order number (default) | removing it costs extra |
| Electrical test | Flying probe, fully tested | free |
| Gold fingers, castellated holes, edge plating, blind slots | No | |
| 4-wire Kelvin test, paper between boards | No | |

Before paying, check that 1.0 mm with the stackup selected costs the same as 1.6 mm with no stackup.
If not, **JLC04161H-3313** (1.6 mm) has the same 0.0994 mm outer prepreg, so every trace width
still applies; only the core becomes 1.265 mm.

---

## 4. Fabrication files

Gerbers and drill files go to `mainboard/outputs/gerbers/` (ignored by git). From the repository
root:

```sh
cd mainboard
kicad-cli pcb export gerbers \
  --layers F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts \
  --subtract-soldermask --no-x2 --check-zones \
  -o outputs/gerbers/ mainboard/mainboard.kicad_pcb
kicad-cli pcb export drill \
  --format excellon --drill-origin absolute --excellon-units mm \
  --excellon-zeros-format decimal --excellon-oval-format alternate \
  --generate-map --map-format gerberx2 \
  -o outputs/gerbers/ mainboard/mainboard.kicad_pcb
cd outputs && zip -j mainboard_gerbers.zip gerbers/*
```

Protel extensions (`.gtl`, `.g1`, `.g2`, `.gbl`, …) are kicad-cli's default and what JLCPCB
prefers. Upload the zip; in the viewer check that 4 copper layers are detected in the order
F.Cu, In1, In2, B.Cu.

The same from the GUI (**File → Fabrication Outputs**):

- **Gerbers:** the layers above; *Use Protel filename extensions*, *Subtract soldermask from
  silkscreen*, *Check zone fills before plotting* on; X2 format off.
- **Drill:** Excellon, absolute origin, millimetres, decimal zeros, oval holes in alternate drill
  mode, map file on.

Run DRC with zero errors and refill zones (**B**) before exporting.

---

## 5. Trace widths

Solved with a 2D field solver: 35 µm copper etched 1 mil narrower at the top, JLCPCB's solder mask.
Check them once in the [JLCPCB calculator](https://jlcpcb.com/pcb-impedance-calculator)
(4 layers, 1.0 mm, JLC04101H-3313, top layer, reference L2) before ordering. Fab tolerance ±10 %.

| target | width | gap | solved |
|---|---|---|---|
| 50 Ω single-ended | **0.16 mm** | — | 51 Ω |
| 90 Ω differential (USB) | **0.16 mm** | **0.15 mm** | 90 Ω |
| 100 Ω differential (spare) | 0.12 mm | 0.15 mm | 100 Ω |
| for reference | 0.10 / 0.127 / 0.20 / 0.25 mm | — | 63 / 57 / 46 / 40 Ω |

Nothing on the mainboard is fast enough to fail at 55–60 Ω. Necking to 0.1 mm to escape a QFN pad
is fine; route the length at 0.16 mm.

---

## 6. Constraints (Board Setup → Constraints and `.kicad_dru`)

| rule | set | JLCPCB minimum |
|---|---|---|
| track width | 0.1 mm | 0.09 mm |
| clearance | 0.1 mm | 0.09 mm |
| via drill / diameter | 0.3 / 0.5 mm | 0.15 / 0.25 mm (below 0.3 mm drill costs more) |
| via annular ring | 0.1 mm | — |
| hole to copper | 0.2 mm; PTH pad 0.3 mm, NPTH 0.25 mm (`.kicad_dru`) | via 0.2, PTH 0.28, NPTH 0.2 |
| hole to hole | 0.25 mm; PTH to PTH 0.45 mm (`.kicad_dru`) | via 0.2, pad 0.45 |
| copper to edge | 0.3 mm; pours 0.5 mm (`.kicad_dru`) | 0.2 mm, ±0.2 mm routing tolerance |
| solder mask | 1:1 openings, 0.1 mm min web | 1:1, bridge 0.10 mm (green) |
| silkscreen | text 1.0 mm / 0.15 mm stroke, 0.15 mm to pads | same |
| blind / buried / micro vias | off | not on this order |

Presets:

| | values |
|---|---|
| tracks | 0.1, 0.12, 0.16, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0 mm |
| vias (drill / diameter) | 0.3 / 0.5, 0.3 / 0.6, 0.4 / 0.8, 0.5 / 1.0 mm |
| diff pairs (width / gap) | 0.16 / 0.15 (90 Ω), 0.12 / 0.15 (100 Ω) |
| zones | min width 0.2 mm, thermal gap and spoke 0.3 mm |

A 0.2 mm drill with a ≥ 0.45 mm pad is also listed without surcharge. Allow it (lower the minimum
through-hole diameter) only if the RP2350B fan-out needs it, and check the quote.

---

## 7. Net classes

| class | nets | track | clearance | via |
|---|---|---|---|---|
| Default | everything else | 0.16 | 0.15 | 0.3 / 0.6 |
| Signal_50R | QSPI (both MCUs), SDIO, SWD, I2S, `/MSPC bus/*` | 0.16 | 0.2 | 0.3 / 0.6 |
| USB_90R | USB D+/D− (MCU, hub ports, probe, connectors) — pair 0.16 / 0.15 | 0.16 | 0.2 | 0.3 / 0.6 |
| XOSC | crystal nets; 0.3 mm to other tracks (`.kicad_dru`) | 0.2 | 0.2 | 0.3 / 0.6 |
| Power_3A | `+5V_CONN_*`, `+5V_DBG`, `+5V` (VSYS) | 1.2 | 0.25 | 0.4 / 0.8 |
| Power | other `+*` rails, buck switch nodes | 0.5 | 0.2 | 0.4 / 0.8 |
| GND | GND | 0.3 | 0.15 | 0.3 / 0.6 |

Widths are for 1 oz outer copper, 10 °C rise (IPC-2221): 0.5 mm ≈ 1.4 A, 1.2 mm ≈ 2.7 A, above the
J_PWR eFuse's 2.57 A maximum limit. Use pours for the input path where there is room, and several
vias per layer change on Power_3A.

USB pairs also get a 3 mm uncoupled limit and 0.5 mm skew per pair. KiCad's diff-pair router only
pairs nets ending in `+`/`-` or `P`/`N`: rename local nets like `Net-(U803-USB_DP)` (between the
RP2040 and its series resistors) before routing.

Classes are assigned by net name pattern (**Schematic Setup → Net Classes**). A renamed net or a new
sheet falls back to Default; check the assignment after schematic changes.
