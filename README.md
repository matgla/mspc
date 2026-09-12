# MSPC

Hardware for MSPC: a small modular computer built from a mainboard and extension cards that plug
into the MSPC bus. The mainboard runs [yasos](https://github.com/matgla/yasos.zig).

## Versions and branches

| branch | contents |
|---|---|
| `main` | the current hardware: the v2 boards (same as `mspc_v2`) |
| `mspc_v1` | the first generation, kept for reference |
| `mspc_v2` | the v2 boards as built |
| `mspc_v2.5` | the next revision, work in progress; `main` moves there when it is done and that point is tagged `mspcv2.5` |

## Repository layout

Every board is a KiCad project with its fabrication outputs next to it in `outputs/`.

| directory | board |
|---|---|
| `mainboard/` | Mainboard (project in `mainboard/mainboard/`). RP2350B (80-QFN), QSPI flash and PSRAM, microSD, HS8836A USB hub, TLV320DAC3203 audio DAC, 12 V input with BD9D321EFJ bucks, three MSPC bus sockets (2×10 and 2×15, 1.27 mm). 4 copper layers. |
| `gpu_vga_extension/` | VGA card: RP2350A (60-QFN), 2× W25Q32JV flash, AP2112K 3.3 V regulator, resistor-ladder RGB DAC and DB15 connector. 24 × 50 mm, 4 layers. |
| `gpu_dvi_card/` | DVI card: same core as the VGA card, with an HDMI-type receptacle carrying DVI. 24 × 50 mm, 4 layers. |
| `fpga_extension/` | FPGA card (project in `fpga_extension/fpga_extension/`): Lattice ECP5 LFE5UM-85F, W958D8NBYA HyperRAM, QSPI configuration flash, microSD, and an on-board RP2040 USB-C JTAG programmer. 47.5 × 50 mm, 6 layers. |
| `bus_adapter_20pin/` | Passive breakout from the 2×10 1.27 mm bus socket to two 1×10 2.54 mm headers. |
| `bus_adapter_30pin/` | Passive breakout from the 2×15 1.27 mm bus socket to two 1×15 2.54 mm headers. |
| `simulation/` | 2022 SPICE sandbox: KiCad/ngspice schematics (audio filter, power switch) and PySpice scripts. Not a board. |
| `libs/` | KiCad libraries shared by all boards (see below). |
| `docs/` | `mspc_v2.5_revision_plan.md` (v2.5 mainboard changes), `pcb_rules_jlcpcb.md` (mainboard stackup, JLCPCB order settings, fab export, design rules). |

## Working with the files

- The projects are in **KiCad 8** format. Newer KiCad versions upgrade them on the first save, and
  the result no longer opens in KiCad 8. Do such an upgrade as its own commit, with no design
  changes, so later diffs stay readable.
- Schematic PDFs (`outputs/schematic.pdf`) are tracked, so the schematics can be read without
  KiCad. Gerbers and drill files are not: plot them from the KiCad project when ordering a board.
  Per-machine files (backups, caches, `*.kicad_prl`, lock files, local history) are ignored too.

## Libraries

Every board uses the libraries in `libs/`. Each project's `sym-lib-table`, `fp-lib-table` and
`design-block-lib-table` point there through `${KIPRJMOD}`, so nothing has to be set up in KiCad.

| path | contents |
|---|---|
| `libs/symbols/` | symbol libraries, one per file; the nickname is the file name (`project_power`, `RP2040`, …) |
| `libs/footprints.pretty/` | footprints (nickname `footprints`) |
| `libs/RP2350_60QFN_minimal.pretty/`, `libs/RP2350_80QFN_minimal.pretty/` | RP2350 footprints and the small parts around them |
| `libs/3dmodels/` | STEP models; the footprints do not reference them yet |
| `libs/mspc.kicad_blocks/` | design blocks (KiCad 9+): circuits to reuse between boards, placed from the Design Blocks panel of the schematic editor. `TLV62569DRL_buck` is the FPGA card's 1.1 V buck. |

Add new parts to `libs/`, not to a project. A new symbol library file has to be added to the
`sym-lib-table` of every project.

## Known issues

- Some outputs are older than the boards they sit next to: the mainboard stencil (from v1), the
  GPU cards' schematic PDFs and stencils (from before the VGA/DVI split), and the FPGA card's
  schematic PDF. Regenerate them before the next order.
- The silkscreen of the 20-pin bus adapter reads "2x15" (copied from the 30-pin adapter).
