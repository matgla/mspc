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

## Working with the files

- The projects are in **KiCad 8** format. Newer KiCad versions upgrade them on the first save, and
  the result no longer opens in KiCad 8. Do such an upgrade as its own commit, with no design
  changes, so later diffs stay readable.
- Fabrication outputs (gerbers, drill files, schematic PDFs, stencils) are tracked on purpose:
  boards are ordered from them. Per-machine files (backups, caches, `*.kicad_prl`, lock files,
  local history) are ignored.

## Known issues

- Some outputs are older than the boards they sit next to: the mainboard stencil (from v1), the
  GPU cards' schematic PDFs and stencils (from before the VGA/DVI split), and the FPGA card's
  schematic PDF. Existing gerbers keep the file names of the old project names
  (`gpu_extension-*`, `bus_adapter_10pin-*`). Regenerate them before the next order.
- The silkscreen of the 20-pin bus adapter reads "2x15" (copied from the 30-pin adapter).
- The libraries are copied into each project and have diverged.
