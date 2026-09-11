# MSPC v2.5 — mainboard revision plan

Summary of the decisions made on 2026-09-11. The board stays on the **RP2350B**. Goals of the
revision: power from USB-C instead of 12 V, fewer ICs, a simpler and faster link to the cards, and
an on-board debug probe.

Confidence marks: **[d]** = from a datasheet or specification, **[m]** = from a published
measurement, **[e]** = estimate. Check everything marked [e] before ordering the PCB.

---

## 1. Summary of changes

| area | v2 | v2.5 |
|---|---|---|
| power input | 12 V jack, 2× BD9D321EFJ, AP2210 (MCU 3.3 V), LD1117S50 (USB 5 V) | **2× USB-C at 5 V** (data + power, power only), priority power mux, 3.3 V buck for a 5 V input, SY6280 load switches |
| programming / console | external debug probe | **on-board RP2040 running debugprobe** (SWD + UART over one cable) |
| USB hub | HS8836A (SOP-16) | **CH334R** (QSOP-16) or **CH334F** (QFN-24) |
| audio | TLV320DAC3203 + I2C | **PCM5100A** (I2S, no I2C, line-level output) |
| I2C / J7 | I2C connector | **removed** (optional pads for an RTC) |
| flash | W25Q64JV (8 MB) | **W25Q128JV-IQ (16 MB)**, room for A/B rootfs slots for OTA |
| PSRAM | APS6404L | unchanged |
| MSPC bus | shared 16-bit, 6× CS, 74HCS151 + 74LVC139 | **point-to-point link per slot**, no CS, no glue logic |
| slots | J13 16-bit, J14 8-bit (upper byte), J15 8-bit | **3 slots: VGA/DVI (4-bit), FPGA (8-bit), network (4-bit)** |
| I/O voltage | 3.3 V | unchanged (1.8 V rejected, see §11) |
| CPU clock | 150 MHz (defconfig) | 150 MHz until the flash sampling fix lands (see §9) |

Removed parts: BD9D321EFJ ×2, AP2210, LD1117S50, 12 V jack, TLV320DAC3203, HS8836A, 74HCS151, 74LVC139.

---

## 2. Power

### 2.1 Trap: the BD9D321EFJ cannot run from 5 V
The BD9D321EFJ output is limited to **0.65 × VIN** [d]. From 5 V it gives at most 3.25 V (not
3.3 V), and the 5 V rail would need ≥ 7.7 V in. A board powered from USB at 5 V therefore needs a
different 3.3 V converter.

### 2.2 Topology

```
J_DBG (USB-C, data) ─ TVS ─┐                 ┌─► 3.3 V buck ─► MCU, probe, memories, SD, audio
  └ D+/D- → RP2040 probe   ├─► TPS2116 ─ eFuse (soft start)
J_PWR (USB-C, power) ─ TVS ─┘ (J_PWR priority) ├─► SY6280 (EN) ─► 5 V to the card slots
                                               └─► SY6280 (EN) ─► USB ports (CH334 hub)
```

- **2× USB-C, each with two separate 5.1 kΩ ±10 % resistors** (CC1 → GND and CC2 → GND) [d].
  Never a single shared one — that was the Raspberry Pi 4 rev 1.0 bug (it did not work with
  e-marked cables) [m].
- **A TVS on each input:** TVS0500 (VBR ≥ 7.5 V) or SMAJ6.0A. **Not SMAJ5.0A** — its 5.0 V
  standoff is below the 5.5 V a charger may deliver [d].
- **TPS2116 priority power mux** (1.6–5.5 V, 2.5 A, 40 mΩ) [d] with priority on J_PWR: when a
  charger is connected it carries the whole load and the PC port is not loaded. A plain diode-OR
  would split the current unpredictably. Status pin to a GPIO. For more current headroom: TPS2121
  (2.8–22 V, 1–4.5 A limit) [d] — not needed without a speaker.
- **eFuse with soft start** after the mux (e.g. TPS25947 or the 5 V TPS2595x variant) [d].
  A USB-C source without PD allows only 1–10 µF directly on VBUS at attach [d]; the board's larger
  capacitance must sit behind the soft start (e.g. 100 µF with a 1 V/ms ramp → 100 mA inrush [e]).
- **3.3 V buck for a 5 V input**, 2 A class, e.g. TLV62569 (used on the ULX3S and OrangeCrab) —
  check the TI datasheet. The 3.3 V load is a few hundred mA [e], so there is plenty of margin.
- **5 V for the cards and USB ports straight from VBUS through SY6280 switches** (2.4–5.5 V, 2 A,
  reverse blocking, limit I_lim = 6800 / R_set) [d]. Each SY6280 EN gets a 100 kΩ pull-down so
  everything is off during reset (RP2350 pads come out of reset with a pull-down and isolation [d]).
  Check the SY6280 variant (they differ in EN polarity).
- **Capacitance on the USB ports' VBUS:** 100–150 µF (USB spec: ≥ 120 µF, ≤ 330 mV droop on attach)
  [d], behind the switch.

### 2.3 Power budget in firmware
- The ADC reads the CC lines of both connectors (through a divider). Spec thresholds [d]:
  - < 0.66 V → "Default" source (USB 2.0: 500 mA, USB 3.x: 900 mA),
  - 0.70–1.16 V → 1.5 A,
  - 1.31–2.04 V → 3.0 A.
- The budget follows the active input (TPS2116 status pin).
- With a weak source yasos does not enable the card or USB-port power (SY6280 EN).

### 2.4 Budget (current drawn from 5 V)

| load | typical | peak | confidence |
|---|---|---|---|
| RP2350 @150 MHz + PSRAM + flash | 20–40 mA at 3.3 V | | [d] RP2350 table 1446, APS6404L, W25Q128JV |
| SD card | 20–80 mA at 3.3 V | 150–300 mA at 3.3 V | [d]+[m] SD spec, Gough Lui |
| RP2040 probe | ~20–30 mA at 3.3 V | | [e] |
| CH334 hub (full-speed upstream) | ~20 mA | | [d] |
| **mainboard total** | **~110 mA** | **~300 mA** | [e] |
| keyboard + mouse + flash drive (FS) | 95–220 mA | RGB keyboard up to 500 mA | [m] |
| FPGA card (ECP5-45F, soft CPU, APS256, RP2040) | ~100–170 mA | ~250 mA | [m]+[e] OrangeCrab/ULX3S |
| VGA/DVI card (RP2350 @252–300 MHz) | ~125–140 mA | | [e], incl. 55 mA on the DVI/HDMI +5 V pin [d] |
| WiFi card (RM2 + RP2040) | ~80–120 mA (transmitting) | ~270 mA; bursts up to ~1 A at 3.3 V | [d] CYW43439 |
| **total** | **~0.55–0.8 A (3–4 W)** | **~1.2 A (6 W)** | [e] |

The audio output is line level (no speaker) — its draw is negligible.

| source | limit | enough for |
|---|---|---|
| USB-A 2.0 on a PC | 0.5 A | mainboard + probe + keyboard/mouse/flash drive (~250–350 mA) |
| USB-A 3.x | 0.9 A | + one card |
| **USB-C 1.5 A** | 7.5 W | **the whole set** |
| USB-C 3 A | 15 W | the whole set with a large margin |

Note: a USB-C source at 3 A may drop to 4.0 V at the end of the cable [d], while USB devices need
≥ 4.75 V (enumeration from 4.4 V) [d] — use a short, thick cable.

---

## 3. On-board RP2040 debug probe
- **debugprobe** firmware (CMSIS-DAP + CDC UART) — the same flow as today (OpenOCD, `flash.sh`).
- Connections to the RP2350: **SWCLK, SWDIO, RUN** and the console **UART** (TX/RX).
- **Power the probe from the board's 3.3 V rail**, not from J_DBG's VBUS — otherwise, with only
  the charger connected, the RP2350 would back-power the unpowered probe through SWD/UART.
- **VBUS detection on J_DBG** (divider → RP2040 GPIO, VBUS detect function): a self-powered device
  must not assert the D+ pull-up while the PC is disconnected.
- ESD protection on D+/D- (e.g. USBLC6-2). Total cost ~$2–3 [e].
- Known limitation: debugprobe drops bytes at 921600 baud (from earlier measurements). MSPC runs at
  115200 — safe; test a faster console through the probe first.

---

## 4. USB hub
- **WCH CH334R** (QSOP-16, 0.635 mm pitch — hand-solderable) or **CH334F** (QFN-24, 4×4 mm, ganged
  PWREN/OVCUR) [d]. All CH334 variants: crystal optional, built-in 5 V → 3.3 V LDO [d].
- Leave an unpopulated footprint for a 12 MHz crystal.
- The RP2350 is a full-speed host, so the hub runs in FS mode: ~20 mA [d].
- The ports are powered from the board's own 5 V through an SY6280 (limit e.g. 1–1.5 A for all), so
  the 100 mA-per-port limit of a bus-powered hub does not apply.

---

## 5. Audio: PCM5100A
- **TI PCM5100A** — the same part as on the Pimoroni Pico VGA Demo Base [d].
- I2S on 3 pins (BCK, LRCK, DIN); no system clock needed (internal PLL from BCK) [d].
- **No I2C:** FMT (low = I2S), XSMT (mute), FLT, DEMP are set by pins [d].
- **2.1 V RMS ground-centred** output (DirectPath charge pump) — no DC-blocking capacitors, no
  op-amp and no filter [d]. Load ≥ 1 kΩ, i.e. a line output [d].
- SNR 100 dB (PCM5101A 106 dB, PCM5102A 112 dB) [d].
- 3.3 V supply (AVDD, CPVDD, DVDD); DVDD ~7–8 mA at 48 kHz [d].
- **XSMT to a GPIO** — no pop at start-up (or a simple RC if pins are short).
- Price ~$1.63 each (LCSC, TSSOP-20) [d].
- The I2S driver (PIO, like `audio_i2s` from pico-extras) can be written and tested **now** on the
  Pico Plus 2 + VGA Demo Base (`configs/pimoroni_pico_plus2_and_vga_defconfig` in yasos).

---

## 6. Memories
- **W25Q128JV-IQ flash (16 MB, 133 MHz SDR)**: kernel ~0.3 MB + rootfs 2 × 3.4 MB (A/B) = ~7.1 MB.
  The DTR variant (-IM) is not needed: DTR on the W25Q64JV/W25Q128JV runs only up to 66 MHz [d],
  and the RP2350 QMI halves the clock in DTR mode anyway — no gain.
- **APS6404L PSRAM** unchanged (the RP2350 QMI supports only 4 lines; the x8/x16 1.8 V APS256XXN does
  not fit — it goes to the FPGA card).
- W25Q64/W25Q128 packages are interchangeable, but the firmware must know the real size (§9).

---

## 7. Link to the cards: point-to-point

### 7.1 Principle
Instead of a shared bus — **a separate link to each slot**, each with its own PIO state machine and
DMA channel (the RP2350 has 12 state machines):
- no stubs → easier to run a high clock,
- transfers to different cards in parallel,
- a hung card blocks only itself,
- no chip selects and no glue logic.

Every card carries an **RP2040/RP2350 bridge** that speaks a common protocol; everything specific to
the card sits behind it. A command protocol, e.g. `[CMD][ADDR][LEN][DATA…]`, clocked by the host.
The card identifies itself on an ID command → yasos picks the driver. Card firmware is updated over
the link.

### 7.2 Slots

| slot | card | link | extra lines | bridge |
|---|---|---|---|---|
| 1 | VGA / DVI | 4-bit + CLK | INT; optionally REQ/GNT (shared-PSRAM experiment) | RP2350B (HSTX for DVI, framebuffer in SRAM) |
| 2 | FPGA (ECP5-45F) | **8-bit (GPIO12–19)** + CLK | INT, SEL (FPGA / RP2040 for configuration) | RP2040 (bitstream loading) |
| 3 | network (Ethernet / WiFi) | 4-bit + CLK | INT | RP2040 |
| — | shared | | RESET | |

**INT doubles as DET:** 100 kΩ pull-down on the host side, the card drives it high once running,
an interrupt is a low level. The host asks the card for its status; no answer → card removed or
hung. Ignore INT for ~100 ms after RESET is released. Level-triggered interrupt with masking; poll a
hung card every few tens of seconds (slot reset + ID). No hot-plug unless the connectors are
designed for it.

### 7.3 Throughput (DDR: link clock = ½ system clock)

| system clock | 4-bit DDR | 8-bit DDR |
|---|---|---|
| 150 MHz (today) | 75 MB/s | 150 MB/s |
| 200 MHz | 100 MB/s | 200 MB/s |
| 300 MHz | ~150 MB/s | ~300 MB/s (pad ceiling: 300 Mbit/s/pin [d]) |

Needs: 640×480×8 bpp @60 Hz graphics = 18.4 MB/s (full frames), 16 bpp = 36.9 MB/s; 100 Mbit
Ethernet = 12.5 MB/s. Above ~66 MB/s the data must be in SRAM (PSRAM on the QMI gives ~66 MB/s).

- **Writes** are easy (the host drives clock and data).
- **Reads**: PIO samples in whole-cycle steps (6.7 ns at 150 MHz) — **the card calibrates the phase**
  (PLL / pin delays on the FPGA, a training pattern at start-up). On the host side, bypass the input
  synchroniser.

### 7.4 Experiment: HSTX writes + PIO reads on the same pins (FPGA)
- A GPIO input is "always connected" to PIO regardless of FUNCSEL [d] → PIO can read HSTX pins.
- Direction: `OEOVER` in `GPIOx_CTRL` disables the HSTX output while reading [d]; switch at block
  boundaries (a few register writes, hundreds of ns).
- HSTX: GPIO12–19, output only, DDR up to 150 MHz, clock generator on any of these pins [d].
  **8 bits + clock = 9 lines** → take the link clock from PIO on a separate pin (e.g. GPIO20), with
  the phase calibrated by the FPGA and a start marker per data block. Or 7 bits + an HSTX clock.
- Read up on the PIO↔HSTX "coupled mode" in the RP2350 datasheet.

### 7.5 Signal integrity
3.3 V; 22–33 Ω series resistors at the drivers; short, impedance-controlled traces; a ground every
2–4 signals in the connectors.

---

## 8. Cards (power requirements and notes)

**FPGA — ECP5 LFE5U-45F + APS256XXN + QSPI flash + RP2040**
- Rails: **1.1 V** (VCC, buck ≥ 1 A — start-up current is unpublished), **1.8 V** (APS256XXN and the
  FPGA bank that talks to it), **2.5 V** (VCCAUX), **3.3 V** (VCCIO) [d].
- Sequencing: with configuration from SPI flash, VCCIO8 must come up before or together with
  VCC/VCCAUX [d].
- Draw: idle ~0.16–0.18 W, soft CPU @50–100 MHz ~0.3–0.6 W (FPGA) [m]+[e].
- APS256XXN: up to 26 mA (x8) / 33 mA (x16) at 200 MHz, 1.8 V [d].

**VGA/DVI — RP2350B + PSRAM + QSPI flash**
- DVI through HSTX (RP2350 only), VGA through a resistor ladder driven by PIO (~30 mA peak on full
  white [e]).
- 640×480×8 bpp framebuffer = 307 KB in SRAM (the RP2040 has too little — 264 KB).
- Data from the host: **copy** (link → DMA → card SRAM/PSRAM). Shared PSRAM with REQ/GNT only as an
  experiment: the PSRAM on the card's PIO pins (not the QMI — the QMI shares pins with the card's
  flash), and the picture not scanned out of PSRAM.
- DVI/HDMI connector +5 V pin: ≥ 55 mA [d].

**Network — RP2040 + W5500 and/or RM2**
- **W5500** (SPI, hardware TCP/IP, 8 sockets, 32 KB buffer) — yasos needs no TCP stack.
- **RM2** (CYW43439, WiFi 4 + BT 5.2, SPI, $4, 14.5×16.5 mm, antenna) [d]. Vin 3.0–4.8 V, full RF
  performance from 3.2 V [d] → **feed the RM2 from 3.6–4.2 V** with its own regulator on the card;
  local capacitance for ~1 A bursts [d]. cyw43 driver + lwIP on the RP2040.

---

## 9. Small fixes and firmware

**Board**
- [ ] Crystal capacitors C1/C2 = 1 pF look too small for the ECS-120-8-36 (CL 8 pF) — compute from
      the crystal datasheet (typically ~8–10 pF each) [e].
- [ ] RP2350 core regulator inductor (L3) as in the RP2350 hardware design guide.
- [ ] Test points: SWD, UART, supply rails.
- [ ] Optional pads for a battery-backed RTC (e.g. PCF8563) on 2 GPIOs — the RP2350 has no
      battery-backed RTC.

**yasos — already done in the working tree (uncommitted, partly in the `hal` submodule)**
- [x] `FIRMWARE_IN_FLASH` for mspc_v2 (Kconfig was ignoring the flash values from the defconfig).
- [x] SDIO selectable on MSPC (SPI by default); SDIO pins taken at run time from `MmcConfig.pins`.
- [x] Build fix: `-MD` + depfile in `addBoardHeaders` (stale C header translations).
- [x] MSPC defconfig back at 150 MHz.

**yasos — to do**
- [ ] Flash sampling (rxdelay): replace `(8·MHz+499)/500` with its `clkdiv-1` clamp by a target of
      ~4.75 ns after the edge (1 @150, 2 @200, 5 @532 MHz), and run
      `overclock_calibrate_flash_rxdelay()` inside `overclock_apply()` (from RAM). Run the 200 MHz
      test on the new board before any overclocking.
- [ ] Read the flash JEDEC ID (`0x9F`, capacity byte 0x17 = 8 MB, 0x18 = 16 MB) instead of relying
      on a constant from the configuration.
- [ ] A/B rootfs OTA: real `erase`/`write` in `flash.zig` (from RAM, leaving continuous-read mode,
      restoring the QMI configuration), a slot header, the romfs offset chosen at run time.
- [ ] Power budget: read CC (ADC), the TPS2116 status, drive the SY6280 EN lines.
- [ ] PCM5100A I2S driver (test on the Pico Plus 2 + VGA Demo Base).
- [ ] Card link protocol (PIO + DMA), enumeration by ID, INT/DET handling.

---

## 10. GPIO budget (RP2350B, 48 pins) — to verify in the pin editor

| function | GPIO |
|---|---|
| VGA link (CLK + D0–3 + INT) | 6 |
| FPGA link (CLK + D0–7 + SEL + INT), D0–7 on GPIO12–19 | 11 |
| network link (CLK + D0–3 + INT) | 6 |
| slot RESET (shared) | 1 |
| PSRAM CS1 (GPIO0 or 47; not 19 — that is an HSTX pin) | 1 |
| SD card (CLK, CMD, D0–3) | 6 |
| console UART (to the probe) | 2 |
| I2S audio (BCK, LRCK, DIN) + XSMT | 4 |
| CC of both USB-C connectors (ADC, GPIO40–47) | 4 |
| TPS2116 status | 1 |
| SY6280 EN (cards shared + USB) | 2 |
| power button + LED data | 2 |
| **total** | **46** |
| spare | 2 (e.g. REQ/GNT for the PSRAM experiment, SD card detect) |

SWD, RUN, QSPI and USB have dedicated pins. If pins run short: XSMT on an RC (−1); the shared card
EN is already counted; a separate EN per slot costs +2.

---

## 11. Rejected options (so we don't revisit them)

- **Moving to an STM32** — not now. Future candidate: **STM32H7S3** (LQFP-176; xSPI1 x16 →
  APS256XXN, xSPI2 → MX25UW25645G ~$1.96 or W25Q128JV, FMC → the bus), test board NUCLEO-H7S3L8
  ~$52. i.MX RT1062 — out of stock; STM32F7 — a single QSPI; H745/H747 — dual-core AMP without
  octal; STM32N6 — lots of RAM, but many supply rails and an irreversible 1.8 V fuse.
- **RISC-V** — no MCU with an FMC-class bus; tinycc has no RV32 backend, porting costs far more than
  an STM32.
- **SMP on the RP2350** — gains nothing anyway (parallel smoke 0.61×, limited by XIP).
- **1.8 V / DDR for speed** — QMI DTR halves the clock (limit 4 bits/cycle); W25Q64JV DTR only
  66 MHz; all GPIOs share one IOVDD; RP2350 pads are not faster at 1.8 V (VOH 1.24 V); HSTX already
  does 300 Mbit/s/pin at 3.3 V.
- **UHS (1.8 V) for the SD card** — the bus is not the bottleneck (writes ~4.4 MB/s limited by the
  card and the software, reads ~13 MB/s ≈ half the bandwidth at 48 MHz).
- **PD trigger (CH224A, 9 V)** — a PC without PD gives only 5 V, so one cable to the probe would no
  longer be enough; the CH224 passes 5 V before the voltage is negotiated.
- **Speaker on the board** — no; line output only.

---

## 12. Sources

- RP2350 datasheet — https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf
- ROHM BD9D321EFJ — https://fscdn.rohm.com/en/products/databook/datasheet/ic/power/switching_regulator/bd9d321efj-e.pdf
- USB Type-C Spec R2.0 — https://www.usb.org/sites/default/files/USB%20Type-C%20Spec%20R2.0%20-%20August%202019.pdf
- TI TPS25947 — https://www.ti.com/lit/ds/symlink/tps25947.pdf
- TI TVS0500 — https://www.ti.com/lit/ds/symlink/tvs0500.pdf
- TI LM66200 — https://www.ti.com/lit/ds/symlink/lm66200.pdf
- TPS2121 (LCSC) — https://www.lcsc.com/product-detail/C485916.html
- SY6280 — https://datasheet.lcsc.com/lcsc/1810121532_Silergy-Corp-SY6280AAC_C55136.pdf
- WCH CH334/335 datasheet — https://cdn-learn.adafruit.com/assets/assets/000/131/435/original/CH334DS1.PDF
- TI PCM510xA datasheet — https://cdn.sparkfun.com/assets/d/4/f/0/4/DS-16325-PCM5100A.pdf
- PCM5100APWR (LCSC) — https://www.lcsc.com/product-detail/C131154.html
- Pimoroni Pico VGA Demo Base — https://shop.pimoroni.com/en-us/products/pimoroni-pico-vga-demo-base
- W25Q128JV datasheet — https://www.winbond.com/resource-files/w25q128jv%20revf%2003272018%20plus.pdf
- W25Q64JV-DTR datasheet — https://www.pjrc.com/teensy/winbond_w25q64jvxgim.pdf
- APS6404L datasheet — https://raw.githubusercontent.com/Edragon/Datasheet/master/APM/APS6404L-3SQR-SN-2.pdf
- APS256XXN datasheet — https://www.apmemory.com/en/downloadFiles/0324112221tz581562
- CYW43439 datasheet — https://www.mouser.com/datasheet/2/196/Infineon_CYW43439_DataSheet_v03_00_EN-3074791.pdf
- RM2 datasheet — https://datasheets.raspberrypi.com/rm2/rm2-datasheet.pdf
- Raspberry Pi Radio Module 2 — https://www.raspberrypi.com/news/raspberry-pi-radio-module-2-available-now-at-4/
- ECP5 datasheet — https://0x04.net/~mwk/doc/lattice/ecp5/FPGA-DS-02012-3-3-ECP5-ECP5G-Family-Data-Sheet.pdf
- ECP5 measurements (OrangeCrab/ULX3S) — https://github.com/targeted/fpga-reg-power
- SD Simplified Spec v6.00 — https://academy.cba.mit.edu/classes/networking_communications/SD/SD.pdf
- microSD measurements — https://goughlui.com/2021/02/27/experiment-microsd-card-power-consumption-spi-performance/
- HDMI/DVI +5 V (EDN) — https://edn.com/Home/PrintView?contentItemId=4013470
