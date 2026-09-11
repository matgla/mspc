# MSPC v2.5 — mainboard revision plan

Summary of the decisions made on 2026-09-11, fact-checked against datasheets the same day. The
board stays on the **RP2350B**. Goals of the revision: power from USB-C instead of 12 V, fewer ICs,
a simpler and faster link to the cards, and an on-board debug probe.

Confidence marks: **[d]** = from a datasheet or specification, **[m]** = from a published
measurement, **[e]** = estimate. Check everything marked [e] before ordering the PCB.

---

## 1. Summary of changes

| area | v2 | v2.5 |
|---|---|---|
| power input | 12 V jack, 2× BD9D321EFJ, AP2210 (MCU 3.3 V), LD1117S50 (USB 5 V) | **2× USB-C at 5 V** (data + power, power only), two TPS25947 eFuses as a priority power mux (overvoltage clamp, current limit, soft start), 3.3 V buck for a 5 V input, SY6280 load switches |
| programming / console | external debug probe | **on-board RP2040 running debugprobe** (SWD + UART over one cable) |
| USB hub | HS8836A (SOP-16) | **CH334R** (QSOP-16) or **CH334F** (QFN-24), powered from 3.3 V |
| audio | TLV320DAC3203 + I2C | **PCM5100A** (I2S, no I2C, line-level output) |
| I2C / J7 | I2C connector | **removed** (optional pads for an RTC) |
| flash | 8 MB W25Q64JVSIM fitted; the schematic specifies W25Q128JVS | **W25Q128JVSIQ (16 MB)** in the BOM, room for A/B rootfs slots for OTA |
| PSRAM | APS6404L fitted, but the schematic shows a second W25Q128JVS in its place | APS6404L, with its own symbol in the schematic |
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
J_DBG (USB-C, data) ─ TVS ─ TPS25947 #1 ─┐
  └ D+/D- → RP2040 probe                 ├─ VSYS (5 V, protected) ─┬─► 3.3 V buck ─► MCU, probe, hub,
J_PWR (USB-C, power) ─ TVS ─ TPS25947 #2 ─┘  (J_PWR has priority)  │                memories, SD, audio
                                                                  ├─► SY6280 (EN) ─► 5 V to the card slots
                                                                  └─► SY6280 (EN) ─► USB ports (CH334 hub)
```

- **Overvoltage protection must sit in front of everything else.** The TPS2116 power mux, the
  TLV62569 buck and the SY6280 switches are all 6 V absolute-maximum parts [d] (the SY6280
  datasheet warns that overshoot above 5.5 V can cause EOS failure). A TVS does not protect them:
  it only starts conducting at 6.4–7.5 V and clamps at 9–10 V [d]. Hence one **TPS25947** per input
  (28 V absolute maximum, overvoltage clamp selectable at 5.7 V, reverse-current blocking, adjustable
  current limit, soft start) [d], and the two together form the priority mux: the datasheet lists a
  priority power mux built with the AUXOFF pin as an application [d]. This replaces both the TPS2116
  and a separate eFuse. J_PWR has priority, so a charger carries the whole load whenever it is
  connected and the PC port is not loaded; a plain diode-OR would split the current unpredictably.
- **2× USB-C, each with two separate 5.1 kΩ resistors** (CC1 → GND and CC2 → GND) [d]. Use 1 %
  parts: only a ±10 % Rd can detect the advertised current, and the ADC divider on CC sits in
  parallel (keep it ≥ 100 kΩ). Never one shared resistor — that was the Raspberry Pi 4 rev 1.1 bug
  (fixed in rev 1.2): with an e-marked cable the shared resistor in parallel with the cable's Ra
  looked like Ra on both CC pins, so the charger took the Pi for an audio accessory and never turned
  on VBUS [m].
- **A TVS on each input, for ESD and surges:** prefer the **TVS0500** (VBR 7.5–8.4 V, leakage
  ≤ 5.5 nA at 5 V, clamps at 9.2 V / 43 A) [d]; the SMAJ6.0A also works (VBR ≥ 6.67 V, but up to
  800 µA leakage at 6 V, clamps at 10.3 V) [d]. Not the SMAJ5.0A: its VBR is only 6.40 V min and
  it may already leak 800 µA at 5.0 V, so its leakage at the 5.5 V a charger may deliver is
  unspecified [d].
- **Bulk capacitance at attach:** a sink may present at most 10 µF ∥ 44 Ω when plugged in (USB 2.0
  §7.2.4.1); USB PD calls this cSnkBulk = 1–10 µF, and only a PD sink with a contract may go up to
  100 µF [d]. The board's larger capacitance sits behind the TPS25947 soft start (e.g. 100 µF with a
  1 V/ms ramp → 100 mA inrush [e]).
- **3.3 V buck for a 5 V input**, 2 A class, e.g. TLV62569 (used on the ULX3S and OrangeCrab; 6 V
  absolute maximum [d], hence the protection above). The 3.3 V load is a few hundred mA [e].
- **5 V for the cards and the USB ports comes from VSYS** (the protected, soft-started but
  unregulated input) **through SY6280 switches** (2.4–5.5 V, 2 A, reverse blocking) [d].
  - EN is active high on both variants (SY6280AAC and SY6280A/AAAC) [d]; they differ only in output
    discharge. Prefer the **SY6280**: it discharges its output when disabled.
  - EN gets a 100 kΩ pull-down, so everything is off during reset (RP2350 pads come out of reset with
    a pull-down and isolation [d]).
  - Current limit I_lim = 6800 / R_SET, but only ±25 % (0.75–1.25 A at 6.8 kΩ); minimum 0.4 A, above
    2 A not recommended [d]. Size R_SET so the *minimum* limit covers the load, e.g. ≈ 4.5 kΩ →
    1.13–1.9 A for the USB ports. The TPS25947 current limit must sit above the sum of the maximum
    limits of both switches.
- **Capacitance on the USB ports' VBUS: ≥ 120 µF of low-ESR capacitance** (USB 2.0 §7.2.4.1; e.g.
  150 µF polymer), behind the SY6280 and close to the connectors; it keeps the droop within 330 mV
  when a device is plugged in [d].

### 2.3 Power budget in firmware
- The ADC reads CC1 and CC2 of both connectors (through a ≥ 100 kΩ divider); use the higher of the
  two (the unused pin reads ~0 V). Sink thresholds (Type-C Table 4-36) [d]:
  - < 0.2 V → no source attached,
  - 0.2–0.66 V → "Default" source (USB 2.0: 500 mA, USB 3.x: 900 mA — after enumeration, see §2.4),
  - 0.66–1.23 V → 1.5 A,
  - ≥ 1.23 V → 3.0 A.
- The budget follows the active input (read which TPS25947 is conducting, e.g. from its
  power-good/fault output).
- With a weak source yasos does not enable the card or USB-port power (SY6280 EN).

### 2.4 Budget (current drawn from 5 V)

| load | typical | peak | confidence |
|---|---|---|---|
| RP2350 @150 MHz + PSRAM + flash | 20–40 mA at 3.3 V | | [d] RP2350 table 1446, APS6404L, W25Q128JV |
| SD card | 20–80 mA at 3.3 V | 150–300 mA at 3.3 V | [d]+[m] SD spec, Gough Lui |
| RP2040 probe | ~20–30 mA at 3.3 V | | [e] |
| CH334 hub (full-speed upstream) | ~20 mA | | [d] |
| PCM5100A | 18–30 mA at 3.3 V | ~45 mA | [d] |
| **mainboard total** | **~120 mA** | **~300 mA** | [e] |
| keyboard + mouse + flash drive (FS) | 95–220 mA | RGB keyboard up to 500 mA | [m] |
| FPGA card (ECP5-45F, soft CPU, APS256, RP2040) | ~100–170 mA | ~250 mA | [d]+[e] |
| VGA/DVI card (RP2350 overclocked to 252–300 MHz) | ~125–140 mA | | [e], incl. 55 mA on the DVI/HDMI +5 V pin [d] |
| WiFi card (RM2 + RP2040) | ~80–120 mA (transmitting) | ~270 mA; bursts up to ~0.8 A at 3.3 V | [d] CYW43439 |
| **total** | **~0.55–0.8 A (3–4 W)** | **~1.5 A (7.3 W)**; ~1.9 A if the WiFi bursts coincide | [e] |

The audio output is line level (no speaker).

| source | limit | enough for |
|---|---|---|
| USB-A 2.0 on a PC | 100 mA until configured, 500 mA after | mainboard + probe; keyboard and mouse only if the probe asks for 500 mA (see §3) |
| USB-A 3.x | 150 mA until configured, 900 mA after | + one card |
| USB-C 1.5 A | 7.5 W | the whole set typically; the peaks are marginal |
| **USB-C 3 A** | 15 W | **the whole set, including the peaks** |

**Voltage at the USB ports:** a charger may legally drop below 4.75 V by 0.25 V per ampere
(4.0–5.5 V at 3 A, measured at its own plug) [d], so a better cable does not help. At the board's
own ~1–1.5 A the input may be 4.4–4.5 V, and the USB-A ports, fed unregulated through the eFuse and
the SY6280, can fall below the 4.75 V a high-power port must provide (USB 2.0 §7.2.2) [d]. Either
accept that (most devices work down to about 4.4 V) or add a buck-boost for the port rail.

---

## 3. On-board RP2040 debug probe
- **debugprobe** firmware (CMSIS-DAP + CDC UART) — the same flow as today (OpenOCD, `flash.sh`).
- Connections to the RP2350: **SWCLK, SWDIO, RUN** and the console **UART** (TX/RX).
- **Power the probe from the board's 3.3 V rail**, not from J_DBG's VBUS — otherwise, with only
  the charger connected, the RP2350 would back-power the unpowered probe through SWD/UART.
- **VBUS detection on J_DBG** (divider → RP2040 GPIO, VBUS detect function): a self-powered device
  must not assert the D+ pull-up while the PC is disconnected. Stock debugprobe ignores VBUS:
  TinyUSB's RP2040 device driver forces VBUS detect on (`VBUS_DETECT | VBUS_DETECT_OVERRIDE_EN`)
  [d]. Honouring the divider needs a modified build that routes a VBUS-detect-capable GPIO, or that
  polls the GPIO and calls `tud_disconnect()` / `tud_connect()`.
- **Current it asks the PC for:** stock debugprobe declares bMaxPower = 100 mA
  (`TUD_CONFIG_DESCRIPTOR(..., 0, 100)`) [d]. A USB 2.0 device may draw only 100 mA until configured
  and then only what it declared [d]. Patch the probe firmware to request 500 mA, and keep the board
  below 100 mA until the probe is configured (cards and USB ports off).
- The RP2040 has no internal flash: the probe needs its own QSPI flash (e.g. W25Q16), a 12 MHz
  crystal with load capacitors (USB needs the crystal) and ESD protection on D+/D- (e.g. USBLC6-2)
  [d]. Total cost ~$2–3 [e]; count the board area too.
- Known limitation: the debugprobe on the current rig silently drops host → target bytes under
  sustained bulk traffic. It was measured at 921600 baud and still happens at 3 Mbaud, where the rig
  console runs with echo-verify and resend [m]. 115200 (the MSPC default) has not been measured, so
  "safe" is only an estimate [e]. Test a zmodem push through the on-board probe before relying on it.

---

## 4. USB hub
- **WCH CH334R** (QSOP-16, 0.635 mm pitch — hand-solderable) or **CH334F** (QFN-24, 4×4 mm) [d].
  - The CH334F has ganged PWREN/OVCUR. The **CH334R has no PWREN#/OVCUR# pins at all** (no power
    control, no overcurrent detection, no LEDs, no EEPROM) [d]: port power is then switched only by
    the MCU through the SY6280 EN, and the hub never reports a port overcurrent to the host. It
    reports itself as self-powered from its default configuration [d].
  - Avoid the CH334Q: it has no V5 pin (no internal LDO) [d].
- **Power the hub from the 3.3 V rail:** tie V5 and VDD33 together to 3.3 V (allowed 3.2–3.4 V,
  "no internal LDO" mode) [d]. The internal LDO needs V5 = 4.5–5.25 V [d], which a sagging USB-C
  input does not guarantee (§2.4).
- **Crystal:** crystal-free operation is optional for every model but ordering-dependent, selected
  by XI = GND and XO open, and WCH notes some parameters may then fall outside the USB spec [d].
  Fit the 12 MHz crystal footprint, and route it so XI can be strapped to GND through a 0 Ω when no
  crystal is fitted.
- The RP2350 is a full-speed host, so the hub runs in FS mode: ~20 mA [d]. Ports are powered from
  the board's own supply through an SY6280, so the 100 mA-per-port limit of a bus-powered hub does
  not apply.

---

## 5. Audio: PCM5100A
- **TI PCM5100A** — the same part as on the Pimoroni Pico VGA Demo Base [d].
- I2S on 3 pins (BCK, LRCK, DIN), using the internal PLL instead of a system clock [d]:
  - **tie SCK to GND** — the PLL starts only after SCK stays low for 16 LRCK periods, and switches
    off as soon as a clock appears on SCK;
  - BCK must be 32×fS or 64×fS; 8 kHz is not supported in PLL mode and 16 kHz only at 64×fS;
    32–384 kHz work [d].
- **No I2C:** FMT (low = I2S), XSMT (mute), FLT, DEMP are set by pins [d].
- **2.1 V RMS ground-centred** output (DirectPath charge pump): no DC-blocking capacitors and no
  op-amp, but **fit TI's RC output filter: 470 Ω in series + 2.2 nF to AGND on each channel**
  (datasheet Figure 33; the performance figures are measured with it into 10 kΩ) [d]. Load ≥ 1 kΩ,
  i.e. a line output [d].
- SNR 100 dB (PCM5101A 106 dB, PCM5102A 112 dB) [d].
- 3.3 V supply (AVDD, CPVDD, DVDD). At 48 kHz: DVDD 7–8 mA typ (13 mA max), AVDD + CPVDD 11 mA typ
  with zero data and 22 mA with a −1 dBFS sine (32 mA max) — about 18–30 mA typ, up to ~45 mA [d].
- **XSMT to a GPIO** — no pop at start-up (or a simple RC if pins are short).
- Price ~$1.63 each (LCSC, TSSOP-20) [d].
- The I2S driver (PIO, like `audio_i2s` from pico-extras) can be written and tested **now** on the
  Pico Plus 2 + VGA Demo Base (`configs/pimoroni_pico_plus2_and_vga_defconfig` in yasos).

---

## 6. Memories
- **W25Q128JV-IQ flash (16 MB, 133 MHz SDR).** Kernel ~0.3 MB + rootfs 2 × 3.4 MB (A/B) = ~7.1 MB:
  that fits in 8 MB only just, so 16 MB gives the rootfs room to grow (up to ~7.8 MB per slot).
  - The DTR variant (-IM) is not needed: DTR on the W25Q64JV/W25Q128JV runs only up to 66 MHz [d],
    and the RP2350 QMI halves the clock in DTR mode anyway — no gain.
  - The v2 schematic already specifies a W25Q128JVS for the MCU flash; the board in use has an 8 MB
    W25Q64JV fitted. Both come in the 208-mil SOIC-8 the footprint uses, so they are interchangeable
    on the board, but the firmware must detect the real part (§9).
- **APS6404L PSRAM** unchanged (the RP2350 QMI supports only 4 lines; the x8/x16 1.8 V APS256XXN does
  not fit — it goes to the FPGA card).

---

## 7. Link to the cards: point-to-point

### 7.1 Principle
Instead of a shared bus — **a separate link to each slot**, each with its own PIO state machine and
DMA channel (the RP2350 has 12 state machines):
- no stubs → easier to run a high clock,
- transfers to different cards in parallel,
- a hung card blocks only itself,
- no chip selects and no glue logic.

**PIO pin windows:** each PIO block reaches only 32 consecutive GPIOs, and its GPIOBASE can be
only 0 (GPIO0–31) or 16 (GPIO16–47) [d]. Every pin a state machine uses (out, in, side-set, jmp)
must lie inside its block's window, and each block has only 32 instruction slots. So the FPGA link
on GPIO12–19 (+ CLK on GPIO20) needs a block with GPIOBASE = 0, while the PIO SDIO on GPIO32–37 and
anything else above 31 needs a block with GPIOBASE = 16. Assign pins per PIO block in the pin
editor.

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
| 150 MHz (today, rated) | 75 MB/s | 150 MB/s |
| 200 MHz (overclock) [e] | 100 MB/s | 200 MB/s |
| 300 MHz (overclock) [e] | ~150 MB/s | ~300 MB/s |

The RP2350 is rated for a 150 MHz system clock [d]; the 200/300 MHz rows are overclocks. The only
published per-pin rate is HSTX's maximum, 300 Mbit/s per pin (150 MHz × DDR) [d]; no toggle rate is
published for PIO-driven pads.

These are host-side figures. **The card bridge must keep up too:** a PIO receiver that follows an
external link clock needs at least a `wait` and an `in` per edge, so an RP2040 bridge (rated
133 MHz, 200 MHz since SDK 2.1.1) caps a 4-bit DDR link at roughly 33–50 MB/s [e], unless the card
runs its system clock from the link clock or oversamples. Size the network and VGA links against
the bridge, not the host.

Needs: 640×480×8 bpp @60 Hz graphics = 18.4 MB/s (full frames), 16 bpp = 36.9 MB/s; 100 Mbit
Ethernet = 12.5 MB/s. **Buffers for the links must be in SRAM:** at today's 150 MHz the PSRAM runs
at SCK 75 MHz (clkdiv 2), i.e. at most 37.5 MB/s raw before command, address and wait overhead, so
even one 4-bit link outruns it. ~66 MB/s needs a 133 MHz SCK, i.e. a system clock of ≥ 266 MHz.

- **Writes** are easy (the host drives clock and data).
- **Reads**: PIO samples in whole-cycle steps (6.7 ns at 150 MHz) — **the card calibrates the phase**
  (PLL / pin delays on the FPGA, a training pattern at start-up). On the host side, bypass the input
  synchroniser.

### 7.4 Experiment: HSTX writes + PIO reads on the same pins (FPGA)
- A GPIO input is "always connected" to PIO regardless of FUNCSEL [d] → PIO can read HSTX pins.
  "Always connected" refers to the function mux only: the pad input must also be enabled —
  PADS_BANK0 GPIOx.IE resets to 0 and ISO to 1 — so set IE = 1 and clear ISO on GPIO12–19 [d].
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

**FPGA — ECP5 LFE5U-45F + APS256XXN + QSPI flash + RP2040** (the future card; the current v2 card
uses an LFE5UM-85F with W958D8NBYA HyperRAM)
- Rails: **1.1 V** (VCC), **1.8 V** (APS256XXN and the FPGA bank that talks to it), **2.5 V**
  (VCCAUX), **3.3 V** (VCCIO) [d].
- Currents: static (typical, TJ = 85 °C) ICC 116 mA, ICCAUX 17 mA, ICCIO 0.5 mA per bank [d]. The
  peak start-up current is not in the datasheet — Lattice points to the Power Calculator in Diamond
  [d] — so size the 1.1 V buck for ≥ 1 A as the open ECP5 boards do [e].
- Sequencing for Master SPI configuration: VCCIO8 must be above the SPI flash's VIH (≈ 2.3 V for a
  3.3 V flash) before the later of VCC/VCCAUX reaches its POR trip point (VCC 0.90–1.00 V, VCCAUX
  2.00–2.20 V) [d]. Rising together is not enough — a 1.1 V rail passes 0.9 V long before a 3.3 V
  rail reaches 2.3 V. If VCCIO8 cannot come up first, hold PROGRAMN or INITN low until it has. All
  rails must rise monotonically at 0.01–10 V/ms, VCCAUX at no more than 30 mV/µs [d].
- Draw: idle ≈ 0.17 W (116 mA × 1.1 V + 17 mA × 2.5 V) [d]; soft CPU @50–100 MHz ~0.3–0.6 W [e].
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
  performance from 3.2 V [d] → **feed the RM2 from 3.6–4.2 V** with its own regulator on the card.
  Transmit current 271 mA at MCS7/16 dBm (RM2 datasheet); the CYW43439's internal PA regulator is
  rated 800 mA peak [d] — size the regulator and local capacitance for ~0.8 A peaks, ~1 A with
  margin [e]. cyw43 driver + lwIP on the RP2040.

---

## 9. Small fixes and firmware

**Board**
- [ ] **Crystal load capacitors are too large.** C1/C2 on the v2 mainboard are 14 pF, and the same
      pair sits at every ECS-120-8-36 in the repo (GPU cards, FPGA card programmer). The crystal
      (ECX-2236 series, CL = 8 pF) then sees 7 pF + 3–5 pF stray ≈ 10–12 pF and runs slow [d].
      Use C1 = C2 = 2 × (CL − Cstray) ≈ 6–10 pF (e.g. 10 pF with ~3 pF stray), or keep ~14–15 pF and
      order the 10 pF load option of the same series (check the 12 MHz part is stocked).
- [ ] Put the real APS6404L symbol in the schematic (v2 shows a second W25Q128JVS on CS1) and give
      the flash as W25Q128JVSIQ in the BOM.
- [ ] RP2350 core regulator inductor (L3) as in the RP2350 hardware design guide.
- [ ] Test points: SWD, UART, supply rails.
- [ ] Optional battery-backed RTC (e.g. PCF8563) — the RP2350 has no battery-backed RTC. It needs 2
      GPIOs the budget does not have (§10); only if pins are freed.

**yasos — already done (committed in yasos.zig `5fb7a7b`, 2026-09-11)**
- [x] `FIRMWARE_IN_FLASH` for mspc_v2 (Kconfig was ignoring the flash values from the defconfig).
- [x] SDIO selectable on MSPC (SPI by default); SDIO pins taken at run time from `MmcConfig.pins`.
- [x] Build fix: `-MD` + depfile in `addBoardHeaders` (stale C header translations).
- [x] MSPC defconfig back at 150 MHz.

**yasos — to do**
- [ ] Flash sampling (rxdelay): replace `(8·MHz+499)/500` with its `clkdiv-1` clamp by a target of
      ~4.75 ns after the edge (1 @150, 2 @200, 5 @532 MHz), and run
      `overclock_calibrate_flash_rxdelay()` inside `overclock_apply()` (from RAM). Run the 200 MHz
      test on the new board before any overclocking.
- [ ] Detect the flash from its JEDEC ID (`0x9F`) instead of relying on a constant from the
      configuration: the capacity byte is 0x17 for 8 MB and 0x18 for 16 MB; the device ID also
      tells the -IQ/JQ (0x4018) from the -IM/JM DTR parts (0x7018), which ship with QE = 0, so quad
      mode must be enabled by software [d]. Today `configs/mspc_defconfig` and
      `hal/boards/mspc_v2/KConfig` both say `FLASH_MEMORY_SIZE="16MB"` while the board carries 8 MB,
      and `get_number_of_blocks()` in `flash.zig` returns a hard-coded 1024 — fix both with the probe.
- [ ] A/B rootfs OTA: real `erase`/`write` in `flash.zig` (from RAM, leaving continuous-read mode,
      restoring the QMI configuration), a slot header, the romfs offset chosen at run time.
- [ ] Power budget: read CC (ADC), the active input, drive the SY6280 EN lines.
- [ ] debugprobe: request 500 mA in the configuration descriptor (§3).
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
| active-input status (TPS25947 power-good/fault) | 1 |
| SY6280 EN (cards shared + USB) | 2 |
| power button + LED data | 2 |
| **total** | **46** |
| spare | 2 |

SWD, RUN, QSPI and USB have dedicated pins. **The spares are over-subscribed:** REQ/GNT for the PSRAM
experiment (2), SD card detect (1) and an RTC on I2C (2) would need 5. Decide which get pins; to free
more, put XSMT on an RC (−1). The shared card EN is already counted; a separate EN per slot costs +2.

---

## 11. Rejected options (so we don't revisit them)

- **Moving to an STM32** — not now. Future candidate: **STM32H7S3** (LQFP-176; xSPI1 x16 →
  APS256XXN, xSPI2 → MX25UW25645G or W25Q128JV, FMC → the bus), test board NUCLEO-H7S3L8 (DigiKey
  $52.52). Before relying on x16 in LQFP-176, check that package brings out all of XSPIM port 1
  (IO8–IO15) — ST's datasheet was not checked. Prices and stock as of 2026-09-11: MX25UW25645G ~$1.96
  at LCSC; i.MX RT1062 out of stock at LCSC and DigiKey. STM32F7 — a single QSPI; H745/H747 —
  dual-core AMP without octal; STM32N6 — lots of RAM, but many supply rails and an irreversible 1.8 V
  fuse.
- **RISC-V** — RV32 MCUs with a parallel memory bus do exist (WCH CH32V307 with FSMC, GigaDevice
  GD32VF103 with EXMC) [d], but they are slower (144/108 MHz) with less RAM than the STM32
  candidates, and tinycc has no RV32 backend (upstream has only riscv64): porting costs far more than
  an STM32.
- **SMP on the RP2350** — does not help the workloads that matter: two tcc compiles in parallel run
  at 0.61× because both cores thrash the shared 16 KiB XIP cache. RAM-resident work does scale
  (1.64× measured) [m].
- **1.8 V / DDR for speed** — QMI DTR halves the clock (limit 4 bits/cycle); W25Q64JV DTR only
  66 MHz; all GPIOs share one IOVDD; the RP2350 datasheet publishes no speed advantage for 1.8 V
  IOVDD (drive is weaker: VOH min 1.24 V), so 1.8 V buys nothing documented; HSTX already does
  300 Mbit/s/pin at 3.3 V.
- **UHS (1.8 V) for the SD card** — measured on the Pico Plus 2 rig in 4-bit SDIO at 48 MHz (bus
  maximum 24 MB/s): writes ~4.4 MB/s are limited by the card and the software, reads ~13 MB/s use
  about half the bus [m]. MSPC defaults to 1-bit SPI, where the bus *is* the limit (at most ~6 MB/s
  at 48 MHz), so the conclusion holds once MSPC runs 4-bit SDIO.
- **USB PD trigger for 9 V (e.g. CH224A)** — every USB-C source starts at 5 V and moves to 9 V only
  after a PD contract, and a PC or non-PD port never does. Every rail would have to run from both
  5 V and 9 V, so it adds a wide-input converter and a 5 V regulator for the USB and card rails and
  removes nothing. (The CH224A is not in the power path; it only reports the contract on its PG pin.)
- **Speaker on the board** — no; line output only.

---

## 12. Sources

- RP2350 datasheet — https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf
- ROHM BD9D321EFJ — https://fscdn.rohm.com/en/products/databook/datasheet/ic/power/switching_regulator/bd9d321efj-e.pdf
- USB Type-C Spec R2.0 — https://www.usb.org/sites/default/files/USB%20Type-C%20Spec%20R2.0%20-%20August%202019.pdf
- USB 2.0 specification (§7.2.2, §7.2.4.1, §9.2.5.1) — https://www.usb.org/document-library/usb-20-specification
- USB Power Delivery (cSnkBulk) — https://www.usb.org/document-library/usb-power-delivery
- TI TPS25947 — https://www.ti.com/lit/ds/symlink/tps25947.pdf
- TI TPS2116 — https://www.ti.com/lit/ds/symlink/tps2116.pdf
- TI TLV62569 — https://www.ti.com/lit/ds/symlink/tlv62569.pdf
- TI TVS0500 — https://www.ti.com/lit/ds/symlink/tvs0500.pdf
- Vishay SMAJ5.0A/6.0A — https://www.vishay.com/docs/88390/smaj50a.pdf
- SY6280 — https://datasheet.lcsc.com/lcsc/1810121532_Silergy-Corp-SY6280AAC_C55136.pdf
- Raspberry Pi revision codes — https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/revision-codes.adoc
- Pi 4 and CC resistors — https://www.scorpia.co.uk/2019/06/28/pi4-not-working-with-some-chargers-or-why-you-need-two-cc-resistors/
- debugprobe USB descriptors — https://github.com/raspberrypi/debugprobe/blob/master/src/usb_descriptors.c
- WCH CH334/335 datasheet — https://cdn-learn.adafruit.com/assets/assets/000/131/435/original/CH334DS1.PDF
- TI PCM510xA datasheet — https://www.ti.com/lit/ds/symlink/pcm5100a.pdf
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
- ECS ECX-2236 crystal — https://ecsxtal.com/store/pdf/ECX-2236.pdf
- SD Simplified Spec v6.00 — https://academy.cba.mit.edu/classes/networking_communications/SD/SD.pdf
- microSD measurements — https://goughlui.com/2021/02/27/experiment-microsd-card-power-consumption-spi-performance/
- HDMI/DVI +5 V (EDN) — https://edn.com/Home/PrintView?contentItemId=4013470
