# MSPC v3 — mainboard changes

Decided 2026-09-11. The board stays on the **RP2350B**. Values marked *(verify)* are estimates to
check before ordering the PCB.

## 1. Summary

| area | v2 | v3 |
|---|---|---|
| power input | 12 V jack | **2× USB-C at 5 V**: J_DBG (data + power), J_PWR (power only, has priority) |
| input protection | — | **TPS25200 eFuse per input** (20 V tolerant, 5.4 V output clamp), the two forming a priority mux |
| 3.3 V | BD9D321EFJ + AP2210 | **TLV62569DRL** buck |
| 5 V | BD9D321EFJ + LD1117S50 | no regulator; the USB input through **SY6280AAC** switches |
| programming / console | external debug probe | **on-board RP2040 running debugprobe** (SWD + UART on J_DBG), on its own LDO; it can reset the RP2350B and power-cycle the board (§3.1) |
| USB hub | HS8836A + 10× LESD5D5.0CT1G | **CH334R** on `+5V_VBUS` through its internal LDO; **2× USBLC6-4SC6** ESD arrays on the USB-A ports |
| audio | TLV320DAC3203 + I2C | **PCM5101A** (I2S only, line out) on its own **TLV70033** LDO |
| I2C / J7 | I2C connector | removed |
| flash | W25Q64JVSIM fitted (8 MB) | **W25Q128JVSIQ** (16 MB, room for A/B rootfs) |
| PSRAM | APS6404L (drawn as a W25Q128JVS) | **APS1604M-3SQR-SN** (16 Mbit / 2 MB, SOP-8 150 mil) with its own symbol |
| card bus | shared 16-bit, 6× CS, 74HCS151 + 74LVC139 | **point-to-point link per slot**, no glue logic; v2 cards are not compatible |
| slots | J13 16-bit, J14 8-bit, J15 8-bit | **3 slots**: FPGA (8-bit), network (1-bit SPI), VGA/DVI (4-bit) (§7) |
| slot 3.3 V | `+3V3_BUS` on every slot | **dropped** — each card makes its own 3.3 V from its switched 5 V (§8.1) |
| front panel | — | **3× JST SH 1.00 mm**: panel LEDs (3 pins), reset and power button (2 pins each) (§9) |

Removed: 12 V jack, BD9D321EFJ ×2, AP2210, LD1117S50, TLV320DAC3203, 74HCS151, 74LVC139, J7,
HS8836A (+ R19, R20, C41–C43), LESD5D5.0CT1G ×10.

Added: 2× USB-C, 2× TPS25200, 1 PMEG3020EP Schottky, TLV62569DRL + 2.2 µH, 4× SY6280AAC, RP2040 +
W25Q16JVSSIQ + 12 MHz crystal + USBLC6-4SC6 + TLV70033 (probe 3.3 V) + 2× 2N7002 (target power and RUN, §3.1), PCM5101A + TLV70033, CH334R (+ 12 MHz crystal),
2× USBLC6-4SC6 (USB-A) + 2× USBLC6-4SC6 (SD), 3× slot power LED + resistor, 3 front-panel JST SH headers (J50 reset, J51 power
button, J52 panel LEDs), I2C1 housekeeping (TCA6408A, INA226, INA3221, MCP7940N, §2.8).

---

## 2. Power

```
J_DBG (USB-C, data)  ─ TPS25200 U3 ─┬─►|─┐   D3: J_PWR wins
  └ D+/D- → RP2040 probe             │    ├──┬── VSYS (5 V, protected; +5V_RAW, then the
                                     │    │  │   R221 shunt → +5V)
       +3V3_PROBE: RP2040 probe ◄─ TLV70033  │   (probe only; up whenever J_DBG is plugged in)
J_PWR (USB-C, power) ─ TPS25200 U1 ───────┘  │
                                             ├─► TLV62569 ─► 3.3 V: MCU, memories, SD,
                                             │   EN ◄─ probe  DAC digital side, I2C devices
                                             │   (§3.1)
                                             ├─► TLV70033 ─► 3.3 V audio: DAC AVDD + CPVDD
                                             ├─► 3× SY6280, one per slot ─► 5 V to slots 0, 1, 2
                                             └─► SY6280 (EN) ─► +5V_VBUS: USB ports + CH334R
```

### 2.1 USB-C inputs
- J1 (J_PWR) and J2 (J_DBG) are HRO **TYPE-C-31-M-12** (LCSC C165948).
- CC1 and CC2 each with their own 5.1 kΩ 1 % to GND (never one shared resistor): R13/R15 on J1,
  R14/R16 on J2.
- C1/C4: 4.7 µF / 25 V X5R 0805 at each eFuse input (the TPS25200 input is rated to 20 V).
- **VBUS bleeders R17/R18**, 47 kΩ 1 % from `+5V_CONN_1` / `+5V_CONN_2` to GND: they hold an
  unplugged connector's VBUS below 0.8 V against the TPS25200's reverse leakage, so a charger that
  checks for a dead VBUS before sourcing will start. No fuse, no TVS (optional unpopulated TVS2200
  footprint). Never a 5–6 V TVS: it burns on a faulty charger.
- USB-C shells straight to GND with several vias at the connector; no chassis, so no RC network
  (USB-A: §4).

### 2.2 eFuses: TPS25200
One per input; each is that input's fuse: current limit, input overvoltage shutoff, output clamp,
reverse-current blocking (while disabled) and soft start. 2.5–6.5 V operating, **IN withstands
20 V**; WSON-6 2 × 2 mm with thermal pad, 60 mΩ, θJA 66.5 °C/W.

| setting | J_PWR (primary) | J_DBG (auxiliary) |
|---|---|---|
| R_ILIM (current limit), 1 % | R4 40.2 kΩ → 2.27–2.57 A (2.42 nom) | R10 110 kΩ → ≈ 0.88 A nom; hardware ceiling, firmware honours CC |
| EN | 300 kΩ pull-up to its own IN (TI's value; an internal zener clamps the pin) | same |
| FAULT | open-drain, 300 kΩ pull-up to 3.3 V, to the expander (§2.8) | same |
| C_IN | 0.1 µF at the pin, plus the 4.7 µF of §2.1 | same |
| C_OUT | 1 µF at OUT (OUT *is* VSYS here, so C8 is its bulk) | 1 µF at OUT (`+5V_DBG`), ahead of D3 — its own node |

- **Overvoltage: fixed thresholds, no dividers.** Above ~5.55 V in, the output is clamped to
  5.25–5.55 V (5.4 V typ) — under the 6 V limit of the TLV62569 and the SY6280s — and above
  **7.6 V** in, the switch disconnects (0.6 µs). Short-circuit response 3.5 µs, current limit
  ±6 %.
- **Priority mux: a Schottky in the J_DBG path, both eFuses always enabled.** With both inputs
  present, VSYS follows J_PWR at ~5 V and D3 is reverse-biased (a few µA), so the charger carries the
  load; when J_PWR drops, D3 conducts at once — no switchover gap. Nothing back-feeds the PC: D3
  blocks it, and an eFuse below its 2.35 V UVLO disables itself and blocks reverse current.
  D3 = **PMEG3020EP** (SOD-128, 30 V, 2 A): V_F 275 mV typ / 310 mV max at 1 A, ~250 mV at 0.5 A,
  I_R 130 µA at 5 V. The leakage only trickles into the disabled auxiliary eFuse's 480–625 Ω
  discharge path (< 1 mW), so the low-V_F part wins over the low-leakage PMEG3020BEP (+90 mV) and
  the 5 A PMEG3050EP (−35 mV for 330 µA). **Not the inventory BAT60J:** SOD-323 is thermally
  limited to 310 mW at 400 °C/W, so its 3 A rating is pulsed, and a 1.2 A fault would cook it.
  - **Not an EN-driven mux** (an N-FET pulling the auxiliary's EN low): the TPS25200 needs 5.1 ms
    typ, 7.3 ms max to turn on, so every source swap would brown the board out — no bulk capacitor
    bridges 5 ms. The TPS259470L's dedicated mux did it in ~90 µs.
  - Cost: ~0.25–0.3 V on the PC path only (VSYS ≈ 4.7 V when J_DBG feeds the board); the buck
    regulates from a 3.4 V input up.
  - C8 (≥ 100 µF) sits on VSYS itself (`+5V_RAW`), where D3's cathode meets U1's OUT, ahead of the
    shunt, the buck and the SY6280s — one shared bulk, not one per eFuse. It is **EEHZC1V151P**
    (Panasonic ZC hybrid polymer aluminium, 150 µF/35 V, ESR 27 mΩ, 1.6 A ripple at 100 kHz,
    Ø8 × 10.2 mm, 4000 h at 125 °C), with 10 µF + 0.1 µF ceramic beside it for the fast edges.
    It is **polarized** — `Device:C_Polarized` on `footprints:CP_Elec_8x10.5`, with the minus
    marking on the silkscreen; the same applies to the VBUS bulk (§2.4).
    No MnO2 tantalum (§2.7); a 100 µF MLCC would work electrically but loses much of its value to
    DC bias and is the part most likely to crack into a short. It also damps the OVP overshoot: the
    output follows the input until the eFuse reacts (0.6 µs), and more output capacitance lowers
    that peak (datasheet Figure 8-3). Each eFuse still gets a local 1 µF at its own OUT pin; on the
    auxiliary that is the only capacitance on its side of D3.
  - **Voltage rating on the output side: 16 V or more** (10 V absolute minimum; the ZC series
    starts at 25 V, so its 35 V rating is simply the family's low end). VSYS normally sits at 5 V
    and the clamp holds 5.4 V, but during that 0.6 µs the output follows the input upward, and
    MLCCs lose a large part of their value under DC bias. Ceramic or polymer only — no MnO2
    tantalum (§2.7).
- FAULT is pulled up to 3.3 V, which the buck makes from VSYS: it reads low until the rail is up,
  and a fault that collapses VSYS takes the flag with it.
- **Why not the TPS259470L** (the earlier choice): it needs ~10 external parts against ~4, and it
  is out of stock at TI; the TPS25200 is $0.44/1, $0.27/100 at LCSC. Given up: 20 V instead of
  28 V, no −15 V reverse polarity (IN-to-OUT −7 V), 2.6 A continuous instead of 5.5 A, a FAULT
  flag instead of an analog current monitor, and auto-retry instead of latch-off (§2.7, §12).

### 2.3 3.3 V: TLV62569DRL
- U23: R31 (VOUT–FB) 453 kΩ (E96; 450 kΩ is E192), R32 (FB–GND) 100 kΩ → 0.6 V × (1 + R31/R32)
  = 3.32 V. R32 must be ≤ 200 kΩ. **C33 6.8 pF C0G feedforward across R31** (TI's value for exactly
  100 kΩ at the bottom): the divider is high-impedance, so a few pF of stray at FB already forms a
  pole near the loop's crossover, and this adds the zero that offsets it — worth it on a rail that
  sees SD card and MCU load steps. The datasheet calls it optional, so it can be depopulated if the
  bench says otherwise.
- **Divider resistors 0.1 %.** The feedback reference is 0.588–0.612 V (±2 %), so with 1 %
  resistors the rail spans ~3.20–3.44 V; 0.1 % tightens it to ~3.23–3.38 V. The CH334R that once
  set this limit now runs from `+5V_VBUS` (§4), so it is margin rather than a requirement.
- C28 10 µF 1206 (CL31A106KAHNNNE) at the input, C31 10 µF at the output.
- **EN: R27 100 kΩ pull-up to VSYS** (net `POWER_OFF#`), pulled low by the probe's 2N7002 to
  power the board off (§3.1). EN is VIN-tolerant, so the pull-up can go to VSYS directly; with the
  probe absent or not driving, the gate pull-down keeps the FET off and the buck runs.
- L20 2.2 µH: **Murata DFE252012F-2R2M=P2** (LCSC C576403) — 2.5 × 2.0 × 1.2 mm, I_sat 3.3 A
  guaranteed / 3.6 A typ, I_rms 2.3 A, DCR 82 mΩ, $0.11–0.15. Its I_sat clears the converter's
  ≥ 2.5 A current limit, so
  it replaces the 4 × 4 mm XAL4020 outright and takes the regulator block from ~24 mm² to ~13 mm².
  Not the FPGA card's 0805 part, and not a low-I_sat 2520 like the SWPA252012S (1.25 A, 216 mΩ).
  Footprint `footprints:L_Murata_DFE252012F` — KiCad has no DFE252012F, and the stock 2520 patterns
  don't match: Murata wants 1.2 × 2.0 mm pads on a 2.8 mm span, while e.g. L_Changjiang_FNR252010S
  has 0.85 × 2.0 mm on 2.5 mm, ending flush with the body and leaving no outer fillet.
- Layout per datasheet Figure 21. Thermal: ~+8 °C at the 0.3 A peak, fine with a ground pour.

### 2.4 Switched 5 V: SY6280AAC
Each switch has its own EN, active high, driven by the TCA6408A U200 (§2.8): P2 `CARD_0_EN` (U20),
P3 `CARD_1_EN` (U22), P4 `CARD_2_EN` (U24), P5 `VBUS_EN` (U21). Each EN net has a **7.5 kΩ
pull-down** (R20, R21, R26, R33). The datasheet says EN must never float; it turns on above 2.4 V
(with VIN 4.2–5.5 V) and off below 0.8 V, and leaks ≤ 1 µA, so the expander drives it directly at
3.3 V and the pull-down holds it low while the expander is in reset or unpowered. No series
resistor. The SY6280AAC discharges its output through 120 Ω when disabled.
I_lim = 6800 / R_SET (±25 %).

| branch | R_SET | limit |
|---|---|---|
| card 0 FPGA (U20) | R22 9.1 kΩ | 0.75 A nom (0.56–0.93 A) — the only card that may reach ~0.4 A |
| card 1 network, ESP32-C3 (U22) | R28 13.7 kΩ | 0.5 A nom (0.37–0.62 A) — ~2.5× the 0.25 A transmit peak |
| card 2 VGA/DVI (U24) | R34 13.7 kΩ | 0.5 A nom (0.37–0.62 A) |
| USB ports (all four, one switch, U21) | R24 6.8 kΩ | 1.0 A nom (0.75–1.25 A) — one 500 mA device plus HID and a drive |

- Slot numbering follows the schematic: card 0, 1, 2.
- Decoupling: 1 µF ceramic at each switch's IN plus C38–C40 (10 µF, 1206) on `+5V` at the switch
  inputs, and 10 µF + 100 nF at each OUT; each slot also gets 10 µF at its connector. This follows
  the Silergy datasheet's application circuit (10 µF in / 10 µF out). All of it **16 V or more,
  X7R/X5R** like the rest of the 5 V side (§2.2). Standard 10 µF part: **CL31A106KAHNNNE** (1206,
  25 V, X5R, ±10 %) — the 1206 case and 25 V rating keep most of the capacitance at 5 V bias, where
  a 0603 6.3 V part would keep little. Use it for the other 10 µF spots too (buck C_IN/C_OUT). The
  USB ports' output bulk is the ≥ 120 µF below.
- ≥ 120 µF low-ESR on the USB ports' VBUS: **C23**, the same **EEHZC1V151P** as C8.
- Give the USB-port switch copper on IN/OUT (~+45 °C at 1.5 A).
- **Slot power LED**, one per slot on the switched output at the connector: **SZYY0402YG** (0402
  yellow-green, 571 nm) with **10 kΩ** to GND, ≈ 0.33 mA each and ~1 mA for all three. Brightness
  scales with current, so the 58 mcd at 20 mA becomes ~1 mcd — a soft glow that still says which
  slots yasos has enabled. Use 6.8 kΩ (0.5 mA) if it reads too dim in daylight, 22 kΩ (0.15 mA)
  for barely-there. Green means "powered"; keep the orange-red ZSR1-1105C-045-Z4 for faults — an
  LED and 1.5 kΩ from 3.3 V to an eFuse FAULT pin lights when that input trips and costs no GPIO
  (FAULT sinks up to 25 mA).

### 2.5 Audio 3.3 V: TLV70033
- `+3V3_AUDIO` from U25, a **TLV70033DDCR** (SOT-23-THIN, fixed 3.3 V, 200 mA) fed from **VSYS**, per the same
  RP2040 reference: a linear regulator ahead of the analogue supply keeps the TLV62569's switching
  ripple off it. Raspberry Pi report that buck's PFM ripple being visible on their VGA DAC output.
- 1 µF in and 1 µF out, EN (pin 3) tied directly to `+3V3` rather than to its own input — so the audio rail comes
  up only after the digital rail, which is free sequencing. **No `AUDIO_EN` GPIO** (decided
  2026-09-13): the PCM5101A sleeps on its own. When the I2S clocks stop it drops to standby, and with
  BCK and LRCK held low for > 1 s it enters power-down (charge pump and reference off) — 0.2 mA
  typ / 0.4 mA max on AVDD+CPVDD and 0.5 / 0.8 mA on DVDD (datasheet SLAS859C §8.5, §11.5.2).
  The LDO's own quiescent current is ~31 µA. Before dormant, firmware drives BCK and LRCK **low**
  (not floating) so the DAC reaches power-down; restarting the clocks wakes it. GPIO46 becomes
  I2C1 SDA (§2.8).
- **It must be fed from VSYS, not `+3V3`**: 3.3 V in for 3.3 V out leaves no headroom, the part
  never leaves dropout and its PSRR is zero, which defeats the whole point.
- Load is ~25 mA, so the 1.7 V drop burns ~45 mW. Operating input is 2–5.5 V with a 6 V abs max;
  VSYS reaches 5.55 V only inside the TPS25200's OV-clamp window, the same fault-window
  exceedance already accepted for the CH334R's V5 (§4).
- Place it beside the DAC, not beside the buck — a clean rail routed across the board past the
  switchers picks the noise back up.

### 2.6 Budget
- Mainboard: ~120 mA typ, ~300 mA peak at 3.3 V (~85 mA from 5 V through the buck). This figure
  still includes the hub; the CH334R now draws its share from `+5V_VBUS` through its internal LDO
  (§4), so the 3.3 V number is a little high — re-measure at bring-up.
  The DAC's analogue side (~25 mA) sits on VSYS through the TLV70033 (§2.5), not on the buck.
- Whole set with cards and USB devices: ~0.55–0.8 A typ, ~1.5–1.9 A peak → needs a USB-C 3 A source.
- A PC port (500 mA) runs the mainboard and probe; 100 mA until the probe enumerates. A USB 3.x
  port changes nothing: its 150 mA/900 mA unit loads apply to SuperSpeed operation, and the probe
  is a USB 2.0 full-speed device, so USB 2.0's 100 mA/500 mA rules hold. More than 500 mA on J_DBG
  only comes from a USB-C source advertising 1.5 A or 3 A on CC (§2.3).

### 2.7 Board rules (unattended CI node)
- Copper and connector pins sized for each limiter's maximum current; card 0 (0.93 A worst case,
  against ~1 A per 1.27 mm pin) gets ≥ 2 pins each for 5 V and GND.
- No tantalum capacitors; MLCCs away from board edges, connectors and mounting holes.
- Keep the VSYS node small. Test points on VSYS, 3.3 V and every switched rail.
- The TPS25200 auto-retries rather than latching off, so a persistent fault drops the probe off
  USB and brings it back in a loop; that pattern is the CI fault signal. Whether auto-retry is
  good enough here is open (§12).
- **Recovery is a probe power cycle (§3.1), not a J_PWR power cycle.** Unplugging or
  uhubctl-cycling J_PWR does not reset the board while J_DBG is connected: VSYS stays up through
  D3 from the PC port (up to the 0.88 A limit of the J_DBG eFuse). The probe cutting the buck EN
  works whichever input feeds the board. A relay on J_PWR is only needed if a card's 5 V side can
  latch up with the 3.3 V rail off, which the SY6280s already rule out (their ENs fall with it).

### 2.8 I2C1 housekeeping bus: power monitor, I/O expander, RTC
Decided 2026-09-13. **I2C1 on GPIO46 (SDA) / GPIO47 (SCL)** — already an I2C1 pin pair, so no
reordering — with 4.7 kΩ pull-ups to `+3V3`. Four devices, no address clash:

| addr | part | job |
|---|---|---|
| 0x20 | **TCA6408APWR** (TSSOP-16) | outputs CARD_0/1/2_EN, VBUS_EN; inputs eFuse FAULT# ×2, power-monitor alert, RTC MFP |
| 0x40 | **INA226AIDGSR** (VSSOP-10) | VSYS current, voltage and power |
| 0x41 | **INA3221AIRGVR** (VQFN-16, 4 × 4 mm) | card 0/1/2 5 V rail current and voltage |
| 0x6F | **MCP7940N-I/SN** (SOIC-8) + 32.768 kHz crystal | real-time clock with battery switchover |

**Interrupts are cascaded through the expander** (revised 2026-09-13, as wired). The open-drain
alert outputs land on TCA6408A inputs: INA226 ALERT, INA3221 CRITICAL# and WARNING# share
`VBUS_POWER_ALERT#` (10 kΩ pull-up) on P6, MCP7940N MFP (10 kΩ pull-up) is on P7, and the two
eFuse FAULT# lines are on P0/P1. Any input change asserts the expander's INT#, which goes to
**GPIO42 with a 10 kΩ pull-up** — the one interrupt pin on the MCU. Firmware reads the expander's
input port to see what fired, then the INA status registers if it was P6. GPIO42 is also the
dormant wake source for an RTC alarm.

**TCA6408A.** Every P-port is an input after power-on with ≤ 1 µA leakage and no internal pull-ups
(datasheet SCPS192), so the 7.5 kΩ pull-downs keep every slot unpowered until firmware configures
P2–P5 as outputs; P0, P1, P6, P7 stay inputs. VCCI = VCCP = `+3V3`, ADDR to GND (0x20).
**RESET# to `MCU_RESET#`**, so the reset header also drops the card rails; an SWD reset from the
probe does not touch RUN, so firmware re-initialises the expander at boot.
- **Card RST# stay on RP2350 GPIOs** (11, 21, 28), not on the expander: a card reset is then
  immediate, and it does not depend on the I2C bus being alive.
- The eFuse FAULT# lines moved to the expander, so a fault is seen only through I2C. Accepted: a
  fault that matters also shows as a VSYS drop on the INA226, and a hung bus is recovered with
  nine SCL pulses and the RP2350 watchdog.
- I2C is now the only path to slot and USB-port power. Defaults are safe (all off).

**INA226 (U220).** High-side 20 mΩ shunt R221 (Panasonic **ERJ-6CWDR020V**, 0805, 0.5 W, C2089540)
right after C8 and before every load (buck, the four SY6280s, the audio TLV70033) — VSYS splits
into a source side `+5V_RAW` (U1 OUT, D3 cathode, C3, C8, TP1) and a load side `+5V` (D1, TP26). IN+ on the source side, IN− and VBUS on the load side, A0 = A1 = GND
(0x40), 100 nF at VS.
- Shunt range ±81.8 mV → ±4.09 A; with a 0.1 mA current LSB it covers U1's 2.57 A worst case.
  Bus voltage LSB 1.25 mV, so a VSYS sag is visible — which a bare current amplifier would miss.
- ALERT is programmable (shunt over-voltage = over-current, bus under-voltage, power over-limit),
  latching or transparent, at conversion times down to 140 µs: a hardware threshold instead of
  polling.
- Shunt loss 50 mV / 125 mW at 2.5 A.

**INA3221 (U221) — per-slot current.** One 50 mΩ 1 % shunt per card rail (R224/R225/R227), between each SY6280 OUT and
its slot connector (ahead of the slot's 10 µF and power LED), so each channel sees one card only
and its bus voltage is what the card actually gets. IN+ on the switch side, IN− on the connector
side (the bus voltage is read at IN−). VS from `+3V3` with 100 nF, **A0 to VS through R226
(5.6 kΩ) → 0x41**.
- Shunt range ±163.8 mV, 40 µV LSB → 0.8 mA per count at 50 mΩ; card 0's 0.93 A worst case is
  46.5 mV. Bus voltage 0–26 V, 8 mV LSB. Loss: 43 mW and a 46 mV drop at 0.93 A, which the cards'
  own regulators absorb. Shunts: Yageo **PT0603FR-7W0R05L** (0603, 0.2 W, C784595).
- 13-bit, no power register: firmware multiplies bus voltage by current.
- CRITICAL# (every conversion, per-channel limit, optionally latched) and WARNING# (on the averaged
  value) go to `VBUS_POWER_ALERT#` on expander P6, together with the INA226 ALERT. PV (power valid)
  and TC (timing control) are open-drain, each pulled up with 10 kΩ to `+3V3` and read on GPIO30
  and GPIO31. VPU to `+3V3`.
- The INA226 still measures the total on VSYS, so the USB-A ports and the board itself are VSYS
  minus the three cards minus the buck input — close enough for load shedding. A fourth channel
  (PAC1954, or a second INA3221 at 0x42) is the upgrade if the USB ports need their own number.
- Not the INA4230: 4 channels, but DSBGA 0.4 mm pitch only, which needs vias in pads (`docs/pcb_rules_jlcpcb.md`).

**J_PWR and J_DBG CC → GPIO40 (ADC0) and GPIO41 (ADC1).** CC1/CC2 are USB-C's configuration-channel
pins. The board's 5.1 kΩ Rd (R13/R15 on J1, R14/R16 on J2) pulls each to GND; a source answers with a
pull-up on the one CC wire the plug orientation connects, and the voltage says what it offers:
0.25–0.61 V default USB (500/900 mA), 0.70–1.16 V 1.5 A, 1.31–2.04 V 3 A.
- Per connector, CC1 and CC2 through **100 kΩ each** (R2/R3 on J1, R7/R8 on J2) into one node,
  **10 nF** to GND (C9, C10). The idle CC
  sits at 0 V, so the node reads half: 0.13–0.31 V default, 0.35–0.58 V 1.5 A, 0.65–1.02 V 3 A.
  The 100 kΩ load shifts Rd by ~5 %, inside the bands.
- **0 V on J_PWR = charger gone**, the moment it goes — before VSYS moves, because D3 takes over.

**RTC: MCP7940N-I/SN** (Microchip, SOIC-8, LCSC **C51106**, ~$0.38, in stock). A plain RTC IC
with an external crystal — replaces the RV-3028-C7 (crystal-in-package, better at 45 nA and
±1 ppm, but out of stock at LCSC, C3019759). Datasheet DS20005010J:
- VCC 1.8–5.5 V from `+3V3`, 100 nF. **VBAT 1.3–5.5 V, automatic switchover**, backup current
  **925 nA typ / 1.2 µA max** at 3 V (vs 45 nA for the RV-3028).
- I2C address fixed at **0x6F**. MFP is open-drain (alarm, square wave or GPIO) → expander P7;
  it is driven from VCC only, so no alarm output while on battery.
- 64 bytes of battery-backed SRAM (lost with the battery — keep the board serial in RP2350 OTP,
  not here). Power-fail and power-restore timestamps, ±129 ppm digital trim in 1 ppm steps.
- **Crystal: CL 6–9 pF only** (12.5 pF parts are not recommended). Seiko **SC-32S 32.768 kHz
  20 ppm 7 pF**, 3.2 × 1.5 mm, LCSC **C97604**. Load capacitors from
  CL = C_X1·C_X2 / (C_X1 + C_X2) + C_stray with C_X = C + 3 pF (C_OSC): for CL 7 pF and ~2 pF
  stray, **6.8 pF C0G** on each pin *(verify on the bench, then trim)*. Crystal and caps on the same
  side as the IC, next to X1/X2, GND guard pour, no traces underneath on B.Cu.
- ±20 ppm is ~50 s/month before trimming; the network card's NTP corrects it anyway.
- Firmware: the oscillator does not start until **ST** is set, and battery switchover is off
  until **VBATEN** is set — both at first boot, or the backup does nothing.

**VBACKUP: CR1220 coin cell** (decided 2026-09-13). The cell goes straight to VBAT — no diode, no
resistor; the MCP7940N only switches over and never charges it, which is what a lithium primary
needs. It drains only while the board is unplugged: ~38 mAh at 0.925 µA typ / 1.2 µA max is
4.7 / 3.6 years unplugged the whole time, so on a normally powered board the cell's ~10-year
shelf life is the limit.
- **Holder: Q&J CR1220-2** (BS-12-B3AA004), SMD, LCSC **C70381** (~52 k in stock, $0.24),
  phosphor-bronze contacts, PPS housing, 4.0 mm high with the cell, −40…+80 °C.
- Library: symbol `mspc_connectors:BatteryHolder_QJ_CR1220-2` (pin 1 +, pin 2 −, LCSC/MPN filled
  in), footprint `footprints:BatteryHolder_QJ_CR1220-2_SMD` from the maker's drawing: + pad
  4.35 × 2.30 mm, − pad 4.04 × 2.30 mm, two Ø1.20 mm NPTH peg holes 7.50 mm apart, body Ø14.8 mm.
- Footprint area ~20.7 × 15.5 mm. Keep it reachable with the cards seated, and away from the
  buck and the USB-A ports. The silkscreen `+` marks the positive contact.
- Rejected: CR2032 (20 mm holder, not needed for the hold-up), a supercap (days of hold-up at µA
  drain, needs a diode and resistor or the BQ32000's charger).

The RP2350's own AON timer keeps time through dormant, but not through unplugging; the network
card's NTP corrects drift either way.

**What firmware does with it**
- *J_PWR lost* (CC → 0 V): VSYS is now on the PC port through D3 and U3's 0.88 A limit. If the
  INA226 reads above ~0.8 A, drop the slots and USB-A ports through the expander before U3 trips,
  instead of the auto-retry loop of §2.7. An INA226 over-current ALERT can trigger this without
  polling.
- *VSYS sag* (bus under-voltage ALERT): shed load in the same order.
- *Weak source* (CC reads default or 1.5 A): refuse the FPGA slot, cap the USB ports.
- *Card fault*: the INA3221 channel for that slot exceeds its budget (CRITICAL# fires, no
  polling) or never settles after EN — switch the slot off and park its link pins.
- Limits: firmware speed (ms) — it acts after the SY6280/TPS25200 hardware limits, not before.

---

## 3. On-board debug probe: RP2040
- RP2040 + W25Q16JVSSIQ (U121, 208-mil SOIC-8, C82317) + 12 MHz crystal, debugprobe firmware.
- ESD: U4 **USBLC6-4SC6** (C111212) right at J_DBG. The two D+ (A6/B6) and two D- (A7/B7) pins are
  joined on `PROBE_USB_D+` / `PROBE_USB_D-`, the same nets that run through the R128/R129 27 Ω
  series resistors to the RP2040. I/O1 (pin 1) on D-, I/O2 (pin 3) on D+, GND (2) by a short via.
  **VBUS (pin 5) to the probe's 3.3 V rail, with 100 nF (≥ 10 V) at the pin** —
  not to VSYS and never to the connector-side VBUS. That pin is the clamp reference, so the data
  lines can only rise to it plus a diode drop: ~4 V from 3.3 V, ~6 V from VSYS, and whatever a
  faulty charger delivers if it were tied to the connector. RP2040/RP2350 USB pins are **not 5 V
  tolerant** — they are USB IO fed from USB_OTP_VDD (3.135–3.63 V), so their limit is that supply
  + 0.5 V, about 3.8 V; the datasheet's 5.5 V applies only to the Digital IO (FT) GPIOs.
- To the RP2350: SWCLK, SWDIO, console UART, and RUN and the buck EN through FETs (§3.1). The
  UART is the bus `UART_DEBUG{MCU_TX MCU_RX}`: `MCU_TX` = RP2350 GPIO44 → R135 → U124 → probe
  GPIO5; `MCU_RX` = probe GPIO4 → R133 → RP2350 GPIO45.
- **Powered from its own TLV70033DDCR (U120) on the J_DBG eFuse output** (`+5V_DBG`, ahead of D3),
  not from the board's 3.3 V rail: the probe stays on the host while the rest of the board is off,
  and it is off whenever J_DBG is unplugged. The USBLC6-4SC6 clamp reference goes to this
  `+3V3_PROBE` rail.
- `+5V_DBG` bulk: **C11, 47 µF / 35 V hybrid polymer, Panasonic EEHZA1V470P** (C178639,
  6.3 × 5.8 mm). It limits the overvoltage spike from a faulty charger before the TPS25200's OVP
  clamps — the TLV70033's absolute maximum is 6 V.
- **R140, 2.2 kΩ from `+5V_DBG` to GND**: sinks D3's reverse leakage so the probe rail does not
  float up when only J_PWR is plugged in.
- The RP2040's GPIO26/ADC0 senses the board's `+3V3` through the R136/R137 100 k / 100 k divider
  (§3.1).
- **BOOTSEL button (SW_DBG_BOOT)**: XUNPU **TS-1088-AR02016**, LCSC **C720477**, a JLCPCB
  basic/preferred part (3.9 × 3.0 × 2.0 mm, 1.6 N, 100 k cycles) — footprint
  `Button_Switch_SMD:SW_SPST_TS-1088-xR020` ships with KiCad 10, no custom library entry.
  Wire it from the W25Q16's `QSPI_SS`/CS net to GND through a **1 kΩ series resistor**; that
  series R is what keeps normal XIP reads undisturbed, and the RP2040 holds CS high internally
  for the boot sample, so no pull-up is needed. This is the only way to reflash the probe
  without a second probe, so it must stay reachable with all cards seated.
- **Probe reset (SW_DBG_RUN)**: same part, straight from the RP2040's `RUN` to GND. Optional,
  but the pair next to each other is the usual reflash gesture (hold BOOT, tap RUN).

### 3.1 Target reset and power cycle from the probe
Decided 2026-09-13. The CI node needs a *real* reset: a hung RP2350B, a card stuck in a bad state
or a wedged I2C device must come back without anyone touching the board. SWD reset is not enough
(it does not touch RUN or the card rails), so the probe gets two outputs.

```
+5V (VSYS) ── 100k ──┬── U23 EN (TLV62569)          probe GPIO10  TARGET_PWR_OFF
                     │                              probe GPIO11  TARGET_RUN_ASSERT
                     D                              probe GPIO26  +3V3 sense (ADC0)
probe GPIO10 ── G  2N7002   G: 100k to GND
                     S ── GND

MCU_RESET# (RP2350B RUN) ── D  2N7002   G ── probe GPIO11, 100k to GND
                             S ── GND

+3V3 ── 100k ──┬── 100k ── GND
               └── probe GPIO26 (ADC0)
```

**Power off.** GPIO10 high turns the FET on and pulls the buck EN low; `+3V3` collapses and takes
with it the RP2350B, flash, PSRAM, SD card, the TCA6408A, INA226, INA3221 (the MCP7940N drops to its
battery), the CH334R
and the audio LDO (its EN hangs off `+3V3`). With the expander unpowered, its 7.5 kΩ pull-downs drop
all three card rails and hold the cards in reset; `VBUS_EN` falls too, so the USB-A ports go off.
VSYS itself stays up, so the probe keeps running and nothing re-enumerates on the host. Power
back on is a cold boot of the whole machine.

**Reset.** GPIO11 high pulls RUN low — the same net as the front-panel reset header J_RST, so the
two never fight (both only pull down).

**Why FETs and not direct GPIOs.** With J_DBG unplugged the probe is unpowered, and an unpowered
RP2040 pin clamps to its 0 V rail through its ESD diode. Wired straight to RUN it would hold the
RP2350B in reset; the EN node is at 5 V, which no RP2040 pin may see. The 2N7002 gates are pulled
down, so an absent, unpowered or booting probe leaves the board running (RP2040 GPIOs also reset
to inputs with pull-downs).

**Firmware sequence (debugprobe fork).**
1. Stop SWD and switch SWCLK (GPIO12), SWDIO (GPIO14) and UART TX (GPIO4) to inputs without pulls.
   Otherwise they feed the unpowered RP2350B through the 100 Ω resistors and its ESD diodes, keep
   `+3V3` partly up and spoil the cold boot. The receive buffers (74AUP1T17, U123/U124) have I_off
   and need nothing.
2. GPIO10 high.
3. Wait until GPIO26 reads `+3V3` < 0.3 V, at least 200 ms, at most 2 s (then report a failure).
   The TLV62569 has no output discharge, so the tail is set by what is left on the rail.
4. GPIO10 low; wait for `+3V3` > 3.0 V plus ~10 ms, then restore the SWD and UART pins.

Host side: CMSIS-DAP vendor commands, so CI scripts need nothing but OpenOCD
(`cmsis-dap cmd 0x80 <op>`) or pyOCD — `0x80` target power (0 off, 1 on, 2 cycle), `0x81` RUN
(0 release, 1 assert, 2 pulse). Also map GPIO11 as the CMSIS-DAP nRESET output, inverted because of
the FET, so `reset_config srst_only` works in OpenOCD. Stock debugprobe has neither, so this is a
fork either way.

**Parts:** 2× 2N7002 (SOT-23), 5× 100 kΩ (EN pull-up, 2 gate pull-downs, `+3V3` divider). The
divider costs ~16 µA from `+3V3`; with the probe unpowered it injects < 30 µA into its ADC pin,
which is harmless.

## 4. USB hub: HS8836A → CH334R
- **CH334R** (QSOP-16, 0.635 mm pitch), bus-powered through its internal LDO: **V5 on
  `+5V_VBUS`** with 1 µF + 100 nF, **VDD33 left as the LDO output** with 0.1 µF + 10 µF and nothing
  else on it. ~20 mA (the datasheet's full-speed-upstream, full-speed-downstream row — the 85 mA
  figure is the all-high-speed case the RP2350's full-speed-only PHY can never reach), so the LDO
  drops 1.7 V at 20 mA, 34 mW. Never tie VDD33 to an external 3.3 V rail while V5 is at 5 V: that
  parallels the internal LDO against the buck. V5's abs max is 5.5 V and its LDO-enabled range
  4.5–5.25 V; `+5V_VBUS` only reaches 5.55 V while the TPS25200 clamps an input overvoltage, i.e.
  inside a fault window — accepted.
- The chip now shares the rail its own ports load, so a downstream hot-plug dip can trip the LVR
  and drop the whole hub (datasheet §6.1). Both fixes it names are already in: C23 = 150 µF
  EEHZC1V151P (C128516) on `+5V_VBUS`, 10 µF on the LDO output.
- **ABM8-272-T3** 12 MHz across XI (16) and XO (15), **no load capacitors** — the CH334R's are
  built in. Crystal-free (XI to GND, XO open) only if the ordered part has it enabled, and the
  datasheet warns those parts "may deviate from the USB specification": fit the crystal.
- RESET# open (internal pull-up). Pull-up and pull-downs are built in: no 1.5 kΩ pull-up on the
  RP2350's D+.
- Upstream: RP2350 USB_DP → DPU (pin 11), USB_DM → DMU (pin 10), through the 27 Ω series resistors.
- Ports powered only through the SY6280 (the CH334R has no PWREN/OVCUR pins).
- USB-A: J90/J91 are Jing Extension **907-111A1022D10200** dual stacked (C12049).
- **ESD: one USBLC6-4SC6 (U90, U91; C111212) per stacked USB-A connector** on its 4 data lines, next to the
  connector, GND on a short via. Pin 5 (VBUS) goes to **`+5V_VBUS`**, the switched port rail, with
  100 nF to GND right at the chip — that ceramic is the return path for the steering-diode current,
  which the 150 µF electrolytic is far too inductive to carry. Referencing the array to the
  connector's own rail keeps a VBUS-to-D− short (adjacent pins on USB-A) from forward-biasing a
  steering diode into the 3.3 V rail, and puts the port's VBUS pin under the internal 6 V Zener.
  Worst case on that rail is 5.55 V (the TPS25200's OV clamp), not 5.9 V, and above V_RM = 5.25 V
  the only consequence is leakage rising from 1 µA — V_BR(min) is 6.0 V.
- USB-A connector shells straight to GND with several vias (bare board, no chassis).

## 5. Audio: TLV320DAC3203 → PCM5101A
Built to the audio section of Raspberry Pi's *Hardware design with RP2040* (the VGA/SD/audio demo
board, §3.1.2 and Figure 30), which uses the same PCM5101A. Checked pin by pin against it.

- I2S from RP2350 GPIO13 = DIN, GPIO14 = BCK, GPIO15 = LRCK (pico-extras:
  `PICO_AUDIO_I2S_DATA_PIN=13`, `PICO_AUDIO_I2S_CLOCK_PIN_BASE=14`). BCK = 32× or 64× fS.
- Hardware configuration, no I2C: **SCK (12), DEMP (10), FLT (11) and FMT (16) all to GND** — I2S
  format, de-emphasis off, normal-latency filter, and a grounded SCK selects the internal PLL.
- **XSMT (17) strapped to `+3V3`, not to a GPIO.** The datasheet bounds it at
  −0.3 V to DVDD + 0.3 V, so it has to sit on **the same rail as DVDD** or it breaches that limit
  while the other rail ramps. Soft mute is given up deliberately; the reference does the same.
  `AUDIO.RESET#` on U9's 2Y3 output is therefore spare.
- **Two rails, and the split is not arbitrary** (§2.5): AVDD (8) and CPVDD (1) on the clean
  `+3V3_AUDIO`, DVDD (20) on the main `+3V3` beside XSMT. The analogue output stage and the charge
  pump that generates its negative rail are what the LDO is there for; the digital core is not.
  Do not "tidy" all three onto one rail — it either re-exposes the outputs to buck ripple or
  breaks the XSMT limit above. 10 µF + 100 nF at each supply pin.
- Charge pump: 2.2 µF flying capacitor across CAPP (2) / CAPM (4), **2.2 µF on VNEG (5)**,
  **100 nF on LDOO (18)**. VNEG is the reservoir for the −3.3 V rail the output stage runs from;
  LDOO only decouples the internal logic rail. Getting these two the wrong way round puts
  charge-pump ripple on the outputs.
- Outputs are ground-centred thanks to that charge pump, so there are **no DC blocking capacitors**.
- Output filter per channel: 470 Ω series + 2.2 nF to AGND (154 kHz corner). Line out
  (load ≥ 1 kΩ) — into headphones it is series-resistance limited and very quiet.
- Jack: tip = left, ring 1 = right, ring 2 and sleeve to GND.

## 6. Memories
- Flash: U51 W25Q128JVSIQ (16 MB, C97521); same 208-mil SOIC-8 footprint.
- PSRAM: U52 AP Memory **APS1604M-3SQR-SN** (16 Mbit / 2 MB, SOP-8 150 mil, C18214056) on CS1
  with its own symbol, chosen instead of the APS6404L (v2 showed a W25Q128JVS there). C80 1 µF at
  its VDD, as AP Memory recommends.

## 7. Card link: shared bus → point-to-point

| slot | card | link (RP2350B GPIO) | extra lines | bridge |
|---|---|---|---|---|
| **card 0** (J150, 2×10) | FPGA (ECP5-45F) | 8-bit: D0–D7 on GPIO1–8, CLK GPIO9 | INT# GPIO10, RST# GPIO11, CS# GPIO12 | RP2040 |
| **card 1** (J151, 2×08) | network (ESP32-C3) | **1-bit SPI** (SPI0): MISO GPIO16, CS# GPIO17, SCK GPIO18, MOSI GPIO19 | INT# GPIO20, RST# GPIO21 | none — the card *is* the bridge |
| **card 2** (J152, 2×08) | VGA / DVI | 4-bit: D0–D3 on GPIO22–25, CLK GPIO26 | INT# GPIO27, RST# GPIO28, CS# GPIO29 | RP2350B |

**Slot 2 is the edge slot and the VGA/DVI card has to live there** — its display connector must
reach the chassis edge, which is what fixes the assignment. The network card takes slot 1; the two
data lines on that connector are MOSI and MISO, which is why the slot reads "2-bit" in the
schematic. **RST# is per slot**, not the shared RESET this section used to specify: one card can be
restarted without disturbing the others, which a shared line cannot do.

**Cards from v2 are not compatible** — the slot pinout changed, so the board becomes v3.
The 1.27 mm sockets have no key: mark pin 1 on the board silkscreen and on every card.

- One PIO state machine + DMA channel per link, DDR, host-clocked; command protocol
  `[CMD][ADDR][LEN][DATA…]`; the card answers an ID command.
- **The link is SPI-shaped**: host-clocked, command-framed, half-duplex with the data lines turning
  around, so the 4-bit form is QSPI and a slot can drop to plain 1-bit SPI. Slot 1 does exactly
  that, because the ESP32-C3's half-duplex slave has **no quad mode** — single line only, specified
  to 60 MHz and realistically 20–40 MHz across a connector, i.e. 2.5–5 MB/s. That is ample for
  WiFi. An SPI slave needs its CS# to frame transactions; the other slots carry one as well.
- INT doubles as DET: 7.5 kΩ pull-down on the host, driven high by a running card, low = interrupt.
- PIO windows: GPIO0–31 (base 0) or GPIO16–47 (base 16) — the FPGA link needs base 0, the PIO
  SDIO on GPIO32–37 base 16.
- 33 Ω series resistor on every line at the host; 7.5 kΩ pull-downs on CS#, RST# and INT#; a
  ground every 2–4 signals in the connectors.
- Link buffers in SRAM (the PSRAM is slower than one link).

## 8. Card power requirements
- **FPGA** (ECP5 LFE5U-45F + APS256XXN + RP2040): 1.1 V (buck ≥ 1 A), 1.8 V, 2.5 V, 3.3 V.
  VCCIO8 above ~2.3 V before VCC/VCCAUX reach POR, or hold PROGRAMN/INITN low.
- **VGA/DVI** (RP2350B + PSRAM + flash): DVI/HDMI +5 V pin ≥ 55 mA.
- **Network** (ESP32-C3-WROOM-02, no bridge MCU): 345 mA peak at 3.3 V transmitting (802.11b at
  20.5 dBm; 280–285 mA for g/n), 82–84 mA receiving, 16–28 mA in modem-sleep — so ~250 mA from 5 V
  at the peak through the card's own buck. That is less than half the RM2's 800 mA burst budget,
  which is what allowed the network slot's limit to come down.
  - Wired alternative on the same SPI slot, also with no bridge: a **W5500** (~132 mA, hardware
    TCP/IP, its interrupt output lands on the slot's INT line). More dependable for an unattended
    CI node than WiFi, and it needs no TCP stack in yasos.
- Each card regulates its own rails from its switched 5 V. A slot carries 5 V and GND only; there
  is no 3.3 V on the connector (§12 decided).
- Every card filters the slot 5 V at the connector before its own regulators: bulk 47 µF plus a
  ferrite/22 µF pi section. Without it each card's switching transients return through the slot
  pins' inductance, which is what made the shared rails noisy on v2.

### 8.1 FPGA card refactor
The VGA/DVI card already regulates locally (`U3` AP2112K-3.3 on its `bus_connector` sheet). The
FPGA card is the only one still taking 3.3 V from the bus, and it needs four changes.

**3.3 V regulator.** Today `+3V3` has exactly one source: `FB1` from `+3V3_BUS` (J4 pin 28), and
there is no 3.3 V regulator on the card at all. It feeds six ECP5 I/O banks (VCCIO0, 1, 2, 3, 6
and 8 — VCCIO7 is the 1.8 V HyperRAM bank), the RP2040 programmer, both W25Q128 flashes, the
SG-8002CE oscillator and the microSD socket: roughly 150–250 mA typical, ~500 mA peak.

Add `U10`, a second **TLV62569DRL** from the local 5 V — the same part, SOT-563 footprint and
2.2 µH inductor as `U6`, so no new line items. Divider `0.6 × (1 + R_top/R_bot) = 3.3` → 475 k /
105 k, with a feedforward cap across the top leg as `C16` does for `U6`. An LDO was considered and
rejected: 1.7 V × 250 mA ≈ 0.43 W is too much for SOT-23-5 at the peak load.

`FB1` stays, repurposed. The buck output feeds the 3.3 V VCCIO banks directly — never put a bead
in series with a switching-I/O rail — and `FB1` becomes a quiet branch for the oscillator and the
programmer's IOVDD.

**Power sequencing becomes explicit.** The VCCIO8-before-VCC/VCCAUX ordering above is currently
satisfied by accident: bus 3.3 V is always present, and the local rails only start when the slot's
5 V switches on. Once 3.3 V is local off that same 5 V, all four rails race. The EN pins of `U4`
(2.5 V), `U5` (1.8 V) and `U6` (1.1 V) are tied straight to `+5V` today — move them to a common EN
net divided down from `+3V3` (≈100 k / 60 k, so EN crosses its ~0.9 V threshold when 3.3 V reaches
~2.4 V). Two resistors, and the ECP5 requirement is met by design rather than by side effect.

**Card detect moves.** `R28` (`<10k`) pulls `BUS.DET` up from `+3V3_BUS` — the card announces
itself on bus power, before it has any of its own. With no bus 3.3 V that cannot work, and pulling
from the card's local 3.3 V is circular if the host uses DET to decide whether to switch the slot
on. Use §7's scheme instead: host-side 7.5 kΩ pull-down on INT/DET, driven high by a running card.
The card currently has INT and DET on separate pins (J4 23 and 24) and does not match §7 yet.

**5 V input path.** `+5V_BUS` and `+5V_USB` have no capacitors on them at all, and the only bulk on
the 5 V node is 3 × 4.7 µF *after* the `Q1`/`D1` OR — thin for a 1 A buck plus two LDOs, and about
to carry the 3.3 V load as well. Add the §8 pi filter at J4, and replace `D1`: it is a SOD-323
part carrying the whole card when running from USB.

## 9. Front panel

Three JST **SH 1.00 mm** headers at the board edge — the family the Pico uses for its debug port.
Keyed, so a panel cable cannot go on backwards, and 1 A / 50 V is ample for two buttons and two
LEDs. Top entry (`BM..B-SRSS-TB`), as placed on the schematic.

| ref | function | part | LCSC | symbol / footprint (`..._Vertical`) |
|---|---|---|---|---|
| J52 (J_LED) | panel LEDs, 3 pins | BM03B-SRSS-TB(LF)(SN) | **C160389** | `mspc_connectors:JST_SH_BM03B-SRSS-TB` / `footprints:JST_SH_BM03B-SRSS-TB_1x03-1MP_P1.00mm_Vertical` |
| J50 (J_RST) | reset button, 2 pins | BM02B-SRSS-TB(LF)(SN) | **C160388** | `mspc_connectors:JST_SH_BM02B-SRSS-TB` / `footprints:JST_SH_BM02B-SRSS-TB_1x02-1MP_P1.00mm_Vertical` |
| J51 (J_PWR_BTN) | power button, 2 pins | BM02B-SRSS-TB(LF)(SN) | **C160388** | as J50 |

The side-entry `SM02B`/`SM03B-SRSS-TB` (C160402/C160403) are also in `mspc_connectors` with the
footprint, datasheet and LCSC code filled in (§13).

### 9.1 Pinout and wiring
- **J_LED** (J52): 1 = `POWER_LED` anode (GPIO38, U50.47, via R66 220 Ω), 2 = `STATUS_LED` anode
  (GPIO39, U50.48, via R67 220 Ω), 3 = GND (common cathode).
- **J_RST** (J50): 1 = RP2350B `RUN` (`MCU_RESET#`, via R50 100 Ω, C53), 2 = GND.
- **J_PWR_BTN** (J51): 1 = `POWER_BUTTON` (GPIO43, U50.54, via R51 100 Ω, C54 100 nF), 2 = GND.
  R69 10 kΩ pulls it up to `+3V3`, because GPIO43 resets with a pull-down.
- **100 Ω in series and 100 nF to GND** on `RUN` and `POWER_BUTTON` at their connectors. These are
  the only nets that leave the board on unshielded wire, so they are the ESD and noise path into
  the MCU. The probe pulls the same `RUN` net low through its 2N7002 (§3.1), so button and probe
  cannot fight.
- Solder both mounting tabs and tie them to GND — that is what holds an SH header down when the
  cable is pulled.
- Cable side: housings `SHR-02V-S-B` / `SHR-03V-S-B` with `SSH-003T-P0.2` contacts (LCSC
  **C160231**). Crimping SH by hand is awkward: buy ready-made pigtails, or run the panel loom in
  **JST PH 2.0 mm** (B2B-PH-K-S **C131337**, B3B-PH-K-S **C131339**) and keep SH for the board.

**Panel LED: plain LEDs driven from GPIOs** (decided 2026-09-13, replaces the WS2812B with its
SN74AHCT1G125 level shifter and switched 5 V supply).

```
GPIO38 POWER_LED  ── 220 Ω ── J52.1 ──►|── ┐   power LED (e.g. yellow-green, 570 nm)
GPIO39 STATUS_LED ── 220 Ω ── J52.2 ──►|── ┤   second colour / status (e.g. amber), optional
                              J52.3 ───────┘   GND, common cathode
```

- **Why not an addressable LED.** WS2812-type parts need ≥ 3.5–3.7 V supply (the blue/green dies
  drop ~3.0–3.2 V plus driver headroom), so on the 3.3 V rail they are out of spec, and on 5 V
  they need a level shifter for data plus a switched supply, because they hold their last colour
  while powered and `+5V` stays up when the probe cuts the buck. A plain LED needs none of that.
- **Off means off.** When `+3V3` goes down, the GPIOs lose power and the LEDs go dark on their own.
- **Firmware:** on, blink, or PWM (e.g. breathing in soft-off standby, §9.2). A bi-colour
  common-cathode LED gives two states on the existing 3-pin header, e.g. yellow-green = running, amber =
  standby/fault. Pin 2 can stay unconnected for a single LED.
- **Resistor 220 Ω, 0402, on the mainboard next to J52.** A red, yellow or yellow-green LED
  (V_F ≈ 2.0 V) draws ≈ 6 mA, inside the RP2350's 12 mA drive setting. Blue and white
  (V_F ≈ 2.8–3.1 V) only get 1–2 mA and look dim, so the panel should use a low-V_F colour or a
  high-efficiency part. The resistor is also the series protection for the panel wire, like the
  100 Ω on `RUN` and `POWER_BUTTON`.

### 9.2 Power button
The board has no battery and USB-C is the only source, so "off" always means *plugged in, rails
down*. Two depths, and the baseline needs no new parts:

- **Soft off (baseline).** The four SY6280 ENs are already MCU-driven through the expander (§2.4, §2.8): a press asks yasos to
  park the cards, then drops the three slot rails and the USB-A ports and puts the RP2350B in
  dormant; a press wakes it. Hold ~4 s to cut the rails without waiting for software. Standby is
  the buck plus a dormant MCU plus the hub — tens of mA, which a wall-powered machine can carry.
  The debug probe stays enumerated on the host, which is what the CI node needs.
- **Hard off, remote (from the probe).** The probe now has its own LDO and owns the buck EN (§3.1),
  so the host can switch the whole board off and on; the probe stays enumerated. This covers the
  CI node without a latch.
- **Hard off from the button (still open, §12).** A local hard off needs the button to reach the
  buck EN while the MCU is down. It can go through the probe instead of a latch: a probe GPIO
  reading `POWER_BUTTON` in parallel (long press → power off, press while off → power on). That only
  works while J_DBG is plugged in; without it, the board stays on soft off.

`POWER_BUTTON` (GPIO43) and the panel LEDs (GPIO38, GPIO39) are the GPIOs reserved in §11; the
probe-side power control uses probe GPIOs only (§3.1) and costs the RP2350B nothing.

## 10. To do

**Board**

*Checked against the schematic netlist on 2026-09-13 (ERC clean).*

- [x] Swap the remaining `ECS-120-8-36-CGN-TR` to **ABM8-272-T3** (`mspc_xtal`) on `rp2350_core`,
      `rp2040_core` and `south_bridge_osc`: the ECS part's 150 Ω max ESR is 3× the 50 Ω the
      RP2350's 1 kΩ XOUT damping resistor is tuned for, and Raspberry Pi tells you to test over
      temperature for any deviation. Keep the 14 pF load capacitors — 7 pF + 3 pF parasitic is the
      ABM8's 10 pF CL exactly, so this closes the old "14 pF → 10 pF" item rather than doing it.
      *(Y50, Y120 and Y90 are ABM8-272-T3; 14 pF kept on Y50 and Y120.)*
- [x] Fix the hub's upstream pair: `USB_D+` sits on RP2350 pin 66 (USB_DM) and `USB_D-` on pin 67
      (USB_DP), so DM reaches DPU+ through R16. The pair is genuinely crossed, not just mislabelled
      — move the labels onto the other pins. *(Pin 66 → R55 → DMU, pin 67 → R54 → DPU.)*
- [x] `usb_hub.kicad_sch` is instantiated twice, in `mainboard.kicad_sch` and in `rp2350_core`.
      The netlist carries two of every part and the top-level copy's upstream dangles. Delete it.
      *(One instance left, on the top level, with its upstream connected.)*
- [x] `mspc_bus.kicad_sch`: J11 is missing `BUS_CARD1.RE#` and carries `INT#` on pins 9 *and* 12;
      J7 is missing `BUS_CARD2.D3` and carries `D2` on pins 5 *and* 6. Both connectors also cluster
      their only two grounds at the far end — tie the spare pins (J19 3/4/15/19/20, J11 3/4/5/6/11/
      15/16, J7 3/4/11/15/16) to GND so the rule above is met, and put one next to CLK.
      *(Now J150/J151/J152: every line is present once, GND pairs at 3/4, 9/10 and next to CLK.
      J151 pins 5 and 6 stay free for the network card's handshake line, §12.)*
- [x] PSRAM with its own symbol; W25Q128JVSIQ in the BOM.
      *(U52 APS1604M-3SQR-SN, C18214056 — chosen instead of the APS6404L; U51 W25Q128JVSIQ,
      C97521.)*
- [x] RP2350 core regulator inductor per the RP2350 hardware design guide. *(L50, see below.)*
- [x] Test points on the supply rails (none on SWD or UART, decided 2026-09-13).
      *(TP1 `+5V_RAW`; TP20–TP28 on the power supply sheet, TP28 on `POWER_OFF#`; TP50
      `+1V1_MCU`; TP120 `+1V1_PROBE`, TP121 `BOOTSEL`, TP122 `+3V3_PROBE`, TP124 `+5V_DBG`. The
      duplicate `+1V1_PROBE` test point is gone.)*
- [x] Remove `+3V3_BUS` from the slot pinout; J4 pin 28 becomes a spare connector pin.
- [x] Front panel (§9): place J_LED, J_RST and J_PWR_BTN (`mspc_connectors`) at the board edge
      with their 100 Ω + 100 nF, mounting tabs on GND; the reset header parallels the RP2350B
      `RUN` line the probe already drives. *(J50 reset: R50 100 Ω + C53 on `MCU_RESET#`; J51
      power: R51 100 Ω + C54 100 nF on `POWER_BUTTON`, GPIO43; J52 LEDs. Top-entry `BM0xB` parts.)*

**Board — from the 2026-09-13 schematic review**
- [x] SD card: CMD pull-up (R60, 10 kΩ); D0–D3 on R58, R61, R57, R59. *(J53 Hirose
      DM3AT-SF-PEJM5 push-push; ESD U53 on D0–D3, U54 on CMD/CLK; C58 10 µF 1206 + C77 100 nF at
      VDD; R68 33 Ω in series on CLK near the RP2350.)*
- [x] L50 → **AOTA-B201610S3R3-101-T**, symbol `mspc_power:AOTA-B201610S3R3-101-T` (footprint
      `footprints:L_Abracon_AOTA-B201610S_Polarity` and LCSC C42411119 filled in), §13.
      *(Done; pin 1 (the dot) on `+1V1_MCU`, pin 2 on `VREG_LX` — RP2350 datasheet §6.3.8,
      Figure 23/25.)* Layout per Figure 23/24: C_IN, L, C_OUT on the same side, no copper under L
      or the LX node on layer 2.
- [x] Remove `AUDIO_EN`: U25 EN back to `+3V3` (§2.5), delete its pull-down R36.
      *(R36 and R37 removed; U25 EN (pin 3) tied directly to `+3V3`.)*
- [x] I2C1 on GPIO46/47 with 4.7 kΩ pull-ups; TCA6408A (0x20), INA226 (0x40), INA3221 (0x41),
      MCP7940N (0x6F) (§2.8). *(R63/R64. INA3221 A0 goes to VS through R226, 5.6 kΩ, rather than
      straight to VS.)*
- [x] Move CARD_0/1/2_EN and VBUS_EN to the TCA6408A (keep the 7.5 kΩ pull-downs); expander
      RESET# to `MCU_RESET#`. Card RST# stay on RP2350 GPIO11/21/28 (decided 2026-09-13, §2.8).
      *(U200 P2–P4 CARD_0/1/2_EN, P5 VBUS_EN; P0/P1 eFuse FAULT#, P6 `VBUS_POWER_ALERT#`,
      P7 `RTC_MFP`.)*
- [x] Housekeeping interrupts: TCA6408A INT#, INA226 ALERT, INA3221 CRITICAL# and WARNING#,
      MCP7940N MFP, reaching one RP2350 GPIO with a 10 kΩ pull-up.
      *(Built as a cascade on **GPIO42**. `EXPANDER.INT#` (R65 pull-up) goes to GPIO42.
      INA226 ALERT and INA3221 WARNING/CRITICAL share `VBUS_POWER_ALERT#` on expander P6, and MFP
      (R240) is on P7. Firmware reads the expander's input port to see which device fired.)*
- [x] INA3221: 50 mΩ shunt per card rail between SY6280 OUT and the slot connector, A0 to VS (§2.8).
      *(R224/R225/R227, IN+ on `_RAW`. PV and TC are pulled up with 10 kΩ and wired to
      GPIO30/31.)*
- [x] MCP7940N + SC-32S 7 pF crystal + 2 × 6.8 pF C0G, VBAT to the coin cell; replaces the RV-3028-C7.
      *(U240, Y240, VBAT to BT240; C241/C242 6.8 pF C0G 0402.)*
- [x] VSYS shunt 20 mΩ, `+5V` split into source and load side, INA226 across it (§2.8).
      *(R221 `+5V_RAW` → `+5V`; U1 OUT, D3, C3, C8 and TP1 are on `+5V_RAW`. TP26 and D1 are on
      the load side.)*
- [x] CC sense: J_PWR on GPIO40, J_DBG on GPIO41, each 2 × 100 kΩ + 10 nF (§2.8).
      *(J1 → GPIO40, J2 → GPIO41.)*
- [x] RTC backup: place BT? `mspc_connectors:BatteryHolder_QJ_CR1220-2`, + to MCP7940N VBAT, − to GND (§2.8).
      *(BT240.)*
- [x] Panel LED (§9.1): J52 pin 1 from GPIO38 through 220 Ω, pin 2 from GPIO39 through 220 Ω,
      pin 3 GND. *(R66 `POWER_LED`, R67 `STATUS_LED`.)*
- [x] Probe power and reset (§3.1): R27 0 Ω → 100 kΩ to VSYS on U23 EN; 2N7002 on U23 EN
      (gate to probe GPIO10, 100 kΩ pull-down); 2N7002 on `MCU_RESET#` (gate to probe GPIO11,
      100 kΩ pull-down); 100 kΩ / 100 kΩ `+3V3` divider to probe GPIO26. Test point on the EN node.
      *(R27, Q120/Q121, R136–R139; TP28 on `POWER_OFF#`.)*
- [x] ~~Update the §2 power tree in the schematic notes: probe on U120 from `+5V_DBG`, not on
      `+3V3`.~~ Dropped: it referred to a v2 text note drawing the power tree on the schematic;
      the v3 sheets have none, and the tree lives in §2 of this document.
- [x] Remove the stray empty `untitled.kicad_sch`.
- [x] Re-annotated into the sheet ranges: card-line resistors R150–R184, R60.
      Ranges: power connector 1–19, power supply 20–49, RP2350 core 50–89, USB hub 90–99,
      audio 100–119, debug probe 120–149, MSPC bus 150–199, expander 200–219,
      power measurement 220–239, RTC 240–259.
- [x] Re-annotated the late additions into their ranges: power connector R2/3/7/8/13–16, C9/C10;
      RP2350 core R63–R65; debug probe Q120/Q121, R136–R139 (R134 left unused, it was removed);
      expander U200, C200; power measurement U220/U221, R220–R227, C220/C221; RTC U240, Y240,
      C240–C242, R240, BT240.
- [x] Re-annotate once the fixes above are in. *(Every reference is in its sheet's range: TP1,
      TP20–TP28, TP50, TP120–TP124, R66/R67.)*

**FPGA card** (§8.1)
- [ ] `U10` TLV62569DRL + 2.2 µH → 3.3 V from the local 5 V; divider 475 k / 105 k + feedforward cap.
- [ ] EN chain: `U4`, `U5`, `U6` enabled from `+3V3` through a ≈100 k / 60 k divider, not from `+5V`.
- [ ] `FB1` repurposed as the quiet branch (oscillator, programmer IOVDD); VCCIO banks on the raw
      buck output.
- [ ] Drop `+3V3_BUS` and `R28`; fold DET into INT per §7.
- [ ] 5 V input filter at J4 (47 µF bulk + ferrite/22 µF pi); replace the SOD-323 `D1`.
- [ ] Reconcile the FPGA part: the schematic has `LFE5UM-85F-7BG381`, §8 says `LFE5U-45F`.
      Different die and different power numbers — settle it before sizing anything.

**yasos**
- [ ] Flash rxdelay by timing target (~4.75 ns after the edge); calibrate inside `overclock_apply()`.
- [ ] Detect the flash by JEDEC ID (0x17 = 8 MB, 0x18 = 16 MB; enable QE on -IM/JM parts); drop
      the hard-coded 1024 blocks in `flash.zig` and the wrong `FLASH_MEMORY_SIZE="16MB"`.
- [ ] A/B rootfs OTA: `erase`/`write` in `flash.zig`, slot header, romfs offset at run time.
- [ ] Power: read CC (ADC) and the eFuse FAULT lines (expander P0/P1); drive the SY6280 ENs
      through the expander; handle the expander INT# on GPIO42 (§2.8);
      keep an unpowered slot's link pins high-Z. The TPS25200 has no analog current
      monitor, so the per-input current watch is gone.
- [ ] Front panel (§9.2): power button on `POWER_BUTTON` (GPIO43) — short press parks the cards
      and drops the slot and USB rails through the SY6280 ENs, then dormant; ~4 s forces the rails
      down. Drive the panel LEDs on GPIO38 (power) and GPIO39 (status), PWM for standby.
- [ ] debugprobe: request 500 mA in the configuration descriptor.
- [ ] debugprobe fork (§3.1): vendor commands `0x80` target power (off/on/cycle) and `0x81` RUN
      (release/assert/pulse); tri-state SWD and UART TX while the target is off; `+3V3` sense on
      ADC0; GPIO11 as the inverted CMSIS-DAP nRESET. CI script: `openocd … -c "cmsis-dap cmd 0x80 2"`.
- [ ] PCM5100A I2S driver (test on the Pico Plus 2 + VGA Demo Base).
- [ ] Card link protocol (PIO + DMA), enumeration by ID, INT/DET.
- [ ] TinyUSB ≥ 0.21.0 for enumeration through the hub (pico-sdk pins 0.18.0).

## 11. GPIO map (RP2350B) — as wired 2026-09-13

May still change during layout; update this table with the schematic.

| GPIO | function | GPIO | function |
|---|---|---|---|
| 0 | PSRAM CS1 | 29 | card 2 CS# |
| 1–8 | card 0 D0–D7 | 30 | INA3221 PV (power valid) |
| 9 | card 0 CLK | 31 | INA3221 TC (timing control) |
| 10 | card 0 INT# | 32–37 | SDIO CLK, CMD, D0–D3 |
| 11 | card 0 RST# | 38 | POWER_LED (panel LED 1, 220 Ω, §9.1) |
| 12 | card 0 CS# | 39 | STATUS_LED (panel LED 2, 220 Ω, §9.1) |
| 13–15 | I2S DIN (13), BCK (14), LRCK (15) | 40 (ADC0) | J_PWR CC sense |
| 16 | card 1 D0 (SPI0 RX = host MISO) | 41 (ADC1) | J_DBG CC sense |
| 17 | card 1 CS# (SPI0 CSn) | 42 | expander INT# (all housekeeping interrupts, §2.8) |
| 18 | card 1 CLK (SPI0 SCK) | 43 | POWER_BUTTON (J51, §9.1) |
| 19 | card 1 D1 (SPI0 TX = host MOSI) | 44 | UART0 TX (`UART_DEBUG.MCU_TX`) → probe GPIO5 |
| 20 | card 1 INT# | 45 | UART0 RX (`UART_DEBUG.MCU_RX`) ← probe GPIO4 |
| 21 | card 1 RST# | 46 | I2C1 SDA |
| 22–25 | card 2 D0–D3 | 47 | I2C1 SCL |
| 26 | card 2 CLK | | |
| 27 | card 2 INT# | | |
| 28 | card 2 RST# | | |

On the TCA6408A (0x20): P0 PRIMARY_POWER_FAULT# (J_PWR eFuse), P1 SECONDARY_POWER_FAULT# (J_DBG
eFuse), P2–P4 CARD_0/1/2_EN, P5 VBUS_EN, P6 VBUS_POWER_ALERT# (INA226 + INA3221), P7 RTC_MFP.
`AUDIO_EN` is gone (§2.5).

No spare GPIOs left (§12). The card links keep bus nomenclature
(D0/D1…) on the schematic — slot 1's mapping to SPI0 is noted above for firmware. Every card line
has a 33 Ω series resistor, and CS#, RST#, INT# a 7.5 kΩ pull-down (INT# doubles as DET, §7); 7.5 kΩ
is also used on the SY6280 EN pull-downs, below the 8.2 kΩ that RP2350 erratum E9 calls for on A2
silicon.

## 12. Open decisions

**Decided 2026-09-12** — 3.3 V on the slots (`+3V3_BUS`) is dropped; each card regulates from its
switched 5 V. The FPGA card was the only one still relying on it (§8.1). The trigger was measured
noise on the shared rails with a card in the slot.

- Power-off depth (§9.2): remote hard off through the probe is decided (§3.1). Open: whether the
  front-panel button also gets a hard off, read by the probe (only with J_DBG plugged in), or stays
  soft off.
**Decided 2026-09-13** — RTC backup is a CR1220 in a Q&J CR1220-2 SMD holder (§2.8).
**Decided 2026-09-13** — no BOOTSEL pads on the RP2350B's `QSPI_CSN0`. Its USB goes to the CH334R
upstream port, so USB boot can't reach a host anyway; flashing and recovery go through the on-board
debug probe's SWD, which is always fitted.
**Decided 2026-09-13** — no test points on SWD or UART (§10).
**Decided** — the mounting holes have no pads, on purpose.
- **No spare GPIO is left** (card RST# stayed on 11/21/28, GPIO39 drives the second panel LED and
  GPIO43 reads the power button). If
  one is needed, INA3221 PV/TC on GPIO30/31 are the cheapest to give up. Candidates, in rough order of value:
  - a *handshake / data-ready* line for the network card — the ESP32 SPI slave needs one to tell
    the host it has data; J151 pins 5 and 6 are already free;
  - SD card detect — J53 (DM3AT-SF-PEJM5) has a CD switch on pads 9/10; it needs the
    `Micro_SD_Card_Det_Hirose_DM3AT` symbol and a GPIO;
  - a JST SH 4-pin **Qwiic/STEMMA QT** header on I2C1 plus one or two spare GPIOs on a header,
    for sensors and bring-up — same connector family as the front panel;
  - a piezo buzzer (PWM) for CI-node alarms that nobody is watching on a screen;
  - card 0 / card 2 REQ-GNT or a frame line — needs connector pins those slots do not have yet.
- The TPS25200 auto-retries instead of latching off after a fault (§2.7). Accept the retry loop on
  the CI node, or add a latch (firmware holding the eFuse EN low once FAULT is seen, which needs
  the 3.3 V rail to survive the fault).

## 13. KiCad libraries
- `mspc_power` (`libs/symbols/mspc_power.kicad_sym`): TPS25200, SY6280AAC, PMEG3020EP (D3, on
  `footprints:D_SOD-128`), the now-unused TPS259470LRPWR, and **AOTA-B201610S3R3-101-T** (L50,
  RP2350 SMPS inductor): `Device:L` graphics with a polarity dot at pin 1, Abracon/LCSC C42411119
  data filled in. Pin 1 = dot = `+1V1_MCU`, pin 2 = `VREG_LX` (RP2350 datasheet Figure 23/25).
- `footprints:L_Abracon_AOTA-B201610S_Polarity`: Abracon's recommended land pattern for the
  2.0 × 1.6 mm AOTA-B201610S (2 pads 0.8 × 1.6 mm, 0.7 mm gap, i.e. on a 1.5 mm pitch), silk and
  fab dot at pad 1. Replaces the old `RP2350_80QFN_minimal:L_pol_2016` (0.7 × 1.7 mm pads, no
  datasheet reference). No 3D model.
- `footprints:TI_DRV0006A_WSON-6-1EP_2x2mm_P0.65mm` (TPS25200; land pattern per TI drawing
  4222173/C: 6 pads 0.45 × 0.3 mm on a 1.95 mm span, thermal pad 1.0 × 1.6 mm).
  `footprints:TI_RPW0010A_VQFN-HR-10_2x2mm_P0.45mm` is left over from the TPS259470L, now unused.
- `footprints:L_Murata_DFE252012F` (buck inductor; Murata's recommended land, hand-made).
- `mspc_xtal` (`libs/symbols/mspc_xtal.kicad_sym`): ABM8-272-T3, KiCad 10 `Device:Crystal_GND24`
  graphics with Abracon part data; footprint `Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm`.
- `mspc_usb` (`libs/symbols/mspc_usb.kicad_sym`): CH334R, USBLC6-4SC6, copied from the KiCad 10
  standard libraries; footprints `footprints:QSOP-16_3.9x4.9mm_P0.635mm` and
  `footprints:SOT-23-6_Handsoldering`.
- `mspc_connectors` (`libs/symbols/mspc_connectors.kicad_sym`): JST SH 1.00 mm front-panel
  headers — `JST_SH_SM02B-SRSS-TB` / `JST_SH_SM03B-SRSS-TB` (side entry) and
  `JST_SH_BM02B-SRSS-TB` / `JST_SH_BM03B-SRSS-TB` (top entry). KiCad 10
  `Connector_Generic_MountingPin` graphics — the `MP` pin is the shell tab, tie it to GND —
  with the footprint, datasheet and LCSC code filled in. Footprints are the KiCad 10
  `Connector_JST` ones, copied into `libs/footprints.pretty/` so no stock library is needed.
  Also `BatteryHolder_QJ_CR1220-2` (KiCad 10 `Device:Battery_Cell` graphics, Q&J part data) with
  `footprints:BatteryHolder_QJ_CR1220-2_SMD`, drawn from the QJ1220-2SMT drawing in the LCSC
  datasheet.
- KiCad standard, for the 2026-09-13 additions: `Timer_RTC:MCP7940N-xSN`,
  `Power_Management:INA3221`, `Sensor_Energy:INA226`, `Interface_Expansion:TCA6408APW`.
- KiCad standard: TLV62569DRL (SOT-563),
  PCM5100A, RP2040, W25Q16JVSS,
  USB_C_Receptacle_USB2.0_16P, TVS2200DRV; the SY6280 uses SOT-23-5_HandSoldering.

## 14. Parts to order — added 2026-09-13
New line items from the 2026-09-13 review. LCSC codes marked *(verify)* were not confirmed on the
LCSC page; order by manufacturer part number and check stock.

| qty / board | ref | part | LCSC | notes |
|---|---|---|---|---|
| 1 | L50 | Abracon **AOTA-B201610S3R3-101-T**, 3.3 µH, 2016, shielded, 2.1 A, 140 mΩ | **C42411119** | RP2350 SMPS inductor; dot (pin 1) to `+1V1_MCU` (RP2350 datasheet Fig. 23) |
| 1 | new | TI **INA226AIDGSR**, VSSOP-10 | **C49851** | VSYS current/voltage/power, I2C 0x40 (§2.8) |
| 1 | new | TI **TCA6408APWR**, TSSOP-16 | **C206177** | I2C I/O expander 0x20: slot/USB enables, FAULT# and alert inputs (§2.8) |
| 1 | new | TI **INA3221AIRGVR**, VQFN-16 | **C181255** | card 0/1/2 rail current/voltage, I2C 0x41 (§2.8) |
| 3 | new | Yageo **PT0603FR-7W0R05L**, 50 mΩ ±1 % 0.2 W, **0603** | **C784595** | INA3221 shunts, one per card rail; 43 mW at 0.93 A |
| 1 | new | Microchip **MCP7940N-I/SN**, SOIC-8 | **C51106** | RTC 0x6F (§2.8); replaces the RV-3028-C7 (C3019759, out of stock) |
| 1 | new | Seiko **SC-32S 32.768 kHz 20 ppm 7 pF**, 3215 | **C97604** | RTC crystal; CL must be 6–9 pF |
| 2 | new | 6.8 pF C0G 0402 | stock part | RTC load capacitors *(verify on the bench)* |
| 1 | new | Q&J **CR1220-2** SMD coin cell holder | **C70381** | MCP7940N VBAT (§2.8) |
| 1 | — | CR1220 lithium cell | — | fitted by hand after assembly, not part of the PCBA |
| 1 | new | Panasonic **ERJ-6CWDR020V**, 20 mΩ ±0.5 % 0.5 W ±75 ppm/°C, **0805** | **C2089540** | VSYS shunt for the INA226; 132 mW at 2.57 A. Alternates: Vishay RCWE120620L0FMEA (1206, ±300 ppm, C2075929), Milliohm HoGXT0805-1/2W-20mR-1% (0805, C2977780) |
| 2 | new | 4.7 kΩ 1 % 0402 | stock part | I2C1 pull-ups |
| 1 | new | 10 kΩ 1 % 0402 | stock part | expander INT# pull-up |
| 4 | new | 100 kΩ 1 % 0402 | stock part | CC1/CC2 → ADC0, ADC1 |
| 2 | new | 10 nF X7R 0402 | stock part | CC sense filters |
| 4 | new | 100 nF X7R 0402 | stock part | decoupling: INA226, INA3221, TCA6408A, MCP7940N |
| 26 | R150–R184 | 33 Ω 1 % 0402, UNI-ROYAL **0402WGF330JTCE** | **C25105** | card-line series resistors (card 0: 12, card 1: 6, card 2: 8) |
| 13 | R150–R184, R20/21/26/33 | 7.5 kΩ 1 % 0402, UNI-ROYAL **0402WGF7501TCE** | *(verify)* | 9 card pull-downs + 4 SY6280 EN pull-downs |
| 2 | new | **2N7002** N-MOSFET, SOT-23 | **C8545** | probe → buck EN and probe → RUN (§3.1) |
| 5 | new, R27 | 100 kΩ 1 % 0402 | stock part | buck EN pull-up (replaces R27 0 Ω), 2 gate pull-downs, `+3V3` sense divider |
| 2 | R66, R67 | 220 Ω 1 % 0402 | stock part | panel LED series resistors, GPIO38 and GPIO39 (§9.1) |
| — (panel) | — | bi-colour common-cathode LED (e.g. yellow-green/amber, both V_F ≈ 2 V), or a single low-V_F LED | — | on the front-panel side, not the mainboard |
| 1 | FB50 | Murata **BLM15AG601SN1D**, 600 Ω @ 100 MHz, 300 mA, 520 mΩ, **0402** | **C76884** | `ADC_AVDD` filter: `+3V3` → FB50 → C74 100 nF + C75 1 µF → U50 pin 59 |
| 1 | J53 | Hirose **DM3AT-SF-PEJM5**, microSD push-push, SMD | **C114218** | replaces the hinged Würth 693072010801 (hard to swap cards); footprint `Connector_Card:microSD_HC_Hirose_DM3AT-SF-PEJM5`. Pads 1–8 match the `Micro_SD_Card` symbol; pads 9/10 are the card-detect switch, not wired (§12) |
| 2 | J1, J2 | HRO **TYPE-C-31-M-12**, USB-C 16-pin SMD, 5 A | **C165948** | J_PWR (J1) and J_DBG (J2); footprint `Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12` was already assigned |
| 2 | U53, U54 | ST **USBLC6-4SC6**, SOT-23-6 (same part as U90/U91) | **C111212** | SD ESD at J53: U53 on DAT0–DAT3, U54 on CMD and CLK (I/O3, I/O4 not connected); VBUS pins on `+3V3` with C78/C79 100 nF |
| 1 | C77 | 100 nF X7R 0402 | stock part | J53 VDD decoupling, next to C58 |
| 1 | C58 | Samsung **CL31A106KAHNNNE**, 10 µF 25 V X5R **1206** | **C9807** | J53 VDD bulk for card-insertion inrush; was 10 µF 0402 6.3 V (≈5 µF left at 3.3 V) |
| 2 | J90, J91 | Jing Extension **907-111A1022D10200**, dual stacked USB-A, THT right angle | **C12049** | fits the assigned `USB_A_Wuerth_61400826021_Horizontal_Stacked` footprint: pins 0/2.5/4.5/7.0 mm, rows 2.60 mm (footprint 2.62), shell legs Ø2.30 at 13.10 mm (13.14) — checked against the datasheet |
| 2 | U90, U91 | ST **USBLC6-4SC6** | **C111212** | USB-A port ESD (§4) |
| 1 | U4 | ST **USBLC6-4SC6** | **C111212** | J_DBG USB ESD (§3) |
| 1 | U52 | AP Memory **APS1604M-3SQR-SN**, 16 Mbit PSRAM, SOP-8 150 mil | **C18214056** | RP2350 PSRAM on CS1 (§6) |
| 2 | U25, U120 | TI **TLV70033DDCR**, SOT-23-THIN | **C11337** | audio LDO (§2.5) and probe LDO (§3) |
| 1 | C11 | Panasonic **EEHZA1V470P**, 47 µF 35 V hybrid polymer, 6.3 × 5.8 mm | **C178639** | `+5V_DBG` bulk against a faulty-charger spike ahead of the TLV70033 (§3) |
| 3 | C38–C40 | Samsung **CL31A106KAHNNNE**, 10 µF 25 V X5R 1206 | **C9807** | SY6280 input capacitors on `+5V` (§2.4) |
| 1 | C80 | 1 µF 16 V X5R 0402 | stock part | PSRAM U52 VDD (§6) |
| 1 | C201 | 100 nF 16 V X7R 0402 | stock part | second TCA6408A supply pin (VCCP) |
| 1 | R69 | 10 kΩ 1 % 0402 | stock part | `POWER_BUTTON` pull-up (§9.1) |
| 2 | R17, R18 | 47 kΩ 1 % 0402 | stock part | USB-C VBUS bleeders on `+5V_CONN_1` / `+5V_CONN_2` (§2.1) |
| 1 | R140 | 2.2 kΩ 1 % 0402 | stock part | `+5V_DBG` bleeder for D3's reverse leakage (§3) |

From own inventory, no LCSC code: status LEDs D1, D2, D20–D24, D120–D124 (0402, colour chosen at
assembly; check the 5.6 k / 10 k series resistors against the chosen V_F), audio jack J100, and the
generic 1.27 mm card sockets J150–J152.

Removed by the same review: R134 (probe GPIO6 path), `AUDIO_EN` with R36 and R37, the probe's
USBLC6-2SC6 (now U4, USBLC6-4SC6), and the INA180A2 plan (replaced by the INA226).
