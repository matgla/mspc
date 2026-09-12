# MSPC v2.5 — mainboard changes

Decided 2026-09-11. The board stays on the **RP2350B**. Values marked *(verify)* are estimates to
check before ordering the PCB.

## 1. Summary

| area | v2 | v2.5 |
|---|---|---|
| power input | 12 V jack | **2× USB-C at 5 V**: J_DBG (data + power), J_PWR (power only, has priority) |
| input protection | — | **TPS25200 eFuse per input** (20 V tolerant, 5.4 V output clamp), the two forming a priority mux |
| 3.3 V | BD9D321EFJ + AP2210 | **TLV62569DRL** buck |
| 5 V | BD9D321EFJ + LD1117S50 | no regulator; the USB input through **SY6280AAC** switches |
| programming / console | external debug probe | **on-board RP2040 running debugprobe** (SWD + UART on J_DBG) |
| USB hub | HS8836A + 10× LESD5D5.0CT1G | **CH334R** on 3.3 V; **2× USBLC6-4SC6** ESD arrays on the USB-A ports |
| audio | TLV320DAC3203 + I2C | **PCM5101A** (I2S only, line out) on its own **TLV70033** LDO |
| I2C / J7 | I2C connector | removed |
| flash | W25Q64JVSIM fitted (8 MB) | **W25Q128JVSIQ** (16 MB, room for A/B rootfs) |
| PSRAM | APS6404L (drawn as a W25Q128JVS) | APS6404L with its own symbol |
| card bus | shared 16-bit, 6× CS, 74HCS151 + 74LVC139 | **point-to-point link per slot**, no glue logic |
| slots | J13 16-bit, J14 8-bit, J15 8-bit | **3 slots**: VGA/DVI (4-bit), FPGA (8-bit), network (4-bit) |
| slot 3.3 V | `+3V3_BUS` on every slot | **dropped** — each card makes its own 3.3 V from its switched 5 V (§8.1) |
| front panel | — | **3× JST SH 1.00 mm**: power LED (3 pins), reset and power button (2 pins each) (§9) |

Removed: 12 V jack, BD9D321EFJ ×2, AP2210, LD1117S50, TLV320DAC3203, 74HCS151, 74LVC139, J7,
HS8836A (+ R19, R20, C41–C43), LESD5D5.0CT1G ×10.

Added: 2× USB-C, 2× TPS25200, 1 PMEG3020EP Schottky, TLV62569DRL + 2.2 µH, 4× SY6280AAC, RP2040 +
W25Q16JVSS + 12 MHz crystal + USBLC6-2SC6, PCM5101A + TLV70033, CH334R (+ 12 MHz crystal),
2× USBLC6-4SC6, 3× slot power LED + resistor, 3 front-panel JST SH headers (J_LED, J_RST,
J_PWR_BTN).

---

## 2. Power

```
J_DBG (USB-C, data)  ─ TPS25200 #1 ─►|─┐   D1: J_PWR wins
  └ D+/D- → RP2040 probe               ├──┬── VSYS (5 V, protected)
J_PWR (USB-C, power) ─ TPS25200 #2 ────┘  │
                                          ├─► TLV62569 ─► 3.3 V: MCU, probe, hub, memories, SD,
                                          │                       DAC digital side
                                          ├─► TLV70033 ─► 3.3 V audio: DAC AVDD + CPVDD
                                          ├─► 3× SY6280, one per slot (shared EN) ─► 5 V to slots 1, 2, 3
                                          └─► SY6280 (EN) ─► 5 V to the USB ports
```

### 2.1 USB-C inputs
- CC1 and CC2 each with their own 5.1 kΩ 1 % to GND (never one shared resistor).
- 4.7 µF / 50 V ceramic at each eFuse input. No fuse, no TVS (optional unpopulated TVS2200
  footprint). Never a 5–6 V TVS: it burns on a faulty charger.
- USB-C shells straight to GND with several vias at the connector; no chassis, so no RC network
  (USB-A: §4).

### 2.2 eFuses: TPS25200
One per input; each is that input's fuse: current limit, input overvoltage shutoff, output clamp,
reverse-current blocking (while disabled) and soft start. 2.5–6.5 V operating, **IN withstands
20 V**; WSON-6 2 × 2 mm with thermal pad, 60 mΩ, θJA 66.5 °C/W.

| setting | J_PWR (primary) | J_DBG (auxiliary) |
|---|---|---|
| R_ILIM (current limit), 1 % | 40.2 kΩ → 2.26–2.57 A (2.42 nom) | 110 kΩ → 0.80–0.97 A (0.88 nom); hardware ceiling, firmware honours CC |
| EN | 300 kΩ pull-up to its own IN (TI's value; an internal zener clamps the pin) | same |
| FAULT | open-drain, 300 kΩ pull-up to 3.3 V (100 kΩ for margin) | same |
| C_IN | 0.1 µF at the pin, plus the 4.7 µF of §2.1 | same |
| C_OUT | 1 µF at OUT (OUT *is* VSYS here, so C9 is its bulk) | 1 µF at OUT, ahead of D1 — its own node |

- **Overvoltage: fixed thresholds, no dividers.** Above ~5.55 V in, the output is clamped to
  5.25–5.55 V (5.4 V typ) — under the 6 V limit of the TLV62569 and the SY6280s — and above
  **7.6 V** in, the switch disconnects (0.6 µs). Short-circuit response 3.5 µs, current limit
  ±6 %.
- **Priority mux: a Schottky in the J_DBG path, both eFuses always enabled.** With both inputs
  present, VSYS follows J_PWR at ~5 V and D1 is reverse-biased (a few µA), so the charger carries the
  load; when J_PWR drops, D1 conducts at once — no switchover gap. Nothing back-feeds the PC: D1
  blocks it, and an eFuse below its 2.35 V UVLO disables itself and blocks reverse current.
  D1 = **PMEG3020EP** (SOD-128, 30 V, 2 A): V_F 275 mV typ / 310 mV max at 1 A, ~250 mV at 0.5 A,
  I_R 130 µA at 5 V. The leakage only trickles into the disabled auxiliary eFuse's 480–625 Ω
  discharge path (< 1 mW), so the low-V_F part wins over the low-leakage PMEG3020BEP (+90 mV) and
  the 5 A PMEG3050EP (−35 mV for 330 µA). **Not the inventory BAT60J:** SOD-323 is thermally
  limited to 310 mW at 400 °C/W, so its 3 A rating is pulsed, and a 1.2 A fault would cook it.
  - **Not an EN-driven mux** (an N-FET pulling the auxiliary's EN low): the TPS25200 needs 5.1 ms
    typ, 7.3 ms max to turn on, so every source swap would brown the board out — no bulk capacitor
    bridges 5 ms. The TPS259470L's dedicated mux did it in ~90 µs.
  - Cost: ~0.25–0.3 V on the PC path only (VSYS ≈ 4.7 V when J_DBG feeds the board); the buck
    regulates from a 3.4 V input up.
  - C9 (≥ 100 µF) sits on VSYS itself, where D1's cathode meets eFuse #1's OUT, ahead of the buck
    and the SY6280s — one shared bulk, not one per eFuse. Make it **EEHZC1V151P**
    (Panasonic ZC hybrid polymer aluminium, 150 µF/35 V, ESR 27 mΩ, 1.6 A ripple at 100 kHz,
    Ø8 × 10.2 mm, 4000 h at 125 °C), with 10 µF + 0.1 µF ceramic beside it for the fast edges.
    It is **polarized** — `Device:C_Polarized` on `footprints:CP_Elec_8x10.5`, with the minus
    marking on the silkscreen; the same applies to the VBUS bulk (§2.4).
    No MnO2 tantalum (§2.7); a 100 µF MLCC would work electrically but loses much of its value to
    DC bias and is the part most likely to crack into a short. It also damps the OVP overshoot: the
    output follows the input until the eFuse reacts (0.6 µs), and more output capacitance lowers
    that peak (datasheet Figure 8-3). Each eFuse still gets a local 1 µF at its own OUT pin; on the
    auxiliary that is the only capacitance on its side of D1.
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
- R1 (VOUT–FB) 453 kΩ (E96; 450 kΩ is E192), R2 (FB–GND) 100 kΩ → 0.6 V × (1 + R1/R2) = 3.32 V.
  R2 must be ≤ 200 kΩ. **Fit the 6.8 pF feedforward across R1** (C0G, TI's value for exactly
  R2 = 100 kΩ): the divider is high-impedance, so a few pF of stray at FB already forms a pole near
  the loop's crossover, and this adds the zero that offsets it — worth it on a rail that sees SD
  card and MCU load steps. The datasheet calls it optional, so it can be depopulated if the bench
  says otherwise. 0603 for the resistors.
- **Divider tolerance matters for the hub.** The feedback reference is 0.588–0.612 V (±2 %), so
  with 1 % resistors the rail spans ~3.20–3.44 V. Everything tolerates that except the CH334R's
  3.2–3.4 V window (§4) — no damage, its absolute maximum is 4.0 V, but out of spec at the top.
  Use 0.1 % resistors for the divider and the span tightens to ~3.23–3.38 V.
- C_IN 4.7–10 µF, C_OUT 10 µF (X7R/X5R, ≥ 10 V), EN to VSYS.
- L 2.2 µH: **Murata DFE252012F-2R2M=P2** (LCSC C576403) — 2.5 × 2.0 × 1.2 mm, I_sat 3.6 A,
  I_rms 2.3 A, DCR 82 mΩ, $0.11–0.15. Its I_sat clears the converter's ≥ 2.5 A current limit, so
  it replaces the 4 × 4 mm XAL4020 outright and takes the regulator block from ~24 mm² to ~13 mm².
  Not the FPGA card's 0805 part, and not a low-I_sat 2520 like the SWPA252012S (1.25 A, 216 mΩ).
  Footprint `footprints:L_Murata_DFE252012F` — KiCad has no DFE252012F, and the stock 2520 patterns
  don't match: Murata wants 1.2 × 2.0 mm pads on a 2.8 mm span, while e.g. L_Changjiang_FNR252010S
  has 0.85 × 2.0 mm on 2.5 mm, ending flush with the body and leaving no outer fillet.
- Ferrite bead + local capacitance to the PCM5100A's AVDD/CPVDD.
- Layout per datasheet Figure 21. Thermal: ~+8 °C at the 0.3 A peak, fine with a ground pour.

### 2.4 Switched 5 V: SY6280AAC
EN active high with a 100 kΩ pull-down per EN net — one for the three slot switches, one for the
USB-port switch. The datasheet says EN must never float; it turns on above 2.4 V (with VIN
4.2–5.5 V) and off below 0.8 V, and leaks ≤ 1 µA, so an RP2350 GPIO drives it directly at 3.3 V
and the 100 kΩ holds it at ≤ 0.1 V while the MCU is in reset. No series resistor.
I_lim = 6800 / R_SET (±25 %).

| branch | R_SET | limit |
|---|---|---|
| card 0 FPGA | 9.1 kΩ | 0.56–0.93 A — the only card that may reach ~0.4 A |
| card 1 network (ESP32-C3) | 13.7 kΩ | 0.37–0.62 A — ~2.5× the 0.25 A transmit peak |
| card 2 VGA/DVI | 13.7 kΩ | 0.37–0.62 A *(verify)* |
| USB ports (all four, one switch) | 6.8 kΩ | 0.75–1.25 A — one 500 mA device plus HID and a drive |

- The three card switches share one EN. Slot numbering follows the schematic: card 0, 1, 2.
- Decoupling: 1 µF ceramic at each switch's IN and 0.1 µF at each OUT; each slot also gets 10 µF at
  its connector *(verify)*. All of it **16 V or more, X7R/X5R** like the rest of the
  5 V side (§2.2). Standard 10 µF part: **CL31A106KAHNNNE** (1206, 25 V, X5R, ±10 %) — the 1206
  case and 25 V rating keep most of the capacitance at 5 V bias, where a 0603 6.3 V part would
  keep little. Use it for the other 10 µF spots too (buck C_OUT, the ceramic beside C9). The USB ports' output bulk is the ≥ 120 µF below. Silergy's
  preliminary datasheet gives no C_IN/C_OUT recommendation — it only tests with C_L = 1 µF — so
  these are defaults, and the switches sit on the VSYS bulk anyway.
- ≥ 120 µF low-ESR on the USB ports' VBUS, at the connectors: the same **EEHZC1V151P** as C9.
- Give the USB-port switch copper on IN/OUT (~+45 °C at 1.5 A).
- **Slot power LED**, one per slot on the switched output at the connector: **SZYY0402YG** (0402
  yellow-green, 571 nm) with **10 kΩ** to GND, ≈ 0.33 mA each and ~1 mA for all three. Brightness
  scales with current, so the 58 mcd at 20 mA becomes ~1 mcd — a soft glow that still says which
  slots yasos has enabled. Use 6.8 kΩ (0.5 mA) if it reads too dim in daylight, 22 kΩ (0.15 mA)
  for barely-there. Green means "powered"; keep the orange-red ZSR1-1105C-045-Z4 for faults — an
  LED and 1.5 kΩ from 3.3 V to an eFuse FAULT pin lights when that input trips and costs no GPIO
  (FAULT sinks up to 25 mA). 0402 is smaller than the SOD-523 parts that were awkward to place;
  use 0603 if these are hand-soldered.

### 2.5 Audio 3.3 V: TLV70033
- `+3V3_AUDIO` from a **TLV70033** (SOT-23-5, fixed 3.3 V, 200 mA) fed from **VSYS**, per the same
  RP2040 reference: a linear regulator ahead of the analogue supply keeps the TLV62569's switching
  ripple off it. Raspberry Pi report that buck's PFM ripple being visible on their VGA DAC output.
- 1 µF in and 1 µF out, EN pulled to `+3V3` rather than to its own input — so the audio rail comes
  up only after the digital rail, which is free sequencing.
- **It must be fed from VSYS, not `+3V3`**: 3.3 V in for 3.3 V out leaves no headroom, the part
  never leaves dropout and its PSRR is zero, which defeats the whole point.
- Load is ~25 mA, so the 1.7 V drop burns ~45 mW. Operating input is 2–5.5 V with a 6 V abs max;
  VSYS reaches 5.55 V only inside the TPS25200's OV-clamp window, the same fault-window
  exceedance already accepted for the CH334R's V5 (§4).
- Place it beside the DAC, not beside the buck — a clean rail routed across the board past the
  switchers picks the noise back up.

### 2.6 Budget
- Mainboard, hub included: ~120 mA typ, ~300 mA peak at 3.3 V (~85 mA from 5 V through the buck).
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
  USB and brings it back in a loop; that pattern is the CI fault signal. Recovery: power-cycle
  J_PWR (uhubctl-capable hub port or a relay on the charger). Whether auto-retry is good enough
  here is open (§12).

---

## 3. On-board debug probe: RP2040
- RP2040 + W25Q16JVSS + 12 MHz crystal, debugprobe firmware.
- ESD: USBLC6-2SC6 right at J_DBG, after the two D+ (A6/B6) and two D- (A7/B7) pins are joined,
  before the RP2040's 27 Ω series resistors. I/O1 (pins 1, 6) on D+, I/O2 (3, 4) on D-, GND (2)
  by a short via. **VBUS (pin 5) to the 3.3 V rail, with 100 nF (≥ 10 V) at the pin** —
  not to VSYS and never to the connector-side VBUS. That pin is the clamp reference, so the data
  lines can only rise to it plus a diode drop: ~4 V from 3.3 V, ~6 V from VSYS, and whatever a
  faulty charger delivers if it were tied to the connector. RP2040/RP2350 USB pins are **not 5 V
  tolerant** — they are USB IO fed from USB_OTP_VDD (3.135–3.63 V), so their limit is that supply
  + 0.5 V, about 3.8 V; the datasheet's 5.5 V applies only to the Digital IO (FT) GPIOs.
- To the RP2350: SWCLK, SWDIO, RUN, console UART (TX/RX).
- Powered from the board's 3.3 V rail, not from J_DBG's VBUS.
- VBUS divider from J_DBG to an RP2040 GPIO (needs a debugprobe build that honours it).
- **BOOTSEL button (SW_DBG_BOOT)**: XUNPU **TS-1088-AR02016**, LCSC **C720477**, a JLCPCB
  basic/preferred part (3.9 × 3.0 × 2.0 mm, 1.6 N, 100 k cycles) — footprint
  `Button_Switch_SMD:SW_SPST_TS-1088-xR020` ships with KiCad 10, no custom library entry.
  Wire it from the W25Q16's `QSPI_SS`/CS net to GND through a **1 kΩ series resistor**; that
  series R is what keeps normal XIP reads undisturbed, and the RP2040 holds CS high internally
  for the boot sample, so no pull-up is needed. This is the only way to reflash the probe
  without a second probe, so it must stay reachable with all cards seated.
- **Probe reset (SW_DBG_RUN)**: same part, straight from the RP2040's `RUN` to GND. Optional,
  but the pair next to each other is the usual reflash gesture (hold BOOT, tap RUN).
  Smaller alternative for both if the area is tight: XKB **TS-1187A-B-A-B**, LCSC **C318884**
  (3.0 × 2.0 × 1.5 mm), footprint `Button_Switch_SMD:SW_Push_1P1T_XKB_TS-1187A`.

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
  and drop the whole hub (datasheet §6.1). Both fixes it names are already in: C54 = 150 µF on
  `+5V_VBUS`, 10 µF on the LDO output.
- **ABM8-272-T3** 12 MHz across XI (16) and XO (15), **no load capacitors** — the CH334R's are
  built in. Crystal-free (XI to GND, XO open) only if the ordered part has it enabled, and the
  datasheet warns those parts "may deviate from the USB specification": fit the crystal.
- RESET# open (internal pull-up). Pull-up and pull-downs are built in: no 1.5 kΩ pull-up on the
  RP2350's D+.
- Upstream: RP2350 USB_DP → DPU (pin 11), USB_DM → DMU (pin 10), through the 27 Ω series resistors.
- Ports powered only through the SY6280 (the CH334R has no PWREN/OVCUR pins).
- **ESD: one USBLC6-4SC6 per stacked USB-A connector** on its 4 data lines, next to the
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

- I2S on BCK (13), DIN (14), LRCK (15). BCK = 32× or 64× fS.
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
- Flash: W25Q128JVSIQ (16 MB) in the BOM; same 208-mil SOIC-8 footprint.
- PSRAM: APS6404L unchanged; give it its real symbol (v2 shows a W25Q128JVS on CS1).

## 7. Card link: shared bus → point-to-point

| slot | card | link | extra lines | bridge |
|---|---|---|---|---|
| **card 0** | FPGA (ECP5-45F) | 8-bit on GPIO12–19 + CLK | INT#, RE#, SEL | RP2040 |
| **card 1** | network (ESP32-C3) | **1-bit SPI**: CLK + MOSI + MISO + CS | INT#, RE# | none — the card *is* the bridge |
| **card 2** | VGA / DVI | 4-bit + CLK | INT#, RE# (+ optional REQ/GNT) | RP2350B |

**Slot 2 is the edge slot and the VGA/DVI card has to live there** — its display connector must
reach the chassis edge, which is what fixes the assignment. The network card takes slot 1; the two
data lines on that connector are MOSI and MISO, which is why the slot reads "2-bit" in the
schematic. **RE# is per slot**, not the shared RESET this section used to specify: one card can be
restarted without disturbing the others, which a shared line cannot do.

- One PIO state machine + DMA channel per link, DDR, host-clocked; command protocol
  `[CMD][ADDR][LEN][DATA…]`; the card answers an ID command.
- **The link is SPI-shaped**: host-clocked, command-framed, half-duplex with the data lines turning
  around, so the 4-bit form is QSPI and a slot can drop to plain 1-bit SPI. Slot 1 does exactly
  that, because the ESP32-C3's half-duplex slave has **no quad mode** — single line only, specified
  to 60 MHz and realistically 20–40 MHz across a connector, i.e. 2.5–5 MB/s. That is ample for
  WiFi and the reason this slot alone carries a CS: an SPI slave needs it to frame transactions.
- INT doubles as DET: 100 kΩ pull-down on the host, driven high by a running card, low = interrupt.
- PIO windows: GPIO0–31 (base 0) or GPIO16–47 (base 16) — the FPGA link needs base 0, the PIO
  SDIO on GPIO32–37 base 16.
- 22–33 Ω series resistors at the drivers; a ground every 2–4 signals in the connectors.
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
on. Use §7's scheme instead: host-side 100 kΩ pull-down on INT/DET, driven high by a running card.
The card currently has INT and DET on separate pins (J4 23 and 24) and does not match §7 yet.

**5 V input path.** `+5V_BUS` and `+5V_USB` have no capacitors on them at all, and the only bulk on
the 5 V node is 3 × 4.7 µF *after* the `Q1`/`D1` OR — thin for a 1 A buck plus two LDOs, and about
to carry the 3.3 V load as well. Add the §8 pi filter at J4, and replace `D1`: it is a SOD-323
part carrying the whole card when running from USB.

## 9. Front panel

Three JST **SH 1.00 mm** headers at the board edge — the family the Pico uses for its debug port.
Keyed, so a panel cable cannot go on backwards, and 1 A / 50 V is ample for two buttons and an LED.
Side entry (`SM..B-SRSS-TB`) lets the cable leave flat along the board; the top-entry
`BM..B-SRSS-TB` parts are in the library as alternates.

| ref | function | part | LCSC | symbol / footprint (both `..._Horizontal`) |
|---|---|---|---|---|
| J_LED | power LED, 3 pins | SM03B-SRSS-TB(LF)(SN) | **C160403** | `mspc_connectors:JST_SH_SM03B-SRSS-TB` / `footprints:JST_SH_SM03B-SRSS-TB_1x03-1MP_P1.00mm_Horizontal` |
| J_RST | reset button, 2 pins | SM02B-SRSS-TB(LF)(SN) | **C160402** | `mspc_connectors:JST_SH_SM02B-SRSS-TB` / `footprints:JST_SH_SM02B-SRSS-TB_1x02-1MP_P1.00mm_Horizontal` |
| J_PWR_BTN | power button, 2 pins | SM02B-SRSS-TB(LF)(SN) | **C160402** | as J_RST |

Top entry if the panel sits above the board: **C160389** (BM03B-SRSS-TB) and **C160388**
(BM02B-SRSS-TB), symbols `JST_SH_BM03B-SRSS-TB` / `JST_SH_BM02B-SRSS-TB`, footprints
`..._1x0n-1MP_P1.00mm_Vertical`. All four are in `mspc_connectors` with the footprint, datasheet
and LCSC code already filled in (§13).

### 9.1 Pinout and wiring
- **J_LED**: 1 = LED supply, 2 = `FP_LED`, 3 = GND. A plain LED wires anode to pin 1 (series
  resistor on the board, sized like the slot LEDs of §2.4) and cathode to pin 3, leaving pin 2
  open; an addressable LED takes supply, `FP_LED` as its data line and GND — the same header
  either way.
- **J_RST**: 1 = RP2350B `RUN`, 2 = GND.
- **J_PWR_BTN**: 1 = `FP_PWR_BTN` (GPIO with the internal pull-up), 2 = GND.
- **100 Ω in series and 100 nF to GND** on `RUN` and `FP_PWR_BTN` at their connectors. These are
  the only nets that leave the board on unshielded wire, so they are the ESD and noise path into
  the MCU; the 100 Ω also caps the current if a probe firmware ever drives `RUN` high (§3) while
  the button is pressed.
- Solder both mounting tabs and tie them to GND — that is what holds an SH header down when the
  cable is pulled.
- Cable side: housings `SHR-02V-S-B` / `SHR-03V-S-B` with `SSH-003T-P0.2` contacts (LCSC
  **C160231**). Crimping SH by hand is awkward: buy ready-made pigtails, or run the panel loom in
  **JST PH 2.0 mm** (B2B-PH-K-S **C131337**, B3B-PH-K-S **C131339**) and keep SH for the board.
- If the panel LED is addressable and fed 5 V, its data threshold is 0.7 × VDD = 3.5 V and a
  3.3 V GPIO does not clear it. Either feed pin 1 from `+3V3` with a part rated for 3.3 V, or take
  pin 1 from the 5 V side through a series Schottky so its VDD sits near 4.4 V.

### 9.2 Power button
The board has no battery and USB-C is the only source, so "off" always means *plugged in, rails
down*. Two depths, and the baseline needs no new parts:

- **Soft off (baseline).** The four SY6280 ENs are already MCU-driven (§2.4): a press asks yasos to
  park the cards, then drops the three slot rails and the USB-A ports and puts the RP2350B in
  dormant; a press wakes it. Hold ~4 s to cut the rails without waiting for software. Standby is
  the buck plus a dormant MCU plus the hub — tens of mA, which a wall-powered machine can carry.
  The debug probe stays enumerated on the host, which is what the CI node needs.
- **Hard off (open, §12).** Make the TLV62569's EN the control node instead of tying it to VSYS
  (§2.3): 1 MΩ to GND so the default is off, the button pulls EN to VSYS through a diode, and the
  MCU holds it with a `PWR_HOLD` GPIO once it boots; releasing `PWR_HOLD` drops everything to the
  eFuse quiescent current. Costs ~5 passives and one of the two spare GPIOs (§11). The catch: §3
  powers the RP2040 probe from that same 3.3 V rail, so a hard off takes the probe off the host —
  no console, no remote power-on. Giving the probe its own small LDO off VSYS fixes that and lets
  the probe own the button, the LED and the buck EN, returning both budgeted GPIOs to the RP2350B.

`FP_PWR_BTN` and `FP_LED` are the two GPIOs already reserved in §11; a hard-off latch adds a third.

## 10. To do

**Board**
- [ ] Swap the remaining `ECS-120-8-36-CGN-TR` to **ABM8-272-T3** (`mspc_xtal`) on `rp2350_core`,
      `rp2040_core` and `south_bridge_osc`: the ECS part's 150 Ω max ESR is 3× the 50 Ω the
      RP2350's 1 kΩ XOUT damping resistor is tuned for, and Raspberry Pi tells you to test over
      temperature for any deviation. Keep the 14 pF load capacitors — 7 pF + 3 pF parasitic is the
      ABM8's 10 pF CL exactly, so this closes the old "14 pF → 10 pF" item rather than doing it.
- [ ] Fix the hub's upstream pair: `USB_D+` sits on RP2350 pin 66 (USB_DM) and `USB_D-` on pin 67
      (USB_DP), so DM reaches DPU+ through R16. The pair is genuinely crossed, not just mislabelled
      — move the labels onto the other pins.
- [ ] `usb_hub.kicad_sch` is instantiated twice, in `mainboard.kicad_sch` and in `rp2350_core`.
      The netlist carries two of every part and the top-level copy's upstream dangles. Delete it.
- [ ] `mspc_bus.kicad_sch`: J11 is missing `BUS_CARD1.RE#` and carries `INT#` on pins 9 *and* 12;
      J7 is missing `BUS_CARD2.D3` and carries `D2` on pins 5 *and* 6. Both connectors also cluster
      their only two grounds at the far end — tie the spare pins (J19 3/4/15/19/20, J11 3/4/5/6/11/
      15/16, J7 3/4/11/15/16) to GND so the rule above is met, and put one next to CLK.
- [ ] APS6404L symbol; W25Q128JVSIQ in the BOM.
- [ ] RP2350 core regulator inductor (L3) per the RP2350 hardware design guide.
- [ ] Test points: SWD, UART, supply rails.
- [ ] Optional RTC (PCF8563) — only if GPIOs are freed.
- [ ] Remove `+3V3_BUS` from the slot pinout; J4 pin 28 becomes a spare connector pin.
- [ ] Front panel (§9): place J_LED, J_RST and J_PWR_BTN (`mspc_connectors`) at the board edge
      with their 100 Ω + 100 nF, mounting tabs on GND; the reset header parallels the RP2350B
      `RUN` line the probe already drives.

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
- [ ] Power: read CC (ADC), both eFuse FAULT lines and the "J_PWR present" divider; drive the
      SY6280 ENs; keep an unpowered slot's link pins high-Z. The TPS25200 has no analog current
      monitor, so the per-input current watch is gone.
- [ ] Front panel (§9.2): power button on `FP_PWR_BTN` — short press parks the cards and drops the
      slot and USB rails through the SY6280 ENs, then dormant; ~4 s forces the rails down. Drive
      `FP_LED` for the panel LED.
- [ ] debugprobe: request 500 mA in the configuration descriptor.
- [ ] PCM5100A I2S driver (test on the Pico Plus 2 + VGA Demo Base).
- [ ] Card link protocol (PIO + DMA), enumeration by ID, INT/DET.
- [ ] TinyUSB ≥ 0.21.0 for enumeration through the hub (pico-sdk pins 0.18.0).

## 11. GPIO budget (RP2350B) — verify in the pin editor

| function | GPIO |
|---|---|
| VGA link (CLK + D0–3 + INT) | 6 |
| FPGA link (CLK + D0–7 + SEL + INT), D0–7 on GPIO12–19 | 11 |
| network link, SPI (CLK + MOSI + MISO + CS + INT) | 5 |
| slot RE# (one per slot) | 3 |
| PSRAM CS1 (GPIO0 or 47) | 1 |
| SD card (CLK, CMD, D0–3) | 6 |
| console UART | 2 |
| I2S (BCK, LRCK, DIN) + XSMT | 4 |
| CC, CC1/CC2 joined by 2 × 100 kΩ per connector (ADC) | 2 |
| eFuse FAULT ×2 | 2 |
| SY6280 EN (slots shared + USB) | 2 |
| front panel: power button + LED data (§9) | 2 |
| **total** | **46** |
| spare | 2 |

Spares: 2, once RE# is per slot (3 pins) instead of one shared RESET. An RTC on I2C (2) fits, or
SD card detect (1) plus one held back for the link's frame line; REQ/GNT (2) would take both.
Dropping `+3V3_BUS` does not help here — it frees a pin on each slot connector, not an RP2350B GPIO.
No "charger present" pin is needed: the CC readings (§2.3) already say which connector has a source.

## 12. Open decisions

**Decided 2026-09-12** — 3.3 V on the slots (`+3V3_BUS`) is dropped; each card regulates from its
switched 5 V. The FPGA card was the only one still relying on it (§8.1). The trigger was measured
noise on the shared rails with a card in the slot.

- 3.3 V buck package: keep the TLV62569DRL (in stock here, scarce) or move to SOT-23-5
  (TLV62569DBV, or the 1 A TLV62568).
- Which features get the 2 spare GPIOs.
- Power-off depth (§9.2): soft off through the SY6280 ENs, which needs no new parts and keeps
  the probe enumerated, or a hard-off latch on the buck EN, which needs ~5 passives and a
  `PWR_HOLD` GPIO and takes the probe off the host unless the probe gets its own LDO.
- The TPS25200 auto-retries instead of latching off after a fault (§2.7). Accept the retry loop on
  the CI node, or add a latch (firmware holding the eFuse EN low once FAULT is seen, which needs
  the 3.3 V rail to survive the fault).

## 13. KiCad libraries
- `mspc_power` (`libs/symbols/mspc_power.kicad_sym`): TPS25200, SY6280AAC, PMEG3020EP (D1, on
  `footprints:D_SOD-128`), and the now-unused TPS259470LRPWR.
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
- KiCad standard: TLV62569DRL (SOT-563),
  PCM5100A, RP2040, W25Q16JVSS, USBLC6-2SC6,
  USB_C_Receptacle_USB2.0_16P, TVS2200DRV; the SY6280 uses SOT-23-5_HandSoldering.
- Hand soldering: the eFuse (WSON-6, 2 × 2 mm, 0.65 mm pitch, thermal pad) and the TLV62569DRL
  (SOT-563) need solder paste and hot air.
