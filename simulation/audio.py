from distutils.command import bdist_dumb
import math

import wave
import struct
from distro import uname_info
import numpy as np

import matplotlib.pyplot as plt


import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *

import soundfile as sf


from OperationalAmplifier import BasicOperationalAmplifier

import math
import numpy as np
import matplotlib.pyplot as plt


import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *



class WavSource(NgSpiceShared):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)

        # self._data, self._samplerate = sf.read(filename)
        fs = 8000
        f = 400 
        t = 2 
        self._data = np.arange(t * fs)
        self._data = np.sin(2 * np.pi * f * self._data)
        self._samplerate = 8000
        
        self._position = 0
        self._prev_time = 0
        self._delta = 1 / self._samplerate
        print("Loaded ", filename, " sample rate: ", self._samplerate)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        if time - self._prev_time > self._delta:
            self._position += 1 
            self._prev_time = time 
        
        if self._position < len(self._data):
            voltage[0] = self.scale(self._data[self._position], 0, 3.3)
        else:
            voltage[0] = 0 
        return 0 

    def get_isrc_data(self, current, time, node, ngspice_id):
        current[0] = 1.
        return 0

    def scale(self, voltage, level_min, level_max):
        return (voltage + 1) * level_max/2

class PwmAudioSource(NgSpiceShared):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)

        self._data, self._samplerate = sf.read(filename)
        self._position = 0
        self._prev_time = 0
        self._delta = 1 / self._samplerate / 4
        self._delta_time = 0
        self._repeat = 0
        self._filltime = 0
        print("Loaded ", filename, " sample rate: ", self._samplerate)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        if self._position >= len(self._data):
            voltage[0] = 0 
            return 0 

        # next byte
        if time - self._prev_time > self._delta * 4:
            self._position += 1 
            self._prev_time = time 
            self._delta_time = time
            self._filltime = 1/self._samplerate/4 * self.scale(self._data[self._position], 0, 1)
        
        if time - self._delta_time > self._delta:
            self._delta_time = time 

        # process current 
        if time - self._delta_time < self._filltime:
            voltage[0] = 3.3 
            return 0 
        
        voltage[0] = 0 
        return 0 

    def get_isrc_data(self, current, time, node, ngspice_id):
        current[0] = 1
        return 0

    def scale(self, voltage, level_min, level_max):
        return (voltage + 1) * level_max/2


circuit = Circuit('Low-Pass RC Filter')
# ngspice_shared = PwmAudioSource("test.wav")

# circuit.V('input', 'input', circuit.gnd, 'dc 0 external')
# R1 = circuit.R(1, 'input', 'o1', 48@u_Ω)
# C1 = circuit.C(1, 'o1', circuit.gnd, 200@u_nF)
# R2 = circuit.R(2, 'o1', 'o2', 68@u_Ω)
# C2 = circuit.C(2, 'o2', circuit.gnd, 100@u_nF)
# R3 = circuit.R(3, 'o2', 'output', 63@u_Ω)
# C3 = circuit.C(3, 'output', circuit.gnd, 100@u_nF)

# circuit.V('input', 'voltage_out', circuit.gnd, 'dc 0 external')

# ngspice_shared = PwmAudioSource("test.wav")
circuit.SinusoidalVoltageSource('input', 'voltage_out', circuit.gnd, amplitude=1.7@u_V, dc_offset=1.7@u_V, offset=1.7@u_V, frequency=800@u_Hz)
# circuit.V('input', 'voltage_out', circuit.gnd, amplitude=1.7@u_V, dc_offset=1.7@u_V, offset=1.7@u_V, frequency=8@u_Hz)
circuit.subcircuit(BasicOperationalAmplifier())
circuit.X('op', 'BasicOperationalAmplifier', "r2_out", "low_pass_filter_out", 'low_pass_filter_out')

circuit.R(1, "voltage_out", "r1_out", 30@u_kΩ)
# circuit.C(1, "r1_out", circuit.gnd, 680@u_pF)
circuit.R(2, "r1_out", "r2_out", 12@u_kΩ)
circuit.C(1, "low_pass_filter_out", "r1_out", 1@u_nF)
circuit.C(2, "r2_out", circuit.gnd, 220@u_pF)

circuit.R(3, "low_pass_filter_out", "low_stage_out", 100@u_Ω)
circuit.C(3, "low_stage_out", circuit.gnd, 100@u_nF)

# voltage divider
circuit.R(4, "low_stage_out", circuit.gnd, 220@u_Ω)

# high pass filter > 1 HZ 
circuit.C(4, "low_stage_out", "output_load", 47@u_uF)
circuit.R(5, "output_load", circuit.gnd, 1.8@u_kΩ)


# load 
circuit.R(6, "output_load", circuit.gnd, 1200@u_Ω)


# circuit.R(8, "output_f", "output_load", 1000@u_Ω)

# voltage divider 
# circuit.R(4, "low_stage_out", circuit.gnd, 340@u_Ω)
# # # load 
# circuit.R(100, "output_load", circuit.gnd, 600@u_Ω)


# circuit.C(4, "input", "low_pass_out", 47@u_uF)
# circuit.R(4, "input", circuit.gnd, 100@u_Ω)




# circuit.R(4, "input", "r4_out", 68@u_Ω)
# circuit.C(4, "r4_out", circuit.gnd, 100@u_nF)




# R2 = circuit.R(2, 'output', "RL", 6@u_Ω)
# L1 = circuit.L(1, "RL", circuit.gnd, 0.43@u_mH)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# simulator = circuit.simulator(temperature=25, nominal_temperature=25,
#                               simulator='ngspice-shared', ngspice_shared=ngspice_shared)
# analysis = simulator.transient(step_time=1/100000000, end_time=0.008)


# circuit.V('input', 'input', circuit.gnd, 'dc 0 external')
# circuit.R(1, 'input', 'output', 220@u_Ω)
# circuit.C(2, 'output', circuit.gnd, 100@u_nF)
# # circuit.R(3, 'r1_o', circuit.gnd, 100@u_Ω)
# # circuit.C(4, 'r1_o', 'output', 47@u_uF)
# circuit.R(5, 'output', circuit.gnd, 1.8@u_kΩ)

# # speaker 
# circuit.R(6, "output", "r6_o", 6@u_Ω)
# circuit.L(1, "r6_o", "l1_o", 0.43@u_mH)
# circuit.C(7, "l1_o", circuit.gnd, 240@u_uF)
# circuit.L(8, "l1_o", circuit.gnd, 126@u_mH)
# circuit.R(9, "l1_o", circuit.gnd, 137@u_Ω)
# circuit.C(10, "l1_o", circuit.gnd, 12@u_uF)



# ngspice_shared = PwmAudioSource("test.wav")
# simulator = circuit.simulator(temperature=25, nominal_temperature=25,
#                               simulator='ngspice-shared', ngspice_shared=ngspice_shared)
                              
analysis = simulator.transient(step_time=1/1000000, end_time=0.1)


figure1 = plt.figure(1, (20, 10))
plt.title('Voltage Divider')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plot(analysis.voltage_out)
plot(analysis.output_load)
plt.legend(('voltage_out', 'output_load'), loc=(.05,.1))

plt.tight_layout()

analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=5@u_MHz, number_of_points=500,  variation='dec')
figure = plt.figure(2, (20, 10))
plt.title("Bode Diagram of a Low-Pass RC Filter")
axes = (plt.subplot(211), plt.subplot(212))
bode_diagram(axes=axes,
             frequency=analysis.frequency,
             gain=20*np.log10(np.absolute(analysis.output_load)),
             phase=np.angle(analysis.output_load, deg=False),
             marker='.',
             color='blue',
             linestyle='-',
         )

plt.tight_layout()
plt.show()
