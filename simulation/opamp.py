####################################################################################################

import numpy as np

import matplotlib.pyplot as plt

####################################################################################################

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

####################################################################################################

from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from OperationalAmplifier import BasicOperationalAmplifier

#f# literal_include('OperationalAmplifier.py')

####################################################################################################

circuit = Circuit('Operational Amplifier')

# AC 1 PWL(0US 0V  0.01US 1V)
circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=1@u_V)
circuit.subcircuit(BasicOperationalAmplifier())
circuit.R(1, "in", "r1_out", 12@u_kΩ)
circuit.R(2, "r1_out", "r2_out", 110@u_kΩ)
circuit.C(1, "r1_out", circuit.gnd, 680@u_pF)
circuit.R(3, "r2_out", "r3_out", 36@u_kΩ)
circuit.C(2, "op_out", "r2_out", 330@u_pF)
circuit.C(3, "r3_out", circuit.gnd, 330@u_pF)

circuit.R(4, "in", "r4_out", 68@u_Ω)
circuit.C(4, "r4_out", circuit.gnd, 100@u_nF)



circuit.X('op', 'BasicOperationalAmplifier', "r3_out", "op_out", 'op_out')

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_MHz, number_of_points=5,  variation='dec')

figure, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

plt.title("Bode Diagram of an Operational Amplifier")
bode_diagram(axes=(ax1, ax2),
             frequency=analysis.frequency,
             gain=20*np.log10(np.absolute(analysis.op_out)),
             phase=np.angle(analysis.op_out, deg=False),
             marker='.',
             color='blue',
             linestyle='-',
            )
bode_diagram(axes=(ax1, ax2),
             frequency=analysis.frequency,
             gain=20*np.log10(np.absolute(analysis.r4_out)),
             phase=np.angle(analysis.r4_out, deg=False),
             marker='.',
             color='blue',
             linestyle='-',
            )
plt.tight_layout()
plt.show()

#f# save_figure('figure', 'operational-amplifier.png')
