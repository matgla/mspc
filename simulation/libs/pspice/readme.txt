================================================================================
                   NXP SEMICONDUCTORS HCMOS SPICE MODELS
================================================================================

This is version 1.2 for the NXP Semiconductors HCMOS SPICE models.  These 
models are in Berkeley SPICE format. To run simulations, read the instructions 
provided in the top level program, hct.cir.  Also refer to the SPICE data book 
for information on individual part types and package modeling.  The following 
files should be included to perform simulations on the HCMOS family: hct.cir, 
hc_tnomi.cir,hc_tfast.cir, hc_tslow.cir and the package libraries: dhvqfn.s,
dip.s, so.s, ssop.s and tssop.s.

The following shows the revision history for the HCMOS models:

Verison 1.2
-----------
Date 30/03/2011 RRV
Rewritten netlist to add package options.
Available packages: DIP, SO, SSOP, TSSOP and DHVQFN

Version 1.12
------------
Date 27/10/2010 RRV
Added HC(T)240/253

Version 1.11
------------
Date 06/07/2010 RRV
Added HC(T)02/08/86

Version 1.10
------------
Date 10/12/2009 RRV
Removed HCT05 (not an active part type)

Version 1.09
------------
Date 09/23/2009 RRV
.SUBSCKT SWITCH3 renamed to SWITCH3N in hc_tnomi.cir

Version 1.08
------------
Date 09/08/2009 RRV
Added models for HC(T)4020/4060

Version 1.07
------------
Date 03/27/2009 RRV
Added models for HC(T)05

Version 1.06
------------
Date 02/20/2006 RRV
Added models for HC(T)366 and HC(T)368

Version 1.05
------------
Date 10/27/2005 RRV
Updated HC/T04 and HCU04 models.
Commented out R1 for test load. Datasheet specifies only Cload.

Version 1.04 
------------
Date 09/22/2003 RM
Added an inverter between the enable input module and the enable input of the 
analog switch in hc4066 and hct4066, in order to realize an active high enable.

Version 1.03 
------------
Date 01/14/2003 RM
Corrected the level converter subcircuit LLCN/LLCF/LLCS, used in 
SWI1, SWI3, SWI1T, SWI3T. The gate connection of MP1 is now connected
to net 4 (it was connected to net 2, which was not correct).

Version 1.02
------------
Date 10/3/95
Deleted UNBUFF circuit in subcircuit files for HCU04 model.  All subcircuit
files were changed.

Version 1.01
------------
Added HCU04 model.  All files were changed.

Version 1.0
-----------
Original release.
