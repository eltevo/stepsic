         _                 _
        | |               (_)
     ___| |_ ___ _ __  ___ _  ___
    / __| __/ _ \ '_ \/ __| |/ __|
    \__ \ ||  __/ |_) \__ \ | (__
    |___/\__\___| .__/|___/_|\___|
                | |
                |_|
stepsic 2.0.0
    An IC generator python script for
    STEreographically Projected cosmological Simulations

Copyright (C) 2018-2025 Gábor Rácz, Balázs Pál
	Jet Propulsion Laboratory, California Institute of Technology | 4800 Oak Grove Drive, Pasadena, CA, 91109, USA
	Department of Physics of Complex Systems, Eotvos Lorand University | Pf. 32, H-1518 Budapest, Hungary
	Department of Physics & Astronomy, Johns Hopkins University | 3400 N. Charles Street, Baltimore, MD 21218
    Heavy-ion Physics Research Group, HUN-REN Wigner RCP | Budapest, Hungary
	Department of Physics, University of Helsinki

+-------------------------------------------------------------------------------+
| stepsic comes with ABSOLUTELY NO WARRANTY.                                    |
| This is free software, and you are welcome to redistribute it                 |
| under certain conditions. See the LICENSE file for details.                   |
+-------------------------------------------------------------------------------+

This is an IC generator script for StePS simulations.
-written in python3
-leverages the Zel'dovich approximation and 2LPT to generate initial conditions
-reads an input glass, and perturbates its particles
-the output can be in ASCII or in HDF5 format.

*********************************************************************************************

Dependencies:
	Python:
	-future
	-glio (https://github.com/spthm/glio)
	-h5py
	-toml
	-tabulate
	-astropy
  	-camb
  	-colossus (https://bdiemer.bitbucket.io/colossus/)

Running the script:
```
python stepsic.py <input config.toml file>
```

The template `config.toml` file:

<explanation of variables>

*********************************************************************************************

Acknowledgement
  The development of this code has been supported by Department of Physics of Complex Systems, ELTE.
  GR would like to thank the Department of Physics & Astronomy, JHU for supporting this work.
  GR acknowledges sponsorship of a NASA Postdoctoral Program Fellowship. GR was supported by JPL, which is run under contract by California Institute of Technology for NASA.
  The developer acknowledges support from the National Science Foundation (NSF) award 1616974.
