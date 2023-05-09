# Mirror of the Si2 LEF/DEF parser (v5.8)

This repository mirrors the LEF/DEF source code from Si2 (v5.8, p007). 
   * The `lef/` directory contains the umodified contents of `lef_5.8-p007.tar.Z`
   * The `def/` directory contains the umodified contents of `def_5.8-p007.tar.Z`

In addition, there is a top-level Makefile that assumes that `$ACT_HOME` is set.
Running `make` followed by `make install` installs the LEF/DEF libraries and header
files to the appropriate directories used by the rest of the ACT tools. 
A few `.gitignore` files are added as well.

