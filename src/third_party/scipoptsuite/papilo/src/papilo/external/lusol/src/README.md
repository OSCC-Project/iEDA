## LUSOL fortran code

This directory contains LUSOL fortran 77 and fortran 90 code.

## Contents

* `clusol.c`: generated interface code
* `clusol.h`: generated interface header
* `lusol6b.f`: contains lu6mul
* `lusol7b.f`: contains lusol helper functions
* `lusol8b.f`: contains lusol update functions (not lu8rpc)
* `lusol.f90`: fortran 90 code for factorization, solve, and replace column
* `lusol_precision.f90`: fortran 90 module to specify precision
* `lusol.txt`: some LUSOL documentation and history
* `lusol_util.f`: fortran 77 code for factorization, solve, and replace column
* `symbols.osx`: list of symbols to export in osx dynamic library
