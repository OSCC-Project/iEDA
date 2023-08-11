## Interface generation files

The files in this directory are used to generate the `clusol.h` and `clusol.c`
interface code.

## Contents

* `interface_files.org`: list of interface functions to include
* `interface.py`: python3 script to generate `clusol` interface code
* `lu1fac.org`: specification for LUSOL factorization function
* `lu6mul.org`: specification for LUSOL multipy function
* `lu6sol.org`: specification for LUSOL solve function
* `lu8adc.org`: specification for LUSOL add column function
* `lu8adr.org`: specification for LUSOL add row function
* `lu8dlc.org`: specification for LUSOL delete column function
* `lu8dlr.org`: specification for LUSOL delete row function
* `lu8mod.org`: specification for LUSOL rank 1 modification function
* `lu8rpc.org`: specification for LUSOL replace column function
* `lu8rpr.org`: specification for LUSOL replace row function

## Note

The interface generation script is called by `make` during the build process.
