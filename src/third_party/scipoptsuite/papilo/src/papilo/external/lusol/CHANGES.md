## 2016-01-26

* `makefile` now works on OSX 10.11 with `gfortran` version 5.3 installed with
  homebrew
* Updated LUSOL code to fix zero-column bug

## 2014-05-18

* Updated build code to work on Mac OS X 10.9 & Matlab 2014a
* Removed compiler selection from `matlab/lusol_build.m`
    * `mex` is now required to be setup to specify C compiler
