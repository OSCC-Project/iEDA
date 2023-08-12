# LUSOL-Matlab interface notes

This document contains some notes on building and installing the LUSOL-Matlab
interface.

## 2016-01-26 osx build notes

Notes:

* Need to build `libclusol.dylib` statically linked against
  * `libgfortran`
  * `libquadmath`
  * `libgcc`
* Dynamically link against `libmwblas` provided by Matlab
* `libclusol` is compiled to use 64-bit integers to match `libmwblas`
* `lusol_build.m` works, but emits a nasty error message (see below)
* `lusol_test.m` works

Build dynamic library:
```
$ make
clang  -fPIC -c src/clusol.c -o src/clusol.o
gfortran -fPIC -Jsrc -O3 -c src/lusol_precision.f90 -o src/lusol_precision.o
gfortran -fPIC -Jsrc -O3 -c src/lusol.f90 -o src/lusol.o
gfortran -fPIC -fdefault-integer-8 -O3 -c src/lusol_util.f -o src/lusol_util.o
gfortran -fPIC -fdefault-integer-8 -O3 -c src/lusol6b.f -o src/lusol6b.o
gfortran -fPIC -fdefault-integer-8 -O3 -c src/lusol7b.f -o src/lusol7b.o
gfortran -fPIC -fdefault-integer-8 -O3 -c src/lusol8b.f -o src/lusol8b.o
clang -dynamiclib -Wl,-twolevel_namespace -Wl,-no_compact_unwind -undefined error -bind_at_load -Wl,-exported_symbols_list,src/symbols.osx /usr/local/opt/gcc/lib/gcc/5/libgfortran.a /usr/local/opt/gcc/lib/gcc/5/libquadmath.a /usr/local/Cellar/gcc/5.3.0/lib/gcc/5/gcc/x86_64-apple-darwin15.0.0/5.3.0/libgcc.a -L/Applications/MATLAB_R2015b.app/bin/maci64 -lmwblas src/clusol.o src/lusol_precision.o src/lusol.o src/lusol_util.o src/lusol6b.o src/lusol7b.o src/lusol8b.o -o src/libclusol.dylib
$
```

Build Matlab prototype file and thunk:
```
$ make matlab
cp src/libclusol.dylib src/clusol.h ./matlab/
matlab -nojvm -nodisplay -r "cd matlab; lusol_build; exit"

                                < M A T L A B (R) >
                      Copyright 1984-2015 The MathWorks, Inc.
                       R2015b (8.6.0.267246) 64-bit (maci64)
                                  August 20, 2015

 
For online documentation, see http://www.mathworks.com/support
For product information, visit www.mathworks.com.
 

	Academic License

Warning: loadlibrary returned warning messages:

Type 'char__signed' was not found.  Defaulting to type error.

Found on line 17 of input from line 30 of file
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/sys/_types/_int8_t.h

Failed to parse type 'union { char *__mbstate8; long long _mbstateL ; } __mbstate_t'
original input 'union { char __mbstate8 [ 128 ]; long long _mbstateL ; } __mbstate_t'
Found on line 112 of input from line 79 of file
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/i386/_types.h

Type '__mbstate_t' was not found.  Defaulting to type error.

Found on line 114 of input from line 81 of file
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/i386/_types.h

 
> In lusol_build (line 17) 
$
```

Test:
```
$ make matlab_test
matlab -nojvm -nodisplay -r "cd matlab; lusol_test; exit"

                                < M A T L A B (R) >
                      Copyright 1984-2015 The MathWorks, Inc.
                       R2015b (8.6.0.267246) 64-bit (maci64)
                                  August 20, 2015

 
For online documentation, see http://www.mathworks.com/support
For product information, visit www.mathworks.com.
 

	Academic License

test_addcol_lhr02: passed
test_addrow_lhr02: passed
test_delcol_lhr02: passed
test_delrow_lhr02: passed
test_factorize_lhr02: passed
test_mulA_lhr02: passed
test_mulAt_lhr02: passed
test_r1mod_lhr02: passed
test_repcol_lhr02: passed
test_reprow_lhr02: passed
test_solveA_lhr02: passed
test_solveAt_lhr02: passed
$ 
```

## Mac OS X dynamic library references

Apple Developer Documentation: <https://developer.apple.com/library/mac/navigation/>

Specific documents:

* [Dynamic Library Programming Topics][dlpt]
* [Porting UNIX/Linux Applications to OS X][pula]

[dlpt]: https://developer.apple.com/library/mac/documentation/DeveloperTools/Conceptual/DynamicLibraries/000-Introduction/Introduction.html#//apple_ref/doc/uid/TP40001908-SW1
[pula]: https://developer.apple.com/library/mac/documentation/Porting/Conceptual/PortingUnix/intro/intro.html#//apple_ref/doc/uid/TP40002847-TPXREF101

## Linux shared object references

### **Linux Programming Interface**:

* <http://proquest.safaribooksonline.com/9781593272203?uicode=stanford>
* See chapters on shared libraries (41 and 42)

### TLDP: Program Library HOWTO

<http://tldp.org/HOWTO/Program-Library-HOWTO/index.html>

### How to write shared libraries by Drepper

<https://www.akkadia.org/drepper/dsohowto.pdf>

