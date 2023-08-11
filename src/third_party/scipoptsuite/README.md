```
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*                  This file is part of the program and library             *#
#*         SCIP --- Solving Constraint Integer Programs                      *#
#*                                                                           *#
#*  Copyright 2002-2022 Zuse Institute Berlin                                *#
#*                                                                           *#
#*  Licensed under the Apache License, Version 2.0 (the "License");          *#
#*  you may not use this file except in compliance with the License.         *#
#*  You may obtain a copy of the License at                                  *#
#*                                                                           *#
#*      http://www.apache.org/licenses/LICENSE-2.0                           *#
#*                                                                           *#
#*  Unless required by applicable law or agreed to in writing, software      *#
#*  distributed under the License is distributed on an "AS IS" BASIS,        *#
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *#
#*  See the License for the specific language governing permissions and      *#
#*  limitations under the License.                                           *#
#*                                                                           *#
#*  You should have received a copy of the Apache-2.0 license                *#
#*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         *#
#*                                                                           *#
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
```

The SCIP Optimization Suite consists of the following six software tools:
  1. ZIMPL  - the Zuse Institute Mathematical Programming language
  2. SoPlex - the Sequential Object Oriented Simplex
  3. SCIP   - Solving Constraint Integer Programs
  4. GCG    - Generic Column Generation
  5. UG     - Ubiquity Generator Framework
  6. PaPILO - Parallel Presolve for Integer and Linear Optimization

We provide two different systems to compile the code: the traditional Makefile
system and the new CMake build system.  Be aware that generated libraries and
binaries of both systems might be different and incompatible.  For further
details please refer to the INSTALL file of SCIP and the online documentation.


Content
=======

1. CMake

2. Makefiles
  - Creating a SCIP binary and the individual libraries
  - Creating a single library containing SCIP, SoPlex, and ZIMPL
  - Creating GCG and UG


CMake
=====

Ensure that you're using an up-to-date [CMake installation](https://cmake.org/).
CMake will automatically configure the code according to your environment and
the available third-party tools, like GMP, etc.
Create a new directory for the build, for instance inside the `scipoptsuite` directory.

    mkdir build
    cd build
    cmake ..
    make

If you want a maximal installation with autodetection of additional dependencies, please use the following cmake call instead:

    cmake .. -DAUTOBUILD=ON

For further information please refer to the online documentation of [SCIP](https://scipopt.org)
or the `INSTALL` in the SCIP subdirectory.


Makefiles
=========

##  Creating a SCIP binary and the individual libraries

SCIP uses the libraries of ZIMPL and SoPlex to be able to read ZIMPL models
and solve the subproblem LP relaxations with SoPlex.

In order to compile the whole bundle, just enter

    make

within the SCIP Optimization Suite main directory. If you are using a
gcc compiler with version less 4.2 you have to compile with the following
additional flags

    make LPSOPT=opt-gccold ZIMPLOPT=opt-gccold OPT=opt-gccold

If all goes well, you should get a final message

** Build complete.
** Find your binary in "<path>/scipoptsuite-<version>/scip-<version>/bin".
** Enter "make test" to solve a number of easy instances in order to verify that SCIP runs correctly.


If this is not the case, there are most probably some libraries missing on
your machine or they are not in the right version. In its default build, the
SCIP Optimization Suite needs the following external libraries:

- the Z Compression Library (ZLIB: `libz.a` or `libz.so` on Unix systems)
  Lets you read in `.gz` compressed data files.
- the GNU Multi Precision Library (GMP: `libgmp.a` or `libgmp.so` on Unix systems)
  Allows ZIMPL to perform calculations in exact arithmetic.
- the Readline Library (READLINE: `libreadline.a` or `libreadline.so` on Unix systems)
  Enables cursor keys and file name completion in the SCIP shell.

You can disable those packets using the following `make` arguments:
- `GMP=false`       (disables GMP support)
- `ZLIB=false`      (disables ZLIB support)
- `READLINE=false`  (disables READLINE support)

You can also disable ZIMPL by specifying `ZIMPL=false` as a `make` argument.
Note, however, that this disables the ZIMPL file reader in SCIP and you can
no longer read in ZIMPL models (the input files with a `.zpl` extension).

Since ZIMPL requires GMP, it is automatically disabled, if GMP is disabled.

Since the GMP is not installed on every machine, and the READLINE library is
sometimes only existing in an old version, these are the two most frequent
candidates for build problems. The following should work on most machines:

    make GMP=false READLINE=false

Note that on some MAC systems, GMP is installed under `/sw/include` and `/sw/lib`.
If these are not contained in the library and include paths, you have to add
them explicitly.

If this still does not work, you should try the following, which is the most
compatible method to build the SCIP Optimization Suite:

    make ZLIB=false GMP=false READLINE=false LPSOPT=opt-gccold OPT=opt-gccold

Note, however, that in this case, you cannot
- read compressed (`.gz`) input files from SCIP and SoPlex
- read ZIMPL models from SCIP
- use the counting of solutions feature exactly for large numbers of solutions
- use the ZIMPL binary to transform ZIMPL (`.zpl`) models into MIP instances
  of `.lp` or `.mps` type
- use the cursor keys and command line completion in SCIP


For more information how to install the components of the SCIP
Optimization Suite and for details concerning special architectures and
operating systems, see the INSTALL file of SCIP and the documentation
of each component.

## Creating a single library containing SCIP, SoPlex, and ZIMPL

In case you need a single library which contains SCIP, SoPlex, and ZIMPL
you can use the command:

    make scipoptlib

This will create a single library containing SCIP, SoPlex, and ZIMPL. It is
placed in the `lib` directory. This library is statically linked. If you
need a shared library use the command:

    make scipoptlib SHARED=true

## Creating GCG and UG

In case for the both SCIP extensions GCG and UG, you can easily compile
these using the commands

    make gcg
    make ug
