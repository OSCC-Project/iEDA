# SoPlex: Sequential object-oriented simPlex

SoPlex is an optimization package for solving linear programming problems (LPs)
based on an advanced implementation of the primal and dual revised simplex
algorithm. It provides special support for the exact solution of LPs with
rational input data. It can be used as a standalone solver reading MPS or LP
format files via a command line interface as well as embedded into other
programs via a C++ class library. The main features of SoPlex are:

- presolving, scaling, exploitation of sparsity, hot-starting from any regular basis,
- column- and row-oriented form of the simplex algorithm,
- an object-oriented software design written in C++,
- a compile-time option to use 80bit extended ("quad") precision for numerically difficult LPs,
- an LP iterative refinement procedure to compute high-precision solution, and
- routines for an exact rational LU factorization and continued fraction approximations in order to compute exact solutions.

SoPlex has been used in numerous research and industry projects and is the standard LP solver linked to the mixed-integer
nonlinear programming and constraint integer programming solver SCIP.

The original instance of this repository is hosted at
[git.zib.de](https://git.zib.de) and a read-only
mirror is available at
[github.com/scipopt/soplex](https://github.com/scipopt/soplex).

SoPlex is part of the SCIP Optimization Suite, online at [scipopt.org](https://scipopt.org).

Further information and resources are available through the official SoPlex website at
[soplex.zib.de](https://soplex.zib.de) including

- [online documentation](https://soplex.zib.de/doc) of the code
- with information how to get started and
- how to cite SoPlex when you use it in scientific publications.

For installation instructions have a look [here](INSTALL.md) or in the [online
documentation](https://soplex.zib.de/doc/html/INSTALL.php).
