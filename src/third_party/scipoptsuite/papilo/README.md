# PaPILO &mdash; Parallel Presolve for Integer and Linear Optimization

PaPILO, a C++14-based software package, provides parallel presolve routines for (mixed integer) linear programming problems. The routines
are implemented using templates which allows switching to higher precision or rational arithmetic using the boost multiprecision package.

Additionally to the distribution here under the LGPLv3, PaPILO is also distributed as part of the
SCIP Optimization Suite which is available under https://scipopt.org/.

PaPILO can be used as a header-based library and also provides an executable.
Using the executable it is possible to presolve and postsolve MILP instances based on files.
Additionally, PaPILO can be linked to SCIP, SoPlex, Gurobi, Glop, and HiGHS (https://github.com/ERGO-Code/HiGHS) solvers and act as a frontend. In this setting PaPILO passes the presolved problem to those solvers and applies the postsolve step to the optimal solution.
When PaPILO is compiled as part of the SCIP Optimization Suite linking of SoPlex and SCIP solvers is performed automatically.
Further, a Julia wrapper is available [here](https://github.com/scipopt/PaPILO.jl).


*Note:* The original instance of this repository is hosted at  [git.zib.de](https://git.zib.de) and a read-only mirror is
available at [github.com/scipopt/papilo](https://github.com/scipopt/papilo).

# Dependencies

External dependencies that need to be installed by the user are the Intel TBB >= 2020, or TBB from oneAPI runtime library and boost >= 1.65 headers.
The executable additionally requires some of the boost runtime libraries that are not required when PaPILO is used as
a library.
Under the folder external/ there are additional packages that are directly included within PaPILO and have a
liberal open-source license.

If TBB is not found, then PaPILO tries to compile a static version. However this may fail on some systems currently and it is strongly recommended to install an Intel TBB runtime library.

# Building

Building PaPILO works with the standard cmake workflow:
(_we recommend running the make command without specifying the number of jobs
that can run simultaneously (no -j n), since this may cause large memory consumption and freeze of the machine_)

```
mkdir build
cd build
cmake ..
make
```

Building PaPILO with SCIP and SOPLEX works also with the standard cmake workflow:
```
mkdir build
cd build
cmake -DSCIP_DIR=PATH_TO_SCIP_BUILD_DIR ..
make
```
After this, your papilo binary should be found in the `bin` folder.
To install into your system, run `sudo make install`.

To install papilo into a folder, add `-DCMAKE_INSTALL_PREFIX=/path/to/install/dir/` to the cmake call and run `make install` after the build.

If you use a relative path to SCIP, then the reference point is the location of the `CMakeLists.txt`.
If you want to build PaPILO with a provided Boost version please add one of these option to the cmake command:
```
-DBOOST_ROOT=../boost_1_66_0
-DBOOST_INCLUDEDIR=../boost_1_66_0/include
```

Solvers that are found in the system are automatically linked to the executable.
Additionally one can specify the locations of solvers, e.g. with `-DSCIP_DIR=<location of scip-config.cmake>`, to allow
PaPILO to find them in non-standard locations.

# Usage of the binary

The PaPILO binary provides a list of all available functionality when the help flag `-h` or `--help` is specified.
The binary provides the three subcommands `solve`, `presolve`, and `postsolve`. If no solvers are linked the `solve` subcommand will fail and print an error message.

Next we provide a small example of how the binary can be used to apply presolving and postsolving based on files.

Assuming a problem instance is stored in the file `problem.mps` the following call will apply presolving with standard settings and write the reduced problem to `reduced.mps` and all information that is needed for postsolve to the binary archive `reduced.postsolve`.
```
papilo presolve -f problem.mps -r reduced.mps -v reduced.postsolve
```

Now we can use the reduced problem `reduced.mps` to obtain a solution
using any solver or from any other source to the file `reduced.sol`.
The format of the solution should be one value per line given like this:
```
<variable name>        <value>
```
This is compatible with the solutions given on the MIPLIB 2017 homepage https://miplib.zib.de and with the solutions written by the SCIP solver.
Variable names that are not found in the reduced problem are ignored.

The command for applying the postsolve step to the solution `reduced.sol` is then
```
papilo postsolve -v reduced.postsolve -u reduced.sol -l problem.sol
```
Giving the parameter `-l problem.sol` is optional and will store the solution transformed to the original space under `problem.sol`.
The output of papilo contains some information about violations and objective value of the solution.

If PaPILO was linked to a suitable solver, then the above can also be achieved by using the `solve` subcommand like this:
```
papilo solve -f problem.mps -l problem.sol
```
This will presolve the problem, pass the reduced problem to a solver, and subsequently transform back the optimal solution returned by the solver and write it to problem.sol.

# Using PaPILO as a library

PaPILO provides a templated C++ interface that allows to specify the type used for numerical computations. During configuration time PaPILO scans the system and provides the fastest available numeric types for quadprecision and for exact rational arithmetic in the file
`papilo/misc/MultiPrecision.hpp`. Including this file will currently introduce the types
```
papilo::Quad
papilo::Float100
papilo::Float500
papilo::Float1000
papilo::Rational
```
The numeric type used by PaPILO will be referred to as REAL in the following section. It can be any of the above types as well as simply `double` for using standard double precision arithmetic.

To avoid confusion with types a short note on types like `papilo:Vec` and `papilo::String`.
Those types are aliases for types from the standard library, `std::vector` and `std::string`, that possibly use an adjusted allocator. If nothing is altered regarding the allocator then the type `papilo::Vec` will be exactly the same as `std::vector`.
It can be changed by adding a partial specialization of `papilo::AllocatorTraits<T>` after including `papilo/misc/Alloc.hpp` but before including any other header of PaPILO.

The C++ interface for using PaPILO mainly revolves around the classes
*   `papilo::Presolve<REAL>`, which controls the presolving routines,
*   `papilo::Problem<REAL>`, which holds the problem instance, and
*   `papilo::Postsolve<REAL>`, which can transform solutions in the reduced space into solutions for the original problem space.
The includes for those classes are under `papilo/core/{Problem,Postsolve,Presolve}.hpp`.

## Creating an instance of `papilo::Problem<REAL>`

The PaPILO binary uses the MPS parsing routine to construct an instance of `papilo::Problem<REAL>` with the call `papilo::MpsParser<REAL>::loadProblem("problem.mps")`.

For feeding a problem to PaPILO programmatically, there is the class
`papilo::ProblemBuilder<REAL>`. This class allows to efficiently build an `papilo::Problem<REAL>` incrementally.
The problem definition that PaPILO supports does not use a row sense, but uses left and right hand sides $l$ and $u$ of rows defined as
$\text{l} \leq a^\top x \leq \text{u}$. For defining a row that is an equation with right hand side $b$ one has to set $l = u = b$. For inequalities either $l$ or $u$ are set to infinity.
One thing where PaPILO differs from many solvers is how infinite values are encoded. Whether for column bounds or left/right hand sides of rows PaPILO encodes the infinite value as a logical condition.
This ensures that regardless of numerical type used for `REAL`, that infinite values are always treated the same.

The member functions for initializing the rows and columns are
```
setNumCols( int ncols )
setNumRows( int nrows )
```
After calling those functions the problem will have no nonzero entries but the given number of columns and rows. The left and right hand side values of rows the rows are set to $0$ as well as the lower and upper bounds of the columns. Additionally all columns are initialized as continuous columns.

Next the following member functions can be used to alter the bound information about rows and columns as well as the objective coefficients and integrality flags of the columns.

*   alter objective coefficient for columns
    ```
    setObj( int col, REAL val )
    ```
*   alter bound information for columns
    ```
    setColLb( int col, REAL lb )
    setColLbInf( int col, bool isInfinite )
    setColUb( int col, REAL ub )
    setColUbInf( int col, bool isInfinite )
    ```
*   mark column to be restricted to integral values or not
    ```
    setColIntegral( int col, bool isIntegral )
    ```
*   alter left and right hand sides of rows
    ```
    setRowLhsInf( int row, bool isInfinite )
    setRowRhsInf( int row, bool isInfinite )
    setRowLhs( int row, REAL lhsval )
    setRowRhs( int row, REAL rhsval )
    ```
*   set names of rows, columns, and the problem
    ```
    setRowName( int row, Str&& name )
    setColName( int col, Str&& name )
    setProblemName( Str&& name )
    ```

Adding nonzeros to the problem can be done with individual nonzero values, row-based, or column-based using the following functions:
```
   /// add the nonzero entries for the given column
   addColEntries( int col, int len, const int* rows, const R* vals )
   /// add a nonzero entry for the given row and column
   addEntry( int row, int col, const REAL& val )
   /// add the nonzero entries for the given row
   addRowEntries( int row, int len, const int* cols, const R* vals )
```
All those functions can be called multiple times, but a nonzero entry for a particular column in a particular row should only be added once.
For maximum efficiency the member function
`papilo::ProblemBuilder<REAL>::reserve(int nnz, int nrows, int ncols)` should be used to reserve all required memory before adding nonzeros.

Finally calling `papilo::ProblemBuilder<REAL>::build()` will return an instance of `papilo::Problem<REAL>` with the information that was given to the builder. The builder can be reused afterwards.

## Presolving an instance of `papilo::Problem<REAL>`

For this section we assume a problem instance is stored in a variable `problem` of type `papilo::Problem<REAL>`.

In order to presolve a problem instance we need to setup an instance of `papilo::Presolve<REAL>` and then call `papilo::Presolve<REAL>::apply(problem)`.

The same instance of `papilo::Presolve<REAL>` can be used for presolving multiple problem instances.
It stores the basic configuration of the presolving routine and constructing it with the default presolvers and settings to presolve the problem is straight forward:
```
papilo::Presolve<REAL> presolve;
presolve.addDefaultPresolvers();
papilo::PresolveResult<REAL> result = presolve.apply(problem);
```

After the above call `result.status` will contain an enum class of the following type:
```
/// result codes of a presolving routine
enum class PresolveStatus : int
{
   /// problem was not changed
   kUnchanged = 0,

   /// problem was reduced
   kReduced = 1,

   /// problem was detected to be unbounded or infeasible
   kUnbndOrInfeas = 2,

   /// problem was detected to be unbounded
   kUnbounded = 3,

   /// problem was detected to be infeasible
   kInfeasible = 4,
};
```

And `result.postsolve` contains an instance of the class `papilo::Postsolve<REAL>`.

## Postsolve of a solution in the reduced problem space

First we construct a `papilo::Solution<REAL>` from a `papilo::Vec<REAL>` of reduced solution values and an empty instance of `papilo::Solution<REAL>` to hold the original space solution.
The interface here is not the simplest for the current functionality. It is like this to support
postsolve of dual solutions in the future. The class `papilo::Solution<REAL>` cannot only hold primal solutions but also dual solution values, even though the postsolve routine does not yet support this step.

Obtaining the original space solution
```
papilo::Vec<REAL> reducedsolvals;
...
// set up the values of the reduced solution in the reduced index space
...
// create reduced solution and original solution
papilo::Solution<REAL> reducedsol(std::move(reducedsolvals));
papilo::Solution<REAL> origsol;

// transform the reduced solution into the original problem space
PostsolveStatus status = result.postsolve.undo(reducedsol, origsol);
```

The value of `status` is `PostsolveStatus::kOk` if everything worked or `PostsolveStatus::kFail` otherwise.
If everything worked then the vector `origsol.primal` contains the primal solution values in the original problem space.

# Presolve parameters

There are several parameters that can be adjusted to influence the behavior during presolving.
All the parameters and their default values are listed in the file `parameters.txt`.
Adjusting a parameter via the command line when using the PaPILO exectuable works like this:
```
papilo solve -f problem.mps -l problem.sol --presolve.randomseed=42
```
This call will use an adjusted random seed for the presolve routine.

Alternatively a file with the same format as `parameters.txt` can be used to set multiple parameters by
passing the setting file with the `-p`/`--parameter-settings` flag.

Passing the `--print-params` flag will print the parameters in a format similar to the one of `parameters.txt` before starting presolving.
The printed parameters will have the values they were set to, not the default values.

For adjusting the parameters programatically there are two ways.
The first way is to obtain an instance of `papilo::ParameterSet` by calling `papilo::Presolve<REAL>::getParameters()`.
It is important to call this member function after all presolvers have been added to the `papilo::Presolve<REAL>` class.
Otherwise not all parameters are available, e.g. the ones that are added by individual presolvers.
Now we can call `papilo::ParameterSet::setParameter( key, val )` to set parameters to their desired values.

If we want to adjust the random seed programatically this would look like
```
papilo::Presolve<REAL> presolve;
...
// add all presolvers
...
papilo::ParameterSet paramset = presolve.getParameters();
paramset.setParameter("presolve.randomseed", 42);
```
The function `papilo::ParameterSet::setParameter()` will throw exceptions if anything goes wrong,
e.g. if the parameter key was not recognized or the type of the value is not suitable.
Possible exceptions are of the types `std::out_of_range`, `std::domain_error`, or `std::invalid_argument` and contain a suitable error message.
For debugging it can be helpful to print the parameters stored within a `papilo::ParameterSet` which can be achieved by calling `paramset.printParams(std::cout)` and produces an output similar to the one in `parameters.txt`.

The second way to set a subset of parameters is by directly accessing the instance of `papilo::PresolveOptions` that is stored within each instance of `papilo::Presolve<REAL>`.
Setting the random seed with this method can simply be achieved by `presolve.getPresolveOptions().randomseed = 42`.
The caveat with directly accessing the `papilo::PresolveOptions` is, that parameters added by individual presolvers cannot be set and that no error checking is performed in case the user sets a parameter to an invalid value.
Nevertheless this can be convenient for setting basic things like tolerances, time limits, and thread limits.

# Adding a presolver

Adding a presolver to PaPILO requires the following steps. First create a class for your presolver that inherits publically from `papilo::PresolveMethod<REAL>`.
In the constructor at least adjust the name and the timing of the new presolver.
For the constraint propagation presolver this is done in the constructor by the following calls:
```
this->setName( "propagation" );
this->setTiming( PresolverTiming::kFast );
```
Use the following rules to set a suitable value for `PresolverTiming`.
* `PresolverTiming::kFast` for presolvers that work only on altered parts of the problem.
  E.g. the constraint propagation presolver will only look at rows for which the activity was adjusted
* `PresolverTiming::kMedium` for presolvers that run in $O(n \log n)$ of the problem size (number of rows/columns/nonzeros).
  E.g. the parallel row/column presolvers compute hash values and sort on the order of the rows and columns.
  Technically the runtime becomes quadratic in the worst case when many collisions occur, but practically the presolvers are fast.
  They are the slowest in the category of medium timing presolvers included in the default.
* `PresolverTiming::kExhaustive` for presolvers that neither fit into the fast or medium categories

In addition, depending on the presolve method, it might be good to add restrictions on what type of problems a presolver is called.
For this the `setType(PresolveType)` member function of `papilo::PresolveMethod<REAL>` should be used.
If the type is set to `PresolverType::KIntegralCols` or `PresolverType::kContinuousCols` then the presolver only runs if integral or continuous columns are present respectively. The type `PresolverType::kMixedCols` runs only if both integral and continuous columns are present which is suitable for the implied integer detection presolver.

For presolve methods that require internal state two additional virtual member functions need to be overriden.
An example for this can be seen in the `papilo::Probing<REAL>` presolver.
The probing presolver stores how often a variable has been probed between calls.
For this it overrides the `papilo::PresolveMethod<REAL>::initialize()` member function which clears the internal data and
adjusts it to the given problems dimension. The function must return a boolean value.
The value `false` indicates that the presolver does not need to be informed in case the index space of the columns changes
and the value `true` indicates the opposite. When `true` is returned in the initialize member function `papilo::PresolveMethod<REAL>::compress()`
should additionally be overriden. The function is called with a mapping for the rows and columns and is called when
PaPILO compresses the index space of the problem.
In the compress callback the probing presolver uses the given column mapping to compress the vector that stores how often each variable has been probed.

If the presolve method needs to add parameters that can be adjusted via the `papilo::ParameterSet`, then the member function `papilo::PresolveMethod<REAL>::addPresolverParams()` should be overridden.

The member function `papilo::PresolveMethod<REAL>::execute()` is a pure virtual function and therefore must be implemented for every presolve method.
This is where the presolver actually runs. The method grants read-only access to the `papilo::Problem<REAL>` that is being presolved
and grants write access to an instance of `papilo::Reductions<REAL>`. A presolve method should scan the problem for possible reductions and add them
to the reductions class. This is only the broad picture and there are some more details that we omit here for brevity.
Looking at some of the default implemented presolvers can help for understanding further details and also the resources in the next section.

Finally when a new presolve method is implemented it needs to be added to the `papilo::Presolve<REAL>` instance that is used for presolving.
Assuming the new presolve method is called `MyPresolveMethod` this is achieved by the following call:
```
presolve.addPresolver( std::unique_ptr<papilo::PresolveMethod<REAL>>( new MyPresolveMethod<REAL>() ) );
```
Getting the PaPILO binary to call your presolver could be achieved by adding an include for your presolver in `papilo/core/Presolve.hpp` and then adding it together with the other default presolvers in the member function `papilo::Presolve<REAL>::addDefaultPresolvers()`.

# References and how to cite

Any publication for which PaPILO is used must include an acknowledgement and a reference to the following article:
> PaPILO: A Parallel Presolving Library for Integer and Linear Programming with Multiprecision Support
>
> Gleixner, Ambros and Gottwald, Leona and Hoen, Alexander
>
> available at https://arxiv.org/abs/2206.10709

Most of the presolve methods implemented in PaPILO are described in the paper "Presolve Reductions in Mixed Integer Programming" by Achterberg et al.
which is available under https://opus4.kobv.de/opus4-zib/files/6037/Presolve.pdf.

Some details on how PaPILO works internally are presented in a talk given during the SCIP workshop 2020 which has been recorded
and is available under https://www.youtube.com/watch?v=JKAyyWXGeQM.

# Licensing

To avoid confusion about licensing a short note on the LGPLv3.
This note is just an explanation and legally only the license text itself is of relevance.

When PaPILO is used as a header-based library then only section 3 of LGPLv3 is relevant, and not section 4. Therefore PaPILO in that setting could be used in a software that is distributed under the terms of a different license when the conditions of section 3 are met, which are

    a) Give prominent notice with each copy of the object code (refers to binary distributions of your software) that the Library (refers to PaPILO) is used in it and that the Library and its use are covered by this License (refers to the LGPLv3).
    b) Accompany the object code with a copy of the GNU GPL and this license document.

Modifications of PaPILO itself, however, must be distributed under LGPLv3.

For other licensing options we refer to https://scipopt.org/, where PaPILO can be obtained as part of the SCIP Optimization Suite.

# Contributors

[Ambros Gleixner](https://www.zib.de/members/gleixner) ([@ambros-gleixner](https://github.com/ambros-gleixner)) &mdash; project head

[Alexander Hoen](https://www.zib.de/members/hoen)  ([@alexhoen](https://github.com/alexhoen)) &mdash; main developer

[Franziska Schlösser](https://www.zib.de/members/schloesser)  ([@fschloesser](https://github.com/fschloesser)) &mdash; build system

#### Former Contributors

Ivet Galabova ([@galabovaa](https://github.com/galabovaa)) &mdash; initial draft for dual postsolve

[Leona Gottwald](https://www.zib.de/members/gottwald)  ([@lgottwald](https://github.com/lgottwald)) &mdash; creator and former main developer

[Katrin Halbig](https://www.datascience.nat.fau.eu/research/groups/amio/members/katrin-halbig/) ([@khalbig](https://github.com/khalbig)) &mdash; presolver for GCD based reductions on inequalities (SimplifyIneq), strengthening in DualFix

Gabriel Kressin ([@GabrielKP](https://github.com/GabrielKP)) &mdash; numerical statistics, testing

Anass Meskini &mdash; general development and contributions to substitution presolver in terms of internship

[Daniel Rehfeldt](https://www.zib.de/members/rehfeldt) ([@dRehfeldt](https://github.com/dRehfeldt)) &mdash; core data structures and MPS parsing
