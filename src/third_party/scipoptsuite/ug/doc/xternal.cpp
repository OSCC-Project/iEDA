/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   xternal.c
 * @brief  main document page
 * @author  Yuji Shinano
 * @author  Franziska Schloesser
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/** @mainpage Overview
 *
 * @tableofcontents
 *
 * @section WHATISUG What is UG?
 *
 * \UG is originally a framework to parallelize state-of-the-art branch-and-bound based solvers to solve optimization problems such
 * as constraint integer programs (CIPs) and mixed-integer (non-)linear programs. In particular
 *
 * - \UG incorporates a mixed-integer programming (MIP) solver as well as
 * - an LP based mixed-integer nonlinear programming (MINLP) solver, and
 * - is a framework for branch-and-cut-and-price programs.
 *
 * \UG is a high-level task parallelization framework that is generalized, that means it can also handle both branch-and-bound-based and non-branch-and-bound based solvers starting from \UG version 1.0.
 *
 * See the web site <a href="http://ug.zib.de">ug.zib.de</a> for more information about licensing and to download the recent release.
 *
 *
 * @section TABLEOFCONTENTS Structure of this manual
 *
 * This manual gives an accessible introduction to the functionality of the UG code in the following chapters
 *
 * - @subpage GETTINGSTARTED      Installation and license information and an interactive shell tutorial
 * - @subpage PARAMETERS          List of all UG parameters
 * - @subpage PROGRAMMING         Important programming concepts for working with(in) UG.
 * - @subpage AUTHORS             UG Authors
 * - @subpage CONCEPT             Concept of UG's high-level task parallelization framework
 * - @subpage QUICKSTART          Quickstart
 * - @subpage DOC                 How to search the documentation for interface methods
 * - @subpage HIERARCHYDIRS       Classes hierarchy and source code directory organization
 * - @subpage CODE                Coding style guidelines
 * - @subpage DEBUG               Debugging
 *
 *
 * @page CONCEPT Concept of UG's high-level task parallelization framework
 *
 * One of the most famous high-level task parallelization frameworks would be one that follows the \b Master-Worker paradigm:
 * A \b Task represents an operation that is running on a \b Worker.
 * In this context, the word \e high-level means that the \e granularity of the \b Task is coarse.
 * The words \b Task and its granularity are generally used in the field of parallel programming,
 * where the granularity indicates how much computation is needed to process one \b Task.
 *
 * \UG is a high-level task parallelization framework following the \b Supervisor-Worker paradigm:
 * From the configuration point of view, \UG is composed by the two elements \b LoadCoordinator and \b Solvers,
 * where the \b LoadCoordinator is the \b Supervisor and a \b Solver is a \b Worker.
 * In the UG's high-level task parallelization framework, the goal is to parallelize a state-of-the-art single-threaded or multi-threaded \b Solver,
 * in which the \b Task operation is very much complicated and is processed by a large number of lines of code.
 * For example, a Mixed Integer Programming problem (MIP) solver was the initial main target.
 * In that case, a \b Task is a (sub-)MIP.
 *
 * To understand \UG, the concepts of \b Master-Worker and \b Supervisor-Worker are compared
 * using the example of the way a parallel Branch-and-bound solver works.
 *
 * @subsection PARADIGMMASTERWORKER "Master-Worker paradigm"
 *
 * In the \b Master-Worker paradigm, the communication is very simple, the key messages are:
 * - \b Task : contains task information, such as a (sub-)problem representation and current best incumbent value
 * - \b Result : contains all information regarding the task execution. In the case of a parallel MIP solver, the best incumbent solution and all open nodes as examples.
 *
 * \image html Master-Worker.png "A parallel MIP solver based on the Master-Worker paradigm" width=500px
 * \image latex Master-Worker.png "A parallel MIP solver based on the Master-Worker paradigm" width=500px
 *
 * The \b granularity of a Task can be controlled in the Worker by a termination criterion for a (sub-)problem computation,
 * such as the number of open nodes generated.
 * Note that all open branch-and-bound nodes are managed by the Master side and the number of transferred nodes is huge in case the solution process
 * generates a huge search tree.
 * In order to reduce the number of open nodes, usually depth-first search is used in the Worker side.
 * An advantage is that the all nodes in the Master are independent, that is, all nodes can be a (sub-)problem root.
 * This fact allows for a simple checkpoint and restart mechanism, that can fully save and resume the tree search.
 * This loosely coupled Master-Worker paradigm is very well suitable for high-throughput computing,
 * but is not so efficient in large-scale high-performance computing.
 *
 * @subsection PARADIGMSUPERVISORWORKER "Supervisor-Worker paradigm"
 *
 * Different from the previously explained Master-Worker paradigm, the \b Supervisor-Worker paradigm used in UG's high-level task
 * parallelization framework defines a very flexible message passing protocol.
 * The key messages are:
 * - \b Task : contains task information, such as a (sub-)problem representation, and it indicates the beginning of the Task computation.
 * - \b Status : contains the Task computation status and the nitification frequency to Supervisor, which can be specified at run-time.
 * - \b Completion : indicates the termination of the task computation.
 *
 * In between the \b Task and \b Completion messages, any message passing protocol can be defined, for example
 * - \b Solution : represents the best incumbent solution. Then, the solution can be share whenever a single Solver found a new one.
 * - \b InCollecting : indicates that the Supervisor needs new Tasks. This message allows to collect (sub-)problems on demand.
 * - \b OutCollecting : indicates that the Supervisor does not need new Tasks.
 * - \b Interrupt : indicates the current executing Task can be interrupted
 * - etc.
 *
 * \image html Supervisor-Worker.png "A parallel MIP solver based on Supervisor-Worker" width=500px
 * \image latex Supervisor-Worker.png "A parallel MIP solver based on Supervisor-Worker" width=500px
 *
 * The \b LoadCoordinator manages and does a dynamic load balancing among \b Solvers.
 * However, the \b LoadCoordinaor only keeps root (sub-)problem nodes at run-time.
 * For checkpoints, an even smaller number of only essential nodes are saved.
 * These essential nodes are checking relation among the root nodes, that is, if its ancestor node exits.
 * If it does exist, then the node does not have to be saved, since it can be regenerated from its ancestor.
 * This looks inefficient, but in a state-of-the-art MIP solver, it sometimes works very well
 * (see <a href="https://ieeexplore.ieee.org/document/6969561">
 * Solving Hard MIPLIB2003 Problems with ParaSCIP on Supercomputers: An Update</a>).
 * For Supervisor-Worker, see
 * <a href="https://link.springer.com/chapter/10.1007/978-3-319-63516-3_8">
 * Parallel Solvers for Mixed Integer Linear Optimization</a> in more detailed.
 *
 * UG can flexibly define the message passing protocol in between \b Task and \b Completion.
 *
 * The key feature of UG's high-level Task parallelization framework is the above described flexible message passing protocol
 * during a \b Task computation.
 *
 *
 * @page QUICKSTART Quickstart
 *
 *  The stand-alone shared memory parallel MIP/MINLP solver FiberSCIP can be used easily via the `fscip` command.
 *  Let's consider the following minimal example in LP format,
 *  a 4-variable problem with a single, general integer variable and three linear constraints:
 *
 *  \verbinclude simple.lp
 *
 *  Saving this file as "simple.lp" allows to read it into FiberSCIP.
 *  Create a default parameter file for FiberSCIP:
 *
 *  \verbinclude default.prm
 *
 *  The column starting with "#" is treated as a comment.
 *  Therefore, this parameter file contains all default settings.
 *  Save this file as "default.prm" and solve "simple.lp" with these settings by running the command:
 *
 * ```
 * fscip default.prm simple.lp -q
 * ```
 *
 * This model is solved by using the maximal number of cores on your PC:
 *
 * \verbinclude output.log
 *
 * The solution file "sample.sol" will be written as below:
 *
 * \verbinclude simple.sol
 *
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/**@page GETTINGSTARTED Getting started
 *
 * - @subpage WHATPROBLEMS "What types of optimization problems does UG solve?"
 *
 * - @subpage LICENSE "License"
 * - @subpage INSTALL "Installation"
 * - @subpage DOC     "How to search the documentation for interface methods"
 */

/**@page PARAMETERS List of all UG parameters
 *
 * This page lists all parameters of the current UG version.
 * It can easily be generated running FiberSCIP with the following parameters:
 *
 * <code>Quiet = FALSE</code><br>
 * <code>OutputParaParams = 4</code>
 *
 * \verbinclude parameters.prm
 *
 *
 */

/**@page PROGRAMMING Programming with UG
 *
 * The UG high-level task parallelization framework provides a systematic way to develop a massively parallel solver,
 * that can run on large scale supercomputers.
 *
 * 1. develop and debug a stand-alone \b Solver, if the \b Solver does not exit.
 * 2. develop and debug a shared-memory parallel solver, in which \b Solvers are managed by \b LoadCoordinator.
 * 3. develop and debug a distributed-memory parallel solver, in which \b Solvers are managed by \b LoadCoordinator.
 *
 * Note that, algorithmically, 2 and 3 work the same.
 * Therefore, a massively parallel solver can be developed on a PC.
 *
 * - @subpage HIERARCHYDIRS "Classes hierarchy and source code directory organization"
 * - @subpage CODE          "Coding style guidelines"
 * - @subpage DEBUG         "Debugging"
 */

/**@page AUTHORS UG Authors
 *
 * The current main developer is Yuji Shinano.
 *
 */

/**@page WHATPROBLEMS What types of optimization problems does UG solve?
 *
 * \UG, as a stand-alone solver using \b SCIP as the underlying base solver,
 * is called FiberSCIP and ParaSCIP can solve mixed-integer linear and nonlinear programs.
 * For solving \b MINLP it applies an LP based spatial branch-and-cut algorithm,
 * that is guaranteed to solve bounded MINLPs within a given numerical tolerance in a finite amount of time.
 *
 * The \b SCIP applications
 *
 * - \b STP (Steiner Tree Problem) and its applications, also known as \b SCIP-Jack, and
 * - \b MISDPs (mixed integer semidefinite programs) provided by \b SCIP-SDP,
 *
 * are parallelized by \UG. They can be solved as stand-alone \UG applications.
 *
 * Any branch-and-bound based solver for a specific problem, which is realized
 * by \b SCIP plugins, can easily parallelized by adding a small lines of glue code.
 *
 * (see, <a href="https://ieeexplore.ieee.org/document/8778206">
 * An Easy Way to Build Parallel State-of-the-art Combinatorial Optimization Problem Solvers:
 * A Computational Study on Solving Steiner Tree Problems and Mixed Integer Semidefinite Programs
 * by using ug[SCIP-*,*]-Libraries</a>)
 *
 * In the \UG framework, the underlying base solver is abstracted.
 * By adding the \UG wrapper code etc. for a specific optimization problem,
 * a massively parallel solver for the problem can be built.
 * Ongoing projects are the following:
 * - Quadratic Assignment Problem (\b QAP):
 *    DNN relaxation with Newton Bracket method based branch-and-bound solver is parallelized
 *    (see, <a href="https://arxiv.org/abs/2101.09629">Solving Challenging Large Scale QAPs</a>).
 * - Traveling Salesman Problem (\b TSP): <a href="http://www.math.uwaterloo.ca/tsp/concorde.html">
 *    Concorde TSP solver</a> has been parallelized by \UG.
 * - \b MIP: Another MIP solver, in which multi-threaded FICO Xpress solver is used as underlying base solver
 *   (see, <a href="https://www.tandfonline.com/doi/abs/10.1080/10556788.2018.1428602?journalCode=goms20">ParaXpress: an experimental extension of the FICO Xpress-Optimizer to solve hard MIPs on supercomputers</a>).
 *
 * Also, \UG version 1.0 can be used to parallelize non-branch-and-bound based solvers:
 * A Shortest Vector Problem (\b SVP : see <a href="https://www.latticechallenge.org/svp-challenge/">
 * SVP Challenge</a>) solver has been parallelized by \UG.
 *
 *
 *
 *
 *
 */

/**@page LICENSE License
 *
 * \verbinclude COPYING
 */

/**@page INSTALL Installing UG
 *
 * This chapter is a detailed guide to the installation procedure of UG.
 *
 * UG lets you freely choose between its own, manually maintained Makefile system
 * or the CMake cross platform build system generator. For new users, we strongly
 * recommend to use CMake, if available on their targeted platform.
 *
 * - @subpage CMAKE   "Installation information using CMake (recommended for new users)"
 * - @subpage MAKE    "Installation information using Makefiles (deprecated)"
 */

/**@page CMAKE Building UG with CMake
 *
 * <a href=https://cmake.org/>CMake</a> is a build system generator that can create, e.g., Makefiles for UNIX and Mac
 * or Visual Studio project files for Windows.
 *
 * CMake provides an <a href="https://cmake.org/cmake/help/latest/manual/cmake.1.html">extensive documentation</a>
 * explaining available features and use cases as well as an <a href="https://cmake.org/Wiki/CMake_FAQ">FAQ section</a>.
 * It's recommended to use the latest stable CMake version available. `cmake --help` is also a good first step to see
 * available options and usage information.
 *
 * Platform independent build instructions:
 *
 * ```
 * cmake -Bbuild -H. [-DSCIP_DIR=/path/to/scip]
 * cmake --build build
 * ```
 *
 * Linux/macOS Makefile-based build instructions:
 *
 * ```
 * mkdir build
 * cd build
 * cmake ..
 * make
 * ```
 *
 * CMake uses an out-of-source build, i.e., compiled binaries and object files are separated from the source tree and
 * located in another directory. Usually this directory is called `build` or `debug` or whatever you prefer. From within
 * this directory, run `cmake <path/to/UG>` to configure your build, followed by `make` to compile the code according
 * to the current configuration (this assumes that you chose Linux Makefiles as CMake Generator). By default, UG
 * searches for SCIP as base solver. If SCIP is not installed systemwide, the path to a CMake build directory
 * of SCIP must be specified (ie one that contains "scip-config.cmake").
 *
 * Afterwards, successive calls to `make` are going to recompile modified source code,
 * without requiring another call to `cmake`.
 *
 * The generated executable and libraries are put in directories `bin` and `lib` respectively and will simply be named `fscip`.
 *
 * @section CMAKE_CONFIG Modifying a CMake configuration
 *
 * There are several options that can be passed to the `cmake <path/to/UG>` call to modify how the code is built.
 * For all of these options and parameters you have to use `-D<Parameter_name>=<value>`. Following a list of available
 * options, for the full list run
 *
 * ```
 * cmake <path/to/UG> -LH
 * ```
 *
 * CMake option         | Available values               | Makefile equivalent    | Remarks                                    |
 * ---------------------|--------------------------------|------------------------|--------------------------------------------|
 * CMAKE_BUILD_TYPE     | Release, Debug, ...            | OPT=[opt, dbg]         |                                            |
 * SCIP_DIR             | /path/to/scip/install          |                        |                                            |
 *
 * Parameters can be set all at once or in subsequent calls to `cmake` - extending or modifying the existing
 * configuration.
 *
 * @section CMAKE_INSTALL Installation
 *
 * CMake uses a default directory for installation, e.g., /usr/local on Linux. This can be modified by either changing
 * the configuration using `-DCMAKE_INSTALL_PREFIX` as explained in \ref CMAKE_CONFIG or by setting the environment
 * variable `DESTDIR` during or before the install command, e.g., `DESTDIR=<custom/install/dir> make install`.
 *
 * @section CMAKE_TARGETS Additional targets
 *
 * There are several further targets available, which can be listed using `make help`. For instance, there are some
 * examples that can be built with `make examples` or by specifying a certain one: `make <example-name>`.
 *
 * | CMake target    | Description                                           | Requirements                          |
 * |-----------------|-------------------------------------------------------|---------------------------------------|
 * | fiberscip       | build fiberscip UG executable                         |                                       |
 * | parascip        | build parascip UG executable                          |                                       |
 * | githash         | build executables for all applications                |                                       |
 */

/**@page MAKE Makefiles / Installation information
 *
 * Compiling with \UG directly can be done as follows:
 *
 * Type `make` in the console from the UG main folder. The system will ask you to provide necessary links and build fiberscip.
 * Options include `OPT=<opt|dbg>` to select optimized or debug mode, `COMM=<pth,mpi,cpp11>` to select pthreads, MPI or cpp11.
 * Also, some Makefile flags from scip can be set, such as `LPS` and `IPOPT`. Please refer to the
 * <a href="https://scipopt.org/doc-8.0.0/html/md_INSTALL.php">SCIP documentation</a> for a full list.
 *
 * With `make doc` you can build a local copy of this documentation if you have doxygen installed in your machine.
 * After the build process completed, open `doc/html/index.html` with your favourite browser.
 *
 */

/**@page DOC How to search the documentation for interface methods
 *
 * If you are looking for a method in order to perform a specific task, the public UG Documentation is the place to look.
 *
 */

/**@page HIERARCHYDIRS Classes hierarchy and source code directory organization
 *
 * The released code in the Ubiquity Generator framework (UG) has the following classes hierarchy and code directory organization.
 *
 * \image html classes.png "Classes hierarchy and source code directory organization" width=500px
 * \image latex classes.png "Classes hierarchy and source code directory organization" width=500px
 *
 * - UG base classes: Classes for high-level task parallelization features
 * - B&B Base classes: Classes for general parallel Branch-and-Bound features
 * - UG_SCIP classes: Classes to parallelize the SCIP solver
 *
 * To add code to a specific baseSolver locate the source code files to parallelize the baseSolver in
 *
 * - src/ug_baseSolver
 *
 * The code specific to baseSolver can be inherited from B&B base classes or from UG base classes depending on the solver being based on B&B or non-B&B.
 *
 */

/**@page CODE Coding style guidelines
 *
 * We follow the following coding style guidelines and recommend them for all developers in coding-style-guidelines.md.
 *  \verbinclude coding-style-guidelines.md
 *
 * If you want to contribute to UG development, please check out the contribution-guidelines.md.
 *  \verbinclude contribution-guidelines.md
 *
 * @section CODESPACING Spacing:
 *
 * - Indentation is 3 spaces. No tabs anywhere in the code.
 * - Every opening parenthesis requires an additional indentation of 3 spaces.
 *
 *   @refsnippet{src/scip/branch_relpscost.c,SnippetCodeStyleParenIndent}
 *
 * @section CODENAMING  Naming:
 *
 * - Use assert() to show preconditions for the parameters, invariants, and postconditions.
 * - Make all functions that are not used outside the module 'static'.
 * - Naming should start with a lower case letter.
 *
 *   @refsnippet{src/ug/branch_relpscost.c,SnippetCodeStyleStaticAsserts}
 *
 * @section CODEDOC Documentation:
 *
 * - Document functions, parameters, and variables in a doxygen conformed way.
 *
 * @section ECLIPSE Customize eclipse
 *
 * Eclipse user can use the profile below. This profile does not match the \UG coding guideline completely.
 *
 * \include codestyle/eclipse_ug_codestyle.xml
 *
 */

/**@page DEBUG Debugging
 *
 *  If you need to debug your own code that uses UG, here are some tips and tricks:
 *
 *  - Use <b>asserts</b> in your code to show preconditions for the parameters, invariants and postconditions.
 *    Assertions are boolean expressions which inevitably have to evaluate to <code>TRUE</code>. Consider the
 *    following example:
 *
 *    @refsnippet{src/scip/cons_linear.c,SnippetDebugAssertions}
 *
 *  - Compile UG in debug mode and run it with <a href="https://github.com/rr-debugger/rr">RR (Record and Replay)</a> or your favourite debug tool.
 *
 */
