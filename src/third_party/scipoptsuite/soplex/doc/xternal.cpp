/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*  Copyright 1996-2022 Zuse Institute Berlin                                */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SoPlex; see the file LICENSE. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   xternal.cpp
 * @brief  SoPlex documentation pages
 * @author Ambros Gleixner
 * @author Thorsten Koch
 * @author Matthias Miltenberger
 * @author Sebastian Orlowski
 * @author Marc Pfetsch
 * @author Andreas Tuchscherer
 */

/**@mainpage Overview
 *
 * @section MAIN What is SoPlex?
 *
 * SoPlex is an optimization package for solving linear programming problems
 * (LPs) that can be used standalone via a command line interface and as a
 * callable library.  Its main features are:
 *
 * - an advanced implementation of the primal and dual revised simplex
 *   algorithm,
 *
 * - an object-oriented software design written in C++,
 *
 * - presolving, scaling, exploitation of sparsity, hot-starting from any
 *   regular basis,
 *
 * - column- and row-oriented form of the simplex algorithm,
 *
 * - a compile-time option to use 80bit extended ("quad") precision for
 *   numerically difficult LPs, and
 *
 * - special support for the exact solution of LPs over the rational numbers.
 *
 * SoPlex has been used in numerous research and industry projects and is the
 * standard LP solver linked to the constraint integer programming solver <a
 * href="http://scipopt.org/">SCIP</a>.
 *
 *@section INST Download, License, and Installation
 *
 * SoPlex can be downloaded in source code and as precompiled binaries from the
 * <a href="http://soplex.zib.de">SoPlex</a> web page.  It is also distributed
 * as part of the <a href="http://scipopt.org/">SCIP Optimization Suite</a>.
 *
 * SoPlex is distributed under the terms of the \ref LICENSE "Apache 2.0"
 * See the <a
 * href="http://soplex.zib.de">SoPlex</a> web page or contact us for more
 * information.
 *
 * For help with the installation please consult the \ref INSTALL "INSTALL" file.
 *
 * @section GETTINGSTARTED Getting started
 *
 * - \ref FAQ      "Frequently Asked Questions"
 *
 * - \ref CMD      "How to use the SoPlex command line"
 *
 * - \ref LIB      "How to use SoPlex as a callable library"
 *
 * - \ref PARS     "Parameters in SoPlex"
 *
 * - \ref PROG     "Programming with SoPlex"
 *
 * - \ref EXACT    "How to use SoPlex as an exact LP solver"
 *
 * - \ref CHANGELOG "CHANGELOG"
 *
 * A tutorial article for getting started with the SCIP Optimization Suite,
 * which includes SoPlex, is available as <a
 * href="http://scipopt.org/doc/ZR-12-27.pdf">ZIB-Report 12-27</a>.
 *
 * @section AUTHORS Authors
 *
 * The initial implementation of SoPlex has been developed by Roland Wunderling
 * as part of his Ph.D. thesis <a
 * href="http://www.zib.de/PaperWeb/abstracts/TR-96-09">"Paralleler und
 * Objektorientierter Simplex-Algorithmus"</a> from 1996.  Since then many
 * developers have maintained and improved the underlying algorithms.  See the
 * <a href="http://soplex.zib.de">SoPlex</a> web page for a comprehensive list
 * of all contributors.
 *
 * @version  6.0.3
 */


/**@namespace soplex
   @brief     Everything should be within this namespace.

   We have put the whole class library in the namespace soplex.
   If anything here is defined outside, this is a mistake and
   should be reported.
*/


/**@defgroup Elementary Elementary Classes
   @brief    General purpose classes.

   Elementary classes are provided for general purpose use in
   projects way beyond the scope of numerical software or linear
   programming.
*/

/**@defgroup Algebra Linear Algebra Classes
   @brief Basic data types for linear algebra computations.

   Linear algebra classes provide basic data types for (sparse)
   linear algebra computations. However, their functionality is
   restricted to simple operations such as addition and scaling.
   For complex tasks, such as solving linear systems of equations,
   algorithmic classes are provided instead.
*/

/**@defgroup Algo Algorithmic Classes
   @brief Implementation of numerical algorithms.

   Algorithmic classes serve for implementing a variety of
   algorithms for solving numerical (sub-)problems.
*/

/**@page DataObjects Data Objects

    \em Data \em objects refer to C++ objects that do not allocate any
    resources, particularly that do not allocate any memory.  This
    makes them behave just like ordinary C structures, in that both,
    the copy constructor and assignment operator are equivalent to a
    memcopy of the memory taken by the object. Examples for data
    objects are all builtin types such as \c int or \c double or
    \e simple classes such as \c complex.

    We distinguish \em data \em objects from general C++ objects that
    may include some allocation of resources. (Note that for general
    C++ objects that do allocate resources, this must be respected by
    providing appropriate copy constructor and assignment operators.)
    An example for a general C++ class is class DataArray.

    The distinction between data and general C++ objects becomes
    relevant when using such objects in container classes such as
    DataArray or Array.
*/


/**@page LICENSE License
 *
 * \verbinclude LICENSE
 */


/**@page CHANGELOG CHANGELOG
 *
 * \verbinclude CHANGELOG
 */


/**@page FAQ Frequently Asked Questions
 * \htmlinclude faq.inc
 */


/**@page CMD How to use the SoPlex command line
 *
 * Running the command line binary of SoPlex without any arguments displays a
 * list of options.  You can write a parameter file with default parameters by
 * using option \-\-saveset=FILENAME.set.  After changing parameter values in this
 * file you can use it by with \-\-loadset=FILENAME.set.  The most frequently used
 * parameters have abbreviations.
*/


/**@page LIB How to use SoPlex as a callable library
 *
 *@section LIB1 Namespace "soplex"
 *
 * The entire SoPlex code is contained in the namespace \ref soplex.  Because of
 * this, either all classes and methods must be qualified by the prefix
 * "soplex::" or a "using namespace soplex;" must be present.
 *
 *@section LIB2 Interface class
 *
 * The main interface is given by the class \ref soplex::SoPlex "SoPlex", which
 * handles the construction and modification of an LP, the solving process,
 * allows to access and change parameters, and retrieve solution information.
 *
 * A basic example on how to construct and solve an LP via the class
 * \ref soplex::SoPlex "SoPlex" is given in the file \ref example.cpp.
 *
 *@section LIB3 Deprecated 1.x interface
 *
 * With version 2.0, the SoPlex class has been updated significantly compared to
 * the 1.x version.  Since version 4.0, the old version cannot be used anymore.
*/


/**@page PARS Parameters in SoPlex
 *
 * Since SoPlex 2.0, the main interface class \ref soplex::SoPlex "SoPlex" provides an improved management of
 * parameters.  Currently, there are three types of parameters for boolean, integer, and real values.  A list of default
 * parameters is provided \ref PARSLIST "here".
 *
 *@section PARS1 Setting parameters at the command line
 *
 * When using the command line interface, parameters can be changed with the generic option
 *
 * <code>\-\-&lt;type&gt;:&lt;name&gt;=&lt;val&gt;</code>
 *
 * where &lt;type&gt; is one of <code>bool</code>, <code>int</code>, or <code>real</code> and name is the name of the
 * parameter.  E.g., in order to deactivate the simplifier, one can use the option <code>\-\-bool:simplifier=0</code>.
 *
 *@section PARS2 Setting parameters via the callable library
 *
 * When using the callable library via the class \ref soplex::SoPlex "SoPlex" (of version 2.0 and above), parameters can
 * be changed by the methods \ref soplex::SoPlex::setBoolParam() "setBoolParam()", \ref soplex::SoPlex::setIntParam()
 * "setIntParam()", and \ref soplex::SoPlex::setRealParam() "setRealParam()".  See their documentation for details.
 */


/**@page PARSLIST List of all SoPlex parameters
 *
 * This page lists all parameters of the current SoPlex version. This list can
 * easily be generated by the SoPlex command line interface using:
 *
 * <code>soplex \-\-saveset=&lt;file name&gt;.set</code>
 *
 * or via the method \ref soplex::SoPlex::saveSettingsFile() "saveSettingsFile(&quot;&lt;file name&gt;.set&quot;, true)" of the class \ref
 * soplex::SoPlex "SoPlex".
 *
 * \verbinclude doc/parameters.set
 */


/**@page PROG Programming with SoPlex
 *
 * Besides the main interface class \ref soplex::SoPlex "SoPlex", the classes of
 * the SoPlex library are categorized into three different types:
 *
 * - Elementary classes are provided for general purpose use in projects way
 *   beyond the scope of numerical software or linear programming.
 *
 * - Linear algebra classes provide basic data types for (sparse) linear algebra
 *   computations. However, their functionality is restricted to simple
 *   operations such as addition and scaling.  For complex tasks, such as
 *   solving linear systems of equations, algorithmic classes are provided
 *   instead.
 *
 * - Algorithmic classes serve for implementing maybe a variety of algorithms
 *   for solving numerical (sub-)problems.
 *
 * The main class implementing the primal and dual simplex algorithm is the
 * class \ref soplex::SPxSolver "SPxSolver".  The following sections are
 * dedicated to users who want to provide own components of the simplex
 * algorithm such as pricers, ratio tests, start basis generators or LP
 * simplifiers to use with SoPlex's standard floating-point simplex
 * implementation.
 *
 *@section Representation Virtualizing the Representation
 *
 * The primal Simplex on the columnwise representation is structurally
 * equivalent to the dual Simplex on the rowwise representation and vice versa
 * (see below). Hence, it is desirable to treat both cases in a very similar
 * manner. This is supported by the programmer's interface of SoPlex which
 * provides access methods for all internal data in two ways: one is relative to
 * the "physical" representation of the LP in rows and columns, while the other
 * is relative to the chosen basis representation.
 *
 * If e.g. a \ref soplex::SPxPricer "SPxPricer" is written using the second type
 * of methods only (which will generally be the case), the same code can be used
 * for running SoPlex's simplex algorithm for both representations.  We will now
 * give two examples for this abstraction from the chosen representation.
 *
 * Methods \c vector() will return a column or a row vector, corresponding to
 * the chosen basis representation.  The other "vectors" will be referred to as
 * \em covectors:
 *
 * <TABLE>
 * <TR><TD>&nbsp;  </TD><TD>ROW      </TD><TD>COLUMN   </TD></TR>
 * <TR><TD>vector  </TD><TD>rowVector</TD><TD>colVector</TD></TR>
 * <TR><TD>coVector</TD><TD>colVector</TD><TD>rowVector</TD></TR>
 * </TABLE>
 *
 * Whether the \ref soplex::SPxBasis::Desc::Status "SPxBasis::Desc::Status" of a
 * variable indicates that the corresponding vector is in the basis matrix or
 * not also depends on the chosen representation. Hence, methods \c isBasic()
 * are provided to get the correct answer for both representations.
 *
 *@section Simplex Vectors and Bounds
 *
 * The Simplex algorithms keeps three vectors which are associated to each
 * basis.  Two of them are required for the pricing, while the third one is
 * needed for detecting feasibility of the basis. For all three vectors, bounds
 * are defined. The Simplex algorithm changes the basis until all three vectors
 * satisfy their bounds, which means that the optimal solution has been found.
 *
 * With each update of the basis, also the three vectors need to be
 * updated. This is best supported by the use of \c UpdateVectors.
 *
 *@subsection Variables
 *
 * The Simplex algorithm works with two types of variables, primals and duals.
 * The primal variables are associated with each column of an LP, whereas the
 * dual variables are associated with each row.  However, for each row a slack
 * variable must be added to the set of primals (to represent inequalities), and
 * a reduced cost variable must be added for each column (to represent upper or
 * lower bounds). Note, that mathematically, one dual variable for each bound
 * (upper and lower) should be added. However, this variable would always yield
 * the same value and can, hence, be implemented as one.
 *
 * To summarize, we have a primal variable for each LP column and row (i.e., its
 * slack) as well as a dual variable for each LP row and column (i.e., its
 * bounds). However, not all these values need to be stored and computed, since
 * the structure of the Simplex algorithms allow to keep them implicitly.
 *
 * If the SPxBasis's Status of a row or column is one of \c P_ON_LOWER, \c
 * P_ON_UPPER, \c P_FIXED or \c P_FREE, the value of the corresponding primal
 * variable is the lower, upper or both bound(s) or 0, respectively.  The
 * corresponding dual variable needs to be computed. Equivalently, for a Status
 * of \c D_FREE, \c D_ON_UPPER, \c D_ON_LOWER, \c D_ON_BOTH or \c D_UNDEFINED,
 * the corresponding dual variable is 0, whereas the primal one needs to be
 * computed.
 *
 * The following vectors are declared for holding the values to be computed: \c
 * primRhs, \c primVec (with dimension \c nCols()) for the primal variables, and
 * \c dualRhs, \c dualVec (with dimension \c nRows()) for the dual
 * variables. The additional variable \c addvec (with dimension \c coDim())
 * depends on the representation.
 *
 * @subsection Bounds
 *
 * Primal and dual variables are bounded (including \f$\pm\infty\f$ as bounds).
 * If all primal variables are within their bounds, the Simplex basis is said to
 * be primal feasible. Analogously, if all dual variables are within their
 * bounds, its is called dual feasible.  If a basis is both, primal and dual
 * feasible, the optimal solution has been found.
 *
 * In the dual Simplex, the basis is maintained dual feasible, while primal
 * feasibility is improved via basis updates. However, for numerical reasons
 * dual feasibility must be relaxed from time to time.  Equivalently, primal
 * feasibility will be relaxed to retain numerical stability in the primal
 * Simplex algorithm.
 *
 * Relaxation of (dual or primal) feasibility is achieved by relaxing the bounds
 * of primal or dual variables. However, for each type of Simplex only the
 * corresponding bounds need to be relaxed. Hence, we define only one vector of
 * upper and lower bound for each row and column and initialize it with primal
 * or dual bound, depending on the Simplex type (see \c theURbound, \c
 * theLRbound, \c theUCbound, \c theLCbound).
 */


/**@page EXACT How to use SoPlex as an exact LP solver
 *
 * Since version 1.7, SoPlex implements an \em iterative \em refinement procedure on the level of linear programs, which
 * allows for computing extended-precision solutions beyond the limits of standard floating-point arithmetic.  It may be
 * particularly helpful for numerically troublesome LPs and applications that require solutions within tight feasibility
 * tolerances.  Since version 2.1 this has been extended to compute exact rational solutions.
 *
 * By default, SoPlex functions as a standard floating-point LP solver.  In order to use SoPlex as an exact LP solver,
 * you need to compile SoPlex with GMP support (default, see the \ref INSTALL "INSTALL" file) and change the following
 * parameters from their default value:
 *
 * - <code>real:feastol = 0</code>
 *
 * - <code>real:opttol = 0</code>
 *
 * - <code>int:solvemode = 2</code>
 *
 * - <code>int:syncmode = 1</code>
 *
 * - <code>int:readmode = 1</code> (optional, activates exact parsing of input files)
 *
 * - <code>int:checkmode = 2</code> (optional, activates exact final check of feasibility and optimality at the command
 *   line)
 *
 * See \ref PARS "this page" how to change parameters and the \ref PARSLIST "list of all SoPlex parameters" for their
 * detailed description.  A settings file <code>exact.set</code> for exact solving is provided in the directory
 * <code>settings</code> of the distribution.  On the command line, this can be read with option
 * <code>\-\-loadset=settings/exact.set</code>.
 *
 * If you have questions on particularly this feature you can contact <a href="http://www.zib.de/gleixner/">Ambros
 * Gleixner</a> or post them on the SoPlex mailing list.
 *
 *
 *@section EXACT4 References
 *
 * The mathematical background of the underlying methods is described in the papers
 *
 * - Ambros M. Gleixner, Daniel E. Steffy. <i>Linear programming using limited-precision oracles</i>.
 * Mathematical Pogramming. 183, pp. 525-554, 2020, available as <a
 *   href="http://nbn-resolving.de/urn:nbn:de:0297-zib-75316">ZIB-Report 19-57</a>.
 *
 * - Ambros M. Gleixner, Daniel E. Steffy, Kati Wolter. <i>Iterative Refinement for Linear Programming</i>.
 *   INFORMS Journal on Computing 28 (3). pp. 449-464, available as <a href="http://nbn-resolving.de/urn/resolver.pl?urn:nbn:de:0297-zib-55118">ZIB-Report 15-15</a>.
 *
 * <b>When using SoPlex as an exact LP solver, please cite the above papers.</b>
 */
