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

#include <iostream>
#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex.h"
#include "soplex/statistics.h"
#include "soplex/sorter.h"

//#define NO_TOL
#define USE_FEASTOL
//#define NO_TRANSFORM
//#define PERFORM_COMPPROB_CHECK

#define MAX_DEGENCHECK     20    /**< the maximum number of degen checks that are performed before the DECOMP is abandoned */
#define DEGENCHECK_OFFSET  50    /**< the number of iteration before the degeneracy check is reperformed */
#define SLACKCOEFF         1.0   /**< the coefficient of the slack variable in the incompatible rows. */
#define TIMELIMIT_FRAC     0.5   /**< the fraction of the total time limit given to the setup of the reduced problem */

/* This file contains the private functions for the Decomposition Based Dual Simplex (DBDS)
 *
 * An important note about the implementation of the DBDS is the reliance on the row representation of the basis matrix.
 * The forming of the reduced and complementary problems is involves identifying rows in the row-form of the basis
 * matrix that have zero dual multipliers.
 *
 * Ideally, this work will be extended such that the DBDS can be executed using the column-form of the basis matrix. */

namespace soplex
{

/// solves LP using the decomposition dual simplex
template <class R>
void SoPlexBase<R>::_solveDecompositionDualSimplex()
{
   assert(_solver.rep() == SPxSolverBase<R>::ROW);
   assert(_solver.type() == SPxSolverBase<R>::LEAVE);

   // flag to indicate that the algorithm must terminate
   bool stop = false;

   // setting the initial status of the reduced problem
   _statistics->redProbStatus = SPxSolverBase<R>::NO_PROBLEM;

   // start timing
   _statistics->solvingTime->start();

   // remember that last solve was in floating-point
   _lastSolveMode = SOLVEMODE_REAL;

   // setting the current solving mode.
   _currentProb = DECOMP_ORIG;

   // setting the sense to maximise. This is to make all matrix operations more consistent.
   // @todo if the objective sense is changed, the output of the current objective value is the negative of the
   // actual value. This needs to be corrected in future versions.
   if(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE)
   {
      assert(_solver.spxSense() == SPxLPBase<R>::MINIMIZE);

      _solver.changeObj(_solver.maxObj());
      _solver.changeSense(SPxLPBase<R>::MAXIMIZE);
   }

   // it is necessary to solve the initial problem to find a starting basis
   _solver.setDecompStatus(SPxSolverBase<R>::FINDSTARTBASIS);

   // setting the decomposition iteration limit to the parameter setting
   _solver.setDecompIterationLimit(intParam(SoPlexBase<R>::DECOMP_ITERLIMIT));

   // variables used in the initialisation phase of the decomposition solve.
   int numDegenCheck = 0;
   R degeneracyLevel = 0;
   _decompFeasVector.reDim(_solver.nCols());
   bool initSolveFromScratch = true;


   /************************/
   /* Initialisation phase */
   /************************/

   // arrays to store the basis status for the rows and columns at the interruption of the original problem solve.
   DataArray< typename SPxSolverBase<R>::VarStatus > basisStatusRows;
   DataArray< typename SPxSolverBase<R>::VarStatus > basisStatusCols;

   // since the original LP may have been shifted, the dual multiplier will not be correct for the original LP. This
   // loop will recheck the degeneracy level and compute the proper dual multipliers.
   do
   {
      // solving the instance.
      _decompSimplifyAndSolve(_solver, _slufactor, initSolveFromScratch, initSolveFromScratch);
      initSolveFromScratch = false;

      // checking whether the initialisation must terminate and the original problem is solved using the dual simplex.
      if(_solver.type() == SPxSolverBase<R>::LEAVE || _solver.status() >= SPxSolverBase<R>::OPTIMAL
            || _solver.status() == SPxSolverBase<R>::ABORT_EXDECOMP || numDegenCheck > MAX_DEGENCHECK)
      {
         // decomposition is deemed not useful. Solving the original problem using regular SoPlexBase.

         // returning the sense to minimise
         if(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE)
         {
            assert(_solver.spxSense() == SPxLPBase<R>::MAXIMIZE);

            _solver.changeObj(-(_solver.maxObj()));
            _solver.changeSense(SPxLPBase<R>::MINIMIZE);
         }

         // switching off the starting basis check. This is only required to initialise the decomposition simplex.
         _solver.setDecompStatus(SPxSolverBase<R>::DONTFINDSTARTBASIS);

         // the basis is not computed correctly is the problem was unfeasible or unbounded.
         if(_solver.status() == SPxSolverBase<R>::UNBOUNDED
               || _solver.status() == SPxSolverBase<R>::INFEASIBLE)
            _hasBasis = false;


         // resolving the problem to update the real lp and solve with the correct objective.
         // TODO: With some infeasible problem (e.g. refinery) the dual is violated. Setting fromScratch to true
         // avoids this issue. Need to check what the problem is.
         // @TODO: Should this be _preprocessAndSolveReal similar to the resolve at the end of the algorithm?
         _decompSimplifyAndSolve(_solver, _slufactor, true, false);

         // retreiving the original problem statistics prior to destroying it.
         getOriginalProblemStatistics();

         // storing the solution from the reduced problem
         _storeSolutionReal();

         // stop timing
         _statistics->solvingTime->stop();
         return;
      }
      else if(_solver.status() == SPxSolverBase<R>::ABORT_TIME
              || _solver.status() == SPxSolverBase<R>::ABORT_ITER
              || _solver.status() == SPxSolverBase<R>::ABORT_VALUE)
      {
         // This cleans up the problem is an abort is reached.

         // at this point, the _decompSimplifyAndSolve does not store the realLP. It stores the _decompLP. As such, it is
         // important to reinstall the _realLP to the _solver.
         if(!_isRealLPLoaded)
         {
            _solver.loadLP(*_decompLP);
            spx_free(_decompLP);
            _decompLP = &_solver;
            _isRealLPLoaded = true;
         }

         // returning the sense to minimise
         if(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE)
         {
            assert(_solver.spxSense() == SPxLPBase<R>::MAXIMIZE);

            _solver.changeObj(-(_solver.maxObj()));
            _solver.changeSense(SPxLPBase<R>::MINIMIZE);
         }

         // resolving the problem to set up all of the solution vectors. This is required because data from the
         // initialisation solve may remain at termination causing an infeasible solution to be reported.
         _preprocessAndSolveReal(false);

         // storing the solution from the reduced problem
         _storeSolutionReal();

         // stop timing
         _statistics->solvingTime->stop();
         return;
      }

      // checking the degeneracy level
      _solver.basis().solve(_decompFeasVector, _solver.maxObj());
      degeneracyLevel = _solver.getDegeneracyLevel(_decompFeasVector);

      _solver.setDegenCompOffset(DEGENCHECK_OFFSET);

      numDegenCheck++;
   }
   while((degeneracyLevel > 0.9 || degeneracyLevel < 0.1)
         || !checkBasisDualFeasibility(_decompFeasVector));

   // decomposition simplex will commence, so the start basis does not need to be found
   _solver.setDecompStatus(SPxSolverBase<R>::DONTFINDSTARTBASIS);

   // if the original problem has a basis, this will be stored
   if(_hasBasis)
   {
      basisStatusRows.reSize(numRows());
      basisStatusCols.reSize(numCols());
      _solver.getBasis(basisStatusRows.get_ptr(), basisStatusCols.get_ptr(), basisStatusRows.size(),
                       basisStatusCols.size());
   }

   // updating the algorithm iterations statistic
   _statistics->callsReducedProb++;

   // setting the statistic information
   numDecompIter = 0;
   numRedProbIter = _solver.iterations();
   numCompProbIter = 0;

   // setting the display information
   _decompDisplayLine = 0;

   /************************/
   /* Decomposition phase  */
   /************************/

   MSG_INFO1(spxout,
             spxout << "========      Degeneracy Detected       ========" << std::endl
             << std::endl
             << "======== Commencing decomposition solve ========" << std::endl
            );

   //spxout.setVerbosity( SPxOut::DEBUG );
   MSG_INFO2(spxout, spxout << "Creating the Reduced and Complementary problems." << std::endl);

   // setting the verbosity level
   const SPxOut::Verbosity orig_verbosity = spxout.getVerbosity();
   const SPxOut::Verbosity decomp_verbosity = (SPxOut::Verbosity)intParam(
            SoPlexBase<R>::DECOMP_VERBOSITY);

   if(decomp_verbosity < orig_verbosity)
      spxout.setVerbosity(decomp_verbosity);

   // creating copies of the original problem that will be manipulated to form the reduced and complementary
   // problems.
   _createDecompReducedAndComplementaryProblems();

   // creating the initial reduced problem from the basis information
   _formDecompReducedProblem(stop);

   // setting flags for the decomposition solve
   _hasBasis = false;
   bool hasRedBasis = false;
   bool redProbError = false;
   bool noRedprobIter = false;
   bool explicitviol = boolParam(SoPlexBase<R>::EXPLICITVIOL);
   int algIterCount = 0;


   // a stop will be triggered if the reduced problem exceeded a specified fraction of the time limit
   if(stop)
   {
      MSG_INFO1(spxout, spxout << "==== Error constructing the reduced problem ====" << std::endl);
      redProbError = true;
      _statistics->redProbStatus = SPxSolverBase<R>::NOT_INIT;
   }

   // the main solving loop of the decomposition simplex.
   // This loop solves the Reduced problem, and if the problem is feasible, the complementary problem is solved.
   while(!stop)
   {
      int previter = _statistics->iterations;

      // setting the current solving mode.
      _currentProb = DECOMP_RED;

      // solve the reduced problem

      MSG_INFO2(spxout,
                spxout << std::endl
                << "=========================" << std::endl
                << "Solving: Reduced Problem." << std::endl
                << "=========================" << std::endl
                << std::endl);

      _hasBasis = hasRedBasis;
      // solving the reduced problem
      _decompSimplifyAndSolve(_solver, _slufactor, !algIterCount, !algIterCount);

      stop = decompTerminate(realParam(
                                SoPlexBase<R>::TIMELIMIT));  // checking whether the algorithm should terminate

      // updating the algorithm iterations statistics
      _statistics->callsReducedProb++;
      _statistics->redProbStatus = _solver.status();

      assert(_isRealLPLoaded);
      hasRedBasis = _hasBasis;
      _hasBasis = false;

      // checking the status of the reduced problem
      // It is expected that infeasibility and unboundedness will be found in the initialisation solve. If this is
      // not the case and the reduced problem is infeasible or unbounded the decomposition simplex will terminate.
      // If the decomposition simplex terminates, then the original problem will be solved using the stored basis.
      if(_solver.status() != SPxSolverBase<R>::OPTIMAL)
      {
         if(_solver.status() == SPxSolverBase<R>::UNBOUNDED)
            MSG_INFO2(spxout, spxout << "Unbounded reduced problem." << std::endl);

         if(_solver.status() == SPxSolverBase<R>::INFEASIBLE)
            MSG_INFO2(spxout, spxout << "Infeasible reduced problem." << std::endl);

         MSG_INFO2(spxout, spxout << "Reduced problem status: " << static_cast<int>
                   (_solver.status()) << std::endl);

         redProbError = true;
         break;
      }

      // as a final check, if no iterations were performed with the reduced problem, then the algorithm terminates
      if(_statistics->iterations == previter)
      {
         MSG_WARNING(spxout,
                     spxout << "WIMDSM02: reduced problem performed zero iterations. Terminating." << std::endl;);

         noRedprobIter = true;
         stop = true;
         break;
      }

      // printing display line
      printDecompDisplayLine(_solver, orig_verbosity, !algIterCount, !algIterCount);

      // get the dual solutions from the reduced problem
      VectorBase<R> reducedLPDualVector(_solver.nRows());
      VectorBase<R> reducedLPRedcostVector(_solver.nCols());
      _solver.getDualSol(reducedLPDualVector);
      _solver.getRedCostSol(reducedLPRedcostVector);


      // Using the solution to the reduced problem:
      // In the first iteration of the algorithm create the complementary problem.
      // In the subsequent iterations of the algorithm update the complementary problem.
      if(algIterCount == 0)
         _formDecompComplementaryProblem();
      else
      {
         if(boolParam(SoPlexBase<R>::USECOMPDUAL))
            _updateDecompComplementaryDualProblem(false);
         else
            _updateDecompComplementaryPrimalProblem(false);
      }

      // setting the current solving mode.
      _currentProb = DECOMP_COMP;

      // solve the complementary problem
      MSG_INFO2(spxout,
                spxout << std::endl
                << "=========================" << std::endl
                << "Solving: Complementary Problem." << std::endl
                << "=========================" << std::endl
                << std::endl);

      if(!explicitviol)
      {
         //_compSolver.writeFileLPBase("comp.lp");

         _decompSimplifyAndSolve(_compSolver, _compSlufactor, true, true);

         MSG_INFO2(spxout, spxout << "Iteration " << algIterCount
                   << "Objective Value: " << std::setprecision(10) << _compSolver.objValue()
                   << std::endl);
      }


      assert(_isRealLPLoaded);


      // Check whether the complementary problem is solved with a non-negative objective function, is infeasible or
      // unbounded. If true, then stop the algorithm.
      if(!explicitviol && (GE(_compSolver.objValue(), R(0.0), R(1e-20))
                           || _compSolver.status() == SPxSolverBase<R>::INFEASIBLE
                           || _compSolver.status() == SPxSolverBase<R>::UNBOUNDED))
      {
         _statistics->compProbStatus = _compSolver.status();
         _statistics->finalCompObj = _compSolver.objValue();

         if(_compSolver.status() == SPxSolverBase<R>::UNBOUNDED)
            MSG_INFO2(spxout, spxout << "Unbounded complementary problem." << std::endl);

         if(_compSolver.status() == SPxSolverBase<R>::INFEASIBLE)
            MSG_INFO2(spxout, spxout << "Infeasible complementary problem." << std::endl);

         if(_compSolver.status() == SPxSolverBase<R>::INFEASIBLE
               || _compSolver.status() == SPxSolverBase<R>::UNBOUNDED)
            explicitviol = true;

         stop = true;
      }

      if(!stop && !explicitviol)
      {
         // get the primal solutions from the complementary problem
         VectorBase<R> compLPPrimalVector(_compSolver.nCols());
         _compSolver.getPrimalSol(compLPPrimalVector);

         // get the dual solutions from the complementary problem
         VectorBase<R> compLPDualVector(_compSolver.nRows());
         _compSolver.getDualSol(compLPDualVector);

         // updating the reduced problem
         _updateDecompReducedProblem(_compSolver.objValue(), reducedLPDualVector, reducedLPRedcostVector,
                                     compLPPrimalVector, compLPDualVector);
      }
      // if the complementary problem is infeasible or unbounded, it is possible that the algorithm can continue.
      // a check of the original problem is required to determine whether there are any violations.
      else if(_compSolver.status() == SPxSolverBase<R>::INFEASIBLE
              || _compSolver.status() == SPxSolverBase<R>::UNBOUNDED
              || explicitviol)
      {
         // getting the primal vector from the reduced problem
         VectorBase<R> reducedLPPrimalVector(_solver.nCols());
         _solver.getPrimalSol(reducedLPPrimalVector);

         // checking the optimality of the reduced problem solution with the original problem
         _checkOriginalProblemOptimality(reducedLPPrimalVector, true);

         // if there are any violated rows or bounds then stop is reset and the algorithm continues.
         if(_nDecompViolBounds > 0 || _nDecompViolRows > 0)
            stop = false;

         if(_nDecompViolBounds == 0 && _nDecompViolRows == 0)
            stop = true;

         // updating the reduced problem with the original problem violated rows
         if(!stop)
            _updateDecompReducedProblemViol(false);
      }


      // =============================================================================
      // Code check completed up to here
      // =============================================================================

      numDecompIter++;
      algIterCount++;
   }

   // if there is an error in solving the reduced problem, i.e. infeasible or unbounded, then we leave the
   // decomposition solve and resolve the original. Infeasibility should be dealt with in the original problem.
   if(!redProbError)
   {
#ifndef NDEBUG
      // computing the solution for the original variables
      VectorBase<R> reducedLPPrimalVector(_solver.nCols());
      _solver.getPrimalSol(reducedLPPrimalVector);

      // checking the optimality of the reduced problem solution with the original problem
      _checkOriginalProblemOptimality(reducedLPPrimalVector, true);

      // resetting the verbosity level
      spxout.setVerbosity(orig_verbosity);
#endif

      // This is an additional check to ensure that the complementary problem is solving correctly.
      // This is only used for debugging
#ifdef PERFORM_COMPPROB_CHECK

      // solving the complementary problem with the original objective function
      if(boolParam(SoPlexBase<R>::USECOMPDUAL))
         _setComplementaryDualOriginalObjective();
      else
         _setComplementaryPrimalOriginalObjective();

      // for clean up
      // the real lp has to be set to not loaded.
      //_isRealLPLoaded = false;
      // the basis has to be set to false
      _hasBasis = false;

      if(boolParam(SoPlexBase<R>::USECOMPDUAL))
      {
         SPxLPBase<R> compDualLP;
         _compSolver.buildDualProblem(compDualLP, _decompPrimalRowIDs.get_ptr(),
                                      _decompPrimalColIDs.get_ptr(),
                                      _decompDualRowIDs.get_ptr(), _decompDualColIDs.get_ptr(), &_nPrimalRows, &_nPrimalCols, &_nDualRows,
                                      &_nDualCols);

         _compSolver.loadLP(compDualLP);
      }

      _decompSimplifyAndSolve(_compSolver, _compSlufactor, true, true);

      _solReal._hasPrimal = true;
      _hasSolReal = true;
      // get the primal solutions from the reduced problem
      VectorBase<R> testPrimalVector(_compSolver.nCols());
      _compSolver.getPrimalSol(testPrimalVector);
      _solReal._primal.reDim(_compSolver.nCols());
      _solReal._primal = testPrimalVector;

      R maxviol = 0;
      R sumviol = 0;

      // Since the original objective value has now been installed in the complementary problem, the solution will be
      // a solution to the original problem.
      // checking the bound violation of the solution from  complementary problem
      if(getDecompBoundViolation(maxviol, sumviol))
         MSG_INFO1(spxout, spxout << "Bound violation - "
                   << "Max: " << std::setprecision(20) << maxviol << " "
                   << "Sum: " << sumviol << std::endl);

      // checking the row violation of the solution from the complementary problem
      if(getDecompRowViolation(maxviol, sumviol))
         MSG_INFO1(spxout, spxout << "Row violation - "
                   << "Max: " << std::setprecision(21) << maxviol << " "
                   << "Sum: " << sumviol << std::endl);

      MSG_INFO1(spxout, spxout << "Objective Value: " << _compSolver.objValue() << std::endl);
#endif
   }

   // resetting the verbosity level
   spxout.setVerbosity(orig_verbosity);

   MSG_INFO1(spxout,
             spxout << "========  Decomposition solve completed ========" << std::endl
             << std::endl
             << "========   Resolving original problem   ========" << std::endl
            );

   // if there is a reduced problem error in the first iteration the complementary problme has not been
   // set up. In this case no memory has been allocated for _decompCompProbColIDsIdx and _fixedOrigVars.
   if((!redProbError && !noRedprobIter) || algIterCount > 0)
   {
      spx_free(_decompCompProbColIDsIdx);
      spx_free(_fixedOrigVars);
   }

   spx_free(_decompViolatedRows);
   spx_free(_decompViolatedBounds);
   spx_free(_decompReducedProbCols);
   spx_free(_decompReducedProbRows);

   // retreiving the original problem statistics prior to destroying it.
   getOriginalProblemStatistics();

   // setting the reduced problem statistics
   _statistics->numRedProbRows = numIncludedRows;
   _statistics->numRedProbCols = _solver.nCols();


   // printing display line for resolve of problem
   _solver.printDisplayLine(false, true);

   // if there is an error solving the reduced problem the LP must be solved from scratch
   if(redProbError)
   {
      MSG_INFO1(spxout, spxout << "=== Reduced problem error - Solving original ===" << std::endl);

      // the solver is loaded with the realLP to solve the problem from scratch
      _solver.loadLP(*_realLP);
      spx_free(_realLP);
      _realLP = &_solver;
      _isRealLPLoaded = true;

      // returning the sense to minimise
      if(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE)
      {
         assert(_solver.spxSense() == SPxLPBase<R>::MAXIMIZE);

         _solver.changeObj(-(_solver.maxObj()));
         _solver.changeSense(SPxLPBase<R>::MINIMIZE);

         // Need to add commands to multiply the objective solution values by -1
      }

      //_solver.setBasis(basisStatusRows.get_const_ptr(), basisStatusCols.get_const_ptr());
      _preprocessAndSolveReal(true);
   }
   else
   {
      // resolving the problem to update the real lp and solve with the correct objective.
      // the realLP is updated with the current solver. This is to resolve the problem to get the correct solution
      // values
      _realLP->~SPxLPBase<R>();
      spx_free(_realLP);
      _realLP = &_solver;
      _isRealLPLoaded = true;

      // returning the sense to minimise
      if(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE)
      {
         assert(_solver.spxSense() == SPxLPBase<R>::MAXIMIZE);

         _solver.changeObj(-(_solver.maxObj()));
         _solver.changeSense(SPxLPBase<R>::MINIMIZE);

         // Need to add commands to multiply the objective solution values by -1
      }

      _preprocessAndSolveReal(false);
   }

   // stop timing
   _statistics->solvingTime->stop();
}



/// creating copies of the original problem that will be manipulated to form the reduced and complementary problems
template <class R>
void SoPlexBase<R>::_createDecompReducedAndComplementaryProblems()
{
   // the reduced problem is formed from the current problem
   // So, we copy the _solver to the _realLP and work on the _solver
   // NOTE: there is no need to preprocess because we always have a starting basis.
   _realLP = nullptr;
   spx_alloc(_realLP);
   _realLP = new(_realLP) SPxLPBase<R>(_solver);

   // allocating memory for the reduced problem rows and cols flag array
   _decompReducedProbRows = nullptr;
   spx_alloc(_decompReducedProbRows, numRows());
   _decompReducedProbCols = nullptr;
   spx_alloc(_decompReducedProbCols, numCols());

   // the complementary problem is formulated with all incompatible rows and those from the reduced problem that have
   // a positive reduced cost.
   _compSolver = _solver;
   _compSolver.setOutstream(spxout);
   _compSolver.setBasisSolver(&_compSlufactor);

   // allocating memory for the violated bounds and rows arrays
   _decompViolatedBounds = nullptr;
   _decompViolatedRows = nullptr;
   spx_alloc(_decompViolatedBounds, numCols());
   spx_alloc(_decompViolatedRows, numRows());
   _nDecompViolBounds = 0;
   _nDecompViolRows = 0;
}



/// forms the reduced problem
template <class R>
void SoPlexBase<R>::_formDecompReducedProblem(bool& stop)
{
   MSG_INFO2(spxout, spxout << "Forming the Reduced problem" << std::endl);
   int* nonposind = 0;
   int* compatind = 0;
   int* rowsforremoval = 0;
   int* colsforremoval = 0;
   int nnonposind = 0;
   int ncompatind = 0;

   assert(_solver.nCols() == numCols());
   assert(_solver.nRows() == numRows());

   // capturing the basis used for the transformation
   _decompTransBasis = _solver.basis();

   // setting row counter to zero
   numIncludedRows = 0;

   // the _decompLP is used as a helper LP object. This is used so that _realLP can still hold the original problem.
   _decompLP = 0;
   spx_alloc(_decompLP);
   _decompLP = new(_decompLP) SPxLPBase<R>(_solver);

   // retreiving the basis information
   _basisStatusRows.reSize(numRows());
   _basisStatusCols.reSize(numCols());
   _solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());

   // get the indices of the rows with positive dual multipliers and columns with positive reduced costs.
   spx_alloc(nonposind, numCols());
   spx_alloc(colsforremoval, numCols());

   if(!stop)
      _getZeroDualMultiplierIndices(_decompFeasVector, nonposind, colsforremoval, &nnonposind, stop);

   // get the compatible columns from the constraint matrix w.r.t the current basis matrix
   MSG_INFO2(spxout, spxout << "Computing the compatible columns" << std::endl
             << "Solving time: " << solveTime() << std::endl);

   spx_alloc(compatind, _solver.nRows());
   spx_alloc(rowsforremoval, _solver.nRows());

   if(!stop)
      _getCompatibleColumns(_decompFeasVector, nonposind, compatind, rowsforremoval, colsforremoval,
                            nnonposind,
                            &ncompatind, true, stop);

   int* compatboundcons = 0;
   int ncompatboundcons = 0;
   spx_alloc(compatboundcons, numCols());

   LPRowSetBase<R> boundcons;

   // identifying the compatible bound constraints
   if(!stop)
      _getCompatibleBoundCons(boundcons, compatboundcons, nonposind, &ncompatboundcons, nnonposind, stop);

   // delete rows and columns from the LP to form the reduced problem
   MSG_INFO2(spxout, spxout << "Deleting rows and columns to form the reduced problem" << std::endl
             << "Solving time: " << solveTime() << std::endl);

   // allocating memory to add bound constraints
   SPxRowId* addedrowids = 0;
   spx_alloc(addedrowids, ncompatboundcons);

   // computing the reduced problem obj coefficient vector and updating the problem
   if(!stop)
   {
      _computeReducedProbObjCoeff(stop);

      _solver.loadLP(*_decompLP);

      // the colsforremoval are the columns with a zero reduced cost.
      // the rowsforremoval are the rows identified as incompatible.
      _solver.removeRows(rowsforremoval);
      // adding the rows for the compatible bound constraints
      _solver.addRows(addedrowids, boundcons);

      for(int i = 0; i < ncompatboundcons; i++)
         _decompReducedProbColRowIDs[compatboundcons[i]] = addedrowids[i];
   }


   // freeing allocated memory
   spx_free(addedrowids);
   spx_free(compatboundcons);
   spx_free(rowsforremoval);
   spx_free(compatind);
   spx_free(colsforremoval);
   spx_free(nonposind);

   _decompLP->~SPxLPBase<R>();
   spx_free(_decompLP);
}



/// forms the complementary problem
template <class R>
void SoPlexBase<R>::_formDecompComplementaryProblem()
{
   // delete rows and columns from the LP to form the reduced problem
   _nElimPrimalRows = 0;
   _decompElimPrimalRowIDs.reSize(
      _realLP->nRows());   // the number of eliminated rows is less than the number of rows
   // in the reduced problem
   _decompCompProbColIDsIdx = 0;
   spx_alloc(_decompCompProbColIDsIdx, _realLP->nCols());

   _fixedOrigVars = 0;
   spx_alloc(_fixedOrigVars, _realLP->nCols());

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      _decompCompProbColIDsIdx[i] = -1;
      _fixedOrigVars[i] = -2; // this must be set to a value differet from -1, 0, 1 to ensure the
      // _updateComplementaryFixedPrimalVars function updates all variables in the problem.
   }

   int naddedrows = 0;
   DataArray < SPxRowId > rangedRowIds(numRows());
   _deleteAndUpdateRowsComplementaryProblem(rangedRowIds.get_ptr(), naddedrows);

   if(boolParam(
            SoPlexBase<R>::USECOMPDUAL))     // if we use the dual formulation of the complementary problem, we must
      // perform the transformation
   {
      // initialising the arrays to store the row id's from the primal and the col id's from the dual
      _decompPrimalRowIDs.reSize(3 * _compSolver.nRows());
      _decompPrimalColIDs.reSize(3 * _compSolver.nCols());
      _decompDualRowIDs.reSize(3 * _compSolver.nCols());
      _decompDualColIDs.reSize(3 * _compSolver.nRows());
      _nPrimalRows = 0;
      _nPrimalCols = 0;
      _nDualRows = 0;
      _nDualCols = 0;

      // convert complementary problem to dual problem
      SPxLPBase<R> compDualLP;
      _compSolver.buildDualProblem(compDualLP, _decompPrimalRowIDs.get_ptr(),
                                   _decompPrimalColIDs.get_ptr(),
                                   _decompDualRowIDs.get_ptr(), _decompDualColIDs.get_ptr(), &_nPrimalRows, &_nPrimalCols, &_nDualRows,
                                   &_nDualCols);
      compDualLP.setOutstream(spxout);

      // setting the id index array for the complementary problem
      for(int i = 0; i < _nPrimalRows; i++)
      {
         // a primal row may result in two dual columns. So the index array points to the first of the indicies for the
         // primal row.
         if(i + 1 < _nPrimalRows &&
               _realLP->number(SPxRowId(_decompPrimalRowIDs[i])) == _realLP->number(SPxRowId(
                        _decompPrimalRowIDs[i + 1])))
            i++;
      }

      for(int i = 0; i < _nPrimalCols; i++)
      {
         if(_decompPrimalColIDs[i].getIdx() != _compSlackColId.getIdx())
            _decompCompProbColIDsIdx[_realLP->number(_decompPrimalColIDs[i])] = i;
      }

      // retrieving the dual row id for the complementary slack column
      // there should be a one to one relationship between the number of primal columns and the number of dual rows.
      // hence, it should be possible to equate the dual row id to the related primal column.
      assert(_nPrimalCols == _nDualRows);
      assert(_compSolver.nCols() == compDualLP.nRows());
      _compSlackDualRowId = compDualLP.rId(_compSolver.number(_compSlackColId));

      _compSolver.loadLP(compDualLP);

      _compSolver.init();

      _updateComplementaryDualSlackColCoeff();

      // initalising the array for the dual columns representing the fixed variables in the complementary problem.
      _decompFixedVarDualIDs.reSize(_realLP->nCols());
      _decompVarBoundDualIDs.reSize(2 * _realLP->nCols());


      // updating the constraints for the complementary problem
      _updateDecompComplementaryDualProblem(false);
   }
   else  // when using the primal of the complementary problem, no transformation of the problem is required.
   {
      // initialising the arrays to store the row and column id's from the original problem the col id's from the dual
      // if a row or column is not included in the complementary problem, the ID is set to INVALID in the
      // _decompPrimalRowIDs or _decompPrimalColIDs respectively.
      _decompPrimalRowIDs.reSize(_compSolver.nRows() * 2);
      _decompPrimalColIDs.reSize(_compSolver.nCols());
      _decompCompPrimalRowIDs.reSize(_compSolver.nRows() * 2);
      _decompCompPrimalColIDs.reSize(_compSolver.nCols());
      _nPrimalRows = 0;
      _nPrimalCols = 0;

      // at this point in time the complementary problem contains all rows and columns from the original problem
      int addedrangedrows = 0;
      _nCompPrimalRows = 0;
      _nCompPrimalCols = 0;

      for(int i = 0; i < _realLP->nRows(); i++)
      {
         assert(_nCompPrimalRows < _compSolver.nRows());
         _decompPrimalRowIDs[_nPrimalRows] = _realLP->rId(i);
         _decompCompPrimalRowIDs[_nCompPrimalRows] = _compSolver.rId(i);
         _nPrimalRows++;
         _nCompPrimalRows++;

         if(_realLP->rowType(i) == LPRowBase<R>::RANGE || _realLP->rowType(i) == LPRowBase<R>::EQUAL)
         {
            assert(EQ(_compSolver.lhs(rangedRowIds[addedrangedrows]), _realLP->lhs(i)));
            assert(LE(_compSolver.rhs(rangedRowIds[addedrangedrows]), R(infinity)));
            assert(LE(_compSolver.lhs(i), R(-infinity)));
            assert(LT(_compSolver.rhs(i), R(infinity)));

            _decompPrimalRowIDs[_nPrimalRows] = _realLP->rId(i);
            _decompCompPrimalRowIDs[_nCompPrimalRows] = rangedRowIds[addedrangedrows];
            _nPrimalRows++;
            _nCompPrimalRows++;
            addedrangedrows++;
         }
      }

      for(int i = 0; i < _compSolver.nCols(); i++)
      {

         if(_compSolver.cId(i).getIdx() != _compSlackColId.getIdx())
         {
            _decompPrimalColIDs[i] = _realLP->cId(i);
            _decompCompPrimalColIDs[i] = _compSolver.cId(i);
            _nPrimalCols++;
            _nCompPrimalCols++;

            _decompCompProbColIDsIdx[_realLP->number(_decompPrimalColIDs[i])] = i;
         }
      }

      // updating the constraints for the complementary problem
      _updateDecompComplementaryPrimalProblem(false);
   }
}



/// simplifies the problem and solves
template <class R>
void SoPlexBase<R>::_decompSimplifyAndSolve(SPxSolverBase<R>& solver, SLUFactor<R>& sluFactor,
      bool fromScratch, bool applyPreprocessing)
{
   if(realParam(SoPlexBase<R>::TIMELIMIT) < realParam(SoPlexBase<R>::INFTY))
      solver.setTerminationTime(Real(realParam(SoPlexBase<R>::TIMELIMIT)) -
                                _statistics->solvingTime->time());

   solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
   _statistics->preprocessingTime->start();

   typename SPxSimplifier<R>::Result result = SPxSimplifier<R>::OKAY;

   /* delete starting basis if solving from scratch */
   if(fromScratch)
   {
      try
      {
         solver.reLoad();
      }
      catch(const SPxException& E)
      {
         MSG_ERROR(spxout << "Caught exception <" << E.what() << "> during simplify and solve.\n");
      }
   }

   //assert(!fromScratch || solver.status() == SPxSolverBase<R>::NO_PROBLEM);

   if(applyPreprocessing)
   {
      _enableSimplifierAndScaler();
      solver.setTerminationValue(realParam(SoPlexBase<R>::INFTY));
   }
   else
   {
      _disableSimplifierAndScaler();
      ///@todo implement for both objective senses
      solver.setTerminationValue(solver.spxSense() == SPxLPBase<R>::MINIMIZE
                                 ? realParam(SoPlexBase<R>::OBJLIMIT_UPPER) : realParam(SoPlexBase<R>::INFTY));
   }

   /* store original lp */
   applyPreprocessing = (_scaler != nullptr || _simplifier != NULL);

   // @TODO The use of _isRealLPLoaded is not correct. An additional parameter would be useful for this algorithm.
   // Maybe a parameter _isDecompLPLoaded?
   if(_isRealLPLoaded)
   {
      //assert(_decompLP == &solver); // there should be an assert here, but I don't know what to use.

      // preprocessing is always applied to the LP in the solver; hence we have to create a copy of the original LP
      // if preprocessing is turned on
      if(applyPreprocessing)
      {
         _decompLP = 0;
         spx_alloc(_decompLP);
         _decompLP = new(_decompLP) SPxLPBase<R>(solver);
         _isRealLPLoaded = false;
      }
      else
         _decompLP = &solver;
   }
   else
   {
      assert(_decompLP != &solver);

      // ensure that the solver has the original problem
      solver.loadLP(*_decompLP);

      // load basis if available
      if(_hasBasis)
      {
         assert(_basisStatusRows.size() == solver.nRows());
         assert(_basisStatusCols.size() == solver.nCols());

         ///@todo this should not fail even if the basis is invalid (wrong dimension or wrong number of basic
         ///      entries); fix either in SPxSolverBase or in SPxBasisBase
         solver.setBasis(_basisStatusRows.get_const_ptr(), _basisStatusCols.get_const_ptr());
      }

      // if there is no preprocessing, then the original and the transformed problem are identical and it is more
      // memory-efficient to keep only the problem in the solver
      if(!applyPreprocessing)
      {
         _decompLP->~SPxLPBase<R>();
         spx_free(_decompLP);
         _decompLP = &solver;
         _isRealLPLoaded = true;
      }
      else
      {
         _decompLP = 0;
         spx_alloc(_decompLP);
         _decompLP = new(_decompLP) SPxLPBase<R>(solver);
      }
   }

   // assert that we have two problems if and only if we apply preprocessing
   assert(_decompLP == &solver || applyPreprocessing);
   assert(_decompLP != &solver || !applyPreprocessing);

   // apply problem simplification
   if(_simplifier != 0)
   {
      Real remainingTime = _solver.getMaxTime() - _solver.time();
      result = _simplifier->simplify(solver, realParam(SoPlexBase<R>::EPSILON_ZERO),
                                     realParam(SoPlexBase<R>::FEASTOL),
                                     realParam(SoPlexBase<R>::OPTTOL),
                                     remainingTime);
      solver.changeObjOffset(_simplifier->getObjoffset() + realParam(SoPlexBase<R>::OBJ_OFFSET));
   }

   _statistics->preprocessingTime->stop();

   // run the simplex method if problem has not been solved by the simplifier
   if(result == SPxSimplifier<R>::OKAY)
   {
      if(_scaler != nullptr)
         _scaler->scale(solver);

      bool _hadBasis = _hasBasis;

      _statistics->simplexTime->start();

      try
      {
         solver.solve();
      }
      catch(const SPxException& E)
      {
         MSG_ERROR(std::cerr << "Caught exception <" << E.what() << "> while solving real LP.\n");
         _status = SPxSolverBase<R>::ERROR;
      }
      catch(...)
      {
         MSG_ERROR(std::cerr << "Caught unknown exception while solving real LP.\n");
         _status = SPxSolverBase<R>::ERROR;
      }

      _statistics->simplexTime->stop();

      // record statistics
      // only record the main statistics for the original problem and reduced problem.
      // the complementary problem is a pivoting rule, so these will be held in other statistics.
      // statistics on the number of iterations for each problem is stored in individual variables.
      if(_currentProb == DECOMP_ORIG || _currentProb == DECOMP_RED)
      {
         _statistics->iterations += solver.iterations();
         _statistics->iterationsPrimal += solver.primalIterations();
         _statistics->iterationsFromBasis += _hadBasis ? solver.iterations() : 0;
         _statistics->boundflips += solver.boundFlips();
         _statistics->luFactorizationTimeReal += sluFactor.getFactorTime();
         _statistics->luSolveTimeReal += sluFactor.getSolveTime();
         _statistics->luFactorizationsReal += sluFactor.getFactorCount();
         _statistics->luSolvesReal += sluFactor.getSolveCount();
         sluFactor.resetCounters();

         _statistics->degenPivotsPrimal += solver.primalDegeneratePivots();
         _statistics->degenPivotsDual += solver.dualDegeneratePivots();

         _statistics->sumDualDegen += _solver.sumDualDegeneracy();
         _statistics->sumPrimalDegen += _solver.sumPrimalDegeneracy();

         if(_currentProb == DECOMP_ORIG)
            _statistics->iterationsInit += solver.iterations();
         else
            _statistics->iterationsRedProb += solver.iterations();
      }

      if(_currentProb == DECOMP_COMP)
         _statistics->iterationsCompProb += solver.iterations();

   }

   // check the result and run again without preprocessing if necessary
   _evaluateSolutionDecomp(solver, sluFactor, result);
}



/// loads original problem into solver and solves again after it has been solved to optimality with preprocessing
template <class R>
void SoPlexBase<R>::_decompResolveWithoutPreprocessing(SPxSolverBase<R>& solver,
      SLUFactor<R>& sluFactor, typename SPxSimplifier<R>::Result result)
{
   assert(_simplifier != 0 || _scaler != nullptr);
   assert(result == SPxSimplifier<R>::VANISHED
          || (result == SPxSimplifier<R>::OKAY
              && (solver.status() == SPxSolverBase<R>::OPTIMAL
                  || solver.status() == SPxSolverBase<R>::ABORT_DECOMP
                  || solver.status() == SPxSolverBase<R>::ABORT_EXDECOMP)));

   // if simplifier is active and LP is solved in presolving or to optimality, then we unsimplify to get the basis
   if(_simplifier != 0)
   {
      assert(!_simplifier->isUnsimplified());

      bool vanished = result == SPxSimplifier<R>::VANISHED;

      // get solution vectors for transformed problem
      VectorBase<R> primal(vanished ? 0 : solver.nCols());
      VectorBase<R> slacks(vanished ? 0 : solver.nRows());
      VectorBase<R> dual(vanished ? 0 : solver.nRows());
      VectorBase<R> redCost(vanished ? 0 : solver.nCols());

      assert(!_isRealLPLoaded);
      _basisStatusRows.reSize(_decompLP->nRows());
      _basisStatusCols.reSize(_decompLP->nCols());
      assert(vanished || _basisStatusRows.size() >= solver.nRows());
      assert(vanished || _basisStatusCols.size() >= solver.nCols());

      if(!vanished)
      {
         assert(solver.status() == SPxSolverBase<R>::OPTIMAL
                || solver.status() == SPxSolverBase<R>::ABORT_DECOMP
                || solver.status() == SPxSolverBase<R>::ABORT_EXDECOMP);

         solver.getPrimalSol(primal);
         solver.getSlacks(slacks);
         solver.getDualSol(dual);
         solver.getRedCostSol(redCost);

         // unscale vectors
         if(_scaler && solver.isScaled())
         {
            _scaler->unscalePrimal(solver, primal);
            _scaler->unscaleSlacks(solver, slacks);
            _scaler->unscaleDual(solver, dual);
            _scaler->unscaleRedCost(solver, redCost);
         }

         // get basis of transformed problem
         solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());
      }

      try
      {
         _simplifier->unsimplify(primal, dual, slacks, redCost, _basisStatusRows.get_ptr(),
                                 _basisStatusCols.get_ptr(), solver.status() == SPxSolverBase<R>::OPTIMAL);
         _simplifier->getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());
         _hasBasis = true;
      }
      catch(const SPxException& E)
      {
         MSG_ERROR(spxout << "Caught exception <" << E.what() <<
                   "> during unsimplification. Resolving without simplifier and scaler.\n");
      }
      catch(...)
      {
         MSG_ERROR(spxout <<
                   "Caught unknown exception during unsimplification. Resolving without simplifier and scaler.\n");
         _status = SPxSolverBase<R>::ERROR;
      }
   }
   // if the original problem is not in the solver because of scaling, we also need to store the basis
   else if(_scaler != nullptr)
   {
      _basisStatusRows.reSize(numRows());
      _basisStatusCols.reSize(numCols());
      assert(_basisStatusRows.size() == solver.nRows());
      assert(_basisStatusCols.size() == solver.nCols());

      solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());
      _hasBasis = true;
   }

   // resolve the original problem
   _decompSimplifyAndSolve(solver, sluFactor, false, false);
   return;
}


/// updates the reduced problem with additional rows using the solution to the complementary problem
template <class R>
void SoPlexBase<R>::_updateDecompReducedProblem(R objValue, VectorBase<R> dualVector,
      VectorBase<R> redcostVector,
      VectorBase<R> compPrimalVector, VectorBase<R> compDualVector)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   R maxDualRatio = R(infinity);

   bool usecompdual = boolParam(SoPlexBase<R>::USECOMPDUAL);

   for(int i = 0; i < _nPrimalRows; i++)
   {
      R reducedProbDual = 0;
      R compProbPrimal = 0;
      R dualRatio = 0;
      int rownumber = _realLP->number(SPxRowId(_decompPrimalRowIDs[i]));

      if(_decompReducedProbRows[rownumber] && _solver.isBasic(_decompReducedProbRowIDs[rownumber]))
      {
         int solverRowNum = _solver.number(_decompReducedProbRowIDs[rownumber]);

         // retreiving the reduced problem dual solutions and the complementary problem primal solutions
         reducedProbDual = dualVector[solverRowNum]; // this is y

         if(usecompdual)
            compProbPrimal = compPrimalVector[_compSolver.number(SPxColId(_decompDualColIDs[i]))]; // this is u
         else
            compProbPrimal = compDualVector[_compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i]))];

         // the variable in the basis is degenerate.
         if(EQ(reducedProbDual, R(0.0), feastol))
         {
            MSG_WARNING(spxout,
                        spxout << "WIMDSM01: reduced problem dual value is very close to zero." << std::endl;);
            continue;
         }

         // the translation of the complementary primal problem to the dual some rows resulted in two columns.
         if(usecompdual && i < _nPrimalRows - 1 &&
               _realLP->number(SPxRowId(_decompPrimalRowIDs[i])) == _realLP->number(SPxRowId(
                        _decompPrimalRowIDs[i + 1])))
         {
            i++;
            compProbPrimal += compPrimalVector[_compSolver.number(SPxColId(_decompDualColIDs[i]))]; // this is u
         }



         // updating the ratio
         SoPlexBase<R>::DualSign varSign = getExpectedDualVariableSign(solverRowNum);

         if(varSign == SoPlexBase<R>::IS_FREE || (varSign == SoPlexBase<R>::IS_POS
               && LE(compProbPrimal, (R)0, feastol)) ||
               (varSign == SoPlexBase<R>::IS_NEG && GE(compProbPrimal, (R)0, feastol)))
         {
            dualRatio = R(infinity);
         }
         else
         {
            dualRatio = reducedProbDual / compProbPrimal;
         }

         if(LT(dualRatio, maxDualRatio, feastol))
            maxDualRatio = dualRatio;
      }
      else
      {
         if(usecompdual)
            compProbPrimal = compPrimalVector[_compSolver.number(SPxColId(_decompDualColIDs[i]))]; // this is v
         else
            compProbPrimal = compDualVector[_compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i]))];
      }

   }

   // setting the maxDualRatio to a maximum of 1
   if(maxDualRatio > 1.0)
      maxDualRatio = 1.0;

   VectorBase<R> compProbRedcost(
      _compSolver.nCols());   // the reduced costs of the complementary problem

   // Retrieving the reduced costs for each original problem row.
   _compSolver.getRedCostSol(compProbRedcost);

   LPRowSetBase<R> updaterows;


   // Identifying the violated rows
   Array<RowViolation> violatedrows;
   int nviolatedrows = 0;
   int* newrowidx = 0;
   int nnewrowidx = 0;
   spx_alloc(newrowidx, _nPrimalRows);

   violatedrows.reSize(_nPrimalRows);

   bool ratioTest = true;

   //ratioTest = false;
   for(int i = 0; i < _nPrimalRows; i++)
   {
      LPRowBase<R> origlprow;
      DSVectorBase<R> rowtoaddVec(_realLP->nCols());
      R compProbPrimal = 0;
      R compRowRedcost = 0;
      int rownumber = _realLP->number(SPxRowId(_decompPrimalRowIDs[i]));

      if(!_decompReducedProbRows[_realLP->number(SPxRowId(_decompPrimalRowIDs[i]))])
      {
         int compRowNumber;

         if(usecompdual)
         {
            compRowNumber = _compSolver.number(_decompDualColIDs[i]);
            // retreiving the complementary problem primal solutions
            compProbPrimal = compPrimalVector[compRowNumber]; // this is v
            compRowRedcost = compProbRedcost[compRowNumber];
         }
         else
         {
            compRowNumber = _compSolver.number(_decompCompPrimalRowIDs[i]);
            // retreiving the complementary problem primal solutions
            compProbPrimal = compDualVector[compRowNumber]; // this is v
         }

         // the translation of the complementary primal problem to the dual some rows resulted in two columns.
         if(usecompdual && i < _nPrimalRows - 1 &&
               _realLP->number(SPxRowId(_decompPrimalRowIDs[i])) == _realLP->number(SPxRowId(
                        _decompPrimalRowIDs[i + 1])))
         {
            i++;
            compRowNumber = _compSolver.number(SPxColId(_decompDualColIDs[i]));
            compProbPrimal += compPrimalVector[compRowNumber]; // this is v
            compRowRedcost += compProbRedcost[compRowNumber];
         }

         SoPlexBase<R>::DualSign varSign = getOrigProbDualVariableSign(rownumber);

         // add row to the reduced problem if the computed dual is of the correct sign for a feasible dual solution
         if(ratioTest && ((varSign == SoPlexBase<R>::IS_FREE && !isZero(compProbPrimal, feastol)) ||
                          (varSign == SoPlexBase<R>::IS_POS && GT(compProbPrimal * maxDualRatio, (R) 0, feastol)) ||
                          (varSign == SoPlexBase<R>::IS_NEG && LT(compProbPrimal * maxDualRatio, (R) 0, feastol))))
         {
            //this set of statements are required to add rows to the reduced problem. This will add a row for every
            //violated constraint. I only want to add a row for the most violated constraint. That is why the row
            //adding functionality is put outside the for loop.
            if(!_decompReducedProbRows[rownumber])
            {
               numIncludedRows++;
               assert(numIncludedRows <= _realLP->nRows());
            }

            violatedrows[nviolatedrows].idx = rownumber;
            violatedrows[nviolatedrows].violation = spxAbs(compProbPrimal * maxDualRatio);
            nviolatedrows++;
         }
      }
   }


   // sorting the violated rows by the absolute violation.
   // Only a predefined number of rows will be added to the reduced problem
   RowViolationCompare compare;
   compare.entry = violatedrows.get_const_ptr();

   // if no rows are identified by the pricing rule, we add rows based upon the constraint violations
   if(!ratioTest || nviolatedrows == 0)
   {
      _findViolatedRows(objValue, violatedrows, nviolatedrows);
   }

   int sorted = 0;
   int sortsize = MINIMUM(intParam(SoPlexBase<R>::DECOMP_MAXADDEDROWS), nviolatedrows);

   // only sorting if the sort size is less than the number of violated rows.
   if(sortsize > 0 && sortsize < nviolatedrows)
      sorted = SPxQuicksortPart(violatedrows.get_ptr(), compare, sorted + 1, nviolatedrows, sortsize);

   // adding the violated rows.
   for(int i = 0; i < sortsize; i++)
   {
      updaterows.add(_transformedRows.lhs(violatedrows[i].idx),
                     _transformedRows.rowVector(violatedrows[i].idx),
                     _transformedRows.rhs(violatedrows[i].idx));

      _decompReducedProbRows[violatedrows[i].idx] = true;
      newrowidx[nnewrowidx] = violatedrows[i].idx;
      nnewrowidx++;
   }

   SPxRowId* addedrowids = 0;
   spx_alloc(addedrowids, nnewrowidx);
   _solver.addRows(addedrowids, updaterows);

   for(int i = 0; i < nnewrowidx; i++)
      _decompReducedProbRowIDs[newrowidx[i]] = addedrowids[i];

   // freeing allocated memory
   spx_free(addedrowids);
   spx_free(newrowidx);
}



/// update the reduced problem with additional columns and rows based upon the violated original bounds and rows
// TODO: Allow for the case that no rows are added. This should terminate the algorithm.
// TODO: Check to make sure that only rows added to the problem do not currently exist in the reduced problem.
template <class R>
void SoPlexBase<R>::_updateDecompReducedProblemViol(bool allrows)
{
#ifdef NO_TOL
   R feastol = 0.0;
#else
#ifdef USE_FEASTOL
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
#else
   R feastol = realParam(SoPlexBase<R>::EPSILON_ZERO);
#endif
#endif
   LPRowSetBase<R> updaterows;

   int* newrowidx = 0;
   int nnewrowidx = 0;
   spx_alloc(newrowidx, _nPrimalRows);

   int rowNumber;
   int bestrow = -1;
   R bestrownorm = R(infinity);
   R percenttoadd = 1;

   int nrowstoadd = MINIMUM(intParam(SoPlexBase<R>::DECOMP_MAXADDEDROWS), _nDecompViolRows);

   if(allrows)
      nrowstoadd = _nDecompViolRows;   // adding all violated rows

   SSVectorBase<R>  y(_solver.nCols());
   y.unSetup();

   // identifying the rows not included in the reduced problem that are violated by the current solution.
   for(int i = 0; i < nrowstoadd; i++)
   {
      rowNumber = _decompViolatedRows[i];

      if(!allrows)
      {
         R norm = 0;

         // the rhs of this calculation are the rows of the constraint matrix
         // so we are solving y B = A_{i,.}
         try
         {
            _solver.basis().solve(y, _solver.vector(rowNumber));
         }
         catch(const SPxException& E)
         {
            MSG_ERROR(spxout << "Caught exception <" << E.what() << "> while computing compatability.\n");
         }


         // comparing the constraints based upon the row norm
         if(y.isSetup())
         {
            for(int j = 0; j < y.size(); j++)
            {
               if(isZero(_solver.fVec()[i], feastol))
                  norm += spxAbs(y.value(j)) * spxAbs(y.value(j));
            }
         }
         else
         {
            for(int j = 0; j < numCols(); j++)
            {
               if(isZero(_solver.fVec()[i], feastol))
                  norm += spxAbs(y[j]) * spxAbs(y[j]);
            }
         }


         // the best row is based upon the row norm
         // the best row is added if no violated row is found
         norm = soplex::spxSqrt(norm);

         if(LT(norm, bestrownorm))
         {
            bestrow = rowNumber;
            bestrownorm = norm;
         }


         // adding the violated row
         if(isZero(norm, feastol) && LT(nnewrowidx / R(numRows()), percenttoadd))
         {
            updaterows.add(_transformedRows.lhs(rowNumber), _transformedRows.rowVector(rowNumber),
                           _transformedRows.rhs(rowNumber));

            _decompReducedProbRows[rowNumber] = true;
            newrowidx[nnewrowidx] = rowNumber;
            nnewrowidx++;
         }

      }
      else
      {
         // adding all violated rows
         updaterows.add(_transformedRows.lhs(rowNumber), _transformedRows.rowVector(rowNumber),
                        _transformedRows.rhs(rowNumber));

         _decompReducedProbRows[rowNumber] = true;
         newrowidx[nnewrowidx] = rowNumber;
         nnewrowidx++;
      }
   }

   // if no violated row is found during the first pass of the available rows, then we add all violated rows.
   if(nnewrowidx == 0)
   {
      for(int i = 0; i < nrowstoadd; i++)
      {
         rowNumber = _decompViolatedRows[i];

         updaterows.add(_transformedRows.lhs(rowNumber), _transformedRows.rowVector(rowNumber),
                        _transformedRows.rhs(rowNumber));

         _decompReducedProbRows[rowNumber] = true;
         newrowidx[nnewrowidx] = rowNumber;
         nnewrowidx++;
      }
   }

   // we will always add the row that is deemed best based upon the row norm.
   // TODO: check whether this should be skipped if a row has already been added.
   if(!allrows && bestrow >= 0)
   {
      updaterows.add(_transformedRows.lhs(bestrow), _transformedRows.rowVector(bestrow),
                     _transformedRows.rhs(bestrow));

      _decompReducedProbRows[bestrow] = true;
      newrowidx[nnewrowidx] = bestrow;
      nnewrowidx++;
   }


   SPxRowId* addedrowids = 0;
   spx_alloc(addedrowids, nnewrowidx);
   _solver.addRows(addedrowids, updaterows);

   for(int i = 0; i < nnewrowidx; i++)
      _decompReducedProbRowIDs[newrowidx[i]] = addedrowids[i];

   // freeing allocated memory
   spx_free(addedrowids);
   spx_free(newrowidx);
}



/// builds the update rows with those violated in the complmentary problem
// A row is violated in the constraint matrix Ax <= b, if b - A_{i}x < 0
// To aid the computation, all violations are translated to <= constraints
template <class R>
void SoPlexBase<R>::_findViolatedRows(R compObjValue, Array<RowViolation>& violatedrows,
                                      int& nviolatedrows)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
   VectorBase<R> compProbRedcost(
      _compSolver.nCols());   // the reduced costs of the complementary problem
   VectorBase<R> compProbPrimal(
      _compSolver.nCols());    // the primal vector of the complementary problem
   VectorBase<R> compProbActivity(
      _compSolver.nRows());  // the activity vector of the complementary problem
   R compProbSlackVal = 0;

   if(boolParam(SoPlexBase<R>::USECOMPDUAL))
   {
      // Retrieving the slacks for each row.
      _compSolver.getRedCostSol(compProbRedcost);
   }
   else
   {
      // Retrieving the primal vector for the complementary problem
      _compSolver.getPrimalSol(compProbPrimal);
      _compSolver.computePrimalActivity(compProbPrimal, compProbActivity);

      compProbSlackVal = compProbPrimal[_compSolver.number(_compSlackColId)];
   }

   // scanning all rows of the complementary problem for violations.
   for(int i = 0; i < _nPrimalRows; i++)
   {
      LPRowBase<R> origlprow;
      DSVectorBase<R> rowtoaddVec(_realLP->nCols());
      R compProbViol = 0;
      R compSlackCoeff = 0;
      int rownumber = _realLP->number(SPxRowId(_decompPrimalRowIDs[i]));
      int comprownum = _compSolver.number(SPxRowId(_decompPrimalRowIDs[i]));

      if(!_decompReducedProbRows[rownumber])
      {
         // retreiving the violation of the complementary problem primal constraints
         if(boolParam(SoPlexBase<R>::USECOMPDUAL))
         {
            compSlackCoeff = getCompSlackVarCoeff(i);
            compProbViol = compProbRedcost[_compSolver.number(SPxColId(
                                              _decompDualColIDs[i]))]; // this is b - Ax
            // subtracting the slack variable value
            compProbViol += compObjValue * compSlackCoeff; // must add on the slack variable value.
            compProbViol *= compSlackCoeff;  // translating the violation to a <= constraint
         }
         else
         {
            R viol = _compSolver.rhs(comprownum) - (compProbActivity[comprownum] + compProbSlackVal);

            if(viol < 0.0)
               compProbViol = viol;

            viol = (compProbActivity[comprownum] - compProbSlackVal) - _compSolver.lhs(comprownum);

            if(viol < 0.0)
               compProbViol = viol;

         }

         // NOTE: if the row was originally a ranged constraint, we are only interest in one of the inequalities.
         // If one inequality of the range violates the bounds, then we will add the row.

         // the translation of the complementary primal problem to the dual some rows resulted in two columns.
         if(boolParam(SoPlexBase<R>::USECOMPDUAL) && i < _nPrimalRows - 1 &&
               _realLP->number(SPxRowId(_decompPrimalRowIDs[i])) == _realLP->number(SPxRowId(
                        _decompPrimalRowIDs[i + 1])))
         {
            i++;
            compSlackCoeff = getCompSlackVarCoeff(i);
            R tempViol = compProbRedcost[_compSolver.number(SPxColId(_decompDualColIDs[i]))]; // this is b - Ax
            tempViol += compObjValue * compSlackCoeff;
            tempViol *= compSlackCoeff;

            // if the other side of the range constraint has a larger violation, then this is used for the
            // computation.
            if(tempViol < compProbViol)
               compProbViol = tempViol;
         }


         // checking the violation of the row.
         if(LT(compProbViol, (R) 0, feastol))
         {
            if(!_decompReducedProbRows[rownumber])
            {
               numIncludedRows++;
               assert(numIncludedRows <= _realLP->nRows());
            }

            violatedrows[nviolatedrows].idx = rownumber;
            violatedrows[nviolatedrows].violation = spxAbs(compProbViol);
            nviolatedrows++;
         }
      }
   }
}



/// identifies the columns of the row-form basis that correspond to rows with zero dual multipliers.
// This function assumes that the basis is in the row form.
// @todo extend this to the case when the basis is in the column form.
template <class R>
void SoPlexBase<R>::_getZeroDualMultiplierIndices(VectorBase<R> feasVector, int* nonposind,
      int* colsforremoval,
      int* nnonposind, bool& stop)
{
   assert(_solver.rep() == SPxSolverBase<R>::ROW);

#ifdef NO_TOL
   R feastol = 0.0;
#else
#ifdef USE_FEASTOL
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
#else
   R feastol = realParam(SoPlexBase<R>::EPSILON_ZERO);
#endif
#endif

   bool delCol;

   _decompReducedProbColIDs.reSize(numCols());
   *nnonposind = 0;

   // iterating over all columns in the basis matrix
   // this identifies the basis indices and the indicies that are positive.
   for(int i = 0; i < _solver.nCols(); ++i)
   {
      _decompReducedProbCols[i] = true;
      _decompReducedProbColIDs[i].inValidate();
      colsforremoval[i] = i;
      delCol = false;

      if(_solver.basis().baseId(i).isSPxRowId())   // find the row id's for rows in the basis
      {
         // record the row if the dual multiple is zero.
         if(isZero(feasVector[i], feastol))
         {
            nonposind[*nnonposind] = i;
            (*nnonposind)++;

            // NOTE: commenting out the delCol flag at this point. The colsforremoval array should indicate the
            // columns that have a zero reduced cost. Hence, the delCol flag should only be set in the isSPxColId
            // branch of the if statement.
            //delCol = true;
         }
      }
      else if(_solver.basis().baseId(
                 i).isSPxColId())    // get the column id's for the columns in the basis
      {
         if(isZero(feasVector[i], feastol))
         {
            nonposind[*nnonposind] = i;
            (*nnonposind)++;

            delCol = true;
         }
      }

      // setting an array to identify the columns to be removed from the LP to form the reduced problem
      if(delCol)
      {
         colsforremoval[i] = -1;
         _decompReducedProbCols[i] = false;
      }
      else if(_solver.basis().baseId(i).isSPxColId())
      {
         if(_solver.basis().baseId(i).isSPxColId())
            _decompReducedProbColIDs[_solver.number(_solver.basis().baseId(i))] = SPxColId(
                     _solver.basis().baseId(i));
         else
            _decompReducedProbCols[_solver.number(_solver.basis().baseId(i))] = false;
      }
   }

   stop = decompTerminate(realParam(SoPlexBase<R>::TIMELIMIT) * TIMELIMIT_FRAC);
}



/// retrieves the compatible columns from the constraint matrix
// This function also updates the constraint matrix of the reduced problem. It is efficient to perform this in the
// following function because the required linear algebra has been performed.
template <class R>
void SoPlexBase<R>::_getCompatibleColumns(VectorBase<R> feasVector, int* nonposind, int* compatind,
      int* rowsforremoval,
      int* colsforremoval, int nnonposind, int* ncompatind, bool formRedProb, bool& stop)
{
#ifdef NO_TOL
   R feastol = 0.0;
#else
#ifdef USE_FEASTOL
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
#else
   R feastol = realParam(SoPlexBase<R>::EPSILON_ZERO);
#endif
#endif

   bool compatible;
   SSVectorBase<R>  y(_solver.nCols());
   y.unSetup();

   *ncompatind  = 0;

#ifndef NDEBUG
   int bind = 0;
   bool* activerows = 0;
   spx_alloc(activerows, numRows());

   for(int i = 0; i < numRows(); ++i)
      activerows[i] = false;

   for(int i = 0; i < numCols(); ++i)
   {
      if(_solver.basis().baseId(i).isSPxRowId())   // find the row id's for rows in the basis
      {
         if(!isZero(feasVector[i], feastol))
         {
            bind = _realLP->number(SPxRowId(_solver.basis().baseId(i))); // getting the corresponding row
            // for the original LP.
            assert(bind >= 0 && bind < numRows());
            activerows[bind] = true;
         }
      }
   }

#endif

   // this function is called from other places where to identify the columns for removal. In these places the
   // reduced problem is not formed.
   if(formRedProb)
   {
      _decompReducedProbRowIDs.reSize(_solver.nRows());
      _decompReducedProbColRowIDs.reSize(_solver.nRows());
   }

   for(int i = 0; i < numRows(); ++i)
   {
      rowsforremoval[i] = i;

      if(formRedProb)
         _decompReducedProbRows[i] = true;

      numIncludedRows++;

      // the rhs of this calculation are the rows of the constraint matrix
      // so we are solving y B = A_{i,.}
      // @NOTE: This may not actually be necessary. The solve process is very time consuming and is a point where the
      // approach breaks down. It could be simplier if we use a faster solve. Maybe something like:
      // Omer, J.; Towhidi, M. & Soumis, F., "The positive edge pricing rule for the dual simplex",
      // Computers & Operations Research , 2015, 61, 135-142
      try
      {
         _solver.basis().solve(y, _solver.vector(i));
      }
      catch(const SPxException& E)
      {
         MSG_ERROR(spxout << "Caught exception <" << E.what() << "> while computing compatability.\n");
      }

      compatible = true;

      // a compatible row is given by zeros in all columns related to the nonpositive indices
      for(int j = 0; j < nnonposind; ++j)
      {
         // @TODO: getting a tolerance issue with this check. Don't know how to fix it.
         if(!isZero(y[nonposind[j]], feastol))
         {
            compatible = false;
            break;
         }
      }

      // checking that the active rows are compatible
      assert(!activerows[i] || compatible);

      // changing the matrix coefficients
      DSVectorBase<R> newRowVector;
      LPRowBase<R> rowtoupdate;

      if(y.isSetup())
      {
         for(int j = 0; j < y.size(); j++)
            // coverity[var_deref_model]
            newRowVector.add(y.index(j), y.value(j));
      }
      else
      {
         for(int j = 0; j < numCols(); j++)
         {
            if(!isZero(y[j], feastol))
               newRowVector.add(j, y[j]);
         }
      }

      // transforming the original problem rows
      _solver.getRow(i, rowtoupdate);

#ifndef NO_TRANSFORM
      rowtoupdate.setRowVector(newRowVector);
#endif

      if(formRedProb)
         _transformedRows.add(rowtoupdate);


      // Making all equality constraints compatible, i.e. they are included in the reduced problem
      if(EQ(rowtoupdate.lhs(), rowtoupdate.rhs()))
         compatible = true;

      if(compatible)
      {
         compatind[*ncompatind] = i;
         (*ncompatind)++;

         if(formRedProb)
         {
            _decompReducedProbRowIDs[i] = _solver.rowId(i);

            // updating the compatible row
            _decompLP->changeRow(i, rowtoupdate);
         }
      }
      else
      {
         // setting an array to identify the rows to be removed from the LP to form the reduced problem
         rowsforremoval[i] = -1;
         numIncludedRows--;

         if(formRedProb)
            _decompReducedProbRows[i] = false;
      }

      // determine whether the reduced problem setup should be terminated
      stop = decompTerminate(realParam(SoPlexBase<R>::TIMELIMIT) * TIMELIMIT_FRAC);

      if(stop)
         break;
   }

   assert(numIncludedRows <= _solver.nRows());

#ifndef NDEBUG
   spx_free(activerows);
#endif
}



/// computes the reduced problem objective coefficients
template <class R>
void SoPlexBase<R>::_computeReducedProbObjCoeff(bool& stop)
{
#ifdef NO_TOL
   R feastol = 0.0;
#else
#ifdef USE_FEASTOL
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
#else
   R feastol = realParam(SoPlexBase<R>::EPSILON_ZERO);
#endif
#endif

   SSVectorBase<R>  y(numCols());
   y.unSetup();

   // the rhs of this calculation is the original objective coefficient vector
   // so we are solving y B = c
   try
   {
      _solver.basis().solve(y, _solver.maxObj());
   }
   catch(const SPxException& E)
   {
      MSG_ERROR(spxout << "Caught exception <" << E.what() << "> while computing compatability.\n");
   }

   _transformedObj.reDim(numCols());

   if(y.isSetup())
   {
      int ycount = 0;

      for(int i = 0; i < numCols(); i++)
      {
         if(ycount < y.size() && i == y.index(ycount))
         {
            _transformedObj[i] = y.value(ycount);
            ycount++;
         }
         else
            _transformedObj[i] = 0.0;
      }
   }
   else
   {
      for(int i = 0; i < numCols(); i++)
      {
         if(isZero(y[i], feastol))
            _transformedObj[i] = 0.0;
         else
            _transformedObj[i] = y[i];
      }
   }

   // setting the updated objective vector
#ifndef NO_TRANSFORM
   _decompLP->changeObj(_transformedObj);
#endif

   // determine whether the reduced problem setup should be terminated
   stop = decompTerminate(realParam(SoPlexBase<R>::TIMELIMIT) * TIMELIMIT_FRAC);
}



/// computes the compatible bound constraints and adds them to the reduced problem
// NOTE: No columns are getting removed from the reduced problem. Only the bound constraints are being removed.
// So in the reduced problem, all variables are free unless the bound constraints are selected as compatible.
template <class R>
void SoPlexBase<R>::_getCompatibleBoundCons(LPRowSetBase<R>& boundcons, int* compatboundcons,
      int* nonposind,
      int* ncompatboundcons, int nnonposind, bool& stop)
{
#ifdef NO_TOL
   R feastol = 0.0;
#else
#ifdef USE_FEASTOL
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
#else
   R feastol = realParam(SoPlexBase<R>::EPSILON_ZERO);
#endif
#endif

   bool compatible;
   int ncols = numCols();
   SSVectorBase<R>  y(ncols);
   y.unSetup();

   _decompReducedProbColRowIDs.reSize(numCols());


   // identifying the compatible bound constraints
   for(int i = 0; i < ncols; i++)
   {
      _decompReducedProbColRowIDs[i].inValidate();

      // setting all variables to free variables.
      // Bound constraints will only be added to the variables with compatible bound constraints.
      _decompLP->changeUpper(i, R(infinity));
      _decompLP->changeLower(i, R(-infinity));

      // the rhs of this calculation are the unit vectors of the bound constraints
      // so we are solving y B = I_{i,.}
      // this solve should not be required. We only need the column of the basis inverse.
      try
      {
         _solver.basis().solve(y, _solver.unitVector(i));
      }
      catch(const SPxException& E)
      {
         MSG_ERROR(spxout << "Caught exception <" << E.what() << "> while computing compatability.\n");
      }

      compatible = true;

      // a compatible row is given by zeros in all columns related to the nonpositive indices
      for(int j = 0; j < nnonposind; ++j)
      {
         // @todo really need to check this part of the code. Run through this with Ambros or Matthias.
         if(!isZero(y[nonposind[j]]))
         {
            compatible = false;
            break;
         }
      }

      // changing the matrix coefficients
      DSVectorBase<R> newRowVector;
      LPRowBase<R> rowtoupdate;

#ifndef NO_TRANSFORM

      if(y.isSetup())
      {
         for(int j = 0; j < y.size(); j++)
            newRowVector.add(y.index(j), y.value(j));
      }
      else
      {
         for(int j = 0; j < ncols; j++)
         {
            if(!isZero(y[j], feastol))
            {
               newRowVector.add(j, y[j]);
            }
         }
      }

#else
      newRowVector.add(i, 1.0);
#endif

      // will probably need to map this set of rows.
      _transformedRows.add(_solver.lower(i), newRowVector, _solver.upper(i));

      // variable bounds are compatible in the current implementation.
      // this is still function is still required to transform the bound constraints with respect to the basis matrix.
      compatible = true;

      // if the bound constraint is compatible, it remains in the reduced problem.
      // Otherwise the bound is removed and the variables are free.
      if(compatible)
      {
         R lhs = R(-infinity);
         R rhs = R(infinity);

         if(GT(_solver.lower(i), R(-infinity)))
            lhs = _solver.lower(i);

         if(LT(_solver.upper(i), R(infinity)))
            rhs = _solver.upper(i);

         if(GT(lhs, R(-infinity)) || LT(rhs, R(infinity)))
         {
            compatboundcons[(*ncompatboundcons)] = i;
            (*ncompatboundcons)++;

            boundcons.add(lhs, newRowVector, rhs);
         }


      }

      // determine whether the reduced problem setup should be terminated
      stop = decompTerminate(realParam(SoPlexBase<R>::TIMELIMIT) * TIMELIMIT_FRAC);

      if(stop)
         break;
   }

}



/// computes the rows to remove from the complementary problem
template <class R>
void SoPlexBase<R>::_getRowsForRemovalComplementaryProblem(int* nonposind, int* bind,
      int* rowsforremoval,
      int* nrowsforremoval, int nnonposind)
{
   *nrowsforremoval = 0;

   for(int i = 0; i < nnonposind; i++)
   {
      if(bind[nonposind[i]] < 0)
      {
         rowsforremoval[*nrowsforremoval] = -1 - bind[nonposind[i]];
         (*nrowsforremoval)++;
      }
   }
}



/// removing rows from the complementary problem.
// the rows that are removed from decompCompLP are the rows from the reduced problem that have a non-positive dual
// multiplier in the optimal solution.
template <class R>
void SoPlexBase<R>::_deleteAndUpdateRowsComplementaryProblem(SPxRowId rangedRowIds[],
      int& naddedrows)
{
   DSVectorBase<R> slackColCoeff;

   // setting the objective coefficients of the original variables to zero
   VectorBase<R> newObjCoeff(numCols());

   for(int i = 0; i < numCols(); i++)
   {
      _compSolver.changeBounds(_realLP->cId(i), R(-infinity), R(infinity));
      newObjCoeff[i] = 0;
   }

   _compSolver.changeObj(newObjCoeff);

   // adding the slack column to the complementary problem
   SPxColId* addedcolid = 0;

   if(boolParam(SoPlexBase<R>::USECOMPDUAL))
   {
      spx_alloc(addedcolid, 1);
      LPColSetBase<R> compSlackCol;
      compSlackCol.add(R(1.0), R(-infinity), slackColCoeff, R(infinity));
      _compSolver.addCols(addedcolid, compSlackCol);
      _compSlackColId = addedcolid[0];
   }
   else
   {
      LPRowSetBase<R>
      addrangedrows;    // the row set of ranged and equality rows that must be added to the complementary problem.
      naddedrows = 0;

      // finding all of the ranged and equality rows and creating two <= constraints.
      for(int i = 0; i < numRows(); i++)
      {
         if(_realLP->rowType(i) == LPRowBase<R>::RANGE || _realLP->rowType(i) == LPRowBase<R>::EQUAL)
         {
            assert(GT(_compSolver.lhs(i), R(-infinity)) && LT(_compSolver.rhs(i), R(infinity)));
            assert(_compSolver.rowType(i) == LPRowBase<R>::RANGE
                   || _compSolver.rowType(i) == LPRowBase<R>::EQUAL);

            _compSolver.changeLhs(i, R(-infinity));
            addrangedrows.add(_realLP->lhs(i), _realLP->rowVector(i), R(infinity));
            naddedrows++;
         }
      }

      // adding the rows for the ranged rows to make <= conatraints
      SPxRowId* addedrowid = 0;
      spx_alloc(addedrowid, naddedrows);
      _compSolver.addRows(addedrowid, addrangedrows);

      // collecting the row ids
      for(int i = 0; i < naddedrows; i++)
         rangedRowIds[i] = addedrowid[i];

      spx_free(addedrowid);


      // adding the slack column
      spx_alloc(addedcolid, 1);
      LPColSetBase<R> compSlackCol;
      compSlackCol.add(R(-1.0), R(0.0), slackColCoeff, R(infinity));

      _compSolver.addCols(addedcolid, compSlackCol);
      _compSlackColId = addedcolid[0];
   }

   // freeing allocated memory
   spx_free(addedcolid);
}



/// update the dual complementary problem with additional columns and rows
// Given the solution to the updated reduced problem, the complementary problem will be updated with modifications to
// the constraints and the removal of variables
template <class R>
void SoPlexBase<R>::_updateDecompComplementaryDualProblem(bool origObj)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   int prevNumCols =
      _compSolver.nCols(); // number of columns in the previous formulation of the complementary prob
   int prevNumRows = _compSolver.nRows();
   int prevPrimalRowIds = _nPrimalRows;
   int prevDualColIds = _nDualCols;

   LPColSetBase<R> addElimCols(_nElimPrimalRows);  // columns previously eliminated from the
   // complementary problem that must be added
   int numElimColsAdded = 0;
   int currnumrows = prevNumRows;
   // looping over all rows from the original LP that were eliminated during the formation of the complementary
   // problem. The eliminated rows will be added if they are basic in the reduced problem.

   for(int i = 0; i < _nElimPrimalRows; i++)
   {
      int rowNumber = _realLP->number(_decompElimPrimalRowIDs[i]);

      int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
      assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

      // checking for the basic rows in the reduced problem
      if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
            _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER ||
            _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED ||
            _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE ||
            (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
             LE(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol)) ||
            (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
             LE(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
      {
         LPRowBase<R> origlprow;
         DSVectorBase<R> coltoaddVec(_realLP->nCols());

         LPRowSetBase<R> additionalrows;
         int nnewrows = 0;

         _realLP->getRow(rowNumber, origlprow);

         for(int j = 0; j < origlprow.rowVector().size(); j++)
         {
            // the column of the new row may not exist in the current complementary problem.
            // if the column does not exist, then it is necessary to create the column.
            int colNumber = origlprow.rowVector().index(j);

            if(_decompCompProbColIDsIdx[colNumber] == -1)
            {
               assert(!_decompReducedProbColIDs[colNumber].isValid());
               _decompPrimalColIDs[_nPrimalCols] = _realLP->cId(colNumber);
               _decompCompProbColIDsIdx[colNumber] = _nPrimalCols;
               _fixedOrigVars[colNumber] = -2;
               _nPrimalCols++;

               // all columns for the complementary problem are converted to unrestricted.
               additionalrows.create(1, _realLP->maxObj(colNumber), _realLP->maxObj(colNumber));
               nnewrows++;

               coltoaddVec.add(currnumrows, origlprow.rowVector().value(j));
               currnumrows++;
            }
            else
               coltoaddVec.add(_compSolver.number(_decompDualRowIDs[_decompCompProbColIDsIdx[colNumber]]),
                               origlprow.rowVector().value(j));
         }

         SPxRowId* addedrowids = 0;
         spx_alloc(addedrowids, nnewrows);
         _compSolver.addRows(addedrowids, additionalrows);

         for(int j = 0; j < nnewrows; j++)
         {
            _decompDualRowIDs[_nDualRows] = addedrowids[j];
            _nDualRows++;
         }

         spx_free(addedrowids);



         if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
               _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED ||
               _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE ||
               (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                LE(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol)))
         {
            assert(LT(_realLP->rhs(_decompElimPrimalRowIDs[i]), R(infinity)));
            addElimCols.add(_realLP->rhs(_decompElimPrimalRowIDs[i]), R(-infinity), coltoaddVec, R(infinity));

            if(_nPrimalRows >= _decompPrimalRowIDs.size())
            {
               _decompPrimalRowIDs.reSize(_nPrimalRows * 2);
               _decompDualColIDs.reSize(_nPrimalRows * 2);
            }

            _decompPrimalRowIDs[_nPrimalRows] = _decompElimPrimalRowIDs[i];
            _nPrimalRows++;

            _decompElimPrimalRowIDs.remove(i);
            _nElimPrimalRows--;
            i--;

            numElimColsAdded++;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER ||
                 (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                  LE(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
         {
            // this assert should stay, but there is an issue with the status and the dual vector
            //assert(LT(dualVector[_solver.number(_decompReducedProbRowIDs[rowNumber])], 0.0));
            assert(GT(_realLP->lhs(_decompElimPrimalRowIDs[i]), R(-infinity)));
            addElimCols.add(_realLP->lhs(_decompElimPrimalRowIDs[i]), R(-infinity), coltoaddVec, R(infinity));

            _decompPrimalRowIDs[_nPrimalRows] = _decompElimPrimalRowIDs[i];
            _nPrimalRows++;

            _decompElimPrimalRowIDs.remove(i);
            _nElimPrimalRows--;
            i--;

            numElimColsAdded++;
         }
      }
   }

   MSG_INFO2(spxout, spxout << "Number of eliminated columns added to complementary problem: "
             << numElimColsAdded << std::endl);

   // updating the _decompDualColIDs with the additional columns from the eliminated rows.
   _compSolver.addCols(addElimCols);

   for(int i = prevNumCols; i < _compSolver.nCols(); i++)
   {
      _decompDualColIDs[prevDualColIds + i - prevNumCols] = _compSolver.colId(i);
      _nDualCols++;
   }

   assert(_nDualCols == _nPrimalRows);

   // looping over all rows from the original problem that were originally contained in the complementary problem.
   // The basic rows will be set as free variables, the non-basic rows will be eliminated from the complementary
   // problem.
   DSVectorBase<R> slackRowCoeff(_compSolver.nCols());

   int* colsforremoval = 0;
   int ncolsforremoval = 0;
   spx_alloc(colsforremoval, prevPrimalRowIds);

   for(int i = 0; i < prevPrimalRowIds; i++)
   {
      int rowNumber = _realLP->number(_decompPrimalRowIDs[i]);

      // this loop runs over all rows previously in the complementary problem. If rows are added to the reduced
      // problem, they will be transfered from the incompatible set to the compatible set in the following if
      // statement.
      if(_decompReducedProbRows[rowNumber])
      {
         // rows added to the reduced problem may have been equality constriants. The equality constraints from the
         // original problem are converted into <= and >= constraints. Upon adding these constraints to the reduced
         // problem, only a single dual column is needed in the complementary problem. Hence, one of the dual columns
         // is removed.
         //
         // 22.06.2015 Testing keeping all constraints in the complementary problem.
         // This requires a dual column to be fixed to zero for the range and equality rows.
#ifdef KEEP_ALL_ROWS_IN_COMP_PROB
         bool incrementI = false;
#endif

         if(i + 1 < prevPrimalRowIds
               && _realLP->number(_decompPrimalRowIDs[i]) == _realLP->number(_decompPrimalRowIDs[i + 1]))
         {
            assert(_decompPrimalRowIDs[i].idx == _decompPrimalRowIDs[i + 1].idx);

#ifdef KEEP_ALL_ROWS_IN_COMP_PROB // 22.06.2015

            if(_realLP->rowType(_decompPrimalRowIDs[i]) == LPRowBase<R>::RANGE)
            {
               _compSolver.changeObj(_decompDualColIDs[i + 1], 0.0);
               _compSolver.changeBounds(_decompDualColIDs[i + 1], 0.0, 0.0);
            }

            incrementI = true;
#else
            colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompDualColIDs[i + 1]));
            ncolsforremoval++;

            _decompPrimalRowIDs.remove(i + 1);
            _nPrimalRows--;
            _decompDualColIDs.remove(i + 1);
            _nDualCols--;

            prevPrimalRowIds--;
#endif
         }

         //assert(i + 1 == prevPrimalRowIds || _decompPrimalRowIDs[i].idx != _decompPrimalRowIDs[i+1].idx);

         int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
         assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

         if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
               _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED ||
               _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE ||
               (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                LE(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol)))
         {
            //assert(GT(dualVector[solverRowNum], 0.0));
            _compSolver.changeObj(_decompDualColIDs[i], _realLP->rhs(SPxRowId(_decompPrimalRowIDs[i])));
            _compSolver.changeBounds(_decompDualColIDs[i], R(-infinity), R(infinity));
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER ||
                 (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                  LE(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
         {
            //assert(LT(dualVector[solverRowNum], 0.0));
            _compSolver.changeObj(_decompDualColIDs[i], _realLP->lhs(SPxRowId(_decompPrimalRowIDs[i])));
            _compSolver.changeBounds(_decompDualColIDs[i], R(-infinity), R(infinity));
         }
         else //if ( _solver.basis().desc().rowStatus(solverRowNum) != SPxBasisBase<R>::Desc::D_FREE )
         {
            //assert(isZero(dualVector[solverRowNum], 0.0));

            // 22.06.2015 Testing keeping all rows in the complementary problem
#ifdef KEEP_ALL_ROWS_IN_COMP_PROB
            switch(_realLP->rowType(_decompPrimalRowIDs[i]))
            {
            case LPRowBase<R>::RANGE:
               assert(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
                      _realLP->number(SPxColId(_decompPrimalRowIDs[i + 1])));
               assert(_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i]))) !=
                      _compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))));

               _compSolver.changeObj(_decompDualColIDs[i], _realLP->rhs(_decompPrimalRowIDs[i]));
               _compSolver.changeBounds(_decompDualColIDs[i], 0.0, R(infinity));
               _compSolver.changeObj(_decompDualColIDs[i + 1], _realLP->lhs(_decompPrimalRowIDs[i]));
               _compSolver.changeBounds(_decompDualColIDs[i + 1], R(-infinity), 0.0);

               i++;
               break;

            case LPRowBase<R>::GREATER_EQUAL:
               _compSolver.changeObj(_decompDualColIDs[i], _realLP->lhs(_decompPrimalRowIDs[i]));
               _compSolver.changeBounds(_decompDualColIDs[i], R(-infinity), 0.0);
               break;

            case LPRowBase<R>::LESS_EQUAL:
               _compSolver.changeObj(_decompDualColIDs[i], _realLP->rhs(_decompPrimalRowIDs[i]));
               _compSolver.changeBounds(_decompDualColIDs[i], 0.0, R(infinity));
               break;

            default:
               throw SPxInternalCodeException("XDECOMPSL01 This should never happen.");
            }

#else // 22.06.2015 testing keeping all rows in the complementary problem
            colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompDualColIDs[i]));
            ncolsforremoval++;

            if(_nElimPrimalRows >= _decompElimPrimalRowIDs.size())
               _decompElimPrimalRowIDs.reSize(_realLP->nRows());

            _decompElimPrimalRowIDs[_nElimPrimalRows] = _decompPrimalRowIDs[i];
            _nElimPrimalRows++;
            _decompPrimalRowIDs.remove(i);
            _nPrimalRows--;
            _decompDualColIDs.remove(i);
            _nDualCols--;

            i--;
            prevPrimalRowIds--;
#endif
         }

#ifdef KEEP_ALL_ROWS_IN_COMP_PROB

         if(incrementI)
            i++;

#endif
      }
      else
      {
         switch(_realLP->rowType(_decompPrimalRowIDs[i]))
         {
         case LPRowBase<R>::RANGE:
            assert(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
                   _realLP->number(SPxColId(_decompPrimalRowIDs[i + 1])));
            assert(_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i]))) !=
                   _compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))));

            if(_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i]))) <
                  _compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))))
            {
               slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i])), -SLACKCOEFF);
               slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i + 1])), SLACKCOEFF);
            }
            else
            {
               slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i])), SLACKCOEFF);
               slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i + 1])), -SLACKCOEFF);
            }


            i++;
            break;

         case LPRowBase<R>::EQUAL:
            assert(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
                   _realLP->number(SPxColId(_decompPrimalRowIDs[i + 1])));

            slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i])), SLACKCOEFF);
            slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i + 1])), SLACKCOEFF);

            i++;
            break;

         case LPRowBase<R>::GREATER_EQUAL:
            slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i])), -SLACKCOEFF);
            break;

         case LPRowBase<R>::LESS_EQUAL:
            slackRowCoeff.add(_compSolver.number(SPxColId(_decompDualColIDs[i])), SLACKCOEFF);
            break;

         default:
            throw SPxInternalCodeException("XDECOMPSL01 This should never happen.");
         }

         if(origObj)
         {
            int numRemove = 1;
            int removeCount = 0;

            if(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
                  _realLP->number(SPxColId(_decompPrimalRowIDs[i + 1])))
               numRemove++;

            do
            {
               colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompDualColIDs[i]));
               ncolsforremoval++;

               if(_nElimPrimalRows >= _decompElimPrimalRowIDs.size())
                  _decompElimPrimalRowIDs.reSize(_realLP->nRows());

               _decompElimPrimalRowIDs[_nElimPrimalRows] = _decompPrimalRowIDs[i];
               _nElimPrimalRows++;
               _decompPrimalRowIDs.remove(i);
               _nPrimalRows--;
               _decompDualColIDs.remove(i);
               _nDualCols--;

               i--;
               prevPrimalRowIds--;

               removeCount++;
            }
            while(removeCount < numRemove);
         }
      }
   }

   // updating the slack column in the complementary problem
   R lhs = 1.0;
   R rhs = 1.0;

   // it is possible that all rows are included in the reduced problem. In this case, the slack row will be empty. To
   // avoid infeasibility, the lhs and rhs are set to 0.
   if(slackRowCoeff.size() == 0)
   {
      lhs = 0.0;
      rhs = 0.0;
   }

   LPRowBase<R> compSlackRow(lhs, slackRowCoeff, rhs);
   _compSolver.changeRow(_compSlackDualRowId, compSlackRow);


   // if the original objective is used, then all dual columns related to primal rows not in the reduced problem are
   // removed from the complementary problem.
   // As a result, the slack row becomes an empty row.
   int* perm = 0;
   spx_alloc(perm, _compSolver.nCols() + numElimColsAdded);
   _compSolver.removeCols(colsforremoval, ncolsforremoval, perm);


   // updating the dual columns to represent the fixed primal variables.
   int* currFixedVars = 0;
   spx_alloc(currFixedVars, _realLP->nCols());
   _identifyComplementaryDualFixedPrimalVars(currFixedVars);
   _removeComplementaryDualFixedPrimalVars(currFixedVars);
   _updateComplementaryDualFixedPrimalVars(currFixedVars);


   // freeing allocated memory
   spx_free(currFixedVars);
   spx_free(perm);
   spx_free(colsforremoval);
}



/// update the primal complementary problem with additional columns and rows
// Given the solution to the updated reduced problem, the complementary problem will be updated with modifications to
// the constraints and the removal of variables
template <class R>
void SoPlexBase<R>::_updateDecompComplementaryPrimalProblem(bool origObj)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   int prevNumRows = _compSolver.nRows();
   int prevPrimalRowIds = _nPrimalRows;

   assert(_nPrimalRows == _nCompPrimalRows);

   LPRowSetBase<R> addElimRows(_nElimPrimalRows);  // rows previously eliminated from the
   // complementary problem that must be added
   int numElimRowsAdded = 0;
   // looping over all rows from the original LP that were eliminated during the formation of the complementary
   // problem. The eliminated rows will be added if they are basic in the reduced problem.

   for(int i = 0; i < _nElimPrimalRows; i++)
   {
      int rowNumber = _realLP->number(_decompElimPrimalRowIDs[i]);

      int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
      assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

      // checking the rows that are basic in the reduced problem that should be added to the complementary problem
      if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER
            || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER
            || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED
            || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE
            || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                EQ(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol))
            || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                EQ(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
      {
         LPRowBase<R> origlprow;
         _realLP->getRow(rowNumber, origlprow);

         // NOTE: 11.02.2016 I am assuming that all columns from the original problem are contained in the
         // complementary problem. Will need to check this. Since nothing is happenning in the
         // _deleteAndUpdateRowsComplementaryProblem function, I am feeling confident that all columns remain.


         if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER
               || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED
               || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE
               || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                   EQ(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol)))
         {
            assert(LT(_realLP->rhs(_decompElimPrimalRowIDs[i]), R(infinity)));

            if(_nPrimalRows >= _decompPrimalRowIDs.size())
            {
               _decompPrimalRowIDs.reSize(_nPrimalRows * 2);
               _decompCompPrimalRowIDs.reSize(_nPrimalRows * 2);
            }

            addElimRows.add(_realLP->rhs(_decompElimPrimalRowIDs[i]), origlprow.rowVector(),
                            _realLP->rhs(_decompElimPrimalRowIDs[i]));

            _decompPrimalRowIDs[_nPrimalRows] = _decompElimPrimalRowIDs[i];
            _nPrimalRows++;

            _decompElimPrimalRowIDs.remove(i);
            _nElimPrimalRows--;
            i--;

            numElimRowsAdded++;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER
                 || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                     EQ(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
         {
            assert(GT(_realLP->lhs(_decompElimPrimalRowIDs[i]), R(-infinity)));

            if(_nPrimalRows >= _decompPrimalRowIDs.size())
            {
               _decompPrimalRowIDs.reSize(_nPrimalRows * 2);
               _decompCompPrimalRowIDs.reSize(_nPrimalRows * 2);
            }

            addElimRows.add(_realLP->lhs(_decompElimPrimalRowIDs[i]), origlprow.rowVector(),
                            _realLP->lhs(_decompElimPrimalRowIDs[i]));

            _decompPrimalRowIDs[_nPrimalRows] = _decompElimPrimalRowIDs[i];
            _nPrimalRows++;

            _decompElimPrimalRowIDs.remove(i);
            _nElimPrimalRows--;
            i--;

            numElimRowsAdded++;
         }
      }
   }

   MSG_INFO2(spxout, spxout << "Number of eliminated rows added to the complementary problem: "
             << numElimRowsAdded << std::endl);

   // adding the eliminated rows to the complementary problem.
   _compSolver.addRows(addElimRows);

   for(int i = prevNumRows; i < _compSolver.nRows(); i++)
   {
      _decompCompPrimalRowIDs[prevPrimalRowIds + i - prevNumRows] = _compSolver.rowId(i);
      _nCompPrimalRows++;
   }

   assert(_nPrimalRows == _nCompPrimalRows);


   // looping over all rows from the original problem that were originally contained in the complementary problem.
   // The basic rows will be set as equalities, the non-basic rows will be eliminated from the complementary
   // problem.
   DSVectorBase<R> slackColCoeff(_compSolver.nRows());

   int* rowsforremoval = 0;
   int nrowsforremoval = 0;
   spx_alloc(rowsforremoval, prevPrimalRowIds);

   for(int i = 0; i < prevPrimalRowIds; i++)
   {
      int rowNumber = _realLP->number(_decompPrimalRowIDs[i]);

      // this loop runs over all rows previously in the complementary problem. If rows are added to the reduced
      // problem, they will be transfered from the incompatible set to the compatible set in the following if
      // statement.
      if(_decompReducedProbRows[rowNumber])
      {
         // rows added to the reduced problem may have been equality constriants. The equality constraints from the
         // original problem are converted into <= and >= constraints. Upon adding these constraints to the reduced
         // problem, only a single dual column is needed in the complementary problem. Hence, one of the dual columns
         // is removed.
         //

         int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
         assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

         if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER
               || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED
               || _solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_FREE
               || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                   EQ(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], R(0.0), feastol)))
         {
            _compSolver.changeLhs(_decompCompPrimalRowIDs[i], _realLP->rhs(SPxRowId(_decompPrimalRowIDs[i])));
            // need to also update the RHS because a ranged row could have previously been fixed to LOWER
            _compSolver.changeRhs(_decompCompPrimalRowIDs[i], _realLP->rhs(SPxRowId(_decompPrimalRowIDs[i])));
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER
                 || (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                     EQ(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), R(0.0), feastol)))
         {
            _compSolver.changeRhs(_decompCompPrimalRowIDs[i], _realLP->lhs(SPxRowId(_decompPrimalRowIDs[i])));
            // need to also update the LHS because a ranged row could have previously been fixed to UPPER
            _compSolver.changeLhs(_decompCompPrimalRowIDs[i], _realLP->lhs(SPxRowId(_decompPrimalRowIDs[i])));
         }
         else //if ( _solver.basis().desc().rowStatus(solverRowNum) != SPxBasisBase<R>::Desc::D_FREE )
         {
            rowsforremoval[nrowsforremoval] = _compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i]));
            nrowsforremoval++;

            if(_nElimPrimalRows >= _decompElimPrimalRowIDs.size())
               _decompElimPrimalRowIDs.reSize(_realLP->nRows());

            _decompElimPrimalRowIDs[_nElimPrimalRows] = _decompPrimalRowIDs[i];
            _nElimPrimalRows++;
            _decompPrimalRowIDs.remove(i);
            _nPrimalRows--;
            _decompCompPrimalRowIDs.remove(i);
            _nCompPrimalRows--;

            i--;
            prevPrimalRowIds--;
         }
      }
      else
      {
         switch(_compSolver.rowType(_decompCompPrimalRowIDs[i]))
         {
         case LPRowBase<R>::RANGE:
            assert(false);
            break;

         case LPRowBase<R>::EQUAL:
            assert(false);
            break;

         case LPRowBase<R>::LESS_EQUAL:
            slackColCoeff.add(_compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i])), -SLACKCOEFF);
            break;

         case LPRowBase<R>::GREATER_EQUAL:
            slackColCoeff.add(_compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i])), SLACKCOEFF);
            break;

         default:
            throw SPxInternalCodeException("XDECOMPSL01 This should never happen.");
         }

         // this is used as a check at the end of the algorithm. If the original objective function is used, then we
         // need to remove all unfixed variables.
         if(origObj)
         {
            rowsforremoval[nrowsforremoval] = _compSolver.number(SPxRowId(_decompCompPrimalRowIDs[i]));
            nrowsforremoval++;

            if(_nElimPrimalRows >= _decompElimPrimalRowIDs.size())
               _decompElimPrimalRowIDs.reSize(_realLP->nRows());

            _decompElimPrimalRowIDs[_nElimPrimalRows] = _decompPrimalRowIDs[i];
            _nElimPrimalRows++;
            _decompPrimalRowIDs.remove(i);
            _nPrimalRows--;
            _decompCompPrimalRowIDs.remove(i);
            _nCompPrimalRows--;

            i--;
            prevPrimalRowIds--;
         }
      }
   }

   // updating the slack column in the complementary problem
   LPColBase<R> compSlackCol(-1, slackColCoeff, R(infinity), 0.0);
   _compSolver.changeCol(_compSlackColId, compSlackCol);


   // if the original objective is used, then all complementary rows related to primal rows not in the reduced problem are
   // removed from the complementary problem.
   // As a result, the slack column becomes an empty column.
   int* perm = 0;
   spx_alloc(perm, _compSolver.nRows() + numElimRowsAdded);
   _compSolver.removeRows(rowsforremoval, nrowsforremoval, perm);

   // updating the dual columns to represent the fixed primal variables.
   int* currFixedVars = 0;
   spx_alloc(currFixedVars, _realLP->nCols());
   _identifyComplementaryPrimalFixedPrimalVars(currFixedVars);
   _updateComplementaryPrimalFixedPrimalVars(currFixedVars);


   // freeing allocated memory
   spx_free(currFixedVars);
   spx_free(perm);
   spx_free(rowsforremoval);
}



/// checking the optimality of the original problem.
// this function is called if the complementary problem is solved with a non-negative objective value. This implies
// that the rows currently included in the reduced problem are sufficient to identify the optimal solution to the
// original problem.
template <class R>
void SoPlexBase<R>::_checkOriginalProblemOptimality(VectorBase<R> primalVector, bool printViol)
{
   SSVectorBase<R>  x(_solver.nCols());
   x.unSetup();

   // multiplying the solution vector of the reduced problem with the transformed basis to identify the original
   // solution vector.
   _decompTransBasis.coSolve(x, primalVector);

   if(printViol)
   {
      MSG_INFO1(spxout, spxout << std::endl
                << "Checking consistency between the reduced problem and the original problem." << std::endl);
   }


   // checking the objective function values of the reduced problem and the original problem.
   R redObjVal = 0;
   R objectiveVal = 0;

   for(int i = 0; i < _solver.nCols(); i++)
   {
      redObjVal += _solver.maxObj(i) * primalVector[i];
      objectiveVal += _realLP->maxObj(i) * x[i];
   }

   if(printViol)
   {
      MSG_INFO1(spxout, spxout << "Reduced Problem Objective Value: " << redObjVal << std::endl
                << "Original Problem Objective Value: " << objectiveVal << std::endl);
   }

   _solReal._isPrimalFeasible = true;
   _hasSolReal = true;
   // get the primal solutions from the reduced problem
   _solReal._primal.reDim(_solver.nCols());
   _solReal._primal = x;

   R maxviol = 0;
   R sumviol = 0;

   // checking the bound violations
   if(getDecompBoundViolation(maxviol, sumviol))
   {
      if(printViol)
         MSG_INFO1(spxout, spxout << "Bound violation - "
                   << "Max violation: " << maxviol << " Sum violation: " << sumviol << std::endl);
   }

   _statistics->totalBoundViol = sumviol;
   _statistics->maxBoundViol = maxviol;

   // checking the row violations
   if(getDecompRowViolation(maxviol, sumviol))
   {
      if(printViol)
         MSG_INFO1(spxout, spxout << "Row violation - "
                   << "Max violation: " << maxviol << " Sum violation: " << sumviol << std::endl);
   }

   _statistics->totalRowViol = sumviol;
   _statistics->maxRowViol = maxviol;

   if(printViol)
      MSG_INFO1(spxout, spxout << std::endl);
}



/// updating the slack column coefficients to adjust for equality constraints
template <class R>
void SoPlexBase<R>::_updateComplementaryDualSlackColCoeff()
{
   // the slack column for the equality constraints is not handled correctly in the dual conversion. Hence, it is
   // necessary to change the equality coefficients of the dual row related to the slack column.
   for(int i = 0; i < _nPrimalRows; i++)
   {
      int rowNumber = _realLP->number(SPxRowId(_decompPrimalRowIDs[i]));

      if(!_decompReducedProbRows[rowNumber])
      {
         if(_realLP->rowType(_decompPrimalRowIDs[i]) == LPRowBase<R>::EQUAL)
         {
            assert(_realLP->lhs(_decompPrimalRowIDs[i]) == _realLP->rhs(_decompPrimalRowIDs[i]));
            _compSolver.changeLower(_decompDualColIDs[i],
                                    0.0);   // setting the lower bound of the dual column to zero.

            LPColBase<R> addEqualityCol(-_realLP->rhs(_decompPrimalRowIDs[i]),
                                        R(-1.0)*_compSolver.colVector(_decompDualColIDs[i]), R(infinity),
                                        0.0);    // adding a new column to the dual

            SPxColId newDualCol;
            _compSolver.addCol(newDualCol, addEqualityCol);

            // inserting the row and col ids for the added column. This is to be next to the original column that has
            // been duplicated.
            _decompPrimalRowIDs.insert(i + 1, 1, _decompPrimalRowIDs[i]);
            _decompDualColIDs.insert(i + 1, 1, newDualCol);
            assert(_realLP->number(_decompPrimalRowIDs[i]) == _realLP->number(_decompPrimalRowIDs[i + 1]));

            i++;
            _nPrimalRows++;
            _nDualCols++;
         }
      }
   }
}



/// identify the dual columns related to the fixed variables
template <class R>
void SoPlexBase<R>::_identifyComplementaryDualFixedPrimalVars(int* currFixedVars)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   int numFixedVar = 0;

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      currFixedVars[i] = 0;

      if(!_decompReducedProbColRowIDs[i].isValid())
         continue;

      int rowNumber = _solver.number(_decompReducedProbColRowIDs[i]);

      if(_decompReducedProbColRowIDs[i].isValid())
      {
         if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
               _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_LOWER ||
               _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_FIXED ||
               _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_FREE)
         {
            // setting the value of the _fixedOrigVars array to indicate which variables are at their bounds.
            currFixedVars[i] = getOrigVarFixedDirection(i);

            numFixedVar++;
         }
         else
         {
            // the dual flags do not imply anything about the primal status of the rows.
            if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                  EQ(_solver.rhs(rowNumber) - _solver.pVec()[rowNumber], R(0.0), feastol))
               currFixedVars[i] = 1;
            else if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                    EQ(_solver.pVec()[rowNumber] - _solver.lhs(rowNumber), R(0.0), feastol))
               currFixedVars[i] = -1;
         }
      }
   }

   MSG_INFO3(spxout, spxout << "Number of fixed primal variables in the complementary (dual) problem: "
             << numFixedVar << std::endl);
}



/// removing the dual columns related to the fixed variables
template <class R>
void SoPlexBase<R>::_removeComplementaryDualFixedPrimalVars(int* currFixedVars)
{
   SPxColId tempId;
   int ncolsforremoval = 0;
   int* colsforremoval = 0;
   spx_alloc(colsforremoval, _realLP->nCols() * 2);

   tempId.inValidate();

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      assert(_decompCompProbColIDsIdx[i] != -1);   // this should be true in the current implementation

      if(_decompCompProbColIDsIdx[i] != -1
            && _fixedOrigVars[i] != -2)  //&& _fixedOrigVars[i] != currFixedVars[i] )
      {
         if(_fixedOrigVars[i] != 0)
         {
            assert(_compSolver.number(SPxColId(_decompFixedVarDualIDs[i])) >= 0);
            assert(_fixedOrigVars[i] == -1 || _fixedOrigVars[i] == 1);
            assert(_decompFixedVarDualIDs[i].isValid());

            colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompFixedVarDualIDs[i]));
            ncolsforremoval++;

            _decompFixedVarDualIDs[i] = tempId;
         }
         else //if( false && !_decompReducedProbColRowIDs[i].isValid() ) // we want to remove all valid columns
            // in the current implementation, the only columns not included in the reduced problem are free columns.
         {
            assert((LE(_realLP->lower(i), R(-infinity)) && GE(_realLP->upper(i), R(infinity))) ||
                   _compSolver.number(SPxColId(_decompVarBoundDualIDs[i * 2])) >= 0);
            int varcount = 0;

            if(GT(_realLP->lower(i), R(-infinity)))
            {
               colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompVarBoundDualIDs[i * 2 +
                                                 varcount]));
               ncolsforremoval++;

               _decompVarBoundDualIDs[i * 2 + varcount] = tempId;
               varcount++;
            }

            if(LT(_realLP->upper(i), R(infinity)))
            {
               colsforremoval[ncolsforremoval] = _compSolver.number(SPxColId(_decompVarBoundDualIDs[i * 2 +
                                                 varcount]));
               ncolsforremoval++;

               _decompVarBoundDualIDs[i * 2 + varcount] = tempId;
            }

         }
      }
   }

   int* perm = 0;
   spx_alloc(perm, _compSolver.nCols());
   _compSolver.removeCols(colsforremoval, ncolsforremoval, perm);

   // freeing allocated memory
   spx_free(perm);
   spx_free(colsforremoval);
}



/// updating the dual columns related to the fixed primal variables.
template <class R>
void SoPlexBase<R>::_updateComplementaryDualFixedPrimalVars(int* currFixedVars)
{
   DSVectorBase<R> col(1);
   LPColSetBase<R> boundConsCols;
   LPColSetBase<R> fixedVarsDualCols(_nPrimalCols);
   int numFixedVars = 0;
   // the solution to the reduced problem results in a number of variables at their bounds. If such variables exist
   // it is necessary to include a dual column to the complementary problem related to a variable fixing. This is
   // equivalent to the tight constraints being converted to equality constraints.
   int numBoundConsCols = 0;
   int* boundConsColsAdded = 0;
   spx_alloc(boundConsColsAdded, _realLP->nCols());

   // NOTE: this loop only goes over the primal columns that are included in the complementary problem, i.e. the
   // columns from the original problem.
   // 29.04.15 in the current implementation, all bound constraints are included in the reduced problem. So, all
   // variables (columns) are included in the reduced problem.
   for(int i = 0; i < _realLP->nCols(); i++)
   {
      boundConsColsAdded[i] = 0;
      assert(_decompCompProbColIDsIdx[i] != -1);

      if(_decompCompProbColIDsIdx[i] != -1)
      {
         int idIndex = _decompCompProbColIDsIdx[i];
         assert(_compSolver.number(SPxRowId(_decompDualRowIDs[idIndex])) >= 0);
         col.add(_compSolver.number(SPxRowId(_decompDualRowIDs[idIndex])), 1.0);

         if(currFixedVars[i] != 0)
         {
            assert(currFixedVars[i] == -1 || currFixedVars[i] == 1);

            assert(_realLP->lower(i) == _solver.lhs(_decompReducedProbColRowIDs[i]));
            assert(_realLP->upper(i) == _solver.rhs(_decompReducedProbColRowIDs[i]));
            R colObjCoeff = 0;

            if(currFixedVars[i] == -1)
               colObjCoeff = _solver.lhs(_decompReducedProbColRowIDs[i]);
            else
               colObjCoeff = _solver.rhs(_decompReducedProbColRowIDs[i]);

            fixedVarsDualCols.add(colObjCoeff, R(-infinity), col, R(infinity));
            numFixedVars++;
         }
         // 09.02.15 I think that the else should only be entered if the column does not exist in the reduced
         // prob. I have tested by just leaving this as an else (without the if), but I think that this is wrong.
         //else if( !_decompReducedProbColRowIDs[i].isValid() )

         // NOTE: in the current implementation all columns, except the free columns, are included int the reduced
         // problem. There in no need to include the variable bounds in the complementary problem.
         else //if( false && _fixedOrigVars[i] == -2 )
         {
            bool isRedProbCol = _decompReducedProbColRowIDs[i].isValid();

            // 29.04.15 in the current implementation only free variables are not included in the reduced problem
            if(GT(_realLP->lower(i), R(-infinity)))
            {
               if(!isRedProbCol)
                  col.add(_compSolver.number(SPxRowId(_compSlackDualRowId)), -SLACKCOEFF);

               boundConsCols.add(_realLP->lower(i), R(-infinity), col, 0.0);

               if(!isRedProbCol)
                  col.remove(col.size() - 1);

               boundConsColsAdded[i]++;
               numBoundConsCols++;
            }

            if(LT(_realLP->upper(i), R(infinity)))
            {
               if(!isRedProbCol)
                  col.add(_compSolver.number(SPxRowId(_compSlackDualRowId)), SLACKCOEFF);

               boundConsCols.add(_realLP->upper(i), 0.0, col, R(infinity));

               if(!isRedProbCol)
                  col.remove(col.size() - 1);

               boundConsColsAdded[i]++;
               numBoundConsCols++;
            }
         }

         col.clear();
         _fixedOrigVars[i] = currFixedVars[i];
      }
   }

   // adding the fixed var dual columns to the complementary problem
   SPxColId* addedcolids = 0;
   spx_alloc(addedcolids, numFixedVars);
   _compSolver.addCols(addedcolids, fixedVarsDualCols);

   SPxColId tempId;
   int addedcolcount = 0;

   tempId.inValidate();

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      if(_fixedOrigVars[i] != 0)
      {
         assert(_fixedOrigVars[i] == -1 || _fixedOrigVars[i] == 1);
         _decompFixedVarDualIDs[i] = addedcolids[addedcolcount];
         addedcolcount++;
      }
      else
         _decompFixedVarDualIDs[i] = tempId;
   }

   // adding the bound cons dual columns to the complementary problem
   SPxColId* addedbndcolids = 0;
   spx_alloc(addedbndcolids, numBoundConsCols);
   _compSolver.addCols(addedbndcolids, boundConsCols);

   addedcolcount = 0;

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      if(boundConsColsAdded[i] > 0)
      {
         for(int j = 0; j < boundConsColsAdded[i]; j++)
         {
            _decompVarBoundDualIDs[i * 2 + j] = addedbndcolids[addedcolcount];
            addedcolcount++;
         }
      }

      switch(boundConsColsAdded[i])
      {
      case 0:
         _decompVarBoundDualIDs[i * 2] = tempId;

      // FALLTHROUGH
      case 1:
         _decompVarBoundDualIDs[i * 2 + 1] = tempId;
         break;
      }
   }

   // freeing allocated memory
   spx_free(addedbndcolids);
   spx_free(addedcolids);
   spx_free(boundConsColsAdded);
}


/// identify the dual columns related to the fixed variables
template <class R>
void SoPlexBase<R>::_identifyComplementaryPrimalFixedPrimalVars(int* currFixedVars)
{
   int numFixedVar = 0;

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      currFixedVars[i] = 0;

      if(!_decompReducedProbColRowIDs[i].isValid())
         continue;

      int rowNumber = _solver.number(_decompReducedProbColRowIDs[i]);

      if(_decompReducedProbColRowIDs[i].isValid() &&
            (_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
             _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_LOWER ||
             _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_FIXED))
      {
         // setting the value of the _fixedOrigVars array to indicate which variables are at their bounds.
         currFixedVars[i] = getOrigVarFixedDirection(i);

         numFixedVar++;
      }
   }

   MSG_INFO3(spxout, spxout <<
             "Number of fixed primal variables in the complementary (primal) problem: "
             << numFixedVar << std::endl);
}



/// updating the dual columns related to the fixed primal variables.
template <class R>
void SoPlexBase<R>::_updateComplementaryPrimalFixedPrimalVars(int* currFixedVars)
{
   int numFixedVars = 0;

   // NOTE: this loop only goes over the primal columns that are included in the complementary problem, i.e. the
   // columns from the original problem.
   // 29.04.15 in the current implementation, all bound constraints are included in the reduced problem. So, all
   // variables (columns) are included in the reduced problem.
   for(int i = 0; i < _nCompPrimalCols; i++)
   {
      int colNumber = _compSolver.number(SPxColId(_decompCompPrimalColIDs[i]));

      if(_fixedOrigVars[colNumber] != currFixedVars[colNumber])
      {
         if(currFixedVars[colNumber] != 0)
         {
            assert(currFixedVars[colNumber] == -1 || currFixedVars[colNumber] == 1);

            if(currFixedVars[colNumber] == -1)
               _compSolver.changeBounds(colNumber, _realLP->lower(SPxColId(_decompPrimalColIDs[i])),
                                        _realLP->lower(SPxColId(_decompPrimalColIDs[i])));
            else
               _compSolver.changeBounds(colNumber, _realLP->upper(SPxColId(_decompPrimalColIDs[i])),
                                        _realLP->upper(SPxColId(_decompPrimalColIDs[i])));

            numFixedVars++;
         }
         else
         {
            _compSolver.changeBounds(colNumber, R(-infinity), R(infinity));
         }
      }

      _fixedOrigVars[colNumber] = currFixedVars[colNumber];
   }
}



/// updating the complementary dual problem with the original objective function
template <class R>
void SoPlexBase<R>::_setComplementaryDualOriginalObjective()
{
   for(int i = 0; i < _realLP->nCols(); i++)
   {
      assert(_decompCompProbColIDsIdx[i] != -1);   // this should be true in the current implementation
      int idIndex = _decompCompProbColIDsIdx[i];
      int compRowNumber = _compSolver.number(_decompDualRowIDs[idIndex]);

      if(_decompCompProbColIDsIdx[i] != -1)
      {
         // In the dual conversion, when a variable has a non-standard bound it is converted to a free variable.
         if(LE(_realLP->lower(i), R(-infinity)) && GE(_realLP->upper(i), R(infinity)))
         {
            // unrestricted variable
            _compSolver.changeLhs(compRowNumber, _realLP->obj(i));
            _compSolver.changeRhs(compRowNumber, _realLP->obj(i));
            assert(LE(_compSolver.lhs(compRowNumber), _compSolver.rhs(compRowNumber)));
         }
         else if(LE(_realLP->lower(i), R(-infinity)))
         {
            // variable with a finite upper bound
            _compSolver.changeRhs(compRowNumber, _realLP->obj(i));

            if(isZero(_realLP->upper(i)))
               _compSolver.changeLhs(compRowNumber, R(-infinity));
            else
               _compSolver.changeLhs(compRowNumber, _realLP->obj(i));
         }
         else if(GE(_realLP->upper(i), R(infinity)))
         {
            // variable with a finite lower bound
            _compSolver.changeLhs(compRowNumber, _realLP->obj(i));

            if(isZero(_realLP->upper(i)))
               _compSolver.changeRhs(compRowNumber, R(infinity));
            else
               _compSolver.changeRhs(compRowNumber, _realLP->obj(i));
         }
         else if(NE(_realLP->lower(i), _realLP->upper(i)))
         {
            // variable with a finite upper and lower bound
            if(isZero(_realLP->upper(i)))
            {
               _compSolver.changeLhs(compRowNumber, _realLP->obj(i));
               _compSolver.changeRhs(compRowNumber, R(infinity));
            }
            else if(isZero(_realLP->upper(i)))
            {
               _compSolver.changeLhs(compRowNumber, R(-infinity));
               _compSolver.changeRhs(compRowNumber, _realLP->obj(i));
            }
            else
            {
               _compSolver.changeLhs(compRowNumber, _realLP->obj(i));
               _compSolver.changeRhs(compRowNumber, _realLP->obj(i));
            }
         }
         else
         {
            // fixed variable
            _compSolver.changeLhs(compRowNumber, _realLP->obj(i));
            _compSolver.changeRhs(compRowNumber, _realLP->obj(i));
         }
      }
   }

   // removing the complementary problem slack column dual row
   _compSolver.removeRow(_compSlackDualRowId);
}



/// updating the complementary primal problem with the original objective function
template <class R>
void SoPlexBase<R>::_setComplementaryPrimalOriginalObjective()
{
   // the comp solver has not removed any columns. Only the slack variables have been added.
   assert(_realLP->nCols() == _compSolver.nCols() - 1);

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      int colNumber = _realLP->number(_decompPrimalColIDs[i]);
      int compColNumber = _compSolver.number(_decompCompPrimalColIDs[i]);
      _compSolver.changeObj(compColNumber, _realLP->maxObj(colNumber));
   }

   // removing the complementary problem slack column dual row
   _compSolver.removeCol(_compSlackColId);
}



/// determining which bound the primal variables will be fixed to.
template <class R>
int SoPlexBase<R>::getOrigVarFixedDirection(int colNum)
{
   if(!_decompReducedProbColRowIDs[colNum].isValid())
      return 0;

   int rowNumber = _solver.number(_decompReducedProbColRowIDs[colNum]);

   // setting the value of the _fixedOrigVars array to indicate which variables are at their bounds.
   if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_UPPER ||
         _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_FIXED ||
         _solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_FREE)
   {
      assert(_solver.rhs(rowNumber) < R(infinity));
      return 1;
   }
   else if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_LOWER)
   {
      assert(_solver.lhs(rowNumber) > R(-infinity));
      return -1;
   }

   return 0;
}



// @todo update this function and related comments. It has only been hacked together.
/// checks result of the solving process and solves again without preprocessing if necessary
// @todo need to evaluate the solution to ensure that it is solved to optimality and then we are able to perform the
// next steps in the algorithm.
template <class R>
void SoPlexBase<R>::_evaluateSolutionDecomp(SPxSolverBase<R>& solver, SLUFactor<R>& sluFactor,
      typename SPxSimplifier<R>::Result result)
{
   typename SPxSolverBase<R>::Status solverStat = SPxSolverBase<R>::UNKNOWN;

   if(result == SPxSimplifier<R>::INFEASIBLE)
      solverStat = SPxSolverBase<R>::INFEASIBLE;
   else if(result == SPxSimplifier<R>::DUAL_INFEASIBLE)
      solverStat = SPxSolverBase<R>::INForUNBD;
   else if(result == SPxSimplifier<R>::UNBOUNDED)
      solverStat = SPxSolverBase<R>::UNBOUNDED;
   else if(result == SPxSimplifier<R>::VANISHED)
      solverStat = SPxSolverBase<R>::OPTIMAL;
   else if(result == SPxSimplifier<R>::OKAY)
      solverStat = solver.status();

   // updating the status of SoPlexBase if the problem solved is the reduced problem.
   if(_currentProb == DECOMP_ORIG || _currentProb == DECOMP_RED)
      _status = solverStat;

   // process result
   // coverity[switch_selector_expr_is_constant]
   switch(solverStat)
   {
   case SPxSolverBase<R>::OPTIMAL:
      if(!_isRealLPLoaded)
      {
         solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
         _decompResolveWithoutPreprocessing(solver, sluFactor, result);
         // Need to solve the complementary problem
         return;
      }
      else
         _hasBasis = true;

      break;

   case SPxSolverBase<R>::UNBOUNDED:
   case SPxSolverBase<R>::INFEASIBLE:
   case SPxSolverBase<R>::INForUNBD:

      // in case of infeasibility or unboundedness, we currently can not unsimplify, but have to solve the original LP again
      if(!_isRealLPLoaded)
      {
         solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
         _decompSimplifyAndSolve(solver, sluFactor, false, false);
         return;
      }
      else
         _hasBasis = (solver.basis().status() > SPxBasisBase<R>::NO_PROBLEM);

      break;

   case SPxSolverBase<R>::ABORT_DECOMP:
   case SPxSolverBase<R>::ABORT_EXDECOMP:

      // in the initialisation of the decomposition simplex, we want to keep the current basis.
      if(!_isRealLPLoaded)
      {
         solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
         _decompResolveWithoutPreprocessing(solver, sluFactor, result);
         //_decompSimplifyAndSolve(solver, sluFactor, false, false);
         return;
      }
      else
         _hasBasis = (solver.basis().status() > SPxBasisBase<R>::NO_PROBLEM);

      break;

   case SPxSolverBase<R>::SINGULAR:
   case SPxSolverBase<R>::ABORT_CYCLING:

      // in case of infeasibility or unboundedness, we currently can not unsimplify, but have to solve the original LP again
      if(!_isRealLPLoaded)
      {
         solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
         _decompSimplifyAndSolve(solver, sluFactor, false, false);
         return;
      }

      break;


   // else fallthrough
   case SPxSolverBase<R>::ABORT_TIME:
   case SPxSolverBase<R>::ABORT_ITER:
   case SPxSolverBase<R>::ABORT_VALUE:
   case SPxSolverBase<R>::REGULAR:
   case SPxSolverBase<R>::RUNNING:

      // store regular basis if there is no simplifier and the original problem is not in the solver because of
      // scaling; non-optimal bases should currently not be unsimplified
      if(_simplifier == 0 && solver.basis().status() > SPxBasisBase<R>::NO_PROBLEM)
      {
         _basisStatusRows.reSize(_decompLP->nRows());
         _basisStatusCols.reSize(_decompLP->nCols());
         assert(_basisStatusRows.size() == solver.nRows());
         assert(_basisStatusCols.size() == solver.nCols());

         solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());
         _hasBasis = true;
      }
      else
         _hasBasis = false;

      break;

   default:
      _hasBasis = false;
      break;
   }
}




/// checks the dual feasibility of the current basis
template <class R>
bool SoPlexBase<R>::checkBasisDualFeasibility(VectorBase<R> feasVec)
{
   assert(_solver.rep() == SPxSolverBase<R>::ROW);
   assert(_solver.spxSense() == SPxLPBase<R>::MAXIMIZE);

   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   // iterating through all elements in the basis to determine whether the dual feasibility is satisfied.
   for(int i = 0; i < _solver.nCols(); i++)
   {
      if(_solver.basis().baseId(i).isSPxRowId())   // find the row id's for rows in the basis
      {
         int rownumber = _solver.number(_solver.basis().baseId(i));

         if(_solver.basis().desc().rowStatus(rownumber) != SPxBasisBase<R>::Desc::P_ON_UPPER &&
               _solver.basis().desc().rowStatus(rownumber) != SPxBasisBase<R>::Desc::P_FIXED)
         {
            if(GT(feasVec[i], (R) 0, feastol))
               return false;
         }

         if(_solver.basis().desc().rowStatus(rownumber) != SPxBasisBase<R>::Desc::P_ON_LOWER &&
               _solver.basis().desc().rowStatus(rownumber) != SPxBasisBase<R>::Desc::P_FIXED)
         {
            if(LT(feasVec[i], (R) 0, feastol))
               return false;
         }

      }
      else if(_solver.basis().baseId(
                 i).isSPxColId())    // get the column id's for the columns in the basis
      {
         int colnumber = _solver.number(_solver.basis().baseId(i));

         if(_solver.basis().desc().colStatus(colnumber) != SPxBasisBase<R>::Desc::P_ON_UPPER &&
               _solver.basis().desc().colStatus(colnumber) != SPxBasisBase<R>::Desc::P_FIXED)
         {
            if(GT(feasVec[i], (R) 0, feastol))
               return false;
         }

         if(_solver.basis().desc().colStatus(colnumber) != SPxBasisBase<R>::Desc::P_ON_LOWER &&
               _solver.basis().desc().colStatus(colnumber) != SPxBasisBase<R>::Desc::P_FIXED)
         {
            if(LT(feasVec[i], (R) 0, feastol))
               return false;
         }
      }
   }

   return true;
}



/// returns the expected sign of the dual variables for the reduced problem
template <class R>
typename SoPlexBase<R>::DualSign SoPlexBase<R>::getExpectedDualVariableSign(int rowNumber)
{
   if(_solver.isRowBasic(rowNumber))
   {
      if(_solver.basis().desc().rowStatus(rowNumber) != SPxBasisBase<R>::Desc::P_ON_UPPER &&
            _solver.basis().desc().rowStatus(rowNumber) != SPxBasisBase<R>::Desc::P_FIXED)
         return SoPlexBase<R>::IS_NEG;

      if(_solver.basis().desc().rowStatus(rowNumber) != SPxBasisBase<R>::Desc::P_ON_LOWER &&
            _solver.basis().desc().rowStatus(rowNumber) != SPxBasisBase<R>::Desc::P_FIXED)
         return SoPlexBase<R>::IS_POS;
   }

   return SoPlexBase<R>::IS_FREE;
}



/// returns the expected sign of the dual variables for the original problem
template <class R>
typename SoPlexBase<R>::DualSign SoPlexBase<R>::getOrigProbDualVariableSign(int rowNumber)
{
   if(_realLP->rowType(rowNumber) == LPRowBase<R>::LESS_EQUAL)
      return IS_POS;

   if(_realLP->rowType(rowNumber) == LPRowBase<R>::GREATER_EQUAL)
      return IS_NEG;

   if(_realLP->rowType(rowNumber) == LPRowBase<R>::EQUAL)
      return IS_FREE;

   // this final statement needs to be checked.
   if(_realLP->rowType(rowNumber) == LPRowBase<R>::RANGE)
      return IS_FREE;

   return IS_FREE;
}


/// print display line of flying table
template <class R>
void SoPlexBase<R>::printDecompDisplayLine(SPxSolverBase<R>& solver,
      const SPxOut::Verbosity origVerb, bool force, bool forceHead)
{
   // setting the verbosity level
   const SPxOut::Verbosity currVerb = spxout.getVerbosity();
   spxout.setVerbosity(origVerb);

   int displayFreq = intParam(SoPlexBase<R>::DECOMP_DISPLAYFREQ);

   MSG_INFO1(spxout,

             if(forceHead || (_decompDisplayLine % (displayFreq * 30) == 0))
{
   spxout << "type |   time |   iters | red iter | alg iter |     rows |     cols |  shift   |    value\n";
}
if(force || (_decompDisplayLine % displayFreq == 0))
{
   Real currentTime = _statistics->solvingTime->time();
      (solver.type() == SPxSolverBase<R>::LEAVE) ? spxout << "  L  |" : spxout << "  E  |";
      spxout << std::fixed << std::setw(7) << std::setprecision(1) << currentTime << " |";
      spxout << std::scientific << std::setprecision(2);
      spxout << std::setw(8) << _statistics->iterations << " | ";
      spxout << std::scientific << std::setprecision(2);
      spxout << std::setw(8) << _statistics->iterationsRedProb << " | ";
      spxout << std::scientific << std::setprecision(2);
      spxout << std::setw(8) << _statistics->callsReducedProb << " | ";
      spxout << std::scientific << std::setprecision(2);
      spxout << std::setw(8) << numIncludedRows << " | ";
      spxout << std::scientific << std::setprecision(2);
      spxout << std::setw(8) << solver.nCols() << " | "
             << solver.shift() << " | "
             << std::setprecision(8) << solver.value() + solver.objOffset()
             << std::endl;

   }
   _decompDisplayLine++;
            );

   spxout.setVerbosity(currVerb);
}



/// stores the problem statistics of the original problem
template <class R>
void SoPlexBase<R>::getOriginalProblemStatistics()
{
   numProbRows = _realLP->nRows();
   numProbCols = _realLP->nCols();
   nNonzeros = _realLP->nNzos();
   minAbsNonzero = _realLP->minAbsNzo();
   maxAbsNonzero = _realLP->maxAbsNzo();

   origCountLower = 0;
   origCountUpper = 0;
   origCountBoxed = 0;
   origCountFreeCol = 0;

   origCountLhs = 0;
   origCountRhs = 0;
   origCountRanged = 0;
   origCountFreeRow = 0;

   for(int i = 0; i < _realLP->nCols(); i++)
   {
      bool hasLower = false;
      bool hasUpper = false;

      if(_realLP->lower(i) > R(-infinity))
      {
         origCountLower++;
         hasLower = true;
      }

      if(_realLP->upper(i) < R(infinity))
      {
         origCountUpper++;
         hasUpper = true;
      }

      if(hasUpper && hasLower)
      {
         origCountBoxed++;
         origCountUpper--;
         origCountLower--;
      }

      if(!hasUpper && !hasLower)
         origCountFreeCol++;
   }

   for(int i = 0; i < _realLP->nRows(); i++)
   {
      bool hasRhs = false;
      bool hasLhs = false;

      if(_realLP->lhs(i) > R(-infinity))
      {
         origCountLhs++;
         hasLhs = true;
      }

      if(_realLP->rhs(i) < R(infinity))
      {
         origCountRhs++;
         hasRhs = true;
      }

      if(hasRhs && hasLhs)
      {
         if(EQ(_realLP->rhs(i), _realLP->lhs(i)))
            origCountEqual++;
         else
            origCountRanged++;

         origCountLhs--;
         origCountRhs--;
      }

      if(!hasRhs && !hasLhs)
         origCountFreeRow++;
   }
}

template <class R>
void SoPlexBase<R>::printOriginalProblemStatistics(std::ostream& os)
{
   os << "  Columns           : " << numProbCols << "\n"
      << "              boxed : " << origCountBoxed << "\n"
      << "        lower bound : " << origCountLower << "\n"
      << "        upper bound : " << origCountUpper << "\n"
      << "               free : " << origCountFreeCol << "\n"
      << "  Rows              : " << numProbRows << "\n"
      << "              equal : " << origCountEqual << "\n"
      << "             ranged : " << origCountRanged << "\n"
      << "                lhs : " << origCountLhs << "\n"
      << "                rhs : " << origCountRhs << "\n"
      << "               free : " << origCountFreeRow << "\n"
      << "  Nonzeros          : " << nNonzeros << "\n"
      << "         per column : " << R(nNonzeros) / R(numProbCols) << "\n"
      << "            per row : " << R(nNonzeros) / R(numProbRows) << "\n"
      << "           sparsity : " << R(nNonzeros) / R(numProbCols) / R(numProbRows) << "\n"
      << "    min. abs. value : " << R(minAbsNonzero) << "\n"
      << "    max. abs. value : " << R(maxAbsNonzero) << "\n";
}



/// gets the coefficient of the slack variable in the primal complementary problem
template <class R>
R SoPlexBase<R>::getCompSlackVarCoeff(int primalRowNum)
{
   int indDir = 1;

   switch(_realLP->rowType(_decompPrimalRowIDs[primalRowNum]))
   {
   // NOTE: check the sign of the slackCoeff for the Range constraints. This will depend on the method of
   // dual conversion.
   case LPRowBase<R>::RANGE:
      assert((primalRowNum < _nPrimalRows - 1
              && _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum])) ==
              _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum + 1]))) ||
             (primalRowNum > 0 && _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum - 1])) ==
              _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum]))));

      // determine with primalRowNum and primalRowNum+1 or primalRowNum-1 and primalRowNum have the same row id.
      if(_realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum - 1])) ==
            _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum])))
         indDir = -1;

      if(_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[primalRowNum]))) <
            _compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[primalRowNum + indDir]))))
         return -SLACKCOEFF;
      else
         return SLACKCOEFF;

      break;

   case LPRowBase<R>::GREATER_EQUAL:
      return -SLACKCOEFF;
      break;

   case LPRowBase<R>::EQUAL:
      assert((primalRowNum < _nPrimalRows - 1
              && _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum])) ==
              _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum + 1]))) ||
             (primalRowNum > 0 && _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum - 1])) ==
              _realLP->number(SPxColId(_decompPrimalRowIDs[primalRowNum]))));

   // FALLTHROUGH
   case LPRowBase<R>::LESS_EQUAL:
      return SLACKCOEFF;
      break;

   default:
      throw SPxInternalCodeException("XDECOMPSL01 This should never happen.");
   }
}



/// gets violation of bounds; returns true on success
template <class R>
bool SoPlexBase<R>::getDecompBoundViolation(R& maxviol, R& sumviol)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   VectorBase<R>& primal = _solReal._primal;
   assert(primal.dim() == _realLP->nCols());

   _nDecompViolBounds = 0;

   maxviol = 0.0;
   sumviol = 0.0;

   bool isViol = false;
   bool isMaxViol = false;

   for(int i = _realLP->nCols() - 1; i >= 0; i--)
   {
      R currviol = 0.0;

      R viol = _realLP->lower(i) - primal[i];

      isViol = false;
      isMaxViol = false;

      if(viol > 0.0)
      {
         sumviol += viol;

         if(viol > maxviol)
         {
            maxviol = viol;
            isMaxViol = true;
         }

         if(currviol < viol)
            currviol = viol;
      }

      if(GT(viol, R(0.0), feastol))
         isViol = true;

      viol = primal[i] - _realLP->upper(i);

      if(viol > 0.0)
      {
         sumviol += viol;

         if(viol > maxviol)
         {
            maxviol = viol;
            isMaxViol = true;
         }

         if(currviol < viol)
            currviol = viol;
      }

      if(GT(viol, R(0.0), feastol))
         isViol = true;

      if(isViol)
      {
         // updating the violated bounds list
         if(isMaxViol)
         {
            _decompViolatedBounds[_nDecompViolBounds] = _decompViolatedBounds[0];
            _decompViolatedBounds[0] = i;
         }
         else
            _decompViolatedBounds[_nDecompViolBounds] = i;

         _nDecompViolBounds++;
      }
   }

   return true;
}



/// gets violation of constraints; returns true on success
template <class R>
bool SoPlexBase<R>::getDecompRowViolation(R& maxviol, R& sumviol)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);

   VectorBase<R>& primal = _solReal._primal;
   assert(primal.dim() == _realLP->nCols());

   VectorBase<R> activity(_realLP->nRows());
   _realLP->computePrimalActivity(primal, activity);

   _nDecompViolRows = 0;

   maxviol = 0.0;
   sumviol = 0.0;

   bool isViol = false;
   bool isMaxViol = false;

   for(int i = _realLP->nRows() - 1; i >= 0; i--)
   {
      R currviol = 0.0;

      isViol = false;
      isMaxViol = false;

      R viol = _realLP->lhs(i) - activity[i];

      if(viol > 0.0)
      {
         sumviol += viol;

         if(viol > maxviol)
         {
            maxviol = viol;
            isMaxViol = true;
         }

         if(viol > currviol)
            currviol = viol;
      }

      if(GT(viol, R(0.0), feastol))
         isViol = true;

      viol = activity[i] - _realLP->rhs(i);

      if(viol > 0.0)
      {
         sumviol += viol;

         if(viol > maxviol)
         {
            maxviol = viol;
            isMaxViol = true;
         }

         if(viol > currviol)
            currviol = viol;
      }

      if(GT(viol, R(0.0), feastol))
         isViol = true;

      if(isViol)
      {
         // updating the violated rows list
         if(isMaxViol)
         {
            _decompViolatedRows[_nDecompViolRows] = _decompViolatedRows[0];
            _decompViolatedRows[0] = i;
         }
         else
            _decompViolatedRows[_nDecompViolRows] = i;

         _nDecompViolRows++;
      }
   }

   return true;
}



/// function call to terminate the decomposition simplex
template <class R>
bool SoPlexBase<R>::decompTerminate(R timeLimit)
{
   R maxTime = timeLimit;

   // check if a time limit is actually set
   if(maxTime < 0 || maxTime >= R(infinity))
      return false;

   Real currentTime = _statistics->solvingTime->time();

   if(currentTime >= maxTime)
   {
      MSG_INFO2(spxout, spxout << " --- timelimit (" << _solver.getMaxTime()
                << ") reached" << std::endl;)
      _solver.setSolverStatus(SPxSolverBase<R>::ABORT_TIME);
      return true;
   }

   return false;
}



/// function to retrieve the original problem row basis status from the reduced and complementary problems
template <class R>
void SoPlexBase<R>::getOriginalProblemBasisRowStatus(DataArray< int >& degenerateRowNums,
      DataArray< typename SPxSolverBase<R>::VarStatus >& degenerateRowStatus, int& nDegenerateRows,
      int& nNonBasicRows)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
   int basicRow = 0;

   // looping over all rows not contained in the complementary problem
   for(int i = 0; i < _nElimPrimalRows; i++)
   {
      int rowNumber = _realLP->number(_decompElimPrimalRowIDs[i]);

      int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
      assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

      if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER) /*||
                                                                                                   (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                                                                                                   EQ(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], 0.0, feastol)) )*/
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_UPPER;
      }
      else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER) /*||
                                                                                                        (_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                                                                                                        EQ(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), 0.0, feastol)) )*/
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_LOWER;
      }
      else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED)
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::FIXED;
      }
      else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FREE)
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::ZERO;
      }
      else
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
         basicRow++;
      }
   }

   for(int i = 0; i < _nPrimalRows; i++)
   {
      int rowNumber = _realLP->number(_decompPrimalRowIDs[i]);

      // this loop runs over all rows previously in the complementary problem.
      if(_decompReducedProbRows[rowNumber])
      {
         int solverRowNum = _solver.number(_decompReducedProbRowIDs[rowNumber]);
         assert(solverRowNum >= 0 && solverRowNum < _solver.nRows());

         if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_UPPER)
         {
            _basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_UPPER;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                 EQ(_solver.rhs(solverRowNum) - _solver.pVec()[solverRowNum], 0.0, feastol))
         {
            // if a row is non basic, but is at its bound then the row number and status is stored
            degenerateRowNums[nDegenerateRows] = rowNumber;
            degenerateRowStatus[nDegenerateRows] = SPxSolverBase<R>::ON_UPPER;
            nDegenerateRows++;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_ON_LOWER)
         {
            _basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_LOWER;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                 EQ(_solver.pVec()[solverRowNum] - _solver.lhs(solverRowNum), 0.0, feastol))
         {
            // if a row is non basic, but is at its bound then the row number and status is stored
            degenerateRowNums[nDegenerateRows] = rowNumber;
            degenerateRowStatus[nDegenerateRows] = SPxSolverBase<R>::ON_LOWER;
            nDegenerateRows++;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FIXED)
         {
            _basisStatusRows[rowNumber] = SPxSolverBase<R>::FIXED;
         }
         else if(_solver.basis().desc().rowStatus(solverRowNum) == SPxBasisBase<R>::Desc::P_FREE)
         {
            _basisStatusRows[rowNumber] = SPxSolverBase<R>::ZERO;
         }
         else
         {
            _basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
            basicRow++;
         }
      }
      else
      {
         _basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
         basicRow++;

         switch(_realLP->rowType(_decompPrimalRowIDs[i]))
         {
         case LPRowBase<R>::RANGE:
            //assert(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
            //_realLP->number(SPxColId(_decompPrimalRowIDs[i+1])));
            //assert(_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i]))) !=
            //_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))));

            //// NOTE: This is not correct at present. Need to modify the inequalities. Look at the dual conversion
            //// function to get this correct.
            //if( _compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i]))) <
            //_compSolver.obj(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))))
            //{
            //if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i]))) ==
            //SPxBasisBase<R>::Desc::D_ON_LOWER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_LOWER;
            //}
            //else if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))) ==
            //SPxBasisBase<R>::Desc::D_ON_UPPER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_UPPER;
            //}
            //else
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
            //basicRow++;
            //}
            //}
            //else
            //{
            //if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i]))) ==
            //SPxBasisBase<R>::Desc::D_ON_UPPER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_UPPER;
            //}
            //else if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i + 1]))) ==
            //SPxBasisBase<R>::Desc::D_ON_LOWER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_LOWER;
            //}
            //else
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
            //basicRow++;
            //}
            //}

            i++;
            break;

         case LPRowBase<R>::EQUAL:
            //assert(_realLP->number(SPxColId(_decompPrimalRowIDs[i])) ==
            //_realLP->number(SPxColId(_decompPrimalRowIDs[i+1])));

            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::FIXED;

            i++;
            break;

         case LPRowBase<R>::GREATER_EQUAL:
            //if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i]))) ==
            //SPxBasisBase<R>::Desc::D_ON_LOWER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_LOWER;
            //}
            //else
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
            //basicRow++;
            //}

            break;

         case LPRowBase<R>::LESS_EQUAL:
            //if( _compSolver.basis().desc().rowStatus(_compSolver.number(SPxColId(_decompDualColIDs[i]))) ==
            //SPxBasisBase<R>::Desc::D_ON_UPPER )
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::ON_UPPER;
            //}
            //else
            //{
            //_basisStatusRows[rowNumber] = SPxSolverBase<R>::BASIC;
            //basicRow++;
            //}

            break;

         default:
            throw SPxInternalCodeException("XDECOMPSL01 This should never happen.");
         }
      }
   }

   nNonBasicRows = _realLP->nRows() - basicRow - nDegenerateRows;
   MSG_INFO2(spxout, spxout << "Number of non-basic rows: " << nNonBasicRows << " (from "
             << _realLP->nRows() << ")" << std::endl);
}


/// function to retrieve the column status for the original problem basis from the reduced and complementary problems
// all columns are currently in the reduced problem, so it is only necessary to retrieve the status from there.
// However, a transformation of the columns has been made, so it is only possible to retrieve the status from the
// variable bound constraints.
template <class R>
void SoPlexBase<R>::getOriginalProblemBasisColStatus(int& nNonBasicCols)
{
   R feastol = realParam(SoPlexBase<R>::FEASTOL);
   int basicCol = 0;
   int numZeroDual = 0;

   // looping over all variable bound constraints
   for(int i = 0; i < _realLP->nCols(); i++)
   {
      if(!_decompReducedProbColRowIDs[i].isValid())
         continue;

      int rowNumber = _solver.number(_decompReducedProbColRowIDs[i]);

      if(_decompReducedProbColRowIDs[i].isValid())
      {
         if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_UPPER) /*||
                                                                                                    (_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_ON_LOWER &&
                                                                                                    EQ(_solver.rhs(rowNumber) - _solver.pVec()[rowNumber], 0.0, feastol)) )*/
         {
            _basisStatusCols[i] = SPxSolverBase<R>::ON_UPPER;
         }
         else if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_ON_LOWER) /*||
                                                                                                         (_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::D_ON_UPPER &&
                                                                                                         EQ(_solver.pVec()[rowNumber] - _solver.lhs(rowNumber), 0.0, feastol)) )*/
         {
            _basisStatusCols[i] = SPxSolverBase<R>::ON_LOWER;
         }
         else if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_FIXED)
         {
            _basisStatusCols[i] = SPxSolverBase<R>::FIXED;
         }
         else if(_solver.basis().desc().rowStatus(rowNumber) == SPxBasisBase<R>::Desc::P_FREE)
         {
            _basisStatusCols[i] = SPxSolverBase<R>::ZERO;
         }
         else
         {
            _basisStatusCols[i] = SPxSolverBase<R>::BASIC;
            basicCol++;
         }
      }

      if(EQ(_solver.fVec()[i], 0.0, feastol))
         numZeroDual++;
   }

   nNonBasicCols = _realLP->nCols() - basicCol;
   MSG_INFO2(spxout, spxout << "Number of non-basic columns: "
             << nNonBasicCols << " (from " << _realLP->nCols() << ")" << std::endl
             << "Number of zero dual columns: " << numZeroDual << " (from " << _realLP->nCols() << ")" <<
             std::endl);
}



/// function to build a basis for the original problem as given by the solution to the reduced problem
template <class R>
void SoPlexBase<R>::_writeOriginalProblemBasis(const char* filename, NameSet* rowNames,
      NameSet* colNames, bool cpxFormat)
{
   int numrows = _realLP->nRows();
   int numcols = _realLP->nCols();
   int nNonBasicRows = 0;
   int nNonBasicCols = 0;
   int nDegenerateRows = 0;
   DataArray< int > degenerateRowNums;   // array to store the orig prob row number of degenerate rows
   DataArray< typename SPxSolverBase<R>::VarStatus >
   degenerateRowStatus;  // possible status for the degenerate rows in the orig prob

   // setting the basis status rows and columns to size of the original problem
   _basisStatusRows.reSize(numrows);
   _basisStatusCols.reSize(numcols);

   degenerateRowNums.reSize(numrows);
   degenerateRowStatus.reSize(numrows);

   // setting the row and column status for the original problem
   getOriginalProblemBasisRowStatus(degenerateRowNums, degenerateRowStatus, nDegenerateRows,
                                    nNonBasicRows);
   getOriginalProblemBasisColStatus(nNonBasicCols);

   // checking the consistency of the non-basic rows and columns for printing out the basis.
   // it is necessary to have numcols of non-basic rows and columns.
   // if there are not enought non-basic rows and columns, then the degenerate rows are set as non-basic.
   // all degenerate rows that are not changed to non-basic must be set to basic.
   assert(nDegenerateRows + nNonBasicRows + nNonBasicCols >= numcols);
   int degenRowsSetNonBasic = 0;

   for(int i = 0; i < nDegenerateRows; i++)
   {
      if(nNonBasicRows + nNonBasicCols + degenRowsSetNonBasic < numcols)
      {
         _basisStatusRows[degenerateRowNums[i]] = degenerateRowStatus[i];
         degenRowsSetNonBasic++;
      }
      else
         _basisStatusRows[degenerateRowNums[i]] = SPxSolverBase<R>::BASIC;
   }

   // writing the basis file for the original problem
   bool wasRealLPLoaded = _isRealLPLoaded;
   _isRealLPLoaded = false;
   writeBasisFile(filename, rowNames, colNames, cpxFormat);
   _isRealLPLoaded = wasRealLPLoaded;
}
} // namespace soplex
