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

#include "soplex.h"
#include "soplex/statistics.h"

#define ALLOWED_UNSCALE_PERCENTAGE    0.1
#define MIN_OPT_CALLS_WITH_SCALING     10

namespace soplex
{
/// solves real LP
template <class R>
void SoPlexBase<R>::_optimize(volatile bool* interrupt)
{
   assert(_realLP != 0);
   assert(_realLP == &_solver);

   _solReal.invalidate();
   ++_optimizeCalls;

   // start timing
   _statistics->solvingTime->start();

   if(boolParam(SoPlexBase<R>::PERSISTENTSCALING))
   {
      // scale original problem; overwriting _realLP
      if(_scaler && !_realLP->isScaled() && _reapplyPersistentScaling())
      {
#ifdef SOPLEX_DEBUG
         SPxLPBase<R>* origLP = 0;
         spx_alloc(origLP);
         origLP = new(origLP) SPxLPBase<R>(*_realLP);
#endif
         _scaler->scale(*_realLP, true);
         _isRealLPScaled = _realLP->isScaled(); // a scaler might decide not to apply scaling
         _solver.invalidateBasis();
#ifdef SOPLEX_DEBUG
         _checkScaling(origLP);
#endif
      }
      // unscale previously scaled problem, overwriting _realLP
      else if(!_scaler && _realLP->isScaled())
      {
         _solver.unscaleLPandReloadBasis();
         _isRealLPScaled = false;
         ++_unscaleCalls;
      }
   }

   // remember that last solve was in floating-point
   _lastSolveMode = SOLVEMODE_REAL;

   // solve and store solution; if we have a starting basis, do not apply preprocessing; if we are solving from
   // scratch, apply preprocessing according to parameter settings
   if(!_hasBasis && realParam(SoPlexBase<R>::OBJLIMIT_LOWER) == -realParam(SoPlexBase<R>::INFTY)
         && realParam(SoPlexBase<R>::OBJLIMIT_UPPER) == realParam(SoPlexBase<R>::INFTY))
      _preprocessAndSolveReal(true, interrupt);
   else
      _preprocessAndSolveReal(false, interrupt);

   _statistics->finalBasisCondition = _solver.getBasisMetric(0);

   // stop timing
   _statistics->solvingTime->stop();
}



/// check whether persistent scaling is supposed to be reapplied again after unscaling
template <class R>
bool SoPlexBase<R>::_reapplyPersistentScaling() const
{
   if((_unscaleCalls > _optimizeCalls * ALLOWED_UNSCALE_PERCENTAGE)
         && _optimizeCalls > MIN_OPT_CALLS_WITH_SCALING)
      return false;
   else
      return true;
}



/// checks result of the solving process and solves again without preprocessing if necessary
template <class R>
void SoPlexBase<R>::_evaluateSolutionReal(typename SPxSimplifier<R>::Result simplificationStatus)
{
   // if the simplifier detected infeasibility or unboundedness we optimize again
   // just to get the proof (primal or dual ray)
   // todo get infeasibility proof from simplifier
   switch(simplificationStatus)
   {
   case SPxSimplifier<R>::INFEASIBLE:
   case SPxSimplifier<R>::DUAL_INFEASIBLE:
   case SPxSimplifier<R>::UNBOUNDED:
      _hasBasis = false;

      if(boolParam(SoPlexBase<R>::ENSURERAY))
      {
         MSG_INFO1(spxout, spxout <<
                   "simplifier detected infeasibility or unboundedness - solve again without simplifying" << std::endl;
                  )
         _preprocessAndSolveReal(false);
      }
      else
      {
         if(simplificationStatus == SPxSimplifier<R>::INFEASIBLE)
            _status = SPxSolverBase<R>::INFEASIBLE;
         else if(simplificationStatus == SPxSimplifier<R>::UNBOUNDED)
            _status = SPxSolverBase<R>::UNBOUNDED;
         else
            _status = SPxSolverBase<R>::INForUNBD;

         // load original LP to restore clean solver state
         _loadRealLP(false);
      }

      return;

   case SPxSimplifier<R>::VANISHED:
      _status = SPxSolverBase<R>::OPTIMAL;
      _storeSolutionRealFromPresol();
      return;

   case SPxSimplifier<R>::OKAY:
      _status = _solver.status();
   }

   // process result
   switch(_status)
   {
   case SPxSolverBase<R>::OPTIMAL:
      _storeSolutionReal(!_isRealLPLoaded || _isRealLPScaled);

      // apply polishing on original problem
      if(_applyPolishing)
      {
         int polishing = intParam(SoPlexBase<R>::SOLUTION_POLISHING);
         setIntParam(SoPlexBase<R>::SOLUTION_POLISHING, polishing);
         _preprocessAndSolveReal(false);
      }

      break;

   case SPxSolverBase<R>::UNBOUNDED:
   case SPxSolverBase<R>::INFEASIBLE:
   case SPxSolverBase<R>::INForUNBD:

      // in case of infeasibility or unboundedness, we currently can not unsimplify, but have to solve the original LP again
      if(!_isRealLPLoaded && boolParam(SoPlexBase<R>::ENSURERAY))
      {
         MSG_INFO1(spxout, spxout << " --- loading original problem" << std::endl;)
         _solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
         // we cannot do more to remove violations
         _resolveWithoutPreprocessing(simplificationStatus);
      }
      else
      {
         _storeSolutionReal(false);
      }

      break;

   case SPxSolverBase<R>::SINGULAR:

      // if preprocessing was applied, try to run again without to avoid singularity
      if(!_isRealLPLoaded)
      {
         MSG_INFO1(spxout, spxout << "encountered singularity - trying to solve again without simplifying" <<
                   std::endl;)
         _preprocessAndSolveReal(false);
         return;
      }

      _hasBasis = false;
      break;

   case SPxSolverBase<R>::ABORT_CYCLING:

      // if preprocessing or scaling was applied, try to run again without to
      // avoid cycling
      if(!_isRealLPLoaded || _isRealLPScaled)
      {
         MSG_INFO1(spxout, spxout << "encountered cycling - trying to solve again without simplifying" <<
                   std::endl;)
         // store and unsimplify sub-optimal solution and basis, may trigger re-solve
         _storeSolutionReal(true);
         return;
      }

      if(_solReal.isPrimalFeasible() || _solReal.isDualFeasible())
         _status = SPxSolverBase<R>::OPTIMAL_UNSCALED_VIOLATIONS;

   // FALLTHROUGH
   case SPxSolverBase<R>::ABORT_TIME:
   case SPxSolverBase<R>::ABORT_ITER:
   case SPxSolverBase<R>::ABORT_VALUE:
   case SPxSolverBase<R>::REGULAR:
   case SPxSolverBase<R>::RUNNING:

      // If we aborted the solve for some reason and there is still a shift, ensure that the basis status is correct
      if(_solver.shift() > _solver.epsilon())
         _solver.setBasisStatus(SPxBasisBase<R>::REGULAR);

      _storeSolutionReal(false);
      break;

   default:
      _hasBasis = false;
      break;
   }
}



/// solves real LP with/without preprocessing
template <class R>
void SoPlexBase<R>::_preprocessAndSolveReal(bool applySimplifier, volatile bool* interrupt)
{
   _solver.changeObjOffset(realParam(SoPlexBase<R>::OBJ_OFFSET));
   _statistics->preprocessingTime->start();

   _applyPolishing = false;

   if(applySimplifier)
      _enableSimplifierAndScaler();
   else
      _disableSimplifierAndScaler();

   // create a copy of the LP when simplifying or when using internal scaling, i.e. w/o persistent scaling
   bool copyLP = (_simplifier != 0 || (_scaler && !_isRealLPScaled));

   _solver.setTerminationValue(intParam(SoPlexBase<R>::OBJSENSE) == SoPlexBase<R>::OBJSENSE_MINIMIZE
                               ? realParam(SoPlexBase<R>::OBJLIMIT_UPPER) : realParam(SoPlexBase<R>::OBJLIMIT_LOWER));

   if(_isRealLPLoaded)
   {
      assert(_realLP == &_solver);

      // preprocessing is always applied to the LP in the solver; hence we have to create a copy of the original LP
      // if simplifier is turned on
      if(copyLP)
      {
         _realLP = nullptr;
         spx_alloc(_realLP);
         _realLP = new(_realLP) SPxLPBase<R>(_solver);
         _isRealLPLoaded = false;
      }
   }
   else
   {
      assert(_realLP != &_solver);

      // load real LP and basis if available
      if(_hasBasis)
      {
         assert(_basisStatusRows.size() == numRows());
         assert(_basisStatusCols.size() == this->numCols());

         _solver.loadLP(*_realLP, false);
         _solver.setBasis(_basisStatusRows.get_const_ptr(), _basisStatusCols.get_const_ptr());
      }
      // load real LP and set up slack basis
      else
         _solver.loadLP(*_realLP, true);

      // if there is no simplifier, then the original and the transformed problem are identical and it is more
      // memory-efficient to keep only the problem in the solver
      if(!copyLP)
      {
         _realLP->~SPxLPBase<R>();
         spx_free(_realLP);
         _realLP = &_solver;
         _isRealLPLoaded = true;
      }
   }

   // assert that we have two problems if and only if we apply the simplifier
   assert(_realLP == &_solver || copyLP);
   assert(_realLP != &_solver || !copyLP);

   // apply problem simplification
   typename SPxSimplifier<R>::Result simplificationStatus = SPxSimplifier<R>::OKAY;

   if(_simplifier)
   {
      assert(!_isRealLPLoaded);
      // do not remove bounds of boxed variables or sides of ranged rows if bound flipping is used; also respect row-boundflip parameter
      bool keepbounds = intParam(SoPlexBase<R>::RATIOTESTER) == SoPlexBase<R>::RATIOTESTER_BOUNDFLIPPING;

      if(intParam(SoPlexBase<R>::REPRESENTATION) == SoPlexBase<R>::REPRESENTATION_ROW
            || (intParam(SoPlexBase<R>::REPRESENTATION) == SoPlexBase<R>::REPRESENTATION_AUTO
                && (_solver.nCols() + 1) * realParam(SoPlexBase<R>::REPRESENTATION_SWITCH) < (_solver.nRows() + 1)))
         keepbounds &= boolParam(SoPlexBase<R>::ROWBOUNDFLIPS);

      Real remainingTime = _solver.getMaxTime() - _solver.time();
      simplificationStatus = _simplifier->simplify(_solver, realParam(SoPlexBase<R>::EPSILON_ZERO),
                             realParam(SoPlexBase<R>::FEASTOL), realParam(SoPlexBase<R>::OPTTOL), remainingTime, keepbounds,
                             _solver.random.getSeed());
      _solver.changeObjOffset(_simplifier->getObjoffset() + realParam(SoPlexBase<R>::OBJ_OFFSET));
      _solver.setScalingInfo(false);
      _applyPolishing = true;

      _solver.setSolutionPolishing(SPxSolverBase<R>::POLISH_OFF);
   }

   _statistics->preprocessingTime->stop();

   // run the simplex method if problem has not been solved by the simplifier
   if(simplificationStatus == SPxSimplifier<R>::OKAY)
   {
      if(_scaler && !_solver.isScaled())
      {
         _scaler->scale(_solver, false);
         _solver.invalidateBasis();
      }

      _solveRealLPAndRecordStatistics(interrupt);
   }

   _evaluateSolutionReal(simplificationStatus);
}



/// loads original problem into solver and solves again after it has been solved to infeasibility or unboundedness with preprocessing
template <class R>
void SoPlexBase<R>::_resolveWithoutPreprocessing(typename SPxSimplifier<R>::Result
      simplificationStatus)
{
   assert(!_isRealLPLoaded || _scaler != nullptr);
   assert(_simplifier != 0 || _scaler != nullptr);
   assert(_status == SPxSolverBase<R>::INFEASIBLE || _status == SPxSolverBase<R>::INForUNBD
          || _status == SPxSolverBase<R>::UNBOUNDED);

   // if simplifier was active, then we unsimplify to get the basis
   if(_simplifier)
   {
      assert(!_simplifier->isUnsimplified());
      assert(simplificationStatus == SPxSimplifier<R>::OKAY);

      // get temporary solution vectors for transformed problem
      VectorBase<R> primal(_solver.nCols());
      VectorBase<R> slacks(_solver.nRows());
      VectorBase<R> dual(_solver.nRows());
      VectorBase<R> redCost(_solver.nCols());

      _basisStatusRows.reSize(numRows());
      _basisStatusCols.reSize(numCols());
      assert(_basisStatusRows.size() >= _solver.nRows());
      assert(_basisStatusCols.size() >= _solver.nCols());

      // get solution data from transformed problem
      _solver.getPrimalSol(primal);
      _solver.getSlacks(slacks);
      _solver.getDualSol(dual);
      _solver.getRedCostSol(redCost);

      // unscale vectors
      if(_scaler && _solver.isScaled())
      {
         _scaler->unscalePrimal(_solver, primal);
         _scaler->unscaleSlacks(_solver, slacks);
         _scaler->unscaleDual(_solver, dual);
         _scaler->unscaleRedCost(_solver, redCost);
      }

      // get basis of transformed problem
      _solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(), _basisStatusRows.size(),
                       _basisStatusCols.size());

      try
      {
         _simplifier->unsimplify(primal, dual, slacks, redCost, _basisStatusRows.get_ptr(),
                                 _basisStatusCols.get_ptr(), false);
         _simplifier->getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(),
                               _basisStatusRows.size(), _basisStatusCols.size());
         _hasBasis = true;
      }
      catch(const SPxException& E)
      {
         MSG_INFO1(spxout, spxout << "Caught exception <" << E.what() <<
                   "> during unsimplification. Resolving without simplifier and scaler.\n");
         _hasBasis = false;
      }
   }
   // if the original problem is not in the solver because of scaling, we also need to store the basis
   else if(_scaler != nullptr)
   {
      _basisStatusRows.reSize(numRows());
      _basisStatusCols.reSize(numCols());
      assert(_basisStatusRows.size() == _solver.nRows());
      assert(_basisStatusCols.size() == _solver.nCols());

      _solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(), _basisStatusRows.size(),
                       _basisStatusCols.size());
      _hasBasis = true;
   }

   // resolve the original problem
   _preprocessAndSolveReal(false);
   return;
}



/// verify computed solution based on status and resolve if claimed primal or dual feasibility is not fulfilled
template <class R>
void SoPlexBase<R>::_verifySolutionReal()
{
   assert(_hasSolReal);

   MSG_INFO1(spxout, spxout << " --- verifying computed solution" << std::endl;)

   R sumviol = 0;
   R boundviol = 0;
   R rowviol = 0;
   R dualviol = 0;
   R redcostviol = 0;

   (void) getBoundViolation(boundviol, sumviol);
   (void) getRowViolation(rowviol, sumviol);
   (void) getDualViolation(dualviol, sumviol);
   (void) getRedCostViolation(redcostviol, sumviol);

   if(boundviol >= _solver.feastol() || rowviol >= _solver.feastol() || dualviol >= _solver.opttol()
         || redcostviol >= _solver.opttol())
   {
      assert(&_solver == _realLP);
      assert(_isRealLPLoaded);
      MSG_INFO3(spxout, spxout << "bound violation: " << boundviol
                << ", row violation: " << rowviol
                << ", dual violation: " << dualviol
                << ", redcost violation: " << redcostviol << std::endl;)
      MSG_INFO1(spxout, spxout <<
                " --- detected violations in original problem space -- solve again without presolving/scaling" <<
                std::endl;)

      if(_isRealLPScaled)
      {
         _solver.unscaleLPandReloadBasis();
         _isRealLPScaled = false;
         ++_unscaleCalls;
      }

      _preprocessAndSolveReal(false);
   }
}


/// stores solution data from the solver, possibly after applying unscaling and unsimplifying
template <class R>
void SoPlexBase<R>::_storeSolutionReal(bool verify)
{
   // prepare storage for basis (enough to fit the original basis)
   _basisStatusRows.reSize(numRows());
   _basisStatusCols.reSize(numCols());

   // prepare storage for the solution data (only in transformed space due to unscaling), w/o setting it to zero
   _solReal._primal.reDim(_solver.nCols(), false);
   _solReal._slacks.reDim(_solver.nRows(), false);
   _solReal._dual.reDim(_solver.nRows(), false);
   _solReal._redCost.reDim(_solver.nCols(), false);

   // check primal status consistency and query solution status
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL || status() != SPxSolverBase<R>::ERROR);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::NO_RATIOTESTER);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::NO_PRICER);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::NO_SOLVER);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::NOT_INIT);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::SINGULAR);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::NO_PROBLEM);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::UNBOUNDED);
   assert(_solver.basis().status() != SPxBasisBase<R>::PRIMAL
          || status() != SPxSolverBase<R>::INFEASIBLE);
   assert(_solver.basis().status() != SPxBasisBase<R>::UNBOUNDED
          || status() == SPxSolverBase<R>::UNBOUNDED);
   assert(_solver.basis().status() == SPxBasisBase<R>::UNBOUNDED
          || _solver.basis().status() == SPxBasisBase<R>::NO_PROBLEM
          || status() != SPxSolverBase<R>::UNBOUNDED);

   _solReal._isPrimalFeasible = (status() == SPxSolverBase<R>::OPTIMAL
                                 || ((_solver.basis().status() == SPxBasisBase<R>::PRIMAL
                                      || _solver.basis().status() == SPxBasisBase<R>::UNBOUNDED)
                                     && _solver.shift() < 10.0 * realParam(SoPlexBase<R>::EPSILON_ZERO)));

   _solReal._hasPrimalRay = (status() == SPxSolverBase<R>::UNBOUNDED && _isRealLPLoaded);

   // check dual status consistency and query solution status
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL || status() != SPxSolverBase<R>::ERROR);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::NO_RATIOTESTER);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::NO_PRICER);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::NO_SOLVER);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL || status() != SPxSolverBase<R>::NOT_INIT);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL || status() != SPxSolverBase<R>::SINGULAR);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::NO_PROBLEM);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::UNBOUNDED);
   assert(_solver.basis().status() != SPxBasisBase<R>::DUAL
          || status() != SPxSolverBase<R>::INFEASIBLE);
   assert(_solver.basis().status() != SPxBasisBase<R>::INFEASIBLE
          || status() == SPxSolverBase<R>::INFEASIBLE);
   assert(_solver.basis().status() == SPxBasisBase<R>::INFEASIBLE
          || _solver.basis().status() == SPxBasisBase<R>::NO_PROBLEM
          || status() != SPxSolverBase<R>::INFEASIBLE);

   _solReal._isDualFeasible = (status() == SPxSolverBase<R>::OPTIMAL
                               || ((_solver.basis().status() == SPxBasisBase<R>::DUAL
                                    || _solver.basis().status() == SPxBasisBase<R>::INFEASIBLE)
                                   && _solver.shift() < 10.0 * realParam(SoPlexBase<R>::EPSILON_ZERO)));

   _solReal._hasDualFarkas = (status() == SPxSolverBase<R>::INFEASIBLE && _isRealLPLoaded);

   // get infeasibility or unboundedness proof if available
   if(_solReal._hasPrimalRay)
   {
      _solReal._primalRay.reDim(_solver.nCols(), false);
      _solver.getPrimalray(_solReal._primalRay);
   }

   if(_solReal._hasDualFarkas)
   {
      _solReal._dualFarkas.reDim(_solver.nRows(), false);
      _solver.getDualfarkas(_solReal._dualFarkas);
   }

   // get solution data from the solver; independent of solution status
   _solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(),
                    _basisStatusRows.size(), _basisStatusCols.size());

   _solver.getPrimalSol(_solReal._primal);
   _solver.getSlacks(_solReal._slacks);
   _solver.getDualSol(_solReal._dual);
   _solver.getRedCostSol(_solReal._redCost);

   _hasBasis = true;

   // get primal and/or dual objective function value depending on status
   _solver.forceRecompNonbasicValue();
   _solReal._objVal = _solver.objValue();

   // infeasible solutions shall also be stored and be accessible
   _hasSolReal = true;

   // unscale vectors
   if(_solver.isScaled() && !_isRealLPLoaded)
      _unscaleSolutionReal(_solver, false);

   // get unsimplified solution data from simplifier
   if(_simplifier)
   {
      assert(!_simplifier->isUnsimplified());
      assert(_simplifier->result() == SPxSimplifier<R>::OKAY);
      assert(_realLP != &_solver);

      typename SPxBasisBase<R>::SPxStatus simplifiedBasisStatus = _solver.getBasisStatus();

      try
      {
         // pass solution data of transformed problem to simplifier
         _simplifier->unsimplify(_solReal._primal, _solReal._dual, _solReal._slacks, _solReal._redCost,
                                 _basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(), status() == SPxSolverBase<R>::OPTIMAL);
      }
      catch(const SPxException& E)
      {
         MSG_INFO1(spxout, spxout << "Caught exception <" << E.what() <<
                   "> during unsimplification. Resolving without simplifier and scaler.\n");
         _hasBasis = false;
         _preprocessAndSolveReal(false);
         return;
      }

      // copy unsimplified solution data from simplifier (size and dimension is adapted during copy)
      _solReal._primal  = _simplifier->unsimplifiedPrimal();
      _solReal._slacks  = _simplifier->unsimplifiedSlacks();
      _solReal._dual    = _simplifier->unsimplifiedDual();
      _solReal._redCost = _simplifier->unsimplifiedRedCost();

      // overwrite the transformed basis with the unsimplified one
      _simplifier->getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(),
                            _basisStatusRows.size(), _basisStatusCols.size());

      // load original problem but don't setup a slack basis
      _loadRealLP(false);

      // since we presolved the problem, we should not allow access to the dual norms
      _solver.weightsAreSetup = false;

      assert(_realLP == &_solver);

      // reset basis status
      _solver.setBasisStatus(simplifiedBasisStatus);

      _solver.setBasis(_basisStatusRows.get_const_ptr(), _basisStatusCols.get_const_ptr());
      // load unsimplified basis into solver
      assert(_basisStatusRows.size() == numRows());
      assert(_basisStatusCols.size() == this->numCols());
      _hasBasis = true;
   }

   // load realLP into the solver again (internal scaling was applied)
   else if(_realLP != &_solver)
   {
      assert(_solver.isScaled());
      _loadRealLP(false);
   }

   // unscale stored solution (removes persistent scaling)
   if(_isRealLPScaled)
      _unscaleSolutionReal(*_realLP, true);

   // check solution for violations and solve again if necessary
   if(verify)
      _verifySolutionReal();

   assert(_solver.nCols() == this->numCols());
   assert(_solver.nRows() == numRows());
}



template <class R>
void SoPlexBase<R>::_storeSolutionRealFromPresol()
{
   assert(_simplifier);
   assert(_simplifier->result() == SPxSimplifier<R>::VANISHED);

   // prepare storage for basis (enough to fit the original basis)
   _basisStatusRows.reSize(numRows());
   _basisStatusCols.reSize(numCols());

   // prepare storage for the solution data and initialize it to zero
   _solReal._primal.reDim(numCols(), true);
   _solReal._slacks.reDim(numRows(), true);
   _solReal._dual.reDim(numRows(), true);
   _solReal._redCost.reDim(numCols(), true);

   // load original LP and setup slack basis for unsimplifying
   _loadRealLP(true);

   // store slack basis
   _solver.getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(),
                    _basisStatusRows.size(), _basisStatusCols.size());

   assert(!_simplifier->isUnsimplified());

   try
   {
      // unsimplify basis and solution data
      _simplifier->unsimplify(_solReal._primal, _solReal._dual, _solReal._slacks, _solReal._redCost,
                              _basisStatusRows.get_ptr(), _basisStatusCols.get_ptr());

   }
   catch(const SPxException& E)
   {
      MSG_INFO1(spxout, spxout << "Caught exception <" << E.what() <<
                "> during unsimplification. Resolving without simplifier and scaler.\n");
      _preprocessAndSolveReal(false);
      return;
   }

   // copy unsimplified solution data from simplifier
   _solReal._primal  = _simplifier->unsimplifiedPrimal();
   _solReal._slacks  = _simplifier->unsimplifiedSlacks();
   _solReal._dual    = _simplifier->unsimplifiedDual();
   _solReal._redCost = _simplifier->unsimplifiedRedCost();

   // unscale stored solution (removes persistent scaling)
   if(_isRealLPScaled)
      _unscaleSolutionReal(*_realLP, true);

   // compute the original objective function value
   StableSum<R> objVal(realParam(SoPlexBase<R>::OBJ_OFFSET));

   for(int i = 0; i < numCols(); ++i)
      objVal += _solReal._primal[i] * objReal(i);

   _solReal._objVal = R(objVal);

   // store the unsimplified basis
   _simplifier->getBasis(_basisStatusRows.get_ptr(), _basisStatusCols.get_ptr(),
                         _basisStatusRows.size(), _basisStatusCols.size());
   _hasBasis = true;
   _hasSolReal = true;
   _solReal._isPrimalFeasible = true;
   _solReal._isDualFeasible = true;
   _solver.setBasisStatus(SPxBasisBase<R>::OPTIMAL);

   // check solution for violations and solve again if necessary
   _verifySolutionReal();
}



/// load original LP and possibly setup a slack basis
template <class R>
void SoPlexBase<R>::_loadRealLP(bool initBasis)
{
   _solver.loadLP(*_realLP, initBasis);
   _isRealLPLoaded = true;
   _realLP->~SPxLPBase<R>();
   spx_free(_realLP);
   _realLP = &_solver;

   if(initBasis)
      _solver.init();
}



/// unscales stored solution to remove internal or external scaling of LP
template <class R>
void SoPlexBase<R>::_unscaleSolutionReal(SPxLPBase<R>& LP, bool persistent)
{
   MSG_INFO1(spxout, spxout << " --- unscaling " << (persistent ? "external" : "internal") <<
             " solution" << std::endl;)
   assert(_scaler);
   assert(!persistent || (boolParam(SoPlexBase<R>::PERSISTENTSCALING) && _isRealLPScaled));
   _scaler->unscalePrimal(LP, _solReal._primal);
   _scaler->unscaleSlacks(LP, _solReal._slacks);
   _scaler->unscaleDual(LP, _solReal._dual);
   _scaler->unscaleRedCost(LP, _solReal._redCost);

   if(_solReal.hasPrimalRay())
      _scaler->unscalePrimalray(LP, _solReal._primalRay);

   if(_solReal.hasDualFarkas())
      _scaler->unscaleDualray(LP, _solReal._dualFarkas);
}
} // namespace soplex
