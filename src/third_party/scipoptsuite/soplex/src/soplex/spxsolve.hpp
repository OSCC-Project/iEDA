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

#include <assert.h>
#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/rational.h"
#include "soplex/spxsolver.h"
#include "soplex/spxpricer.h"
#include "soplex/spxratiotester.h"
#include "soplex/spxdefaultrt.h"
#include "soplex/spxstarter.h"
#include "soplex/spxout.h"

#define MAXCYCLES 400
#define MAXSTALLS 10000
#define MAXSTALLRECOVERS 10
#define MAXREFACPIVOTS 10

namespace soplex
{

/**@todo check separately for ENTER and LEAVE algorithm */
template <class R>
bool SPxSolverBase<R>::precisionReached(R& newpricertol) const
{
   R maxViolRedCost;
   R sumViolRedCost;
   R maxViolBounds;
   R sumViolBounds;
   R maxViolConst;
   R sumViolConst;

   qualRedCostViolation(maxViolRedCost, sumViolRedCost);
   qualBoundViolation(maxViolBounds, sumViolBounds);
   qualConstraintViolation(maxViolConst, sumViolConst);

   // is the solution good enough ?
   bool reached = maxViolRedCost < opttol() && maxViolBounds < feastol() && maxViolConst < feastol();

   if(!reached)
   {
      newpricertol = thepricer->epsilon() / 10.0;

      MSG_INFO3((*this->spxout), (*this->spxout) << "Precision not reached: Pricer tolerance = "
                << thepricer->epsilon()
                << " new tolerance = " << newpricertol
                << std::endl
                << " maxViolRedCost= " << maxViolRedCost
                << " maxViolBounds= " << maxViolBounds
                << " maxViolConst= " << maxViolConst
                << std::endl
                << " sumViolRedCost= " << sumViolRedCost
                << " sumViolBounds= " << sumViolBounds
                << " sumViolConst= " << sumViolConst
                << std::endl;);
   }

   return reached;
}

template <class R>
void SPxSolverBase<R>::calculateProblemRanges()
{
   // only collect absolute values
   R minobj = R(infinity);
   R maxobj = 0.0;
   R minbound = R(infinity);
   R maxbound = 0.0;
   R minside = R(infinity);
   R maxside = 0.0;

   // get min and max absolute values of bounds and objective
   for(int j = 0; j < this->nCols(); ++j)
   {
      R abslow = spxAbs(this->lower(j));
      R absupp = spxAbs(this->lower(j));
      R absobj = spxAbs(this->obj(j));

      if(abslow < R(infinity))
      {
         minbound = MINIMUM(minbound, abslow);
         maxbound = MAXIMUM(maxbound, abslow);
      }

      if(absupp < R(infinity))
      {
         minbound = MINIMUM(minbound, absupp);
         maxbound = MAXIMUM(maxbound, absupp);
      }

      minobj = MINIMUM(minobj, absobj);
      maxobj = MAXIMUM(maxobj, absobj);
   }

   // get min and max absoute values of sides
   for(int i = 0; i < this->nRows(); ++i)
   {
      R abslhs = spxAbs(this->lhs(i));
      R absrhs = spxAbs(this->rhs(i));

      if(abslhs > R(infinity))
      {
         minside = MINIMUM(minside, abslhs);
         maxside = MAXIMUM(maxside, abslhs);
      }

      if(absrhs < R(infinity))
      {
         minside = MINIMUM(minside, absrhs);
         maxside = MAXIMUM(maxside, absrhs);
      }
   }

   boundrange = maxbound - minbound;
   siderange = maxside - minside;
   objrange = maxobj - minobj;
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::solve(volatile bool* interrupt)
{

   SPxId enterId;
   int   leaveNum;
   int   loopCount = 0;
   R  minShift = R(infinity);
   int   cycleCount = 0;
   bool  priced = false;
   R  lastDelta = 1;

   /* allow clean up step only once */
   recomputedVectors = false;

   /* store the last (primal or dual) feasible objective value to recover/abort in case of stalling */
   R  stallRefValue;
   R  stallRefShift;
   int   stallRefIter;
   int   stallNumRecovers;

   if(dim() <= 0 && coDim() <= 0)  // no problem loaded
   {
      m_status = NO_PROBLEM;
      throw SPxStatusException("XSOLVE01 No Problem loaded");
   }

   if(slinSolver() == 0)  // linear system solver is required.
   {
      m_status = NO_SOLVER;
      throw SPxStatusException("XSOLVE02 No Solver loaded");
   }

   if(thepricer == 0)  // pricer is required.
   {
      m_status = NO_PRICER;
      throw SPxStatusException("XSOLVE03 No Pricer loaded");
   }

   if(theratiotester == 0)  // ratiotester is required.
   {
      m_status = NO_RATIOTESTER;
      throw SPxStatusException("XSOLVE04 No RatioTester loaded");
   }

   theTime->reset();
   theTime->start();

   m_numCycle = 0;
   this->iterCount  = 0;
   this->lastIterCount = 0;
   this->iterDegenCheck = 0;

   if(!isInitialized())
   {
      /*
        if(SPxBasisBase<R>::status() <= NO_PROBLEM)
        SPxBasisBase<R>::load(this);
      */
      /**@todo != REGULAR is not enough. Also OPTIMAL/DUAL/PRIMAL should
       * be tested and acted accordingly.
       */
      if(thestarter != 0 && status() != REGULAR
            && this->theLP->status() == NO_PROBLEM)   // no basis and no starter.
         thestarter->generate(*this);              // generate start basis.

      init();

      // Inna/Tobi: init might fail, if the basis is singular
      if(!isInitialized())
      {
         assert(SPxBasisBase<R>::status() == SPxBasisBase<R>::SINGULAR);
         m_status = UNKNOWN;
         return status();
      }
   }

   //setType(type());

   if(!this->matrixIsSetup)
      SPxBasisBase<R>::load(this);

   //factorized = false;

   assert(thepricer->solver()      == this);
   assert(theratiotester->solver() == this);

   // maybe this should be done in init() ?
   thepricer->setType(type());
   theratiotester->setType(type());

   MSG_INFO3((*this->spxout),
             (*this->spxout) << "starting value = " << value() << std::endl
             << "starting shift = " << shift() << std::endl;
            )

   if(SPxBasisBase<R>::status() == SPxBasisBase<R>::OPTIMAL)
      setBasisStatus(SPxBasisBase<R>::REGULAR);

   m_status   = RUNNING;
   bool stop  = terminate();
   leaveCount = 0;
   enterCount = 0;
   primalCount = 0;
   polishCount = 0;
   boundflips = 0;
   totalboundflips = 0;
   enterCycles = 0;
   leaveCycles = 0;
   primalDegenSum = 0;
   dualDegenSum = 0;

   multSparseCalls = 0;
   multFullCalls = 0;
   multColwiseCalls = 0;
   multUnsetupCalls = 0;

   stallNumRecovers = 0;

   /* if we run into a singular basis, we will retry from regulardesc with tighter tolerance in the ratio test */
   typename SPxSolverBase<R>::Type tightenedtype = type();
   bool tightened = false;

   while(!stop)
   {
      const typename SPxBasisBase<R>::Desc regulardesc = this->desc();

      // we need to reset these pointers to avoid unnecessary/wrong solves in leave() or enter()
      solveVector2 = 0;
      solveVector3 = 0;
      coSolveVector2 = 0;
      coSolveVector3 = 0;

      updateViols.clear();
      updateViolsCo.clear();

      try
      {

         if(type() == ENTER)
         {
            forceRecompNonbasicValue();

            int enterCycleCount = 0;
            int enterFacPivotCount = 0;

            instableEnterVal = 0;
            instableEnterId = SPxId();
            instableEnter = false;

            stallRefIter = this->iteration() - 1;
            stallRefShift = shift();
            stallRefValue = value();

            /* in the entering algorithm, entertol() should be maintained by the ratio test and leavetol() should be
             * reached by the pricer
             */
            R maxpricertol = leavetol();
            R minpricertol = 0.01 * maxpricertol;

            thepricer->setEpsilon(maxpricertol);
            priced = false;

            // to avoid shifts we restrict tolerances in the ratio test
            if(loopCount > 0)
            {
               lastDelta = (lastDelta < entertol()) ? lastDelta : entertol();
               lastDelta *= 0.01;
               theratiotester->setDelta(lastDelta);
               assert(theratiotester->getDelta() > 0);
               MSG_DEBUG(std::cout << "decreased delta for ratiotest to: " << theratiotester->getDelta() <<
                         std::endl;)
            }
            else
            {
               lastDelta = 1;
               theratiotester->setDelta(entertol());
            }

            printDisplayLine(true);

            do
            {
               printDisplayLine();

               enterId = thepricer->selectEnter();

               if(!enterId.isValid() && instableEnterId.isValid() && this->lastUpdate() == 0)
               {
                  /* no entering variable was found, but because of valid instableEnterId we know
                     that this is due to the scaling of the test values. Thus, we use
                     instableEnterId and SPxFastRT<R>::selectEnter shall accept even an instable
                     leaving variable. */
                  MSG_INFO3((*this->spxout), (*this->spxout) << " --- trying instable enter iteration" << std::endl;)

                  enterId = instableEnterId;
                  instableEnter = true;
                  // we also need to reset the test() or coTest() value for getEnterVals()
                  assert(instableEnterVal < 0);

                  if(enterId.isSPxColId())
                  {
                     int idx = this->number(SPxColId(enterId));

                     if(rep() == COLUMN)
                     {
                        theTest[idx] = instableEnterVal;

                        if(sparsePricingEnterCo && isInfeasibleCo[idx] == SPxPricer<R>::NOT_VIOLATED)
                        {
                           infeasibilitiesCo.addIdx(idx);
                           isInfeasibleCo[idx] = SPxPricer<R>::VIOLATED;
                        }

                        if(hyperPricingEnter)
                           updateViolsCo.addIdx(idx);
                     }
                     else
                     {
                        theCoTest[idx] = instableEnterVal;

                        if(sparsePricingEnter && isInfeasible[idx] == SPxPricer<R>::NOT_VIOLATED)
                        {
                           infeasibilities.addIdx(idx);
                           isInfeasible[idx] = SPxPricer<R>::VIOLATED;
                        }

                        if(hyperPricingEnter)
                           updateViols.addIdx(idx);
                     }
                  }
                  else
                  {
                     int idx = this->number(SPxRowId(enterId));

                     if(rep() == COLUMN)
                     {
                        theCoTest[idx] = instableEnterVal;

                        if(sparsePricingEnter && isInfeasible[idx] == SPxPricer<R>::NOT_VIOLATED)
                        {
                           infeasibilities.addIdx(idx);
                           isInfeasible[idx] = SPxPricer<R>::VIOLATED;
                        }

                        if(hyperPricingEnter)
                           updateViols.addIdx(idx);
                     }
                     else
                     {
                        theTest[idx] = instableEnterVal;

                        if(sparsePricingEnterCo && isInfeasibleCo[idx] == SPxPricer<R>::NOT_VIOLATED)
                        {
                           infeasibilitiesCo.addIdx(idx);
                           isInfeasibleCo[idx] = SPxPricer<R>::VIOLATED;
                        }

                        if(hyperPricingEnter)
                           updateViolsCo.addIdx(idx);
                     }
                  }
               }
               else
               {
                  instableEnter = false;
               }

               if(!enterId.isValid())
               {
                  // we are not infeasible and have no shift
                  if(shift() <= epsilon()
                        && (SPxBasisBase<R>::status() == SPxBasisBase<R>::REGULAR
                            || SPxBasisBase<R>::status() == SPxBasisBase<R>::DUAL
                            || SPxBasisBase<R>::status() == SPxBasisBase<R>::PRIMAL))
                  {
                     R newpricertol = minpricertol;

                     // refactorize to eliminate accumulated errors from LU updates
                     if(this->lastUpdate() > 0)
                        factorize();

                     // recompute Fvec, Pvec and CoPvec to get a more precise solution and obj value
                     computeFrhs();
                     SPxBasisBase<R>::solve(*theFvec, *theFrhs);

                     computeEnterCoPrhs();
                     SPxBasisBase<R>::coSolve(*theCoPvec, *theCoPrhs);
                     computePvec();

                     forceRecompNonbasicValue();

                     MSG_INFO2((*this->spxout), (*this->spxout) << " --- checking feasibility and optimality\n")
                     computeCoTest();
                     computeTest();

                     // is the solution good enough ?
                     // max three times reduced
                     if((thepricer->epsilon() > minpricertol) && !precisionReached(newpricertol))
                     {
                        // no!
                        // we reduce the pricer tolerance. Note that if the pricer does not find a candiate
                        // with the reduced tolerance, we quit, regardless of the violations.
                        if(newpricertol < minpricertol)
                           newpricertol = minpricertol;

                        thepricer->setEpsilon(newpricertol);

                        MSG_INFO2((*this->spxout), (*this->spxout) << " --- setting pricer tolerance to "
                                  << thepricer->epsilon()
                                  << std::endl;)
                     }
                  }

                  // if the factorization is not fresh, we better refactorize and call the pricer again; however, this can
                  // create cycling, so it is performed only a limited number of times per ENTER round
                  if(this->lastUpdate() > 0 && enterFacPivotCount < MAXREFACPIVOTS)
                  {
                     MSG_INFO3((*this->spxout), (*this->spxout) << " --- solve(enter) triggers refactorization" <<
                               std::endl;)

                     factorize();

                     // if the factorization was found out to be singular, we have to quit
                     if(SPxBasisBase<R>::status() < SPxBasisBase<R>::REGULAR)
                     {
                        MSG_INFO1((*this->spxout), (*this->spxout) << "Something wrong with factorization, Basis status: "
                                  << static_cast<int>(SPxBasisBase<R>::status()) << std::endl;)
                        stop = true;
                        break;
                     }

                     // call pricer again
                     enterId = thepricer->selectEnter();

                     // count how often the pricer has found something only after refactorizing
                     if(enterId.isValid())
                        enterFacPivotCount++;
                  }

                  if(!enterId.isValid())
                  {
                     priced = true;
                     break;
                  }
               }

               /* check if we have iterations left */
               if(maxIters >= 0 && iterations() >= maxIters)
               {
                  MSG_INFO2((*this->spxout), (*this->spxout) << " --- maximum number of iterations (" << maxIters
                            << ") reached" << std::endl;)
                  m_status = ABORT_ITER;
                  stop = true;
                  break;
               }

               if(interrupt != NULL && *interrupt)
               {
                  MSG_INFO2((*this->spxout), (*this->spxout) << " --- aborted due to interrupt signal" << std::endl;)
                  m_status = ABORT_TIME;
                  stop = true;
                  break;
               }

               enter(enterId);
               assert((testBounds(), 1));
               thepricer->entered4(this->lastEntered(), this->lastIndex());
               stop = terminate();
               clearUpdateVecs();

               /* if a successful pivot was performed or a nonbasic variable was flipped to its other bound, we reset the
                * cycle counter
                */
               if(this->lastEntered().isValid())
                  enterCycleCount = 0;
               else if(basis().status() != SPxBasisBase<R>::INFEASIBLE
                       && basis().status() != SPxBasisBase<R>::UNBOUNDED)
               {
                  enterCycleCount++;

                  if(enterCycleCount > MAXCYCLES)
                  {
                     MSG_INFO2((*this->spxout), (*this->spxout) << " --- abort solving due to cycling in "
                               << "entering algorithm" << std::endl;);
                     m_status = ABORT_CYCLING;
                     stop = true;
                  }
               }

               /* only if the basis has really changed, we increase the iterations counter; this is not the case when only
                * a nonbasic variable was flipped to its other bound
                */
               if(this->lastIndex() >= 0)
               {
                  enterCount++;
                  assert(this->lastEntered().isValid());
               }

               /* check every MAXSTALLS iterations whether shift and objective value have not changed */
               if((this->iteration() - stallRefIter) % MAXSTALLS == 0
                     && basis().status() != SPxBasisBase<R>::INFEASIBLE)
               {
                  if(spxAbs(value() - stallRefValue) <= epsilon() && spxAbs(shift() - stallRefShift) <= epsilon())
                  {
                     if(stallNumRecovers < MAXSTALLRECOVERS)
                     {
                        /* try to recover by unshifting/switching algorithm up to MAXSTALLRECOVERS times (just a number picked) */
                        MSG_INFO3((*this->spxout), (*this->spxout) <<
                                  " --- stalling detected - trying to recover by switching to LEAVING algorithm." << std::endl;)

                        ++stallNumRecovers;
                        break;
                     }
                     else
                     {
                        /* giving up */
                        MSG_INFO2((*this->spxout), (*this->spxout) <<
                                  " --- abort solving due to stalling in entering algorithm." << std::endl;);

                        m_status = ABORT_CYCLING;
                        stop = true;
                     }
                  }
                  else
                  {
                     /* merely update reference values */
                     stallRefIter = this->iteration() - 1;
                     stallRefShift = shift();
                     stallRefValue = value();
                  }
               }

               //@ assert(isConsistent());
            }
            while(!stop);

            MSG_INFO3((*this->spxout),
                      (*this->spxout) << " --- enter finished. iteration: " << this->iteration()
                      << ", value: " << value()
                      << ", shift: " << shift()
                      << ", epsilon: " << epsilon()
                      << ", feastol: " << feastol()
                      << ", opttol: " << opttol()
                      << std::endl
                      << "ISOLVE56 stop: " << stop
                      << ", basis status: " << static_cast<int>(SPxBasisBase<R>::status()) << " (" << static_cast<int>
                      (SPxBasisBase<R>::status()) << ")"
                      << ", solver status: " << static_cast<int>(m_status) << " (" << static_cast<int>
                      (m_status) << ")" << std::endl;
                     )

            if(!stop)
            {
               /**@todo technically it would be ok to finish already when (priced && maxinfeas + shift() <= entertol()) is
                *  satisfied; maybe at least in the case when SoPlex keeps jumping back between ENTER and LEAVE always
                *  shifting (looping), we may relax this condition here;
                *  note also that unShift may increase shift() slightly due to roundoff errors
                */
               if(shift() <= epsilon())
               {
                  // factorize();
                  unShift();

                  R maxinfeas = maxInfeas();

                  MSG_INFO3((*this->spxout),
                            (*this->spxout) << " --- maxInfeas: " << maxinfeas
                            << ", shift: " << shift()
                            << ", entertol: " << entertol() << std::endl;
                           )

                  if(priced && maxinfeas + shift() <= entertol())
                  {
                     setBasisStatus(SPxBasisBase<R>::OPTIMAL);
                     m_status = OPTIMAL;
                     break;
                  }
                  else if(loopCount > 2)
                  {
                     // calculate problem ranges if not done already
                     if(boundrange == 0.0 || siderange == 0.0 || objrange == 0.0)
                        calculateProblemRanges();

                     if(MAXIMUM(MAXIMUM(boundrange, siderange), objrange) >= 1e9)
                     {
                        SPxOut::setScientific(spxout->getCurrentStream(), 0);
                        MSG_INFO1((*this->spxout), (*this->spxout) <<
                                  " --- termination despite violations (numerical difficulties,"
                                  << " bound range = " << boundrange
                                  << ", side range = " << siderange
                                  << ", obj range = " << objrange
                                  << ")" << std::endl;)
                        setBasisStatus(SPxBasisBase<R>::OPTIMAL);
                        m_status = OPTIMAL;
                        break;
                     }
                     else
                     {
                        m_status = ABORT_CYCLING;
                        throw SPxStatusException("XSOLVE14 Abort solving due to looping");
                     }
                  }

                  loopCount++;
               }

               setType(LEAVE);
               init();
               thepricer->setType(type());
               theratiotester->setType(type());
            }
         }
         else
         {
            assert(type() == LEAVE);

            forceRecompNonbasicValue();

            int leaveCycleCount = 0;
            int leaveFacPivotCount = 0;

            instableLeaveNum = -1;
            instableLeave = false;
            instableLeaveVal = 0;

            stallRefIter = this->iteration() - 1;
            stallRefShift = shift();
            stallRefValue = value();

            /* in the leaving algorithm, leavetol() should be maintained by the ratio test and entertol() should be reached
             * by the pricer
             */
            R maxpricertol = entertol();
            R minpricertol = 0.01 * maxpricertol;

            thepricer->setEpsilon(maxpricertol);
            priced = false;

            // to avoid shifts we restrict tolerances in the ratio test
            if(loopCount > 0)
            {
               lastDelta = (lastDelta < leavetol()) ? lastDelta : leavetol();
               lastDelta *= 0.01;
               theratiotester->setDelta(lastDelta);
               assert(theratiotester->getDelta() > 0);
               MSG_DEBUG(std::cout << "decreased delta for ratiotest to: " << theratiotester->getDelta() <<
                         std::endl;)
            }
            else
            {
               lastDelta = 1;
               theratiotester->setDelta(leavetol());
            }

            printDisplayLine(true);

            do
            {
               printDisplayLine();

               leaveNum = thepricer->selectLeave();

               if(leaveNum < 0 && instableLeaveNum >= 0 && this->lastUpdate() == 0)
               {
                  /* no leaving variable was found, but because of instableLeaveNum >= 0 we know
                     that this is due to the scaling of theCoTest[...]. Thus, we use
                     instableLeaveNum and SPxFastRT<R>::selectEnter shall accept even an instable
                     entering variable. */
                  MSG_INFO3((*this->spxout),
                            (*this->spxout) << " --- trying instable leave iteration" << std::endl;
                           )

                  leaveNum = instableLeaveNum;
                  instableLeave = true;
                  // we also need to reset the fTest() value for getLeaveVals()
                  assert(instableLeaveVal < 0);
                  theCoTest[instableLeaveNum] = instableLeaveVal;

                  if(sparsePricingLeave)
                  {
                     if(isInfeasible[instableLeaveNum] == SPxPricer<R>::NOT_VIOLATED)
                     {
                        infeasibilities.addIdx(instableLeaveNum);
                        isInfeasible[instableLeaveNum] = SPxPricer<R>::VIOLATED;
                     }

                     if(hyperPricingLeave)
                        updateViols.addIdx(instableLeaveNum);
                  }
               }
               else
               {
                  instableLeave = false;
               }

               if(leaveNum < 0)
               {
                  // we are not infeasible and have no shift
                  if(shift() <= epsilon()
                        && (SPxBasisBase<R>::status() == SPxBasisBase<R>::REGULAR
                            || SPxBasisBase<R>::status() == SPxBasisBase<R>::DUAL
                            || SPxBasisBase<R>::status() == SPxBasisBase<R>::PRIMAL))
                  {
                     R newpricertol = minpricertol;

                     // refactorize to eliminate accumulated errors from LU updates
                     if(this->lastUpdate() > 0)
                        factorize();

                     // recompute Fvec, Pvec and CoPvec to get a more precise solution and obj value
                     computeFrhs();
                     SPxBasisBase<R>::solve(*theFvec, *theFrhs);

                     computeLeaveCoPrhs();
                     SPxBasisBase<R>::coSolve(*theCoPvec, *theCoPrhs);
                     computePvec();

                     forceRecompNonbasicValue();

                     MSG_INFO2((*this->spxout), (*this->spxout) << " --- checking feasibility and optimality\n")
                     computeFtest();

                     // is the solution good enough ?
                     // max three times reduced
                     if((thepricer->epsilon() > minpricertol) && !precisionReached(newpricertol))
                     {
                        // no
                        // we reduce the pricer tolerance. Note that if the pricer does not find a candiate
                        // with the reduced pricer tolerance, we quit, regardless of the violations.
                        if(newpricertol < minpricertol)
                           newpricertol = minpricertol;

                        thepricer->setEpsilon(newpricertol);

                        MSG_INFO2((*this->spxout), (*this->spxout) << " --- setting pricer tolerance to "
                                  << thepricer->epsilon()
                                  << std::endl;);
                     }
                  }

                  // if the factorization is not fresh, we better refactorize and call the pricer again; however, this can
                  // create cycling, so it is performed only a limited number of times per LEAVE round
                  if(this->lastUpdate() > 0 && leaveFacPivotCount < MAXREFACPIVOTS)
                  {
                     MSG_INFO3((*this->spxout), (*this->spxout) << " --- solve(leave) triggers refactorization" <<
                               std::endl;)

                     factorize();

                     // Inna/Tobi: if the factorization was found out to be singular, we have to quit
                     if(SPxBasisBase<R>::status() < SPxBasisBase<R>::REGULAR)
                     {
                        MSG_INFO1((*this->spxout), (*this->spxout) << "Something wrong with factorization, Basis status: "
                                  << static_cast<int>(SPxBasisBase<R>::status()) << std::endl;)
                        stop = true;
                        break;
                     }

                     // call pricer again
                     leaveNum = thepricer->selectLeave();

                     // count how often the pricer has found something only after refactorizing
                     if(leaveNum >= 0)
                        leaveFacPivotCount++;
                  }

                  if(leaveNum < 0)
                  {
                     priced = true;
                     break;
                  }
               }

               /* check if we have iterations left */
               if(maxIters >= 0 && iterations() >= maxIters)
               {
                  MSG_INFO2((*this->spxout), (*this->spxout) << " --- maximum number of iterations (" << maxIters
                            << ") reached" << std::endl;)
                  m_status = ABORT_ITER;
                  stop = true;
                  break;
               }

               if(interrupt != NULL && *interrupt)
               {
                  MSG_INFO2((*this->spxout), (*this->spxout) << " --- aborted due to interrupt signal" << std::endl;)
                  m_status = ABORT_TIME;
                  stop = true;
                  break;
               }

               leave(leaveNum);
               assert((testBounds(), 1));
               thepricer->left4(this->lastIndex(), this->lastLeft());
               stop = terminate();
               clearUpdateVecs();

               /* if a successful pivot was performed or a nonbasic variable was flipped to its other bound, we reset the
                * cycle counter
                */
               if(this->lastIndex() >= 0)
                  leaveCycleCount = 0;
               else if(basis().status() != SPxBasisBase<R>::INFEASIBLE
                       && basis().status() != SPxBasisBase<R>::UNBOUNDED)
               {
                  leaveCycleCount++;

                  if(leaveCycleCount > MAXCYCLES)
                  {
                     MSG_INFO2((*this->spxout), (*this->spxout) <<
                               " --- abort solving due to cycling in leaving algorithm" << std::endl;);
                     m_status = ABORT_CYCLING;
                     stop = true;
                  }
               }

               /* only if the basis has really changed, we increase the iterations counter; this is not the case when only
                * a nonbasic variable was flipped to its other bound
                */
               if(this->lastEntered().isValid())
               {
                  leaveCount++;
                  assert(this->lastIndex() >= 0);
               }

               /* check every MAXSTALLS iterations whether shift and objective value have not changed */
               if((this->iteration() - stallRefIter) % MAXSTALLS == 0
                     && basis().status() != SPxBasisBase<R>::INFEASIBLE)
               {
                  if(spxAbs(value() - stallRefValue) <= epsilon() && spxAbs(shift() - stallRefShift) <= epsilon())
                  {
                     if(stallNumRecovers < MAXSTALLRECOVERS)
                     {
                        /* try to recover by switching algorithm up to MAXSTALLRECOVERS times */
                        MSG_INFO3((*this->spxout), (*this->spxout) <<
                                  " --- stalling detected - trying to recover by switching to ENTERING algorithm." << std::endl;)

                        ++stallNumRecovers;
                        break;
                     }
                     else
                     {
                        /* giving up */
                        MSG_INFO2((*this->spxout), (*this->spxout) <<
                                  " --- abort solving due to stalling in leaving algorithm" << std::endl;);

                        m_status = ABORT_CYCLING;
                        stop = true;
                     }
                  }
                  else
                  {
                     /* merely update reference values */
                     stallRefIter = this->iteration() - 1;
                     stallRefShift = shift();
                     stallRefValue = value();
                  }
               }

               //@ assert(isConsistent());
            }
            while(!stop);

            MSG_INFO3((*this->spxout),
                      (*this->spxout) << " --- leave finished. iteration: " << this->iteration()
                      << ", value: " << value()
                      << ", shift: " << shift()
                      << ", epsilon: " << epsilon()
                      << ", feastol: " << feastol()
                      << ", opttol: " << opttol()
                      << std::endl
                      << "ISOLVE57 stop: " << stop
                      << ", basis status: " << static_cast<int>(SPxBasisBase<R>::status()) << " (" << static_cast<int>
                      (SPxBasisBase<R>::status()) << ")"
                      << ", solver status: " << static_cast<int>(m_status) << " (" << static_cast<int>
                      (m_status) << ")" << std::endl;
                     )

            if(!stop)
            {
               if(shift() < minShift)
               {
                  minShift = shift();
                  cycleCount = 0;
               }
               else
               {
                  cycleCount++;

                  if(cycleCount > MAXCYCLES)
                  {
                     m_status = ABORT_CYCLING;
                     throw SPxStatusException("XSOLVE13 Abort solving due to cycling");
                  }

                  MSG_INFO3((*this->spxout),
                            (*this->spxout) << " --- maxInfeas: " << maxInfeas()
                            << ", shift: " << shift()
                            << ", leavetol: " << leavetol()
                            << ", cycle count: " << cycleCount << std::endl;
                           )
               }

               /**@todo technically it would be ok to finish already when (priced && maxinfeas + shift() <= entertol()) is
                *  satisfied; maybe at least in the case when SoPlex keeps jumping back between ENTER and LEAVE always
                *  shifting (looping), we may relax this condition here;
                *  note also that unShift may increase shift() slightly due to roundoff errors
                */
               if(shift() <= epsilon())
               {
                  cycleCount = 0;
                  // factorize();
                  unShift();

                  R maxinfeas = maxInfeas();

                  MSG_INFO3((*this->spxout),
                            (*this->spxout) << " --- maxInfeas: " << maxinfeas
                            << ", shift: " << shift()
                            << ", leavetol: " << leavetol() << std::endl;
                           )

                  // We stop if we are indeed optimal, or if we have already been
                  // two times at this place. In this case it seems futile to
                  // continue.
                  if(priced && maxinfeas + shift() <= leavetol())
                  {
                     setBasisStatus(SPxBasisBase<R>::OPTIMAL);
                     m_status = OPTIMAL;
                     break;
                  }
                  else if(loopCount > 2)
                  {
                     // calculate problem ranges if not done already
                     if(boundrange == 0.0 || siderange == 0.0 || objrange == 0.0)
                        calculateProblemRanges();

                     if(MAXIMUM(MAXIMUM(boundrange, siderange), objrange) >= 1e9)
                     {
                        SPxOut::setScientific(spxout->getCurrentStream(), 0);
                        MSG_INFO1((*this->spxout), (*this->spxout) <<
                                  " --- termination despite violations (numerical difficulties,"
                                  << " bound range = " << boundrange
                                  << ", side range = " << siderange
                                  << ", obj range = " << objrange
                                  << ")" << std::endl;)
                        setBasisStatus(SPxBasisBase<R>::OPTIMAL);
                        m_status = OPTIMAL;
                        break;
                     }
                     else
                     {
                        m_status = ABORT_CYCLING;
                        throw SPxStatusException("XSOLVE14 Abort solving due to looping");
                     }
                  }

                  loopCount++;
               }

               setType(ENTER);
               init();
               thepricer->setType(type());
               theratiotester->setType(type());
            }
         }

         assert(m_status != SINGULAR);

      }
      catch(const SPxException& E)
      {
         // if we stopped due to a singular basis, we reload the original basis and try again with tighter
         // tolerance (only once)
         if(m_status == SINGULAR && !tightened)
         {
            tightenedtype = type();

            if(tightenedtype == ENTER)
            {
               m_entertol = 0.01 * m_entertol;

               MSG_INFO2((*this->spxout), (*this->spxout) <<
                         " --- basis singular: reloading basis and solving with tighter ratio test tolerance " << m_entertol
                         << std::endl;)
            }
            else
            {
               m_leavetol = 0.01 * m_leavetol;

               MSG_INFO2((*this->spxout), (*this->spxout) <<
                         " --- basis singular: reloading basis and solving with tighter ratio test tolerance " << m_leavetol
                         << std::endl;)
            }

            // load original basis
            int niters = iterations();
            loadBasis(regulardesc);

            // remember iteration count
            this->iterCount = niters;

            // try initializing basis (might fail if starting basis was already singular)
            try
            {
               init();
               theratiotester->setType(type());
            }
            catch(const SPxException& Ex)
            {
               MSG_INFO2((*this->spxout), (*this->spxout) <<
                         " --- reloaded basis singular, resetting original tolerances" << std::endl;)

               if(tightenedtype == ENTER)
                  m_entertol = 100.0 * m_entertol;
               else
                  m_leavetol = 100.0 * m_leavetol;

               theratiotester->setType(type());

               throw Ex;
            }

            // reset status and counters
            m_status = RUNNING;
            m_numCycle = 0;
            leaveCount = 0;
            enterCount = 0;
            stallNumRecovers = 0;

            // continue
            stop = false;
            tightened = true;
         }
         // reset tolerance to its original value and pass on the exception
         else if(tightened)
         {
            if(tightenedtype == ENTER)
               m_entertol = 100.0 * m_entertol;
            else
               m_leavetol = 100.0 * m_leavetol;

            theratiotester->setType(type());

            throw E;
         }
         // pass on the exception
         else
            throw E;
      }
   }

   // reset tolerance to its original value
   if(tightened)
   {
      if(tightenedtype == ENTER)
         m_entertol = 100.0 * m_entertol;
      else
         m_leavetol = 100.0 * m_leavetol;

      theratiotester->setType(type());
   }

   theTime->stop();
   theCumulativeTime += time();

   if(m_status == RUNNING)
   {
      m_status = ERROR;
      throw SPxStatusException("XSOLVE05 Status is still RUNNING when it shouldn't be");
   }

   MSG_INFO3((*this->spxout),
             (*this->spxout) << "Finished solving (status=" << static_cast<int>(status())
             << ", iters=" << this->iterCount
             << ", leave=" << leaveCount
             << ", enter=" << enterCount
             << ", flips=" << totalboundflips;

             if(status() == OPTIMAL)
             (*this->spxout) << ", objValue=" << value();
             (*this->spxout) << ")" << std::endl;
            )

#ifdef ENABLE_ADDITIONAL_CHECKS

      /* check if solution is really feasible */
      if(status() == OPTIMAL)
      {
         int     c;
         R    val;
         VectorBase<R> sol(this->nCols());

         getPrimalSol(sol);

         for(int row = 0; row < this->nRows(); ++row)
         {
            const SVectorBase<R>& rowvec = this->rowVector(row);
            val = 0.0;

            for(c = 0; c < rowvec.size(); ++c)
               val += rowvec.value(c) * sol[rowvec.index(c)];

            if(LT(val, this->lhs(row), feastol()) ||
                  GT(val, this->rhs(row), feastol()))
            {
               // Minor rhs violations happen frequently, so print these
               // warnings only with verbose level INFO2 and higher.
               MSG_INFO2((*this->spxout), (*this->spxout) << "WSOLVE88 Warning! Constraint " << row
                         << " is violated by solution" << std::endl
                         << "   lhs:" << this->lhs(row)
                         << " <= val:" << val
                         << " <= rhs:" << this->rhs(row) << std::endl;)

               if(type() == LEAVE && isRowBasic(row))
               {
                  // find basis variable
                  for(c = 0; c < this->nRows(); ++c)
                     if(basis().baseId(c).isSPxRowId()
                           && (this->number(basis().baseId(c)) == row))
                        break;

                  assert(c < this->nRows());

                  MSG_WARNING((*this->spxout), (*this->spxout) << "WSOLVE90 basis idx:" << c
                              << " fVec:" << fVec()[c]
                              << " fRhs:" << fRhs()[c]
                              << " fTest:" << fTest()[c] << std::endl;)
               }
            }
         }

         for(int col = 0; col < this->nCols(); ++col)
         {
            if(LT(sol[col], this->lower(col), feastol()) ||
                  GT(sol[col], this->upper(col), feastol()))
            {
               // Minor bound violations happen frequently, so print these
               // warnings only with verbose level INFO2 and higher.
               MSG_INFO2((*this->spxout), (*this->spxout) << "WSOLVE91 Warning! Bound for column " << col
                         << " is violated by solution" << std::endl
                         << "   lower:" << this->lower(col)
                         << " <= val:" << sol[col]
                         << " <= upper:" << this->upper(col) << std::endl;)

               if(type() == LEAVE && isColBasic(col))
               {
                  for(c = 0; c < this->nRows() ; ++c)
                     if(basis().baseId(c).isSPxColId()
                           && (this->number(basis().baseId(c)) == col))
                        break;

                  assert(c < this->nRows());
                  MSG_WARNING((*this->spxout), (*this->spxout) << "WSOLVE92 basis idx:" << c
                              << " fVec:" << fVec()[c]
                              << " fRhs:" << fRhs()[c]
                              << " fTest:" << fTest()[c] << std::endl;)
               }
            }
         }
      }

#endif  // ENABLE_ADDITIONAL_CHECKS


   primalCount = (rep() == SPxSolverBase<R>::COLUMN)
                 ? enterCount
                 : leaveCount;

   printDisplayLine(true);
   performSolutionPolishing();

   return status();
}

template <class R>
void SPxSolverBase<R>::performSolutionPolishing()
{
   // catch rare case that the iteration limit is exactly reached at optimality
   bool stop = (maxIters >= 0 && iterations() >= maxIters && !isTimeLimitReached());

   // only polish an already optimal basis
   if(stop || polishObj == POLISH_OFF || status() != OPTIMAL)
      return;

   int nSuccessfulPivots;
   const typename SPxBasisBase<R>::Desc& ds = this->desc();
   const typename SPxBasisBase<R>::Desc::Status* rowstatus = ds.rowStatus();
   const typename SPxBasisBase<R>::Desc::Status* colstatus = ds.colStatus();
   typename SPxBasisBase<R>::Desc::Status stat;
   SPxId polishId;
   bool success = false;

   MSG_INFO2((*this->spxout), (*this->spxout) << " --- perform solution polishing" << std::endl;)

   if(rep() == COLUMN)
   {
      setType(ENTER); // use primal simplex to preserve feasibility
      init();
#ifndef NDEBUG
      // allow a small relative deviation from the original values
      R alloweddeviation = entertol();
      R origval = value();
      R origshift = shift();
#endif
      instableEnter = false;
      theratiotester->setType(type());

      int nrows = this->nRows();
      int ncols = this->nCols();

      if(polishObj == POLISH_INTEGRALITY)
      {
         DIdxSet slackcandidates(nrows);
         DIdxSet continuousvars(ncols);

         // collect nonbasic slack variables that could be made basic
         for(int i = 0; i < nrows; ++i)
         {
            // only check nonbasic rows, skip equations
            if(rowstatus[i] ==  SPxBasisBase<R>::Desc::P_ON_LOWER
                  || rowstatus[i] == SPxBasisBase<R>::Desc::P_ON_UPPER)
            {
               // only consider rows with zero dual multiplier to preserve optimality
               if(EQrel((*theCoPvec)[i], (R) 0))
                  slackcandidates.addIdx(i);
            }
         }

         // collect continuous variables that could be made basic
         if(integerVariables.size() == ncols)
         {
            for(int i = 0; i < ncols; ++i)
            {
               // skip fixed variables
               if(colstatus[i] == SPxBasisBase<R>::Desc::P_ON_LOWER
                     || colstatus[i] ==  SPxBasisBase<R>::Desc::P_ON_UPPER)
               {
                  // only consider continuous variables with zero dual multiplier to preserve optimality
                  if(EQrel(this->maxObj(i) - (*thePvec)[i], (R) 0) && integerVariables[i] == 0)
                     continuousvars.addIdx(i);
               }
            }
         }

         while(!stop)
         {
            nSuccessfulPivots = 0;

            // identify nonbasic slack variables, i.e. rows, that may be moved into the basis
            for(int i = slackcandidates.size() - 1; i >= 0 && !stop; --i)
            {
               polishId = coId(slackcandidates.index(i));
               MSG_DEBUG(std::cout << "try pivoting: " << polishId << " stat: " << rowstatus[slackcandidates.index(
                            i)];)
               success = enter(polishId, true);
               clearUpdateVecs();
#ifndef NDEBUG
               assert(EQrel(value(), origval, alloweddeviation));
               assert(LErel(shift(), origshift, alloweddeviation));
#endif

               if(success)
               {
                  MSG_DEBUG(std::cout << " -> success!";)
                  ++nSuccessfulPivots;
                  slackcandidates.remove(i);

                  if(maxIters >= 0 && iterations() >= maxIters)
                     stop = true;
               }

               MSG_DEBUG(std::cout << std::endl;)

               if(isTimeLimitReached())
                  stop = true;
            }

            // identify nonbasic variables that may be moved into the basis
            for(int i = continuousvars.size() - 1; i >= 0 && !stop; --i)
            {
               polishId = id(continuousvars.index(i));
               MSG_DEBUG(std::cout << "try pivoting: " << polishId << " stat: " << colstatus[continuousvars.index(
                            i)];)
               success = enter(polishId, true);
               clearUpdateVecs();

               if(success)
               {
                  MSG_DEBUG(std::cout << " -> success!";)
                  ++nSuccessfulPivots;
                  continuousvars.remove(i);

                  if(maxIters >= 0 && iterations() >= maxIters)
                     stop = true;
               }

               MSG_DEBUG(std::cout << std::endl;)

               if(isTimeLimitReached())
                  stop = true;
            }

            // terminate if in the last round no more polishing steps were performed
            if(nSuccessfulPivots == 0)
               stop = true;

            polishCount += nSuccessfulPivots;
         }
      }
      else
      {
         assert(polishObj == POLISH_FRACTIONALITY);
         DIdxSet candidates(dim());

         // identify nonbasic variables, i.e. columns, that may be moved into the basis
         for(int i = 0; i < this->nCols() && !stop; ++i)
         {
            if(colstatus[i] == SPxBasisBase<R>::Desc::P_ON_LOWER
                  || colstatus[i] == SPxBasisBase<R>::Desc::P_ON_UPPER)
            {
               // only consider variables with zero reduced costs to preserve optimality
               if(EQrel(this->maxObj(i) - (*thePvec)[i], (R) 0))
                  candidates.addIdx(i);
            }
         }

         while(!stop)
         {
            nSuccessfulPivots = 0;

            for(int i = candidates.size() - 1; i >= 0 && !stop; --i)
            {
               polishId = id(candidates.index(i));
               MSG_DEBUG(std::cout << "try pivoting: " << polishId << " stat: " << colstatus[candidates.index(i)];)
               success = enter(polishId, true);
               clearUpdateVecs();
#ifndef NDEBUG
               assert(EQrel(value(), origval, alloweddeviation));
               assert(LErel(shift(), origshift, alloweddeviation));
#endif

               if(success)
               {
                  MSG_DEBUG(std::cout << " -> success!";)
                  ++nSuccessfulPivots;
                  candidates.remove(i);

                  if(maxIters >= 0 && iterations() >= maxIters)
                     stop = true;
               }

               MSG_DEBUG(std::cout << std::endl;)

               if(isTimeLimitReached())
                  stop = true;
            }

            // terminate if in the last round no more polishing steps were performed
            if(nSuccessfulPivots == 0)
               stop = true;

            polishCount += nSuccessfulPivots;
         }
      }
   }
   else
   {
      setType(LEAVE); // use primal simplex to preserve feasibility
      init();
#ifndef NDEBUG
      // allow a small relative deviation from the original values
      R alloweddeviation = leavetol();
      R origval = value();
      R origshift = shift();
#endif
      instableLeave = false;
      theratiotester->setType(type());
      bool useIntegrality = false;
      int ncols = this->nCols();

      if(integerVariables.size() == ncols)
         useIntegrality = true;

      // in ROW rep: pivot slack out of the basis
      if(polishObj == POLISH_INTEGRALITY)
      {
         DIdxSet basiccandidates(dim());

         // collect basic candidates that may be moved out of the basis
         for(int i = 0; i < dim(); ++i)
         {
            polishId = this->baseId(i);

            if(polishId.isSPxRowId())
               stat = ds.rowStatus(this->number(polishId));
            else
            {
               // skip (integer) variables
               if(!useIntegrality || integerVariables[this->number(SPxColId(polishId))] == 1)
                  continue;

               stat = ds.colStatus(this->number(polishId));
            }

            if(stat == SPxBasisBase<R>::Desc::P_ON_LOWER || stat ==  SPxBasisBase<R>::Desc::P_ON_UPPER)
            {
               if(EQrel((*theFvec)[i], (R) 0))
                  basiccandidates.addIdx(i);
            }
         }

         while(!stop)
         {
            nSuccessfulPivots = 0;

            for(int i = basiccandidates.size() - 1; i >= 0 && !stop; --i)
            {

               MSG_DEBUG(std::cout << "try pivoting: " << this->baseId(basiccandidates.index(i));)
               success = leave(basiccandidates.index(i), true);
               clearUpdateVecs();
#ifndef NDEBUG
               // assert(EQrel(value(), origval, alloweddeviation));
               assert(LErel(shift(), origshift, alloweddeviation));
#endif

               if(success)
               {
                  MSG_DEBUG(std::cout << " -> success!";)
                  ++nSuccessfulPivots;
                  basiccandidates.remove(i);

                  if(maxIters >= 0 && iterations() >= maxIters)
                     stop = true;
               }

               MSG_DEBUG(std::cout << std::endl;)

               if(isTimeLimitReached())
                  stop = true;
            }

            // terminate if in the last round no more polishing steps were performed
            if(nSuccessfulPivots == 0)
               stop = true;

            polishCount += nSuccessfulPivots;
         }
      }
      else
      {
         assert(polishObj == POLISH_FRACTIONALITY);
         DIdxSet basiccandidates(dim());

         // collect basic (integer) variables, that may be moved out of the basis
         for(int i = 0; i < dim(); ++i)
         {
            polishId = this->baseId(i);

            if(polishId.isSPxRowId())
               continue;
            else
            {
               if(useIntegrality && integerVariables[this->number(SPxColId(polishId))] == 0)
                  continue;

               stat = ds.colStatus(i);
            }

            if(stat == SPxBasisBase<R>::Desc::P_ON_LOWER || stat ==  SPxBasisBase<R>::Desc::P_ON_UPPER)
            {
               if(EQrel((*theFvec)[i], (R) 0))
                  basiccandidates.addIdx(i);
            }
         }

         while(!stop)
         {
            nSuccessfulPivots = 0;

            for(int i = basiccandidates.size() - 1; i >= 0 && !stop; --i)
            {
               MSG_DEBUG(std::cout << "try pivoting: " << this->baseId(basiccandidates.index(i));)
               success = leave(basiccandidates.index(i), true);
               clearUpdateVecs();
#ifndef NDEBUG
               assert(EQrel(value(), origval, alloweddeviation));
               assert(LErel(shift(), origshift, alloweddeviation));
#endif

               if(success)
               {
                  MSG_DEBUG(std::cout << " -> success!";)
                  ++nSuccessfulPivots;
                  basiccandidates.remove(i);

                  if(maxIters >= 0 && iterations() >= maxIters)
                     stop = true;
               }

               MSG_DEBUG(std::cout << std::endl;)

               if(isTimeLimitReached())
                  stop = true;
            }

            // terminate if in the last round no more polishing steps were performed
            if(nSuccessfulPivots == 0)
               stop = true;

            polishCount += nSuccessfulPivots;
         }
      }
   }

   MSG_INFO1((*this->spxout),
             (*this->spxout) << " --- finished solution polishing (" << polishCount << " pivots)" << std::endl;)

   this->setStatus(SPxBasisBase<R>::OPTIMAL);
}


template <class R>
void SPxSolverBase<R>::testVecs()
{

   assert(SPxBasisBase<R>::status() > SPxBasisBase<R>::SINGULAR);

   VectorBase<R> tmp(dim());

   tmp = *theCoPvec;
   this->multWithBase(tmp);
   tmp -= *theCoPrhs;

   if(tmp.length() > leavetol())
   {
      MSG_INFO3((*this->spxout), (*this->spxout) << "ISOLVE93 " << this->iteration() <<
                ":\tcoP error = \t"
                << tmp.length() << std::endl;)

      tmp.clear();
      SPxBasisBase<R>::coSolve(tmp, *theCoPrhs);
      this->multWithBase(tmp);
      tmp -= *theCoPrhs;
      MSG_INFO3((*this->spxout), (*this->spxout) << "ISOLVE94\t\t" << tmp.length() << std::endl;)

      tmp.clear();
      SPxBasisBase<R>::coSolve(tmp, *theCoPrhs);
      tmp -= *theCoPvec;
      MSG_INFO3((*this->spxout), (*this->spxout) << "ISOLVE95\t\t" << tmp.length() << std::endl;)
   }

   tmp = *theFvec;
   this->multBaseWith(tmp);
   tmp -= *theFrhs;

   if(tmp.length() > entertol())
   {
      MSG_INFO3((*this->spxout), (*this->spxout) << "ISOLVE96 " << this->iteration() <<
                ":\t  F error = \t"
                << tmp.length() << std::endl;)

      tmp.clear();
      SPxBasisBase<R>::solve(tmp, *theFrhs);
      tmp -= *theFvec;
      MSG_INFO3((*this->spxout), (*this->spxout) << "ISOLVE97\t\t" << tmp.length() << std::endl;)
   }

   if(type() == ENTER)
   {
      for(int i = 0; i < dim(); ++i)
      {
         if(theCoTest[i] < -leavetol() && isCoBasic(i))
         {
            /// @todo Error message "this shalt not be": shalt this be an assert (also below)?
            MSG_INFO1((*this->spxout), (*this->spxout) << "ESOLVE98 testVecs: theCoTest: this shalt not be!"
                      << std::endl
                      << "  i=" << i
                      << ", theCoTest[i]=" << theCoTest[i]
                      << ", leavetol()=" << leavetol() << std::endl;)
         }
      }

      for(int i = 0; i < coDim(); ++i)
      {
         if(theTest[i] < -leavetol() && isBasic(i))
         {
            MSG_INFO1((*this->spxout), (*this->spxout) << "ESOLVE99 testVecs: theTest: this shalt not be!"
                      << std::endl
                      << "  i=" << i
                      << ", theTest[i]=" << theTest[i]
                      << ", leavetol()=" << leavetol() << std::endl;)
         }
      }
   }
}


/// print display line of flying table
template <class R>
void SPxSolverBase<R>::printDisplayLine(const bool force, const bool forceHead)
{
   MSG_INFO1((*this->spxout),

             if(forceHead || displayLine % (displayFreq * 30) == 0)
{
   (*this->spxout)
            << "type |   time |   iters | facts |    shift | viol sum | viol num | obj value ";

      if(printBasisMetric >= 0)
         (*this->spxout) << " | basis metric";

      (*this->spxout) << std::endl;
   }
   if((force || (displayLine % displayFreq == 0)) && !forceHead)
{
   (type() == LEAVE)
      ? (*this->spxout) << "  L  |" : (*this->spxout) << "  E  |";
      (*this->spxout) << std::fixed << std::setw(7) << std::setprecision(1) << time() << " |";
      (*this->spxout) << std::scientific << std::setprecision(2);
      (*this->spxout) << std::setw(8) << this->iteration() << " | "
                      << std::setw(5) << slinSolver()->getFactorCount() << " | "
                      << shift() << " | "
                      << MAXIMUM(0.0, m_pricingViol + m_pricingViolCo) << " | "
                      << std::setw(8) << MAXIMUM(0, m_numViol) << " | "
                      << std::setprecision(8) << value();

      if(getStartingDecompBasis && rep() == SPxSolverBase<R>::ROW)
         (*this->spxout) << " (" << std::fixed << std::setprecision(2) << getDegeneracyLevel(fVec()) << ")";

      if(printBasisMetric == 0)
         (*this->spxout) << " | " << std::scientific << std::setprecision(2) << getBasisMetric(0);

      if(printBasisMetric == 1)
         (*this->spxout) << " | " << std::scientific << std::setprecision(2) << getBasisMetric(1);

      if(printBasisMetric == 2)
         (*this->spxout) << " | " << std::scientific << std::setprecision(2) << getBasisMetric(2);

      if(printBasisMetric == 3)
         (*this->spxout) << " | " << std::scientific << std::setprecision(2) <<
                         basis().getEstimatedCondition();

      (*this->spxout) << std::endl;
   }
   displayLine++;
            );
}


template <class R>
bool SPxSolverBase<R>::terminate()
{
#ifdef ENABLE_ADDITIONAL_CHECKS

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::SINGULAR)
      testVecs();

#endif

   int redo = dim();

   if(redo < 1000)
      redo = 1000;

   if(this->iteration() > 10 && this->iteration() % redo == 0)
   {
#ifdef ENABLE_ADDITIONAL_CHECKS
      VectorBase<R> cr(*theCoPrhs);
      VectorBase<R> fr(*theFrhs);
#endif

      if(type() == ENTER)
         computeEnterCoPrhs();
      else
         computeLeaveCoPrhs();

      computeFrhs();

#ifdef ENABLE_ADDITIONAL_CHECKS
      cr -= *theCoPrhs;
      fr -= *theFrhs;

      if(cr.length() > leavetol())
         MSG_WARNING((*this->spxout), (*this->spxout) << "WSOLVE50 unexpected change of coPrhs "
                     << cr.length() << std::endl;)
         if(fr.length() > entertol())
            MSG_WARNING((*this->spxout), (*this->spxout) << "WSOLVE51 unexpected change of   Frhs "
                        << fr.length() << std::endl;)
#endif

            if(this->updateCount > 1)
            {
               MSG_INFO3((*this->spxout), (*this->spxout) << " --- terminate triggers refactorization"
                         << std::endl;)
               factorize();
            }

      SPxBasisBase<R>::coSolve(*theCoPvec, *theCoPrhs);
      SPxBasisBase<R>::solve(*theFvec, *theFrhs);

      if(pricing() == FULL)
      {
         computePvec();

         if(type() == ENTER)
         {
            computeCoTest();
            computeTest();
         }
      }

      if(shift() > 0.0)
         unShift();
   }

   // check time limit and objective limit only for non-terminal bases
   if(SPxBasisBase<R>::status() >= SPxBasisBase<R>::OPTIMAL  ||
         SPxBasisBase<R>::status() <= SPxBasisBase<R>::SINGULAR)
   {
      m_status = UNKNOWN;
      return true;
   }

   if(isTimeLimitReached())
   {
      MSG_INFO2((*this->spxout), (*this->spxout) << " --- timelimit (" << maxTime
                << ") reached" << std::endl;)
      m_status = ABORT_TIME;
      return true;
   }

   // objLimit is set and we are running DUAL:
   // - objLimit is set if objLimit < R(infinity)
   // - DUAL is running if rep() * type() > 0 == DUAL (-1 == PRIMAL)
   //
   // In this case we have given a objective value limit, e.g, through a
   // MIP solver, and we want stop solving the LP if we figure out that the
   // optimal value of the current LP can not be better then this objective
   // limit. More precisely:
   // - MINIMIZATION Problem
   //   We want stop the solving process if
   //   objLimit <= current objective value of the DUAL LP
   // - MAXIMIZATION Problem
   //   We want stop the solving process if
   //   objLimit >= current objective value of the DUAL LP
   if(objLimit < R(infinity) && type() * rep() > 0)
   {
      // We have no bound shifts; therefore, we can trust the current
      // objective value.
      // It might be even possible to use this termination value in case of
      // bound violations (shifting) but in this case it is quite difficult
      // to determine if we already reached the limit.
      if(shift() < epsilon() && noViols(opttol() - shift()))
      {
         // SPxSense::MINIMIZE == -1, so we have sign = 1 on minimizing
         if(int(this->spxSense()) * value() <= int(this->spxSense()) * objLimit)
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << " --- objective value limit (" << objLimit
                      << ") reached" << std::endl;)
            MSG_DEBUG(
               (*this->spxout) << " --- objective value limit reached" << std::endl
               << " (value: " << value()
               << ", limit: " << objLimit << ")" << std::endl
               << " (spxSense: " << int(this->spxSense())
               << ", rep: " << int(rep())
               << ", type: " << int(type()) << ")" << std::endl;
            )

            m_status = ABORT_VALUE;
            return true;
         }
      }
   }



   if(getComputeDegeneracy() && this->iteration() > this->prevIteration())
   {
      VectorBase<R> degenvec(this->nCols());

      if(rep() == ROW)
      {
         if(type() == ENTER)     // dual simplex
            dualDegenSum += getDegeneracyLevel(fVec());
         else                    // primal simplex
         {
            getPrimalSol(degenvec);
            primalDegenSum += getDegeneracyLevel(degenvec);
         }
      }
      else
      {
         assert(rep() == COLUMN);

         if(type() == LEAVE)     // dual simplex
            dualDegenSum += getDegeneracyLevel(pVec());
         else
         {
            getPrimalSol(degenvec);
            primalDegenSum += getDegeneracyLevel(degenvec);
         }
      }
   }


   // the improved dual simplex requires a starting basis
   // if the flag getStartingDecompBasis is set to true the simplex will terminate when a dual basis is found
   if(getStartingDecompBasis)
   {
      R iterationFrac = 0.6;

      if(type() == ENTER && SPxBasisBase<R>::status() == SPxBasisBase<R>::DUAL &&
            this->iteration() - this->lastDegenCheck() > getDegenCompOffset()/*iteration() % 10 == 0*/)
      {
         this->iterDegenCheck = this->iterCount;

         if(SPxBasisBase<R>::status() >= SPxBasisBase<R>::OPTIMAL)
         {
            m_status = RUNNING;
            return true;
         }

         R degeneracyLevel = 0;
         R degeneracyLB = 0.1;
         R degeneracyUB = 0.9;
         degeneracyLevel = getDegeneracyLevel(fVec());

         if((degeneracyLevel < degeneracyUB && degeneracyLevel > degeneracyLB)
               && this->iteration() > this->nRows() * 0.2)
         {
            m_status = ABORT_DECOMP;
            return true;
         }

         if(degeneracyLevel < degeneracyLB
               && this->iteration() > MINIMUM(getDecompIterationLimit(), int(this->nCols()*iterationFrac)))
         {
            setDecompIterationLimit(0);
            setDegenCompOffset(0);
            m_status = ABORT_EXDECOMP;
            return true;
         }
      }
      else if(type() == LEAVE
              && this->iteration() > MINIMUM(getDecompIterationLimit(), int(this->nCols()*iterationFrac)))
      {
         setDecompIterationLimit(0);
         setDegenCompOffset(0);
         m_status = ABORT_EXDECOMP;
         return true;
      }
   }

   this->lastIterCount = this->iterCount;

   return false;
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getPrimalSol(VectorBase<R>& p_vector) const
{

   if(!isInitialized())
   {
      /* exit if presolving/simplifier cleared the problem */
      if(status() == NO_PROBLEM)
         return status();

      throw SPxStatusException("XSOLVE06 Not Initialized");
   }

   if(rep() == ROW)
      p_vector = coPvec();
   else
   {
      const typename SPxBasisBase<R>::Desc& ds = this->desc();

      for(int i = 0; i < this->nCols(); ++i)
      {
         switch(ds.colStatus(i))
         {
         case SPxBasisBase<R>::Desc::P_ON_LOWER :
            p_vector[i] = SPxLPBase<R>::lower(i);
            break;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :
         case SPxBasisBase<R>::Desc::P_FIXED :
            p_vector[i] = SPxLPBase<R>::upper(i);
            break;

         case SPxBasisBase<R>::Desc::P_FREE :
            p_vector[i] = 0;
            break;

         case SPxBasisBase<R>::Desc::D_FREE :
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
         case SPxBasisBase<R>::Desc::D_ON_BOTH :
         case SPxBasisBase<R>::Desc::D_UNDEFINED :
            break;

         default:
            throw SPxInternalCodeException("XSOLVE07 This should never happen.");
         }
      }

      for(int j = 0; j < dim(); ++j)
      {
         if(this->baseId(j).isSPxColId())
            p_vector[ this->number(SPxColId(this->baseId(j))) ] = fVec()[j];
      }
   }

   return status();
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getDualSol(VectorBase<R>& p_vector) const
{

   assert(isInitialized());

   if(!isInitialized())
   {
      /* exit if presolving/simplifier cleared the problem */
      if(status() == NO_PROBLEM)
         return status();

      throw SPxStatusException("XSOLVE08 No Problem loaded");
   }

   if(rep() == ROW)
   {
      int i;
      p_vector = this->maxRowObj();

      for(i = this->nCols() - 1; i >= 0; --i)
      {
         if(this->baseId(i).isSPxRowId())
            p_vector[ this->number(SPxRowId(this->baseId(i))) ] = fVec()[i];
      }
   }
   else
   {
      const typename SPxBasisBase<R>::Desc& ds = this->desc();

      for(int i = 0; i < this->nRows(); ++i)
      {
         switch(ds.rowStatus(i))
         {
         case SPxBasisBase<R>::Desc::D_FREE:
         case SPxBasisBase<R>::Desc::D_ON_UPPER:
         case SPxBasisBase<R>::Desc::D_ON_LOWER:
         case SPxBasisBase<R>::Desc::D_ON_BOTH:
         case SPxBasisBase<R>::Desc::D_UNDEFINED:
            p_vector[i] = 0;
            break;

         default:
            p_vector[i] = (*theCoPvec)[i];
         }
      }
   }

   p_vector *= Real(this->spxSense());

   return status();
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getRedCostSol(VectorBase<R>& p_vector) const
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE09 No Problem loaded");
      // return NOT_INIT;
   }

   if(rep() == ROW)
   {
      int i;
      p_vector.clear();

      if(this->spxSense() == SPxLPBase<R>::MINIMIZE)
      {
         for(i = dim() - 1; i >= 0; --i)
         {
            if(this->baseId(i).isSPxColId())
               p_vector[ this->number(SPxColId(this->baseId(i))) ] = -fVec()[i];
         }
      }
      else
      {
         for(i = dim() - 1; i >= 0; --i)
         {
            if(this->baseId(i).isSPxColId())
               p_vector[ this->number(SPxColId(this->baseId(i))) ] = fVec()[i];
         }
      }
   }
   else
   {
      const typename SPxBasisBase<R>::Desc& ds = this->desc();

      for(int i = 0; i < this->nCols(); ++i)
      {
         switch(ds.colStatus(i))
         {
         case SPxBasisBase<R>::Desc::D_FREE:
         case SPxBasisBase<R>::Desc::D_ON_UPPER:
         case SPxBasisBase<R>::Desc::D_ON_LOWER:
         case SPxBasisBase<R>::Desc::D_ON_BOTH:
         case SPxBasisBase<R>::Desc::D_UNDEFINED:
            p_vector[i] = 0;
            break;

         default:
            p_vector[i] = this->maxObj()[i] - (*thePvec)[i];
         }
      }

      if(this->spxSense() == SPxLPBase<R>::MINIMIZE)
         p_vector *= -1.0;
   }

   return status();
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getPrimalray(VectorBase<R>& p_vector) const
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE10 No Problem loaded");
      // return NOT_INIT;
   }

   assert(SPxBasisBase<R>::status() == SPxBasisBase<R>::UNBOUNDED);
   p_vector.clear();
   p_vector = primalRay;

   return status();
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getDualfarkas(VectorBase<R>& p_vector) const
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE10 No Problem loaded");
      // return NOT_INIT;
   }

   assert(SPxBasisBase<R>::status() == SPxBasisBase<R>::INFEASIBLE);
   p_vector.clear();
   p_vector = dualFarkas;

   return status();
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getSlacks(VectorBase<R>& p_vector) const
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE11 No Problem loaded");
      // return NOT_INIT;
   }

   if(rep() == COLUMN)
   {
      int i;
      const typename SPxBasisBase<R>::Desc& ds = this->desc();

      for(i = this->nRows() - 1; i >= 0; --i)
      {
         switch(ds.rowStatus(i))
         {
         case SPxBasisBase<R>::Desc::P_ON_LOWER :
            p_vector[i] = this->lhs(i);
            break;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :
         case SPxBasisBase<R>::Desc::P_FIXED :
            p_vector[i] = this->rhs(i);
            break;

         case SPxBasisBase<R>::Desc::P_FREE :
            p_vector[i] = 0;
            break;

         case SPxBasisBase<R>::Desc::D_FREE :
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
         case SPxBasisBase<R>::Desc::D_ON_BOTH :
         case SPxBasisBase<R>::Desc::D_UNDEFINED :
            break;

         default:
            throw SPxInternalCodeException("XSOLVE12 This should never happen.");
         }
      }

      for(i = dim() - 1; i >= 0; --i)
      {
         if(this->baseId(i).isSPxRowId())
            p_vector[ this->number(SPxRowId(this->baseId(i))) ] = -(*theFvec)[i];
      }
   }
   else
      p_vector = pVec();

   return status();
}

template <class R>
void SPxSolverBase<R>::setPrimal(VectorBase<R>& p_vector)
{

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE20 Not Initialized");
   }

   if(rep() == ROW)
      coPvec() = p_vector;
   else
   {
      for(int j = 0; j < dim(); ++j)
      {
         if(this->baseId(j).isSPxColId())
            fVec()[j] = p_vector[ this->number(SPxColId(this->baseId(j))) ];
      }
   }
}

template <class R>
void SPxSolverBase<R>::setDual(VectorBase<R>& p_vector)
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE21 Not Initialized");
   }

   if(rep() == ROW)
   {
      for(int i = this->nCols() - 1; i >= 0; --i)
      {
         if(this->baseId(i).isSPxRowId())
         {
            if(this->spxSense() == SPxLPBase<R>::MAXIMIZE)
               fVec()[i] = p_vector[ this->number(SPxRowId(this->baseId(i))) ];
            else
               fVec()[i] = -p_vector[ this->number(SPxRowId(this->baseId(i))) ];
         }
      }
   }
   else
   {
      coPvec() = p_vector;

      if(this->spxSense() == SPxLPBase<R>::MINIMIZE)
         coPvec() *= -1.0;
   }
}

template <class R>
void SPxSolverBase<R>::setRedCost(VectorBase<R>& p_vector)
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE22 Not Initialized");
   }

   if(rep() == ROW)
   {
      for(int i = dim() - 1; i >= 0; --i)
      {
         if(this->baseId(i).isSPxColId())
         {
            if(this->spxSense() == SPxLPBase<R>::MINIMIZE)
               fVec()[i] = -p_vector[ this->number(SPxColId(this->baseId(i))) ];
            else
               fVec()[i] = p_vector[ this->number(SPxColId(this->baseId(i))) ];
         }
      }
   }
   else
   {
      pVec() = this->maxObj();

      if(this->spxSense() == SPxLPBase<R>::MINIMIZE)
         pVec() += p_vector;
      else
         pVec() -= p_vector;
   }
}

template <class R>
void SPxSolverBase<R>::setSlacks(VectorBase<R>& p_vector)
{

   assert(isInitialized());

   if(!isInitialized())
   {
      throw SPxStatusException("XSOLVE23 Not Initialized");
   }

   if(rep() == COLUMN)
   {
      for(int i = dim() - 1; i >= 0; --i)
      {
         if(this->baseId(i).isSPxRowId())
            (*theFvec)[i] = -p_vector[ this->number(SPxRowId(this->baseId(i))) ];
      }
   }
   else
      pVec() = p_vector;
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::status() const
{
   switch(m_status)
   {
   case UNKNOWN :
      switch(SPxBasisBase<R>::status())
      {
      case SPxBasisBase<R>::NO_PROBLEM :
         return NO_PROBLEM;

      case SPxBasisBase<R>::SINGULAR :
         return SINGULAR;

      case SPxBasisBase<R>::REGULAR :
      case SPxBasisBase<R>::DUAL :
      case SPxBasisBase<R>::PRIMAL :
         return UNKNOWN;

      case SPxBasisBase<R>::OPTIMAL :
         return OPTIMAL;

      case SPxBasisBase<R>::UNBOUNDED :
         return UNBOUNDED;

      case SPxBasisBase<R>::INFEASIBLE :
         return INFEASIBLE;

      default:
         return ERROR;
      }

   case SINGULAR :
      return m_status;

   case OPTIMAL :
      assert(SPxBasisBase<R>::status() == SPxBasisBase<R>::OPTIMAL);

   /*lint -fallthrough*/
   case ABORT_EXDECOMP :
   case ABORT_DECOMP :
   case ABORT_CYCLING :
   case ABORT_TIME :
   case ABORT_ITER :
   case ABORT_VALUE :
   case RUNNING :
   case REGULAR :
   case NOT_INIT :
   case NO_SOLVER :
   case NO_PRICER :
   case NO_RATIOTESTER :
   case ERROR:
      return m_status;

   default:
      return ERROR;
   }
}

template <class R>
typename SPxSolverBase<R>::Status SPxSolverBase<R>::getResult(
   R* p_value,
   VectorBase<R>* p_primal,
   VectorBase<R>* p_slacks,
   VectorBase<R>* p_dual,
   VectorBase<R>* reduCosts)
{
   if(p_value)
      *p_value = this->value();

   if(p_primal)
      getPrimalSol(*p_primal);

   if(p_slacks)
      getSlacks(*p_slacks);

   if(p_dual)
      getDualSol(*p_dual);

   if(reduCosts)
      getRedCostSol(*reduCosts);

   return status();
}
} // namespace soplex
