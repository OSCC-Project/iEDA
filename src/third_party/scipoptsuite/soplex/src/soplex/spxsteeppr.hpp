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

//TODO may be faster to have a greater zero tolerance for sparse pricing vectors
//     to reduce the number of nonzero entries, e.g. for workVec

#include <assert.h>
#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/spxsteeppr.h"
#include "soplex/random.h"

#define STEEP_REFINETOL 2.0

namespace soplex
{
// #define EQ_PREF 1000

template <class R>
void SPxSteepPR<R>::clear()
{
   this->thesolver = 0;
}

template <class R>
void SPxSteepPR<R>::load(SPxSolverBase<R>* base)
{
   this->thesolver = base;

   if(base)
   {
      workVec.clear();
      workVec.reDim(base->dim());
      workRhs.clear();
      workRhs.reDim(base->dim());
   }
}

template <class R>
void SPxSteepPR<R>::setType(typename SPxSolverBase<R>::Type type)
{
   workRhs.setEpsilon(this->thesolver->epsilon());

   setupWeights(type);
   workVec.clear();
   workRhs.clear();
   refined = false;

   bestPrices.clear();
   bestPrices.setMax(this->thesolver->dim());
   prices.reSize(this->thesolver->dim());

   if(type == SPxSolverBase<R>::ENTER)
   {
      bestPricesCo.clear();
      bestPricesCo.setMax(this->thesolver->coDim());
      pricesCo.reSize(this->thesolver->coDim());
   }
}

template <class R>
void SPxSteepPR<R>::setupWeights(typename SPxSolverBase<R>::Type type)
{
   int i;
   int endDim = 0;
   int endCoDim = 0;
   VectorBase<R>& weights = this->thesolver->weights;
   VectorBase<R>& coWeights = this->thesolver->coWeights;

   if(setup == DEFAULT)
   {
      if(type == SPxSolverBase<R>::ENTER)
      {
         if(this->thesolver->weightsAreSetup)
         {
            // check for added/removed rows and adapt norms accordingly
            if(coWeights.dim() < this->thesolver->dim())
               endDim = coWeights.dim();
            else
               endDim = this->thesolver->dim();

            if(weights.dim() < this->thesolver->coDim())
               endCoDim = weights.dim();
            else
               endCoDim = this->thesolver->coDim();
         }

         coWeights.reDim(this->thesolver->dim(), false);

         for(i = this->thesolver->dim() - 1; i >= endDim; --i)
            coWeights[i] = 2.0;

         weights.reDim(this->thesolver->coDim(), false);

         for(i = this->thesolver->coDim() - 1; i >= endCoDim; --i)
            weights[i] = 1.0;
      }
      else
      {
         assert(type == SPxSolverBase<R>::LEAVE);

         if(this->thesolver->weightsAreSetup)
         {
            // check for added/removed rows and adapt norms accordingly
            if(coWeights.dim() < this->thesolver->dim())
               endDim = coWeights.dim();
            else
               endDim = this->thesolver->dim();
         }

         coWeights.reDim(this->thesolver->dim(), false);

         for(i = this->thesolver->dim() - 1; i >= endDim; --i)
            coWeights[i]   = 1.0;
      }
   }
   else
   {
      MSG_INFO1((*this->thesolver->spxout),
                (*this->thesolver->spxout) << " --- initializing steepest edge multipliers" << std::endl;)

      if(type == SPxSolverBase<R>::ENTER)
      {
         coWeights.reDim(this->thesolver->dim(), false);

         for(i = this->thesolver->dim() - 1; i >= endDim; --i)
            coWeights[i] = 1.0;

         weights.reDim(this->thesolver->coDim(), false);

         for(i = this->thesolver->coDim() - 1; i >= endCoDim; --i)
            weights[i] = 1.0 + this->thesolver->vector(i).length2();
      }
      else
      {
         assert(type == SPxSolverBase<R>::LEAVE);
         coWeights.reDim(this->thesolver->dim(), false);
         SSVectorBase<R>  tmp(this->thesolver->dim(), this->thesolver->epsilon());

         for(i = this->thesolver->dim() - 1; i >= endDim && !this->thesolver->isTimeLimitReached(); --i)
         {
            this->thesolver->basis().coSolve(tmp, this->thesolver->unitVector(i));
            coWeights[i] = tmp.length2();
         }
      }
   }

   this->thesolver->weightsAreSetup = true;
}

template <class R>
void SPxSteepPR<R>::setRep(typename SPxSolverBase<R>::Representation)
{
   if(workVec.dim() != this->thesolver->dim())
   {
      VectorBase<R> tmp = this->thesolver->weights;
      this->thesolver->weights = this->thesolver->coWeights;
      this->thesolver->coWeights = tmp;

      workVec.clear();
      workVec.reDim(this->thesolver->dim());
   }
}

template <class R>
void SPxSteepPR<R>::left4(int n, SPxId id)
{
   assert(this->thesolver->type() == SPxSolverBase<R>::LEAVE);

   if(id.isValid())
   {
      R        delta         = 0.1 + 1.0 / this->thesolver->basis().iteration();
      R*       coWeights_ptr = this->thesolver->coWeights.get_ptr();
      const R* workVec_ptr   = workVec.get_const_ptr();
      const R* rhoVec        = this->thesolver->fVec().delta().values();
      R        rhov_1        = 1.0 / rhoVec[n];
      R        beta_q        = this->thesolver->coPvec().delta().length2() * rhov_1 * rhov_1;

#ifndef NDEBUG

      if(spxAbs(rhoVec[n]) < this->theeps * 0.5)
      {
         MSG_INFO3((*this->thesolver->spxout), (*this->thesolver->spxout) << "WSTEEP04: rhoVec = "
                   << rhoVec[n] << " with smaller absolute value than 0.5*theeps = " << 0.5 * this->theeps <<
                   std::endl;)
      }

#endif

      const IdxSet& rhoIdx = this->thesolver->fVec().idx();
      int           len    = this->thesolver->fVec().idx().size();

      for(int i = 0; i < len; ++i)
      {
         int  j = rhoIdx.index(i);
         coWeights_ptr[j] += rhoVec[j] * (beta_q * rhoVec[j] - 2.0 * rhov_1 * workVec_ptr[j]);

         if(coWeights_ptr[j] < delta)
            coWeights_ptr[j] = delta; // coWeights_ptr[j] = delta / (1+delta-x);
         else if(coWeights_ptr[j] >= R(infinity))
            coWeights_ptr[j] = 1.0 / this->theeps;
      }

      coWeights_ptr[n] = beta_q;
   }
}


namespace steeppr
{
template <class R>
R computePrice(R viol, R weight, R tol)
{
   if(weight < tol)
      return viol * viol / tol;
   else
      return viol * viol / weight;
}
}

template <class R>
int SPxSteepPR<R>::buildBestPriceVectorLeave(R feastol)
{
   int idx;
   int nsorted;
   R x;
   const R* fTest = this->thesolver->fTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   typename SPxPricer<R>::IdxElement price;
   prices.clear();
   bestPrices.clear();

   // construct vector of all prices
   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = fTest[idx];

      if(x < -feastol)
      {
         // it might happen that we call the pricer with a tighter tolerance than what was used when computing the violations
         this->thesolver->isInfeasible[idx] = this->VIOLATED;
         price.val = steeppr::computePrice(x, cpen[idx], feastol);
         price.idx = idx;
         prices.append(price);
      }
   }

   // set up structures for the quicksort implementation
   this->compare.elements = prices.get_const_ptr();
   // do a partial sort to move the best ones to the front
   // TODO this can be done more efficiently, since we only need the indices
   nsorted = SPxQuicksortPart(prices.get_ptr(), this->compare, 0, prices.size(), HYPERPRICINGSIZE);

   // copy indices of best values to bestPrices
   for(int i = 0; i < nsorted; ++i)
   {
      bestPrices.addIdx(prices[i].idx);
      this->thesolver->isInfeasible[prices[i].idx] = this->VIOLATED_AND_CHECKED;
   }

   if(nsorted > 0)
      return prices[0].idx;
   else
      return -1;
}


template <class R>
int SPxSteepPR<R>::selectLeave()
{
   assert(isConsistent());

   int retid;

   if(this->thesolver->hyperPricingLeave && this->thesolver->sparsePricingLeave)
   {
      if(bestPrices.size() < 2 || this->thesolver->basis().lastUpdate() == 0)
      {
         // call init method to build up price-vector and return index of largest price
         retid = buildBestPriceVectorLeave(this->theeps);
      }
      else
         retid = selectLeaveHyper(this->theeps);
   }
   else if(this->thesolver->sparsePricingLeave)
      retid = selectLeaveSparse(this->theeps);
   else
      retid = selectLeaveX(this->theeps);

   if(retid < 0 && !refined)
   {
      refined = true;
      MSG_INFO3((*this->thesolver->spxout),
                (*this->thesolver->spxout) << "WSTEEP03 trying refinement step..\n";)
      retid = selectLeaveX(this->theeps / STEEP_REFINETOL);
   }

   if(retid >= 0)
   {
      assert(this->thesolver->coPvec().delta().isConsistent());
      // coPvec().delta() might be not setup after the solve when it contains too many nonzeros.
      // This is intended and forcing to keep the sparsity information leads to a slowdown
      // TODO implement a dedicated solve method for unitvectors
      this->thesolver->basis().coSolve(this->thesolver->coPvec().delta(),
                                       this->thesolver->unitVector(retid));
      assert(this->thesolver->coPvec().delta().isConsistent());
      workRhs.setup_and_assign(this->thesolver->coPvec().delta());
      this->thesolver->setup4solve(&workVec, &workRhs);
   }

   return retid;
}

template <class R>
int SPxSteepPR<R>::selectLeaveX(R tol)
{
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();
   const R* fTest         = this->thesolver->fTest().get_const_ptr();
   R best = R(-infinity);
   R x;
   int lastIdx = -1;

   for(int i = this->thesolver->dim() - 1; i >= 0; --i)
   {
      x = fTest[i];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, coWeights_ptr[i], tol);

         if(x > best)
         {
            best = x;
            lastIdx = i;
         }
      }
   }

   return lastIdx;
}

template <class R>
int SPxSteepPR<R>::selectLeaveSparse(R tol)
{
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();
   const R* fTest         = this->thesolver->fTest().get_const_ptr();
   R best = R(-infinity);
   R x;
   int lastIdx = -1;
   int idx;

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = fTest[idx];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, coWeights_ptr[idx], tol);

         if(x > best)
         {
            best = x;
            lastIdx = idx;
         }
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);
         assert(this->thesolver->isInfeasible[idx] == this->VIOLATED
                || this->thesolver->isInfeasible[idx] == this->VIOLATED_AND_CHECKED);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   return lastIdx;
}

template <class R>
int SPxSteepPR<R>::selectLeaveHyper(R tol)
{
   const R* coPen = this->thesolver->coWeights.get_const_ptr();
   const R* fTest = this->thesolver->fTest().get_const_ptr();
   R leastBest = -1;
   R best = R(-infinity);
   R x;
   int bestIdx = -1;
   int idx = 0;

   // find the best price from the short candidate list
   for(int i = bestPrices.size() - 1; i >= 0; --i)
   {
      idx = bestPrices.index(i);
      x = fTest[idx];

      if(x < -tol)
      {
         assert(this->thesolver->isInfeasible[idx] == this->VIOLATED
                || this->thesolver->isInfeasible[idx] == this->VIOLATED_AND_CHECKED);
         x = steeppr::computePrice(x, coPen[idx], tol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            bestIdx = idx;
         }

         // update the smallest price of candidate list
         if(x < leastBest || leastBest < 0)
            leastBest = x;
      }
      else
      {
         bestPrices.remove(i);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   // scan the updated indices for a better price
   for(int i = this->thesolver->updateViols.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->updateViols.index(i);

      // is this index a candidate for bestPrices?
      if(this->thesolver->isInfeasible[idx] == this->VIOLATED)
      {
         x = fTest[idx];
         assert(x < -tol);
         x = steeppr::computePrice(x, coPen[idx], tol);

         if(x > leastBest)
         {
            if(x > best)
            {
               best = x;
               bestIdx = idx;
            }

            this->thesolver->isInfeasible[idx] = this->VIOLATED_AND_CHECKED;
            bestPrices.addIdx(idx);
         }
      }
   }

   return bestIdx;
}

/* Entering Simplex
 */
template <class R>
void SPxSteepPR<R>::entered4(SPxId /* id */, int n)
{
   assert(this->thesolver->type() == SPxSolverBase<R>::ENTER);

   if(n >= 0 && n < this->thesolver->dim())
   {
      R delta = 2 + 1.0 / this->thesolver->basis().iteration();
      R* coWeights_ptr = this->thesolver->coWeights.get_ptr();
      R* weights_ptr = this->thesolver->weights.get_ptr();
      const R* workVec_ptr = workVec.get_const_ptr();
      const R* pVec = this->thesolver->pVec().delta().values();
      const IdxSet& pIdx = this->thesolver->pVec().idx();
      const R* coPvec = this->thesolver->coPvec().delta().values();
      const IdxSet& coPidx = this->thesolver->coPvec().idx();
      R xi_p = 1 / this->thesolver->fVec().delta()[n];
      int i, j;
      R xi_ip;

      assert(this->thesolver->fVec().delta()[n] > this->thesolver->epsilon()
             || this->thesolver->fVec().delta()[n] < -this->thesolver->epsilon());

      for(j = coPidx.size() - 1; j >= 0; --j)
      {
         i = coPidx.index(j);
         xi_ip = xi_p * coPvec[i];
         coWeights_ptr[i] += xi_ip * (xi_ip * pi_p - 2.0 * workVec_ptr[i]);

         /*
           if(coWeights_ptr[i] < 1)
           coWeights_ptr[i] = 1 / (2-x);
         */
         if(coWeights_ptr[i] < delta)
            coWeights_ptr[i] = delta;
         // coWeights_ptr[i] = 1;
         else if(coWeights_ptr[i] > R(infinity))
            coWeights_ptr[i] = 1 / this->thesolver->epsilon();
      }

      for(j = pIdx.size() - 1; j >= 0; --j)
      {
         i = pIdx.index(j);
         xi_ip = xi_p * pVec[i];
         weights_ptr[i] += xi_ip * (xi_ip * pi_p - 2.0 * (this->thesolver->vector(i) * workVec));

         /*
           if(weights_ptr[i] < 1)
           weights_ptr[i] = 1 / (2-x);
         */
         if(weights_ptr[i] < delta)
            weights_ptr[i] = delta;
         // weights_ptr[i] = 1;
         else if(weights_ptr[i] > R(infinity))
            weights_ptr[i] = 1.0 / this->thesolver->epsilon();
      }
   }

   /*@
     if(this->thesolver->isId(id))
     weights[   this->thesolver->number(id) ] *= 1.0001;
     else if(this->thesolver->isCoId(id))
     coWeights[ this->thesolver->number(id) ] *= 1.0001;
   */

}


template <class R>
SPxId SPxSteepPR<R>::buildBestPriceVectorEnterDim(R& best, R feastol)
{
   const R* coTest        = this->thesolver->coTest().get_const_ptr();
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();
   int idx;
   int nsorted;
   R x;
   typename SPxPricer<R>::IdxElement price;

   prices.clear();
   bestPrices.clear();

   // construct vector of all prices
   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = coTest[idx];

      if(x < -feastol)
      {
         // it might happen that we call the pricer with a tighter tolerance than what was used when computing the violations
         this->thesolver->isInfeasible[idx] = this->VIOLATED;
         price.val = steeppr::computePrice(x, coWeights_ptr[idx], feastol);
         price.idx = idx;
         prices.append(price);
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   // set up structures for the quicksort implementation
   this->compare.elements = prices.get_const_ptr();
   // do a partial sort to move the best ones to the front
   // TODO this can be done more efficiently, since we only need the indices
   nsorted = SPxQuicksortPart(prices.get_ptr(), this->compare, 0, prices.size(), HYPERPRICINGSIZE);

   // copy indices of best values to bestPrices
   for(int i = 0; i < nsorted; ++i)
   {
      bestPrices.addIdx(prices[i].idx);
      this->thesolver->isInfeasible[prices[i].idx] = this->VIOLATED_AND_CHECKED;
   }

   if(nsorted > 0)
   {
      best = prices[0].val;
      return this->thesolver->coId(prices[0].idx);
   }
   else
      return SPxId();
}


template <class R>
SPxId SPxSteepPR<R>::buildBestPriceVectorEnterCoDim(R& best, R feastol)
{
   const R* test        = this->thesolver->test().get_const_ptr();
   const R* weights_ptr = this->thesolver->weights.get_const_ptr();
   int idx;
   int nsorted;
   R x;
   typename SPxPricer<R>::IdxElement price;

   pricesCo.clear();
   bestPricesCo.clear();

   // construct vector of all prices
   for(int i = this->thesolver->infeasibilitiesCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilitiesCo.index(i);
      x = test[idx];

      if(x < -feastol)
      {
         // it might happen that we call the pricer with a tighter tolerance than what was used when computing the violations
         this->thesolver->isInfeasibleCo[idx] = this->VIOLATED;
         price.val = steeppr::computePrice(x, weights_ptr[idx], feastol);
         price.idx = idx;
         pricesCo.append(price);
      }
      else
      {
         this->thesolver->infeasibilitiesCo.remove(i);
         this->thesolver->isInfeasibleCo[idx] = this->NOT_VIOLATED;
      }
   }

   // set up structures for the quicksort implementation
   this->compare.elements = pricesCo.get_const_ptr();
   // do a partial sort to move the best ones to the front
   // TODO this can be done more efficiently, since we only need the indices
   nsorted = SPxQuicksortPart(pricesCo.get_ptr(), this->compare, 0, pricesCo.size(), HYPERPRICINGSIZE);

   // copy indices of best values to bestPrices
   for(int i = 0; i < nsorted; ++i)
   {
      bestPricesCo.addIdx(pricesCo[i].idx);
      this->thesolver->isInfeasibleCo[pricesCo[i].idx] = this->VIOLATED_AND_CHECKED;
   }

   if(nsorted > 0)
   {
      best = pricesCo[0].val;
      return this->thesolver->id(pricesCo[0].idx);
   }
   else
      return SPxId();
}


template <class R>
SPxId SPxSteepPR<R>::selectEnter()
{
   assert(this->thesolver != 0);
   SPxId enterId;

   enterId = selectEnterX(this->theeps);

   if(!enterId.isValid() && !refined)
   {
      refined = true;
      MSG_INFO3((*this->thesolver->spxout),
                (*this->thesolver->spxout) << "WSTEEP05 trying refinement step..\n";)
      enterId = selectEnterX(this->theeps / STEEP_REFINETOL);
   }

   assert(isConsistent());

   if(enterId.isValid())
   {
      SSVectorBase<R>& delta = this->thesolver->fVec().delta();

      this->thesolver->basis().solve4update(delta, this->thesolver->vector(enterId));

      workRhs.setup_and_assign(delta);
      pi_p = 1 + delta.length2();

      this->thesolver->setup4coSolve(&workVec, &workRhs);
   }

   return enterId;
}

template <class R>
SPxId SPxSteepPR<R>::selectEnterX(R tol)
{
   SPxId enterId;
   SPxId enterCoId;
   R best;
   R bestCo;

   best = R(-infinity);
   bestCo = R(-infinity);

   if(this->thesolver->hyperPricingEnter && !refined)
   {
      if(bestPrices.size() < 2 || this->thesolver->basis().lastUpdate() == 0)
         enterCoId = (this->thesolver->sparsePricingEnter) ? buildBestPriceVectorEnterDim(best,
                     tol) : selectEnterDenseDim(best, tol);
      else
         enterCoId = (this->thesolver->sparsePricingEnter) ? selectEnterHyperDim(best,
                     tol) : selectEnterDenseDim(best, tol);

      if(bestPricesCo.size() < 2 || this->thesolver->basis().lastUpdate() == 0)
         enterId = (this->thesolver->sparsePricingEnterCo) ? buildBestPriceVectorEnterCoDim(bestCo,
                   tol) : selectEnterDenseCoDim(bestCo, tol);
      else
         enterId = (this->thesolver->sparsePricingEnterCo) ? selectEnterHyperCoDim(bestCo,
                   tol) : selectEnterDenseCoDim(bestCo, tol);
   }
   else
   {
      enterCoId = (this->thesolver->sparsePricingEnter
                   && !refined) ? selectEnterSparseDim(best, tol) : selectEnterDenseDim(best, tol);
      enterId = (this->thesolver->sparsePricingEnterCo
                 && !refined) ? selectEnterSparseCoDim(bestCo, tol) : selectEnterDenseCoDim(bestCo, tol);
   }

   // prefer slack indices to reduce nonzeros in basis matrix
   if(enterCoId.isValid() && (best > SPARSITY_TRADEOFF * bestCo || !enterId.isValid()))
      return enterCoId;
   else
      return enterId;
}


template <class R>
SPxId SPxSteepPR<R>::selectEnterHyperDim(R& best, R tol)
{
   const R* coTest        = this->thesolver->coTest().get_const_ptr();
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();

   R leastBest = -1;
   R x;
   int enterIdx = -1;
   int idx;

   // find the best price from short candidate list
   for(int i = bestPrices.size() - 1; i >= 0; --i)
   {
      idx = bestPrices.index(i);
      x = coTest[idx];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, coWeights_ptr[idx], tol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            enterIdx = idx;
         }

         // update the smallest price of candidate list
         if(x < leastBest || leastBest < 0)
            leastBest = x;
      }
      else
      {
         bestPrices.remove(i);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   // scan the updated indices for a better price
   for(int i = this->thesolver->updateViols.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->updateViols.index(i);

      // only look at indices that were not checked already
      if(this->thesolver->isInfeasible[idx] == this->VIOLATED)
      {
         x = coTest[idx];

         if(x < -tol)
         {
            x = steeppr::computePrice(x, coWeights_ptr[idx], tol);

            if(x > leastBest)
            {
               if(x > best)
               {
                  best = x;
                  enterIdx = idx;
               }

               // put index into candidate list
               this->thesolver->isInfeasible[idx] = this->VIOLATED_AND_CHECKED;
               bestPrices.addIdx(idx);
            }
         }
         else
         {
            this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
         }
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->coId(enterIdx);
   else
      return SPxId();
}


template <class R>
SPxId SPxSteepPR<R>::selectEnterHyperCoDim(R& best, R tol)
{
   const R* test        = this->thesolver->test().get_const_ptr();
   const R* weights_ptr = this->thesolver->weights.get_const_ptr();

   R leastBest = -1;
   R x;
   int enterIdx = -1;
   int idx;

   // find the best price from short candidate list
   for(int i = bestPricesCo.size() - 1; i >= 0; --i)
   {
      idx = bestPricesCo.index(i);
      x = test[idx];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, weights_ptr[idx], tol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            enterIdx = idx;
         }

         // update the smallest price of candidate list
         if(x < leastBest || leastBest < 0)
            leastBest = x;
      }
      else
      {
         bestPricesCo.remove(i);
         this->thesolver->isInfeasibleCo[idx] = this->NOT_VIOLATED;
      }
   }

   // scan the updated indices for a better price
   for(int i = this->thesolver->updateViolsCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->updateViolsCo.index(i);

      // only look at indices that were not checked already
      if(this->thesolver->isInfeasibleCo[idx] == this->VIOLATED)
      {
         x = test[idx];

         if(x < -tol)
         {
            x = steeppr::computePrice(x, weights_ptr[idx], tol);

            if(x > leastBest)
            {
               if(x > best)
               {
                  best = x;
                  enterIdx = idx;
               }

               // put index into candidate list
               this->thesolver->isInfeasibleCo[idx] = this->VIOLATED_AND_CHECKED;
               bestPricesCo.addIdx(idx);
            }
         }
         else
         {
            this->thesolver->isInfeasibleCo[idx] = this->NOT_VIOLATED;
         }
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->id(enterIdx);
   else
      return SPxId();
}


template <class R>
SPxId SPxSteepPR<R>::selectEnterSparseDim(R& best, R tol)
{
   SPxId enterId;
   const R* coTest        = this->thesolver->coTest().get_const_ptr();
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();

   int idx;
   R x;

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = coTest[idx];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, coWeights_ptr[idx], tol);

         if(x > best)
         {
            best = x;
            enterId = this->thesolver->coId(idx);
         }
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   return enterId;
}

template <class R>
SPxId SPxSteepPR<R>::selectEnterSparseCoDim(R& best, R tol)
{
   SPxId enterId;
   const R* test          = this->thesolver->test().get_const_ptr();
   const R* weights_ptr   = this->thesolver->weights.get_const_ptr();

   int idx;
   R x;

   for(int i = this->thesolver->infeasibilitiesCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilitiesCo.index(i);
      x = test[idx];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, weights_ptr[idx], tol);

         if(x > best)
         {
            best   = x;
            enterId = this->thesolver->id(idx);
         }
      }
      else
      {
         this->thesolver->infeasibilitiesCo.remove(i);
         this->thesolver->isInfeasibleCo[idx] = this->NOT_VIOLATED;
      }
   }

   return enterId;
}

template <class R>
SPxId SPxSteepPR<R>::selectEnterDenseDim(R& best, R tol)
{
   SPxId enterId;
   const R* coTest        = this->thesolver->coTest().get_const_ptr();
   const R* coWeights_ptr = this->thesolver->coWeights.get_const_ptr();

   R x;

   for(int i = 0, end = this->thesolver->dim(); i < end; ++i)
   {
      x = coTest[i];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, coWeights_ptr[i], tol);

         if(x > best)
         {
            best = x;
            enterId = this->thesolver->coId(i);
         }
      }
   }

   return enterId;
}

template <class R>
SPxId SPxSteepPR<R>::selectEnterDenseCoDim(R& best, R tol)
{
   SPxId enterId;
   const R* test          = this->thesolver->test().get_const_ptr();
   const R* weights_ptr   = this->thesolver->weights.get_const_ptr();

   R x;

   for(int i = 0, end = this->thesolver->coDim(); i < end; ++i)
   {
      x = test[i];

      if(x < -tol)
      {
         x = steeppr::computePrice(x, weights_ptr[i], tol);

         if(x > best)
         {
            best   = x;
            enterId = this->thesolver->id(i);
         }
      }
   }

   return enterId;
}


template <class R>
void SPxSteepPR<R>::addedVecs(int n)
{
   VectorBase<R>& weights = this->thesolver->weights;
   n = weights.dim();
   weights.reDim(this->thesolver->coDim());

   if(this->thesolver->type() == SPxSolverBase<R>::ENTER)
   {
      for(; n < weights.dim(); ++n)
         weights[n] = 2;
   }
}

template <class R>
void SPxSteepPR<R>::addedCoVecs(int n)
{
   VectorBase<R>& coWeights = this->thesolver->coWeights;
   n = coWeights.dim();
   workVec.reDim(this->thesolver->dim());
   coWeights.reDim(this->thesolver->dim());

   for(; n < coWeights.dim(); ++n)
      coWeights[n] = 1;
}

template <class R>
void SPxSteepPR<R>::removedVec(int i)
{
   assert(this->thesolver != 0);
   VectorBase<R>& weights = this->thesolver->weights;
   weights[i] = weights[weights.dim()];
   weights.reDim(this->thesolver->coDim());
}

template <class R>
void SPxSteepPR<R>::removedVecs(const int perm[])
{
   assert(this->thesolver != 0);
   VectorBase<R>& weights = this->thesolver->weights;

   if(this->thesolver->type() == SPxSolverBase<R>::ENTER)
   {
      int i;
      int j = weights.dim();

      for(i = 0; i < j; ++i)
      {
         if(perm[i] >= 0)
            weights[perm[i]] = weights[i];
      }
   }

   weights.reDim(this->thesolver->coDim());
}

template <class R>
void SPxSteepPR<R>::removedCoVec(int i)
{
   assert(this->thesolver != 0);
   VectorBase<R>& coWeights = this->thesolver->coWeights;
   coWeights[i] = coWeights[coWeights.dim()];
   coWeights.reDim(this->thesolver->dim());
}

template <class R>
void SPxSteepPR<R>::removedCoVecs(const int perm[])
{
   assert(this->thesolver != 0);
   VectorBase<R>& coWeights = this->thesolver->coWeights;
   int i;
   int j = coWeights.dim();

   for(i = 0; i < j; ++i)
   {
      if(perm[i] >= 0)
         coWeights[perm[i]] = coWeights[i];
   }

   coWeights.reDim(this->thesolver->dim());
}

template <class R>
bool SPxSteepPR<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   VectorBase<R>& w = this->thesolver->weights;
   VectorBase<R>& coW = this->thesolver->coWeights;

   if(this->thesolver != 0 && this->thesolver->type() == SPxSolverBase<R>::LEAVE && setup == EXACT)
   {
      int i;
      SSVectorBase<R>  tmp(this->thesolver->dim(), this->thesolver->epsilon());
      R x;

      for(i = this->thesolver->dim() - 1; i >= 0; --i)
      {
         this->thesolver->basis().coSolve(tmp, this->thesolver->unitVector(i));
         x = coW[i] - tmp.length2();

         if(x > this->thesolver->leavetol() || -x > this->thesolver->leavetol())
         {
            MSG_ERROR(std::cerr << "ESTEEP03 x[" << i << "] = " << x << std::endl;)
         }
      }
   }

   if(this->thesolver != 0 && this->thesolver->type() == SPxSolverBase<R>::ENTER)
   {
      int i;

      for(i = this->thesolver->dim() - 1; i >= 0; --i)
      {
         if(coW[i] < this->thesolver->epsilon())
            return MSGinconsistent("SPxSteepPR");
      }

      for(i = this->thesolver->coDim() - 1; i >= 0; --i)
      {
         if(w[i] < this->thesolver->epsilon())
            return MSGinconsistent("SPxSteepPR");
      }
   }

#endif

   return true;
}
} // namespace soplex
