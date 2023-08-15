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

#include "soplex/spxdefines.h"

#define DEVEX_REFINETOL 2.0

namespace soplex
{
// Definition of signature to avoid the specialization after instantiation error

template <class R>
void SPxDevexPR<R>::load(SPxSolverBase<R>* base)
{
   this->thesolver = base;
   setRep(base->rep());
   assert(isConsistent());
}

template <class R>
bool SPxDevexPR<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS

   if(this->thesolver != 0)
      if(this->thesolver->weights.dim() != this->thesolver->coDim()
            || this->thesolver->coWeights.dim() != this->thesolver->dim())
         return MSGinconsistent("SPxDevexPR");

#endif

   return true;
}

template <class R>
void SPxDevexPR<R>::setupWeights(typename SPxSolverBase<R>::Type tp)
{
   int i;
   int coWeightSize = 0;
   int weightSize = 0;

   VectorBase<R>& weights = this->thesolver->weights;
   VectorBase<R>& coWeights = this->thesolver->coWeights;

   if(tp == SPxSolverBase<R>::ENTER)
   {
      coWeights.reDim(this->thesolver->dim(), false);

      for(i = this->thesolver->dim() - 1; i >= coWeightSize; --i)
         coWeights[i] = 2.0;

      weights.reDim(this->thesolver->coDim(), false);

      for(i = this->thesolver->coDim() - 1; i >= weightSize; --i)
         weights[i] = 2.0;
   }
   else
   {
      coWeights.reDim(this->thesolver->dim(), false);

      for(i = this->thesolver->dim() - 1; i >= coWeightSize; --i)
         coWeights[i] = 1.0;
   }

   this->thesolver->weightsAreSetup = true;
}

template <class R>
void SPxDevexPR<R>::setType(typename SPxSolverBase<R>::Type tp)
{
   setupWeights(tp);
   refined = false;

   bestPrices.clear();
   bestPrices.setMax(this->thesolver->dim());
   prices.reSize(this->thesolver->dim());

   if(tp == SPxSolverBase<R>::ENTER)
   {
      bestPricesCo.clear();
      bestPricesCo.setMax(this->thesolver->coDim());
      pricesCo.reSize(this->thesolver->coDim());
   }

   assert(isConsistent());
}

/**@todo suspicious: Shouldn't the relation between dim, coDim, Vecs,
 *       and CoVecs be influenced by the representation ?
 */
template <class R>
void SPxDevexPR<R>::setRep(typename SPxSolverBase<R>::Representation)
{
   if(this->thesolver != 0)
   {
      // resize weights and initialize new entries
      addedVecs(this->thesolver->coDim());
      addedCoVecs(this->thesolver->dim());
      assert(isConsistent());
   }
}

// A namespace to avoid collision with other pricers
namespace devexpr
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
int SPxDevexPR<R>::buildBestPriceVectorLeave(R feastol)
{
   int idx;
   int nsorted;
   R fTesti;
   const R* fTest = this->thesolver->fTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   typename SPxPricer<R>::IdxElement price;
   prices.clear();
   bestPrices.clear();

   // TODO we should check infeasiblities for duplicates or loop over dimension
   //      bestPrices may then also contain duplicates!
   // construct vector of all prices
   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      fTesti = fTest[idx];

      if(fTesti < -feastol)
      {
         this->thesolver->isInfeasible[idx] = this->VIOLATED;
         price.idx = idx;
         price.val = devexpr::computePrice(fTesti, cpen[idx], feastol);
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
int SPxDevexPR<R>::selectLeave()
{
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
                (*this->thesolver->spxout) << "WDEVEX02 trying refinement step..\n";)
      retid = selectLeaveX(this->theeps / DEVEX_REFINETOL);
   }

   assert(retid < this->thesolver->dim());

   return retid;
}

template <class R>
int SPxDevexPR<R>::selectLeaveX(R feastol, int start, int incr)
{
   R x;

   const R* fTest = this->thesolver->fTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   R best = 0;
   int bstI = -1;
   int end = this->thesolver->coWeights.dim();

   for(; start < end; start += incr)
   {
      if(fTest[start] < -feastol)
      {
         x = devexpr::computePrice(fTest[start], cpen[start], feastol);

         if(x > best)
         {
            best = x;
            bstI = start;
            last = cpen[start];
         }
      }
   }

   return bstI;
}

template <class R>
int SPxDevexPR<R>::selectLeaveSparse(R feastol)
{
   R x;

   const R* fTest = this->thesolver->fTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   R best = 0;
   int bstI = -1;
   int idx = -1;

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = fTest[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, cpen[idx], feastol);

         if(x > best)
         {
            best = x;
            bstI = idx;
            last = cpen[idx];
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

   return bstI;
}

template <class R>
int SPxDevexPR<R>::selectLeaveHyper(R feastol)
{
   R x;

   const R* fTest = this->thesolver->fTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   R best = 0;
   R leastBest = -1;
   int bstI = -1;
   int idx = -1;

   // find the best price from the short candidate list
   for(int i = bestPrices.size() - 1; i >= 0; --i)
   {
      idx = bestPrices.index(i);
      x = fTest[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, cpen[idx], feastol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            bstI = idx;
            last = cpen[idx];
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

      // only look at indeces that were not checked already
      if(this->thesolver->isInfeasible[idx] == this->VIOLATED)
      {
         x = fTest[idx];
         assert(x < -feastol);
         x = devexpr::computePrice(x, cpen[idx], feastol);

         if(x > leastBest)
         {
            if(x > best)
            {
               best = x;
               bstI = idx;
               last = cpen[idx];
            }

            // put index into candidate list
            this->thesolver->isInfeasible[idx] = this->VIOLATED_AND_CHECKED;
            bestPrices.addIdx(idx);
         }
      }
   }

   return bstI;
}

template <class R>
void SPxDevexPR<R>::left4(int n, SPxId id)
{
   VectorBase<R>& coWeights = this->thesolver->coWeights;

   if(id.isValid())
   {
      int i, j;
      R x;
      const R* rhoVec = this->thesolver->fVec().delta().values();
      R rhov_1 = 1 / rhoVec[n];
      R beta_q = this->thesolver->coPvec().delta().length2() * rhov_1 * rhov_1;

#ifndef NDEBUG

      if(spxAbs(rhoVec[n]) < this->theeps)
      {
         MSG_INFO3((*this->thesolver->spxout), (*this->thesolver->spxout) << "WDEVEX01: rhoVec = "
                   << rhoVec[n] << " with smaller absolute value than this->theeps = " << this->theeps << std::endl;)
      }

#endif  // NDEBUG

      //  Update #coPenalty# vector
      const IdxSet& rhoIdx = this->thesolver->fVec().idx();
      int len = this->thesolver->fVec().idx().size();

      for(i = len - 1; i >= 0; --i)
      {
         j = rhoIdx.index(i);
         x = rhoVec[j] * rhoVec[j] * beta_q;
         // if(x > coPenalty[j])
         coWeights[j] += x;
      }

      coWeights[n] = beta_q;
   }
}

template <class R>
SPxId SPxDevexPR<R>::buildBestPriceVectorEnterDim(R& best, R feastol)
{
   int idx;
   int nsorted;
   R x;
   const R* coTest = this->thesolver->coTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
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
         this->thesolver->isInfeasible[idx] = this->VIOLATED;
         price.idx = idx;
         price.val = devexpr::computePrice(x, cpen[idx], feastol);
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
SPxId SPxDevexPR<R>::buildBestPriceVectorEnterCoDim(R& best, R feastol)
{
   int idx;
   int nsorted;
   R x;
   const R* test = this->thesolver->test().get_const_ptr();
   const R* pen = this->thesolver->weights.get_const_ptr();
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
         this->thesolver->isInfeasibleCo[idx] = this->VIOLATED;
         price.idx = idx;
         price.val = devexpr::computePrice(x, pen[idx], feastol);
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
SPxId SPxDevexPR<R>::selectEnter()
{
   assert(this->thesolver != 0);

   SPxId enterId;

   enterId = selectEnterX(this->theeps);

   if(enterId.isSPxColId() && this->thesolver->isBasic(SPxColId(enterId)))
      enterId.info = 0;

   if(enterId.isSPxRowId() && this->thesolver->isBasic(SPxRowId(enterId)))
      enterId.info = 0;

   if(!enterId.isValid() && !refined)
   {
      refined = true;
      MSG_INFO3((*this->thesolver->spxout),
                (*this->thesolver->spxout) << "WDEVEX02 trying refinement step..\n";)
      enterId = selectEnterX(this->theeps / DEVEX_REFINETOL);

      if(enterId.isSPxColId() && this->thesolver->isBasic(SPxColId(enterId)))
         enterId.info = 0;

      if(enterId.isSPxRowId() && this->thesolver->isBasic(SPxRowId(enterId)))
         enterId.info = 0;
   }

   return enterId;
}

// choose the best entering index among columns and rows but prefer sparsity
template <class R>
SPxId SPxDevexPR<R>::selectEnterX(R tol)
{
   SPxId enterId;
   SPxId enterCoId;
   R best;
   R bestCo;

   best = 0;
   bestCo = 0;
   last = 1.0;

   // avoid uninitialized value later on in entered4X()
   last = 1.0;

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

   // prefer coIds to increase the number of unit vectors in the basis matrix, i.e., rows in colrep and cols in rowrep
   if(enterCoId.isValid() && (best > SPARSITY_TRADEOFF * bestCo || !enterId.isValid()))
      return enterCoId;
   else
      return enterId;
}

template <class R>
SPxId SPxDevexPR<R>::selectEnterHyperDim(R& best, R feastol)
{
   const R* cTest = this->thesolver->coTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   R leastBest = -1;
   R x;
   int enterIdx = -1;
   int idx;

   // find the best price from short candidate list
   for(int i = bestPrices.size() - 1; i >= 0; --i)
   {
      idx = bestPrices.index(i);
      x = cTest[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, cpen[idx], feastol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            enterIdx = idx;
            last = cpen[idx];
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

   // scan the updated indeces for a better price
   for(int i = this->thesolver->updateViols.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->updateViols.index(i);

      // only look at indeces that were not checked already
      if(this->thesolver->isInfeasible[idx] == this->VIOLATED)
      {
         x = cTest[idx];

         if(x < -feastol)
         {
            x = devexpr::computePrice(x, cpen[idx], feastol);

            if(x > leastBest)
            {
               if(x > best)
               {
                  best = x;
                  enterIdx = idx;
                  last = cpen[idx];
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
SPxId SPxDevexPR<R>::selectEnterHyperCoDim(R& best, R feastol)
{
   const R* test = this->thesolver->test().get_const_ptr();
   const R* pen = this->thesolver->weights.get_const_ptr();
   R leastBest = -1;
   R x;
   int enterIdx = -1;
   int idx;

   // find the best price from short candidate list
   for(int i = bestPricesCo.size() - 1; i >= 0; --i)
   {
      idx = bestPricesCo.index(i);
      x = test[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, pen[idx], feastol);

         assert(x >= 0);

         // update the best price of candidate list
         if(x > best)
         {
            best = x;
            enterIdx = idx;
            last = pen[idx];
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

   //scan the updated indeces for a better price
   for(int i = this->thesolver->updateViolsCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->updateViolsCo.index(i);

      // only look at indeces that were not checked already
      if(this->thesolver->isInfeasibleCo[idx] == this->VIOLATED)
      {
         x = test[idx];

         if(x < -feastol)
         {
            x = devexpr::computePrice(x, pen[idx], feastol);

            if(x > leastBest)
            {
               if(x > best)
               {
                  best = x;
                  enterIdx = idx;
                  last = pen[idx];
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
SPxId SPxDevexPR<R>::selectEnterSparseDim(R& best, R feastol)
{
   const R* cTest = this->thesolver->coTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   int enterIdx = -1;
   int idx;
   R x;

   assert(this->thesolver->coWeights.dim() == this->thesolver->coTest().dim());

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = cTest[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, cpen[idx], feastol);

         if(x > best)
         {
            best = x;
            enterIdx = idx;
            last = cpen[idx];
         }
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);
         this->thesolver->isInfeasible[idx] = this->NOT_VIOLATED;
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->coId(enterIdx);

   return SPxId();
}


template <class R>
SPxId SPxDevexPR<R>::selectEnterSparseCoDim(R& best, R feastol)
{
   const R* test = this->thesolver->test().get_const_ptr();
   const R* pen = this->thesolver->weights.get_const_ptr();
   int enterIdx = -1;
   int idx;
   R x;

   assert(this->thesolver->weights.dim() == this->thesolver->test().dim());

   for(int i = this->thesolver->infeasibilitiesCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilitiesCo.index(i);
      x = test[idx];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, pen[idx], feastol);

         if(x > best)
         {
            best = x;
            enterIdx = idx;
            last = pen[idx];
         }
      }
      else
      {
         this->thesolver->infeasibilitiesCo.remove(i);
         this->thesolver->isInfeasibleCo[idx] = this->NOT_VIOLATED;
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->id(enterIdx);

   return SPxId();
}


template <class R>
SPxId SPxDevexPR<R>::selectEnterDenseDim(R& best, R feastol, int start, int incr)
{
   const R* cTest = this->thesolver->coTest().get_const_ptr();
   const R* cpen = this->thesolver->coWeights.get_const_ptr();
   int end = this->thesolver->coWeights.dim();
   int enterIdx = -1;
   R x;

   assert(end == this->thesolver->coTest().dim());

   for(; start < end; start += incr)
   {
      x = cTest[start];

      if(x < -feastol)
      {
         x = devexpr::computePrice(x, cpen[start], feastol);

         if(x > best)
         {
            best = x;
            enterIdx = start;
            last = cpen[start];
         }
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->coId(enterIdx);

   return SPxId();
}

template <class R>
SPxId SPxDevexPR<R>::selectEnterDenseCoDim(R& best, R feastol, int start, int incr)
{
   const R* test = this->thesolver->test().get_const_ptr();
   const R* pen = this->thesolver->weights.get_const_ptr();
   int end = this->thesolver->weights.dim();
   int enterIdx = -1;
   R x;

   assert(end == this->thesolver->test().dim());

   for(; start < end; start += incr)
   {
      x = test[start];

      if(test[start] < -feastol)
      {
         x = devexpr::computePrice(x, pen[start], feastol);

         if(x > best)
         {
            best = x;
            enterIdx = start;
            last = pen[start];
         }
      }
   }

   if(enterIdx >= 0)
      return this->thesolver->id(enterIdx);

   return SPxId();
}


/**@todo suspicious: the pricer should be informed, that variable id
   has entered the basis at position n, but the id is not used here
   (this is true for all pricers)
*/
template <class R>
void SPxDevexPR<R>::entered4(SPxId /*id*/, int n)
{
   VectorBase<R>& weights = this->thesolver->weights;
   VectorBase<R>& coWeights = this->thesolver->coWeights;

   if(n >= 0 && n < this->thesolver->dim())
   {
      const R* pVec = this->thesolver->pVec().delta().values();
      const IdxSet& pIdx = this->thesolver->pVec().idx();
      const R* coPvec = this->thesolver->coPvec().delta().values();
      const IdxSet& coPidx = this->thesolver->coPvec().idx();
      R xi_p = 1 / this->thesolver->fVec().delta()[n];
      int i, j;

      assert(this->thesolver->fVec().delta()[n] > this->thesolver->epsilon()
             || this->thesolver->fVec().delta()[n] < -this->thesolver->epsilon());

      xi_p = xi_p * xi_p * last;

      for(j = coPidx.size() - 1; j >= 0; --j)
      {
         i = coPidx.index(j);
         coWeights[i] += xi_p * coPvec[i] * coPvec[i];

         if(coWeights[i] <= 1 || coWeights[i] > 1e+6)
         {
            setupWeights(SPxSolverBase<R>::ENTER);
            return;
         }
      }

      for(j = pIdx.size() - 1; j >= 0; --j)
      {
         i = pIdx.index(j);
         weights[i] += xi_p * pVec[i] * pVec[i];

         if(weights[i] <= 1 || weights[i] > 1e+6)
         {
            setupWeights(SPxSolverBase<R>::ENTER);
            return;
         }
      }
   }
}

template <class R>
void SPxDevexPR<R>::addedVecs(int n)
{
   int initval = (this->thesolver->type() == SPxSolverBase<R>::ENTER) ? 2 : 1;
   VectorBase<R>& weights = this->thesolver->weights;
   n = weights.dim();
   weights.reDim(this->thesolver->coDim());

   for(int i = weights.dim() - 1; i >= n; --i)
      weights[i] = initval;
}

template <class R>
void SPxDevexPR<R>::addedCoVecs(int n)
{
   int initval = (this->thesolver->type() == SPxSolverBase<R>::ENTER) ? 2 : 1;
   VectorBase<R>& coWeights = this->thesolver->coWeights;
   n = coWeights.dim();
   coWeights.reDim(this->thesolver->dim());

   for(int i = coWeights.dim() - 1; i >= n; --i)
      coWeights[i] = initval;
}

} // namespace soplex
