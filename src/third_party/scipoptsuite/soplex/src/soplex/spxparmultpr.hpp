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

namespace soplex
{

template <class R>
void SPxParMultPR<R>::setType(typename SPxSolverBase<R>::Type tp)
{
   if(tp == SPxSolverBase<R>::ENTER)
   {
      used = 0;
      this->thesolver->setPricing(SPxSolverBase<R>::PARTIAL);
   }
   else
   {
      this->thesolver->setPricing(SPxSolverBase<R>::FULL);
   }

   this->thesolver->weights.reDim(0);
   this->thesolver->coWeights.reDim(0);
   this->thesolver->weightsAreSetup = false;

   last = 0;
   min = partialSize / 2;
}

template <class R>
void SPxParMultPR<R>::load(SPxSolverBase<R>* p_solver)
{
   this->thesolver = p_solver;
   multiParts = (this->thesolver->dim() + this->thesolver->coDim()) / partialSize + 1;
   pricSet.reSize(10 * partialSize);
}

template <class R>
SPxId SPxParMultPR<R>::selectEnter()
{
   SPxId id;
   R x;
   int i;
   int best = -1;
   //    const SPxBasisBase<R>::Desc& ds   = this->thesolver->basis().desc();

   assert(this->thesolver != 0);
   int lastlast = -1;

   if(this->thesolver->pricing() == SPxSolverBase<R>::PARTIAL)
   {
      R val;
      R eps = -this->theeps;
      lastlast = last;

      for(i = used - 1; i >= 0; --i)
      {
         int n = this->thesolver->number(pricSet[i].id);

         if(this->thesolver->isId(pricSet[i].id))
         {
            this->thesolver->computePvec(n);
            pricSet[i].test = val = this->thesolver->computeTest(n);
         }
         else
            pricSet[i].test = val = this->thesolver->coTest()[n];

         if(val >= eps)
            pricSet[i] = pricSet[--used];
      }

      while(pricSet.size() - used < partialSize)
      {
         best = 0;

         for(i = 1; i < used; ++i)
         {
            if(pricSet[i].test > pricSet[best].test)
               best = i;
         }

         pricSet[best] = pricSet[--used];
      }

      do
      {
         last = (last + 1) % multiParts;

         for(i = this->thesolver->coDim() - last - 1;
               i >= 0; i -= multiParts)
         {
            this->thesolver->computePvec(i);
            x = this->thesolver->computeTest(i);

            if(x < eps)
            {
               pricSet[used].id = this->thesolver->id(i);
               pricSet[used].test = x;
               used++;
            }
         }

         for(i = this->thesolver->dim() - last - 1;
               i >= 0; i -= multiParts)
         {
            x = this->thesolver->coTest()[i];

            if(x < eps)
            {
               pricSet[used].id = this->thesolver->coId(i);
               pricSet[used].test = x;
               used++;
            }
         }

         assert(used < pricSet.size());
      }
      while(used < min && last != lastlast);

      if(used > 0)
      {
         min = (used + 1);

         if(min < 1)
            min = 1;

         if(min > partialSize)
            min = partialSize;

         best = 0;

         for(i = 1; i < used; ++i)
         {
            if(pricSet[i].test < pricSet[best].test)
               best = i;
         }

         id = pricSet[best].id;
      }

      return id;
   }

   else
   {
      assert(this->thesolver->pricing() == SPxSolverBase<R>::FULL);
      R bestx = -this->theeps;

      for(i = this->thesolver->dim() - 1; i >= 0; --i)
      {
         x = this->thesolver->coTest()[i];

         // x *= EQ_PREF * (1 + (ds.coStatus(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.coStatus(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < bestx)
         {
            id = this->thesolver->coId(i);
            bestx = this->thesolver->coTest()[i];
         }
      }

      for(i = this->thesolver->coDim() - 1; i >= 0; --i)
      {
         x = this->thesolver->test()[i];

         // x *= EQ_PREF * (1 + (ds.status(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.status(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < bestx)
         {
            id = this->thesolver->id(i);
            bestx = this->thesolver->test()[i];
         }
      }

      return id;
   }
}

template <class R>
int SPxParMultPR<R>::selectLeave()
{
   int i, n;
   R x;
   R best = -this->theeps;
   //    const R* up  = this->thesolver->ubBound();
   //    const R* low = this->thesolver->lbBound();

   assert(this->thesolver != 0);
   n = -1;

   for(i = this->thesolver->dim() - 1; i >= 0; --i)
   {
      x = this->thesolver->fTest()[i];

      // x *= EQ_PREF * (1 + (up[i] == low[i]));
      if(x < best)
      {
         n = i;
         best = this->thesolver->fTest()[i];
      }
   }

   return n;
}
} // namespace soplex
