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

// #define EQ_PREF 1000

#include "soplex/spxdefines.h"

namespace soplex
{

template <class R>
int SPxDantzigPR<R>::selectLeave()
{
   assert(this->thesolver != 0);

   if(this->thesolver->sparsePricingLeave)
      return selectLeaveSparse();

   //    const R* up  = this->thesolver->ubBound();
   //    const R* low = this->thesolver->lbBound();

   R best = -this->theeps;
   int  n    = -1;

   for(int i = this->thesolver->dim() - 1; i >= 0; --i)
   {
      R x = this->thesolver->fTest()[i];

      if(x < -this->theeps)
      {
         // x *= EQ_PREF * (1 + (up[i] == low[i]));
         if(x < best)
         {
            n    = i;
            best = x;
         }
      }
   }

   return n;
}

template <class R>
int SPxDantzigPR<R>::selectLeaveSparse()
{
   assert(this->thesolver != 0);

   R best   = -this->theeps;
   R x;
   int  n      = -1;
   int  index;

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      index = this->thesolver->infeasibilities.index(i);
      x = this->thesolver->fTest()[index];

      if(x < -this->theeps)
      {
         if(x < best)
         {
            n    = index;
            best = x;
         }
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);
         assert(this->thesolver->isInfeasible[index] > 0);
         this->thesolver->isInfeasible[index] = 0;
      }
   }

   return n;
}

template <class R>
SPxId SPxDantzigPR<R>::selectEnter()
{
   assert(this->thesolver != 0);

   // const SPxBasisBase<R>::Desc&    ds   = this->thesolver->basis().desc();

   SPxId enterId;
   enterId = selectEnterX();

   return enterId;
}

template <class R>
SPxId SPxDantzigPR<R>::selectEnterX()
{
   SPxId enterId;
   SPxId enterIdCo;
   R best;
   R bestCo;

   best = -this->theeps;
   bestCo = -this->theeps;
   enterId = (this->thesolver->sparsePricingEnter) ? selectEnterSparseDim(best,
             enterId) : selectEnterDenseDim(best, enterId);
   enterIdCo = (this->thesolver->sparsePricingEnterCo) ? selectEnterSparseCoDim(bestCo,
               enterId) : selectEnterDenseCoDim(bestCo, enterId);

   // prefer slack indices to reduce nonzeros in basis matrix
   if(enterId.isValid() && (best > SPARSITY_TRADEOFF * bestCo || !enterIdCo.isValid()))
      return enterId;
   else
      return enterIdCo;
}


template <class R>
SPxId SPxDantzigPR<R>::selectEnterSparseDim(R& best, SPxId& enterId)
{
   assert(this->thesolver != 0);

   int idx;
   R x;

   for(int i = this->thesolver->infeasibilities.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilities.index(i);
      x = this->thesolver->coTest()[idx];

      if(x < -this->theeps)
      {
         // x *= EQ_PREF * (1 + (ds.coStatus(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.coStatus(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < best)
         {
            enterId = this->thesolver->coId(idx);
            best = x;
         }
      }
      else
      {
         this->thesolver->infeasibilities.remove(i);

         assert(this->thesolver->isInfeasible[idx]);
         this->thesolver->isInfeasible[idx] = 0;
      }
   }

   return enterId;
}

template <class R>
SPxId SPxDantzigPR<R>::selectEnterSparseCoDim(R& best, SPxId& enterId)
{
   assert(this->thesolver != 0);

   int idx;
   R x;

   for(int i = this->thesolver->infeasibilitiesCo.size() - 1; i >= 0; --i)
   {
      idx = this->thesolver->infeasibilitiesCo.index(i);
      x = this->thesolver->test()[idx];

      if(x < -this->theeps)
      {
         // x *= EQ_PREF * (1 + (ds.coStatus(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.coStatus(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < best)
         {
            enterId = this->thesolver->id(idx);
            best = x;
         }
      }
      else
      {
         this->thesolver->infeasibilitiesCo.remove(i);
         assert(this->thesolver->isInfeasibleCo[idx] > 0);
         this->thesolver->isInfeasibleCo[idx] = 0;
      }
   }

   return enterId;
}

template <class R>
SPxId SPxDantzigPR<R>::selectEnterDenseDim(R& best, SPxId& enterId)
{
   assert(this->thesolver != 0);

   R x;

   for(int i = this->thesolver->dim() - 1; i >= 0; --i)
   {
      x = this->thesolver->coTest()[i];

      if(x < -this->theeps)
      {
         // x *= EQ_PREF * (1 + (ds.coStatus(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.coStatus(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < best)
         {
            enterId   = this->thesolver->coId(i);
            best = x;
         }
      }
   }

   return enterId;
}

template <class R>
SPxId SPxDantzigPR<R>::selectEnterDenseCoDim(R& best, SPxId& enterId)
{
   assert(this->thesolver != 0);

   R x;

   for(int i = this->thesolver->coDim() - 1; i >= 0; --i)
   {
      x = this->thesolver->test()[i];

      if(x < -this->theeps)
      {
         // x *= EQ_PREF * (1 + (ds.status(i) == SPxBasisBase<R>::Desc::P_FREE
         //                || ds.status(i) == SPxBasisBase<R>::Desc::D_FREE));
         if(x < best)
         {
            enterId   = this->thesolver->id(i);
            best = x;
         }
      }
   }

   return enterId;
}

} // namespace soplex
