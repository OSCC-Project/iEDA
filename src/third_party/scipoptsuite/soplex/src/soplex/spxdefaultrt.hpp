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
/**
 * Here comes the ratio test for selecting a variable to leave the basis.
 * It is assumed that Vec.delta() and fVec.idx() have been setup
 * correctly!
 *
 * The leaving variable is selected such that the update of fVec() (using
 * fVec.value() * fVec.delta()) keeps the basis feasible within
 * solver()->entertol(). Hence, fVec.value() must be chosen such that one
 * updated value of theFvec just reaches its bound and no other one exceeds
 * them by more than solver()->entertol(). Further, fVec.value() must have the
 * same sign as argument \p val.
 *
 * The return value of selectLeave() is the number of a variable in the
 * basis selected to leave the basis. -1 indicates that no variable could be
 * selected. Otherwise, parameter \p val contains the chosen fVec.value().
 */
template <class R>
int SPxDefaultRT<R>::selectLeave(R& val, R, bool)
{
   this->solver()->fVec().delta().setup();

   const R*   vec = this->solver()->fVec().get_const_ptr();
   const R*   upd = this->solver()->fVec().delta().values();
   const IdxSet& idx = this->solver()->fVec().idx();
   const R*   ub  = this->solver()->ubBound().get_const_ptr();
   const R*   lb  = this->solver()->lbBound().get_const_ptr();

   R epsilon = this->solver()->epsilon();
   int  leave   = -1;

   R x;
   int  i;
   int  j;

   // PARALLEL the j loop could be parallelized
   if(val > 0)
   {
      // Loop over NZEs of delta vector.
      for(j = 0; j < idx.size(); ++j)
      {
         i = idx.index(j);
         x = upd[i];

         if(x > epsilon)
         {
            if(ub[i] < R(infinity))
            {
               R y = (ub[i] - vec[i] + this->delta) / x;

               if(y < val)
               {
                  leave = i;
                  val   = y;
               }
            }
         }
         else if(x < -epsilon)
         {
            if(lb[i] > R(-infinity))
            {
               R y = (lb[i] - vec[i] - this->delta) / x;

               if(y < val)
               {
                  leave = i;
                  val   = y;
               }
            }
         }
      }

      if(leave >= 0)
      {
         x   = upd[leave];

         // BH 2005-11-30: It may well happen that the basis is degenerate and the
         // selected leaving variable is (at most this->delta) beyond its bound. (This
         // happens for instance on LP/netlib/adlittle.mps with setting -r -t0.)
         // In that case we do a pivot step with length zero to avoid difficulties.
         if((x > epsilon  && vec[leave] >= ub[leave]) ||
               (x < -epsilon && vec[leave] <= lb[leave]))
         {
            val = 0.0;
         }
         else
         {
            val = (x > epsilon) ? ub[leave] : lb[leave];
            val = (val - vec[leave]) / x;
         }
      }

      ASSERT_WARN("WDEFRT01", val > -epsilon);
   }
   else
   {
      for(j = 0; j < idx.size(); ++j)
      {
         i = idx.index(j);
         x = upd[i];

         if(x < -epsilon)
         {
            if(ub[i] < R(infinity))
            {
               R y = (ub[i] - vec[i] + this->delta) / x;

               if(y > val)
               {
                  leave = i;
                  val   = y;
               }
            }
         }
         else if(x > epsilon)
         {
            if(lb[i] > R(-infinity))
            {
               R y = (lb[i] - vec[i] - this->delta) / x;

               if(y > val)
               {
                  leave = i;
                  val   = y;
               }
            }
         }
      }

      if(leave >= 0)
      {
         x   = upd[leave];

         // See comment above.
         if((x < -epsilon && vec[leave] >= ub[leave]) ||
               (x > epsilon  && vec[leave] <= lb[leave]))
         {
            val = 0.0;
         }
         else
         {
            val = (x < epsilon) ? ub[leave] : lb[leave];
            val = (val - vec[leave]) / x;
         }
      }

      ASSERT_WARN("WDEFRT02", val < epsilon);
   }

   return leave;
}

/**
   Here comes the ratio test. It is assumed that theCoPvec.this->delta() and
   theCoPvec.idx() have been setup correctly!
*/
template <class R>
SPxId SPxDefaultRT<R>::selectEnter(R& max, int, bool)
{
   this->solver()->coPvec().delta().setup();
   this->solver()->pVec().delta().setup();

   const R*   pvec = this->solver()->pVec().get_const_ptr();
   const R*   pupd = this->solver()->pVec().delta().values();
   const IdxSet& pidx = this->solver()->pVec().idx();
   const R*   lpb  = this->solver()->lpBound().get_const_ptr();
   const R*   upb  = this->solver()->upBound().get_const_ptr();

   const R*   cvec = this->solver()->coPvec().get_const_ptr();
   const R*   cupd = this->solver()->coPvec().delta().values();
   const IdxSet& cidx = this->solver()->coPvec().idx();
   const R*   lcb  = this->solver()->lcBound().get_const_ptr();
   const R*   ucb  = this->solver()->ucBound().get_const_ptr();

   R epsilon = this->solver()->epsilon();
   R val     = max;
   int  pnum    = -1;
   int  cnum    = -1;

   SPxId enterId;
   int   i;
   int   j;
   R  x;

   // PARALLEL the j loops could be parallelized
   if(val > 0)
   {
      for(j = 0; j < pidx.size(); ++j)
      {
         i = pidx.index(j);
         x = pupd[i];

         if(x > epsilon)
         {
            if(upb[i] < R(infinity))
            {
               R y = (upb[i] - pvec[i] + this->delta) / x;

               if(y < val)
               {
                  enterId = this->solver()->id(i);
                  val     = y;
                  pnum    = j;
               }
            }
         }
         else if(x < -epsilon)
         {
            if(lpb[i] > R(-infinity))
            {
               R y = (lpb[i] - pvec[i] - this->delta) / x;

               if(y < val)
               {
                  enterId = this->solver()->id(i);
                  val     = y;
                  pnum    = j;
               }
            }
         }
      }

      for(j = 0; j < cidx.size(); ++j)
      {
         i = cidx.index(j);
         x = cupd[i];

         if(x > epsilon)
         {
            if(ucb[i] < R(infinity))
            {
               R y = (ucb[i] - cvec[i] + this->delta) / x;

               if(y < val)
               {
                  enterId = this->solver()->coId(i);
                  val     = y;
                  cnum    = j;
               }
            }
         }
         else if(x < -epsilon)
         {
            if(lcb[i] > R(-infinity))
            {
               R y = (lcb[i] - cvec[i] - this->delta) / x;

               if(y < val)
               {
                  enterId = this->solver()->coId(i);
                  val     = y;
                  cnum    = j;
               }
            }
         }
      }

      if(cnum >= 0)
      {
         i   = cidx.index(cnum);
         x   = cupd[i];
         val = (x > epsilon) ? ucb[i] : lcb[i];
         val = (val - cvec[i]) / x;
      }
      else if(pnum >= 0)
      {
         i   = pidx.index(pnum);
         x   = pupd[i];
         val = (x > epsilon) ? upb[i] : lpb[i];
         val = (val - pvec[i]) / x;
      }
   }
   else
   {
      for(j = 0; j < pidx.size(); ++j)
      {
         i = pidx.index(j);
         x = pupd[i];

         if(x > epsilon)
         {
            if(lpb[i] > R(-infinity))
            {
               R y = (lpb[i] - pvec[i] - this->delta) / x;

               if(y > val)
               {
                  enterId = this->solver()->id(i);
                  val     = y;
                  pnum    = j;
               }
            }
         }
         else if(x < -epsilon)
         {
            if(upb[i] < R(infinity))
            {
               R y = (upb[i] - pvec[i] + this->delta) / x;

               if(y > val)
               {
                  enterId = this->solver()->id(i);
                  val     = y;
                  pnum    = j;
               }
            }
         }
      }

      for(j = 0; j < cidx.size(); ++j)
      {
         i = cidx.index(j);
         x = cupd[i];

         if(x > epsilon)
         {
            if(lcb[i] > R(-infinity))
            {
               R y = (lcb[i] - cvec[i] - this->delta) / x;

               if(y > val)
               {
                  enterId = this->solver()->coId(i);
                  val     = y;
                  cnum    = j;
               }
            }
         }
         else if(x < -epsilon)
         {
            if(ucb[i] < R(infinity))
            {
               R y = (ucb[i] - cvec[i] + this->delta) / x;

               if(y > val)
               {
                  enterId = this->solver()->coId(i);
                  val     = y;
                  cnum    = j;
               }
            }
         }
      }

      if(cnum >= 0)
      {
         i   = cidx.index(cnum);
         x   = cupd[i];
         val = (x < epsilon) ? ucb[i] : lcb[i];
         val = (val - cvec[i]) / x;
      }
      else if(pnum >= 0)
      {
         i   = pidx.index(pnum);
         x   = pupd[i];
         val = (x < epsilon) ? upb[i] : lpb[i];
         val = (val - pvec[i]) / x;
      }
   }

   if(enterId.isValid() && this->solver()->isBasic(enterId))
   {
      MSG_DEBUG(std::cout << "DDEFRT01 isValid() && isBasic(): max=" << max
                << std::endl;)

      if(cnum >= 0)
         this->solver()->coPvec().delta().clearNum(cnum);
      else if(pnum >= 0)
         this->solver()->pVec().delta().clearNum(pnum);

      return SPxDefaultRT<R>::selectEnter(max, 0, false);
   }

   MSG_DEBUG(

      if(!enterId.isValid())
      std::cout << "DDEFRT02 !isValid(): max=" << max << ", x=" << x << std::endl;
   )
      max = val;

   return enterId;
}

} // namespace soplex
