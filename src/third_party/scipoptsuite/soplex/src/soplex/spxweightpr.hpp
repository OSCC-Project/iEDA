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

#include "soplex/spxdefines.h"
#include "soplex/spxweightpr.h"
#include "soplex/exceptions.h"

namespace soplex
{
template <class R>
void SPxWeightPR<R>::computeLeavePenalty(int start, int end);

template <class R>
bool SPxWeightPR<R>::isConsistent() const;

template <class R>
void SPxWeightPR<R>::setRep(typename SPxSolverBase<R>::Representation rep)
{
   if(rep == SPxSolverBase<R>::ROW)
   {
      penalty = rPenalty.get_const_ptr();
      coPenalty = cPenalty.get_const_ptr();
   }
   else
   {
      penalty = cPenalty.get_const_ptr();
      coPenalty = rPenalty.get_const_ptr();
   }
}

template <class R>
void SPxWeightPR<R>::setType(typename SPxSolverBase<R>::Type tp)
{
   if(this->thesolver && tp == SPxSolverBase<R>::LEAVE)
   {
      leavePenalty.reDim(this->thesolver->dim());
      computeLeavePenalty(0, this->thesolver->dim());
   }
}

template <class R>
void SPxWeightPR<R>::computeLeavePenalty(int start, int end)
{
   const SPxBasisBase<R>& basis = this->solver()->basis();

   for(int i = start; i < end; ++i)
   {
      SPxId id = basis.baseId(i);

      if(id.type() == SPxId::ROW_ID)
         leavePenalty[i] = rPenalty[ this->thesolver->number(id) ];
      else
         leavePenalty[i] = cPenalty[ this->thesolver->number(id) ];
   }
}

template <class R>
void SPxWeightPR<R>::computeRP(int start, int end)
{
   for(int i = start; i < end; ++i)
   {
      /**@todo TK04NOV98 here is a bug.
       *       this->solver()->rowVector(i).length() could be zero, so
       *       this->solver()->rowVector(i).length2() is also zero and we
       *       get an arithmetic exception.
       */
      assert(this->solver()->rowVector(i).length() > 0);

      rPenalty[i] = (this->solver()->rowVector(i) * this->solver()->maxObj()) * objlength
                    / this->solver()->rowVector(i).length2();
      ASSERT_WARN("WWGTPR01", rPenalty[i] > -1 - this->solver()->epsilon());
   }
}

template <class R>
void SPxWeightPR<R>::computeCP(int start, int end)
{
   for(int i = start; i < end; ++i)
   {
      cPenalty[i] = this->solver()->maxObj(i) * objlength;
      ASSERT_WARN("WWGTPR02", cPenalty[i] > -1 - this->solver()->epsilon());
   }
}

template <class R>
void SPxWeightPR<R>::load(SPxSolverBase<R>* base)
{
   this->thesolver = base;

   rPenalty.reDim(base->nRows());
   cPenalty.reDim(base->nCols());

   objlength = 1 / this->solver()->maxObj().length();
   computeCP(0, base->nCols());
   computeRP(0, base->nRows());
}

template <class R>
int SPxWeightPR<R>::selectLeave()
{
   const R* test = this->thesolver->fTest().get_const_ptr();
   R type = 1 - 2 * (this->thesolver->rep() == SPxSolverBase<R>::COLUMN ? 1 : 0);
   R best = type * R(infinity);
   int lastIdx = -1;
   R x;
   int i;

   for(i = this->solver()->dim() - 1; i >= 0; --i)
   {
      x = test[i];

      if(x < -this->theeps)
      {
         x *= leavePenalty[i];

         if(type * (x - best) < 0.0)
         {
            best = x;
            lastIdx = i;
         }
      }
   }

   assert(isConsistent());
   return lastIdx;
}

template <class R>
SPxId SPxWeightPR<R>::selectEnter()
{
   const VectorBase<R>& rTest = (this->solver()->rep() == SPxSolverBase<R>::ROW)
                                ? this->solver()->test() : this->solver()->coTest();
   const VectorBase<R>& cTest = (this->solver()->rep() == SPxSolverBase<R>::ROW)
                                ? this->solver()->coTest() : this->solver()->test();
   const typename SPxBasisBase<R>::Desc& ds = this->solver()->basis().desc();
   R best = R(infinity);
   SPxId lastId;
   R x;
   int i;

   for(i = this->solver()->nRows() - 1; i >= 0; --i)
   {
      x = rTest[i];

      if(x < -this->theeps)
      {
         x *= -x;

         switch(ds.rowStatus(i))
         {
         case SPxBasisBase<R>::Desc::P_ON_LOWER :
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
            x *= 1 + rPenalty[i];
            break;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
            x *= 1 - rPenalty[i];
            break;

         case SPxBasisBase<R>::Desc::P_FREE :
         case SPxBasisBase<R>::Desc::D_FREE :
            return SPxId(this->solver()->rId(i));

         case SPxBasisBase<R>::Desc::D_ON_BOTH :
            if(this->solver()->pVec()[i] > this->solver()->upBound()[i])
               x *= 1 + rPenalty[i];
            else
               x *= 1 - rPenalty[i];

            break;

         case SPxBasisBase<R>::Desc::D_UNDEFINED :
         case SPxBasisBase<R>::Desc::P_FIXED :
         default:
            throw SPxInternalCodeException("XWGTPR01 This should never happen.");
         }

         if(x < best)
         {
            best = x;
            lastId = this->solver()->rId(i);
         }
      }
   }

   for(i = this->solver()->nCols() - 1; i >= 0; --i)
   {
      x = cTest[i];

      if(x < -this->theeps)
      {
         x *= -x;

         switch(ds.colStatus(i))
         {
         case SPxBasisBase<R>::Desc::P_ON_LOWER :
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
            x *= 1 + cPenalty[i];
            break;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
            x *= 1 - cPenalty[i];
            break;

         case SPxBasisBase<R>::Desc::P_FREE :
         case SPxBasisBase<R>::Desc::D_FREE :
            return SPxId(this->solver()->cId(i));

         case SPxBasisBase<R>::Desc::D_ON_BOTH :
            if(this->solver()->coPvec()[i] > this->solver()->ucBound()[i])
               x *= 1 + cPenalty[i];
            else
               x *= 1 - cPenalty[i];

            break;

         case SPxBasisBase<R>::Desc::P_FIXED :
         case SPxBasisBase<R>::Desc::D_UNDEFINED :
         default:
            throw SPxInternalCodeException("XWGTPR02 This should never happen.");
         }

         if(x < best)
         {
            best = x;
            lastId = this->solver()->cId(i);
         }
      }
   }

   assert(isConsistent());
   return lastId;
}

template <class R>
void SPxWeightPR<R>::addedVecs(int)
{
   if(this->solver()->rep() == SPxSolverBase<R>::ROW)
   {
      int start = rPenalty.dim();
      rPenalty.reDim(this->solver()->nRows());
      computeRP(start, this->solver()->nRows());
   }
   else
   {
      int start = cPenalty.dim();
      cPenalty.reDim(this->solver()->nCols());
      computeCP(start, this->solver()->nCols());
   }

   if(this->solver()->type() == SPxSolverBase<R>::LEAVE)
   {
      int start = leavePenalty.dim();
      leavePenalty.reDim(this->solver()->dim());
      computeLeavePenalty(start, this->solver()->dim());
   }
}

template <class R>
void SPxWeightPR<R>::addedCoVecs(int)
{
   if(this->solver()->rep() == SPxSolverBase<R>::COLUMN)
   {
      int start = rPenalty.dim();
      rPenalty.reDim(this->solver()->nRows());
      computeRP(start, this->solver()->nRows());
   }
   else
   {
      int start = cPenalty.dim();
      cPenalty.reDim(this->solver()->nCols());
      computeCP(start, this->solver()->nCols());
   }

   if(this->solver()->type() == SPxSolverBase<R>::LEAVE)
   {
      int start = leavePenalty.dim();
      leavePenalty.reDim(this->solver()->dim());
      computeLeavePenalty(start, this->solver()->dim());
   }
}

template <class R>
void SPxWeightPR<R>::removedVec(int i)
{
   assert(this->solver() != 0);

   if(this->solver()->rep() == SPxSolverBase<R>::ROW)
   {
      rPenalty[i] = rPenalty[rPenalty.dim()];
      rPenalty.reDim(this->solver()->nRows());
   }
   else
   {
      cPenalty[i] = cPenalty[cPenalty.dim()];
      cPenalty.reDim(this->solver()->nCols());
   }
}

template <class R>
void SPxWeightPR<R>::removedVecs(const int perm[])
{
   assert(this->solver() != 0);

   if(this->solver()->rep() == SPxSolverBase<R>::ROW)
   {
      int j = rPenalty.dim();

      for(int i = 0; i < j; ++i)
      {
         if(perm[i] >= 0)
            rPenalty[perm[i]] = rPenalty[i];
      }

      rPenalty.reDim(this->solver()->nRows());
   }
   else
   {
      int j = cPenalty.dim();

      for(int i = 0; i < j; ++i)
      {
         if(perm[i] >= 0)
            cPenalty[perm[i]] = cPenalty[i];
      }

      cPenalty.reDim(this->solver()->nCols());
   }
}

template <class R>
void SPxWeightPR<R>::removedCoVec(int i)
{
   assert(this->solver() != 0);

   if(this->solver()->rep() == SPxSolverBase<R>::COLUMN)
   {
      rPenalty[i] = rPenalty[rPenalty.dim()];
      rPenalty.reDim(this->solver()->nRows());
   }
   else
   {
      cPenalty[i] = cPenalty[cPenalty.dim()];
      cPenalty.reDim(this->solver()->nCols());
   }
}

template <class R>
void SPxWeightPR<R>::removedCoVecs(const int perm[])
{
   assert(this->solver() != 0);

   if(this->solver()->rep() == SPxSolverBase<R>::COLUMN)
   {
      int j = rPenalty.dim();

      for(int i = 0; i < j; ++i)
      {
         if(perm[i] >= 0)
            rPenalty[perm[i]] = rPenalty[i];
      }

      rPenalty.reDim(this->solver()->nRows());
   }
   else
   {
      int j = cPenalty.dim();

      for(int i = 0; i < j; ++i)
      {
         if(perm[i] >= 0)
            cPenalty[perm[i]] = cPenalty[i];
      }

      cPenalty.reDim(this->solver()->nCols());
   }
}

template <class R>
bool SPxWeightPR<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS

   if(this->solver() != 0)
   {
      if(rPenalty.dim() != this->solver()->nRows())
         return MSGinconsistent("SPxWeightPR");

      if(cPenalty.dim() != this->solver()->nCols())
         return MSGinconsistent("SPxWeightPR");
   }

#endif

   return true;
}
} // namespace soplex
