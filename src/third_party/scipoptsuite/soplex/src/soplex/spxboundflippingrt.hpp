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
#include "soplex/sorter.h"
#include "soplex/spxsolver.h"
#include "soplex/spxout.h"
#include "soplex/spxid.h"

namespace soplex
{

#define LOWSTAB          1e-10
#define MAX_RELAX_COUNT  2
#define LONGSTEP_FREQ    100


/** perform necessary bound flips to restore dual feasibility */
template <class R>
void SPxBoundFlippingRT<R>::flipAndUpdate(
   int&                  nflips              /**< number of bounds that should be flipped */
)
{
   assert(nflips > 0);

   // number of bound flips that are not performed
   int skipped;

   updPrimRhs.setup();
   updPrimRhs.reDim(this->thesolver->dim());
   updPrimVec.reDim(this->thesolver->dim());
   updPrimRhs.clear();
   updPrimVec.clear();

   skipped = 0;

   for(int i = 0; i < nflips; ++i)
   {
      int idx;
      idx = breakpoints[i].idx;

      if(idx < 0)
      {
         ++skipped;
         continue;
      }

      R range;
      R upper;
      R lower;
      R objChange = 0.0;
      typename SPxBasisBase<R>::Desc::Status stat;
      typename SPxBasisBase<R>::Desc& ds = this->thesolver->basis().desc();

      range = 0;

      if(breakpoints[i].src == PVEC)
      {
         assert(this->thesolver->rep() == SPxSolverBase<R>::COLUMN);
         stat = ds.status(idx);
         upper = this->thesolver->upper(idx);
         lower = this->thesolver->lower(idx);

         switch(stat)
         {
         case SPxBasisBase<R>::Desc::P_ON_UPPER :
            ds.status(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
            range = lower - upper;
            assert((*this->thesolver->theLbound)[idx] == R(-infinity));
            (*this->thesolver->theLbound)[idx] = (*this->thesolver->theUbound)[idx];
            (*this->thesolver->theUbound)[idx] = R(infinity);
            objChange = range * (*this->thesolver->theLbound)[idx];
            break;

         case SPxBasisBase<R>::Desc::P_ON_LOWER :
            ds.status(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
            range = upper - lower;
            assert((*this->thesolver->theUbound)[idx] == R(infinity));
            (*this->thesolver->theUbound)[idx] = (*this->thesolver->theLbound)[idx];
            (*this->thesolver->theLbound)[idx] = R(-infinity);
            objChange = range * (*this->thesolver->theUbound)[idx];
            break;

         default :
            ++skipped;
            MSG_WARNING((*this->thesolver->spxout),
                        (*this->thesolver->spxout) << "PVEC unexpected status: " << static_cast<int>(stat)
                        << " index: " << idx
                        << " val: " << this->thesolver->pVec()[idx]
                        << " upd: " << this->thesolver->pVec().delta()[idx]
                        << " lower: " << lower
                        << " upper: " << upper
                        << " bp.val: " << breakpoints[i].val
                        << std::endl;)
         }

         MSG_DEBUG(std::cout << "PVEC flipped from: " << stat
                   << " index: " << idx
                   << " val: " << this->thesolver->pVec()[idx]
                   << " upd: " << this->thesolver->pVec().delta()[idx]
                   << " lower: " << lower
                   << " upper: " << upper
                   << " bp.val: " << breakpoints[i].val
                   << " UCbound: " << this->thesolver->theUCbound[idx]
                   << " LCbound: " << this->thesolver->theLCbound[idx]
                   << std::endl;)
         assert(spxAbs(range) < 1e20);
         updPrimRhs.multAdd(range, this->thesolver->vector(idx));

         if(objChange != 0.0)
            this->thesolver->updateNonbasicValue(objChange);
      }
      else if(breakpoints[i].src == COPVEC)
      {
         assert(this->thesolver->rep() == SPxSolverBase<R>::COLUMN);
         stat = ds.coStatus(idx);
         upper = this->thesolver->rhs(idx);
         lower = this->thesolver->lhs(idx);

         switch(stat)
         {
         case SPxBasisBase<R>::Desc::P_ON_UPPER :
            ds.coStatus(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
            range = lower - upper;
            assert((*this->thesolver->theCoUbound)[idx] == R(infinity));
            (*this->thesolver->theCoUbound)[idx] = -(*this->thesolver->theCoLbound)[idx];
            (*this->thesolver->theCoLbound)[idx] = R(-infinity);
            objChange = range * (*this->thesolver->theCoUbound)[idx];
            break;

         case SPxBasisBase<R>::Desc::P_ON_LOWER :
            ds.coStatus(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
            range = upper - lower;
            assert((*this->thesolver->theCoLbound)[idx] == R(-infinity));
            (*this->thesolver->theCoLbound)[idx] = -(*this->thesolver->theCoUbound)[idx];
            (*this->thesolver->theCoUbound)[idx] = R(infinity);
            objChange = range * (*this->thesolver->theCoLbound)[idx];
            break;

         default :
            ++skipped;
            MSG_WARNING((*this->thesolver->spxout),
                        (*this->thesolver->spxout) << "COPVEC unexpected status: " << static_cast<int>(stat)
                        << " index: " << idx
                        << " val: " << this->thesolver->coPvec()[idx]
                        << " upd: " << this->thesolver->coPvec().delta()[idx]
                        << " lower: " << lower
                        << " upper: " << upper
                        << " bp.val: " << breakpoints[i].val
                        << std::endl;)
         }

         MSG_DEBUG(std::cout << "COPVEC flipped from: " << stat
                   << " index: " << idx
                   << " val: " << this->thesolver->coPvec()[idx]
                   << " upd: " << this->thesolver->coPvec().delta()[idx]
                   << " lower: " << lower
                   << " upper: " << upper
                   << " bp.val: " << breakpoints[i].val
                   << " URbound: " << this->thesolver->theURbound[idx]
                   << " LRbound: " << this->thesolver->theLRbound[idx]
                   << std::endl;)
         assert(spxAbs(range) < 1e20);
         updPrimRhs.setValue(idx, updPrimRhs[idx] - range);

         if(objChange != 0.0)
            this->thesolver->updateNonbasicValue(objChange);
      }
      else if(breakpoints[i].src == FVEC)
      {
         assert(this->thesolver->rep() == SPxSolverBase<R>::ROW);
         SPxId baseId = this->thesolver->basis().baseId(idx);
         int IdNumber;

         if(baseId.isSPxRowId())
         {
            IdNumber = this->thesolver->number(SPxRowId(baseId));
            stat = ds.rowStatus(IdNumber);
            upper = this->thesolver->rhs(IdNumber);
            lower = this->thesolver->lhs(IdNumber);

            switch(stat)
            {
            case SPxBasisBase<R>::Desc::P_ON_UPPER :
               ds.rowStatus(IdNumber) = SPxBasisBase<R>::Desc::P_ON_LOWER;
               range = upper - lower;
               assert(this->thesolver->theUBbound[idx] == R(infinity));
               this->thesolver->theUBbound[idx] = -this->thesolver->theLBbound[idx];
               this->thesolver->theLBbound[idx] = R(-infinity);
               break;

            case SPxBasisBase<R>::Desc::P_ON_LOWER :
               ds.rowStatus(IdNumber) = SPxBasisBase<R>::Desc::P_ON_UPPER;
               range = lower - upper;
               assert(this->thesolver->theLBbound[idx] == R(-infinity));
               this->thesolver->theLBbound[idx] = -this->thesolver->theUBbound[idx];
               this->thesolver->theUBbound[idx] = R(infinity);
               break;

            default :
               ++skipped;
               MSG_WARNING((*this->thesolver->spxout),
                           (*this->thesolver->spxout) << "unexpected basis status: " << static_cast<int>(stat)
                           << " index: " << idx
                           << " val: " << this->thesolver->fVec()[idx]
                           << " upd: " << this->thesolver->fVec().delta()[idx]
                           << " lower: " << lower
                           << " upper: " << upper
                           << " bp.val: " << breakpoints[i].val
                           << std::endl;)
            }
         }
         else
         {
            assert(baseId.isSPxColId());
            IdNumber = this->thesolver->number(SPxColId(baseId));
            stat = ds.colStatus(IdNumber);
            upper = this->thesolver->upper(IdNumber);
            lower = this->thesolver->lower(IdNumber);

            switch(stat)
            {
            case SPxBasisBase<R>::Desc::P_ON_UPPER :
               ds.colStatus(IdNumber) = SPxBasisBase<R>::Desc::P_ON_LOWER;
               range = upper - lower;
               assert(this->thesolver->theUBbound[idx] == R(infinity));
               this->thesolver->theUBbound[idx] = -this->thesolver->theLBbound[idx];
               this->thesolver->theLBbound[idx] = R(-infinity);
               break;

            case SPxBasisBase<R>::Desc::P_ON_LOWER :
               ds.colStatus(IdNumber) = SPxBasisBase<R>::Desc::P_ON_UPPER;
               range = lower - upper;
               assert(this->thesolver->theLBbound[idx] == R(-infinity));
               this->thesolver->theLBbound[idx] = -this->thesolver->theUBbound[idx];
               this->thesolver->theUBbound[idx] = R(infinity);
               break;

            default :
               ++skipped;
               MSG_WARNING((*this->thesolver->spxout),
                           (*this->thesolver->spxout) << "FVEC unexpected status: " << static_cast<int>(stat)
                           << " index: " << idx
                           << " val: " << this->thesolver->fVec()[idx]
                           << " upd: " << this->thesolver->fVec().delta()[idx]
                           << " lower: " << lower
                           << " upper: " << upper
                           << " bp.val: " << breakpoints[i].val
                           << std::endl;)
            }
         }

         MSG_DEBUG(std::cout << "basic row/col flipped from: " << stat
                   << " index: " << idx
                   << " val: " << this->thesolver->fVec()[idx]
                   << " upd: " << this->thesolver->fVec().delta()[idx]
                   << " lower: " << lower
                   << " upper: " << upper
                   << " bp.val: " << breakpoints[i].val
                   << std::endl;)
         assert(spxAbs(range) < 1e20);
         assert(updPrimRhs[idx] == 0);
         updPrimRhs.add(idx, range);
      }
   }

   nflips -= skipped;

   if(nflips > 0)
   {
      if(this->thesolver->rep() == SPxSolverBase<R>::ROW)
      {
         assert(this->m_type == SPxSolverBase<R>::ENTER);
         (*this->thesolver->theCoPrhs) -= updPrimRhs;
         this->thesolver->setup4coSolve2(&updPrimVec, &updPrimRhs);
      }
      else
      {
         assert(this->thesolver->rep() == SPxSolverBase<R>::COLUMN);
         assert(this->m_type == SPxSolverBase<R>::LEAVE);
         (*this->thesolver->theFrhs) -= updPrimRhs;
         this->thesolver->setup4solve2(&updPrimVec, &updPrimRhs);
      }
   }

   return;
}

/** store all available pivots/breakpoints in an array (positive pivot search direction) */
template <class R>
void SPxBoundFlippingRT<R>::collectBreakpointsMax(
   int&                  nBp,                /**< number of found breakpoints so far */
   int&                  minIdx,             /**< index to current minimal breakpoint */
   const int*            idx,                /**< pointer to indices of current VectorBase<R> */
   int                   nnz,                /**< number of nonzeros in current VectorBase<R> */
   const R*           upd,                /**< pointer to update values of current VectorBase<R> */
   const R*           vec,                /**< pointer to values of current VectorBase<R> */
   const R*           upp,                /**< pointer to upper bound/rhs of current VectorBase<R> */
   const R*           low,                /**< pointer to lower bound/lhs of current VectorBase<R> */
   BreakpointSource      src                 /**< type of VectorBase<R> (pVec, coPvec or fVec)*/
)
{
   R minVal;
   R curVal;
   const int* last;

   minVal = (nBp == 0) ? R(infinity) : breakpoints[minIdx].val;

   last = idx + nnz;

   for(; idx < last; ++idx)
   {
      int i = *idx;
      R x = upd[i];

      if(x > this->epsilon)
      {
         if(upp[i] < R(infinity))
         {
            R y = upp[i] - vec[i];
            curVal = (y <= 0) ? this->fastDelta / x : (y + this->fastDelta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if(curVal < minVal)
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      else if(x < -this->epsilon)
      {
         if(low[i] > R(-infinity))
         {
            R y = low[i] - vec[i];
            curVal = (y >= 0) ? -this->fastDelta / x : (y - this->fastDelta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if(curVal < minVal)
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }

      if(nBp >= breakpoints.size())
         breakpoints.reSize(nBp * 2);
   }

   return;
}

/** store all available pivots/breakpoints in an array (negative pivot search direction) */
template <class R>
void SPxBoundFlippingRT<R>::collectBreakpointsMin(
   int&                  nBp,                /**< number of found breakpoints so far */
   int&                  minIdx,             /**< index to current minimal breakpoint */
   const int*            idx,                /**< pointer to indices of current VectorBase<R> */
   int                   nnz,                /**< number of nonzeros in current VectorBase<R> */
   const R*           upd,                /**< pointer to update values of current VectorBase<R> */
   const R*           vec,                /**< pointer to values of current VectorBase<R> */
   const R*           upp,                /**< pointer to upper bound/rhs of current VectorBase<R> */
   const R*           low,                /**< pointer to lower bound/lhs of current VectorBase<R> */
   BreakpointSource      src                 /**< type of VectorBase<R> (pVec, coPvec or fVec)*/
)
{
   R minVal;
   R curVal;
   const int* last;

   minVal = (nBp == 0) ? R(infinity) : breakpoints[minIdx].val;

   last = idx + nnz;

   for(; idx < last; ++idx)
   {
      int i = *idx;
      R x = upd[i];

      if(x > this->epsilon)
      {
         if(low[i] > R(-infinity))
         {
            R y = low[i] - vec[i];

            curVal = (y >= 0) ? this->fastDelta / x : (this->fastDelta - y) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if(curVal < minVal)
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      else if(x < -this->epsilon)
      {
         if(upp[i] < R(infinity))
         {
            R y = upp[i] - vec[i];
            curVal = (y <= 0) ? -this->fastDelta / x : -(y + this->fastDelta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if(curVal < minVal)
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }

      if(nBp >= breakpoints.size())
         breakpoints.reSize(nBp * 2);
   }

   return;
}

/** get values for entering index and perform shifts if necessary */
template <class R>
bool SPxBoundFlippingRT<R>::getData(
   R&                 val,
   SPxId&                enterId,
   int                   idx,
   R                  stab,
   R                  degeneps,
   const R*           upd,
   const R*           vec,
   const R*           low,
   const R*           upp,
   BreakpointSource      src,
   R                  max
)
{
   if(src == PVEC)
   {
      this->thesolver->pVec()[idx] = this->thesolver->vector(idx) * this->thesolver->coPvec();
      R x = upd[idx];

      // skip breakpoint if it is too small
      if(spxAbs(x) < stab)
      {
         return false;
      }

      enterId = this->thesolver->id(idx);
      val = (max * x > 0) ? upp[idx] : low[idx];
      val = (val - vec[idx]) / x;

      if(upp[idx] == low[idx])
      {
         val = 0.0;

         if(vec[idx] > upp[idx])
            this->thesolver->theShift += vec[idx] - upp[idx];
         else
            this->thesolver->theShift += low[idx] - vec[idx];

         this->thesolver->upBound()[idx] = this->thesolver->lpBound()[idx] = vec[idx];
      }
      else if((max > 0 && val < -degeneps) || (max < 0 && val > degeneps))
      {
         val = 0.0;

         if(max * x > 0)
            this->thesolver->shiftUPbound(idx, vec[idx]);
         else
            this->thesolver->shiftLPbound(idx, vec[idx]);
      }
   }
   else // src == COPVEC
   {
      R x = upd[idx];

      if(spxAbs(x) < stab)
      {
         return false;
      }

      enterId = this->thesolver->coId(idx);
      val = (max * x > 0.0) ? upp[idx] : low[idx];
      val = (val - vec[idx]) / x;

      if(upp[idx] == low[idx])
      {
         val = 0.0;

         if(vec[idx] > upp[idx])
            this->thesolver->theShift += vec[idx] - upp[idx];
         else
            this->thesolver->theShift += low[idx] - vec[idx];

         this->thesolver->ucBound()[idx] = this->thesolver->lcBound()[idx] = vec[idx];
      }
      else if((max > 0 && val < -degeneps) || (max < 0 && val > degeneps))
      {
         val = 0.0;

         if(max * x > 0)
            this->thesolver->shiftUCbound(idx, vec[idx]);
         else
            this->thesolver->shiftLCbound(idx, vec[idx]);
      }
   }

   return true;
}

/** get values for leaving index and perform shifts if necessary */
template <class R>
bool SPxBoundFlippingRT<R>::getData(
   R&                 val,
   int&                  leaveIdx,
   int                   idx,
   R                  stab,
   R                  degeneps,
   const R*           upd,
   const R*           vec,
   const R*           low,
   const R*           upp,
   BreakpointSource      src,
   R                  max
)
{
   assert(src == FVEC);

   R x = upd[idx];

   // skip breakpoint if it is too small
   if(spxAbs(x) < stab)
   {
      return false;
   }

   leaveIdx = idx;
   val = (max * x > 0) ? upp[idx] : low[idx];
   val = (val - vec[idx]) / x;

   if(upp[idx] == low[idx])
   {
      val = 0.0;
      this->thesolver->shiftLBbound(idx, vec[idx]);
      this->thesolver->shiftUBbound(idx, vec[idx]);
   }
   else if((max > 0 && val < -degeneps) || (max < 0 && val > degeneps))
   {
      val = 0.0;

      if(this->thesolver->dualStatus(this->thesolver->baseId(idx)) != SPxBasisBase<R>::Desc::D_ON_BOTH)
      {
         if(max * x > 0)
            this->thesolver->shiftUBbound(idx, vec[idx]);
         else
            this->thesolver->shiftLBbound(idx, vec[idx]);
      }
   }

   return true;
}

/** determine entering row/column */
template <class R>
SPxId SPxBoundFlippingRT<R>::selectEnter(
   R&                 val,
   int                   leaveIdx,
   bool                  polish
)
{
   assert(this->m_type == SPxSolverBase<R>::LEAVE);
   assert(this->thesolver->boundflips == 0);

   // reset the history and try again to do some long steps
   if(this->thesolver->leaveCount % LONGSTEP_FREQ == 0)
   {
      MSG_DEBUG(std::cout << "DLBFRT06 resetting long step history" << std::endl;)
      flipPotential = 1;
   }

   if(!enableBoundFlips || polish || this->thesolver->rep() == SPxSolverBase<R>::ROW
         || flipPotential <= 0)
   {
      MSG_DEBUG(std::cout << "DLBFRT07 switching to fast ratio test" << std::endl;)
      return SPxFastRT<R>::selectEnter(val, leaveIdx, polish);
   }

   const R*  pvec = this->thesolver->pVec().get_const_ptr();
   const R*  pupd = this->thesolver->pVec().delta().values();
   const int*   pidx = this->thesolver->pVec().delta().indexMem();
   int          pupdnnz = this->thesolver->pVec().delta().size();
   const R*  lpb  = this->thesolver->lpBound().get_const_ptr();
   const R*  upb  = this->thesolver->upBound().get_const_ptr();

   const R*  cvec = this->thesolver->coPvec().get_const_ptr();
   const R*  cupd = this->thesolver->coPvec().delta().values();
   const int*   cidx = this->thesolver->coPvec().delta().indexMem();
   int          cupdnnz = this->thesolver->coPvec().delta().size();
   const R*  lcb  = this->thesolver->lcBound().get_const_ptr();
   const R*  ucb  = this->thesolver->ucBound().get_const_ptr();

   this->resetTols();

   R max;

   // index in breakpoint array of minimal value (i.e. choice of normal RT)
   int minIdx;

   // temporary breakpoint data structure to make swaps possible
   Breakpoint tmp;

   // most stable pivot value in candidate set
   R moststable;

   // initialize invalid enterId
   SPxId enterId;

   // slope of objective function improvement
   R slope;

   // number of found breakpoints
   int nBp;

   // number of passed breakpoints
   int npassedBp;

   R degeneps;
   R stab;
   bool instable;

   max = val;
   val = 0.0;
   moststable = 0.0;
   nBp = 0;
   minIdx = -1;

   // get breakpoints and and determine the index of the minimal value
   if(max > 0)
   {
      collectBreakpointsMax(nBp, minIdx, pidx, pupdnnz, pupd, pvec, upb, lpb, PVEC);
      // coverity[negative_returns]
      collectBreakpointsMax(nBp, minIdx, cidx, cupdnnz, cupd, cvec, ucb, lcb, COPVEC);
   }
   else
   {
      collectBreakpointsMin(nBp, minIdx, pidx, pupdnnz, pupd, pvec, upb, lpb, PVEC);
      // coverity[negative_returns]
      collectBreakpointsMin(nBp, minIdx, cidx, cupdnnz, cupd, cvec, ucb, lcb, COPVEC);
   }

   if(nBp == 0)
   {
      val = max;
      return enterId;
   }

   assert(minIdx >= 0);

   // swap smallest breakpoint to the front to skip the sorting phase if no bound flip is possible
   tmp = breakpoints[minIdx];
   breakpoints[minIdx] = breakpoints[0];
   breakpoints[0] = tmp;

   // get initial slope
   slope = spxAbs(this->thesolver->fTest()[leaveIdx]);

   if(slope == 0)
   {
      // this may only happen if SoPlex decides to make an instable pivot
      assert(this->thesolver->instableLeaveNum >= 0);
      // restore original slope
      slope = spxAbs(this->thesolver->instableLeaveVal);
   }

   // set up structures for the quicksort implementation
   BreakpointCompare compare;
   compare.entry = breakpoints.get_const_ptr();

   // pointer to end of sorted part of breakpoints
   int sorted = 0;
   // minimum number of entries that are supposed to be sorted by partial sort
   int sortsize = 4;

   // get all skipable breakpoints
   for(npassedBp = 0; npassedBp < nBp && slope > 0; ++npassedBp)
   {
      // sort breakpoints only partially to save time
      if(npassedBp > sorted)
      {
         sorted = SPxQuicksortPart(breakpoints.get_ptr(), compare, sorted + 1, nBp, sortsize);
      }

      int i = breakpoints[npassedBp].idx;

      // compute new slope
      if(breakpoints[npassedBp].src == PVEC)
      {
         if(this->thesolver->isBasic(i))
         {
            // mark basic indices
            breakpoints[npassedBp].idx = -1;
            this->thesolver->pVec().delta().clearIdx(i);
         }
         else
         {
            R absupd = spxAbs(pupd[i]);
            slope -= (this->thesolver->upper(i) * absupd) - (this->thesolver->lower(i) * absupd);

            // get most stable pivot
            if(absupd > moststable)
               moststable = absupd;
         }
      }
      else
      {
         assert(breakpoints[npassedBp].src == COPVEC);

         if(this->thesolver->isCoBasic(i))
         {
            // mark basic indices
            breakpoints[npassedBp].idx = -1;
            this->thesolver->coPvec().delta().clearIdx(i);
         }
         else
         {
            R absupd = spxAbs(cupd[i]);
            slope -= (this->thesolver->rhs(i) * absupd) - (this->thesolver->lhs(i) * absupd);

            if(absupd > moststable)
               moststable = absupd;
         }
      }
   }

   --npassedBp;
   assert(npassedBp >= 0);

   // check for unboundedness/infeasibility
   if(slope > this->delta && npassedBp >= nBp - 1)
   {
      MSG_DEBUG(std::cout << "DLBFRT02 " << this->thesolver->basis().iteration()
                << ": unboundedness in ratio test" << std::endl;)
      flipPotential -= 0.5;
      val = max;
      return SPxFastRT<R>::selectEnter(val, leaveIdx);
   }

   MSG_DEBUG(std::cout << "DLBFRT01 "
             << this->thesolver->basis().iteration()
             << ": number of flip candidates: "
             << npassedBp
             << std::endl;)

   // try to get a more stable pivot by looking at those with similar step length
   int stableBp;              // index to walk over additional breakpoints (after slope change)
   int bestBp = -1;           // breakpoints index with best possible stability
   R bestDelta = breakpoints[npassedBp].val;  // best step length (after bound flips)

   for(stableBp = npassedBp + 1; stableBp < nBp; ++stableBp)
   {
      R stableDelta = 0;

      // get next breakpoints in increasing order
      if(stableBp > sorted)
      {
         sorted = SPxQuicksortPart(breakpoints.get_ptr(), compare, sorted + 1, nBp, sortsize);
      }

      int idx = breakpoints[stableBp].idx;

      if(breakpoints[stableBp].src == PVEC)
      {
         if(this->thesolver->isBasic(idx))
         {
            // mark basic indices
            breakpoints[stableBp].idx = -1;
            this->thesolver->pVec().delta().clearIdx(idx);
            continue;
         }

         R x = pupd[idx];

         if(spxAbs(x) > moststable)
         {
            this->thesolver->pVec()[idx] = this->thesolver->vector(idx) * this->thesolver->coPvec();
            stableDelta = (x > 0.0) ? upb[idx] : lpb[idx];
            stableDelta = (stableDelta - pvec[idx]) / x;

            if(stableDelta <= bestDelta)
            {
               moststable = spxAbs(x);
               bestBp = stableBp;
            }
         }
      }
      else
      {
         if(this->thesolver->isCoBasic(idx))
         {
            // mark basic indices
            breakpoints[stableBp].idx = -1;
            this->thesolver->coPvec().delta().clearIdx(idx);
            continue;
         }

         R x = cupd[idx];

         if(spxAbs(x) > moststable)
         {
            stableDelta = (x > 0.0) ? ucb[idx] : lcb[idx];
            stableDelta = (stableDelta - cvec[idx]) / x;

            if(stableDelta <= bestDelta)
            {
               moststable = spxAbs(x);
               bestBp = stableBp;
            }
         }
      }

      // stop searching if the step length is too big
      if(stableDelta > this->delta + bestDelta)
         break;
   }

   degeneps = this->fastDelta / moststable;  /* as in SPxFastRT */
   // get stability requirements
   instable = this->thesolver->instableLeave;
   assert(!instable || this->thesolver->instableLeaveNum >= 0);
   stab = instable ? LOWSTAB : SPxFastRT<R>::minStability(moststable);

   bool foundStable = false;

   if(bestBp >= 0)
   {
      // found a more stable pivot
      if(moststable > stab)
      {
         // stability requirements are satisfied
         int idx = breakpoints[bestBp].idx;
         assert(idx >= 0);

         if(breakpoints[bestBp].src == PVEC)
            foundStable = getData(val, enterId, idx, stab, degeneps, pupd, pvec, lpb, upb, PVEC, max);
         else
            foundStable = getData(val, enterId, idx, stab, degeneps, cupd, cvec, lcb, ucb, COPVEC, max);
      }
   }

   else
   {
      // scan passed breakpoints from back to front and stop as soon as a good one is found
      while(!foundStable && npassedBp >= 0)
      {
         int idx = breakpoints[npassedBp].idx;

         // only look for non-basic variables
         if(idx >= 0)
         {
            if(breakpoints[npassedBp].src == PVEC)
               foundStable = getData(val, enterId, idx, stab, degeneps, pupd, pvec, lpb, upb, PVEC, max);
            else
               foundStable = getData(val, enterId, idx, stab, degeneps, cupd, cvec, lcb, ucb, COPVEC, max);
         }

         --npassedBp;
      }

      ++npassedBp;
   }

   if(!foundStable)
   {
      assert(!enterId.isValid());

      if(relax_count < MAX_RELAX_COUNT)
      {
         MSG_DEBUG(std::cout << "DLBFRT04 "
                   << this->thesolver->basis().iteration()
                   << ": no valid enterId found - relaxing..."
                   << std::endl;)
         this->relax();
         ++relax_count;
         // restore original value
         val = max;
         // try again with relaxed delta
         return SPxBoundFlippingRT<R>::selectEnter(val, leaveIdx);
      }
      else
      {
         MSG_DEBUG(std::cout << "DLBFRT05 "
                   << this->thesolver->basis().iteration()
                   << " no valid enterId found - breaking..."
                   << std::endl;)
         return enterId;
      }
   }
   else
   {
      relax_count = 0;
      this->tighten();
   }

   // flip bounds of skipped breakpoints only if a nondegenerate step is to be performed
   if(npassedBp > 0 && spxAbs(breakpoints[npassedBp].val) > this->fastDelta)
   {
      flipAndUpdate(npassedBp);
      this->thesolver->boundflips = npassedBp;

      if(npassedBp >= 10)
         flipPotential = 1;
      else
         flipPotential -= 0.05;
   }
   else
   {
      this->thesolver->boundflips = 0;
      flipPotential -= 0.1;
   }

   MSG_DEBUG(std::cout << "DLBFRT06 "
             << this->thesolver->basis().iteration()
             << ": selected Id: "
             << enterId
             << " number of candidates: "
             << nBp
             << std::endl;)
   return enterId;
}

/** determine leaving row/column */
template <class R>
int SPxBoundFlippingRT<R>::selectLeave(
   R&                 val,
   R                  enterTest,
   bool                  polish
)
{
   assert(this->m_type == SPxSolverBase<R>::ENTER);
   assert(this->thesolver->boundflips == 0);

   // reset the history and try again to do some long steps
   if(this->thesolver->enterCount % LONGSTEP_FREQ == 0)
   {
      MSG_DEBUG(std::cout << "DEBFRT06 resetting long step history" << std::endl;)
      flipPotential = 1;
   }

   if(polish || !enableBoundFlips || !enableRowBoundFlips
         || this->thesolver->rep() == SPxSolverBase<R>::COLUMN || flipPotential <= 0)
   {
      MSG_DEBUG(std::cout << "DEBFRT07 switching to fast ratio test" << std::endl;)
      return SPxFastRT<R>::selectLeave(val, enterTest, polish);
   }

   const R*  vec =
      this->thesolver->fVec().get_const_ptr();         /**< pointer to values of current VectorBase<R> */
   const R*  upd =
      this->thesolver->fVec().delta().values();        /**< pointer to update values of current VectorBase<R> */
   const int*   idx =
      this->thesolver->fVec().delta().indexMem();      /**< pointer to indices of current VectorBase<R> */
   int          updnnz =
      this->thesolver->fVec().delta().size();       /**< number of nonzeros in update VectorBase<R> */
   const R*  lb  =
      this->thesolver->lbBound().get_const_ptr();      /**< pointer to lower bound/lhs of current VectorBase<R> */
   const R*  ub  =
      this->thesolver->ubBound().get_const_ptr();      /**< pointer to upper bound/rhs of current VectorBase<R> */

   this->resetTols();

   R max;

   // index in breakpoint array of minimal value (i.e. choice of normal RT)
   int minIdx;

   // temporary breakpoint data structure to make swaps possible
   Breakpoint tmp;

   // most stable pivot value in candidate set
   R moststable;

   // initialize invalid leaving index
   int leaveIdx = -1;

   // slope of objective function improvement
   R slope;

   // number of found breakpoints
   int nBp;

   // number of passed breakpoints
   int npassedBp;

   R degeneps;
   R stab;
   bool instable;

   max = val;
   val = 0.0;
   moststable = 0.0;
   nBp = 0;
   minIdx = -1;

   assert(this->thesolver->fVec().delta().isSetup());

   // get breakpoints and and determine the index of the minimal value
   if(max > 0)
   {
      collectBreakpointsMax(nBp, minIdx, idx, updnnz, upd, vec, ub, lb, FVEC);
   }
   else
   {
      collectBreakpointsMin(nBp, minIdx, idx, updnnz, upd, vec, ub, lb, FVEC);
   }

   // return -1 if no BP was found
   if(nBp == 0)
   {
      val = max;
      return leaveIdx;
   }

   assert(minIdx >= 0);

   // swap smallest breakpoint to the front to skip the sorting phase if no bound flip is possible
   tmp = breakpoints[minIdx];
   breakpoints[minIdx] = breakpoints[0];
   breakpoints[0] = tmp;

   // get initial slope
   slope = spxAbs(enterTest);

   if(slope == 0)
   {
      // this may only happen if SoPlex decides to make an instable pivot
      assert(this->thesolver->instableEnterId.isValid());
      // restore original slope
      slope = this->thesolver->instableEnterVal;
   }

   // set up structures for the quicksort implementation
   BreakpointCompare compare;
   compare.entry = breakpoints.get_const_ptr();

   // pointer to end of sorted part of breakpoints
   int sorted = 0;
   // minimum number of entries that are supposed to be sorted by partial sort
   int sortsize = 4;

   // get all skipable breakpoints
   for(npassedBp = 0; npassedBp < nBp && slope > 0; ++npassedBp)
   {
      // sort breakpoints only partially to save time
      if(npassedBp > sorted)
      {
         sorted = SPxQuicksortPart(breakpoints.get_ptr(), compare, sorted + 1, nBp, sortsize);
      }

      assert(breakpoints[npassedBp].src == FVEC);
      int breakpointidx = breakpoints[npassedBp].idx;
      // compute new slope
      R upper;
      R lower;
      R absupd = spxAbs(upd[breakpointidx]);
      SPxId baseId = this->thesolver->baseId(breakpointidx);
      int i = this->thesolver->number(baseId);

      if(baseId.isSPxColId())
      {
         upper = this->thesolver->upper(i);
         lower = this->thesolver->lower(i);
      }
      else
      {
         assert(baseId.isSPxRowId());
         upper = this->thesolver->rhs(i);
         lower = this->thesolver->lhs(i);
      }

      slope -= (upper * absupd) - (lower * absupd);

      // get most stable pivot
      if(absupd > moststable)
         moststable = absupd;
   }

   --npassedBp;
   assert(npassedBp >= 0);

   // check for unboundedness/infeasibility
   if(slope > this->delta && npassedBp >= nBp - 1)
   {
      MSG_DEBUG(std::cout << "DEBFRT02 " << this->thesolver->basis().iteration()
                << ": unboundedness in ratio test" << std::endl;)
      flipPotential -= 0.5;
      val = max;
      return SPxFastRT<R>::selectLeave(val, enterTest);
   }

   MSG_DEBUG(std::cout << "DEBFRT01 "
             << this->thesolver->basis().iteration()
             << ": number of flip candidates: "
             << npassedBp
             << std::endl;)

   // try to get a more stable pivot by looking at those with similar step length
   int stableBp;              // index to walk over additional breakpoints (after slope change)
   int bestBp = -1;           // breakpoints index with best possible stability
   R bestDelta = breakpoints[npassedBp].val;  // best step length (after bound flips)

   for(stableBp = npassedBp + 1; stableBp < nBp; ++stableBp)
   {
      R stableDelta = 0;

      // get next breakpoints in increasing order
      if(stableBp > sorted)
      {
         sorted = SPxQuicksortPart(breakpoints.get_ptr(), compare, sorted + 1, nBp, sortsize);
      }

      int breakpointidx = breakpoints[stableBp].idx;
      assert(breakpoints[stableBp].src == FVEC);
      R x = upd[breakpointidx];

      if(spxAbs(x) > moststable)
      {
         stableDelta = (x > 0.0) ? ub[breakpointidx] : lb[breakpointidx];
         stableDelta = (stableDelta - vec[breakpointidx]) / x;

         if(stableDelta <= bestDelta)
         {
            moststable = spxAbs(x);
            bestBp = stableBp;
         }
      }
      // stop searching if the step length is too big
      else if(stableDelta > this->delta + bestDelta)
         break;
   }

   degeneps = this->fastDelta / moststable;  /* as in SPxFastRT */
   // get stability requirements
   instable = this->thesolver->instableEnter;
   assert(!instable || this->thesolver->instableEnterId.isValid());
   stab = instable ? LOWSTAB : SPxFastRT<R>::minStability(moststable);

   bool foundStable = false;

   if(bestBp >= 0)
   {
      // found a more stable pivot
      if(moststable > stab)
      {
         // stability requirements are satisfied
         int breakpointidx = breakpoints[bestBp].idx;
         assert(breakpointidx >= 0);
         foundStable = getData(val, leaveIdx, breakpointidx, moststable, degeneps, upd, vec, lb, ub, FVEC,
                               max);
      }
   }

   else
   {
      // scan passed breakpoints from back to front and stop as soon as a good one is found
      while(!foundStable && npassedBp >= 0)
      {
         int breakpointidx = breakpoints[npassedBp].idx;

         // only look for non-basic variables
         if(breakpointidx >= 0)
         {
            foundStable = getData(val, leaveIdx, breakpointidx, moststable, degeneps, upd, vec, lb, ub, FVEC,
                                  max);
         }

         --npassedBp;
      }

      ++npassedBp;
   }

   if(!foundStable)
   {
      assert(leaveIdx < 0);

      if(relax_count < MAX_RELAX_COUNT)
      {
         MSG_DEBUG(std::cout << "DEBFRT04 "
                   << this->thesolver->basis().iteration()
                   << ": no valid leaveIdx found - relaxing..."
                   << std::endl;)
         this->relax();
         ++relax_count;
         // restore original value
         val = max;
         // try again with relaxed delta
         return SPxBoundFlippingRT<R>::selectLeave(val, enterTest);
      }
      else
      {
         MSG_DEBUG(std::cout << "DEBFRT05 "
                   << this->thesolver->basis().iteration()
                   << " no valid leaveIdx found - breaking..."
                   << std::endl;)
         return leaveIdx;
      }
   }
   else
   {
      relax_count = 0;
      this->tighten();
   }

   // flip bounds of skipped breakpoints only if a nondegenerate step is to be performed
   if(npassedBp > 0 && spxAbs(breakpoints[npassedBp].val) > this->fastDelta)
   {
      flipAndUpdate(npassedBp);
      this->thesolver->boundflips = npassedBp;

      if(npassedBp >= 10)
         flipPotential = 1;
      else
         flipPotential -= 0.05;
   }
   else
   {
      this->thesolver->boundflips = 0;
      flipPotential -= 0.1;
   }

   MSG_DEBUG(std::cout << "DEBFRT06 "
             << this->thesolver->basis().iteration()
             << ": selected Index: "
             << leaveIdx
             << " number of candidates: "
             << nBp
             << std::endl;)

   return leaveIdx;
}


} // namespace soplex
