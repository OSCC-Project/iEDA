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

/* Updating the Basis for Leaving Variables
 */
#include <assert.h>
#include <stdio.h>

#include "soplex/spxdefines.h"
#include "soplex/spxpricer.h"
#include "soplex/spxsolver.h"
#include "soplex/spxratiotester.h"
#include "soplex/spxout.h"
#include "soplex/exceptions.h"

namespace soplex
{
static const Real reject_leave_tol = 1e-10; // = LOWSTAB as defined in spxfastrt.hpp

/*
  VectorBase<R> |fTest| gives the feasibility test of all basic variables. For its
  computation |fVec|, |theUBbound| and |theLBbound| must be setup correctly.
  Values of |fTest| $<0$ represent infeasible variables, which are eligible
  for leaving the basis in the simplex loop.
*/
template <class R>
void SPxSolverBase<R>::computeFtest()
{

   assert(type() == LEAVE);

   R theeps = entertol();
   m_pricingViolUpToDate = true;
   m_pricingViolCoUpToDate = true;
   m_pricingViol = 0;
   m_pricingViolCo = 0;
   m_numViol = 0;
   infeasibilities.clear();
   int sparsitythreshold = (int)(sparsePricingFactor * dim());

   for(int i = 0; i < dim(); ++i)
   {
      theCoTest[i] = ((*theFvec)[i] > theUBbound[i])
                     ? theUBbound[i] - (*theFvec)[i]
                     : (*theFvec)[i] - theLBbound[i];

      if(remainingRoundsLeave == 0)
      {
         if(theCoTest[i] < -theeps)
         {
            m_pricingViol -= theCoTest[i];
            infeasibilities.addIdx(i);
            isInfeasible[i] = SPxPricer<R>::VIOLATED;
            ++m_numViol;
         }
         else
            isInfeasible[i] = SPxPricer<R>::NOT_VIOLATED;

         if(infeasibilities.size() > sparsitythreshold)
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << " --- using dense pricing"
                      << std::endl;)
            remainingRoundsLeave = DENSEROUNDS;
            sparsePricingLeave = false;
            infeasibilities.clear();
         }
      }
      else if(theCoTest[i] < -theeps)
      {
         m_pricingViol -= theCoTest[i];
         m_numViol++;
      }
   }

   if(infeasibilities.size() == 0 && !sparsePricingLeave)
   {
      --remainingRoundsLeave;
   }
   else if(infeasibilities.size() <= sparsitythreshold && !sparsePricingLeave)
   {
      MSG_INFO2((*this->spxout),
                std::streamsize prec = spxout->precision();

                if(hyperPricingLeave)
                (*this->spxout) << " --- using hypersparse pricing, ";
                else
                   (*this->spxout) << " --- using sparse pricing, ";
                   (*this->spxout) << "sparsity: "
                   << std::setw(6) << std::fixed << std::setprecision(4)
                   << (R) m_numViol / dim()
                   << std::scientific << std::setprecision(int(prec))
                   << std::endl;
                  )
            sparsePricingLeave = true;
   }
}

template <class R>
void SPxSolverBase<R>::updateFtest()
{
   const IdxSet& idx = theFvec->idx();
   VectorBase<R>& ftest = theCoTest;      // |== fTest()|
   assert(&ftest == &fTest());

   assert(type() == LEAVE);

   updateViols.clear();
   R theeps = entertol();

   for(int j = idx.size() - 1; j >= 0; --j)
   {
      int i = idx.index(j);

      if(m_pricingViolUpToDate && ftest[i] < -theeps)
         // violation was present before this iteration
         m_pricingViol += ftest[i];

      ftest[i] = ((*theFvec)[i] > theUBbound[i])
                 ? theUBbound[i] - (*theFvec)[i]
                 : (*theFvec)[i] - theLBbound[i];

      if(sparsePricingLeave && ftest[i] < -theeps)
      {
         assert(remainingRoundsLeave == 0);

         if(m_pricingViolUpToDate)
            m_pricingViol -= ftest[i];

         if(isInfeasible[i] == SPxPricer<R>::NOT_VIOLATED)
         {
            // this can cause problems - we cannot keep on adding indeces to infeasibilities,
            // because they are not deleted in hyper mode...
            //             if( !hyperPricingLeave )
            infeasibilities.addIdx(i);
            isInfeasible[i] = SPxPricer<R>::VIOLATED;
         }

         if(hyperPricingLeave)
            updateViols.addIdx(i);
      }
      else if(m_pricingViolUpToDate && ftest[i] < -theeps)
         m_pricingViol -= ftest[i];
   }

   // if boundflips were performed, we need to update these indices as well
   if(boundflips > 0)
   {
      R eps = epsilon();

      for(int j = 0; j < solveVector3->size(); ++j)
      {
         if(spxAbs(solveVector3->value(j)) > eps)
         {
            int i = solveVector3->index(j);

            if(m_pricingViolUpToDate && ftest[i] < -theeps)
               m_pricingViol += ftest[i];

            ftest[i] = ((*theFvec)[i] > theUBbound[i]) ? theUBbound[i] - (*theFvec)[i] :
                       (*theFvec)[i] - theLBbound[i];

            if(sparsePricingLeave && ftest[i] < -theeps)
            {
               assert(remainingRoundsLeave == 0);

               if(m_pricingViolUpToDate)
                  m_pricingViol -= ftest[i];

               if(!isInfeasible[i])
               {
                  infeasibilities.addIdx(i);
                  isInfeasible[i] = true;
               }
            }
            else if(m_pricingViolUpToDate && ftest[i] < -theeps)
               m_pricingViol -= ftest[i];
         }
      }
   }
}


/* compute statistics on leaving variable
   Compute a set of statistical values on the variable selected for leaving the
   basis.
*/
template <class R>
void SPxSolverBase<R>::getLeaveVals(
   int leaveIdx,
   typename SPxBasisBase<R>::Desc::Status& leaveStat,
   SPxId& leaveId,
   R& leaveMax,
   R& leavebound,
   int& leaveNum,
   StableSum<R>& objChange)
{
   typename SPxBasisBase<R>::Desc& ds = this->desc();
   leaveId = this->baseId(leaveIdx);

   if(leaveId.isSPxRowId())
   {
      leaveNum = this->number(SPxRowId(leaveId));
      leaveStat = ds.rowStatus(leaveNum);

      assert(isBasic(leaveStat));

      switch(leaveStat)
      {
      case SPxBasisBase<R>::Desc::P_ON_UPPER :
         assert(rep() == ROW);
         ds.rowStatus(leaveNum) = this->dualRowStatus(leaveNum);
         leavebound = 0;
         leaveMax = R(-infinity);
         break;

      case SPxBasisBase<R>::Desc::P_ON_LOWER :
         assert(rep() == ROW);
         ds.rowStatus(leaveNum) = this->dualRowStatus(leaveNum);
         leavebound = 0;
         leaveMax = R(infinity);
         break;

      case SPxBasisBase<R>::Desc::P_FREE :
         assert(rep() == ROW);
         throw SPxInternalCodeException("XLEAVE01 This should never happen.");

      case SPxBasisBase<R>::Desc::D_FREE :
         assert(rep() == COLUMN);
         ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_FIXED;
         assert(this->lhs(leaveNum) == this->rhs(leaveNum));
         leavebound = -this->rhs(leaveNum);

         if((*theFvec)[leaveIdx] < theLBbound[leaveIdx])
            leaveMax = R(infinity);
         else
            leaveMax = R(-infinity);

         break;

      case SPxBasisBase<R>::Desc::D_ON_LOWER :
         assert(rep() == COLUMN);
         ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         leavebound = -this->rhs(leaveNum);                // slack !!
         leaveMax = R(infinity);
         objChange += theLRbound[leaveNum] * this->rhs(leaveNum);
         break;

      case SPxBasisBase<R>::Desc::D_ON_UPPER :
         assert(rep() == COLUMN);
         ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
         leavebound = -this->lhs(leaveNum);                // slack !!
         leaveMax = R(-infinity);
         objChange += theURbound[leaveNum] * this->lhs(leaveNum);
         break;

      case SPxBasisBase<R>::Desc::D_ON_BOTH :
         assert(rep() == COLUMN);

         if((*theFvec)[leaveIdx] > theLBbound[leaveIdx])
         {
            ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
            theLRbound[leaveNum] = R(-infinity);
            leavebound = -this->lhs(leaveNum);            // slack !!
            leaveMax = R(-infinity);
            objChange += theURbound[leaveNum] * this->lhs(leaveNum);
         }
         else
         {
            ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
            theURbound[leaveNum] = R(infinity);
            leavebound = -this->rhs(leaveNum);            // slack !!
            leaveMax = R(infinity);
            objChange += theLRbound[leaveNum] * this->rhs(leaveNum);
         }

         break;

      default:
         throw SPxInternalCodeException("XLEAVE02 This should never happen.");
      }

      MSG_DEBUG(std::cout << "DLEAVE51 SPxSolverBase<R>::getLeaveVals() : row " << leaveNum
                << ": " << leaveStat
                << " -> " << ds.rowStatus(leaveNum)
                << " objChange: " << objChange
                << std::endl;)
   }

   else
   {
      assert(leaveId.isSPxColId());
      leaveNum = this->number(SPxColId(leaveId));
      leaveStat = ds.colStatus(leaveNum);

      assert(isBasic(leaveStat));

      switch(leaveStat)
      {
      case SPxBasisBase<R>::Desc::P_ON_UPPER :
         assert(rep() == ROW);
         ds.colStatus(leaveNum) = this->dualColStatus(leaveNum);
         leavebound = 0;
         leaveMax = R(-infinity);
         break;

      case SPxBasisBase<R>::Desc::P_ON_LOWER :
         assert(rep() == ROW);
         ds.colStatus(leaveNum) = this->dualColStatus(leaveNum);
         leavebound = 0;
         leaveMax = R(infinity);
         break;

      case SPxBasisBase<R>::Desc::P_FREE :
         assert(rep() == ROW);
         ds.colStatus(leaveNum) = this->dualColStatus(leaveNum);

         if((*theFvec)[leaveIdx] < theLBbound[leaveIdx])
         {
            leavebound = theLBbound[leaveIdx];
            leaveMax = R(-infinity);
         }
         else
         {
            leavebound = theUBbound[leaveIdx];
            leaveMax = R(infinity);
         }

         break;

      case SPxBasisBase<R>::Desc::D_FREE :
         assert(rep() == COLUMN);
         assert(SPxLPBase<R>::upper(leaveNum) == SPxLPBase<R>::lower(leaveNum));
         ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_FIXED;
         leavebound = SPxLPBase<R>::upper(leaveNum);
         objChange += this->maxObj(leaveNum) * leavebound;

         if((*theFvec)[leaveIdx] < theLBbound[leaveIdx])
            leaveMax = R(infinity);
         else
            leaveMax = R(-infinity);

         break;

      case SPxBasisBase<R>::Desc::D_ON_LOWER :
         assert(rep() == COLUMN);
         ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         leavebound = SPxLPBase<R>::upper(leaveNum);
         objChange += theUCbound[leaveNum] * leavebound;
         leaveMax = R(-infinity);
         break;

      case SPxBasisBase<R>::Desc::D_ON_UPPER :
         assert(rep() == COLUMN);
         ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
         leavebound = SPxLPBase<R>::lower(leaveNum);
         objChange += theLCbound[leaveNum] * leavebound;
         leaveMax = R(infinity);
         break;

      case SPxBasisBase<R>::Desc::D_ON_BOTH :
         assert(rep() == COLUMN);

         if((*theFvec)[leaveIdx] > theUBbound[leaveIdx])
         {
            leaveMax = R(-infinity);
            leavebound = SPxLPBase<R>::upper(leaveNum);
            objChange += theUCbound[leaveNum] * leavebound;
            theLCbound[leaveNum] = R(-infinity);
            ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         }
         else
         {
            leaveMax = R(infinity);
            leavebound = SPxLPBase<R>::lower(leaveNum);
            objChange += theLCbound[leaveNum] * leavebound;
            theUCbound[leaveNum] = R(infinity);
            ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
         }

         break;

      default:
         throw SPxInternalCodeException("XLEAVE03 This should never happen.");
      }

      MSG_DEBUG(std::cout << "DLEAVE52 SPxSolverBase<R>::getLeaveVals() : col " << leaveNum
                << ": " << leaveStat
                << " -> " << ds.colStatus(leaveNum)
                << " objChange: " << objChange
                << std::endl;)
   }
}

template <class R>
void SPxSolverBase<R>::getLeaveVals2(
   R leaveMax,
   SPxId enterId,
   R& enterBound,
   R& newUBbound,
   R& newLBbound,
   R& newCoPrhs,
   StableSum<R>& objChange
)
{
   typename SPxBasisBase<R>::Desc& ds = this->desc();

   enterBound = 0;

   if(enterId.isSPxRowId())
   {
      int idx = this->number(SPxRowId(enterId));
      typename SPxBasisBase<R>::Desc::Status enterStat = ds.rowStatus(idx);

      // coverity[switch_selector_expr_is_constant]
      switch(enterStat)
      {
      case SPxBasisBase<R>::Desc::D_FREE :
         assert(rep() == ROW);

         if(thePvec->delta()[idx] * leaveMax < 0)
            newCoPrhs = theLRbound[idx];
         else
            newCoPrhs = theURbound[idx];

         newUBbound = R(infinity);
         newLBbound = R(-infinity);
         ds.rowStatus(idx) = SPxBasisBase<R>::Desc::P_FIXED;
         break;

      case SPxBasisBase<R>::Desc::D_ON_UPPER :
         assert(rep() == ROW);
         newUBbound = 0;
         newLBbound = R(-infinity);
         ds.rowStatus(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
         newCoPrhs = theLRbound[idx];
         break;

      case SPxBasisBase<R>::Desc::D_ON_LOWER :
         assert(rep() == ROW);
         newUBbound = R(infinity);
         newLBbound = 0;
         ds.rowStatus(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         newCoPrhs = theURbound[idx];
         break;

      case SPxBasisBase<R>::Desc::D_ON_BOTH :
         assert(rep() == ROW);

         if(leaveMax * thePvec->delta()[idx] < 0)
         {
            newUBbound = 0;
            newLBbound = R(-infinity);
            ds.rowStatus(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
            newCoPrhs = theLRbound[idx];
         }
         else
         {
            newUBbound = R(infinity);
            newLBbound = 0;
            ds.rowStatus(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
            newCoPrhs = theURbound[idx];
         }

         break;

      case SPxBasisBase<R>::Desc::P_ON_UPPER :
         assert(rep() == COLUMN);
         ds.rowStatus(idx) = this->dualRowStatus(idx);

         if(this->lhs(idx) > R(-infinity))
            theURbound[idx] = theLRbound[idx];

         newCoPrhs = theLRbound[idx];        // slack !!
         newUBbound = -this->lhs(idx);
         newLBbound = -this->rhs(idx);
         enterBound = -this->rhs(idx);
         objChange -= newCoPrhs * this->rhs(idx);
         break;

      case SPxBasisBase<R>::Desc::P_ON_LOWER :
         assert(rep() == COLUMN);
         ds.rowStatus(idx) = this->dualRowStatus(idx);

         if(this->rhs(idx) < R(infinity))
            theLRbound[idx] = theURbound[idx];

         newCoPrhs = theURbound[idx];        // slack !!
         newLBbound = -this->rhs(idx);
         newUBbound = -this->lhs(idx);
         enterBound = -this->lhs(idx);
         objChange -= newCoPrhs * this->lhs(idx);
         break;

      case SPxBasisBase<R>::Desc::P_FREE :
         assert(rep() == COLUMN);
#if 1
         throw SPxInternalCodeException("XLEAVE04 This should never happen.");
#else
         MSG_ERROR(std::cerr << "ELEAVE53 ERROR: not yet debugged!" << std::endl;)
         ds.rowStatus(idx) = this->dualRowStatus(idx);
         newCoPrhs = theURbound[idx];        // slack !!
         newUBbound = R(infinity);
         newLBbound = R(-infinity);
         enterBound = 0;
#endif
         break;

      case SPxBasisBase<R>::Desc::P_FIXED :
         assert(rep() == COLUMN);
         MSG_ERROR(std::cerr << "ELEAVE54 "
                   << "ERROR! Tried to put a fixed row variable into the basis: "
                   << "idx="   << idx
                   << ", lhs=" << this->lhs(idx)
                   << ", rhs=" << this->rhs(idx) << std::endl;)
         throw SPxInternalCodeException("XLEAVE05 This should never happen.");

      default:
         throw SPxInternalCodeException("XLEAVE06 This should never happen.");
      }

      MSG_DEBUG(std::cout << "DLEAVE55 SPxSolverBase<R>::getLeaveVals2(): row " << idx
                << ": " << enterStat
                << " -> " << ds.rowStatus(idx)
                << " objChange: " << objChange
                << std::endl;)
   }

   else
   {
      assert(enterId.isSPxColId());
      int idx = this->number(SPxColId(enterId));
      typename SPxBasisBase<R>::Desc::Status enterStat = ds.colStatus(idx);

      // coverity[switch_selector_expr_is_constant]
      switch(enterStat)
      {
      case SPxBasisBase<R>::Desc::D_ON_UPPER :
         assert(rep() == ROW);
         newUBbound = 0;
         newLBbound = R(-infinity);
         ds.colStatus(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
         newCoPrhs = theLCbound[idx];
         break;

      case SPxBasisBase<R>::Desc::D_ON_LOWER :
         assert(rep() == ROW);
         newUBbound = R(infinity);
         newLBbound = 0;
         ds.colStatus(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         newCoPrhs = theUCbound[idx];
         break;

      case SPxBasisBase<R>::Desc::D_FREE :
         assert(rep() == ROW);
         newUBbound = R(infinity);
         newLBbound = R(-infinity);
         newCoPrhs = theLCbound[idx];
         ds.colStatus(idx) = SPxBasisBase<R>::Desc::P_FIXED;
         break;

      case SPxBasisBase<R>::Desc::D_ON_BOTH :
         assert(rep() == ROW);

         if(leaveMax * theCoPvec->delta()[idx] < 0)
         {
            newUBbound = 0;
            newLBbound = R(-infinity);
            ds.colStatus(idx) = SPxBasisBase<R>::Desc::P_ON_LOWER;
            newCoPrhs = theLCbound[idx];
         }
         else
         {
            newUBbound = R(infinity);
            newLBbound = 0;
            ds.colStatus(idx) = SPxBasisBase<R>::Desc::P_ON_UPPER;
            newCoPrhs = theUCbound[idx];
         }

         break;

      case SPxBasisBase<R>::Desc::P_ON_UPPER :
         assert(rep() == COLUMN);
         ds.colStatus(idx) = this->dualColStatus(idx);

         if(SPxLPBase<R>::lower(idx) > R(-infinity))
            theLCbound[idx] = theUCbound[idx];

         newCoPrhs = theUCbound[idx];
         newUBbound = SPxLPBase<R>::upper(idx);
         newLBbound = SPxLPBase<R>::lower(idx);
         enterBound = SPxLPBase<R>::upper(idx);
         objChange -= newCoPrhs * enterBound;
         break;

      case SPxBasisBase<R>::Desc::P_ON_LOWER :
         assert(rep() == COLUMN);
         ds.colStatus(idx) = this->dualColStatus(idx);

         if(SPxLPBase<R>::upper(idx) < R(infinity))
            theUCbound[idx] = theLCbound[idx];

         newCoPrhs = theLCbound[idx];
         newUBbound = SPxLPBase<R>::upper(idx);
         newLBbound = SPxLPBase<R>::lower(idx);
         enterBound = SPxLPBase<R>::lower(idx);
         objChange -= newCoPrhs * enterBound;
         break;

      case SPxBasisBase<R>::Desc::P_FREE :
         assert(rep() == COLUMN);
         ds.colStatus(idx) = this->dualColStatus(idx);

         if(thePvec->delta()[idx] * leaveMax > 0)
            newCoPrhs = theUCbound[idx];
         else
            newCoPrhs = theLCbound[idx];

         newUBbound = SPxLPBase<R>::upper(idx);
         newLBbound = SPxLPBase<R>::lower(idx);
         enterBound = 0;
         break;

      case SPxBasisBase<R>::Desc::P_FIXED :
         assert(rep() == COLUMN);
         MSG_ERROR(std::cerr << "ELEAVE56 "
                   << "ERROR! Tried to put a fixed column variable into the basis. "
                   << "idx="     << idx
                   << ", lower=" << this->lower(idx)
                   << ", upper=" << this->upper(idx) << std::endl;)
         throw SPxInternalCodeException("XLEAVE07 This should never happen.");

      default:
         throw SPxInternalCodeException("XLEAVE08 This should never happen.");
      }

      MSG_DEBUG(std::cout << "DLEAVE57 SPxSolverBase<R>::getLeaveVals2(): col " << idx
                << ": " << enterStat
                << " -> " << ds.colStatus(idx)
                << " objChange: " << objChange
                << std::endl;)
   }

}

template <class R>
void SPxSolverBase<R>::rejectLeave(
   int leaveNum,
   SPxId leaveId,
   typename SPxBasisBase<R>::Desc::Status leaveStat,
   const SVectorBase<R>* //newVec
)
{
   typename SPxBasisBase<R>::Desc& ds = this->desc();

   if(leaveId.isSPxRowId())
   {
      MSG_DEBUG(std::cout << "DLEAVE58 rejectLeave()  : row " << leaveNum
                << ": " << ds.rowStatus(leaveNum)
                << " -> " << leaveStat << std::endl;)

      if(leaveStat == SPxBasisBase<R>::Desc::D_ON_BOTH)
      {
         if(ds.rowStatus(leaveNum) == SPxBasisBase<R>::Desc::P_ON_LOWER)
            theLRbound[leaveNum] = theURbound[leaveNum];
         else
            theURbound[leaveNum] = theLRbound[leaveNum];
      }

      ds.rowStatus(leaveNum) = leaveStat;
   }
   else
   {
      MSG_DEBUG(std::cout << "DLEAVE59 rejectLeave()  : col " << leaveNum
                << ": " << ds.colStatus(leaveNum)
                << " -> " << leaveStat << std::endl;)

      if(leaveStat == SPxBasisBase<R>::Desc::D_ON_BOTH)
      {
         if(ds.colStatus(leaveNum) == SPxBasisBase<R>::Desc::P_ON_UPPER)
            theLCbound[leaveNum] = theUCbound[leaveNum];
         else
            theUCbound[leaveNum] = theLCbound[leaveNum];
      }

      ds.colStatus(leaveNum) = leaveStat;
   }
}


template <class R>
void SPxSolverBase<R>::computePrimalray4Row(R direction)
{
   R sign = (direction > 0 ? 1.0 : -1.0);

   primalRay.clear();
   primalRay.setMax(coPvec().delta().size());

   for(int i = 0; i < coPvec().delta().size(); ++i)
      primalRay.add(coPvec().delta().index(i), sign * coPvec().delta().value(i));
}

template <class R>
void SPxSolverBase<R>::computeDualfarkas4Col(R direction)
{
   R sign = (direction > 0 ? -1.0 : 1.0);

   dualFarkas.clear();
   dualFarkas.setMax(coPvec().delta().size());

   for(int i = 0; i < coPvec().delta().size(); ++i)
      dualFarkas.add(coPvec().delta().index(i), sign * coPvec().delta().value(i));
}

template <class R>
bool SPxSolverBase<R>::leave(int leaveIdx, bool polish)
{
   assert(leaveIdx < dim() && leaveIdx >= 0);
   assert(type() == LEAVE);
   assert(initialized);

   bool instable = instableLeave;
   assert(!instable || instableLeaveNum >= 0);

   /*
     Before performing the actual basis update, we must determine, how this
     is to be accomplished.
     When using steepest edge pricing this solve is already performed by the pricer
   */
   if(theCoPvec->delta().isSetup() && theCoPvec->delta().size() == 0)
   {
      this->coSolve(theCoPvec->delta(), unitVecs[leaveIdx]);
   }

#ifdef ENABLE_ADDITIONAL_CHECKS
   else
   {
      SSVectorBase<R>  tmp(dim(), epsilon());
      tmp.clear();
      this->coSolve(tmp, unitVecs[leaveIdx]);
      tmp -= theCoPvec->delta();

      if(tmp.length() > leavetol())
      {
         // This happens very frequently and does usually not hurt, so print
         // these warnings only with verbose level INFO2 and higher.
         MSG_INFO2((*this->spxout), (*this->spxout) << "WLEAVE60 iteration=" << basis().iteration()
                   << ": coPvec.delta error = " << tmp.length()
                   << std::endl;)
      }
   }

#endif  // ENABLE_ADDITIONAL_CHECKS

   setupPupdate();

   assert(thePvec->isConsistent());
   assert(theCoPvec->isConsistent());

   typename SPxBasisBase<R>::Desc::Status leaveStat;      // status of leaving var
   SPxId leaveId;        // id of leaving var
   SPxId none;           // invalid id used if leave fails
   R leaveMax;       // maximium lambda of leaving var
   R leavebound;     // current fVec value of leaving var
   int  leaveNum;       // number of leaveId in bounds
   StableSum<R> objChange; // amount of change in the objective function

   getLeaveVals(leaveIdx, leaveStat, leaveId, leaveMax, leavebound, leaveNum, objChange);

   if(!polish && m_numCycle > m_maxCycle)
   {
      if(leaveMax > 0)
         perturbMaxLeave();
      else
         perturbMinLeave();

      //@ m_numCycle /= 2;
      // perturbation invalidates the currently stored nonbasic value
      forceRecompNonbasicValue();
   }

   //@ testBounds();

   R enterVal = leaveMax;
   boundflips = 0;
   R oldShift = theShift;
   SPxId enterId = theratiotester->selectEnter(enterVal, leaveIdx, polish);

   if(NE(theShift, oldShift))
   {
      MSG_DEBUG(std::cout << "DLEAVE71 trigger recomputation of nonbasic value due to shifts in ratiotest"
                << std::endl;)
      forceRecompNonbasicValue();
   }

   assert(!enterId.isValid() || !isBasic(enterId));

   instableLeaveNum = -1;
   instableLeave = false;

   /*
     No variable could be selected to enter the basis and even the leaving
     variable is unbounded.
   */
   if(!enterId.isValid())
   {
      /* the following line originally was below in "rejecting leave" case;
         we need it in the unbounded/infeasible case, too, to have the
         correct basis size */
      rejectLeave(leaveNum, leaveId, leaveStat);
      this->change(-1, none, 0);
      objChange = R(0.0); // the nonbasicValue is not supposed to be updated in this case

      if(polish)
         return false;

      if(NE(enterVal, leaveMax))
      {
         MSG_DEBUG(std::cout << "DLEAVE61 rejecting leave A (leaveIdx=" << leaveIdx
                   << ", theCoTest=" << theCoTest[leaveIdx] << ")"
                   << std::endl;)

         /* In the LEAVE algorithm, when for a selected leaving variable we find only
            an instable entering variable, then the basis change is not conducted.
            Instead, we save the leaving variable's index in instableLeaveNum and scale
            theCoTest[leaveIdx] down by some factor, hoping to find a different leaving
            variable with a stable entering variable.
            If this fails, however, and no more leaving variable is found, we have to
            perform the instable basis change using instableLeaveNum. In this (and only
            in this) case, the flag instableLeave is set to true.

            enterVal != leaveMax is the case that selectEnter has found only an instable entering
            variable. We store this leaving variable for later -- if we are not already in the
            instable case: then we continue and conclude unboundedness/infeasibility */
         if(!instable)
         {
            instableLeaveNum = leaveIdx;

            // Note: These changes do not survive a refactorization
            instableLeaveVal = theCoTest[leaveIdx];
            theCoTest[leaveIdx] = instableLeaveVal / 10.0;

            return true;
         }
      }

      if(this->lastUpdate() > 1)
      {
         MSG_INFO3((*this->spxout), (*this->spxout) << "ILEAVE01 factorization triggered in "
                   << "leave() for feasibility test" << std::endl;)

         try
         {
            factorize();
         }
         catch(const SPxStatusException& E)
         {
            // don't exit immediately but handle the singularity correctly
            assert(SPxBasisBase<R>::status() == SPxBasisBase<R>::SINGULAR);
            MSG_INFO3((*this->spxout), (*this->spxout) << "Caught exception in factorization: " << E.what() <<
                      std::endl;)
         }

         /* after a factorization, the leaving column/row might not be infeasible or suboptimal anymore, hence we do
          * not try to call leave(leaveIdx), but rather return to the main solving loop and call the pricer again
          */
         return true;
      }

      /* do not exit with status infeasible or unbounded if there is only a very small violation */
      if(!recomputedVectors && spxAbs(enterVal) < leavetol())
      {
         MSG_INFO3((*this->spxout), (*this->spxout) << "ILEAVE11 clean up step to reduce numerical errors" <<
                   std::endl;)

         computeFrhs();
         SPxBasisBase<R>::solve(*theFvec, *theFrhs);
         computeFtest();

         /* only do this once per solve */
         recomputedVectors = true;

         return true;
      }

      MSG_INFO3((*this->spxout), (*this->spxout) << "ILEAVE02 unboundedness/infeasibility found "
                << "in leave()" << std::endl;)

      if(rep() != COLUMN)
      {
         computePrimalray4Row(enterVal);
         setBasisStatus(SPxBasisBase<R>::UNBOUNDED);
      }
      else
      {
         computeDualfarkas4Col(enterVal);
         setBasisStatus(SPxBasisBase<R>::INFEASIBLE);
      }

      return false;
   }
   else
   {
      /*
        If an entering variable has been found, a regular basis update is to
        be performed.
      */
      if(enterId != this->baseId((leaveIdx)))
      {
         const SVectorBase<R>& newVector = *enterVector(enterId);

         // update feasibility vectors
         if(solveVector2 != NULL && solveVector3 != NULL)
         {
            assert(solveVector2->isConsistent());
            assert(solveVector2rhs->isSetup());
            assert(solveVector3->isConsistent());
            assert(solveVector3rhs->isSetup());
            assert(boundflips > 0);
            SPxBasisBase<R>::solve4update(theFvec->delta(),
                                          *solveVector2,
                                          *solveVector3,
                                          newVector,
                                          *solveVector2rhs,
                                          *solveVector3rhs);

            // perform update of basic solution
            primVec -= (*solveVector3);
            MSG_DEBUG(std::cout << "ILBFRT02 breakpoints passed / bounds flipped = " << boundflips << std::endl;
                     )
            totalboundflips += boundflips;
         }
         else if(solveVector2 != NULL)
         {
            assert(solveVector2->isConsistent());
            assert(solveVector2rhs->isSetup());

            SPxBasisBase<R>::solve4update(theFvec->delta(),
                                          *solveVector2,
                                          newVector,
                                          *solveVector2rhs);
         }
         else if(solveVector3 != NULL)
         {
            assert(solveVector3->isConsistent());
            assert(solveVector3rhs->isSetup());
            assert(boundflips > 0);
            SPxBasisBase<R>::solve4update(theFvec->delta(),
                                          *solveVector3,
                                          newVector,
                                          *solveVector3rhs);

            // perform update of basic solution
            primVec -= (*solveVector3);
            MSG_DEBUG(std::cout << "ILBFRT02 breakpoints passed / bounds flipped = " << boundflips << std::endl;
                     )
            totalboundflips += boundflips;
         }
         else
            SPxBasisBase<R>::solve4update(theFvec->delta(), newVector);

#ifdef ENABLE_ADDITIONAL_CHECKS
         {
            SSVectorBase<R>  tmp(dim(), epsilon());
            SPxBasisBase<R>::solve(tmp, newVector);
            tmp -= fVec().delta();

            if(tmp.length() > entertol())
            {
               // This happens very frequently and does usually not hurt, so print
               // these warnings only with verbose level INFO2 and higher.
               MSG_INFO2((*this->spxout), (*this->spxout) << "WLEAVE62\t(" << tmp.length() << ")\n";)
            }
         }
#endif  // ENABLE_ADDITIONAL_CHECKS


         if(spxAbs(theFvec->delta()[leaveIdx]) < reject_leave_tol)
         {
            if(instable)
            {
               /* We are in the case that for all leaving variables only instable entering
                  variables were found: Thus, above we already accepted such an instable
                  entering variable. Now even this seems to be impossible, thus we conclude
                  unboundedness/infeasibility. */
               MSG_INFO3((*this->spxout), (*this->spxout) << "ILEAVE03 unboundedness/infeasibility found "
                         << "in leave()" << std::endl;)

               rejectLeave(leaveNum, leaveId, leaveStat);
               this->change(-1, none, 0);
               objChange = R(0.0); // the nonbasicValue is not supposed to be updated in this case

               /**@todo if shift() is not zero we must not conclude unboundedness */
               if(rep() == ROW)
               {
                  computePrimalray4Row(enterVal);
                  setBasisStatus(SPxBasisBase<R>::UNBOUNDED);
               }
               else
               {
                  computeDualfarkas4Col(enterVal);
                  setBasisStatus(SPxBasisBase<R>::INFEASIBLE);
               }

               return false;
            }
            else
            {
               theFvec->delta().clear();
               rejectLeave(leaveNum, leaveId, leaveStat, &newVector);
               this->change(-1, none, 0);
               objChange = R(0.0); // the nonbasicValue is not supposed to be updated in this case

               MSG_DEBUG(std::cout << "DLEAVE63 rejecting leave B (leaveIdx=" << leaveIdx
                         << ", theCoTest=" << theCoTest[leaveIdx]
                         << ")" << std::endl;)

               // Note: These changes do not survive a refactorization
               theCoTest[leaveIdx] *= 0.01;

               return true;
            }
         }

         //      process leaving variable
         if(leavebound > epsilon() || leavebound < -epsilon())
            theFrhs->multAdd(-leavebound, this->baseVec(leaveIdx));

         //      process entering variable
         R enterBound;
         R newUBbound;
         R newLBbound;
         R newCoPrhs;

         try
         {
            getLeaveVals2(leaveMax, enterId, enterBound, newUBbound, newLBbound, newCoPrhs, objChange);
         }
         catch(const SPxException& F)
         {
            rejectLeave(leaveNum, leaveId, leaveStat);
            this->change(-1, none, 0);
            objChange = R(0.0); // the nonbasicValue is not supposed to be updated in this case
            throw F;
         }

         theUBbound[leaveIdx] = newUBbound;
         theLBbound[leaveIdx] = newLBbound;
         (*theCoPrhs)[leaveIdx] = newCoPrhs;

         if(enterBound > epsilon() || enterBound < -epsilon())
            theFrhs->multAdd(enterBound, newVector);

         // update pricing vectors
         theCoPvec->value() = enterVal;
         thePvec->value() = enterVal;

         if(enterVal > epsilon() || enterVal < -epsilon())
            doPupdate();

         // update feasibility vector
         theFvec->value() = -((*theFvec)[leaveIdx] - leavebound)
                            / theFvec->delta()[leaveIdx];
         theFvec->update();
         (*theFvec)[leaveIdx] = enterBound - theFvec->value();
         updateFtest();

         // update objective funtion value
         updateNonbasicValue(objChange);

         //  change basis matrix
         this->change(leaveIdx, enterId, &newVector, &(theFvec->delta()));
      }

      /*
        No entering vector has been selected from the basis. However, if the
        shift amount for |coPvec| is bounded, we are in the case, that the
        entering variable is moved from one bound to its other, before any of
        the basis feasibility variables reaches their bound. This may only
        happen in primal/columnwise case with upper and lower bounds on
        variables.
      */
      else
      {
         // @todo update obj function value here!!!
         assert(rep() == ROW);
         typename SPxBasisBase<R>::Desc& ds = this->desc();

         this->change(leaveIdx, none, 0);

         if(leaveStat == SPxBasisBase<R>::Desc::P_ON_UPPER)
         {
            if(leaveId.isSPxRowId())
            {
               ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
               (*theCoPrhs)[leaveIdx] = theLRbound[leaveNum];
            }
            else
            {
               ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_LOWER;
               (*theCoPrhs)[leaveIdx] = theLCbound[leaveNum];
            }

            theUBbound[leaveIdx] = 0;
            theLBbound[leaveIdx] = R(-infinity);
         }
         else
         {
            assert(leaveStat == SPxBasisBase<R>::Desc::P_ON_LOWER);

            if(leaveId.isSPxRowId())
            {
               ds.rowStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
               (*theCoPrhs)[leaveIdx] = theURbound[leaveNum];
            }
            else
            {
               ds.colStatus(leaveNum) = SPxBasisBase<R>::Desc::P_ON_UPPER;
               (*theCoPrhs)[leaveIdx] = theUCbound[leaveNum];
            }

            theUBbound[leaveIdx] = R(infinity);
            theLBbound[leaveIdx] = 0;
         }

         // update copricing vector
         theCoPvec->value() = enterVal;
         thePvec->value() = enterVal;

         if(enterVal > epsilon() || enterVal < -epsilon())
            doPupdate();

         // update feasibility vectors
         theFvec->value() = 0;
         assert(theCoTest[leaveIdx] < 0.0);
         m_pricingViol += theCoTest[leaveIdx];
         theCoTest[leaveIdx] *= -1;
      }

      if((leaveMax > entertol() && enterVal <= entertol()) || (leaveMax < -entertol()
            && enterVal >= -entertol()))
      {
         if((theUBbound[leaveIdx] < R(infinity) || theLBbound[leaveIdx] > R(-infinity))
               && leaveStat != SPxBasisBase<R>::Desc::P_FREE
               && leaveStat != SPxBasisBase<R>::Desc::D_FREE)
         {
            m_numCycle++;
            leaveCycles++;
         }
      }
      else
         m_numCycle /= 2;

#ifdef ENABLE_ADDITIONAL_CHECKS
      {
         VectorBase<R> tmp = fVec();
         this->multBaseWith(tmp);
         tmp -= fRhs();

         if(tmp.length() > entertol())
         {
            // This happens very frequently and does usually not hurt, so print
            // these warnings only with verbose level INFO2 and higher.
            MSG_INFO2((*this->spxout), (*this->spxout) << "WLEAVE64\t" << basis().iteration()
                      << ": fVec error = " << tmp.length() << std::endl;)
            SPxBasisBase<R>::solve(tmp, fRhs());
            tmp -= fVec();
            MSG_INFO2((*this->spxout), (*this->spxout) << "WLEAVE65\t(" << tmp.length() << ")\n";)
         }
      }
#endif  // ENABLE_ADDITIONAL_CHECKS

      return true;
   }
}
} // namespace soplex
