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
#include "soplex/spxsolver.h"

namespace soplex
{
/** Setting up the feasiblity bound for normal primal variables is
    straightforward. However, slack variables need some more details
    on how we treat them. This is slightly different from usual
    textbook versions. Let \f$l_i \le A_i^T x \le u_i\f$. This can be
    transformed to \f$A_i^Tx + s_i = 0\f$, with \f$-u_i \le s_i \le
    -l_i\f$. Hence, with this definition of slack variables \f$s_i\f$, we
    can directly use vectors \f$l\f$ and \f$u\f$ as feasibility bounds.
*/
template <class R>
void SPxSolverBase<R>::setPrimalBounds()
{

   theUCbound = SPxLPBase<R>::upper();
   theLCbound = SPxLPBase<R>::lower();

   if(rep() == ROW)
   {
      theURbound = this->rhs();
      theLRbound = this->lhs();
   }
   else
   {
      theURbound = this->lhs();
      theLRbound = this->rhs();
      theURbound *= -1.0;
      theLRbound *= -1.0;
   }
}

/** Setting up the basis for dual simplex requires to install upper and lower
    feasibility bounds for dual variables (|Lbound| and |Ubound|). Here is a
    list of how these must be set for inequalities of type \f$l \le a^Tx \le u\f$:

    \f[
    \begin{tabular}{cccc}
    $l$         &       $u$     & |Lbound|      & |Ubound|      \\
    \hline
    $-\infty=l$     & $u=\infty$    &       0       &       0       \\
    $-\infty<l$     & $u=\infty$    &       0       & $\infty$      \\
    $-\infty=l$     & $u<\infty$    & $-\infty$     &       0       \\
    \multicolumn{2}{c}{
    $-\infty<l \ne u<\infty$}       &       0       &       0       \\
    \multicolumn{2}{c}{
    $-\infty<l  =  u<\infty$}       & $-\infty$     & $\infty$      \\
    \end{tabular}
    \f]
    The case \f$l = -\infty\f$, \f$u = \infty\f$ occurs for unbounded primal variables.
    Such must be treated differently from the general case.

    Given possible upper and lower bounds to a dual variable with |Status stat|,
    this function clears the bounds according to |stat| by setting them to
    \f$\infty\f$ or \f$-\infty\f$, respectively.
*/
template <class R>
void SPxSolverBase<R>::clearDualBounds(
   typename SPxBasisBase<R>::Desc::Status stat,
   R&                  upp,
   R&                  lw) const
{

   switch(stat)
   {
   case SPxBasisBase<R>::Desc::P_ON_UPPER + SPxBasisBase<R>::Desc::P_ON_LOWER :
   case SPxBasisBase<R>::Desc::D_FREE :
      upp = R(infinity);
      lw  = R(-infinity);
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER :
   case SPxBasisBase<R>::Desc::D_ON_LOWER :
      upp = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_ON_LOWER :
   case SPxBasisBase<R>::Desc::D_ON_UPPER :
      lw  = R(-infinity);
      break;

   default:
      break;
   }
}

template <class R>
void SPxSolverBase<R>::setDualColBounds()
{

   assert(rep() == COLUMN);

   const typename SPxBasisBase<R>::Desc& ds = this->desc();

   int i;

   for(i = 0; i < this->nRows(); ++i)
   {
      theURbound[i] = this->maxRowObj(i);
      theLRbound[i] = this->maxRowObj(i);

      clearDualBounds(ds.rowStatus(i), theURbound[i], theLRbound[i]);
   }

   for(i = 0; i < this->nCols(); ++i)
   {
      theUCbound[i] = -this->maxObj(i);
      theLCbound[i] = -this->maxObj(i);

      // exchanged ...                 due to definition of slacks!
      clearDualBounds(ds.colStatus(i), theLCbound[i], theUCbound[i]);

      theUCbound[i] *= -1.0;
      theLCbound[i] *= -1.0;
   }
}

template <class R>
void SPxSolverBase<R>::setDualRowBounds()
{

   assert(rep() == ROW);

   int i;

   for(i = 0; i < this->nRows(); ++i)
   {
      theURbound[i] = 0.0;
      theLRbound[i] = 0.0;

      clearDualBounds(this->dualRowStatus(i), theURbound[i], theLRbound[i]);
   }

   for(i = 0; i < this->nCols(); ++i)
   {
      theUCbound[i] = 0.0;
      theLCbound[i] = 0.0;

      clearDualBounds(this->dualColStatus(i), theUCbound[i], theLCbound[i]);
   }
}


/** This sets up the bounds for basic variables for entering simplex algorithms.
    It requires, that all upper lower feasibility bounds have allready been
    setup. Method |setEnterBound4Row(i, n)| does so for the |i|-th basis
    variable being row index |n|. Equivalently, method
    |setEnterBound4Col(i, n)| does so for the |i|-th basis variable being
    column index |n|.
*/
template <class R>
void SPxSolverBase<R>::setEnterBound4Row(int i, int n)
{
   assert(this->baseId(i).isSPxRowId());
   assert(this->number(SPxRowId(this->baseId(i))) == n);

   switch(this->desc().rowStatus(n))
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = theURbound[n];
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      theLBbound[i] = theLRbound[n];
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      theLBbound[i] = R(-infinity);
      theUBbound[i] = R(infinity);
      break;

   default:
      theUBbound[i] = theURbound[n];
      theLBbound[i] = theLRbound[n];
      break;
   }
}

template <class R>
void SPxSolverBase<R>::setEnterBound4Col(int i, int n)
{
   assert(this->baseId(i).isSPxColId());
   assert(this->number(SPxColId(this->baseId(i))) == n);

   switch(this->desc().colStatus(n))
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = theUCbound[n];
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      theLBbound[i] = theLCbound[n];
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      theLBbound[i] = R(-infinity);
      theUBbound[i] = R(infinity);
      break;

   default:
      theUBbound[i] = theUCbound[n];
      theLBbound[i] = theLCbound[n];
      break;
   }
}
template <class R>
void SPxSolverBase<R>::setEnterBounds()
{

   for(int i = 0; i < dim(); ++i)
   {
      SPxId base_id = this->baseId(i);

      if(base_id.isSPxRowId())
         setEnterBound4Row(i, this->number(SPxRowId(base_id)));
      else
         setEnterBound4Col(i, this->number(SPxColId(base_id)));
   }
}


/** This sets up the bounds for basic variables for leaving simplex algorithms.
    While method |setLeaveBound4Row(i,n)| does so for the |i|-th basic variable
    being the |n|-th row, |setLeaveBound4Col(i,n)| does so for the |i|-th basic
    variable being the |n|-th column.
*/
template <class R>
void SPxSolverBase<R>::setLeaveBound4Row(int i, int n)
{
   assert(this->baseId(i).isSPxRowId());
   assert(this->number(SPxRowId(this->baseId(i))) == n);

   switch(this->desc().rowStatus(n))
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = -this->maxRowObj(n);
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      theLBbound[i] = -this->maxRowObj(n);
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER + SPxBasisBase<R>::Desc::P_ON_LOWER :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_FREE :
      theLBbound[i] = -this->maxRowObj(n);
      theUBbound[i] = -this->maxRowObj(n);
      break;

   default:
      assert(rep() == COLUMN);
      theLBbound[i] = -this->rhs(n);                // slacks !
      theUBbound[i] = -this->lhs(n);                // slacks !
      break;
   }
}
template <class R>
void SPxSolverBase<R>::setLeaveBound4Col(int i, int n)
{

   assert(this->baseId(i).isSPxColId());
   assert(this->number(SPxColId(this->baseId(i))) == n);

   switch(this->desc().colStatus(n))
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = 0;
      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      theLBbound[i] = 0;
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_FIXED :
      theLBbound[i] = R(-infinity);
      theUBbound[i] = R(infinity);
      break;

   case SPxBasisBase<R>::Desc::P_FREE :
      theLBbound[i] = theUBbound[i] = 0;
      break;

   default:
      theUBbound[i] = SPxLPBase<R>::upper(n);
      theLBbound[i] = SPxLPBase<R>::lower(n);
      break;
   }
}

template <class R>
void SPxSolverBase<R>::setLeaveBounds()
{

   for(int i = 0; i < dim(); ++i)
   {
      SPxId base_id = this->baseId(i);

      if(base_id.isSPxRowId())
         setLeaveBound4Row(i, this->number(SPxRowId(base_id)));
      else
         setLeaveBound4Col(i, this->number(SPxColId(base_id)));
   }
}

template <class R>
void SPxSolverBase<R>::testBounds() const
{

   if(type() == ENTER)
   {
      R viol_max = (1 + this->iterCount) * entertol();
      int nlinesprinted = 0;
      int m = dim();

      for(int i = 0; i < m; ++i)
      {
         // Minor bound violations happen frequently, so print these
         // warnings only with verbose level INFO2 and higher.
         if((*theFvec)[i] > theUBbound[i] + viol_max)   //@ &&  theUBbound[i] != theLBbound[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND01 Invalid upper enter bound " << i
                      << " Fvec: " << (*theFvec)[i]
                      << " UBbound: " << theUBbound[i]
                      << " tolerance: " << viol_max
                      << " violation: " << (*theFvec)[i] - theUBbound[i] << std::endl;)
            nlinesprinted++;
         }

         if((*theFvec)[i] < theLBbound[i] - viol_max)   //@ &&  theUBbound[i] != theLBbound[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND02 Invalid lower enter bound " << i
                      << " Fvec: " << (*theFvec)[i]
                      << " LBbound: " << theLBbound[i]
                      << " tolerance: " << viol_max
                      << " violation: " << theLBbound[i] - (*theFvec)[i] << std::endl;)
            nlinesprinted++;
         }

         if(nlinesprinted >= 3)
         {
            MSG_INFO2((*this->spxout), (*this->spxout) <<
                      "WBOUND10 suppressing further warnings of type WBOUND{01,02} in this round" << std::endl);
            break;
         }
      }
   }
   else
   {
      R viol_max = (1 + this->iterCount) * leavetol();
      int nlinesprinted = 0;
      int m = dim();
      int n = coDim();

      for(int i = 0; i < m; ++i)
      {
         if((*theCoPvec)[i] > (*theCoUbound)[i] + viol_max)  // && (*theCoUbound)[i] != (*theCoLbound)[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND03 Invalid upper cobound " << i
                      << " CoPvec: " << (*theCoPvec)[i]
                      << " CoUbound: " << (*theCoUbound)[i]
                      << " tolerance: " << viol_max
                      << " violation: " << (*theCoPvec)[i] - (*theCoUbound)[i] << std::endl;)
            nlinesprinted++;
         }

         if((*theCoPvec)[i] < (*theCoLbound)[i] - viol_max)  // && (*theCoUbound)[i] != (*theCoLbound)[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND04 Invalid lower cobound " << i
                      << " CoPvec: " << (*theCoPvec)[i]
                      << " CoLbound: " << (*theCoLbound)[i]
                      << " tolerance: " << viol_max
                      << " violation: " << (*theCoLbound)[i] - (*theCoPvec)[i] << std::endl;)
            nlinesprinted++;
         }

         if(nlinesprinted >= 3)
         {
            MSG_INFO2((*this->spxout), (*this->spxout) <<
                      "WBOUND11 suppressing further warnings of type WBOUND{03,04} in this round" << std::endl);
            break;
         }
      }

      nlinesprinted = 0;

      for(int i = 0; i < n; ++i)
      {
         if((*thePvec)[i] > (*theUbound)[i] + viol_max)   // &&  (*theUbound)[i] != (*theLbound)[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND05 Invalid upper bound " << i
                      << " Pvec: " << (*thePvec)[i]
                      << " Ubound: " << (*theUbound)[i]
                      << " tolerance: " << viol_max
                      << " violation: " << (*thePvec)[i] - (*theUbound)[i] << std::endl;)
            nlinesprinted++;
         }

         if((*thePvec)[i] < (*theLbound)[i] - viol_max)   // &&  (*theUbound)[i] != (*theLbound)[i])
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << "WBOUND06 Invalid lower bound " << i
                      << " Pvec: " << (*thePvec)[i]
                      << " Lbound: " << (*theLbound)[i]
                      << " tolerance: " << viol_max
                      << " violation: " << (*theLbound)[i] - (*thePvec)[i] << std::endl;)
            nlinesprinted++;
         }

         if(nlinesprinted >= 3)
         {
            MSG_INFO2((*this->spxout), (*this->spxout) <<
                      "WBOUND12 suppressing further warnings of type WBOUND{05,06} in this round" << std::endl);
            break;
         }
      }
   }
}
} // namespace soplex
