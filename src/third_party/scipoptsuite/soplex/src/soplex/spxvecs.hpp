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
#include "soplex/exceptions.h"

namespace soplex
{
/** Initialize Vectors

    Computing the right hand side vector for the feasibility vector depends on
    the chosen representation and type of the basis.

    In columnwise case, |theFvec| = \f$x_B = A_B^{-1} (- A_N x_N)\f$, where \f$x_N\f$
    are either upper or lower bounds for the nonbasic variables (depending on
    the variables |Status|). If these values remain unchanged throughout the
    simplex algorithm, they may be taken directly from LP. However, in the
    entering type algorith they are changed and, hence, retreived from the
    column or row upper or lower bound vectors.

    In rowwise case, |theFvec| = \f$\pi^T_B = (c^T - 0^T A_N) A_B^{-1}\f$. However,
    this applies only to leaving type algorithm, where no bounds on dual
    variables are altered. In entering type algorithm they are changed and,
    hence, retreived from the column or row upper or lower bound vectors.
*/
template <class R>
void SPxSolverBase<R>::computeFrhs()
{

   if(rep() == COLUMN)
   {
      theFrhs->clear();

      if(type() == LEAVE)
      {
         computeFrhsXtra();

         for(int i = 0; i < this->nRows(); i++)
         {
            R x;

            typename SPxBasisBase<R>::Desc::Status stat = this->desc().rowStatus(i);

            if(!isBasic(stat))
            {
               // coverity[switch_selector_expr_is_constant]
               switch(stat)
               {
               // columnwise cases:
               case SPxBasisBase<R>::Desc::P_FREE :
                  continue;

               case(SPxBasisBase<R>::Desc::P_FIXED) :
                  assert(EQ(this->lhs(i), this->rhs(i)));

               //lint -fallthrough
               case SPxBasisBase<R>::Desc::P_ON_UPPER :
                  x = this->rhs(i);
                  break;

               case SPxBasisBase<R>::Desc::P_ON_LOWER :
                  x = this->lhs(i);
                  break;

               default:
                  MSG_ERROR(std::cerr << "ESVECS01 ERROR: "
                            << "inconsistent basis must not happen!"
                            << std::endl;)
                  throw SPxInternalCodeException("XSVECS01 This should never happen.");
               }

               assert(x < R(infinity));
               assert(x > R(-infinity));
               (*theFrhs)[i] += x;     // slack !
            }
         }
      }
      else
      {
         computeFrhs1(*theUbound, *theLbound);
         computeFrhs2(*theCoUbound, *theCoLbound);
      }
   }
   else
   {
      assert(rep() == ROW);

      if(type() == ENTER)
      {
         theFrhs->clear();
         computeFrhs1(*theUbound, *theLbound);
         computeFrhs2(*theCoUbound, *theCoLbound);
         *theFrhs += this->maxObj();
      }
      else
      {
         ///@todo put this into a separate method
         *theFrhs = this->maxObj();
         const typename SPxBasisBase<R>::Desc& ds = this->desc();

         for(int i = 0; i < this->nRows(); ++i)
         {
            typename SPxBasisBase<R>::Desc::Status stat = ds.rowStatus(i);

            if(!isBasic(stat))
            {
               R x;

               switch(stat)
               {
               case SPxBasisBase<R>::Desc::D_FREE :
                  continue;

               case SPxBasisBase<R>::Desc::D_ON_UPPER :
               case SPxBasisBase<R>::Desc::D_ON_LOWER :
               case(SPxBasisBase<R>::Desc::D_ON_BOTH) :
                  x = this->maxRowObj(i);
                  break;

               default:
                  assert(this->lhs(i) <= R(-infinity) && this->rhs(i) >= R(infinity));
                  x = 0.0;
                  break;
               }

               assert(x < R(infinity));
               assert(x > R(-infinity));
               // assert(x == 0.0);

               if(x != 0.0)
                  theFrhs->multAdd(x, vector(i));
            }
         }
      }
   }
}

template <class R>
void SPxSolverBase<R>::computeFrhsXtra()
{

   assert(rep()  == COLUMN);
   assert(type() == LEAVE);

   for(int i = 0; i < this->nCols(); ++i)
   {
      typename SPxBasisBase<R>::Desc::Status stat = this->desc().colStatus(i);

      if(!isBasic(stat))
      {
         R x;

         // coverity[switch_selector_expr_is_constant]
         switch(stat)
         {
         // columnwise cases:
         case SPxBasisBase<R>::Desc::P_FREE :
            continue;

         case(SPxBasisBase<R>::Desc::P_FIXED) :
            assert(EQ(SPxLPBase<R>::lower(i), SPxLPBase<R>::upper(i)));

         //lint -fallthrough
         case SPxBasisBase<R>::Desc::P_ON_UPPER :
            x = SPxLPBase<R>::upper(i);
            break;

         case SPxBasisBase<R>::Desc::P_ON_LOWER :
            x = SPxLPBase<R>::lower(i);
            break;

         default:
            MSG_ERROR(std::cerr << "ESVECS02 ERROR: "
                      << "inconsistent basis must not happen!"
                      << std::endl;)
            throw SPxInternalCodeException("XSVECS02 This should never happen.");
         }

         assert(x < R(infinity));
         assert(x > R(-infinity));

         if(x != 0.0)
            theFrhs->multAdd(-x, vector(i));
      }
   }
}


/** This methods subtracts \f$A_N x_N\f$ or \f$\pi_N^T A_N\f$ from |theFrhs| as
    specified by the |Status| of all nonbasic variables. The values of \f$x_N\f$ or
    \f$\pi_N\f$ are taken from the passed arrays.
*/
template <class R>
void SPxSolverBase<R>::computeFrhs1(
   const VectorBase<R>& ufb,    ///< upper feasibility bound for variables
   const VectorBase<R>& lfb)    ///< lower feasibility bound for variables
{

   const typename SPxBasisBase<R>::Desc& ds = this->desc();

   for(int i = 0; i < coDim(); ++i)
   {
      typename SPxBasisBase<R>::Desc::Status stat = ds.status(i);

      if(!isBasic(stat))
      {
         R x;

         // coverity[switch_selector_expr_is_constant]
         switch(stat)
         {
         case SPxBasisBase<R>::Desc::D_FREE :
         case SPxBasisBase<R>::Desc::D_UNDEFINED :
         case SPxBasisBase<R>::Desc::P_FREE :
            continue;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
            x = ufb[i];
            break;

         case SPxBasisBase<R>::Desc::P_ON_LOWER :
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
            x = lfb[i];
            break;

         case(SPxBasisBase<R>::Desc::P_FIXED) :
            assert(EQ(lfb[i], ufb[i]));

         //lint -fallthrough
         case(SPxBasisBase<R>::Desc::D_ON_BOTH) :
            x = lfb[i];
            break;

         default:
            MSG_ERROR(std::cerr << "ESVECS03 ERROR: "
                      << "inconsistent basis must not happen!"
                      << std::endl;)
            throw SPxInternalCodeException("XSVECS04 This should never happen.");
         }

         assert(x < R(infinity));
         assert(x > R(-infinity));

         if(x != 0.0)
            theFrhs->multAdd(-x, vector(i));
      }
   }
}

/** This methods subtracts \f$A_N x_N\f$ or \f$\pi_N^T A_N\f$ from |theFrhs| as
    specified by the |Status| of all nonbasic variables. The values of
    \f$x_N\f$ or \f$\pi_N\f$ are taken from the passed arrays.
*/
template <class R>
void SPxSolverBase<R>::computeFrhs2(
   VectorBase<R>& coufb,   ///< upper feasibility bound for covariables
   VectorBase<R>& colfb)   ///< lower feasibility bound for covariables
{
   const typename SPxBasisBase<R>::Desc& ds = this->desc();

   for(int i = 0; i < dim(); ++i)
   {
      typename SPxBasisBase<R>::Desc::Status stat = ds.coStatus(i);

      if(!isBasic(stat))
      {
         R x;

         // coverity[switch_selector_expr_is_constant]
         switch(stat)
         {
         case SPxBasisBase<R>::Desc::D_FREE :
         case SPxBasisBase<R>::Desc::D_UNDEFINED :
         case SPxBasisBase<R>::Desc::P_FREE :
            continue;

         case SPxBasisBase<R>::Desc::P_ON_LOWER :            // negative slack bounds!
         case SPxBasisBase<R>::Desc::D_ON_UPPER :
            x = coufb[i];
            break;

         case SPxBasisBase<R>::Desc::P_ON_UPPER :            // negative slack bounds!
         case SPxBasisBase<R>::Desc::D_ON_LOWER :
            x = colfb[i];
            break;

         case SPxBasisBase<R>::Desc::P_FIXED :
         case SPxBasisBase<R>::Desc::D_ON_BOTH :

            if(colfb[i] != coufb[i])
            {
               MSG_WARNING((*this->spxout), (*this->spxout) << "WSVECS04 Frhs2[" << i << "]: " << static_cast<int>
                           (stat) << " "
                           << colfb[i] << " " << coufb[i]
                           << " shouldn't be" << std::endl;)

               if(isZero(colfb[i]) || isZero(coufb[i]))
                  colfb[i] = coufb[i] = 0.0;
               else
               {
                  R mid = (colfb[i] + coufb[i]) / 2.0;
                  colfb[i] = coufb[i] = mid;
               }
            }

            assert(EQ(colfb[i], coufb[i]));
            x = colfb[i];
            break;

         default:
            MSG_ERROR(std::cerr << "ESVECS05 ERROR: "
                      << "inconsistent basis must not happen!"
                      << std::endl;)
            throw SPxInternalCodeException("XSVECS05 This should never happen.");
         }

         assert(x < R(infinity));
         assert(x > R(-infinity));

         (*theFrhs)[i] -= x; // This is a slack, so no need to multiply a vector.
      }
   }
}

/** Computing the right hand side vector for |theCoPvec| depends on
    the type of the simplex algorithm. In entering algorithms, the
    values are taken from the inequality's right handside or the
    column's objective value.

    In contrast to this leaving algorithms take the values from vectors
    |theURbound| and so on.

    We reflect this difference by providing two pairs of methods
    |computeEnterCoPrhs(n, stat)| and |computeLeaveCoPrhs(n, stat)|. The first
    pair operates for entering algorithms, while the second one is intended for
    leaving algorithms.  The return value of these methods is the right hand
    side value for the \f$n\f$-th row or column id, respectively, if it had the
    passed |Status| for both.

    Both methods are again split up into two methods named |...4Row(i,n)| and
    |...4Col(i,n)|, respectively. They do their job for the |i|-th basis
    variable, being the |n|-th row or column.
*/
template <class R>
void SPxSolverBase<R>::computeEnterCoPrhs4Row(int i, int n)
{
   assert(this->baseId(i).isSPxRowId());
   assert(this->number(SPxRowId(this->baseId(i))) == n);

   switch(this->desc().rowStatus(n))
   {
   // rowwise representation:
   case SPxBasisBase<R>::Desc::P_FIXED :
      assert(this->lhs(n) > R(-infinity));
      assert(EQ(this->rhs(n), this->lhs(n)));

   //lint -fallthrough
   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      assert(rep() == ROW);
      assert(this->rhs(n) < R(infinity));
      (*theCoPrhs)[i] = this->rhs(n);
      break;

   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      assert(rep() == ROW);
      assert(this->lhs(n) > R(-infinity));
      (*theCoPrhs)[i] = this->lhs(n);
      break;

   // columnwise representation:
   // slacks must be left 0!
   default:
      (*theCoPrhs)[i] = this->maxRowObj(n);
      break;
   }
}

template <class R>
void SPxSolverBase<R>::computeEnterCoPrhs4Col(int i, int n)
{
   assert(this->baseId(i).isSPxColId());
   assert(this->number(SPxColId(this->baseId(i))) == n);

   switch(this->desc().colStatus(n))
   {
   // rowwise representation:
   case SPxBasisBase<R>::Desc::P_FIXED :
      assert(EQ(SPxLPBase<R>::upper(n), SPxLPBase<R>::lower(n)));
      assert(SPxLPBase<R>::lower(n) > R(-infinity));

   //lint -fallthrough
   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      assert(rep() == ROW);
      assert(SPxLPBase<R>::upper(n) < R(infinity));
      (*theCoPrhs)[i] = SPxLPBase<R>::upper(n);
      break;

   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      assert(rep() == ROW);
      assert(SPxLPBase<R>::lower(n) > R(-infinity));
      (*theCoPrhs)[i] = SPxLPBase<R>::lower(n);
      break;

   // columnwise representation:
   case SPxBasisBase<R>::Desc::D_UNDEFINED :
   case SPxBasisBase<R>::Desc::D_ON_BOTH :
   case SPxBasisBase<R>::Desc::D_ON_UPPER :
   case SPxBasisBase<R>::Desc::D_ON_LOWER :
   case SPxBasisBase<R>::Desc::D_FREE :
      assert(rep() == COLUMN);
      (*theCoPrhs)[i] = this->maxObj(n);
      break;

   default:             // variable left 0
      (*theCoPrhs)[i] = 0;
      break;
   }
}

template <class R>
void SPxSolverBase<R>::computeEnterCoPrhs()
{
   assert(type() == ENTER);

   for(int i = dim() - 1; i >= 0; --i)
   {
      SPxId l_id = this->baseId(i);

      if(l_id.isSPxRowId())
         computeEnterCoPrhs4Row(i, this->number(SPxRowId(l_id)));
      else
         computeEnterCoPrhs4Col(i, this->number(SPxColId(l_id)));
   }
}

template <class R>
void SPxSolverBase<R>::computeLeaveCoPrhs4Row(int i, int n)
{
   assert(this->baseId(i).isSPxRowId());
   assert(this->number(SPxRowId(this->baseId(i))) == n);

   switch(this->desc().rowStatus(n))
   {
   case SPxBasisBase<R>::Desc::D_ON_BOTH :
   case SPxBasisBase<R>::Desc::P_FIXED :
      assert(theLRbound[n] > R(-infinity));
      assert(EQ(theURbound[n], theLRbound[n]));

   //lint -fallthrough
   case SPxBasisBase<R>::Desc::D_ON_UPPER :
   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      assert(theURbound[n] < R(infinity));
      (*theCoPrhs)[i] = theURbound[n];
      break;

   case SPxBasisBase<R>::Desc::D_ON_LOWER :
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      assert(theLRbound[n] > R(-infinity));
      (*theCoPrhs)[i] = theLRbound[n];
      break;

   default:
      (*theCoPrhs)[i] = this->maxRowObj(n);
      break;
   }
}

template <class R>
void SPxSolverBase<R>::computeLeaveCoPrhs4Col(int i, int n)
{
   assert(this->baseId(i).isSPxColId());
   assert(this->number(SPxColId(this->baseId(i))) == n);

   switch(this->desc().colStatus(n))
   {
   case SPxBasisBase<R>::Desc::D_UNDEFINED :
   case SPxBasisBase<R>::Desc::D_ON_BOTH :
   case SPxBasisBase<R>::Desc::P_FIXED :
      assert(theLCbound[n] > R(-infinity));
      assert(EQ(theUCbound[n], theLCbound[n]));

   //lint -fallthrough
   case SPxBasisBase<R>::Desc::D_ON_LOWER :
   case SPxBasisBase<R>::Desc::P_ON_UPPER :
      assert(theUCbound[n] < R(infinity));
      (*theCoPrhs)[i] = theUCbound[n];
      break;

   case SPxBasisBase<R>::Desc::D_ON_UPPER :
   case SPxBasisBase<R>::Desc::P_ON_LOWER :
      assert(theLCbound[n] > R(-infinity));
      (*theCoPrhs)[i] = theLCbound[n];
      break;

   default:
      (*theCoPrhs)[i] = this->maxObj(n);
      //      (*theCoPrhs)[i] = 0;
      break;
   }
}

template <class R>
void SPxSolverBase<R>::computeLeaveCoPrhs()
{
   assert(type() == LEAVE);

   for(int i = dim() - 1; i >= 0; --i)
   {
      SPxId l_id = this->baseId(i);

      if(l_id.isSPxRowId())
         computeLeaveCoPrhs4Row(i, this->number(SPxRowId(l_id)));
      else
         computeLeaveCoPrhs4Col(i, this->number(SPxColId(l_id)));
   }
}


/** When computing the copricing vector, we expect the pricing vector to be
    setup correctly. Then computing the copricing vector is nothing but
    computing all scalar products of the pricing vector with the vectors of the
    LPs constraint matrix.
*/
template <class R>
void SPxSolverBase<R>::computePvec()
{
   int i;

   for(i = coDim() - 1; i >= 0; --i)
      (*thePvec)[i] = vector(i) * (*theCoPvec);
}

template <class R>
void SPxSolverBase<R>::setupPupdate(void)
{
   SSVectorBase<R>& p = thePvec->delta();
   SSVectorBase<R>& c = theCoPvec->delta();

   if(c.isSetup())
   {
      if(c.size() < 0.95 * theCoPvec->dim())
         p.assign2product4setup(*thecovectors, c,
                                multTimeSparse, multTimeFull,
                                multSparseCalls, multFullCalls);
      else
      {
         multTimeColwise->start();
         p.assign2product(c, *thevectors);
         multTimeColwise->stop();
         ++multColwiseCalls;
      }
   }
   else
   {
      multTimeUnsetup->start();
      p.assign2productAndSetup(*thecovectors, c);
      multTimeUnsetup->stop();
      ++multUnsetupCalls;
   }

   p.setup();
}

template <class R>
void SPxSolverBase<R>::doPupdate(void)
{
   theCoPvec->update();

   if(pricing() == FULL)
   {
      // thePvec->delta().setup();
      thePvec->update();
   }
}
} // namespace soplex
