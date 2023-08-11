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

#include <iostream>

#include "soplex/array.h"
#include "soplex/dataarray.h"
#include "soplex/sorter.h"
#include "soplex/spxout.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <memory>


//rows
#define FREE_LHS_RHS            1
#define FREE_CONSTRAINT         1
#define EMPTY_CONSTRAINT        1
#define ROW_SINGLETON           1
#define AGGREGATE_VARS          1
#define FORCE_CONSTRAINT        1
//cols
#define FREE_BOUNDS             1
#define EMPTY_COLUMN            1
#define FIX_VARIABLE            1
#define FREE_ZERO_OBJ_VARIABLE  1
#define ZERO_OBJ_COL_SINGLETON  1
#define DOUBLETON_EQUATION      1
#define FREE_COL_SINGLETON      1
//dual
#define DOMINATED_COLUMN        1
#define WEAKLY_DOMINATED_COLUMN 1
#define MULTI_AGGREGATE         1
//other
#define TRIVIAL_HEURISTICS      1
#define PSEUDOOBJ               1


#define EXTREMES                1
#define ROWS_SPXMAINSM                    1
#define COLS_SPXMAINSM                    1
#define DUAL_SPXMAINSM                    1
///@todo check: with this simplification step, the unsimplified basis seems to be slightly suboptimal for some instances
#define DUPLICATE_ROWS          1
#define DUPLICATE_COLS          1


#ifndef NDEBUG
#define CHECK_BASIC_DIM
#endif  // NDEBUG

namespace soplex
{

template <class R>
bool SPxMainSM<R>::PostStep::checkBasisDim(DataArray<typename SPxSolverBase<R>::VarStatus> rows,
      DataArray<typename SPxSolverBase<R>::VarStatus> cols) const
{
   int numBasis = 0;

   for(int rs = 0; rs < nRows; ++rs)
   {
      if(rows[rs] == SPxSolverBase<R>::BASIC)
         numBasis++;
   }

   for(int cs = 0; cs < nCols; ++cs)
   {
      if(cols[cs] == SPxSolverBase<R>::BASIC)
         numBasis++;
   }

   assert(numBasis == nRows);
   return numBasis == nRows;
}

template <class R>
void SPxMainSM<R>::RowObjPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
                                     VectorBase<R>&,
                                     DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
                                     DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   s[m_i] = s[m_i] - x[m_j];

   assert(rStatus[m_i] != SPxSolverBase<R>::UNDEFINED);
   assert(cStatus[m_j] != SPxSolverBase<R>::UNDEFINED);
   assert(rStatus[m_i] != SPxSolverBase<R>::BASIC || cStatus[m_j] != SPxSolverBase<R>::BASIC);

   MSG_DEBUG(std::cout << "RowObjPS: removing slack column " << m_j << " (" << cStatus[m_j] <<
             ") for row " << m_i << " (" << rStatus[m_i] << ").\n");

   if(rStatus[m_i] != SPxSolverBase<R>::BASIC)
   {
      switch(cStatus[m_j])
      {
      case SPxSolverBase<R>::ON_UPPER:
         rStatus[m_i] = SPxSolverBase<R>::ON_LOWER;
         break;

      case SPxSolverBase<R>::ON_LOWER:
         rStatus[m_i] = SPxSolverBase<R>::ON_UPPER;
         break;

      default:
         rStatus[m_i] = cStatus[m_j];
      }

      // otherwise checkBasisDim() may fail
      cStatus[m_j] = SPxSolverBase<R>::ZERO;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      assert(false);
      throw SPxInternalCodeException("XMAISM15 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::FreeConstraintPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>&,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // primal:
   R slack = 0.0;

   for(int k = 0; k < m_row.size(); ++k)
      slack += m_row.value(k) * x[m_row.index(k)];

   s[m_i] = slack;

   // dual:
   y[m_i] = m_row_obj;

   // basis:
   rStatus[m_i] = SPxSolverBase<R>::BASIC;

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM15 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::EmptyConstraintPS::execute(VectorBase<R>&, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>&,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // primal:
   s[m_i] = 0.0;

   // dual:
   y[m_i] = m_row_obj;

   // basis:
   rStatus[m_i] = SPxSolverBase<R>::BASIC;

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM16 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::RowSingletonPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      y[m_old_i] = y[m_i];
      s[m_old_i] = s[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   R aij = m_col[m_i];
   assert(aij != 0.0);

   // primal:
   s[m_i] = aij * x[m_j];

   // dual & basis:
   R val = m_obj;

   for(int k = 0; k < m_col.size(); ++k)
   {
      if(m_col.index(k) != m_i)
         val -= m_col.value(k) * y[m_col.index(k)];
   }

   R newLo = (aij > 0) ? m_lhs / aij : m_rhs / aij; // implicit lhs
   R newUp = (aij > 0) ? m_rhs / aij : m_lhs / aij; // implicit rhs

   switch(cStatus[m_j])
   {
   case SPxSolverBase<R>::FIXED:
      if(newLo <= m_oldLo && newUp >= m_oldUp)
      {
         // this row is totally redundant, has not changed bound of xj
         rStatus[m_i] = SPxSolverBase<R>::BASIC;
         y[m_i] = m_row_obj;
      }
      else if(EQrel(newLo, newUp, this->eps()))
      {
         // row is in the type  aij * xj = b
         assert(EQrel(newLo, x[m_j], this->eps()));

         if(EQrel(m_oldLo, m_oldUp, this->eps()))
         {
            // xj has been fixed in other row
            rStatus[m_i] = SPxSolverBase<R>::BASIC;
            y[m_i] = m_row_obj;
         }
         else if((EQrel(m_oldLo, x[m_j], this->eps()) && r[m_j] <= -this->eps())
                 || (EQrel(m_oldUp, x[m_j], this->eps()) && r[m_j] >= this->eps())
                 || (!EQrel(m_oldLo, x[m_j], this->eps()) && !(EQrel(m_oldUp, x[m_j], this->eps()))))
         {
            // if x_j on lower but reduced cost is negative, or x_j on upper but reduced cost is positive, or x_j not on bound: basic
            rStatus[m_i] = (EQrel(m_lhs, x[m_j] * aij,
                                  this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            y[m_i] = val / aij;
            r[m_j] = 0.0;
         }
         else
         {
            // set x_j on one of the bound
            assert(EQrel(m_oldLo, x[m_j], this->eps()) || EQrel(m_oldUp, x[m_j], this->eps()));

            cStatus[m_j] = EQrel(m_oldLo, x[m_j],
                                 this->eps()) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
            rStatus[m_i] = SPxSolverBase<R>::BASIC;
            y[m_i] = m_row_obj;
            r[m_j] = val;
         }
      }
      else if(EQrel(newLo, m_oldUp, this->eps()))
      {
         // row is in the type  xj >= b/aij, try to set xj on upper
         if(r[m_j] >= this->eps())
         {
            // the reduced cost is positive, xj should in the basic
            assert(EQrel(m_rhs / aij, x[m_j], this->eps()) || EQrel(m_lhs / aij, x[m_j], this->eps()));

            rStatus[m_i] = (EQrel(m_lhs / aij, x[m_j],
                                  this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            y[m_i] = val / aij;
            r[m_j] = 0.0;
         }
         else
         {
            assert(EQrel(m_oldUp, x[m_j], this->eps()));

            cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
            rStatus[m_i] = SPxSolverBase<R>::BASIC;
            y[m_i] = m_row_obj;
            r[m_j] = val;
         }
      }
      else if(EQrel(newUp, m_oldLo, this->eps()))
      {
         // row is in the type  xj <= b/aij, try to set xj on lower
         if(r[m_j] <= -this->eps())
         {
            // the reduced cost is negative, xj should in the basic
            assert(EQrel(m_rhs / aij, x[m_j], this->eps()) || EQrel(m_lhs / aij, x[m_j], this->eps()));

            rStatus[m_i] = (EQrel(m_lhs / aij, x[m_j],
                                  this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            y[m_i] = val / aij;
            r[m_j] = 0.0;
         }
         else
         {
            assert(EQrel(m_oldLo, x[m_j], this->eps()));

            cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
            rStatus[m_i] = SPxSolverBase<R>::BASIC;
            y[m_i] = m_row_obj;
            r[m_j] = val;
         }
      }
      else
      {
         // the variable is set to FIXED by other constraints, i.e., this singleton row is redundant
         rStatus[m_i] = SPxSolverBase<R>::BASIC;
         y[m_i] = m_row_obj;
      }

      break;

   case SPxSolverBase<R>::BASIC:
      rStatus[m_i] = SPxSolverBase<R>::BASIC;
      y[m_i] = m_row_obj;
      r[m_j] = 0.0;
      break;

   case SPxSolverBase<R>::ON_LOWER:
      if(EQrel(m_oldLo, x[m_j], this->eps())) // xj may stay on lower
      {
         rStatus[m_i] = SPxSolverBase<R>::BASIC;
         y[m_i] = m_row_obj;
         r[m_j] = val;
      }
      else // if reduced costs are negative or old lower bound not equal to xj, we need to change xj into the basis
      {
         assert(!isOptimal || EQrel(m_rhs / aij, x[m_j], this->eps())
                || EQrel(m_lhs / aij, x[m_j], this->eps()));

         cStatus[m_j] = SPxSolverBase<R>::BASIC;
         rStatus[m_i] = (EQrel(m_lhs / aij, x[m_j],
                               this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
         y[m_i] = val / aij;
         r[m_j] = 0.0;
      }

      break;

   case SPxSolverBase<R>::ON_UPPER:
      if(EQrel(m_oldUp, x[m_j], this->eps())) // xj may stay on upper
      {
         rStatus[m_i] = SPxSolverBase<R>::BASIC;
         y[m_i] = m_row_obj;
         r[m_j] = val;
      }
      else // if reduced costs are positive or old upper bound not equal to xj, we need to change xj into the basis
      {
         assert(!isOptimal || EQrel(m_rhs / aij, x[m_j], this->eps())
                || EQrel(m_lhs / aij, x[m_j], this->eps()));

         cStatus[m_j] = SPxSolverBase<R>::BASIC;
         rStatus[m_i] = (EQrel(m_lhs / aij, x[m_j],
                               this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;
         y[m_i] = val / aij;
         r[m_j] = 0.0;
      }

      break;

   case SPxSolverBase<R>::ZERO:
      rStatus[m_i] = SPxSolverBase<R>::BASIC;
      y[m_i] = m_row_obj;
      r[m_j] = val;
      break;

   default:
      break;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM17 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::ForceConstraintPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // primal:
   s[m_i] = m_lRhs;

   // basis:
   int cBasisCandidate = -1;
   R maxViolation = -1.0;
   int bas_k = -1;

   for(int k = 0; k < m_row.size(); ++k)
   {
      int  cIdx  = m_row.index(k);
      R aij   = m_row.value(k);
      R oldLo = m_oldLowers[k];
      R oldUp = m_oldUppers[k];

      switch(cStatus[cIdx])
      {
      case SPxSolverBase<R>::FIXED:
         if(m_fixed[k])
         {
            assert(EQrel(oldLo, x[cIdx], this->eps()) || EQrel(oldUp, x[cIdx], this->eps()));

            R violation = spxAbs(r[cIdx] / aij);

            cStatus[cIdx] = EQrel(oldLo, x[cIdx],
                                  this->eps()) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;

            if(violation > maxViolation && ((EQrel(oldLo, x[cIdx], this->eps()) && r[cIdx] < -this->eps())
                                            || (EQrel(oldUp, x[cIdx], this->eps()) && r[cIdx] > this->eps())))
            {
               maxViolation = violation;
               cBasisCandidate = cIdx;
               bas_k = k;
            }
         } // do nothing, if the old bounds are equal, i.e. variable has been not fixed in this row

         break;

      case SPxSolverBase<R>::ON_LOWER:
      case SPxSolverBase<R>::ON_UPPER:
      case SPxSolverBase<R>::BASIC:
         break;

      default:
         break;
      }
   }

   // dual and basis :
   if(cBasisCandidate >= 0)  // one of the variable in the row should in the basis
   {
      assert(EQrel(m_lRhs, m_rhs, this->eps()) || EQrel(m_lRhs, m_lhs, this->eps()));
      assert(bas_k >= 0);
      assert(cBasisCandidate == m_row.index(bas_k));

      cStatus[cBasisCandidate] = SPxSolverBase<R>::BASIC;
      rStatus[m_i] = (EQrel(m_lRhs, m_lhs,
                            this->eps())) ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER;

      R aij = m_row.value(bas_k);
      R multiplier = r[cBasisCandidate] / aij;
      r[cBasisCandidate] = 0.0;

      for(int k = 0; k < m_row.size(); ++k)  // update the reduced cost
      {
         if(k == bas_k)
         {
            continue;
         }

         r[m_row.index(k)] -= m_row.value(k) * multiplier;
      }

      // compute the value of new dual variable (because we have a new row)
      R val = m_objs[bas_k];
      DSVectorBase<R> basis_col = m_cols[bas_k];

      for(int k = 0; k < basis_col.size(); ++k)
      {
         if(basis_col.index(k) != m_i)
            val -= basis_col.value(k) * y[basis_col.index(k)];
      }

      y[m_i] = val / aij;
   }
   else // slack in the basis
   {
      rStatus[m_i] = SPxSolverBase<R>::BASIC;
      y[m_i] = m_rowobj;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM18 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::FixVariablePS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // update the index mapping; if m_correctIdx is false, we assume that this has happened already
   if(m_correctIdx)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   // primal:
   x[m_j] = m_val;

   for(int k = 0; k < m_col.size(); ++k)
      s[m_col.index(k)] += m_col.value(k) * x[m_j];

   // dual:
   R val = m_obj;

   for(int k = 0; k < m_col.size(); ++k)
      val -= m_col.value(k) * y[m_col.index(k)];

   r[m_j] = val;

   // basis:
   if(m_lower == m_upper)
   {
      assert(EQrel(m_lower, m_val));

      cStatus[m_j] = SPxSolverBase<R>::FIXED;
   }
   else
   {
      assert(EQrel(m_val, m_lower) || EQrel(m_val, m_upper) || m_val == 0.0);

      cStatus[m_j] = EQrel(m_val, m_lower) ? SPxSolverBase<R>::ON_LOWER : (EQrel(m_val,
                     m_upper) ? SPxSolverBase<R>::ON_UPPER : SPxSolverBase<R>::ZERO);
   }

#ifdef CHECK_BASIC_DIM

   if(m_correctIdx)
   {
      if(!this->checkBasisDim(rStatus, cStatus))
      {
         throw SPxInternalCodeException("XMAISM19 Dimension doesn't match after this step.");
      }
   }

#endif
}

template <class R>
void SPxMainSM<R>::FixBoundsPS::execute(VectorBase<R>&, VectorBase<R>&, VectorBase<R>&,
                                        VectorBase<R>&,
                                        DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
                                        DataArray<typename SPxSolverBase<R>::VarStatus>&, bool isOptimal) const
{
   // basis:
   cStatus[m_j] = m_status;
}

template <class R>
void SPxMainSM<R>::FreeZeroObjVariablePS::execute(VectorBase<R>& x, VectorBase<R>& y,
      VectorBase<R>& s, VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the column and corresponding rows:
   if(m_j != m_old_j)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   int rIdx = m_old_i - m_col.size() + 1;

   for(int k = 0; k < m_col.size(); ++k)
   {
      int rIdx_new = m_col.index(k);
      s[rIdx] = s[rIdx_new];
      y[rIdx] = y[rIdx_new];
      rStatus[rIdx] = rStatus[rIdx_new];
      rIdx++;
   }

   // primal:
   int      domIdx = -1;
   DSVectorBase<R> slack(m_col.size());

   if(m_loFree)
   {
      R minRowUp = R(infinity);

      for(int k = 0; k < m_rows.size(); ++k)
      {
         R           val = 0.0;
         const SVectorBase<R>& row = m_rows[k];

         for(int l = 0; l < row.size(); ++l)
         {
            if(row.index(l) != m_j)
               val += row.value(l) * x[row.index(l)];
         }

         R scale = maxAbs(m_lRhs[k], val);

         if(scale < 1.0)
            scale = 1.0;

         R z = (m_lRhs[k] / scale) - (val / scale);

         if(isZero(z))
            z = 0.0;

         R up = z * scale / row[m_j];
         slack.add(k, val);

         if(up < minRowUp)
         {
            minRowUp = up;
            domIdx   = k;
         }
      }

      if(m_bnd < minRowUp)
      {
         x[m_j] = m_bnd;
         domIdx = -1;
      }
      else
         x[m_j] = minRowUp;
   }
   else
   {
      R maxRowLo = R(-infinity);

      for(int k = 0; k < m_rows.size(); ++k)
      {
         R val = 0.0;
         const SVectorBase<R>& row = m_rows[k];

         for(int l = 0; l < row.size(); ++l)
         {
            if(row.index(l) != m_j)
               val += row.value(l) * x[row.index(l)];
         }

         R scale = maxAbs(m_lRhs[k], val);

         if(scale < 1.0)
            scale = 1.0;

         R z = (m_lRhs[k] / scale) - (val / scale);

         if(isZero(z))
            z = 0.0;

         R lo = z * scale / row[m_j];
         slack.add(k, val);

         if(lo > maxRowLo)
         {
            maxRowLo = lo;
            domIdx   = k;
         }
      }

      if(m_bnd > maxRowLo)
      {
         x[m_j] = m_bnd;
         domIdx = -1;
      }
      else
         x[m_j] = maxRowLo;
   }

   for(int k = 0; k < m_col.size(); ++k)
      s[m_col.index(k)] = slack[k] + m_col.value(k) * x[m_j];

   // dual:
   r[m_j] = 0.0;

   for(int k = 0; k < m_col.size(); ++k)
   {
      int idx = m_col.index(k);
      y[idx] = m_rowObj[idx];
   }

   // basis:
   for(int k = 0; k < m_col.size(); ++k)
   {
      if(k != domIdx)
         rStatus[m_col.index(k)] = SPxSolverBase<R>::BASIC;

      else
      {
         cStatus[m_j] = SPxSolverBase<R>::BASIC;

         if(m_loFree)
            rStatus[m_col.index(k)] = (m_col.value(k) > 0) ? SPxSolverBase<R>::ON_UPPER :
                                      SPxSolverBase<R>::ON_LOWER;
         else
            rStatus[m_col.index(k)] = (m_col.value(k) > 0) ? SPxSolverBase<R>::ON_LOWER :
                                      SPxSolverBase<R>::ON_UPPER;
      }
   }

   if(domIdx == -1)
   {
      if(m_loFree)
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      else
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM20 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::ZeroObjColSingletonPS::execute(VectorBase<R>& x, VectorBase<R>& y,
      VectorBase<R>& s, VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the column and corresponding rows:
   if(m_j != m_old_j)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   // primal & basis:
   R aij = m_row[m_j];

   if(isZero(s[m_i], R(1e-6)))
      s[m_i] = 0.0;
   else if(s[m_i] >= R(infinity))
      // this is a fix for a highly ill conditioned instance that is "solved" in presolving (ilaser0 from MINLP, mittelmann)
      throw SPxException("Simplifier: infinite activities - aborting unsimplification");

   R scale1 = maxAbs(m_lhs, s[m_i]);
   R scale2 = maxAbs(m_rhs, s[m_i]);

   if(scale1 < 1.0)
      scale1 = 1.0;

   if(scale2 < 1.0)
      scale2 = 1.0;

   R z1 = (m_lhs / scale1) - (s[m_i] / scale1);
   R z2 = (m_rhs / scale2) - (s[m_i] / scale2);

   if(isZero(z1))
      z1 = 0.0;

   if(isZero(z2))
      z2 = 0.0;

   R lo = (aij > 0) ? z1 * scale1 / aij : z2 * scale2 / aij;
   R up = (aij > 0) ? z2 * scale2 / aij : z1 * scale1 / aij;

   if(isZero(lo, this->eps()))
      lo = 0.0;

   if(isZero(up, this->eps()))
      up = 0.0;

   assert(LErel(lo, up));
   ASSERT_WARN("WMAISM01", isNotZero(aij, R(1.0 / R(infinity))));

   if(rStatus[m_i] == SPxSolverBase<R>::ON_LOWER)
   {
      if(m_lower <= R(-infinity) && m_upper >= R(infinity))
      {
         x[m_j] = 0.0;
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
      }
      else if(m_lower == m_upper)
      {
         x[m_j]       = m_lower;
         cStatus[m_j] = SPxSolverBase<R>::FIXED;
      }
      else if(aij > 0)
      {
         x[m_j]       = m_upper;
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      }
      else if(aij < 0)
      {
         x[m_j]       = m_lower;
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
      }
      else
         throw SPxInternalCodeException("XMAISM01 This should never happen.");
   }
   else if(rStatus[m_i] == SPxSolverBase<R>::ON_UPPER)
   {
      if(m_lower <= R(-infinity) && m_upper >= R(infinity))
      {
         x[m_j] = 0.0;
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
      }
      else if(m_lower == m_upper)
      {
         x[m_j]       = m_lower;
         cStatus[m_j] = SPxSolverBase<R>::FIXED;
      }
      else if(aij > 0)
      {
         x[m_j]       = m_lower;
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
      }
      else if(aij < 0)
      {
         x[m_j]       = m_upper;
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      }
      else
         throw SPxInternalCodeException("XMAISM02 This should never happen.");
   }
   else if(rStatus[m_i] == SPxSolverBase<R>::FIXED)
   {
      if(m_lower <= R(-infinity) && m_upper >= R(infinity))
      {
         x[m_j] = 0.0;
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
      }
      else
      {
         assert(EQrel(m_lower, m_upper, this->eps()));

         x[m_j]        = (m_lower + m_upper) / 2.0;
         cStatus[m_j]  = SPxSolverBase<R>::FIXED;
      }
   }
   else if(rStatus[m_i] == SPxSolverBase<R>::BASIC)
   {
      if(GErel(m_lower, lo, this->eps()) && m_lower > R(-infinity))
      {
         x[m_j]       = m_lower;
         cStatus[m_j] = (m_lower == m_upper) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
      }
      else if(LErel(m_upper, up, this->eps()) && m_upper < R(infinity))
      {
         x[m_j]       = m_upper;
         cStatus[m_j] = (m_lower == m_upper) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
      }
      else if(lo > R(-infinity))
      {
         // make m_i non-basic and m_j basic
         x[m_j]       = lo;
         cStatus[m_j] = SPxSolverBase<R>::BASIC;
         rStatus[m_i] = (aij > 0 ? SPxSolverBase<R>::ON_LOWER : SPxSolverBase<R>::ON_UPPER);
      }
      else if(up < R(infinity))
      {
         // make m_i non-basic and m_j basic
         x[m_j]       = up;
         cStatus[m_j] = SPxSolverBase<R>::BASIC;
         rStatus[m_i] = (aij > 0 ? SPxSolverBase<R>::ON_UPPER : SPxSolverBase<R>::ON_LOWER);
      }
      else
         throw SPxInternalCodeException("XMAISM03 This should never happen.");
   }
   else
      throw SPxInternalCodeException("XMAISM04 This should never happen.");

   s[m_i] += aij * x[m_j];

   // dual:
   r[m_j] = -1.0 * aij * y[m_i];

   assert(!isOptimal || (cStatus[m_j] != SPxSolverBase<R>::BASIC || isZero(r[m_j], this->eps())));

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM21 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::FreeColSingletonPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{

   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // correcting the change of idx by deletion of the column:
   if(m_j != m_old_j)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   // primal:
   R val = 0.0;
   R aij = m_row[m_j];

   for(int k = 0; k < m_row.size(); ++k)
   {
      if(m_row.index(k) != m_j)
         val += m_row.value(k) * x[m_row.index(k)];
   }

   R scale = maxAbs(m_lRhs, val);

   if(scale < 1.0)
      scale = 1.0;

   R z = (m_lRhs / scale) - (val / scale);

   if(isZero(z))
      z = 0.0;

   x[m_j] = z * scale / aij;
   s[m_i] = m_lRhs;

   // dual:
   y[m_i] = m_obj / aij;
   r[m_j] = 0.0;

   // basis:
   cStatus[m_j] = SPxSolverBase<R>::BASIC;

   if(m_eqCons)
      rStatus[m_i] = SPxSolverBase<R>::FIXED;
   else if(m_onLhs)
      rStatus[m_i] = SPxSolverBase<R>::ON_LOWER;
   else
      rStatus[m_i] = SPxSolverBase<R>::ON_UPPER;

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM22 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::DoubletonEquationPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>&,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // dual:
   if((cStatus[m_k]  != SPxSolverBase<R>::BASIC) &&
         ((cStatus[m_k] == SPxSolverBase<R>::ON_LOWER && m_strictLo) ||
          (cStatus[m_k] == SPxSolverBase<R>::ON_UPPER && m_strictUp) ||
          (cStatus[m_k] == SPxSolverBase<R>::FIXED    &&
           ((m_maxSense && ((r[m_j] > 0 && m_strictUp) || (r[m_j] < 0 && m_strictLo))) ||
            (!m_maxSense && ((r[m_j] > 0 && m_strictLo) || (r[m_j] < 0 && m_strictUp)))))))
   {
      R val  = m_kObj;
      R aik  = m_col[m_i];

      for(int _k = 0; _k < m_col.size(); ++_k)
      {
         if(m_col.index(_k) != m_i)
            val -= m_col.value(_k) * y[m_col.index(_k)];
      }

      y[m_i] = val / aik;
      r[m_k] = 0.0;

      r[m_j] = m_jObj - val * m_aij / aik;

      ASSERT_WARN("WMAISM73", isNotZero(m_aij * aik));

      // basis:
      if(m_jFixed)
         cStatus[m_j] = SPxSolverBase<R>::FIXED;
      else
      {
         if(GT(r[m_j], (R) 0) || (isZero(r[m_j]) && EQ(x[m_j], m_Lo_j)))
            cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
         else
            cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      }

      cStatus[m_k] = SPxSolverBase<R>::BASIC;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM23 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::DuplicateRowsPS::execute(VectorBase<R>&, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>&,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the duplicated rows:
   if(m_isLast)
   {
      for(int i = m_perm.size() - 1; i >= 0; --i)
      {
         if(m_perm[i] >= 0)
         {
            int rIdx_new = m_perm[i];
            int rIdx = i;
            s[rIdx] = s[rIdx_new];
            y[rIdx] = y[rIdx_new];
            rStatus[rIdx] = rStatus[rIdx_new];
         }
      }
   }

   // primal:
   for(int k = 0; k < m_scale.size(); ++k)
   {
      if(m_scale.index(k) != m_i)
         s[m_scale.index(k)] = s[m_i] / m_scale.value(k);
   }

   // dual & basis:
   bool haveSetBasis = false;

   for(int k = 0; k < m_scale.size(); ++k)
   {
      int i = m_scale.index(k);

      if(rStatus[m_i] == SPxSolverBase<R>::BASIC || (haveSetBasis && i != m_i))
         // if the row with tightest lower and upper bound in the basic, every duplicate row should in basic
         // or basis status of row m_i has been set, this row should be in basis
      {
         y[i]       = m_rowObj.value(k);
         rStatus[i] = SPxSolverBase<R>::BASIC;
         continue;
      }

      ASSERT_WARN("WMAISM02", isNotZero(m_scale.value(k)));

      if(rStatus[m_i] == SPxSolverBase<R>::FIXED && (i == m_maxLhsIdx || i == m_minRhsIdx))
      {
         // this row leads to the tightest lower or upper bound, slack should not be in the basis
         y[i]   = y[m_i] * m_scale.value(k);
         y[m_i] = m_i_rowObj;

         if(m_isLhsEqualRhs[k])
         {
            rStatus[i] = SPxSolverBase<R>::FIXED;
         }
         else if(i == m_maxLhsIdx)
         {
            rStatus[i] = m_scale.value(k) * m_scale.value(0) > 0 ? SPxSolverBase<R>::ON_LOWER :
                         SPxSolverBase<R>::ON_UPPER;
         }
         else
         {
            assert(i == m_minRhsIdx);

            rStatus[i] = m_scale.value(k) * m_scale.value(0) > 0 ? SPxSolverBase<R>::ON_UPPER :
                         SPxSolverBase<R>::ON_LOWER;
         }

         if(i != m_i)
            rStatus[m_i] = SPxSolverBase<R>::BASIC;

         haveSetBasis = true;
      }
      else if(i == m_maxLhsIdx && rStatus[m_i] == SPxSolverBase<R>::ON_LOWER)
      {
         // this row leads to the tightest lower bound, slack should not be in the basis
         y[i]   = y[m_i] * m_scale.value(k);
         y[m_i] = m_i_rowObj;

         rStatus[i] = m_scale.value(k) * m_scale.value(0) > 0 ? SPxSolverBase<R>::ON_LOWER :
                      SPxSolverBase<R>::ON_UPPER;

         if(i != m_i)
            rStatus[m_i] = SPxSolverBase<R>::BASIC;

         haveSetBasis = true;
      }
      else if(i == m_minRhsIdx && rStatus[m_i] == SPxSolverBase<R>::ON_UPPER)
      {
         // this row leads to the tightest upper bound, slack should not be in the basis
         y[i]   = y[m_i] * m_scale.value(k);
         y[m_i] = m_i_rowObj;

         rStatus[i] = m_scale.value(k) * m_scale.value(0) > 0 ? SPxSolverBase<R>::ON_UPPER :
                      SPxSolverBase<R>::ON_LOWER;

         if(i != m_i)
            rStatus[m_i] = SPxSolverBase<R>::BASIC;

         haveSetBasis = true;
      }
      else if(i != m_i)
      {
         // this row does not lead to the tightest lower or upper bound, slack should be in the basis
         y[i]       = m_rowObj.value(k);
         rStatus[i] = SPxSolverBase<R>::BASIC;
      }
   }

#ifdef CHECK_BASIC_DIM

   if(m_isFirst && !this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM24 Dimension doesn't match after this step.");
   }

#endif

   // nothing to do for the reduced cost values
}

template <class R>
void SPxMainSM<R>::DuplicateColsPS::execute(VectorBase<R>& x,
      VectorBase<R>&,
      VectorBase<R>&,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{

   if(m_isFirst)
   {
#ifdef CHECK_BASIC_DIM

      if(!this->checkBasisDim(rStatus, cStatus))
      {
         throw SPxInternalCodeException("XMAISM25 Dimension doesn't match after this step.");
      }

#endif
      return;
   }


   // correcting the change of idx by deletion of the columns:
   if(m_isLast)
   {
      for(int i = m_perm.size() - 1; i >= 0; --i)
      {
         if(m_perm[i] >= 0)
         {
            int cIdx_new = m_perm[i];
            int cIdx = i;
            x[cIdx] = x[cIdx_new];
            r[cIdx] = r[cIdx_new];
            cStatus[cIdx] = cStatus[cIdx_new];
         }
      }

      return;
   }

   // primal & basis:
   ASSERT_WARN("WMAISM03", isNotZero(m_scale));

   if(cStatus[m_k] == SPxSolverBase<R>::ON_LOWER)
   {
      x[m_k] = m_loK;

      if(m_scale > 0)
      {
         x[m_j]       = m_loJ;
         cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
      }
      else
      {
         x[m_j]       = m_upJ;
         cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
      }
   }
   else if(cStatus[m_k] == SPxSolverBase<R>::ON_UPPER)
   {
      x[m_k] = m_upK;

      if(m_scale > 0)
      {
         x[m_j]       = m_upJ;
         cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
      }
      else
      {
         x[m_j]       = m_loJ;
         cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
      }
   }
   else if(cStatus[m_k] == SPxSolverBase<R>::FIXED)
   {
      // => x[m_k] and x[m_j] are also fixed before the corresponding preprocessing step
      x[m_j]       = m_loJ;
      cStatus[m_j] = SPxSolverBase<R>::FIXED;
   }
   else if(cStatus[m_k] == SPxSolverBase<R>::ZERO)
   {
      /* we only aggregate duplicate columns if 0 is contained in their bounds, so we can handle this case properly */
      assert(isZero(x[m_k]));
      assert(LErel(m_loJ, R(0.0)));
      assert(GErel(m_upJ, R(0.0)));
      assert(LErel(m_loK, R(0.0)));
      assert(GErel(m_upK, R(0.0)));

      if(isZero(m_loK) && isZero(m_upK) && m_loK == m_upK)
         cStatus[m_k] = SPxSolverBase<R>::FIXED;
      else if(isZero(m_loK))
         cStatus[m_k] = SPxSolverBase<R>::ON_LOWER;
      else if(isZero(m_upK))
         cStatus[m_k] = SPxSolverBase<R>::ON_UPPER;
      else if(LErel(m_loK, R(0.0)) && GErel(m_upK, R(0.0)))
         cStatus[m_k] = SPxSolverBase<R>::ZERO;
      else
         throw SPxInternalCodeException("XMAISM05 This should never happen.");

      x[m_j] = 0.0;

      if(isZero(m_loJ) && isZero(m_upJ) && m_loJ == m_upJ)
         cStatus[m_j] = SPxSolverBase<R>::FIXED;
      else if(isZero(m_loJ))
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
      else if(isZero(m_upJ))
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      else if(LErel(m_loJ, R(0.0)) && GErel(m_upJ, R(0.0)))
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
      else
         throw SPxInternalCodeException("XMAISM06 This should never happen.");
   }
   else if(cStatus[m_k] == SPxSolverBase<R>::BASIC)
   {
      R scale1 = maxAbs(x[m_k], m_loK);
      R scale2 = maxAbs(x[m_k], m_upK);

      if(scale1 < 1.0)
         scale1 = 1.0;

      if(scale2 < 1.0)
         scale2 = 1.0;

      R z1 = (x[m_k] / scale1) - (m_loK / scale1);
      R z2 = (x[m_k] / scale2) - (m_upK / scale2);

      if(isZero(z1))
         z1 = 0.0;

      if(isZero(z2))
         z2 = 0.0;

      if(m_loJ <= R(-infinity) && m_upJ >= R(infinity) && m_loK <= R(-infinity) && m_upK >= R(infinity))
      {
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
         x[m_j] = 0.0;
      }
      else if(m_scale > 0.0)
      {
         if(GErel(x[m_k], m_upK + m_scale * m_upJ))
         {
            assert(m_upJ < R(infinity));
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_j] = m_upJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_upJ) && m_upJ < R(infinity))
         {
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_j] = m_upJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_upK + m_scale * m_loJ) && m_upK < R(infinity))
         {
            cStatus[m_k] = (m_loK == m_upK) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_k] = m_upK;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            x[m_j] = z2 * scale2 / m_scale;
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_loJ) && m_loJ > R(-infinity))
         {
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_j] = m_loJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_loJ) && m_loK > R(-infinity))
         {
            cStatus[m_k] = (m_loK == m_upK) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_k] = m_loK;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            x[m_j] = z1 * scale1 / m_scale;
         }
         else if(LTrel(x[m_k], m_loK + m_scale * m_loJ))
         {
            assert(m_loJ > R(-infinity));
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_j] = m_loJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else
         {
            throw SPxInternalCodeException("XMAISM08 This should never happen.");
         }
      }
      else
      {
         assert(m_scale < 0.0);

         if(GErel(x[m_k], m_upK + m_scale * m_loJ))
         {
            assert(m_loJ > R(-infinity));
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_j] = m_loJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_loJ) && m_loJ > R(-infinity))
         {
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_j] = m_loJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_upK + m_scale * m_upJ) && m_upK < R(infinity))
         {
            cStatus[m_k] = (m_loK == m_upK) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_k] = m_upK;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            x[m_j] = z2 * scale2 / m_scale;
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_upJ) && m_upJ < R(infinity))
         {
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_j] = m_upJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else if(GErel(x[m_k], m_loK + m_scale * m_upJ) && m_loK > R(-infinity))
         {
            cStatus[m_k] = (m_loK == m_upK) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_LOWER;
            x[m_k] = m_loK;
            cStatus[m_j] = SPxSolverBase<R>::BASIC;
            x[m_j] = z1 * scale1 / m_scale;
         }
         else if(LTrel(x[m_k], m_loK + m_scale * m_upJ))
         {
            assert(m_upJ < R(infinity));
            cStatus[m_j] = (m_loJ == m_upJ) ? SPxSolverBase<R>::FIXED : SPxSolverBase<R>::ON_UPPER;
            x[m_j] = m_upJ;
            x[m_k] -= m_scale * x[m_j];
         }
         else
         {
            throw SPxInternalCodeException("XMAISM09 This should never happen.");
         }
      }
   }

   // dual:
   r[m_j] = m_scale * r[m_k];
}

template <class R>
void SPxMainSM<R>::AggregationPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // correcting the change of idx by deletion of the column:
   if(m_j != m_old_j)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   // primal:
   R val = 0.0;
   R aij = m_row[m_j];
   int active_idx = -1;

   assert(m_row.size() == 2);

   for(int k = 0; k < 2; ++k)
   {
      if(m_row.index(k) != m_j)
      {
         active_idx = m_row.index(k);
         val = m_row.value(k) * x[active_idx];
      }
   }

   assert(active_idx >= 0);

   R scale = maxAbs(m_rhs, val);

   if(scale < 1.0)
      scale = 1.0;

   R z = (m_rhs / scale) - (val / scale);

   if(isZero(z))
      z = 0.0;

   x[m_j] = z * scale / aij;
   s[m_i] = m_rhs;

   if(isOptimal && (LT(x[m_j], m_lower, this->eps()) || GT(x[m_j], m_upper, this->eps())))
   {
      MSG_ERROR(std::cerr << "EMAISM: numerical violation after disaggregating variable" << std::endl;)
   }

   // dual:
   R dualVal = 0.0;

   for(int k = 0; k < m_col.size(); ++k)
   {
      if(m_col.index(k) != m_i)
         dualVal += m_col.value(k) * y[m_col.index(k)];
   }

   z = m_obj - dualVal;

   y[m_i] = z / aij;
   r[m_j] = 0.0;

   // basis:
   if(((cStatus[active_idx] == SPxSolverBase<R>::ON_UPPER
         || cStatus[active_idx] == SPxSolverBase<R>::FIXED)
         && NE(x[active_idx], m_oldupper, this->eps())) ||
         ((cStatus[active_idx] == SPxSolverBase<R>::ON_LOWER
           || cStatus[active_idx] == SPxSolverBase<R>::FIXED)
          && NE(x[active_idx], m_oldlower, this->eps())))
   {
      cStatus[active_idx] = SPxSolverBase<R>::BASIC;
      r[active_idx] = 0.0;
      assert(NE(m_upper, m_lower));

      if(EQ(x[m_j], m_upper, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;
      else if(EQ(x[m_j], m_lower, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
      else if(m_upper >= R(infinity) && m_lower <= R(-infinity))
         cStatus[m_j] = SPxSolverBase<R>::ZERO;
      else
         throw SPxInternalCodeException("XMAISM unexpected basis status in aggregation unsimplifier.");
   }
   else
   {
      cStatus[m_j] = SPxSolverBase<R>::BASIC;
   }

   // sides may not be equal and we always only consider the rhs during aggregation, so set ON_UPPER
   // (in theory and with exact arithmetic setting it to FIXED would be correct)
   rStatus[m_i] = SPxSolverBase<R>::ON_UPPER;

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM22 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::MultiAggregationPS::execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s,
      VectorBase<R>& r,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{

   // correcting the change of idx by deletion of the row:
   if(m_i != m_old_i)
   {
      s[m_old_i] = s[m_i];
      y[m_old_i] = y[m_i];
      rStatus[m_old_i] = rStatus[m_i];
   }

   // correcting the change of idx by deletion of the column:
   if(m_j != m_old_j)
   {
      x[m_old_j] = x[m_j];
      r[m_old_j] = r[m_j];
      cStatus[m_old_j] = cStatus[m_j];
   }

   // primal:
   R val = 0.0;
   R aij = m_row[m_j];

   for(int k = 0; k < m_row.size(); ++k)
   {
      if(m_row.index(k) != m_j)
         val += m_row.value(k) * x[m_row.index(k)];
   }

   R scale = maxAbs(m_const, val);

   if(scale < 1.0)
      scale = 1.0;

   R z = (m_const / scale) - (val / scale);

   if(isZero(z))
      z = 0.0;

   x[m_j] = z * scale / aij;
   s[m_i] = 0.0;

#ifndef NDEBUG

   if(isOptimal && (LT(x[m_j], m_lower, this->eps()) || GT(x[m_j], m_upper, this->eps())))
      MSG_ERROR(std::cerr << "numerical violation in original space due to MultiAggregation\n";)
#endif

      // dual:
      R dualVal = 0.0;

   for(int k = 0; k < m_col.size(); ++k)
   {
      if(m_col.index(k) != m_i)
         dualVal += m_col.value(k) * y[m_col.index(k)];
   }

   z = m_obj - dualVal;

   y[m_i] = z / aij;
   r[m_j] = 0.0;

   // basis:
   cStatus[m_j] = SPxSolverBase<R>::BASIC;

   if(m_eqCons)
      rStatus[m_i] = SPxSolverBase<R>::FIXED;
   else if(m_onLhs)
      rStatus[m_i] = SPxSolverBase<R>::ON_LOWER;
   else
      rStatus[m_i] = SPxSolverBase<R>::ON_UPPER;

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM22 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::TightenBoundsPS::execute(VectorBase<R>& x, VectorBase<R>&, VectorBase<R>&,
      VectorBase<R>&,
      DataArray<typename SPxSolverBase<R>::VarStatus>& cStatus,
      DataArray<typename SPxSolverBase<R>::VarStatus>& rStatus, bool isOptimal) const
{
   // basis:
   switch(cStatus[m_j])
   {
   case SPxSolverBase<R>::FIXED:
      if(LT(x[m_j], m_origupper, this->eps()) && GT(x[m_j], m_origlower, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::BASIC;
      else if(LT(x[m_j], m_origupper, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::ON_LOWER;
      else if(GT(x[m_j], m_origlower, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::ON_UPPER;

      break;

   case SPxSolverBase<R>::ON_LOWER:
      if(GT(x[m_j], m_origlower, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::BASIC;

      break;

   case SPxSolverBase<R>::ON_UPPER:
      if(LT(x[m_j], m_origupper, this->eps()))
         cStatus[m_j] = SPxSolverBase<R>::BASIC;

      break;

   default:
      break;
   }

#ifdef CHECK_BASIC_DIM

   if(!this->checkBasisDim(rStatus, cStatus))
   {
      throw SPxInternalCodeException("XMAISM22 Dimension doesn't match after this step.");
   }

#endif
}

template <class R>
void SPxMainSM<R>::handleRowObjectives(SPxLPBase<R>& lp)
{
   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      if(lp.maxRowObj(i) != 0.0)
      {
         std::shared_ptr<PostStep> ptr(new RowObjPS(lp, i, lp.nCols()));
         m_hist.append(ptr);
         lp.addCol(lp.rowObj(i), -lp.rhs(i), UnitVectorBase<R>(i), -lp.lhs(i));
         lp.changeRange(i, R(0.0), R(0.0));
         lp.changeRowObj(i, R(0.0));
         m_addedcols++;
      }
   }
}

template <class R>
void SPxMainSM<R>::handleExtremes(SPxLPBase<R>& lp)
{

   // This method handles extreme value of the given LP by
   //
   // 1. setting numbers of very small absolute values to zero and
   // 2. setting numbers of very large absolute values to R(-infinity) or +R(infinity), respectively.

   R maxVal  = R(infinity) / 5.0;
   R tol = feastol() * 1e-2;
   tol = (tol < this->epsZero()) ? this->epsZero() : tol;
   int  remRows = 0;
   int  remNzos = 0;
   int  chgBnds = 0;
   int  chgLRhs = 0;
   int  objCnt  = 0;

   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      // lhs
      R lhs = lp.lhs(i);

      if(lhs != 0.0 && isZero(lhs, this->epsZero()))
      {
         lp.changeLhs(i, R(0.0));
         ++chgLRhs;
      }
      else if(lhs > R(-infinity) && lhs < -maxVal)
      {
         lp.changeLhs(i, R(-infinity));
         ++chgLRhs;
      }
      else if(lhs <  R(infinity) && lhs >  maxVal)
      {
         lp.changeLhs(i,  R(infinity));
         ++chgLRhs;
      }

      // rhs
      R rhs = lp.rhs(i);

      if(rhs != 0.0 && isZero(rhs, this->epsZero()))
      {
         lp.changeRhs(i, R(0.0));
         ++chgLRhs;
      }
      else if(rhs > R(-infinity) && rhs < -maxVal)
      {
         lp.changeRhs(i, R(-infinity));
         ++chgLRhs;
      }
      else if(rhs <  R(infinity) && rhs >  maxVal)
      {
         lp.changeRhs(i,  R(infinity));
         ++chgLRhs;
      }

      if(lp.lhs(i) <= R(-infinity) && lp.rhs(i) >= R(infinity))
      {
         std::shared_ptr<PostStep> ptr(new FreeConstraintPS(lp, i));
         m_hist.append(ptr);

         removeRow(lp, i);
         ++remRows;

         ++m_stat[FREE_ROW];
      }
   }

   for(int j = 0; j < lp.nCols(); ++j)
   {
      // lower bound
      R lo = lp.lower(j);

      if(lo != 0.0 && isZero(lo, this->epsZero()))
      {
         lp.changeLower(j, R(0.0));
         ++chgBnds;
      }
      else if(lo > R(-infinity) && lo < -maxVal)
      {
         lp.changeLower(j, R(-infinity));
         ++chgBnds;
      }
      else if(lo <  R(infinity) && lo >  maxVal)
      {
         lp.changeLower(j,  R(infinity));
         ++chgBnds;
      }

      // upper bound
      R up = lp.upper(j);

      if(up != 0.0 && isZero(up, this->epsZero()))
      {
         lp.changeUpper(j, R(0.0));
         ++chgBnds;
      }
      else if(up > R(-infinity) && up < -maxVal)
      {
         lp.changeUpper(j, R(-infinity));
         ++chgBnds;
      }
      else if(up <  R(infinity) && up >  maxVal)
      {
         lp.changeUpper(j,  R(infinity));
         ++chgBnds;
      }

      // fixed columns will be eliminated later
      if(NE(lo, up))
      {
         lo = spxAbs(lo);
         up = spxAbs(up);

         R absBnd = (lo > up) ? lo : up;

         if(absBnd < 1.0)
            absBnd = 1.0;

         // non-zeros
         SVectorBase<R>& col = lp.colVector_w(j);
         int        i = 0;

         while(i < col.size())
         {
            R aij = spxAbs(col.value(i));

            if(isZero(aij * absBnd, tol))
            {
               SVectorBase<R>& row = lp.rowVector_w(col.index(i));
               int row_j = row.pos(j);

               // this changes col.size()
               if(row_j >= 0)
                  row.remove(row_j);

               col.remove(i);

               MSG_DEBUG((*this->spxout) << "IMAISM04 aij=" << aij
                         << " removed, absBnd=" << absBnd
                         << std::endl;)
               ++remNzos;
            }
            else
            {
               if(aij > maxVal)
               {
                  MSG_WARNING((*this->spxout), (*this->spxout) << "WMAISM05 Warning! Big matrix coefficient " << aij
                              << std::endl);
               }
               else if(isZero(aij, tol))
               {
                  MSG_WARNING((*this->spxout), (*this->spxout) << "WMAISM06 Warning! Tiny matrix coefficient " << aij
                              << std::endl);
               }

               ++i;
            }
         }
      }

      // objective
      R obj = lp.obj(j);

      if(obj != 0.0 && isZero(obj, this->epsZero()))
      {
         lp.changeObj(j, R(0.0));
         ++objCnt;
      }
      else if(obj > R(-infinity) && obj < -maxVal)
      {
         lp.changeObj(j, R(-infinity));
         ++objCnt;
      }
      else if(obj <  R(infinity) && obj >  maxVal)
      {
         lp.changeObj(j,  R(infinity));
         ++objCnt;
      }
   }

   if(remRows + remNzos + chgLRhs + chgBnds + objCnt > 0)
   {
      this->m_remRows += remRows;
      this->m_remNzos += remNzos;
      this->m_chgLRhs += chgLRhs;
      this->m_chgBnds += chgBnds;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (extremes) removed "
                << remRows << " rows, "
                << remNzos << " non-zeros, "
                << chgBnds << " col bounds, "
                << chgLRhs << " row bounds, "
                << objCnt  << " objective coefficients" << std::endl;)
   }

   assert(lp.isConsistent());
}

/// computes the minimum and maximum residual activity for a given variable
template <class R>
void SPxMainSM<R>::computeMinMaxResidualActivity(SPxLPBase<R>& lp, int rowNumber, int colNumber,
      R& minAct, R& maxAct)
{
   const SVectorBase<R>& row = lp.rowVector(rowNumber);
   bool minNegInfinite = false;
   bool maxInfinite = false;

   minAct = 0;   // this is the minimum value that the aggregation can attain
   maxAct = 0;   // this is the maximum value that the aggregation can attain

   for(int l = 0; l < row.size(); ++l)
   {
      if(colNumber < 0 || row.index(l) != colNumber)
      {
         // computing the minimum activity of the aggregated variables
         if(GT(row.value(l), R(0.0)))
         {
            if(LE(lp.lower(row.index(l)), R(-infinity)))
               minNegInfinite = true;
            else
               minAct += row.value(l) * lp.lower(row.index(l));
         }
         else if(LT(row.value(l), R(0.0)))
         {
            if(GE(lp.upper(row.index(l)), R(infinity)))
               minNegInfinite = true;
            else
               minAct += row.value(l) * lp.upper(row.index(l));
         }

         // computing the maximum activity of the aggregated variables
         if(GT(row.value(l), R(0.0)))
         {
            if(GE(lp.upper(row.index(l)), R(infinity)))
               maxInfinite = true;
            else
               maxAct += row.value(l) * lp.upper(row.index(l));
         }
         else if(LT(row.value(l), R(0.0)))
         {
            if(LE(lp.lower(row.index(l)), R(-infinity)))
               maxInfinite = true;
            else
               maxAct += row.value(l) * lp.lower(row.index(l));
         }
      }
   }

   // if an infinite value exists for the minimum activity, then that it taken
   if(minNegInfinite)
      minAct = R(-infinity);

   // if an -infinite value exists for the maximum activity, then that value is taken
   if(maxInfinite)
      maxAct = R(infinity);
}


/// calculate min/max value for the multi aggregated variables
template <class R>
void SPxMainSM<R>::computeMinMaxValues(SPxLPBase<R>& lp, R side, R val, R minRes, R maxRes,
                                       R& minVal, R& maxVal)
{
   minVal = 0;
   maxVal = 0;

   if(LT(val, R(0.0)))
   {
      if(LE(minRes, R(-infinity)))
         minVal = R(-infinity);
      else
         minVal = (side - minRes) / val;

      if(GE(maxRes, R(infinity)))
         maxVal = R(infinity);
      else
         maxVal = (side - maxRes) / val;
   }
   else if(GT(val, R(0.0)))
   {
      if(GE(maxRes, R(infinity)))
         minVal = R(-infinity);
      else
         minVal = (side - maxRes) / val;

      if(LE(minRes, R(-infinity)))
         maxVal = R(infinity);
      else
         maxVal = (side - minRes) / val;
   }
}


/// tries to find good lower bound solutions by applying some trivial heuristics
template <class R>
void SPxMainSM<R>::trivialHeuristic(SPxLPBase<R>& lp)
{
   VectorBase<R>         zerosol(lp.nCols());  // the zero solution VectorBase<R>
   VectorBase<R>         lowersol(lp.nCols()); // the lower bound solution VectorBase<R>
   VectorBase<R>         uppersol(lp.nCols()); // the upper bound solution VectorBase<R>
   VectorBase<R>         locksol(lp.nCols());  // the locks solution VectorBase<R>

   VectorBase<R>         upLocks(lp.nCols());
   VectorBase<R>         downLocks(lp.nCols());

   R            zeroObj = this->m_objoffset;
   R            lowerObj = this->m_objoffset;
   R            upperObj = this->m_objoffset;
   R            lockObj = this->m_objoffset;

   bool            zerovalid = true;

   R largeValue = R(infinity);

   if(LT(R(1.0 / feastol()), R(infinity)))
      largeValue = 1.0 / feastol();



   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      upLocks[j] = 0;
      downLocks[j] = 0;

      // computing the locks on the variables
      const SVectorBase<R>& col = lp.colVector(j);

      for(int k = 0; k < col.size(); ++k)
      {
         R aij = col.value(k);

         ASSERT_WARN("WMAISM45", isNotZero(aij, R(1.0 / R(infinity))));

         if(GT(lp.lhs(col.index(k)), R(-infinity)) && LT(lp.rhs(col.index(k)), R(infinity)))
         {
            upLocks[j]++;
            downLocks[j]++;
         }
         else if(GT(lp.lhs(col.index(k)), R(-infinity)))
         {
            if(aij > 0)
               downLocks[j]++;
            else if(aij < 0)
               upLocks[j]++;
         }
         else if(LT(lp.rhs(col.index(k)), R(infinity)))
         {
            if(aij > 0)
               upLocks[j]++;
            else if(aij < 0)
               downLocks[j]++;
         }
      }

      R lower = lp.lower(j);
      R upper = lp.upper(j);

      if(LE(lower, R(-infinity)))
         lower = MINIMUM(-largeValue, upper);

      if(GE(upper, R(infinity)))
         upper = MAXIMUM(lp.lower(j), largeValue);

      if(zerovalid)
      {
         if(LE(lower, R(0.0), feastol()) && GE(upper, R(0.0), feastol()))
            zerosol[j] = 0.0;
         else
            zerovalid = false;
      }

      lowersol[j] = lower;
      uppersol[j] = upper;

      if(downLocks[j] > upLocks[j])
         locksol[j] = upper;
      else if(downLocks[j] < upLocks[j])
         locksol[j] = lower;
      else
         locksol[j] = (lower + upper) / 2.0;

      lowerObj += lp.maxObj(j) * lowersol[j];
      upperObj += lp.maxObj(j) * uppersol[j];
      lockObj += lp.maxObj(j) * locksol[j];
   }

   // trying the lower bound solution
   if(checkSolution(lp, lowersol))
   {
      if(lowerObj > m_cutoffbound)
         m_cutoffbound = lowerObj;
   }

   // trying the upper bound solution
   if(checkSolution(lp, uppersol))
   {
      if(upperObj > m_cutoffbound)
         m_cutoffbound = upperObj;
   }

   // trying the zero solution
   if(zerovalid && checkSolution(lp, zerosol))
   {
      if(zeroObj > m_cutoffbound)
         m_cutoffbound = zeroObj;
   }

   // trying the lock solution
   if(checkSolution(lp, locksol))
   {
      if(lockObj > m_cutoffbound)
         m_cutoffbound = lockObj;
   }
}



/// checks a solution for feasibility
template <class R>
bool SPxMainSM<R>::checkSolution(SPxLPBase<R>& lp, VectorBase<R> sol)
{
   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      const SVectorBase<R>& row = lp.rowVector(i);
      R activity = 0;

      for(int k = 0; k < row.size(); k++)
         activity += row.value(k) * sol[row.index(k)];

      if(!GE(activity, lp.lhs(i), feastol()) || !LE(activity, lp.rhs(i), feastol()))
         return false;
   }

   return true;
}


/// tightens variable bounds by propagating the pseudo objective function value.
template <class R>
void SPxMainSM<R>::propagatePseudoobj(SPxLPBase<R>& lp)
{
   R pseudoObj = this->m_objoffset;

   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      R val = lp.maxObj(j);

      if(val < 0)
      {
         if(lp.lower(j) <= R(-infinity))
            return;

         pseudoObj += val * lp.lower(j);
      }
      else if(val > 0)
      {
         if(lp.upper(j) >= R(-infinity))
            return;

         pseudoObj += val * lp.upper(j);
      }
   }

   if(GT(m_cutoffbound, R(-infinity)) && LT(m_cutoffbound, R(infinity)))
   {
      if(pseudoObj > m_pseudoobj)
         m_pseudoobj = pseudoObj;

      for(int j = lp.nCols() - 1; j >= 0; --j)
      {
         R objval = lp.maxObj(j);

         if(EQ(objval, R(0.0)))
            continue;

         if(objval < 0.0)
         {
            R newbound = lp.lower(j) + (m_cutoffbound - m_pseudoobj) / objval;

            if(LT(newbound, lp.upper(j)))
            {
               std::shared_ptr<PostStep> ptr(new TightenBoundsPS(lp, j, lp.upper(j), lp.lower(j)));
               m_hist.append(ptr);
               lp.changeUpper(j, newbound);
            }
         }
         else if(objval > 0.0)
         {
            R newbound = lp.upper(j) + (m_cutoffbound - m_pseudoobj) / objval;

            if(GT(newbound, lp.lower(j)))
            {
               std::shared_ptr<PostStep> ptr(new TightenBoundsPS(lp, j, lp.upper(j), lp.lower(j)));
               m_hist.append(ptr);
               lp.changeLower(j, newbound);
            }
         }
      }
   }
}



template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::removeEmpty(SPxLPBase<R>& lp)
{

   // This method removes empty rows and columns from the LP.

   int remRows = 0;
   int remCols = 0;

   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      const SVectorBase<R>& row = lp.rowVector(i);

      if(row.size() == 0)
      {
         MSG_DEBUG((*this->spxout) << "IMAISM07 row " << i
                   << ": empty ->";)

         if(LT(lp.rhs(i), R(0.0), feastol()) || GT(lp.lhs(i), R(0.0), feastol()))
         {
            MSG_DEBUG((*this->spxout) << " infeasible lhs=" << lp.lhs(i)
                      << " rhs=" << lp.rhs(i) << std::endl;)
            return this->INFEASIBLE;
         }

         MSG_DEBUG((*this->spxout) << " removed" << std::endl;)

         std::shared_ptr<PostStep> ptr(new EmptyConstraintPS(lp, i));
         m_hist.append(ptr);

         ++remRows;
         removeRow(lp, i);

         ++m_stat[EMPTY_ROW];
      }
   }

   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      const SVectorBase<R>& col = lp.colVector(j);

      if(col.size() == 0)
      {
         MSG_DEBUG((*this->spxout) << "IMAISM08 col " << j
                   << ": empty -> maxObj=" << lp.maxObj(j)
                   << " lower=" << lp.lower(j)
                   << " upper=" << lp.upper(j);)

         R val;

         if(GT(lp.maxObj(j), R(0.0), this->epsZero()))
         {
            if(lp.upper(j) >= R(infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)
               return this->UNBOUNDED;
            }

            val = lp.upper(j);
         }
         else if(LT(lp.maxObj(j), R(0.0), this->epsZero()))
         {
            if(lp.lower(j) <= R(-infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)
               return this->UNBOUNDED;
            }

            val = lp.lower(j);
         }
         else
         {
            ASSERT_WARN("WMAISM09", isZero(lp.maxObj(j), this->epsZero()));

            // any value within the bounds is ok
            if(lp.lower(j) > R(-infinity))
               val = lp.lower(j);
            else if(lp.upper(j) < R(infinity))
               val = lp.upper(j);
            else
               val = 0.0;
         }

         MSG_DEBUG((*this->spxout) << " removed" << std::endl;)

         std::shared_ptr<PostStep> ptr1(new FixBoundsPS(lp, j, val));
         std::shared_ptr<PostStep> ptr2(new FixVariablePS(lp, *this, j, val));
         m_hist.append(ptr1);
         m_hist.append(ptr2);

         ++remCols;
         removeCol(lp, j);

         ++m_stat[EMPTY_COL];
      }
   }

   if(remRows + remCols > 0)
   {
      this->m_remRows += remRows;
      this->m_remCols += remCols;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (empty rows/colums) removed "
                << remRows << " rows, "
                << remCols << " cols"
                << std::endl;)

   }

   return this->OKAY;
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::removeRowSingleton(SPxLPBase<R>& lp,
      const SVectorBase<R>& row, int& i)
{
   assert(row.size() == 1);

   R aij = row.value(0);
   int  j   = row.index(0);
   R lo  = R(-infinity);
   R up  =  R(infinity);

   MSG_DEBUG((*this->spxout) << "IMAISM22 row " << i
             << ": singleton -> val=" << aij
             << " lhs=" << lp.lhs(i)
             << " rhs=" << lp.rhs(i);)

   if(GT(aij, R(0.0), this->epsZero()))            // aij > 0
   {
      lo = (lp.lhs(i) <= R(-infinity)) ? R(-infinity) : lp.lhs(i) / aij;
      up = (lp.rhs(i) >=  R(infinity)) ?  R(infinity) : lp.rhs(i) / aij;
   }
   else if(LT(aij, R(0.0), this->epsZero()))       // aij < 0
   {
      lo = (lp.rhs(i) >=  R(infinity)) ? R(-infinity) : lp.rhs(i) / aij;
      up = (lp.lhs(i) <= R(-infinity)) ?  R(infinity) : lp.lhs(i) / aij;
   }
   else if(LT(lp.rhs(i), R(0.0), feastol()) || GT(lp.lhs(i), R(0.0), feastol()))
   {
      // aij == 0, rhs < 0 or lhs > 0
      MSG_DEBUG((*this->spxout) << " infeasible" << std::endl;)
      return this->INFEASIBLE;
   }

   if(isZero(lo, this->epsZero()))
      lo = 0.0;

   if(isZero(up, this->epsZero()))
      up = 0.0;

   MSG_DEBUG((*this->spxout) << " removed, lower=" << lo
             << " (" << lp.lower(j)
             << ") upper=" << up
             << " (" << lp.upper(j)
             << ")" << std::endl;)

   bool stricterUp = false;
   bool stricterLo = false;

   R oldLo = lp.lower(j);
   R oldUp = lp.upper(j);

   if(LTrel(up, lp.upper(j), feastol()))
   {
      lp.changeUpper(j, up);
      stricterUp = true;
   }

   if(GTrel(lo, lp.lower(j), feastol()))
   {
      lp.changeLower(j, lo);
      stricterLo = true;
   }

   std::shared_ptr<PostStep> ptr(new RowSingletonPS(lp, i, j, stricterLo, stricterUp, lp.lower(j),
                                 lp.upper(j), oldLo, oldUp));
   m_hist.append(ptr);

   removeRow(lp, i);

   this->m_remRows++;
   this->m_remNzos++;
   ++m_stat[SINGLETON_ROW];

   return this->OKAY;
}

/// aggregate variable x_j to x_j = (rhs - aik * x_k) / aij from row i: aij * x_j + aik * x_k = rhs
template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::aggregateVars(SPxLPBase<R>& lp,
      const SVectorBase<R>& row, int& i)
{
   assert(row.size() == 2);
   assert(EQrel(lp.lhs(i), lp.rhs(i), feastol()));

   R rhs = lp.rhs(i);
   assert(rhs < R(infinity) && rhs > R(-infinity));

   int j = row.index(0);
   int k = row.index(1);
   R aij = row.value(0);
   R aik = row.value(1);
   R lower_j = lp.lower(j);
   R upper_j = lp.upper(j);
   R lower_k = lp.lower(k);
   R upper_k = lp.upper(k);

   // fixed variables should be removed by simplifyCols()
   if(EQrel(lower_j, upper_j, feastol()) || EQrel(lower_k, upper_k, feastol()))
      return this->OKAY;

   assert(isNotZero(aij, this->epsZero()) && isNotZero(aik, this->epsZero()));

   MSG_DEBUG((*this->spxout) << "IMAISM22 row " << i << ": doubleton equation -> "
             << aij << " x_" << j << " + " << aik << " x_" << k << " = " << rhs;)

   // determine which variable can be aggregated without requiring bound tightening of the other variable
   R new_lo_j;
   R new_up_j;
   R new_lo_k;
   R new_up_k;

   if(aij * aik < 0.0)
   {
      // orientation persists
      new_lo_j = (upper_k >=  R(infinity)) ? R(-infinity) : (rhs - aik * upper_k) / aij;
      new_up_j = (lower_k <= R(-infinity)) ?  R(infinity) : (rhs - aik * lower_k) / aij;
      new_lo_k = (upper_j >=  R(infinity)) ? R(-infinity) : (rhs - aij * upper_j) / aik;
      new_up_k = (lower_j <= R(-infinity)) ?  R(infinity) : (rhs - aij * lower_j) / aik;
   }
   else if(aij * aik > 0.0)
   {
      // orientation is reversed
      new_lo_j = (lower_k <= R(-infinity)) ? R(-infinity) : (rhs - aik * lower_k) / aij;
      new_up_j = (upper_k >=  R(infinity)) ?  R(infinity) : (rhs - aik * upper_k) / aij;
      new_lo_k = (lower_j <= R(-infinity)) ? R(-infinity) : (rhs - aij * lower_j) / aik;
      new_up_k = (upper_j >=  R(infinity)) ?  R(infinity) : (rhs - aij * upper_j) / aik;
   }
   else
      throw SPxInternalCodeException("XMAISM12 This should never happen.");

   bool flip_jk = false;

   if(new_lo_j <= R(-infinity) && new_up_j >= R(infinity))
   {
      // no bound tightening on x_j when x_k is aggregated
      flip_jk = true;
   }
   else if(new_lo_k <= R(-infinity) && new_up_k >= R(infinity))
   {
      // no bound tightening on x_k when x_j is aggregated
      flip_jk = false;
   }
   else if(LE(new_lo_j, lower_j) && GE(new_up_j, upper_j))
   {
      if(LE(new_lo_k, lower_k) && GE(new_up_k, upper_k))
      {
         // both variables' bounds are not affected by aggregation; choose the better aggregation coeff (aik/aij)
         if(spxAbs(aij) > spxAbs(aik))
            flip_jk = false;
         else
            flip_jk = true;
      }
      else
         flip_jk = false;
   }
   else if(LE(new_lo_k, lower_k) && GE(new_up_k, upper_k))
   {
      flip_jk = true;
   }
   else
   {
      if(spxAbs(aij) > spxAbs(aik))
         flip_jk = false;
      else
         flip_jk = true;
   }

   if(flip_jk)
   {
      int _j = j;
      R _aij = aij;
      R _lower_j = lower_j;
      R _upper_j = upper_j;
      j = k;
      k = _j;
      aij = aik;
      aik = _aij;
      lower_j = lower_k;
      lower_k = _lower_j;
      upper_j = upper_k;
      upper_k = _upper_j;
   }

   const SVectorBase<R>& col_j = lp.colVector(j);
   const SVectorBase<R>& col_k = lp.colVector(k);

   // aggregation coefficients (x_j = aggr_coef * x_k + aggr_const)
   R aggr_coef = - (aik / aij);
   R aggr_const = rhs / aij;

   MSG_DEBUG((*this->spxout) << " removed, replacing x_" << j << " with "
             << aggr_const << " + " << aggr_coef << " * x_" << k << std::endl;)

   // replace all occurrences of x_j
   for(int r = 0; r < col_j.size(); ++r)
   {
      int row_r = col_j.index(r);
      R arj = col_j.value(r);

      // skip row i
      if(row_r == i)
         continue;

      // adapt sides of row r
      R lhs_r = lp.lhs(row_r);
      R rhs_r = lp.rhs(row_r);

      if(lhs_r > R(-infinity))
      {
         lp.changeLhs(row_r, lhs_r - aggr_const * arj);
         this->m_chgLRhs++;
      }

      if(rhs_r < R(infinity))
      {
         lp.changeRhs(row_r, rhs_r - aggr_const * arj);
         this->m_chgLRhs++;
      }

      R newcoef = aggr_coef * arj;
      int pos_rk = col_k.pos(row_r);

      // check whether x_k is also present in row r and get its coefficient
      if(pos_rk >= 0)
      {
         R ark = col_k.value(pos_rk);
         newcoef += ark;
         this->m_remNzos++;
      }

      // add new column k to row r or adapt the coefficient a_rk
      lp.changeElement(row_r, k, newcoef);
   }

   // adapt objective function
   R obj_j = lp.obj(j);

   if(isNotZero(obj_j, this->epsZero()))
   {
      this->addObjoffset(aggr_const * obj_j);
      R obj_k = lp.obj(k);
      lp.changeObj(k, obj_k + aggr_coef * obj_j);
   }

   // adapt bounds of x_k
   R scale1 = maxAbs(rhs, aij * upper_j);
   R scale2 = maxAbs(rhs, aij * lower_j);

   if(scale1 < 1.0)
      scale1 = 1.0;

   if(scale2 < 1.0)
      scale2 = 1.0;

   R z1 = (rhs / scale1) - (aij * upper_j / scale1);
   R z2 = (rhs / scale2) - (aij * lower_j / scale2);

   // just some rounding
   if(isZero(z1, this->epsZero()))
      z1 = 0.0;

   if(isZero(z2, this->epsZero()))
      z2 = 0.0;

   // determine which side has to be used for the bounds comparison below
   if(aik * aij > 0.0)
   {
      new_lo_k = (upper_j >=  R(infinity)) ? R(-infinity) : z1 * scale1 / aik;
      new_up_k = (lower_j <= R(-infinity)) ?  R(infinity) : z2 * scale2 / aik;
   }
   else if(aik * aij < 0.0)
   {
      new_lo_k = (lower_j <= R(-infinity)) ? R(-infinity) : z2 * scale2 / aik;
      new_up_k = (upper_j >=  R(infinity)) ?  R(infinity) : z1 * scale1 / aik;
   }
   else
      throw SPxInternalCodeException("XMAISM12 This should never happen.");

   // change bounds of x_k if the new ones are tighter
   R oldlower_k = lower_k;
   R oldupper_k = upper_k;

   if(GT(new_lo_k, lower_k, this->epsZero()))
   {
      lp.changeLower(k, new_lo_k);
      this->m_chgBnds++;
   }

   if(LT(new_up_k, upper_k, this->epsZero()))
   {
      lp.changeUpper(k, new_up_k);
      this->m_chgBnds++;
   }

   std::shared_ptr<PostStep> ptr(new AggregationPS(lp, i, j, rhs, oldupper_k, oldlower_k));
   m_hist.append(ptr);

   removeRow(lp, i);
   removeCol(lp, j);

   this->m_remRows++;
   this->m_remCols++;
   this->m_remNzos += 2;

   ++m_stat[AGGREGATION];

   return this->OKAY;
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::simplifyRows(SPxLPBase<R>& lp, bool& again)
{

   // This method simplifies the rows of the LP.
   //
   // The following operations are done:
   // 1. detect implied free variables
   // 2. detect implied free constraints
   // 3. detect infeasible constraints
   // 4. remove unconstrained constraints
   // 5. remove empty constraints
   // 6. remove row singletons and tighten the corresponding variable bounds if necessary
   // 7. remove doubleton equation, aka aggregation
   // 8. detect forcing rows and fix the corresponding variables

   int remRows = 0;
   int remNzos = 0;
   int chgLRhs = 0;
   int chgBnds = 0;
   int keptBnds = 0;
   int keptLRhs = 0;

   int oldRows = lp.nRows();

   bool redundantLower;
   bool redundantUpper;
   bool redundantLhs;
   bool redundantRhs;

   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      const SVectorBase<R>& row = lp.rowVector(i);


      // compute bounds on constraint value
      R lhsBnd = 0.0; // minimal activity (finite summands)
      R rhsBnd = 0.0; // maximal activity (finite summands)
      int  lhsCnt = 0; // number of R(-infinity) summands in minimal activity
      int  rhsCnt = 0; // number of +R(infinity) summands in maximal activity

      for(int k = 0; k < row.size(); ++k)
      {
         R aij = row.value(k);
         int  j   = row.index(k);

         if(!isNotZero(aij, R(1.0 / R(infinity))))
         {
            MSG_WARNING((*this->spxout), (*this->spxout) << "Warning: tiny nonzero coefficient " << aij <<
                        " in row " << i << "\n");
         }

         if(aij > 0.0)
         {
            if(lp.lower(j) <= R(-infinity))
               ++lhsCnt;
            else
               lhsBnd += aij * lp.lower(j);

            if(lp.upper(j) >= R(infinity))
               ++rhsCnt;
            else
               rhsBnd += aij * lp.upper(j);
         }
         else if(aij < 0.0)
         {
            if(lp.lower(j) <= R(-infinity))
               ++rhsCnt;
            else
               rhsBnd += aij * lp.lower(j);

            if(lp.upper(j) >= R(infinity))
               ++lhsCnt;
            else
               lhsBnd += aij * lp.upper(j);
         }
      }

#if FREE_BOUNDS

      // 1. detect implied free variables
      if(rhsCnt <= 1 || lhsCnt <= 1)
      {
         for(int k = 0; k < row.size(); ++k)
         {
            R aij = row.value(k);
            int  j   = row.index(k);

            redundantLower = false;
            redundantUpper = false;

            ASSERT_WARN("WMAISM12", isNotZero(aij, R(1.0 / R(infinity))));

            if(aij > 0.0)
            {
               if(lp.lhs(i) > R(-infinity) && lp.lower(j) > R(-infinity) && rhsCnt <= 1
                     && NErel(lp.lhs(i), rhsBnd, feastol())
                     // do not perform if strongly different orders of magnitude occur
                     && spxAbs(lp.lhs(i) / maxAbs(rhsBnd, R(1.0))) > Param::epsilon())
               {
                  R lo    = R(-infinity);
                  R scale = maxAbs(lp.lhs(i), rhsBnd);

                  if(scale < 1.0)
                     scale = 1.0;

                  R z = (lp.lhs(i) / scale) - (rhsBnd / scale);

                  if(isZero(z, this->epsZero()))
                     z = 0.0;

                  assert(rhsCnt > 0 || lp.upper(j) < R(infinity));

                  if(rhsCnt == 0)
                     lo = lp.upper(j) + z * scale / aij;
                  else if(lp.upper(j) >= R(infinity))
                     lo = z * scale / aij;

                  if(isZero(lo, this->epsZero()))
                     lo = 0.0;

                  if(GErel(lo, lp.lower(j), feastol()))
                  {
                     MSG_DEBUG((*this->spxout) << "IMAISM13 row " << i
                               << ": redundant lower bound on x" << j
                               << " -> lower=" << lo
                               << " (" << lp.lower(j)
                               << ")" << std::endl;)

                     redundantLower = true;
                  }

               }

               if(lp.rhs(i) < R(infinity) && lp.upper(j) < R(infinity) && lhsCnt <= 1
                     && NErel(lp.rhs(i), lhsBnd, feastol())
                     // do not perform if strongly different orders of magnitude occur
                     && spxAbs(lp.rhs(i) / maxAbs(lhsBnd, R(1.0))) > Param::epsilon())
               {
                  R up    = R(infinity);
                  R scale = maxAbs(lp.rhs(i), lhsBnd);

                  if(scale < 1.0)
                     scale = 1.0;

                  R z = (lp.rhs(i) / scale) - (lhsBnd / scale);

                  if(isZero(z, this->epsZero()))
                     z = 0.0;

                  assert(lhsCnt > 0 || lp.lower(j) > R(-infinity));

                  if(lhsCnt == 0)
                     up = lp.lower(j) + z * scale / aij;
                  else if(lp.lower(j) <= R(-infinity))
                     up = z * scale / aij;

                  if(isZero(up, this->epsZero()))
                     up = 0.0;

                  if(LErel(up, lp.upper(j), feastol()))
                  {
                     MSG_DEBUG((*this->spxout) << "IMAISM14 row " << i
                               << ": redundant upper bound on x" << j
                               << " -> upper=" << up
                               << " (" << lp.upper(j)
                               << ")" << std::endl;)

                     redundantUpper = true;
                  }
               }

               if(redundantLower)
               {
                  // no upper bound on x_j OR redundant upper bound
                  if((lp.upper(j) >= R(infinity)) || redundantUpper || (!m_keepbounds))
                  {
                     ++lhsCnt;
                     lhsBnd -= aij * lp.lower(j);

                     lp.changeLower(j, R(-infinity));
                     ++chgBnds;
                  }
                  else
                     ++keptBnds;
               }

               if(redundantUpper)
               {
                  // no lower bound on x_j OR redundant lower bound
                  if((lp.lower(j) <= R(-infinity)) || redundantLower || (!m_keepbounds))
                  {
                     ++rhsCnt;
                     rhsBnd -= aij * lp.upper(j);

                     lp.changeUpper(j, R(infinity));
                     ++chgBnds;
                  }
                  else
                     ++keptBnds;
               }
            }
            else if(aij < 0.0)
            {
               if(lp.lhs(i) > R(-infinity) && lp.upper(j) < R(infinity) && rhsCnt <= 1
                     && NErel(lp.lhs(i), rhsBnd, feastol())
                     // do not perform if strongly different orders of magnitude occur
                     && spxAbs(lp.lhs(i) / maxAbs(rhsBnd, R(1.0))) > Param::epsilon())
               {
                  R up    = R(infinity);
                  R scale = maxAbs(lp.lhs(i), rhsBnd);

                  if(scale < 1.0)
                     scale = 1.0;

                  R z = (lp.lhs(i) / scale) - (rhsBnd / scale);

                  if(isZero(z, this->epsZero()))
                     z = 0.0;

                  assert(rhsCnt > 0 || lp.lower(j) > R(-infinity));

                  if(rhsCnt == 0)
                     up = lp.lower(j) + z * scale / aij;
                  else if(lp.lower(j) <= R(-infinity))
                     up = z * scale / aij;

                  if(isZero(up, this->epsZero()))
                     up = 0.0;

                  if(LErel(up, lp.upper(j), feastol()))
                  {
                     MSG_DEBUG((*this->spxout) << "IMAISM15 row " << i
                               << ": redundant upper bound on x" << j
                               << " -> upper=" << up
                               << " (" << lp.upper(j)
                               << ")" << std::endl;)

                     redundantUpper = true;
                  }
               }

               if(lp.rhs(i) < R(infinity) && lp.lower(j) > R(-infinity) && lhsCnt <= 1
                     && NErel(lp.rhs(i), lhsBnd, feastol())
                     // do not perform if strongly different orders of magnitude occur
                     && spxAbs(lp.rhs(i) / maxAbs(lhsBnd, R(1.0))) > Param::epsilon())
               {
                  R lo    = R(-infinity);
                  R scale = maxAbs(lp.rhs(i), lhsBnd);

                  if(scale < 1.0)
                     scale = 1.0;

                  R z = (lp.rhs(i) / scale) - (lhsBnd / scale);

                  if(isZero(z, this->epsZero()))
                     z = 0.0;

                  assert(lhsCnt > 0 || lp.upper(j) < R(infinity));

                  if(lhsCnt == 0)
                     lo = lp.upper(j) + z * scale / aij;
                  else if(lp.upper(j) >= R(infinity))
                     lo = z * scale / aij;

                  if(isZero(lo, this->epsZero()))
                     lo = 0.0;

                  if(GErel(lo, lp.lower(j)))
                  {
                     MSG_DEBUG((*this->spxout) << "IMAISM16 row " << i
                               << ": redundant lower bound on x" << j
                               << " -> lower=" << lo
                               << " (" << lp.lower(j)
                               << ")" << std::endl;)

                     redundantLower = true;
                  }
               }

               if(redundantUpper)
               {
                  // no lower bound on x_j OR redundant lower bound
                  if((lp.lower(j) <= R(-infinity)) || redundantLower || (!m_keepbounds))
                  {
                     ++lhsCnt;
                     lhsBnd -= aij * lp.upper(j);

                     lp.changeUpper(j, R(infinity));
                     ++chgBnds;
                  }
                  else
                     ++keptBnds;
               }

               if(redundantLower)
               {
                  // no upper bound on x_j OR redundant upper bound
                  if((lp.upper(j) >= R(infinity)) || redundantUpper || (!m_keepbounds))
                  {
                     ++rhsCnt;
                     rhsBnd -= aij * lp.lower(j);

                     lp.changeLower(j, R(-infinity));
                     ++chgBnds;
                  }
                  else
                     ++keptBnds;
               }
            }
         }
      }

#endif

#if FREE_LHS_RHS

      redundantLhs = false;
      redundantRhs = false;

      // 2. detect implied free constraints
      if(lp.lhs(i) > R(-infinity) && lhsCnt == 0 && GErel(lhsBnd, lp.lhs(i), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM17 row " << i
                   << ": redundant lhs -> lhsBnd=" << lhsBnd
                   << " lhs=" << lp.lhs(i)
                   << std::endl;)

         redundantLhs = true;
      }

      if(lp.rhs(i) <  R(infinity) && rhsCnt == 0 && LErel(rhsBnd, lp.rhs(i), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM18 row " << i
                   << ": redundant rhs -> rhsBnd=" << rhsBnd
                   << " rhs=" << lp.rhs(i)
                   << std::endl;)

         redundantRhs = true;
      }

      if(redundantLhs)
      {
         // no rhs for constraint i OR redundant rhs
         if((lp.rhs(i) >= R(infinity)) || redundantRhs || (!m_keepbounds))
         {
            lp.changeLhs(i, R(-infinity));
            ++chgLRhs;
         }
         else
            ++keptLRhs;
      }

      if(redundantRhs)
      {
         // no lhs for constraint i OR redundant lhs
         if((lp.lhs(i) <= R(-infinity)) || redundantLhs || (!m_keepbounds))
         {
            lp.changeRhs(i, R(infinity));
            ++chgLRhs;
         }
         else
            ++keptLRhs;
      }

#endif

      // 3. infeasible constraint
      if(LTrel(lp.rhs(i), lp.lhs(i), feastol())                 ||
            (LTrel(rhsBnd,   lp.lhs(i), feastol()) && rhsCnt == 0) ||
            (GTrel(lhsBnd,   lp.rhs(i), feastol()) && lhsCnt == 0))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM19 row " << std::setprecision(20) << i
                   << ": infeasible -> lhs=" << lp.lhs(i)
                   << " rhs=" << lp.rhs(i)
                   << " lhsBnd=" << lhsBnd
                   << " rhsBnd=" << rhsBnd
                   << std::endl;)
         return this->INFEASIBLE;
      }

#if FREE_CONSTRAINT

      // 4. unconstrained constraint
      if(lp.lhs(i) <= R(-infinity) && lp.rhs(i) >= R(infinity))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM20 row " << i
                   << ": unconstrained -> removed" << std::endl;)

         std::shared_ptr<PostStep> ptr(new FreeConstraintPS(lp, i));
         m_hist.append(ptr);

         ++remRows;
         remNzos += row.size();
         removeRow(lp, i);

         ++m_stat[FREE_ROW];

         continue;
      }

#endif

#if EMPTY_CONSTRAINT

      // 5. empty constraint
      if(row.size() == 0)
      {
         MSG_DEBUG((*this->spxout) << "IMAISM21 row " << i
                   << ": empty ->";)

         if(LT(lp.rhs(i), R(0.0), feastol()) || GT(lp.lhs(i), R(0.0), feastol()))
         {
            MSG_DEBUG((*this->spxout) << " infeasible lhs=" << lp.lhs(i)
                      << " rhs=" << lp.rhs(i) << std::endl;)
            return this->INFEASIBLE;
         }

         MSG_DEBUG((*this->spxout) << " removed" << std::endl;)

         std::shared_ptr<PostStep> ptr(new EmptyConstraintPS(lp, i));
         m_hist.append(ptr);

         ++remRows;
         removeRow(lp, i);

         ++m_stat[EMPTY_ROW];

         continue;
      }

#endif

#if ROW_SINGLETON

      // 6. row singleton
      if(row.size() == 1)
      {
         removeRowSingleton(lp, row, i);
         continue;
      }

#endif

#if AGGREGATE_VARS

      // 7. row doubleton, aka. simple aggregation of two variables in an equation
      if(row.size() == 2 && EQrel(lp.lhs(i), lp.rhs(i), feastol()))
      {
         aggregateVars(lp, row, i);
         continue;
      }

#endif

#if FORCE_CONSTRAINT

      // 8. forcing constraint (postsolving)
      // fix variables to obtain the upper bound on constraint value
      if(rhsCnt == 0 && EQrel(rhsBnd, lp.lhs(i), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM24 row " << i
                   << ": forcing constraint fix on lhs ->"
                   << " lhs=" << lp.lhs(i)
                   << " rhsBnd=" << rhsBnd
                   << std::endl;)

         DataArray<bool> fixedCol(row.size());
         Array<R> lowers(row.size());
         Array<R> uppers(row.size());

         for(int k = 0; k < row.size(); ++k)
         {
            R aij = row.value(k);
            int  j   = row.index(k);

            fixedCol[k] = !(EQrel(lp.upper(j), lp.lower(j), m_epsilon));

            lowers[k] = lp.lower(j);
            uppers[k] = lp.upper(j);

            ASSERT_WARN("WMAISM25", isNotZero(aij, R(1.0 / R(infinity))));

            if(aij > 0.0)
               lp.changeLower(j, lp.upper(j));
            else
               lp.changeUpper(j, lp.lower(j));
         }

         std::shared_ptr<PostStep> ptr(new ForceConstraintPS(lp, i, true, fixedCol, lowers, uppers));
         m_hist.append(ptr);

         ++remRows;
         remNzos += row.size();
         removeRow(lp, i);

         ++m_stat[FORCE_ROW];

         continue;
      }

      // fix variables to obtain the lower bound on constraint value
      if(lhsCnt == 0 && EQrel(lhsBnd, lp.rhs(i), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM26 row " << i
                   << ": forcing constraint fix on rhs ->"
                   << " rhs=" << lp.rhs(i)
                   << " lhsBnd=" << lhsBnd
                   << std::endl;)

         DataArray<bool> fixedCol(row.size());
         Array<R> lowers(row.size());
         Array<R> uppers(row.size());

         for(int k = 0; k < row.size(); ++k)
         {
            R aij   = row.value(k);
            int  j     = row.index(k);

            fixedCol[k] = !(EQrel(lp.upper(j), lp.lower(j), m_epsilon));

            lowers[k] = lp.lower(j);
            uppers[k] = lp.upper(j);

            ASSERT_WARN("WMAISM27", isNotZero(aij, R(1.0 / R(infinity))));

            if(aij > 0.0)
               lp.changeUpper(j, lp.lower(j));
            else
               lp.changeLower(j, lp.upper(j));
         }

         std::shared_ptr<PostStep> ptr(new ForceConstraintPS(lp, i, false, fixedCol, lowers, uppers));
         m_hist.append(ptr);

         ++remRows;
         remNzos += row.size();
         removeRow(lp, i);

         ++m_stat[FORCE_ROW];

         continue;
      }

#endif
   }

   assert(remRows > 0 || remNzos == 0);

   if(remRows + chgLRhs + chgBnds > 0)
   {
      this->m_remRows += remRows;
      this->m_remNzos += remNzos;
      this->m_chgLRhs += chgLRhs;
      this->m_chgBnds += chgBnds;
      this->m_keptBnds += keptBnds;
      this->m_keptLRhs += keptLRhs;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (rows) removed "
                << remRows << " rows, "
                << remNzos << " non-zeros, "
                << chgBnds << " col bounds, "
                << chgLRhs << " row bounds; kept "
                << keptBnds << " column bounds, "
                << keptLRhs << " row bounds"
                << std::endl;)

      if(remRows > this->m_minReduction * oldRows)
         again = true;
   }

   return this->OKAY;
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::simplifyCols(SPxLPBase<R>& lp, bool& again)
{

   // This method simplifies the columns of the LP.
   //
   // The following operations are done:
   // 1. detect empty columns and fix corresponding variables
   // 2. detect variables that are unconstrained from below or above
   //    and fix corresponding variables or remove involved constraints
   // 3. fix variables
   // 4. use column singleton variables with zero objective to adjust constraint bounds
   // 5. (not free) column singleton combined with doubleton equation are
   //    used to make the column singleton variable free
   // 6. substitute (implied) free column singletons

   int remRows = 0;
   int remCols = 0;
   int remNzos = 0;
   int chgBnds = 0;

   int oldCols = lp.nCols();
   int oldRows = lp.nRows();

   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      const SVectorBase<R>& col = lp.colVector(j);

      // infeasible bounds
      if(GTrel(lp.lower(j), lp.upper(j), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM29 col " << j
                   << ": infeasible bounds on x" << j
                   << " -> lower=" << lp.lower(j)
                   << " upper=" << lp.upper(j)
                   << std::endl;)
         return this->INFEASIBLE;
      }

      // 1. empty column
      if(col.size() == 0)
      {
#if EMPTY_COLUMN
         MSG_DEBUG((*this->spxout) << "IMAISM30 col " << j
                   << ": empty -> maxObj=" << lp.maxObj(j)
                   << " lower=" << lp.lower(j)
                   << " upper=" << lp.upper(j);)

         R val;

         if(GT(lp.maxObj(j), R(0.0), this->epsZero()))
         {
            if(lp.upper(j) >= R(infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)
               return this->UNBOUNDED;
            }

            val = lp.upper(j);
         }
         else if(LT(lp.maxObj(j), R(0.0), this->epsZero()))
         {
            if(lp.lower(j) <= R(-infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)
               return this->UNBOUNDED;
            }

            val = lp.lower(j);
         }
         else
         {
            assert(isZero(lp.maxObj(j), this->epsZero()));

            // any value within the bounds is ok
            if(lp.lower(j) > R(-infinity))
               val = lp.lower(j);
            else if(lp.upper(j) < R(infinity))
               val = lp.upper(j);
            else
               val = 0.0;
         }

         MSG_DEBUG((*this->spxout) << " removed" << std::endl;)

         std::shared_ptr<PostStep> ptr1(new FixBoundsPS(lp, j, val));
         std::shared_ptr<PostStep> ptr2(new FixVariablePS(lp, *this, j, val));
         m_hist.append(ptr1);
         m_hist.append(ptr2);

         ++remCols;
         removeCol(lp, j);

         ++m_stat[EMPTY_COL];

         continue;
#endif
      }

      if(NErel(lp.lower(j), lp.upper(j), feastol()))
      {
         // will be set to false if any constraint implies a bound on the variable
         bool loFree = true;
         bool upFree = true;

         // 1. fix and remove variables
         for(int k = 0; k < col.size(); ++k)
         {
            if(!loFree && !upFree)
               break;

            int i = col.index(k);

            // warn since this unhandled case may slip through unnoticed otherwise
            ASSERT_WARN("WMAISM31", isNotZero(col.value(k), R(1.0 / R(infinity))));

            if(col.value(k) > 0.0)
            {
               if(lp.rhs(i) <  R(infinity))
                  upFree = false;

               if(lp.lhs(i) > R(-infinity))
                  loFree = false;
            }
            else if(col.value(k) < 0.0)
            {
               if(lp.rhs(i) <  R(infinity))
                  loFree = false;

               if(lp.lhs(i) > R(-infinity))
                  upFree = false;
            }
         }

         // 2. detect variables that are unconstrained from below or above
         // max  3 x
         // s.t. 5 x >= 8
         if(GT(lp.maxObj(j), R(0.0), this->epsZero()) && upFree)
         {
#if FIX_VARIABLE
            MSG_DEBUG((*this->spxout) << "IMAISM32 col " << j
                      << ": x" << j
                      << " unconstrained above ->";)

            if(lp.upper(j) >= R(infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)

               return this->UNBOUNDED;
            }

            MSG_DEBUG((*this->spxout) << " fixed at upper=" << lp.upper(j) << std::endl;)

            std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.upper(j)));
            m_hist.append(ptr);
            lp.changeLower(j, lp.upper(j));
         }
         // max -3 x
         // s.t. 5 x <= 8
         else if(LT(lp.maxObj(j), R(0.0), this->epsZero()) && loFree)
         {
            MSG_DEBUG((*this->spxout) << "IMAISM33 col " << j
                      << ": x" << j
                      << " unconstrained below ->";)

            if(lp.lower(j) <= R(-infinity))
            {
               MSG_DEBUG((*this->spxout) << " unbounded" << std::endl;)

               return this->UNBOUNDED;
            }

            MSG_DEBUG((*this->spxout) << " fixed at lower=" << lp.lower(j) << std::endl;)

            std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.lower(j)));
            m_hist.append(ptr);
            lp.changeUpper(j, lp.lower(j));
#endif
         }
         else if(isZero(lp.maxObj(j), this->epsZero()))
         {
#if FREE_ZERO_OBJ_VARIABLE
            bool unconstrained_below = loFree && lp.lower(j) <= R(-infinity);
            bool unconstrained_above = upFree && lp.upper(j) >= R(infinity);

            if(unconstrained_below || unconstrained_above)
            {
               MSG_DEBUG((*this->spxout) << "IMAISM34 col " << j
                         << ": x" << j
                         << " unconstrained "
                         << (unconstrained_below ? "below" : "above")
                         << " with zero objective (" << lp.maxObj(j)
                         << ")" << std::endl;)

               DSVectorBase<R> col_idx_sorted(col);

               // sort col elements by increasing idx
               IdxCompare compare;
               SPxQuicksort(col_idx_sorted.mem(), col_idx_sorted.size(), compare);

               std::shared_ptr<PostStep> ptr(new FreeZeroObjVariablePS(lp, j, unconstrained_below,
                                             col_idx_sorted));
               m_hist.append(ptr);

               // we have to remove the rows with larger idx first, because otherwise the rows are reorder and indices
               // are out-of-date
               remRows += col.size();

               for(int k = col_idx_sorted.size() - 1; k >= 0; --k)
                  removeRow(lp, col_idx_sorted.index(k));

               // remove column
               removeCol(lp, j);

               // statistics
               for(int k = 0; k < col.size(); ++k)
               {
                  int l   =  col.index(k);
                  remNzos += lp.rowVector(l).size();
               }

               ++m_stat[FREE_ZOBJ_COL];
               ++remCols;

               continue;
            }

#endif
         }
      }

#if FIX_VARIABLE

      // 3. fix variable
      if(EQrel(lp.lower(j), lp.upper(j), feastol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM36 col " << j
                   << ": x" << j
                   << " fixed -> lower=" << lp.lower(j)
                   << " upper=" << lp.upper(j) << std::endl;)

         fixColumn(lp, j);

         ++remCols;
         remNzos += col.size();
         removeCol(lp, j);

         ++m_stat[FIX_COL];

         continue;
      }

#endif

      // handle column singletons
      if(col.size() == 1)
      {
         R aij = col.value(0);
         int  i   = col.index(0);

         // 4. column singleton with zero objective
         if(isZero(lp.maxObj(j), this->epsZero()))
         {
#if ZERO_OBJ_COL_SINGLETON
            MSG_DEBUG((*this->spxout) << "IMAISM37 col " << j
                      << ": singleton in row " << i
                      << " with zero objective";)

            R lhs = R(-infinity);
            R rhs = +R(infinity);

            if(GT(aij, R(0.0), this->epsZero()))
            {
               if(lp.lhs(i) > R(-infinity) && lp.upper(j) <  R(infinity))
                  lhs = lp.lhs(i) - aij * lp.upper(j);

               if(lp.rhs(i) <  R(infinity) && lp.lower(j) > R(-infinity))
                  rhs = lp.rhs(i) - aij * lp.lower(j);
            }
            else if(LT(aij, R(0.0), this->epsZero()))
            {
               if(lp.lhs(i) > R(-infinity) && lp.lower(j) > R(-infinity))
                  lhs = lp.lhs(i) - aij * lp.lower(j);

               if(lp.rhs(i) <  R(infinity) && lp.upper(j) <  R(infinity))
                  rhs = lp.rhs(i) - aij * lp.upper(j);
            }
            else
            {
               lhs = lp.lhs(i);
               rhs = lp.rhs(i);
            }

            if(isZero(lhs, this->epsZero()))
               lhs = 0.0;

            if(isZero(rhs, this->epsZero()))
               rhs = 0.0;

            MSG_DEBUG((*this->spxout) << " removed -> lhs=" << lhs
                      << " (" << lp.lhs(i)
                      << ") rhs=" << rhs
                      << " (" << lp.rhs(i)
                      << ")" << std::endl;)

            std::shared_ptr<PostStep> ptr(new ZeroObjColSingletonPS(lp, *this, j, i));
            m_hist.append(ptr);

            lp.changeRange(i, lhs, rhs);

            ++remCols;
            ++remNzos;
            removeCol(lp, j);

            ++m_stat[ZOBJ_SINGLETON_COL];

            if(lp.lhs(i) <= R(-infinity) && lp.rhs(i) >= R(infinity))
            {
               std::shared_ptr<PostStep> ptr2(new FreeConstraintPS(lp, i));
               m_hist.append(ptr2);

               ++remRows;
               removeRow(lp, i);

               ++m_stat[FREE_ROW];
            }

            continue;
#endif
         }

         // 5. not free column singleton combined with doubleton equation
         else if(EQrel(lp.lhs(i), lp.rhs(i), feastol())             &&
                 lp.rowVector(i).size() == 2                         &&
                 (lp.lower(j) > R(-infinity) || lp.upper(j) < R(infinity)))
         {
#if DOUBLETON_EQUATION
            MSG_DEBUG((*this->spxout) << "IMAISM38 col " << j
                      << ": singleton in row " << i
                      << " with doubleton equation ->";)

            R lhs = lp.lhs(i);

            const SVectorBase<R>& row = lp.rowVector(i);

            R aik;
            int  k;

            if(row.index(0) == j)
            {
               aik = row.value(1);
               k   = row.index(1);
            }
            else if(row.index(1) == j)
            {
               aik = row.value(0);
               k   = row.index(0);
            }
            else
               throw SPxInternalCodeException("XMAISM11 This should never happen.");

            ASSERT_WARN("WMAISM39", isNotZero(aik, R(1.0 / R(infinity))));

            R lo, up;
            R oldLower = lp.lower(k);
            R oldUpper = lp.upper(k);

            R scale1 = maxAbs(lhs, aij * lp.upper(j));
            R scale2 = maxAbs(lhs, aij * lp.lower(j));

            if(scale1 < 1.0)
               scale1 = 1.0;

            if(scale2 < 1.0)
               scale2 = 1.0;

            R z1 = (lhs / scale1) - (aij * lp.upper(j) / scale1);
            R z2 = (lhs / scale2) - (aij * lp.lower(j) / scale2);

            if(isZero(z1, this->epsZero()))
               z1 = 0.0;

            if(isZero(z2, this->epsZero()))
               z2 = 0.0;

            if(aij * aik > 0.0)
            {
               lo = (lp.upper(j) >=  R(infinity)) ? R(-infinity) : z1 * scale1 / aik;
               up = (lp.lower(j) <= R(-infinity)) ?  R(infinity) : z2 * scale2 / aik;
            }
            else if(aij * aik < 0.0)
            {
               lo = (lp.lower(j) <= R(-infinity)) ? R(-infinity) : z2 * scale2 / aik;
               up = (lp.upper(j) >=  R(infinity)) ?  R(infinity) : z1 * scale1 / aik;
            }
            else
               throw SPxInternalCodeException("XMAISM12 This should never happen.");

            if(GTrel(lo, lp.lower(k), this->epsZero()))
               lp.changeLower(k, lo);

            if(LTrel(up, lp.upper(k), this->epsZero()))
               lp.changeUpper(k, up);

            MSG_DEBUG((*this->spxout) << " made free, bounds on x" << k
                      << ": lower=" << lp.lower(k)
                      << " (" << oldLower
                      << ") upper=" << lp.upper(k)
                      << " (" << oldUpper
                      << ")" << std::endl;)

            // infeasible bounds
            if(GTrel(lp.lower(k), lp.upper(k), feastol()))
            {
               MSG_DEBUG((*this->spxout) << "new bounds are infeasible"
                         << std::endl;)
               return this->INFEASIBLE;
            }

            std::shared_ptr<PostStep> ptr(new DoubletonEquationPS(lp, j, k, i, oldLower, oldUpper));
            m_hist.append(ptr);

            if(lp.lower(j) > R(-infinity) && lp.upper(j) < R(infinity))
               chgBnds += 2;
            else
               ++chgBnds;

            lp.changeBounds(j, R(-infinity), R(infinity));

            ++m_stat[DOUBLETON_ROW];
#endif
         }

         // 6. (implied) free column singleton
         if(lp.lower(j) <= R(-infinity) && lp.upper(j) >= R(infinity))
         {
#if FREE_COL_SINGLETON
            R slackVal = lp.lhs(i);

            // constraint i is an inequality constraint -> transform into equation type
            if(NErel(lp.lhs(i), lp.rhs(i), feastol()))
            {
               MSG_DEBUG((*this->spxout) << "IMAISM40 col " << j
                         << ": free singleton in inequality constraint" << std::endl;)

               // do nothing if constraint i is unconstrained
               if(lp.lhs(i) <= R(-infinity) && lp.rhs(i) >= R(infinity))
                  continue;

               // introduce slack variable to obtain equality constraint
               R sMaxObj = lp.maxObj(j) / aij; // after substituting variable j in objective
               R sLo     = lp.lhs(i);
               R sUp     = lp.rhs(i);

               if(GT(sMaxObj, R(0.0), this->epsZero()))
               {
                  if(sUp >= R(infinity))
                  {
                     MSG_DEBUG((*this->spxout) << " -> problem unbounded" << std::endl;)
                     return this->UNBOUNDED;
                  }

                  slackVal = sUp;
               }
               else if(LT(sMaxObj, R(0.0), this->epsZero()))
               {
                  if(sLo <= R(-infinity))
                  {
                     MSG_DEBUG((*this->spxout) << " -> problem unbounded" << std::endl;)
                     return this->UNBOUNDED;
                  }

                  slackVal = sLo;
               }
               else
               {
                  assert(isZero(sMaxObj, this->epsZero()));

                  // any value within the bounds is ok
                  if(sLo > R(-infinity))
                     slackVal = sLo;
                  else if(sUp < R(infinity))
                     slackVal = sUp;
                  else
                     throw SPxInternalCodeException("XMAISM13 This should never happen.");
               }
            }

            std::shared_ptr<PostStep> ptr(new FreeColSingletonPS(lp, *this, j, i, slackVal));
            m_hist.append(ptr);

            MSG_DEBUG((*this->spxout) << "IMAISM41 col " << j
                      << ": free singleton removed" << std::endl;)

            const SVectorBase<R>& row = lp.rowVector(i);

            for(int h = 0; h < row.size(); ++h)
            {
               int k = row.index(h);

               if(k != j)
               {
                  R new_obj = lp.obj(k) - (lp.obj(j) * row.value(h) / aij);
                  lp.changeObj(k, new_obj);
               }
            }

            ++remRows;
            ++remCols;
            remNzos += row.size();
            removeRow(lp, i);
            removeCol(lp, j);

            ++m_stat[FREE_SINGLETON_COL];

            continue;
#endif
         }
      }
   }

   if(remCols + remRows > 0)
   {
      this->m_remRows += remRows;
      this->m_remCols += remCols;
      this->m_remNzos += remNzos;
      this->m_chgBnds += chgBnds;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (columns) removed "
                << remRows << " rows, "
                << remCols << " cols, "
                << remNzos << " non-zeros, "
                << chgBnds << " col bounds"
                << std::endl;)

      if(remCols + remRows > this->m_minReduction * (oldCols + oldRows))
         again = true;
   }

   return this->OKAY;
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::simplifyDual(SPxLPBase<R>& lp, bool& again)
{

   // This method simplifies LP using the following dual structures:
   //
   // 1. dominated columns
   // 2. weakly dominated columns
   //
   // For constructing the dual variables, it is assumed that the objective sense is max

   int remRows = 0;
   int remCols = 0;
   int remNzos = 0;

   int oldRows = lp.nRows();
   int oldCols = lp.nCols();

   DataArray<bool> colSingleton(lp.nCols());
   VectorBase<R>         dualVarLo(lp.nRows());
   VectorBase<R>         dualVarUp(lp.nRows());
   VectorBase<R>         dualConsLo(lp.nCols());
   VectorBase<R>         dualConsUp(lp.nCols());

   // init
   for(int i = lp.nRows() - 1; i >= 0; --i)
   {
      // check for unconstrained constraints
      if(lp.lhs(i) <= R(-infinity) && lp.rhs(i) >= R(infinity))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM43 row " << i
                   << ": unconstrained" << std::endl;)

         std::shared_ptr<PostStep> ptr(new FreeConstraintPS(lp, i));
         m_hist.append(ptr);

         ++remRows;
         remNzos += lp.rowVector(i).size();
         removeRow(lp, i);

         ++m_stat[FREE_ROW];

         continue;
      }

      // corresponds to maximization sense
      dualVarLo[i] = (lp.lhs(i) <= R(-infinity)) ? 0.0 : R(-infinity);
      dualVarUp[i] = (lp.rhs(i) >=  R(infinity)) ? 0.0 :  R(infinity);
   }

   // compute bounds on the dual variables using column singletons
   for(int j = 0; j < lp.nCols(); ++j)
   {
      if(lp.colVector(j).size() == 1)
      {
         int  i   = lp.colVector(j).index(0);
         R aij = lp.colVector(j).value(0);

         ASSERT_WARN("WMAISM44", isNotZero(aij, R(1.0 / R(infinity))));

         R bound = lp.maxObj(j) / aij;

         if(aij > 0)
         {
            if(lp.lower(j) <= R(-infinity) && bound < dualVarUp[i])
               dualVarUp[i] = bound;

            if(lp.upper(j) >=  R(infinity) && bound > dualVarLo[i])
               dualVarLo[i] = bound;
         }
         else if(aij < 0)
         {
            if(lp.lower(j) <= R(-infinity) && bound > dualVarLo[i])
               dualVarLo[i] = bound;

            if(lp.upper(j) >=  R(infinity) && bound < dualVarUp[i])
               dualVarUp[i] = bound;
         }
      }

   }

   // compute bounds on the dual constraints
   for(int j = 0; j < lp.nCols(); ++j)
   {
      dualConsLo[j] = dualConsUp[j] = 0.0;

      const SVectorBase<R>& col = lp.colVector(j);

      for(int k = 0; k < col.size(); ++k)
      {
         if(dualConsLo[j] <= R(-infinity) && dualConsUp[j] >= R(infinity))
            break;

         R aij = col.value(k);
         int  i   = col.index(k);

         ASSERT_WARN("WMAISM45", isNotZero(aij, R(1.0 / R(infinity))));

         if(aij > 0)
         {
            if(dualVarLo[i] <= R(-infinity))
               dualConsLo[j] = R(-infinity);
            else
               dualConsLo[j] += aij * dualVarLo[i];

            if(dualVarUp[i] >= R(infinity))
               dualConsUp[j] = R(infinity);
            else
               dualConsUp[j] += aij * dualVarUp[i];
         }
         else if(aij < 0)
         {
            if(dualVarLo[i] <= R(-infinity))
               dualConsUp[j] = R(infinity);
            else
               dualConsUp[j] += aij * dualVarLo[i];

            if(dualVarUp[i] >= R(infinity))
               dualConsLo[j] = R(-infinity);
            else
               dualConsLo[j] += aij * dualVarUp[i];
         }
      }
   }

   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      if(lp.colVector(j).size() <= 1)
         continue;

      // dual infeasibility checks
      if(LTrel(dualConsUp[j], dualConsLo[j], opttol()))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM46 col " << j
                   << ": dual infeasible -> dual lhs bound=" << dualConsLo[j]
                   << " dual rhs bound=" << dualConsUp[j] << std::endl;)
         return this->DUAL_INFEASIBLE;
      }

      R obj = lp.maxObj(j);

      // 1. dominated column
      // Is the problem really unbounded in the cases below ??? Or is only dual infeasibility be shown
      if(GTrel(obj, dualConsUp[j], opttol()))
      {
#if DOMINATED_COLUMN
         MSG_DEBUG((*this->spxout) << "IMAISM47 col " << j
                   << ": dominated -> maxObj=" << obj
                   << " dual rhs bound=" << dualConsUp[j] << std::endl;)

         if(lp.upper(j) >= R(infinity))
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << " unbounded" << std::endl;)
            return this->UNBOUNDED;
         }

         MSG_DEBUG((*this->spxout) << " fixed at upper=" << lp.upper(j) << std::endl;)

         std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.upper(j)));
         m_hist.append(ptr);
         lp.changeLower(j, lp.upper(j));

         ++m_stat[DOMINATED_COL];
#endif
      }
      else if(LTrel(obj, dualConsLo[j], opttol()))
      {
#if DOMINATED_COLUMN
         MSG_DEBUG((*this->spxout) << "IMAISM48 col " << j
                   << ": dominated -> maxObj=" << obj
                   << " dual lhs bound=" << dualConsLo[j] << std::endl;)

         if(lp.lower(j) <= R(-infinity))
         {
            MSG_INFO2((*this->spxout), (*this->spxout) << " unbounded" << std::endl;)
            return this->UNBOUNDED;
         }

         MSG_DEBUG((*this->spxout) << " fixed at lower=" << lp.lower(j) << std::endl;)

         std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.lower(j)));
         m_hist.append(ptr);
         lp.changeUpper(j, lp.lower(j));

         ++m_stat[DOMINATED_COL];
#endif
      }

      // 2. weakly dominated column (no postsolving)
      else if(lp.upper(j) < R(infinity) && EQrel(obj, dualConsUp[j], opttol()))
      {
#if WEAKLY_DOMINATED_COLUMN
         MSG_DEBUG((*this->spxout) << "IMAISM49 col " << j
                   << ": weakly dominated -> maxObj=" << obj
                   << " dual rhs bound=" << dualConsUp[j] << std::endl;)

         std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.upper(j)));
         m_hist.append(ptr);
         lp.changeLower(j, lp.upper(j));

         ++m_stat[WEAKLY_DOMINATED_COL];
#endif
      }
      else if(lp.lower(j) > R(-infinity) && EQrel(obj, dualConsLo[j], opttol()))
      {
#if WEAKLY_DOMINATED_COLUMN
         MSG_DEBUG((*this->spxout) << "IMAISM50 col " << j
                   << ": weakly dominated -> maxObj=" << obj
                   << " dual lhs bound=" << dualConsLo[j] << std::endl;)

         std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j, lp.lower(j)));
         m_hist.append(ptr);
         lp.changeUpper(j, lp.lower(j));

         ++m_stat[WEAKLY_DOMINATED_COL];
#endif
      }

      // fix column
      if(EQrel(lp.lower(j), lp.upper(j), feastol()))
      {
#if FIX_VARIABLE
         fixColumn(lp, j);

         ++remCols;
         remNzos += lp.colVector(j).size();
         removeCol(lp, j);

         ++m_stat[FIX_COL];
#endif
      }
   }


   assert(remRows > 0 || remCols > 0 || remNzos == 0);

   if(remCols + remRows > 0)
   {
      this->m_remRows += remRows;
      this->m_remCols += remCols;
      this->m_remNzos += remNzos;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (dual) removed "
                << remRows << " rows, "
                << remCols << " cols, "
                << remNzos << " non-zeros"
                << std::endl;)

      if(remCols + remRows > this->m_minReduction * (oldCols + oldRows))
         again = true;
   }

   return this->OKAY;
}



template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::multiaggregation(SPxLPBase<R>& lp, bool& again)
{
   // this simplifier eliminates rows and columns by performing multi aggregations as identified by the constraint
   // activities.
   int remRows = 0;
   int remCols = 0;
   int remNzos = 0;

   int oldRows = lp.nRows();
   int oldCols = lp.nCols();

   VectorBase<R> upLocks(lp.nCols());
   VectorBase<R> downLocks(lp.nCols());

   for(int j = lp.nCols() - 1; j >= 0; --j)
   {
      // setting the locks on the variables
      upLocks[j] = 0;
      downLocks[j] = 0;

      if(lp.colVector(j).size() <= 1)
         continue;

      const SVectorBase<R>& col = lp.colVector(j);

      for(int k = 0; k < col.size(); ++k)
      {
         R aij = col.value(k);

         ASSERT_WARN("WMAISM45", isNotZero(aij, R(1.0 / R(infinity))));

         if(GT(lp.lhs(col.index(k)), R(-infinity)) && LT(lp.rhs(col.index(k)), R(infinity)))
         {
            upLocks[j]++;
            downLocks[j]++;
         }
         else if(GT(lp.lhs(col.index(k)), R(-infinity)))
         {
            if(aij > 0)
               downLocks[j]++;
            else if(aij < 0)
               upLocks[j]++;
         }
         else if(LT(lp.rhs(col.index(k)), R(infinity)))
         {
            if(aij > 0)
               upLocks[j]++;
            else if(aij < 0)
               downLocks[j]++;
         }
      }

      // multi-aggregate column
      if(upLocks[j] == 1 || downLocks[j] == 1)
      {
         R lower = lp.lower(j);
         R upper = lp.upper(j);
         int maxOtherLocks;
         int bestpos = -1;
         bool bestislhs = true;

         for(int k = 0; k < col.size(); ++k)
         {
            int rowNumber;
            R lhs;
            R rhs;
            bool lhsExists;
            bool rhsExists;
            bool aggLhs;
            bool aggRhs;

            R val = col.value(k);

            rowNumber = col.index(k);
            lhs = lp.lhs(rowNumber);
            rhs = lp.rhs(rowNumber);

            if(EQ(lhs, rhs, feastol()))
               continue;

            lhsExists = GT(lhs, R(-infinity));
            rhsExists = LT(rhs, R(infinity));

            if(lp.rowVector(rowNumber).size() <= 2)
               maxOtherLocks = INT_MAX;
            else if(lp.rowVector(rowNumber).size() == 3)
               maxOtherLocks = 3;
            else if(lp.rowVector(rowNumber).size() == 4)
               maxOtherLocks = 2;
            else
               maxOtherLocks = 1;

            aggLhs = lhsExists
                     && ((col.value(k) > 0.0 && lp.maxObj(j) <= 0.0 && downLocks[j] == 1 && upLocks[j] <= maxOtherLocks)
                         || (col.value(k) < 0.0 && lp.maxObj(j) >= 0.0 && upLocks[j] == 1 && downLocks[j] <= maxOtherLocks));
            aggRhs = rhsExists
                     && ((col.value(k) > 0.0 && lp.maxObj(j) >= 0.0 && upLocks[j] == 1 && downLocks[j] <= maxOtherLocks)
                         || (col.value(k) < 0.0 && lp.maxObj(j) <= 0.0 && downLocks[j] == 1 && upLocks[j] <= maxOtherLocks));

            if(aggLhs || aggRhs)
            {
               R minRes = 0;   // this is the minimum value that the aggregation can attain
               R maxRes = 0;   // this is the maximum value that the aggregation can attain

               // computing the minimum and maximum residuals if variable j is set to zero.
               computeMinMaxResidualActivity(lp, rowNumber, j, minRes, maxRes);

               // we will try to aggregate to the lhs
               if(aggLhs)
               {
                  R minVal;
                  R maxVal;

                  // computing the values of the upper and lower bounds for the aggregated variables
                  computeMinMaxValues(lp, lhs, val, minRes, maxRes, minVal, maxVal);

                  assert(LE(minVal, maxVal));

                  // if the bounds of the aggregation and the original variable are equivalent, then we can reduce
                  if((minVal > R(-infinity) && GT(minVal, lower, feastol()))
                        && (maxVal < R(infinity) && LT(maxVal, upper, feastol())))
                  {
                     bestpos = col.index(k);
                     bestislhs = true;
                     break;
                  }
               }

               // we will try to aggregate to the rhs
               if(aggRhs)
               {
                  R minVal;
                  R maxVal;

                  // computing the values of the upper and lower bounds for the aggregated variables
                  computeMinMaxValues(lp, rhs, val, minRes, maxRes, minVal, maxVal);

                  assert(LE(minVal, maxVal));

                  if((minVal > R(-infinity) && GT(minVal, lower, feastol()))
                        && (maxVal < R(infinity) && LT(maxVal, upper, feastol())))
                  {
                     bestpos = col.index(k);
                     bestislhs = false;
                     break;
                  }
               }
            }
         }

         // it is only possible to aggregate if a best position has been found
         if(bestpos >= 0)
         {
            const SVectorBase<R>& bestRow = lp.rowVector(bestpos);
            // aggregating the variable and applying the fixings to the all other constraints
            R aggConstant = (bestislhs ? lp.lhs(bestpos) : lp.rhs(
                                bestpos));   // this is the lhs or rhs of the aggregated row
            R aggAij =
               bestRow[j];                                   // this is the coefficient of the deleted col

            MSG_DEBUG(
               (*this->spxout) << "IMAISM51 col " << j
               << ": Aggregating row: " << bestpos
               << " Aggregation Constant=" << aggConstant
               << " Coefficient of aggregated col=" << aggAij << std::endl;
            )

            std::shared_ptr<PostStep> ptr(new MultiAggregationPS(lp, *this, bestpos, j, aggConstant));
            m_hist.append(ptr);

            for(int k = 0; k < col.size(); ++k)
            {
               if(col.index(k) != bestpos)
               {
                  int rowNumber = col.index(k);
                  VectorBase<R> updateRow(lp.nCols());
                  R updateRhs = lp.rhs(col.index(k));
                  R updateLhs = lp.lhs(col.index(k));

                  updateRow = lp.rowVector(col.index(k));

                  // updating the row with the best row
                  for(int l = 0; l < bestRow.size(); l++)
                  {
                     if(bestRow.index(l) != j)
                     {
                        if(lp.rowVector(rowNumber).pos(bestRow.index(l)) >= 0)
                           lp.changeElement(rowNumber, bestRow.index(l), updateRow[bestRow.index(l)]
                                            - updateRow[j]*bestRow.value(l) / aggAij);
                        else
                           lp.changeElement(rowNumber, bestRow.index(l), -1.0 * updateRow[j]*bestRow.value(l) / aggAij);
                     }
                  }

                  // NOTE: I don't know whether we should change the LHS and RHS if they are currently at R(infinity)
                  if(LT(lp.rhs(rowNumber), R(infinity)))
                     lp.changeRhs(rowNumber, updateRhs - updateRow[j]*aggConstant / aggAij);

                  if(GT(lp.lhs(rowNumber), R(-infinity)))
                     lp.changeLhs(rowNumber, updateLhs - updateRow[j]*aggConstant / aggAij);

                  assert(LE(lp.lhs(rowNumber), lp.rhs(rowNumber)));
               }
            }

            for(int l = 0; l < bestRow.size(); l++)
            {
               if(bestRow.index(l) != j)
                  lp.changeMaxObj(bestRow.index(l),
                                  lp.maxObj(bestRow.index(l)) - lp.maxObj(j)*bestRow.value(l) / aggAij);
            }

            ++remCols;
            remNzos += lp.colVector(j).size();
            removeCol(lp, j);
            ++remRows;
            remNzos += lp.rowVector(bestpos).size();
            removeRow(lp, bestpos);

            ++m_stat[MULTI_AGG];
         }
      }
   }


   assert(remRows > 0 || remCols > 0 || remNzos == 0);

   if(remCols + remRows > 0)
   {
      this->m_remRows += remRows;
      this->m_remCols += remCols;
      this->m_remNzos += remNzos;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (multi-aggregation) removed "
                << remRows << " rows, "
                << remCols << " cols, "
                << remNzos << " non-zeros"
                << std::endl;)

      if(remCols + remRows > this->m_minReduction * (oldCols + oldRows))
         again = true;
   }

   return this->OKAY;
}



template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::duplicateRows(SPxLPBase<R>& lp, bool& again)
{

   // This method simplifies the LP by removing duplicate rows
   // Duplicates are detected using the algorithm of Bixby and Wagner [1987]

   // Possible extension: use generalized definition of duplicate rows according to Andersen and Andersen
   // However: the resulting sparsification is often very small since the involved rows are usually very sparse

   int remRows = 0;
   int remNzos = 0;

   int oldRows = lp.nRows();

   // remove empty rows and columns
   typename SPxSimplifier<R>::Result ret = removeEmpty(lp);

   if(ret != this->OKAY)
      return ret;

#if ROW_SINGLETON
   int rs_remRows = 0;

   for(int i = 0; i < lp.nRows(); ++i)
   {
      const SVectorBase<R>& row = lp.rowVector(i);

      if(row.size() == 1)
      {
         removeRowSingleton(lp, row, i);
         rs_remRows++;
      }
   }

   if(rs_remRows > 0)
   {
      MSG_INFO2((*this->spxout), (*this->spxout) <<
                "Simplifier duplicate rows (row singleton stage) removed "
                << rs_remRows << " rows, "
                << rs_remRows << " non-zeros"
                << std::endl;)
   }

#endif

   if(lp.nRows() < 2)
      return this->OKAY;

   DataArray<int>    pClass(lp.nRows());           // class of parallel rows
   DataArray<int>    classSize(lp.nRows());        // size of each class
   Array<R>   scale(lp.nRows());            // scaling factor for each row
   int*              idxMem = 0;

   try
   {
      spx_alloc(idxMem, lp.nRows());
   }
   catch(const SPxMemoryException& x)
   {
      spx_free(idxMem);
      throw x;
   }

   IdxSet idxSet(lp.nRows(), idxMem);           // set of feasible indices for new pClass

   // init
   pClass[0]    = 0;
   scale[0]     = 0.0;
   classSize[0] = lp.nRows();

   for(int i = 1; i < lp.nRows(); ++i)
   {
      pClass[i] = 0;
      scale[i]  = 0.0;
      classSize[i] = 0;
      idxSet.addIdx(i);
   }

   R oldVal = 0.0;

   // main loop
   for(int j = 0; j < lp.nCols(); ++j)
   {
      const SVectorBase<R>& col = lp.colVector(j);

      for(int k = 0; k < col.size(); ++k)
      {
         R aij = col.value(k);
         int  i   = col.index(k);

         if(scale[i] == 0.0)
            scale[i] = aij;

         m_classSetRows[pClass[i]].add(i, aij / scale[i]);

         if(--classSize[pClass[i]] == 0)
            idxSet.addIdx(pClass[i]);
      }

      // update each parallel class with non-zero column entry
      for(int m = 0; m < col.size(); ++m)
      {
         int k = pClass[col.index(m)];

         if(m_classSetRows[k].size() > 0)
         {
            // sort classSet[k] w.r.t. scaled column values
            ElementCompare compare;

            if(m_classSetRows[k].size() > 1)
               SPxQuicksort(m_classSetRows[k].mem(), m_classSetRows[k].size(), compare);

            // use new index first
            int classIdx = idxSet.index(0);
            idxSet.remove(0);

            for(int l = 0; l < m_classSetRows[k].size(); ++l)
            {
               if(l != 0 && NErel(m_classSetRows[k].value(l), oldVal, this->epsZero()))
               {
                  classIdx = idxSet.index(0);
                  idxSet.remove(0);
               }

               pClass[m_classSetRows[k].index(l)] = classIdx;
               ++classSize[classIdx];

               oldVal = m_classSetRows[k].value(l);
            }

            m_classSetRows[k].clear();
         }
      }
   }

   spx_free(idxMem);

   DataArray<bool> remRow(lp.nRows());

   for(int k = 0; k < lp.nRows(); ++k)
      m_dupRows[k].clear();

   for(int k = 0; k < lp.nRows(); ++k)
   {
      remRow[k] = false;
      m_dupRows[pClass[k]].add(k, 0.0);
   }

   const int nRowsOld_tmp = lp.nRows();
   int* perm_tmp = 0;
   spx_alloc(perm_tmp, nRowsOld_tmp);

   for(int j = 0; j < nRowsOld_tmp; ++j)
   {
      perm_tmp[j] = 0;
   }

   int idxFirstDupRows = -1;
   int idxLastDupRows = -1;
   int numDelRows = 0;

   for(int k = 0; k < lp.nRows(); ++k)
   {
      if(m_dupRows[k].size() > 1 && !(lp.rowVector(m_dupRows[k].index(0)).size() == 1))
      {
         idxLastDupRows = k;

         if(idxFirstDupRows < 0)
         {
            idxFirstDupRows = k;
         }

         for(int l = 1; l < m_dupRows[k].size(); ++l)
         {
            int i = m_dupRows[k].index(l);
            perm_tmp[i] = -1;
         }

         numDelRows += (m_dupRows[k].size() - 1);
      }
   }

   {
      int k_tmp, j_tmp = -1;

      for(k_tmp = j_tmp = 0; k_tmp < nRowsOld_tmp; ++k_tmp)
      {
         if(perm_tmp[k_tmp] >= 0)
            perm_tmp[k_tmp] = j_tmp++;
      }
   }

   // store rhs and lhs changes for combined update
   bool doChangeRanges = false;
   VectorBase<R> newLhsVec(lp.lhs());
   VectorBase<R> newRhsVec(lp.rhs());

   for(int k = 0; k < lp.nRows(); ++k)
   {
      if(m_dupRows[k].size() > 1 && !(lp.rowVector(m_dupRows[k].index(0)).size() == 1))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM53 " << m_dupRows[k].size()
                   << " duplicate rows found" << std::endl;)

         m_stat[DUPLICATE_ROW] += m_dupRows[k].size() - 1;

         // index of one non-column singleton row in dupRows[k]
         int  rowIdx    = -1;
         int  maxLhsIdx = -1;
         int  minRhsIdx = -1;
         R maxLhs    = R(-infinity);
         R minRhs    = +R(infinity);

         DataArray<bool> isLhsEqualRhs(m_dupRows[k].size());

         // determine strictest bounds on constraint
         for(int l = 0; l < m_dupRows[k].size(); ++l)
         {
            int i = m_dupRows[k].index(l);
            isLhsEqualRhs[l] = (lp.lhs(i) == lp.rhs(i));

            ASSERT_WARN("WMAISM54", isNotZero(scale[i], R(1.0 / R(infinity))));

            if(rowIdx == -1)
            {
               rowIdx = i;
               maxLhs = lp.lhs(rowIdx);
               minRhs = lp.rhs(rowIdx);
            }
            else
            {
               R scaledLhs, scaledRhs;
               R factor = scale[rowIdx] / scale[i];

               if(factor > 0)
               {
                  scaledLhs = (lp.lhs(i) <= R(-infinity)) ? R(-infinity) : lp.lhs(i) * factor;
                  scaledRhs = (lp.rhs(i) >=  R(infinity)) ?  R(infinity) : lp.rhs(i) * factor;
               }
               else
               {
                  scaledLhs = (lp.rhs(i) >=  R(infinity)) ? R(-infinity) : lp.rhs(i) * factor;
                  scaledRhs = (lp.lhs(i) <= R(-infinity)) ?  R(infinity) : lp.lhs(i) * factor;
               }

               if(scaledLhs > maxLhs)
               {
                  maxLhs    = scaledLhs;
                  maxLhsIdx = i;
               }

               if(scaledRhs < minRhs)
               {
                  minRhs    = scaledRhs;
                  minRhsIdx = i;
               }

               remRow[i] = true;
            }
         }

         if(rowIdx != -1)
         {
            R newLhs = (maxLhs > lp.lhs(rowIdx)) ? maxLhs : lp.lhs(rowIdx);
            R newRhs = (minRhs < lp.rhs(rowIdx)) ? minRhs : lp.rhs(rowIdx);

            if(k == idxLastDupRows)
            {
               DataArray<int> da_perm(nRowsOld_tmp);

               for(int j = 0; j < nRowsOld_tmp; ++j)
               {
                  da_perm[j] = perm_tmp[j];
               }

               std::shared_ptr<PostStep> ptr(new DuplicateRowsPS(lp, rowIdx, maxLhsIdx, minRhsIdx,
                                             m_dupRows[k], scale, da_perm, isLhsEqualRhs, true,
                                             EQrel(newLhs, newRhs), k == idxFirstDupRows));
               m_hist.append(ptr);
            }
            else
            {
               DataArray<int> da_perm_empty(0);
               std::shared_ptr<PostStep> ptr(new DuplicateRowsPS(lp, rowIdx, maxLhsIdx, minRhsIdx,
                                             m_dupRows[k], scale, da_perm_empty, isLhsEqualRhs, false, EQrel(newLhs, newRhs),
                                             k == idxFirstDupRows));
               m_hist.append(ptr);
            }

            if(maxLhs > lp.lhs(rowIdx) || minRhs < lp.rhs(rowIdx))
            {
               // modify lhs and rhs of constraint rowIdx
               doChangeRanges = true;

               if(LTrel(newRhs, newLhs, feastol()))
               {
                  MSG_DEBUG((*this->spxout) << "IMAISM55 duplicate rows yield infeasible bounds:"
                            << " lhs=" << newLhs
                            << " rhs=" << newRhs << std::endl;)
                  spx_free(perm_tmp);
                  return this->INFEASIBLE;
               }

               // if we accept the infeasibility we should clean up the values to avoid problems later
               if(newRhs < newLhs)
                  newRhs = newLhs;

               newLhsVec[rowIdx] = newLhs;
               newRhsVec[rowIdx] = newRhs;
            }
         }
      }
   }

   // change ranges for all modified constraints by one single call (more efficient)
   if(doChangeRanges)
   {
      lp.changeRange(newLhsVec, newRhsVec);
   }

   // remove all rows by one single method call (more efficient)
   const int nRowsOld = lp.nRows();
   int* perm = 0;
   spx_alloc(perm, nRowsOld);

   for(int i = 0; i < nRowsOld; ++i)
   {
      if(remRow[i])
      {
         perm[i] = -1;
         ++remRows;
         remNzos += lp.rowVector(i).size();
      }
      else
         perm[i] = 0;
   }

   lp.removeRows(perm);

   for(int i = 0; i < nRowsOld; ++i)
   {
      // assert that the pre-computed permutation was correct
      assert(perm[i] == perm_tmp[i]);

      // update the global index mapping
      if(perm[i] >= 0)
         m_rIdx[perm[i]] = m_rIdx[i];
   }

   spx_free(perm);
   spx_free(perm_tmp);

   if(remRows + remNzos > 0)
   {
      this->m_remRows += remRows;
      this->m_remNzos += remNzos;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (duplicate rows) removed "
                << remRows << " rows, "
                << remNzos << " non-zeros"
                << std::endl;)

      if(remRows > this->m_minReduction * oldRows)
         again = true;

   }

   return this->OKAY;
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::duplicateCols(SPxLPBase<R>& lp, bool& again)
{

   // This method simplifies the LP by removing duplicate columns
   // Duplicates are detected using the algorithm of Bixby and Wagner [1987]

   int remCols = 0;
   int remNzos = 0;

   // remove empty rows and columns
   typename SPxSimplifier<R>::Result ret = removeEmpty(lp);

   if(ret != this->OKAY)
      return ret;

   if(lp.nCols() < 2)
      return this->OKAY;

   DataArray<int>    pClass(lp.nCols());          // class of parallel columns
   DataArray<int>    classSize(lp.nCols());       // size of each class
   Array<R>   scale(lp.nCols());           // scaling factor for each column
   int*              idxMem = 0;

   try
   {
      spx_alloc(idxMem, lp.nCols());
   }
   catch(const SPxMemoryException& x)
   {
      spx_free(idxMem);
      throw x;
   }

   IdxSet idxSet(lp.nCols(), idxMem);  // set of feasible indices for new pClass

   // init
   pClass[0]    = 0;
   scale[0]     = 0.0;
   classSize[0] = lp.nCols();

   for(int j = 1; j < lp.nCols(); ++j)
   {
      pClass[j] = 0;
      scale[j]  = 0.0;
      classSize[j] = 0;
      idxSet.addIdx(j);
   }

   R oldVal = 0.0;

   // main loop
   for(int i = 0; i < lp.nRows(); ++i)
   {
      const SVectorBase<R>& row = lp.rowVector(i);

      for(int k = 0; k < row.size(); ++k)
      {
         R aij = row.value(k);
         int  j   = row.index(k);

         if(scale[j] == 0.0)
            scale[j] = aij;

         m_classSetCols[pClass[j]].add(j, aij / scale[j]);

         if(--classSize[pClass[j]] == 0)
            idxSet.addIdx(pClass[j]);
      }

      // update each parallel class with non-zero row entry
      for(int m = 0; m < row.size(); ++m)
      {
         int k = pClass[row.index(m)];

         if(m_classSetCols[k].size() > 0)
         {
            // sort classSet[k] w.r.t. scaled row values
            ElementCompare compare;

            if(m_classSetCols[k].size() > 1)
               SPxQuicksort(m_classSetCols[k].mem(), m_classSetCols[k].size(), compare);

            // use new index first
            int classIdx = idxSet.index(0);
            idxSet.remove(0);

            for(int l = 0; l < m_classSetCols[k].size(); ++l)
            {
               if(l != 0 && NErel(m_classSetCols[k].value(l), oldVal, this->epsZero()))
               {
                  // start new parallel class
                  classIdx = idxSet.index(0);
                  idxSet.remove(0);
               }

               pClass[m_classSetCols[k].index(l)] = classIdx;
               ++classSize[classIdx];

               oldVal = m_classSetCols[k].value(l);
            }

            m_classSetCols[k].clear();
         }
      }
   }

   spx_free(idxMem);

   DataArray<bool> remCol(lp.nCols());
   DataArray<bool> fixAndRemCol(lp.nCols());

   for(int k = 0; k < lp.nCols(); ++k)
      m_dupCols[k].clear();

   for(int k = 0; k < lp.nCols(); ++k)
   {
      remCol[k] = false;
      fixAndRemCol[k] = false;
      m_dupCols[pClass[k]].add(k, 0.0);
   }

   bool hasDuplicateCol = false;
   DataArray<int>  m_perm_empty(0);

   for(int k = 0; k < lp.nCols(); ++k)
   {
      if(m_dupCols[k].size() > 1 && !(lp.colVector(m_dupCols[k].index(0)).size() == 1))
      {
         MSG_DEBUG((*this->spxout) << "IMAISM58 " << m_dupCols[k].size()
                   << " duplicate columns found" << std::endl;)

         if(!hasDuplicateCol)
         {
            std::shared_ptr<PostStep> ptr(new DuplicateColsPS(lp, 0, 0, 1.0, m_perm_empty, true));
            m_hist.append(ptr);
            hasDuplicateCol = true;
         }

         for(int l = 0; l < m_dupCols[k].size(); ++l)
         {
            for(int m = 0; m < m_dupCols[k].size(); ++m)
            {
               int j1  = m_dupCols[k].index(l);
               int j2  = m_dupCols[k].index(m);

               if(l != m && !remCol[j1] && !remCol[j2])
               {
                  R cj1 = lp.maxObj(j1);
                  R cj2 = lp.maxObj(j2);

                  // A.j1 = factor * A.j2
                  R factor = scale[j1] / scale[j2];
                  R objDif = cj1 - cj2 * scale[j1] / scale[j2];

                  ASSERT_WARN("WMAISM59", isNotZero(factor, this->epsZero()));

                  if(isZero(objDif, this->epsZero()))
                  {
                     // case 1: objectives also duplicate

                     // if 0 is not within the column bounds, we are not able to postsolve if the aggregated column has
                     // status ZERO, hence we skip this case
                     if(LErel(lp.lower(j1), R(0.0)) && GErel(lp.upper(j1), R(0.0))
                           && LErel(lp.lower(j2), R(0.0)) && GErel(lp.upper(j2), R(0.0)))
                     {
                        std::shared_ptr<PostStep> ptr(new DuplicateColsPS(lp, j1, j2, factor, m_perm_empty));
                        // variable substitution xj2' := xj2 + factor * xj1 <=> xj2 = -factor * xj1 + xj2'
                        m_hist.append(ptr);

                        // update bounds of remaining column j2 (new column j2')
                        if(factor > 0)
                        {
                           if(lp.lower(j2) <= R(-infinity) || lp.lower(j1) <= R(-infinity))
                              lp.changeLower(j2, R(-infinity));
                           else
                              lp.changeLower(j2, lp.lower(j2) + factor * lp.lower(j1));

                           if(lp.upper(j2) >= R(infinity) || lp.upper(j1) >= R(infinity))
                              lp.changeUpper(j2, R(infinity));
                           else
                              lp.changeUpper(j2, lp.upper(j2) + factor * lp.upper(j1));
                        }
                        else if(factor < 0)
                        {
                           if(lp.lower(j2) <= R(-infinity) || lp.upper(j1) >= R(infinity))
                              lp.changeLower(j2, R(-infinity));
                           else
                              lp.changeLower(j2, lp.lower(j2) + factor * lp.upper(j1));

                           if(lp.upper(j2) >= R(infinity) || lp.lower(j1) <= R(-infinity))
                              lp.changeUpper(j2, R(infinity));
                           else
                              lp.changeUpper(j2, lp.upper(j2) + factor * lp.lower(j1));
                        }

                        MSG_DEBUG((*this->spxout) << "IMAISM60 two duplicate columns " << j1
                                  << ", " << j2
                                  << " replaced by one" << std::endl;)

                        remCol[j1] = true;

                        ++m_stat[SUB_DUPLICATE_COL];
                     }
                     else
                     {
                        MSG_DEBUG((*this->spxout) << "IMAISM80 not removing two duplicate columns " << j1
                                  << ", " << j2
                                  << " because zero not contained in their bounds" << std::endl;)
                     }
                  }
                  else
                  {
                     // case 2: objectives not duplicate
                     // considered for maximization sense
                     if(lp.lower(j2) <= R(-infinity))
                     {
                        if(factor > 0 && objDif > 0)
                        {
                           if(lp.upper(j1) >= R(infinity))
                           {
                              MSG_DEBUG((*this->spxout) << "IMAISM75 LP unbounded" << std::endl;)
                              return this->UNBOUNDED;
                           }

                           // fix j1 at upper bound
                           MSG_DEBUG((*this->spxout) << "IMAISM61 two duplicate columns " << j1
                                     << ", " << j2
                                     << " first one fixed at upper bound=" << lp.upper(j1) << std::endl;)

                           std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j1, lp.upper(j1)));
                           m_hist.append(ptr);
                           lp.changeLower(j1, lp.upper(j1));
                        }
                        else if(factor < 0 && objDif < 0)
                        {
                           if(lp.lower(j1) <= R(-infinity))
                           {
                              MSG_DEBUG((*this->spxout) << "IMAISM76 LP unbounded" << std::endl;)
                              return this->UNBOUNDED;
                           }

                           // fix j1 at lower bound
                           MSG_DEBUG((*this->spxout) << "IMAISM62 two duplicate columns " << j1
                                     << ", " << j2
                                     << " first one fixed at lower bound=" << lp.lower(j1) << std::endl;)

                           std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j1, lp.lower(j1)));
                           m_hist.append(ptr);
                           lp.changeUpper(j1, lp.lower(j1));
                        }
                     }
                     else if(lp.upper(j2) >= R(infinity))
                     {
                        // fix j1 at upper bound
                        if(factor < 0 && objDif > 0)
                        {
                           if(lp.upper(j1) >= R(infinity))
                           {
                              MSG_DEBUG((*this->spxout) << "IMAISM77 LP unbounded" << std::endl;)
                              return this->UNBOUNDED;
                           }

                           // fix j1 at upper bound
                           MSG_DEBUG((*this->spxout) << "IMAISM63 two duplicate columns " << j1
                                     << ", " << j2
                                     << " first one fixed at upper bound=" << lp.upper(j1) << std::endl;)

                           std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j1, lp.upper(j1)));
                           m_hist.append(ptr);
                           lp.changeLower(j1, lp.upper(j1));
                        }

                        // fix j1 at lower bound
                        else if(factor > 0 && objDif < 0)
                        {
                           if(lp.lower(j1) <= R(-infinity))
                           {
                              MSG_DEBUG((*this->spxout) << "IMAISM78 LP unbounded" << std::endl;)
                              return this->UNBOUNDED;
                           }

                           // fix j1 at lower bound
                           MSG_DEBUG((*this->spxout) << "IMAISM64 two duplicate columns " << j1
                                     << ", " << j2
                                     << " first one fixed at lower bound=" << lp.lower(j1) << std::endl;)

                           std::shared_ptr<PostStep> ptr(new FixBoundsPS(lp, j1, lp.lower(j1)));
                           m_hist.append(ptr);
                           lp.changeUpper(j1, lp.lower(j1));
                        }
                     }

                     if(EQrel(lp.lower(j1), lp.upper(j1), feastol()))
                     {
                        remCol[j1] = true;
                        fixAndRemCol[j1] = true;

                        ++m_stat[FIX_DUPLICATE_COL];
                     }
                  }
               }
            }
         }
      }
   }

   for(int j = 0; j < lp.nCols(); ++j)
   {
      if(fixAndRemCol[j])
      {
         assert(remCol[j]);

         // correctIdx == false, because the index mapping will be handled by the postsolving in DuplicateColsPS
         fixColumn(lp, j, false);
      }
   }

   // remove all columns by one single method call (more efficient)
   const int nColsOld = lp.nCols();
   int* perm = 0;
   spx_alloc(perm, nColsOld);

   for(int j = 0; j < nColsOld; ++j)
   {
      if(remCol[j])
      {
         perm[j] = -1;
         ++remCols;
         remNzos += lp.colVector(j).size();
      }
      else
         perm[j] = 0;
   }

   lp.removeCols(perm);

   for(int j = 0; j < nColsOld; ++j)
   {
      if(perm[j] >= 0)
         m_cIdx[perm[j]] = m_cIdx[j];
   }

   DataArray<int> da_perm(nColsOld);

   for(int j = 0; j < nColsOld; ++j)
   {
      da_perm[j] = perm[j];
   }

   if(hasDuplicateCol)
   {
      std::shared_ptr<PostStep> ptr(new DuplicateColsPS(lp, 0, 0, 1.0, da_perm, false, true));
      m_hist.append(ptr);
   }

   spx_free(perm);

   assert(remCols > 0 || remNzos == 0);

   if(remCols > 0)
   {
      this->m_remCols += remCols;
      this->m_remNzos += remNzos;

      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier (duplicate columns) removed "
                << remCols << " cols, "
                << remNzos << " non-zeros"
                << std::endl;)

      if(remCols > this->m_minReduction * nColsOld)
         again = true;
   }

   return this->OKAY;
}

template <class R>
void SPxMainSM<R>::fixColumn(SPxLPBase<R>& lp, int j, bool correctIdx)
{

   assert(EQrel(lp.lower(j), lp.upper(j), feastol()));

   R lo            = lp.lower(j);
   R up            = lp.upper(j);
   const SVectorBase<R>& col = lp.colVector(j);
   R mid           = lo;

   // use the center value between slightly different bounds to improve numerics
   if(NE(lo, up))
      mid = (up + lo) / 2.0;

   assert(LT(lo, R(infinity)) && GT(lo, R(-infinity)));
   assert(LT(up, R(infinity)) && GT(up, R(-infinity)));

   MSG_DEBUG((*this->spxout) << "IMAISM66 fix variable x" << j
             << ": lower=" << lo
             << " upper=" << up
             << "to new value: " << mid
             << std::endl;)

   if(isNotZero(lo, this->epsZero()))
   {
      for(int k = 0; k < col.size(); ++k)
      {
         int i = col.index(k);

         if(lp.rhs(i) < R(infinity))
         {
            R y     = mid * col.value(k);
            R scale = maxAbs(lp.rhs(i), y);

            if(scale < 1.0)
               scale = 1.0;

            R rhs = (lp.rhs(i) / scale) - (y / scale);

            if(isZero(rhs, this->epsZero()))
               rhs = 0.0;
            else
               rhs *= scale;

            MSG_DEBUG((*this->spxout) << "IMAISM67 row " << i
                      << ": rhs=" << rhs
                      << " (" << lp.rhs(i)
                      << ") aij=" << col.value(k)
                      << std::endl;)

            lp.changeRhs(i, rhs);
         }

         if(lp.lhs(i) > R(-infinity))
         {
            R y     = mid * col.value(k);
            R scale = maxAbs(lp.lhs(i), y);

            if(scale < 1.0)
               scale = 1.0;

            R lhs = (lp.lhs(i) / scale) - (y / scale);

            if(isZero(lhs, this->epsZero()))
               lhs = 0.0;
            else
               lhs *= scale;

            MSG_DEBUG((*this->spxout) << "IMAISM68 row " << i
                      << ": lhs=" << lhs
                      << " (" << lp.lhs(i)
                      << ") aij=" << col.value(k)
                      << std::endl;)

            lp.changeLhs(i, lhs);
         }

         assert(lp.lhs(i) <= lp.rhs(i) + feastol());
      }
   }

   std::shared_ptr<PostStep> ptr(new FixVariablePS(lp, *this, j, lp.lower(j), correctIdx));
   m_hist.append(ptr);
}

template <class R>
typename SPxSimplifier<R>::Result SPxMainSM<R>::simplify(SPxLPBase<R>& lp, R eps, R ftol, R otol,
      Real remainingTime,
      bool keepbounds, uint32_t seed)
{
   // transfer message handler
   this->spxout = lp.spxout;
   assert(this->spxout != 0);

   m_thesense = lp.spxSense();
   this->m_timeUsed->reset();
   this->m_timeUsed->start();

   this->m_objoffset = 0.0;
   m_cutoffbound = R(-infinity);
   m_pseudoobj = R(-infinity);

   this->m_remRows = 0;
   this->m_remCols = 0;
   this->m_remNzos = 0;
   this->m_chgBnds = 0;
   this->m_chgLRhs = 0;
   this->m_keptBnds = 0;
   this->m_keptLRhs = 0;

   m_result     = this->OKAY;
   bool   again = true;
   int nrounds = 0;

   if(m_hist.size() > 0)
   {
      m_hist.clear();
   }

   m_hist.reSize(0);
   m_postsolved = false;

   if(eps < 0.0)
      throw SPxInterfaceException("XMAISM30 Cannot use negative this->epsilon in simplify().");

   if(ftol < 0.0)
      throw SPxInterfaceException("XMAISM31 Cannot use negative feastol in simplify().");

   if(otol < 0.0)
      throw SPxInterfaceException("XMAISM32 Cannot use negative opttol in simplify().");

   m_epsilon = eps;
   m_feastol = ftol;
   m_opttol = otol;


   MSG_INFO2((*this->spxout),
             int numRangedRows = 0;
             int numBoxedCols = 0;
             int numEqualities = 0;

             for(int i = 0; i < lp.nRows(); ++i)
{
   if(lp.lhs(i) > R(-infinity) && lp.rhs(i) < R(infinity))
      {
         if(EQ(lp.lhs(i), lp.rhs(i)))
            ++numEqualities;
         else
            ++numRangedRows;
      }
   }
   for(int j = 0; j < lp.nCols(); ++j)
   if(lp.lower(j) > R(-infinity) && lp.upper(j) < R(infinity))
      ++numBoxedCols;

      (*this->spxout) << "LP has "
                      << numEqualities << " equations, "
                      << numRangedRows << " ranged rows, "
                      << numBoxedCols << " boxed columns"
                      << std::endl;
               )

         m_stat.reSize(17);

   for(int k = 0; k < m_stat.size(); ++k)
      m_stat[k] = 0;

   m_addedcols = 0;
   handleRowObjectives(lp);

   m_prim.reDim(lp.nCols());
   m_slack.reDim(lp.nRows());
   m_dual.reDim(lp.nRows());
   m_redCost.reDim(lp.nCols());
   m_cBasisStat.reSize(lp.nCols());
   m_rBasisStat.reSize(lp.nRows());
   m_cIdx.reSize(lp.nCols());
   m_rIdx.reSize(lp.nRows());

   m_classSetRows.reSize(lp.nRows());
   m_classSetCols.reSize(lp.nCols());
   m_dupRows.reSize(lp.nRows());
   m_dupCols.reSize(lp.nCols());

   m_keepbounds = keepbounds;

   for(int i = 0; i < lp.nRows(); ++i)
      m_rIdx[i] = i;

   for(int j = 0; j < lp.nCols(); ++j)
      m_cIdx[j] = j;

   // round extreme values (set all values smaller than this->eps to zero and all values bigger than R(infinity)/5 to R(infinity))
#if EXTREMES
   handleExtremes(lp);
#endif

   // main presolving loop
   while(again && m_result == this->OKAY)
   {
      nrounds++;
      MSG_INFO3((*this->spxout), (*this->spxout) << "Round " << nrounds << ":" << std::endl;)
      again = false;

#if ROWS_SPXMAINSM

      if(m_result == this->OKAY)
         m_result = simplifyRows(lp, again);

#endif

#if COLS_SPXMAINSM

      if(m_result == this->OKAY)
         m_result = simplifyCols(lp, again);

#endif

#if DUAL_SPXMAINSM

      if(m_result == this->OKAY)
         m_result = simplifyDual(lp, again);

#endif

#if DUPLICATE_ROWS

      if(m_result == this->OKAY)
         m_result = duplicateRows(lp, again);

#endif

#if DUPLICATE_COLS

      if(m_result == this->OKAY)
         m_result = duplicateCols(lp, again);

#endif

      if(!again)
      {
#if TRIVIAL_HEURISTICS
         trivialHeuristic(lp);
#endif

#if PSEUDOOBJ
         propagatePseudoobj(lp);
#endif

#if MULTI_AGGREGATE

         if(m_result == this->OKAY)
            m_result = multiaggregation(lp, again);

#endif
      }

   }

   // preprocessing detected infeasibility or unboundedness
   if(m_result != this->OKAY)
   {
      MSG_INFO1((*this->spxout), (*this->spxout) << "Simplifier result: " << static_cast<int>
                (m_result) << std::endl;)
      return m_result;
   }

   this->m_remCols -= m_addedcols;
   this->m_remNzos -= m_addedcols;
   MSG_INFO1((*this->spxout), (*this->spxout) << "Simplifier removed "
             << this->m_remRows << " rows, "
             << this->m_remCols << " columns, "
             << this->m_remNzos << " nonzeros, "
             << this->m_chgBnds << " col bounds, "
             << this->m_chgLRhs << " row bounds"
             << std::endl;)

   if(keepbounds)
      MSG_INFO2((*this->spxout), (*this->spxout) << "Simplifier kept "
                << this->m_keptBnds << " column bounds, "
                << this->m_keptLRhs << " row bounds"
                << std::endl;)

      MSG_INFO1((*this->spxout), (*this->spxout) << "Reduced LP has "
                << lp.nRows() << " rows "
                << lp.nCols() << " columns "
                << lp.nNzos() << " nonzeros"
                << std::endl;)

      MSG_INFO2((*this->spxout),
                int numRangedRows = 0;
                int numBoxedCols  = 0;
                int numEqualities = 0;

                for(int i = 0; i < lp.nRows(); ++i)
   {
      if(lp.lhs(i) > R(-infinity) && lp.rhs(i) < R(infinity))
         {
            if(EQ(lp.lhs(i), lp.rhs(i)))
               ++numEqualities;
            else
               ++numRangedRows;
         }
      }
   for(int j = 0; j < lp.nCols(); ++j)
   if(lp.lower(j) > R(-infinity) && lp.upper(j) < R(infinity))
      ++numBoxedCols;

      (*this->spxout) << "Reduced LP has "
                      << numEqualities << " equations, "
                      << numRangedRows << " ranged rows, "
                      << numBoxedCols << " boxed columns"
                      << std::endl;
               )

         if(lp.nCols() == 0 && lp.nRows() == 0)
         {
            MSG_INFO1((*this->spxout), (*this->spxout) << "Simplifier removed all rows and columns" <<
                      std::endl;)
            m_result = this->VANISHED;
         }

   MSG_INFO2((*this->spxout), (*this->spxout) << "\nSimplifier performed:\n"
             << m_stat[EMPTY_ROW]            << " empty rows\n"
             << m_stat[FREE_ROW]             << " free rows\n"
             << m_stat[SINGLETON_ROW]        << " singleton rows\n"
             << m_stat[FORCE_ROW]            << " forcing rows\n"
             << m_stat[EMPTY_COL]            << " empty columns\n"
             << m_stat[FIX_COL]              << " fixed columns\n"
             << m_stat[FREE_ZOBJ_COL]        << " free columns with zero objective\n"
             << m_stat[ZOBJ_SINGLETON_COL]   << " singleton columns with zero objective\n"
             << m_stat[DOUBLETON_ROW]        << " singleton columns combined with a doubleton equation\n"
             << m_stat[FREE_SINGLETON_COL]   << " free singleton columns\n"
             << m_stat[DOMINATED_COL]        << " dominated columns\n"
             << m_stat[WEAKLY_DOMINATED_COL] << " weakly dominated columns\n"
             << m_stat[DUPLICATE_ROW]        << " duplicate rows\n"
             << m_stat[FIX_DUPLICATE_COL]    << " duplicate columns (fixed)\n"
             << m_stat[SUB_DUPLICATE_COL]    << " duplicate columns (substituted)\n"
             << m_stat[AGGREGATION]          << " variable aggregations\n"
             << m_stat[MULTI_AGG]            << " multi aggregations\n"
             << std::endl;);

   this->m_timeUsed->stop();

   return m_result;
}

template <class R>
void SPxMainSM<R>::unsimplify(const VectorBase<R>& x, const VectorBase<R>& y,
                              const VectorBase<R>& s, const VectorBase<R>& r,
                              const typename SPxSolverBase<R>::VarStatus rows[],
                              const typename SPxSolverBase<R>::VarStatus cols[], bool isOptimal)
{
   MSG_INFO1((*this->spxout), (*this->spxout) << " --- unsimplifying solution and basis" << std::endl;)
   assert(x.dim() <= m_prim.dim());
   assert(y.dim() <= m_dual.dim());
   assert(x.dim() == r.dim());
   assert(y.dim() == s.dim());

   // assign values of variables in reduced LP
   // NOTE: for maximization problems, we have to switch signs of dual and reduced cost values,
   // since simplifier assumes minimization problem
   for(int j = 0; j < x.dim(); ++j)
   {
      m_prim[j] = isZero(x[j], this->epsZero()) ? 0.0 : x[j];
      m_redCost[j] = isZero(r[j], this->epsZero()) ? 0.0 : (m_thesense == SPxLPBase<R>::MAXIMIZE ? -r[j] :
                     r[j]);
      m_cBasisStat[j] = cols[j];
   }

   for(int i = 0; i < y.dim(); ++i)
   {
      m_dual[i] = isZero(y[i], this->epsZero()) ? 0.0 : (m_thesense == SPxLPBase<R>::MAXIMIZE ? -y[i] :
                  y[i]);
      m_slack[i] = isZero(s[i], this->epsZero()) ? 0.0 : s[i];
      m_rBasisStat[i] = rows[i];
   }

   // undo preprocessing
   for(int k = m_hist.size() - 1; k >= 0; --k)
   {
      MSG_DEBUG(std::cout << "unsimplifying " << m_hist[k]->getName() << "\n");

      try
      {
         m_hist[k]->execute(m_prim, m_dual, m_slack, m_redCost, m_cBasisStat, m_rBasisStat, isOptimal);
      }
      catch(const SPxException& ex)
      {
         MSG_INFO1((*this->spxout), (*this->spxout) << "Exception thrown while unsimplifying " <<
                   m_hist[k]->getName() << ":\n" << ex.what() << "\n");
         throw SPxInternalCodeException("XMAISM00 Exception thrown during unsimply().");
      }

      m_hist.reSize(k);
   }

   // for maximization problems, we have to switch signs of dual and reduced cost values back
   if(m_thesense == SPxLPBase<R>::MAXIMIZE)
   {
      for(int j = 0; j < m_redCost.dim(); ++j)
         m_redCost[j] = -m_redCost[j];

      for(int i = 0; i < m_dual.dim(); ++i)
         m_dual[i] = -m_dual[i];
   }

   if(m_addedcols > 0)
   {
      assert(m_prim.dim() >= m_addedcols);
      m_prim.reDim(m_prim.dim() - m_addedcols);
      m_redCost.reDim(m_redCost.dim() - m_addedcols);
      m_cBasisStat.reSize(m_cBasisStat.size() - m_addedcols);
      m_cIdx.reSize(m_cIdx.size() - m_addedcols);
   }

#ifdef CHECK_BASIC_DIM
   int numBasis = 0;

   for(int rs = 0; rs < m_rBasisStat.size(); ++rs)
   {
      if(m_rBasisStat[rs] == SPxSolverBase<R>::BASIC)
         numBasis ++;
   }

   for(int cs = 0; cs < m_cBasisStat.size(); ++cs)
   {
      if(m_cBasisStat[cs] == SPxSolverBase<R>::BASIC)
         numBasis ++;
   }

   if(numBasis != m_rBasisStat.size())
   {
      throw SPxInternalCodeException("XMAISM26 Dimension doesn't match after this step.");
   }

#endif

   m_hist.clear();
   m_postsolved = true;
}

// Pretty-printing of solver status.
template <class R>
std::ostream& operator<<(std::ostream& os, const typename SPxSimplifier<R>::Result& status)
{
   switch(status)
   {
   case SPxSimplifier<R>::OKAY:
      os << "SUCCESS";
      break;

   case SPxSimplifier<R>::INFEASIBLE:
      os << "INFEASIBLE";
      break;

   case SPxSimplifier<R>::DUAL_INFEASIBLE:
      os << "DUAL_INFEASIBLE";
      break;

   case SPxSimplifier<R>::UNBOUNDED:
      os << "UNBOUNDED";
      break;

   case SPxSimplifier<R>::VANISHED:
      os << "VANISHED";
      break;

   default:
      os << "UNKNOWN";
      break;
   }

   return os;
}

} //namespace soplex
