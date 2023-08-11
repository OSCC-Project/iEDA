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

/**@file  spxscaler.hpp
 * @brief LP scaling base class.
 */

#include <cmath>

#include <iostream>
#include <assert.h>
#include "soplex/dsvector.h"
#include "soplex/lprowsetbase.h"
#include "soplex/lpcolsetbase.h"
#include <limits>

namespace soplex
{

template <class R>
std::ostream& operator<<(std::ostream& s, const SPxScaler<R>& sc)
{
   const DataArray < int >& colscaleExp = *(sc.m_activeColscaleExp);
   DataArray < int > rowccaleExp = *(sc.m_activeRowscaleExp);

   s << sc.getName() << " scaler:" << std::endl;
   s << "colscale = [ ";

   for(int ci = 0; ci < colscaleExp.size(); ++ci)
      s << colscaleExp[ci] << " ";

   s << "]" << std::endl;

   s << "rowscale = [ ";

   for(int ri = 0; ri < rowccaleExp.size(); ++ri)
      s << rowccaleExp[ri] << " ";

   s << "]" << std::endl;

   return s;
}


template <class R>
SPxScaler<R>::SPxScaler(
   const char* name,
   bool        colFirst,
   bool        doBoth,
   SPxOut*     outstream)
   : m_name(name)
   , m_activeColscaleExp(0)
   , m_activeRowscaleExp(0)
   , m_colFirst(colFirst)
   , m_doBoth(doBoth)
   , spxout(outstream)
{
   assert(SPxScaler<R>::isConsistent());
}

template <class R>
SPxScaler<R>::SPxScaler(const SPxScaler<R>& old)
   : m_name(old.m_name)
   , m_activeColscaleExp(old.m_activeColscaleExp)
   , m_activeRowscaleExp(old.m_activeRowscaleExp)
   , m_colFirst(old.m_colFirst)
   , m_doBoth(old.m_doBoth)
   , spxout(old.spxout)
{
   assert(SPxScaler<R>::isConsistent());
}

template <class R>
SPxScaler<R>::~SPxScaler()
{
   m_name = 0;
}

template <class R>
SPxScaler<R>& SPxScaler<R>::operator=(const SPxScaler<R>& rhs)
{
   if(this != &rhs)
   {
      m_name     = rhs.m_name;
      m_activeColscaleExp = rhs.m_activeColscaleExp;
      m_activeRowscaleExp = rhs.m_activeRowscaleExp;
      m_colFirst = rhs.m_colFirst;
      m_doBoth   = rhs.m_doBoth;
      spxout     = rhs.spxout;

      assert(SPxScaler<R>::isConsistent());
   }

   return *this;
}

template <class R>
const char* SPxScaler<R>::getName() const
{

   return m_name;
}

template <class R>
void SPxScaler<R>::setOrder(bool colFirst)
{

   m_colFirst = colFirst;
}

template <class R>
void SPxScaler<R>::setBoth(bool both)
{

   m_doBoth = both;
}

template <class R>
void SPxScaler<R>::setRealParam(R param, const char* name)
{}

template <class R>
void SPxScaler<R>::setIntParam(int param, const char* name)
{}

template <class R>
void SPxScaler<R>::setup(SPxLPBase<R>& lp)
{
   assert(lp.isConsistent());
   m_activeColscaleExp = &lp.LPColSetBase<R>::scaleExp;
   m_activeRowscaleExp = &lp.LPRowSetBase<R>::scaleExp;
   m_activeColscaleExp->reSize(lp.nCols());
   m_activeRowscaleExp->reSize(lp.nRows());

   for(int i = 0; i < lp.nCols(); ++i)
      (*m_activeColscaleExp)[i] = 0;

   for(int i = 0; i < lp.nRows(); ++i)
      (*m_activeRowscaleExp)[i] = 0;

   lp.lp_scaler = this;
}


template <class R>
int SPxScaler<R>::computeScaleExp(const SVectorBase<R>& vec,
                                  const DataArray<int>& oldScaleExp) const
{
   R maxi = 0.0;

   // find largest absolute value after applying existing scaling factors
   for(int i = 0; i < vec.size(); ++i)
   {
      R x = spxAbs(spxLdexp(vec.value(i), oldScaleExp[vec.index(i)]));

      if(GT(x, maxi))
         maxi = x;
   }

   // empty rows/cols are possible
   if(maxi == 0.0)
      return 0;
   // get exponent corresponding to new scaling factor
   else
   {
      int scaleExp;
      spxFrexp(R(1.0 / maxi), &(scaleExp));
      return scaleExp - 1;
   }
}

template <class R>
void SPxScaler<R>::applyScaling(SPxLPBase<R>& lp)
{
   assert(lp.nCols() == m_activeColscaleExp->size());
   assert(lp.nRows() == m_activeRowscaleExp->size());

   DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   for(int i = 0; i < lp.nRows(); ++i)
   {
      SVectorBase<R>& vec = lp.rowVector_w(i);
      int exp1;
      int exp2 = rowscaleExp[i];

      for(int j = 0; j < vec.size(); ++j)
      {
         exp1 = colscaleExp[vec.index(j)];
         vec.value(j) = spxLdexp(vec.value(j), exp1 + exp2);
      }

      lp.maxRowObj_w(i) = spxLdexp(lp.maxRowObj(i), exp2);

      if(lp.rhs(i) < R(infinity))
         lp.rhs_w(i) = spxLdexp(lp.rhs_w(i), exp2);

      if(lp.lhs(i) > R(-infinity))
         lp.lhs_w(i) = spxLdexp(lp.lhs_w(i), exp2);

      MSG_DEBUG(std::cout << "DEBUG: rowscaleExp(" << i << "): " << exp2 << std::endl;)
   }

   for(int i = 0; i < lp.nCols(); ++i)
   {
      SVectorBase<R>& vec = lp.colVector_w(i);
      int exp1;
      int exp2 = colscaleExp[i];

      for(int j = 0; j < vec.size(); ++j)
      {
         exp1 = rowscaleExp[vec.index(j)];
         vec.value(j) = spxLdexp(vec.value(j), exp1 + exp2);
      }

      lp.maxObj_w(i) = spxLdexp(lp.maxObj_w(i), exp2);

      if(lp.upper(i) < R(infinity))
         lp.upper_w(i) = spxLdexp(lp.upper_w(i), -exp2);

      if(lp.lower(i) > R(-infinity))
         lp.lower_w(i) = spxLdexp(lp.lower_w(i), -exp2);

      MSG_DEBUG(std::cout << "DEBUG: colscaleExp(" << i << "): " << exp2 << std::endl;)
   }

   lp.setScalingInfo(true);
   assert(lp.isConsistent());
}

/// unscale SPxLP
template <class R>
void SPxScaler<R>::unscale(SPxLPBase<R>& lp)
{
   assert(lp.isScaled());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   for(int i = 0; i < lp.nRows(); ++i)
   {
      SVectorBase<R>& vec = lp.rowVector_w(i);

      int exp1;
      int exp2 = rowscaleExp[i];

      for(int j = 0; j < vec.size(); ++j)
      {
         exp1 = colscaleExp[vec.index(j)];
         vec.value(j) = spxLdexp(vec.value(j), -exp1 - exp2);
      }

      lp.maxRowObj_w(i) = spxLdexp(lp.maxRowObj(i), -exp2);

      if(lp.rhs(i) < R(infinity))
         lp.rhs_w(i) = spxLdexp(lp.rhs_w(i), -exp2);

      if(lp.lhs(i) > R(-infinity))
         lp.lhs_w(i) = spxLdexp(lp.lhs_w(i), -exp2);
   }

   for(int i = 0; i < lp.nCols(); ++i)
   {
      SVectorBase<R>& vec = lp.colVector_w(i);

      int exp1;
      int exp2 = colscaleExp[i];

      for(int j = 0; j < vec.size(); ++j)
      {
         exp1 = rowscaleExp[vec.index(j)];
         vec.value(j) = spxLdexp(vec.value(j), -exp1 - exp2);
      }

      lp.maxObj_w(i) = spxLdexp(lp.maxObj_w(i), -exp2);

      if(lp.upper(i) < R(infinity))
         lp.upper_w(i) = spxLdexp(lp.upper_w(i), exp2);

      if(lp.lower(i) > R(-infinity))
         lp.lower_w(i) = spxLdexp(lp.lower_w(i), exp2);
   }

   lp._isScaled = false;
   assert(lp.isConsistent());
}

/// returns scaling factor for column \p i
/// todo pass the LP?!
template <class R>
int SPxScaler<R>::getColScaleExp(int i) const
{
   return (*m_activeColscaleExp)[i];
}

/// returns scaling factor for row \p i
/// todo pass the LP?!
template <class R>
int SPxScaler<R>::getRowScaleExp(int i) const
{
   return (*m_activeRowscaleExp)[i];
}

/// gets unscaled column \p i
template <class R>
void SPxScaler<R>::getColUnscaled(const SPxLPBase<R>& lp, int i, DSVectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(i < lp.nCols());
   assert(i >= 0);
   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   vec = lp.LPColSetBase<R>::colVector(i);

   int exp1;
   int exp2 = colscaleExp[i];

   const SVectorBase<R>& col = lp.colVector(i);
   vec.setMax(col.size());
   vec.clear();

   for(int j = 0; j < col.size(); j++)
   {
      exp1 = rowscaleExp[col.index(j)];
      vec.add(col.index(j), spxLdexp(col.value(j), -exp1 - exp2));
   }
}

/// returns maximum absolute value of unscaled column \p i
template <class R>
R SPxScaler<R>::getColMaxAbsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(i < lp.nCols());
   assert(i >= 0);

   DataArray < int >& colscaleExp = *m_activeColscaleExp;
   DataArray < int >& rowscaleExp = *m_activeRowscaleExp;
   const SVectorBase<R>& colVec = lp.LPColSetBase<R>::colVector(i);

   R max = 0.0;
   int exp1;
   int exp2 = colscaleExp[i];

   for(int j = 0; j < colVec.size(); j++)
   {
      exp1 = rowscaleExp[colVec.index(j)];
      R abs = spxAbs(spxLdexp(colVec.value(j), -exp1 - exp2));

      if(abs > max)
         max = abs;
   }

   return max;
}

/// returns minimum absolute value of unscaled column \p i
template <class R>
R SPxScaler<R>::getColMinAbsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(i < lp.nCols());
   assert(i >= 0);

   DataArray < int >& colscaleExp = *m_activeColscaleExp;
   DataArray < int >& rowscaleExp = *m_activeRowscaleExp;
   const SVectorBase<R>& colVec = lp.LPColSetBase<R>::colVector(i);

   R min = R(infinity);
   int exp1;
   int exp2 = colscaleExp[i];

   for(int j = 0; j < colVec.size(); j++)
   {
      exp1 = rowscaleExp[colVec.index(j)];
      R abs = spxAbs(spxLdexp(colVec.value(j), -exp1 - exp2));

      if(abs < min)
         min = abs;
   }

   return min;
}


/// returns unscaled upper bound \p i
template <class R>
R SPxScaler<R>::upperUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(lp.isScaled());
   assert(i < lp.nCols());
   assert(i >= 0);

   if(lp.LPColSetBase<R>::upper(i) < R(infinity))
   {
      const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
      return spxLdexp(lp.LPColSetBase<R>::upper(i), colscaleExp[i]);
   }
   else
      return lp.LPColSetBase<R>::upper(i);
}


/// gets unscaled upper bound vector
template <class R>
void SPxScaler<R>::getUpperUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(lp.LPColSetBase<R>::upper().dim() == vec.dim());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   for(int i = 0; i < lp.LPColSetBase<R>::upper().dim(); i++)
      vec[i] = spxLdexp(lp.LPColSetBase<R>::upper()[i], colscaleExp[i]);
}


/// returns unscaled upper bound VectorBase<R> of \p lp
template <class R>
R SPxScaler<R>::lowerUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(lp.isScaled());
   assert(i < lp.nCols());
   assert(i >= 0);

   if(lp.LPColSetBase<R>::lower(i) > R(-infinity))
   {
      const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
      return spxLdexp(lp.LPColSetBase<R>::lower(i), colscaleExp[i]);
   }
   else
      return lp.LPColSetBase<R>::lower(i);
}


/// returns unscaled lower bound VectorBase<R> of \p lp
template <class R>
void SPxScaler<R>::getLowerUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(lp.LPColSetBase<R>::lower().dim() == vec.dim());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   for(int i = 0; i < lp.LPColSetBase<R>::lower().dim(); i++)
      vec[i] = spxLdexp(lp.LPColSetBase<R>::lower()[i], colscaleExp[i]);
}

/// returns unscaled objective function coefficient of \p i
template <class R>
R SPxScaler<R>::maxObjUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(lp.isScaled());
   assert(i < lp.nCols());
   assert(i >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   return spxLdexp(lp.LPColSetBase<R>::maxObj(i), -colscaleExp[i]);
}


/// gets unscaled objective function coefficient of \p i
template <class R>
void SPxScaler<R>::getMaxObjUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(lp.LPColSetBase<R>::maxObj().dim() == vec.dim());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   for(int i = 0; i < lp.LPColSetBase<R>::maxObj().dim(); i++)
      vec[i] = spxLdexp(lp.LPColSetBase<R>::maxObj()[i], -colscaleExp[i]);
}

/// gets unscaled row \p i
template <class R>
void SPxScaler<R>::getRowUnscaled(const SPxLPBase<R>& lp, int i, DSVectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(i < lp.nRows());
   assert(i >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;
   int exp1;
   int exp2 = rowscaleExp[i];

   const SVectorBase<R>& row = lp.rowVector(i);
   vec.setMax(row.size());
   vec.clear();

   for(int j = 0; j < row.size(); j++)
   {
      exp1 = colscaleExp[row.index(j)];
      vec.add(row.index(j), spxLdexp(row.value(j), -exp1 - exp2));
   }
}

/// returns maximum absolute value of unscaled row \p i
template <class R>
R SPxScaler<R>::getRowMaxAbsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(i < lp.nRows());
   assert(i >= 0);
   DataArray < int >& colscaleExp = *m_activeColscaleExp;
   DataArray < int >& rowscaleExp = *m_activeRowscaleExp;
   const SVectorBase<R>& rowVec = lp.LPRowSetBase<R>::rowVector(i);

   R max = 0.0;

   int exp1;
   int exp2 = rowscaleExp[i];

   for(int j = 0; j < rowVec.size(); j++)
   {
      exp1 = colscaleExp[rowVec.index(j)];
      R abs = spxAbs(spxLdexp(rowVec.value(j), -exp1 - exp2));

      if(GT(abs, max))
         max = abs;
   }

   return max;
}

/// returns minimum absolute value of unscaled row \p i
template <class R>
R SPxScaler<R>::getRowMinAbsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(i < lp.nRows());
   assert(i >= 0);
   DataArray < int >& colscaleExp = *m_activeColscaleExp;
   DataArray < int >& rowscaleExp = *m_activeRowscaleExp;
   const SVectorBase<R>& rowVec = lp.LPRowSetBase<R>::rowVector(i);

   R min = R(infinity);

   int exp1;
   int exp2 = rowscaleExp[i];

   for(int j = 0; j < rowVec.size(); j++)
   {
      exp1 = colscaleExp[rowVec.index(j)];
      R abs = spxAbs(spxLdexp(rowVec.value(j), -exp1 - exp2));

      if(LT(abs, min))
         min = abs;
   }

   return min;
}

/// returns unscaled right hand side \p i
template <class R>
R SPxScaler<R>::rhsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(lp.isScaled());
   assert(i < lp.nRows());
   assert(i >= 0);

   if(lp.LPRowSetBase<R>::rhs(i) < R(infinity))
   {
      const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;
      return spxLdexp(lp.LPRowSetBase<R>::rhs(i), -rowscaleExp[i]);
   }
   else
      return lp.LPRowSetBase<R>::rhs(i);
}


/// gets unscaled right hand side vector
template <class R>
void SPxScaler<R>::getRhsUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(lp.LPRowSetBase<R>::rhs().dim() == vec.dim());

   for(int i = 0; i < lp.LPRowSetBase<R>::rhs().dim(); i++)
   {
      const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;
      vec[i] = spxLdexp(lp.LPRowSetBase<R>::rhs()[i], -rowscaleExp[i]);
   }
}


/// returns unscaled left hand side \p i of \p lp
template <class R>
R SPxScaler<R>::lhsUnscaled(const SPxLPBase<R>& lp, int i) const
{
   assert(lp.isScaled());
   assert(i < lp.nRows());
   assert(i >= 0);

   if(lp.LPRowSetBase<R>::lhs(i) > R(-infinity))
   {
      const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;
      return spxLdexp(lp.LPRowSetBase<R>::lhs(i), -rowscaleExp[i]);
   }
   else
      return lp.LPRowSetBase<R>::lhs(i);
}

/// returns unscaled left hand side VectorBase<R> of \p lp
template <class R>
void SPxScaler<R>::getLhsUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const
{
   assert(lp.isScaled());
   assert(lp.LPRowSetBase<R>::lhs().dim() == vec.dim());

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   for(int i = 0; i < lp.LPRowSetBase<R>::lhs().dim(); i++)
      vec[i] = spxLdexp(lp.LPRowSetBase<R>::lhs()[i], -rowscaleExp[i]);
}

/// returns unscaled coefficient of \p lp
template <class R>
R SPxScaler<R>::getCoefUnscaled(const SPxLPBase<R>& lp, int row, int col) const
{
   assert(lp.isScaled());
   assert(row < lp.nRows());
   assert(col < lp.nCols());

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;
   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   return spxLdexp(lp.colVector(col)[row], - rowscaleExp[row] - colscaleExp[col]);
}

template <class R>
void SPxScaler<R>::unscalePrimal(const SPxLPBase<R>& lp, VectorBase<R>& x) const
{
   assert(lp.isScaled());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   assert(x.dim() == colscaleExp.size());

   for(int j = 0; j < x.dim(); ++j)
      x[j] = spxLdexp(x[j], colscaleExp[j]);
}

template <class R>
void SPxScaler<R>::unscaleSlacks(const SPxLPBase<R>& lp, VectorBase<R>& s) const
{
   assert(lp.isScaled());

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   assert(s.dim() == rowscaleExp.size());

   for(int i = 0; i < s.dim(); ++i)
      s[i] = spxLdexp(s[i], -rowscaleExp[i]);
}

template <class R>
void SPxScaler<R>::unscaleDual(const SPxLPBase<R>& lp, VectorBase<R>& pi) const
{
   assert(lp.isScaled());

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   assert(pi.dim() == rowscaleExp.size());

   for(int i = 0; i < pi.dim(); ++i)
      pi[i] = spxLdexp(pi[i], rowscaleExp[i]);
}

template <class R>
void SPxScaler<R>::unscaleRedCost(const SPxLPBase<R>& lp, VectorBase<R>& r) const
{
   assert(lp.isScaled());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   assert(r.dim() == colscaleExp.size());

   for(int j = 0; j < r.dim(); ++j)
      r[j] = spxLdexp(r[j], -colscaleExp[j]);
}

template <class R>
void SPxScaler<R>::unscalePrimalray(const SPxLPBase<R>& lp, VectorBase<R>& ray) const
{
   assert(lp.isScaled());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   assert(ray.dim() == colscaleExp.size());

   for(int j = 0; j < ray.dim(); ++j)
      ray[j] = spxLdexp(ray[j], colscaleExp[j]);
}

template <class R>
void SPxScaler<R>::unscaleDualray(const SPxLPBase<R>& lp, VectorBase<R>& ray) const
{
   assert(lp.isScaled());

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   assert(ray.dim() == rowscaleExp.size());

   for(int i = 0; i < ray.dim(); ++i)
      ray[i] = spxLdexp(ray[i], rowscaleExp[i]);
}

template <class R>
void SPxScaler<R>::scaleObj(const SPxLPBase<R>& lp, VectorBase<R>& origObj) const
{
   assert(lp.isScaled());

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   for(int i = 0; i < origObj.dim(); ++i)
   {
      origObj[i] = spxLdexp(origObj[i], colscaleExp[i]);
   }
}

template <class R>
R SPxScaler<R>::scaleObj(const SPxLPBase<R>& lp, int i, R origObj) const
{
   assert(lp.isScaled());
   assert(i < lp.nCols());
   assert(i >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   int exp = colscaleExp[i];

   return spxLdexp(origObj, exp);
}

template <class R>
R SPxScaler<R>::scaleElement(const SPxLPBase<R>& lp, int row, int col, R val) const
{
   assert(lp.isScaled());
   assert(col < lp.nCols());
   assert(col >= 0);
   assert(row < lp.nRows());
   assert(row >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;
   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   return spxLdexp(val, colscaleExp[col] + rowscaleExp[row]);
}

template <class R>
R SPxScaler<R>::scaleLower(const SPxLPBase<R>& lp, int col, R lower) const
{
   assert(lp.isScaled());
   assert(col < lp.nCols());
   assert(col >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   return spxLdexp(lower, -colscaleExp[col]);
}

template <class R>
R SPxScaler<R>::scaleUpper(const SPxLPBase<R>& lp, int col, R upper) const
{
   assert(lp.isScaled());
   assert(col < lp.nCols());
   assert(col >= 0);

   const DataArray < int >& colscaleExp = lp.LPColSetBase<R>::scaleExp;

   return spxLdexp(upper, -colscaleExp[col]);
}

template <class R>
R SPxScaler<R>::scaleLhs(const SPxLPBase<R>& lp, int row, R lhs) const
{
   assert(lp.isScaled());
   assert(row < lp.nRows());
   assert(row >= 0);

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   return spxLdexp(lhs, rowscaleExp[row]);
}

template <class R>
R SPxScaler<R>::scaleRhs(const SPxLPBase<R>& lp, int row, R rhs) const
{
   assert(lp.isScaled());
   assert(row < lp.nRows());
   assert(row >= 0);

   const DataArray < int >& rowscaleExp = lp.LPRowSetBase<R>::scaleExp;

   return spxLdexp(rhs, rowscaleExp[row]);
}

template <class R>
R SPxScaler<R>::minAbsColscale() const
{
   const DataArray < int >& colscaleExp = *m_activeColscaleExp;

   R mini = R(infinity);

   for(int i = 0; i < colscaleExp.size(); ++i)
      if(spxAbs(spxLdexp(1.0, colscaleExp[i])) < mini)
         mini = spxAbs(spxLdexp(1.0, colscaleExp[i]));

   return mini;
}

template <class R>
R SPxScaler<R>::maxAbsColscale() const
{
   const DataArray < int >& colscaleExp = *m_activeColscaleExp;

   R maxi = 0.0;

   for(int i = 0; i < colscaleExp.size(); ++i)
      if(spxAbs(spxLdexp(1.0, colscaleExp[i])) > maxi)
         maxi = spxAbs(spxLdexp(1.0, colscaleExp[i]));


   return maxi;
}

template <class R>
R SPxScaler<R>::minAbsRowscale() const
{
   const DataArray < int >& rowscaleExp = *m_activeRowscaleExp;

   int mini = std::numeric_limits<int>::max();

   for(int i = 0; i < rowscaleExp.size(); ++i)
      if(rowscaleExp[i] < mini)
         mini = rowscaleExp[i];

   return spxLdexp(1.0, mini);
}

template <class R>
R SPxScaler<R>::maxAbsRowscale() const
{
   const DataArray < int >& rowscaleExp = *m_activeRowscaleExp;

   int maxi = std::numeric_limits<int>::min();

   for(int i = 0; i < rowscaleExp.size(); ++i)
      if(rowscaleExp[i] > maxi)
         maxi = rowscaleExp[i];

   return spxLdexp(1.0, maxi);
}

/** \f$\max_{j\in\mbox{ cols}}
 *   \left(\frac{\max_{i\in\mbox{ rows}}|a_ij|}
 *              {\min_{i\in\mbox{ rows}}|a_ij|}\right)\f$
 */
template <class R>
R SPxScaler<R>::maxColRatio(const SPxLPBase<R>& lp) const
{

   R pmax = 0.0;

   for(int i = 0; i < lp.nCols(); ++i)
   {
      const SVectorBase<R>& vec  = lp.colVector(i);
      R           mini = R(infinity);
      R           maxi = 0.0;

      for(int j = 0; j < vec.size(); ++j)
      {
         R x = spxAbs(vec.value(j));

         if(isZero(x))
            continue;

         if(x < mini)
            mini = x;

         if(x > maxi)
            maxi = x;
      }

      if(mini == R(infinity))
         continue;

      R p = maxi / mini;

      if(p > pmax)
         pmax = p;
   }

   return pmax;
}

/** \f$\max_{i\in\mbox{ rows}}
 *   \left(\frac{\max_{j\in\mbox{ cols}}|a_ij|}
 *              {\min_{j\in\mbox{ cols}}|a_ij|}\right)\f$
 */
template <class R>
R SPxScaler<R>::maxRowRatio(const SPxLPBase<R>& lp) const
{

   R pmax = 0.0;

   for(int i = 0; i < lp.nRows(); ++i)
   {
      const SVectorBase<R>& vec  = lp.rowVector(i);
      R           mini = R(infinity);
      R           maxi = 0.0;

      for(int j = 0; j < vec.size(); ++j)
      {
         R x = spxAbs(vec.value(j));

         if(isZero(x))
            continue;

         if(x < mini)
            mini = x;

         if(x > maxi)
            maxi = x;
      }

      if(mini == R(infinity))
         continue;

      R p = maxi / mini;

      if(p > pmax)
         pmax = p;
   }

   return pmax;
}

template <class R>
void SPxScaler<R>::computeExpVec(const std::vector<R>& vec, DataArray<int>& vecExp)
{
   assert(vec.size() == unsigned(vecExp.size()));

   for(unsigned i = 0; i < vec.size(); ++i)
   {
      spxFrexp(vec[i], &(vecExp[int(i)]));
      vecExp[int(i)] -= 1;
   }
}

template <class R>
bool SPxScaler<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   return m_activeColscaleExp->isConsistent() && m_activeRowscaleExp->isConsistent();
#else
   return true;
#endif
}


} // namespace soplex
