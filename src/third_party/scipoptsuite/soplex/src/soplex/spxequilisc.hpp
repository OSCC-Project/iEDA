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

/**@file  spxequilisc.hpp
 * @brief Equilibrium row/column scaling.
 */
#include <assert.h>

#include "soplex/spxout.h"
#include "soplex/spxlpbase.h"
#include "soplex/spxlp.h"
#include "soplex.h"

namespace soplex
{
static inline const char* makename(bool doBoth)
{
   return doBoth ? "bi-Equilibrium" : "uni-Equilibrium";
}

/// maximum ratio between absolute biggest and smallest element in any scaled row/column.
template <class R>
static R maxPrescaledRatio(const SPxLPBase<R>& lp, const std::vector<R>& coScaleval, bool rowRatio)
{
   R pmax = 0.0;
   const int n = rowRatio ? lp.nRows() : lp.nCols();

   for(int i = 0; i < n; ++i)
   {
      const SVectorBase<R>& vec = rowRatio ? lp.rowVector(i) : lp.colVector(i);
      R mini = R(infinity);
      R maxi = 0.0;

      for(int j = 0; j < vec.size(); ++j)
      {
         assert(vec.index(j) >= 0);
         const R x = spxAbs(vec.value(j)) * coScaleval[unsigned(vec.index(j))];

         if(isZero(x))
            continue;

         if(x < mini)
            mini = x;

         if(x > maxi)
            maxi = x;
      }

      if(mini == R(infinity))
         continue;

      const R p = maxi / mini;

      if(p > pmax)
         pmax = p;
   }

   return pmax;
}

template <class R>
void SPxEquiliSC<R>::computeEquiExpVec(const SVSetBase<R>* vecset, const DataArray<int>& coScaleExp,
                                       DataArray<int>& scaleExp)
{
   assert(vecset != nullptr);

   for(int i = 0; i < vecset->num(); ++i)
   {
      const SVectorBase<R>& vec = (*vecset)[i];

      R maxi = 0.0;

      for(int j = 0; j < vec.size(); ++j)
      {
         const R x = spxAbs(spxLdexp(vec.value(j), coScaleExp[vec.index(j)]));

         if(GT(x, maxi))
            maxi = x;
      }

      // empty rows/cols are possible
      if(maxi == 0.0)
         maxi = 1.0;

      assert(maxi > 0.0);

      spxFrexp(Real(1.0 / maxi), &(scaleExp[i]));

      scaleExp[i] -= 1;
   }
}

template <class R>
void SPxEquiliSC<R>::computeEquiExpVec(const SVSetBase<R>* vecset, const std::vector<R>& coScaleVal,
                                       DataArray<int>& scaleExp)
{
   assert(vecset != nullptr);

   for(int i = 0; i < vecset->num(); ++i)
   {
      const SVectorBase<R>& vec = (*vecset)[i];

      R maxi = 0.0;

      for(int j = 0; j < vec.size(); ++j)
      {
         assert(vec.index(j) >= 0);
         const R x = spxAbs(vec.value(j) * coScaleVal[unsigned(vec.index(j))]);

         if(GT(x, maxi))
            maxi = x;
      }

      // empty rows/cols are possible
      if(maxi == 0.0)
         maxi = 1.0;

      assert(maxi > 0.0);

      spxFrexp(Real(1.0 / maxi), &(scaleExp[i]));

      scaleExp[i] -= 1;
   }
}

template <class R>
void SPxEquiliSC<R>::computePostequiExpVecs(const SPxLPBase<R>& lp,
      const std::vector<R>& preRowscale, const std::vector<R>& preColscale,
      DataArray<int>& rowscaleExp, DataArray<int>& colscaleExp)
{
   const R colratio = maxPrescaledRatio(lp, preRowscale, false);
   const R rowratio = maxPrescaledRatio(lp, preColscale, true);

   const bool colFirst = colratio < rowratio;

   // see SPxEquiliSC<R>::scale for reason behind this branch
   if(colFirst)
   {
      computeEquiExpVec(lp.colSet(), preRowscale, colscaleExp);
      computeEquiExpVec(lp.rowSet(), colscaleExp, rowscaleExp);
   }
   else
   {
      computeEquiExpVec(lp.rowSet(), preColscale, rowscaleExp);
      computeEquiExpVec(lp.colSet(), rowscaleExp, colscaleExp);
   }
}

template <class R>
SPxEquiliSC<R>::SPxEquiliSC(bool doBoth)
   : SPxScaler<R>(makename(doBoth), false, doBoth)
{}

template <class R>
SPxEquiliSC<R>::SPxEquiliSC(const SPxEquiliSC<R>& old)
   : SPxScaler<R>(old)
{}

template <class R>
SPxEquiliSC<R>& SPxEquiliSC<R>::operator=(const SPxEquiliSC<R>& rhs)
{
   if(this != &rhs)
   {
      SPxScaler<R>::operator=(rhs);
   }

   return *this;
}

template <class R>
void SPxEquiliSC<R>::scale(SPxLPBase<R>& lp, bool persistent)
{

   MSG_INFO1((*this->spxout), (*this->spxout) << "Equilibrium scaling LP" <<
             (persistent ? " (persistent)" : "") << std::endl;)

   this->setup(lp);

   /* We want to do the direction first, which has a lower maximal ratio,
    * since the lowest value in the scaled matrix is bounded from below by
    * the inverse of the maximum ratio of the direction that is done first
    * Example:
    *
    *                     Rowratio
    *            0.1  1   10
    *            10   1   10
    *
    * Colratio   100  1
    *
    * Row first =>         Col next =>
    *            0.1  1          0.1  1
    *            1    0.1        1    0.1
    *
    * Col first =>         Row next =>
    *            0.01 1          0.01 1
    *            1    1          1    1
    *
    */
   R colratio = this->maxColRatio(lp);
   R rowratio = this->maxRowRatio(lp);

   bool colFirst = colratio < rowratio;

   MSG_INFO2((*this->spxout), (*this->spxout) << "before scaling:"
             << " min= " << lp.minAbsNzo()
             << " max= " << lp.maxAbsNzo()
             << " col-ratio= " << colratio
             << " row-ratio= " << rowratio
             << std::endl;)

   if(colFirst)
   {
      computeEquiExpVec(lp.colSet(), *this->m_activeRowscaleExp, *this->m_activeColscaleExp);

      if(this->m_doBoth)
         computeEquiExpVec(lp.rowSet(), *this->m_activeColscaleExp, *this->m_activeRowscaleExp);
   }
   else
   {
      computeEquiExpVec(lp.rowSet(), *this->m_activeColscaleExp, *this->m_activeRowscaleExp);

      if(this->m_doBoth)
         computeEquiExpVec(lp.colSet(), *this->m_activeRowscaleExp, *this->m_activeColscaleExp);
   }

   /* scale */
   this->applyScaling(lp);

   MSG_INFO3((*this->spxout), (*this->spxout) << "Row scaling min= " << this->minAbsRowscale()
             << " max= " << this->maxAbsRowscale()
             << std::endl
             << "Col scaling min= " << this->minAbsColscale()
             << " max= " << this->maxAbsColscale()
             << std::endl;)

   MSG_INFO2((*this->spxout), (*this->spxout) << "after scaling: "
             << " min= " << lp.minAbsNzo(false)
             << " max= " << lp.maxAbsNzo(false)
             << " col-ratio= " << this->maxColRatio(lp)
             << " row-ratio= " << this->maxRowRatio(lp)
             << std::endl;)

}

} // namespace soplex
