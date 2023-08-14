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

/**@file  spxgeometsc.hpp
 * @brief Geometric mean row/column scaling.
 */
#include <assert.h>

#include "soplex/spxout.h"
#include "soplex/spxlpbase.h"
#include "soplex/spxequilisc.h"

namespace soplex
{

template <class R>
static R computeScalingVec(
   const SVSetBase<R>*             vecset,
   const std::vector<R>& coScaleval,
   std::vector<R>&       scaleval)
{
   R pmax = 0.0;

   assert(scaleval.size() == unsigned(vecset->num()));

   for(int i = 0; i < vecset->num(); ++i)
   {
      const SVectorBase<R>& vec = (*vecset)[i];

      R maxi = 0.0;
      R mini = R(infinity);

      for(int j = 0; j < vec.size(); ++j)
      {
         const R x = spxAbs(vec.value(j) * coScaleval[unsigned(vec.index(j))]);

         if(!isZero(x))
         {
            if(x > maxi)
               maxi = x;

            if(x < mini)
               mini = x;
         }
      }

      // empty rows/cols are possible
      if(mini == R(infinity) || maxi == 0.0)
      {
         mini = 1.0;
         maxi = 1.0;
      }

      assert(mini < R(infinity));
      assert(maxi > 0.0);

      scaleval[unsigned(i)] = 1.0 / spxSqrt(mini * maxi);

      const R p = maxi / mini;

      if(p > pmax)
         pmax = p;
   }

   return pmax;
}

template <class R>
SPxGeometSC<R>::SPxGeometSC(bool equilibrate, int maxIters, R minImpr, R goodEnough)
   : SPxScaler<R>("Geometric")
   , postequilibration(equilibrate)
   , m_maxIterations(maxIters)
   , m_minImprovement(minImpr)
   , m_goodEnoughRatio(goodEnough)
{
   assert(maxIters > 0);
   assert(minImpr > 0.0 && minImpr <= 1.0);
   assert(goodEnough >= 0.0);
}

template <class R>
SPxGeometSC<R>::SPxGeometSC(const SPxGeometSC<R>& old)
   : SPxScaler<R>(old)
   , postequilibration(old.postequilibration)
   , m_maxIterations(old.m_maxIterations)
   , m_minImprovement(old.m_minImprovement)
   , m_goodEnoughRatio(old.m_goodEnoughRatio)
{
   assert(m_maxIterations > 0);
   assert(m_minImprovement > 0.0 && m_minImprovement <= 1.0);
   assert(m_goodEnoughRatio >= 0.0);
}

template <class R>
SPxGeometSC<R>& SPxGeometSC<R>::operator=(const SPxGeometSC<R>& rhs)
{
   if(this != &rhs)
   {
      SPxScaler<R>::operator=(rhs);
   }

   return *this;
}

template <class R>
void SPxGeometSC<R>::scale(SPxLPBase<R>& lp, bool persistent)
{

   MSG_INFO1((*this->spxout), (*this->spxout) << "Geometric scaling LP" <<
             (persistent ? " (persistent)" : "") << (postequilibration ? " with post-equilibration" : "") <<
             std::endl;)

   this->setup(lp);

   /* We want to do that direction first, with the lower ratio.
    * See SPxEquiliSC<R>::scale() for a reasoning.
    */
   const R colratio = this->maxColRatio(lp);
   const R rowratio = this->maxRowRatio(lp);

   const bool colFirst = colratio < rowratio;

   R p0start;
   R p1start;

   if(colFirst)
   {
      p0start = colratio;
      p1start = rowratio;
   }
   else
   {
      p0start = rowratio;
      p1start = colratio;
   }

   MSG_INFO2((*this->spxout), (*this->spxout) << "before scaling:"
             << " min= " << lp.minAbsNzo()
             << " max= " << lp.maxAbsNzo()
             << " col-ratio= " << colratio
             << " row-ratio= " << rowratio
             << std::endl;)

   // perform geometric scaling only if maximum ratio is above threshold
   bool geoscale = p1start > m_goodEnoughRatio;

   if(!geoscale)
   {
      MSG_INFO2((*this->spxout), (*this->spxout) << "No geometric scaling done, ratio good enough" <<
                std::endl;)

      if(!postequilibration)
      {
         lp.setScalingInfo(true);
         return;
      }

      MSG_INFO2((*this->spxout), (*this->spxout) << " ... but will still perform equilibrium scaling" <<
                std::endl;)
   }

   std::vector<R> rowscale(unsigned(lp.nRows()), 1.0);
   std::vector<R> colscale(unsigned(lp.nCols()), 1.0);

   R p0 = 0.0;
   R p1 = 0.0;

   if(geoscale)
   {
      R p0prev = p0start;
      R p1prev = p1start;

      // we make at most maxIterations.
      for(int count = 0; count < m_maxIterations; count++)
      {
         if(colFirst)
         {
            p0 = computeScalingVec(lp.colSet(), rowscale, colscale);
            p1 = computeScalingVec(lp.rowSet(), colscale, rowscale);
         }
         else
         {
            p0 = computeScalingVec(lp.rowSet(), colscale, rowscale);
            p1 = computeScalingVec(lp.colSet(), rowscale, colscale);
         }

         MSG_INFO3((*this->spxout), (*this->spxout) << "Geometric scaling round " << count
                   << " col-ratio= " << (colFirst ? p0 : p1)
                   << " row-ratio= " << (colFirst ? p1 : p0)
                   << std::endl;)

         if(p0 > m_minImprovement * p0prev && p1 > m_minImprovement * p1prev)
            break;

         p0prev = p0;
         p1prev = p1;
      }

      // perform geometric scaling only if there is enough (default 15%) improvement.
      geoscale = (p0 <= m_minImprovement * p0start || p1 <= m_minImprovement * p1start);
   }

   if(!geoscale && !postequilibration)
   {
      MSG_INFO2((*this->spxout), (*this->spxout) << "No geometric scaling done." << std::endl;)
      lp.setScalingInfo(true);
   }
   else
   {
      DataArray<int>& colscaleExp = *this->m_activeColscaleExp;
      DataArray<int>& rowscaleExp = *this->m_activeRowscaleExp;

      if(postequilibration)
      {
         if(!geoscale)
         {
            std::fill(rowscale.begin(), rowscale.end(), 1.0);
            std::fill(colscale.begin(), colscale.end(), 1.0);
         }

         SPxEquiliSC<R>::computePostequiExpVecs(lp, rowscale, colscale, rowscaleExp, colscaleExp);
      }
      else
      {
         this->computeExpVec(colscale, colscaleExp);
         this->computeExpVec(rowscale, rowscaleExp);
      }

      this->applyScaling(lp);

      MSG_INFO3((*this->spxout), (*this->spxout) << "Row scaling min= " << this->minAbsRowscale()
                << " max= " << this->maxAbsRowscale()
                << std::endl
                << "IGEOSC06 Col scaling min= " << this->minAbsColscale()
                << " max= " << this->maxAbsColscale()
                << std::endl;)

      MSG_INFO2((*this->spxout), (*this->spxout) << "after scaling: "
                << " min= " << lp.minAbsNzo(false)
                << " max= " << lp.maxAbsNzo(false)
                << " col-ratio= " << this->maxColRatio(lp)
                << " row-ratio= " << this->maxRowRatio(lp)
                << std::endl;)
   }
}


} // namespace soplex
