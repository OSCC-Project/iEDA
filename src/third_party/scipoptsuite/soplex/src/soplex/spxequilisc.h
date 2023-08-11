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

/**@file  spxequilisc.h
 * @brief LP equilibrium scaling.
 */
#ifndef _SPXEQUILISC_H_
#define _SPXEQUILISC_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxscaler.h"

namespace soplex
{
/**@brief Equilibrium row/column scaling.
   @ingroup Algo

   This SPxScaler implementation performs equilibrium scaling of the
   LPs rows and columns.
*/
template <class R>
class SPxEquiliSC : public SPxScaler<R>
{
public:
   /// compute equilibrium scaling vector rounded to power of two
   static void computeEquiExpVec(const SVSetBase<R>* vecset, const DataArray<int>& coScaleExp,
                                 DataArray<int>& scaleExp);

   /// compute equilibrium scaling vector rounded to power of two
   static void computeEquiExpVec(const SVSetBase<R>* vecset, const std::vector<R>& coScaleVal,
                                 DataArray<int>& scaleExp);

   /// compute equilibrium scaling rounded to power of 2 for existing R scaling factors (preRowscale, preColscale)
   static void computePostequiExpVecs(const SPxLPBase<R>& lp, const std::vector<R>& preRowscale,
                                      const std::vector<R>& preColscale,
                                      DataArray<int>& rowscaleExp, DataArray<int>& colscaleExp);
   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor (this scaler makes no use of inherited member m_colFirst)
   explicit SPxEquiliSC(bool doBoth = true);
   /// copy constructor
   SPxEquiliSC(const SPxEquiliSC& old);
   /// assignment operator
   SPxEquiliSC& operator=(const SPxEquiliSC&);
   /// destructor
   virtual ~SPxEquiliSC()
   {}
   /// clone function for polymorphism
   inline virtual SPxScaler<R>* clone() const override
   {
      return new SPxEquiliSC<R>(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Scaling */
   ///@{
   /// Scale the loaded SPxLP.
   virtual void scale(SPxLPBase<R>& lp, bool persistent = false) override;
   ///@}
};
} // namespace soplex

#include "spxequilisc.hpp"

#endif // _SPXEQUILISC_H_
