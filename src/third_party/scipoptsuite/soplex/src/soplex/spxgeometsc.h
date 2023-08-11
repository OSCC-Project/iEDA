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

/**@file  spxgeometsc.h
 * @brief LP geometric mean scaling.
 */
#ifndef _SPXGEOMETSC_H_
#define _SPXGEOMETSC_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxscaler.h"

namespace soplex
{
/**@brief Geometric mean row/column scaling.
   @ingroup Algo

   This SPxScaler implementation performs geometric mean scaling of the
   LPs rows and columns.
*/
template <class R>
class SPxGeometSC : public SPxScaler<R>
{
protected:

   //-------------------------------------
   /**@name Data */
   ///@{
   const bool postequilibration;  ///< equilibrate after geometric scaling?
   const int  m_maxIterations;    ///< maximum number of scaling iterations.
   const R m_minImprovement;   ///< improvement necessary to carry on. (Bixby said Fourer said in MP 23, 274 ff. that 0.9 is a good value)
   const R m_goodEnoughRatio;  ///< no scaling needed if ratio is less than this.
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor (this scaler makes no use of inherited members m_colFirst and m_doBoth)
   explicit SPxGeometSC(bool equilibrate = false, int maxIters = 8, R minImpr = 0.85,
                        R goodEnough = 1e3);
   /// copy constructor
   SPxGeometSC(const SPxGeometSC& old);
   /// assignment operator
   SPxGeometSC& operator=(const SPxGeometSC&);
   /// destructor
   virtual ~SPxGeometSC()
   {}
   /// clone function for polymorphism
   inline virtual SPxScaler<R>* clone() const override
   {
      return new SPxGeometSC(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Scaling */
   ///@{
   /// Scale the loaded SPxLPBase<R>.
   virtual void scale(SPxLPBase<R>& lp, bool persistent = true) override;
   ///@}

};
} // namespace soplex

#include "spxgeometsc.hpp"

#endif // _SPXGEOMETSC_H_
