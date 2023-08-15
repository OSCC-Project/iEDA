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


/**@file  spxsteepexpr.h
 * @brief Steepest edge pricer with exact initialization of weights.
 */
#ifndef _SPXSTEEPEXPR_H_
#define _SPXSTEEPEXPR_H_


#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxsteeppr.h"

namespace soplex
{

/**@brief   Steepest edge pricer.
   @ingroup Algo

   Class SPxSteepExPR implements a steepest edge pricer to be used with
   SoPlex. Exact initialization of weights is used.

   See SPxPricer for a class documentation.
*/
template <class R>
class SPxSteepExPR : public SPxSteepPR<R>
{

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   ///
   SPxSteepExPR()
      : SPxSteepPR<R>("SteepEx", SPxSteepPR<R>::EXACT)
   {
      assert(this->isConsistent());
   }
   /// copy constructor
   SPxSteepExPR(const SPxSteepExPR& old)
      : SPxSteepPR<R>(old)
   {
      assert(this->isConsistent());
   }
   /// assignment operator
   SPxSteepExPR& operator=(const SPxSteepExPR& rhs)
   {
      if(this != &rhs)
      {
         SPxSteepPR<R>::operator=(rhs);

         assert(this->isConsistent());
      }

      return *this;
   }
   /// destructor
   virtual ~SPxSteepExPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxSteepPR<R>* clone()  const
   {
      return new SPxSteepExPR(*this);
   }
   ///@}
};

} // namespace soplex

#include "spxsteeppr.hpp"

#endif // _SPXSTEEPPR_H_
