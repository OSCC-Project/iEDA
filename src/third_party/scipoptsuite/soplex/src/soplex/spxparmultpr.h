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

/**@file  spxparmultpr.h
 * @brief Partial multiple pricing.
 */
#ifndef _SPXPARMULTPR_H_
#define _SPXPARMULTPR_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxpricer.h"
#include "soplex/dataarray.h"
#include "soplex/array.h"
#include "soplex/ssvector.h"

namespace soplex
{

/**@brief   Partial multiple pricing.
   @ingroup Algo

   Class SPxParMultPr is an implementation class for SPxPricer implementing
   Dantzig's default pricing strategy with partial multiple pricing.
   Partial multiple pricing applies to the entering Simplex only. A set of
   #partialSize eligible pivot indices is selected (partial pricing). In the
   following Simplex iterations pricing is restricted to these indices
   (multiple pricing) until no more eliiable pivots are available. Partial
   multiple pricing significantly reduces the computation time for computing
   the matrix-vector-product in the Simplex algorithm.

   See SPxPricer for a class documentation.
*/
template <class R>
class SPxParMultPR : public SPxPricer<R>
{
private:

   //-------------------------------------
   /**@name Private types */
   ///@{
   /// Helper structure.
   struct SPxParMultPr_Tmp
   {
      ///
      SPxId id;
      ///
      R test;
   };
   ///@}

   //-------------------------------------
   /**@name Helper data */
   ///@{
   ///
   Array < SPxParMultPr_Tmp > pricSet;
   ///
   int multiParts;
   ///
   int used;
   ///
   int min;
   ///
   int last;
   /// Set size for partial pricing.
   int partialSize;
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxParMultPR()
      : SPxPricer<R>("ParMult")
      , multiParts(0)
      , used(0)
      , min(0)
      , last(0)
      , partialSize(17)
   {}
   /// copy constructor
   SPxParMultPR(const SPxParMultPR& old)
      : SPxPricer<R>(old)
      , pricSet(old.pricSet)
      , multiParts(old.multiParts)
      , used(old.used)
      , min(old.min)
      , last(old.last)
      , partialSize(old.partialSize)
   {}
   /// assignment operator
   SPxParMultPR& operator=(const SPxParMultPR& rhs)
   {
      if(this != &rhs)
      {
         SPxPricer<R>::operator=(rhs);
         pricSet = rhs.pricSet;
         multiParts = rhs.multiParts;
         used = rhs.used;
         min = rhs.min;
         last = rhs.last;
         partialSize = rhs.partialSize;
      }

      return *this;
   }
   /// destructor
   virtual ~SPxParMultPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxPricer<R>* clone()  const
   {
      return new SPxParMultPR(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Interface */
   ///@{
   /// set the solver
   virtual void load(SPxSolverBase<R>* solver);
   /// set entering or leaving algorithm
   virtual void setType(typename SPxSolverBase<R>::Type tp);
   ///
   virtual int selectLeave();
   ///
   virtual SPxId selectEnter();
   ///@}

};

} // namespace soplex

#include "spxparmultpr.hpp"
#endif // _SPXPARMULTPRR_H_
