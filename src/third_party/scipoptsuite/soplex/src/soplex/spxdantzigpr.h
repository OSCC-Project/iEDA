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

/**@file  spxdantzigpr.h
 * @brief Dantzig pricer.
 */
#ifndef _SPXDEFAULTPR_H_
#define _SPXDEFAULTPR_H_

#include <assert.h>

#include "soplex/spxpricer.h"

namespace soplex
{

/**@brief   Dantzig pricer.
   @ingroup Algo

   Class SPxDantzigPR is an implementation class of an SPxPricer implementing
   Dantzig's default pricing strategy, i.e., maximal/minimal reduced cost or
   maximally violated constraint.

   See SPxPricer for a class documentation.
*/
template <class R>
class SPxDantzigPR : public SPxPricer<R>
{
private:
   int                   selectLeaveSparse();/**< sparse pricing method for leaving Simplex */

   SPxId
   selectEnterX();                                /**< choose the best entering index among columns and rows but prefer sparsity */
   SPxId                 selectEnterSparseDim(R& best,
         SPxId& id);   /**< sparse pricing method for entering Simplex (slack variables)*/
   SPxId                 selectEnterSparseCoDim(R& best,
         SPxId& id); /**< sparse pricing method for entering Simplex */
   SPxId                 selectEnterDenseDim(R& best,
         SPxId& id);    /**< selectEnter() in dense case (slack variables) */
   SPxId                 selectEnterDenseCoDim(R& best,
         SPxId& id);  /**< selectEnter() in dense case */
public:

   //-------------------------------------
   /**@name Constructors / destructors */
   ///@{
   /// default constructor
   SPxDantzigPR()
      : SPxPricer<R>("Dantzig")
   {}
   /// copy constructor
   SPxDantzigPR(const SPxDantzigPR& old)
      : SPxPricer<R>(old)
   {}
   /// assignment operator
   SPxDantzigPR& operator=(const SPxDantzigPR& rhs)
   {
      if(this != &rhs)
      {
         SPxPricer<R>::operator=(rhs);
      }

      return *this;
   }
   /// destructor
   virtual ~SPxDantzigPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxPricer<R>* clone() const
   {
      return new SPxDantzigPR(*this);
   }
   ///@}


   //-------------------------------------
   /**@name Select enter/leave */
   ///@{
   ///
   virtual int selectLeave();
   ///
   virtual SPxId selectEnter();
   ///@}
};
} // namespace soplex

// For general tempalted functions
#include "spxdantzigpr.hpp"

#endif // _SPXDEFAULTPRR_H_
