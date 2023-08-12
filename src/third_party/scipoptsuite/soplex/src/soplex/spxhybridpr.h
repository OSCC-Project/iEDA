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

/**@file  spxhybridpr.h
 * @brief Hybrid pricer.
 */
#ifndef _SPXHYBRIDPR_H_
#define _SPXHYBRIDPR_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxpricer.h"
#include "soplex/spxdevexpr.h"
#include "soplex/spxparmultpr.h"
#include "soplex/spxsteeppr.h"

namespace soplex
{

/**@brief   Hybrid pricer.
   @ingroup Algo

   The hybrid pricer for SoPlex tries to guess the best pricing strategy to
   use for pricing the loaded LP with the loaded algorithm type and basis
   representation. Currently it does so by switching between SPxSteepPR,
   SPxDevexPR and SPxParMultPR.

   See SPxPricer for a class documentation.
*/
template <class R>
class SPxHybridPR : public SPxPricer<R>
{
   //-------------------------------------
   /**@name Data */
   ///@{
   /// steepest edge pricer
   SPxSteepPR<R>   steep;
   /// partial multiple pricer
   SPxParMultPR<R> parmult;
   /// devex pricer
   SPxDevexPR<R>   devex;
   /// the currently used pricer
   SPxPricer<R>*   thepricer;
   /// factor between dim and coDim of the problem to decide about the pricer
   R hybridFactor;
   ///@}

public:

   //-------------------------------------
   /**@name Access / modification */
   ///@{
   /// sets the epsilon
   virtual void setEpsilon(R eps);
   /// sets the solver
   virtual void load(SPxSolverBase<R>* solver);
   /// clears all pricers and unselects the current pricer
   virtual void clear();
   /// sets entering or leaving algorithm
   virtual void setType(typename SPxSolverBase<R>::Type tp);
   /// sets row or column representation
   virtual void setRep(typename SPxSolverBase<R>::Representation rep);
   /// selects the leaving algorithm
   virtual int selectLeave();
   /// selects the entering algorithm
   virtual SPxId selectEnter();
   /// calls left4 on the current pricer
   virtual void left4(int n, SPxId id);
   /// calls entered4 on the current pricer
   virtual void entered4(SPxId id, int n);
   /// calls addedVecs(n) on all pricers
   virtual void addedVecs(int n);
   /// calls addedCoVecs(n) on all pricers
   virtual void addedCoVecs(int n);
   ///@}

   //-------------------------------------
   /**@name Consistency check */
   ///@{
   /// consistency check
   virtual bool isConsistent() const;
   ///@}

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxHybridPR()
      : SPxPricer<R>("Hybrid")
      , thepricer(0)
      , hybridFactor(3.0) // we want the ParMult pricer
   {}
   /// copy constructor
   SPxHybridPR(const SPxHybridPR& old)
      : SPxPricer<R>(old)
      , steep(old.steep)
      , parmult(old.parmult)
      , devex(old.devex)
      , hybridFactor(old.hybridFactor)
   {
      if(old.thepricer == &old.steep)
      {
         thepricer = &steep;
      }
      else if(old.thepricer == &old.parmult)
      {
         thepricer = &parmult;
      }
      else if(old.thepricer == &old.devex)
      {
         thepricer = &devex;
      }
      else // old.thepricer should be 0
      {
         thepricer = 0;
      }
   }
   /// assignment operator
   SPxHybridPR& operator=(const SPxHybridPR& rhs)
   {
      if(this != &rhs)
      {
         SPxPricer<R>::operator=(rhs);
         steep = rhs.steep;
         parmult = rhs.parmult;
         devex = rhs.devex;
         hybridFactor = rhs.hybridFactor;

         if(rhs.thepricer == &rhs.steep)
         {
            thepricer = &steep;
         }
         else if(rhs.thepricer == &rhs.parmult)
         {
            thepricer = &parmult;
         }
         else if(rhs.thepricer == &rhs.devex)
         {
            thepricer = &devex;
         }
         else // rhs.thepricer should be 0
         {
            thepricer = 0;
         }
      }

      return *this;
   }
   /// destructor
   virtual ~SPxHybridPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxPricer<R>* clone()  const
   {
      return new SPxHybridPR(*this);
   }
   ///@}
};

} // namespace soplex

// For general templated functions
#include "spxhybridpr.hpp"
#endif // _SPXHYBRIDPR_H_
