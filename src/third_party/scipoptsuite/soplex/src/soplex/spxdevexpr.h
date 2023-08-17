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

/**@file  spxdevexpr.h
 * @brief Devex pricer.
 */
#ifndef _SPXDEVEXPR_H_
#define _SPXDEVEXPR_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxpricer.h"

namespace soplex
{

/**@brief   Devex pricer.
   @ingroup Algo

   The Devex Pricer for SoPlex implements an approximate steepest edge pricing,
   that does without solving an extra linear system and computing the scalar
   products.

   See SPxPricer for a class documentation.

   @todo There seem to be problems with this pricer especially on the
         greenbe[ab] problems with the entering algorithm
         (row representation?).
*/
template <class R>
class SPxDevexPR : public SPxPricer<R>
{
private:

   //-------------------------------------
   /**@name Data */
   ///@{
   R  last;           ///< penalty, selected at last iteration.
   Array<typename SPxPricer<R>::IdxElement>
   prices;   ///< temporary array of precomputed pricing values
   Array<typename SPxPricer<R>::IdxElement>
   pricesCo; ///< temporary array of precomputed pricing values
   DIdxSet bestPrices;   ///< set of best pricing candidates
   DIdxSet bestPricesCo; ///< set of best pricing candidates
   bool refined;         ///< has a refinement step already been tried?
   ///@}

   //-------------------------------------
   /**@name Private helpers */
   ///@{
   /// set entering/leaving algorithm
   void setupWeights(typename SPxSolverBase<R>::Type);
   /// build up vector of pricing values for later use
   int buildBestPriceVectorLeave(R feastol);
   /// internal implementation of SPxPricer::selectLeave()
   int selectLeaveX(R feastol, int start = 0, int incr = 1);
   /// implementation of sparse pricing in the leaving Simplex
   int selectLeaveSparse(R feastol);
   /// implementation of hyper sparse pricing in the leaving Simplex
   int selectLeaveHyper(R feastol);
   /// build up vector of pricing values for later use
   SPxId buildBestPriceVectorEnterDim(R& best, R feastol);
   SPxId buildBestPriceVectorEnterCoDim(R& best, R feastol);
   /// choose the best entering index among columns and rows but prefer sparsity
   SPxId selectEnterX(R tol);
   /// implementation of sparse pricing in the entering Simplex (slack variables)
   SPxId selectEnterSparseDim(R& best, R feastol);
   /// implementation of sparse pricing in the entering Simplex
   SPxId selectEnterSparseCoDim(R& best, R feastol);
   /// SPxPricer::selectEnter() in dense case (slack variabels)
   SPxId selectEnterDenseDim(R& best, R feastol, int start = 0, int incr = 1);
   /// SPxPricer::selectEnter() in dense case
   SPxId selectEnterDenseCoDim(R& best, R feastol, int start = 0, int incr = 1);
   /// implementation of hyper sparse pricing in the entering Simplex
   SPxId selectEnterHyperDim(R& best, R feastol);
   /// implementation of hyper sparse pricing in the entering Simplex
   SPxId selectEnterHyperCoDim(R& best, R feastol);
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxDevexPR()
      : SPxPricer<R>("Devex")
      , last(0)
      , refined(false)
   {}
   /// copy constructor
   SPxDevexPR(const SPxDevexPR& old)
      : SPxPricer<R>(old)
      , last(old.last)
      , refined(false)
   {}
   /// assignment operator
   SPxDevexPR& operator=(const SPxDevexPR& rhs)
   {
      if(this != &rhs)
      {
         SPxPricer<R>::operator=(rhs);
         last = rhs.last;
      }

      return *this;
   }
   /// destructor
   virtual ~SPxDevexPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxPricer<R>* clone()  const
   {
      return new SPxDevexPR(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Access / modification */
   ///@{
   /// sets the solver
   virtual void load(SPxSolverBase<R>* base);
   /// set entering/leaving algorithm
   virtual void setType(typename SPxSolverBase<R>::Type);
   /// set row/column representation
   virtual void setRep(typename SPxSolverBase<R>::Representation);
   ///
   virtual int selectLeave();
   ///
   virtual SPxId selectEnter();
   ///
   virtual void left4(int n, SPxId id);
   ///
   virtual void entered4(SPxId id, int n);
   /// \p n vectors have been added to loaded LP.
   virtual void addedVecs(int n);
   /// \p n covectors have been added to loaded LP.
   virtual void addedCoVecs(int n);
   ///@}

   //-------------------------------------
   /**@name Consistency check */
   ///@{
   /// consistency check
   virtual bool isConsistent() const;
   ///@}
};

} // namespace soplex

#include "spxdevexpr.hpp"

#endif // _SPXDEVEXPR_H_
