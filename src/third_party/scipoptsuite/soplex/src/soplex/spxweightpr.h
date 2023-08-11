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

/**@file  spxweightpr.h
 * @brief Weighted pricing.
 */
#ifndef _SPXWEIGHTPR_H_
#define _SPXWEIGHTPR_H_

#include "soplex/spxdefines.h"
#include "soplex/spxpricer.h"

namespace soplex
{

/**@brief   Weighted pricing.
   @ingroup Algo

   Class SPxWeightPR is an implemantation class of SPxPricer that uses
   weights for columns and rows for selecting the Simplex pivots. The weights
   are computed by methods #computeCP() and #computeRP() which may be
   overridden by derived classes.

   The weights are interpreted as follows: The higher a value is, the more
   likely the corresponding row or column is set on one of its bounds.

   See SPxPricer for a class documentation.
*/
template <class R>
class SPxWeightPR : public SPxPricer<R>
{
private:

   //-------------------------------------
   /**@name Data */
   ///@{
   /// column penalties
   VectorBase<R> cPenalty;
   /// row penalties
   VectorBase<R> rPenalty;
   /// penalties for leaving alg
   VectorBase<R> leavePenalty;
   ///
   const R* penalty;
   ///
   const R* coPenalty;
   /// length of objective vector.
   R objlength;
   ///@}

   //-------------------------------------
   /**@name Private helpers */
   ///@{
   /// compute leave penalties.
   void computeLeavePenalty(int start, int end);
   /// compute weights for columns.
   void computeCP(int start, int end);
   /// compute weights for rows.
   void computeRP(int start, int end);
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxWeightPR()
      : SPxPricer<R>("Weight")
      , penalty(0)
      , coPenalty(0)
      , objlength(0)
   {}
   /// copy constructor
   SPxWeightPR(const SPxWeightPR& old)
      : SPxPricer<R>(old)
      , cPenalty(old.cPenalty)
      , rPenalty(old.rPenalty)
      , leavePenalty(old.leavePenalty)
      , penalty(0)
      , coPenalty(0)
      , objlength(old.objlength)
   {
      if(old.penalty == old.rPenalty.get_const_ptr())
      {
         penalty = rPenalty.get_const_ptr();
         coPenalty = cPenalty.get_const_ptr();
      }
      else if(old.penalty == old.cPenalty.get_const_ptr())
      {
         penalty = cPenalty.get_const_ptr();
         coPenalty = rPenalty.get_const_ptr();
      }

      // otherwise, old.penalty and old.coPenalty are not set and do not have to be copied
   }
   /// assignment operator
   SPxWeightPR& operator=(const SPxWeightPR& rhs)
   {
      if(this != &rhs)
      {
         SPxPricer<R>::operator=(rhs);
         cPenalty = rhs.cPenalty;
         rPenalty = rhs.rPenalty;
         leavePenalty = rhs.leavePenalty;
         objlength = rhs.objlength;

         if(rhs.penalty == rhs.rPenalty.get_const_ptr())
         {
            penalty = rPenalty.get_const_ptr();
            coPenalty = cPenalty.get_const_ptr();
         }
         else if(rhs.penalty == rhs.cPenalty.get_const_ptr())
         {
            penalty = cPenalty.get_const_ptr();
            coPenalty = rPenalty.get_const_ptr();
         }

         // otherwise, old.penalty and old.coPenalty are not set and do not have to be copied
      }

      return *this;
   }
   /// destructor
   virtual ~SPxWeightPR()
   {}
   /// clone function for polymorphism
   inline virtual SPxPricer<R>* clone()  const
   {
      return new SPxWeightPR(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Access / modification */
   ///@{
   /// sets the solver
   virtual void load(SPxSolverBase<R>* base);
   /// set entering/leaving algorithm
   void setType(typename SPxSolverBase<R>::Type tp);
   /// set row/column representation
   void setRep(typename SPxSolverBase<R>::Representation rep);
   ///
   virtual int selectLeave();
   ///
   virtual SPxId selectEnter();
   /// \p n vectors have been added to the loaded LP.
   virtual void addedVecs(int n);
   /// \p n covectors have been added to the loaded LP.
   virtual void addedCoVecs(int n);
   /// \p the i'th vector has been removed from the loaded LP.
   virtual void removedVec(int i);
   /// \p the i'th covector has been removed from the loaded LP.
   virtual void removedCoVec(int i);
   /// \p n vectors have been removed from the loaded LP.
   virtual void removedVecs(const int perm[]);
   /// \p n covectors have been removed from the loaded LP.
   virtual void removedCoVecs(const int perm[]);
   ///@}

   //-------------------------------------
   /**@name Consistency check */
   ///@{
   /// checks for consistency
   virtual bool isConsistent() const;
   ///@}
};
} // namespace soplex

// For general templated functions
#include "spxweightpr.hpp"

#endif // _SPXWEIGHTPR_H_
