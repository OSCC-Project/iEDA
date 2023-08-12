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


/**@file  spxweightst.h
 * @brief Weighted start basis.
 */
#ifndef _SPXWEIGHTST_H_
#define _SPXWEIGHTST_H_


#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxstarter.h"
#include "soplex/dataarray.h"

namespace soplex
{

/**@brief   Weighted start basis.
   @ingroup Algo

   Class SPxWeightST is an implementation of a SPxStarter for generating a
   Simplex starting basis. Using method #setupWeights() it sets up arrays
   #weight and #coWeight, or equivalently #rowWeight and #colWeight.
   (#rowWeight and #colWeight are just pointers initialized to #weight and
   #coWeight according to the representation of SoPlex \p base passed to
   method #generate().)

   The weight values are then used to setup a starting basis for the LP:
   vectors with low values are likely to become dual (i.e. basic for a column
   basis) and such with high values are likely to become primal (i.e. nonbasic
   for a column basis).

   However, if a variable having an upper and lower bound is to become primal,
   there is still a choice for setting it either to its upper or lower bound.
   Members #rowRight and #colUp are used to determine where to set a primal
   variable. If #rowRight[i] is set to a nonzero value, the right-hand side
   inequality is set tightly for the \p i 'th to become primal. Analogously, If
   #colUp[j] is nonzero, the \p j 'th variable will be set to its upper bound
   if it becomes primal.
*/
template <class R>
class SPxWeightST : public SPxStarter<R>
{
private:

   //-----------------------------------
   /**@name Private data */
   ///@{
   ///
   DataArray < int > forbidden;
   ///
   Array < R >* weight;
   ///
   Array < R >* coWeight;
   ///@}

   //-----------------------------------
   /**@name Private helpers */
   ///@{
   ///
   void setPrimalStatus(typename SPxBasisBase<R>::Desc&, const SPxSolverBase<R>&, const SPxId&);
   ///@}

protected:

   //-----------------------------------
   /**@name Protected data */
   ///@{
   /// weight value for LP rows.
   Array < R > rowWeight;
   /// weight value for LP columns.
   Array < R > colWeight;
   /// set variable to rhs?.
   DataArray < bool > rowRight;
   /// set primal variable to upper bound.
   DataArray < bool > colUp;
   ///@}

   //-----------------------------------
   /**@name Protected helpers */
   ///@{
   /// sets up variable weights.
   /** This method is called in order to setup the weights for all
       variables. It has been declared \c virtual in order to allow for
       derived classes to compute other weight values.
   */
   virtual void setupWeights(SPxSolverBase<R>& base);
   ///@}

public:

   //-----------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor.
   SPxWeightST()
      : SPxStarter<R>("Weight")
   {
      weight = 0;
      coWeight = 0;
      assert(isConsistent());
   }
   /// copy constructor
   SPxWeightST(const SPxWeightST& old)
      : SPxStarter<R>(old)
      , forbidden(old.forbidden)
      , rowWeight(old.rowWeight)
      , colWeight(old.colWeight)
      , rowRight(old.rowRight)
      , colUp(old.colUp)
   {
      if(old.weight == &old.colWeight)
      {
         weight   = &colWeight;
         coWeight = &rowWeight;
      }
      else if(old.weight == &old.rowWeight)
      {
         weight   = &rowWeight;
         coWeight = &colWeight;
      }
      else  // old.weight and old.coWeight are not set correctly, do nothing.
      {
         weight = 0;
         coWeight = 0;
      }

      assert(isConsistent());
   }
   /// assignment operator
   SPxWeightST& operator=(const SPxWeightST& rhs)
   {
      if(this != &rhs)
      {
         SPxStarter<R>::operator=(rhs);
         forbidden = rhs.forbidden;
         rowWeight = rhs.rowWeight;
         colWeight = rhs.colWeight;
         rowRight = rhs.rowRight;
         colUp = rhs.colUp;

         if(rhs.weight == &rhs.colWeight)
         {
            weight   = &colWeight;
            coWeight = &rowWeight;
         }
         else if(rhs.weight == &rhs.rowWeight)
         {
            weight   = &rowWeight;
            coWeight = &colWeight;
         }
         else  // old.weight and old.coWeight are not set correctly, do nothing.
         {}

         assert(isConsistent());
      }

      return *this;
   }
   /// destructor.
   virtual ~SPxWeightST()
   {
      weight   = 0;
      coWeight = 0;
   }
   /// clone function for polymorphism
   inline virtual SPxStarter<R>* clone() const
   {
      return new SPxWeightST(*this);
   }
   ///@}

   //-----------------------------------
   /**@name Generation of a start basis */
   ///@{
   /// generates start basis for loaded basis.
   void generate(SPxSolverBase<R>& base);
   ///@}

   //-----------------------------------
   /**@name Debugging */
   ///@{
   /// consistency check.
   virtual bool isConsistent() const;
   ///@}

};

} // namespace soplex

// For general templated functions
#include "spxweightst.hpp"

#endif // _SPXWEIGHTST_H_
