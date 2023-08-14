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

/**@file  spxvectorst.h
 * @brief Solution vector based start basis.
 */
#ifndef _SPXVECTORST_H_
#define _SPXVECTORST_H_

#include <assert.h>

#include "soplex/spxweightst.h"
#include "soplex/vector.h"

namespace soplex
{

/**@brief   Solution vector based start basis.
   @ingroup Algo

   This version of SPxWeightST can be used to construct a starting basis for
   an LP to be solved with SoPlex if an approximate solution vector or dual
   vector (possibly optained by a heuristic) is available. This is done by
   setting up weights for the SPxWeightST it is derived from.

   The primal vector to be used is loaded by calling method #primal() while
   #dual() setups for the dual vector. Methods #primal() or #dual() must be
   called \em before #generate() is called by SoPlex to set up a
   starting basis. If more than one call of method #primal() or #dual()
   occurred only the most recent one is valid for generating the starting base.
*/
template <class R>
class SPxVectorST : public SPxWeightST<R>
{
private:

   //-------------------------------------
   /**@name Types */
   ///@{
   /// specifies whether to work on the primal, the dual, or not at all.
   enum { NONE, PVEC, DVEC } state;
   ///@}

   //-------------------------------------
   /**@name Data */
   ///@{
   /// the current (approximate) primal or dual vector
   VectorBase<R> vec;
   ///@}

protected:

   //-------------------------------------
   /**@name Protected helpers */
   ///@{
   /// sets up variable weights.
   void setupWeights(SPxSolverBase<R>& base);
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor.
   SPxVectorST()
      : state(NONE)
   {
      this->m_name = "vector";
   }
   /// copy constructor
   SPxVectorST(const SPxVectorST& old)
      : SPxWeightST<R>(old)
      , state(old.state)
      , vec(old.vec)
   {
      assert(this->isConsistent());
   }
   /// assignment operator
   SPxVectorST& operator=(const SPxVectorST& rhs)
   {
      if(this != &rhs)
      {
         SPxWeightST<R>::operator=(rhs);
         state = rhs.state;
         vec = rhs.vec;

         assert(this->isConsistent());
      }

      return *this;
   }
   /// destructor.
   virtual ~SPxVectorST()
   {}
   /// clone function for polymorphism
   inline virtual SPxStarter<R>* clone() const
   {
      return new SPxVectorST(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Modification */
   ///@{
   /// sets up primal solution vector.
   void primal(const VectorBase<R>& v)
   {
      vec = v;
      state = PVEC;
   }
   /// sets up primal solution vector.
   void dual(const VectorBase<R>& v)
   {
      vec = v;
      state = DVEC;
   }
   ///@}

};

} // namespace soplex

// For general templated files
#include "spxvectorst.hpp"
#endif // _SPXVECTORST_H_
