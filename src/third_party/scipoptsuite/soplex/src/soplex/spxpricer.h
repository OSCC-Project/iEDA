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


/**@file  spxpricer.h
 * @brief Abstract pricer base class.
 */
#ifndef _SPXPRICE_H_
#define _SPXPRICE_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxsolver.h"
#include "soplex/sorter.h"

namespace soplex
{

/**@brief   Abstract pricer base class.
   @ingroup Algo

   Class SPxPricer is a pure virtual class defining the interface for pricer
   classes to be used by SoPlex. The pricer's task is to select a vector to
   enter or leave the simplex basis, depending on the chosen simplex type.

   An SPxPricer first #load%s the SoPlex object for which pricing is to
   be performed. Then, depending of the SPxSolverBase<R>::Type, methods
   #selectEnter() and #entered4() (for entering Simplex) or #selectLeave()
   and #left4() (for leaving Simplex) are called by SoPlex. The SPxPricer
   object is informed of a change of the SPxSolverBase<R>::Type by calling method
   #setType().
*/
template <class R>
class SPxPricer
{
protected:

   //-------------------------------------
   /**@name Data */
   ///@{
   /// name of the pricer
   const char* m_name;
   /// the solver

   SPxSolverBase<R>*
   thesolver; //@todo The template type should be identified? Do I have to defined two of them?
   /// violation bound
   R        theeps;
   ///@}


   struct IdxElement
   {
      int idx;
      R val;
   };

   /// Compare class to sort idx/val pairs, used for hypersparse pricing leaving
   struct IdxCompare
   {
   public:
      /// constructor
      IdxCompare()
         : elements(0)
      {}

      const IdxElement*  elements;

      R operator()(
         IdxElement      a,
         IdxElement      b
      ) const
      {
         //the first case is needed to handle inf-values
         return (a.val == b.val) ? 0 : b.val - a.val;
      }
   };

   IdxCompare compare;

public:

   // violation types used for (hyper) sparse pricing
   enum ViolationType
   {
      NOT_VIOLATED         = 0,
      VIOLATED             = 1,
      VIOLATED_AND_CHECKED = 2
   };

   //-------------------------------------
   /**@name Initialization */
   ///@{
   /// get name of pricer.
   virtual const char* getName() const
   {
      return m_name;
   }

   /// loads LP.
   /** Loads the solver and LP for which pricing steps are to be performed.
    */
   virtual void load(SPxSolverBase<R>* p_solver)
   {
      thesolver = p_solver;
   }

   /// unloads LP.
   virtual void clear()
   {
      thesolver = 0;
   }

   /// returns loaded SPxSolverBase object.
   virtual SPxSolverBase<R>* solver() const
   {
      return thesolver;
   }

   /// returns violation bound \ref soplex::SPxPricer::theeps "theeps".
   virtual R epsilon() const
   {
      return theeps;
   }

   /// sets violation bound.
   /** Inequality violations are accepted, if their size is less than \p eps.
    */
   virtual void setEpsilon(R eps)
   {
      assert(eps >= 0.0);

      theeps = eps;
   }

   /// sets pricing type.
   /** Informs pricer about (a change of) the loaded SoPlex's Type. In
       the sequel, only the corresponding select methods may be called.
   */
   virtual void setType(typename SPxSolverBase<R>::Type)
   {
      this->thesolver->weights.reDim(0);
      this->thesolver->coWeights.reDim(0);
      this->thesolver->weightsAreSetup = false;
   }

   /// sets basis representation.
   /** Informs pricer about (a change of) the loaded SoPlex's
       Representation.
   */
   virtual void setRep(typename SPxSolverBase<R>::Representation)
   {}
   ///@}

   //-------------------------------------
   /**@name Pivoting */
   ///@{
   /// returns selected index to leave basis.
   /** Selects the index of a vector to leave the basis. The selected index
       i, say, must be in the range 0 <= i < solver()->dim() and its
       tested value must fullfill solver()->test()[i] < -#epsilon().
   */
   virtual int selectLeave() = 0;

   /// performs leaving pivot.
   /** Method #left4() is called after each simplex iteration in LEAVE
       mode. It informs the SPxPricer that the \p n 'th variable has left
       the basis for \p id to come in at this position. When being called,
       all vectors of SoPlex involved in such an entering update are
       setup correctly and may be accessed via the corresponding methods
       (\ref SPxSolverBase<R>::fVec() "fVec()", \ref SPxSolverBase<R>::pVec() "pVec()",
       etc.). In general, argument \p n will be the one returned by the
       SPxPricer at the previous call to #selectLeave(). However, one can not
       rely on this.
   */
   virtual void left4(int /*n*/, SPxId /*id*/) {}

   /// selects Id to enter basis.
   /** Selects the SPxId of a vector to enter the basis. The selected
       id, must not represent a basic index (i.e. solver()->isBasic(id) must
       be false). However, the corresponding test value needs not to be less
       than -#epsilon(). If not, SoPlex will discard the pivot.

       Note:
       When method #selectEnter() is called by the loaded SoPlex
       object, all values from \ref SPxSolverBase<R>::coTest() "coTest()" are
       up to date. However, whether the elements of
       \ref SPxSolverBase<R>::test() "test()" are up to date depends on the
       SPxSolverBase<R>::Pricing type.
   */
   virtual SPxId selectEnter() = 0;

   /// performs entering pivot.
   /** Method #entered4() is called after each simplex iteration in ENTER
       mode. It informs the SPxPricer that variable \p id has entered
       at the \p n 'th position. When being called, all vectors of SoPlex
       involved in such an entering update are setup correctly and may be
       accessed via the corresponding methods
       (\ref SPxSolverBase<R>::fVec() "fVec()", \ref SPxSolverBase<R>::pVec() "pVec()",
       etc.). In general, argument \p id will be the one returned by the
       SPxPricer at the previous call to #selectEnter(). However, one can not
       rely on this.
   */
   virtual void entered4(SPxId /*id*/, int /*n*/)
   {}
   ///@}


   //-------------------------------------
   /**@name Extension */
   ///@{
   /// \p n vectors have been added to loaded LP.
   virtual void addedVecs(int /*n*/)
   {}
   /// \p n covectors have been added to loaded LP.
   virtual void addedCoVecs(int /*n*/)
   {}
   ///@}

   //-------------------------------------
   /**@name Shrinking */
   ///@{
   /// vector \p i was removed from loaded LP.
   virtual void removedVec(int /*i*/)
   {}
   /// vectors given by \p perm have been removed from loaded LP.
   virtual void removedVecs(const int* /*perm*/)
   {}
   /// covector \p i was removed from loaded LP.
   virtual void removedCoVec(int /*i*/)
   {}
   /// covectors given by \p perm have been removed from loaded LP.
   virtual void removedCoVecs(const int* /*perm*/)
   {}
   ///@}

   //-------------------------------------
   /**@name Debugging */
   ///@{
   virtual bool isConsistent() const
   {
#ifdef ENABLE_CONSISTENCY_CHECKS
      return thesolver != 0;
#else
      return true;
#endif
   }
   ///@}

   //-------------------------------------
   /**@name Constructors / Destructors */
   ///@{
   /// constructor
   explicit SPxPricer(const char* p_name)
      : m_name(p_name)
      , thesolver(0)
      , theeps(0.0)
   {}

   /// copy constructor
   SPxPricer(const SPxPricer& old)
      : m_name(old.m_name)
      , thesolver(old.thesolver)
      , theeps(old.theeps)
   {}

   /// assignment operator
   SPxPricer& operator=(const SPxPricer& rhs)
   {
      if(this != &rhs)
      {
         m_name = rhs.m_name;
         thesolver = rhs.thesolver;
         theeps = rhs.theeps;
         assert(isConsistent());
      }

      return *this;
   }

   /// destructor.
   virtual ~SPxPricer()
   {
      m_name    = 0;
      thesolver = 0;
   }

   /// clone function for polymorphism
   virtual SPxPricer* clone()  const  = 0;
   ///@}

};


} // namespace soplex
#endif // _SPXPRICER_H_
