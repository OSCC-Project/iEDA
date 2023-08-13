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


/**@file  spxstarter.h
 * @brief SoPlex start basis generation base class.
 */
#ifndef _SPXDSTARTER_H_
#define _SPXDSTARTER_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxsolver.h"

namespace soplex
{

/**@brief   SoPlex start basis generation base class.
   @ingroup Algo

   SPxStarter is the virtual base class for classes generating a starter basis
   for the Simplex solver SoPlex. When a SPxStarter object has been loaded
   to a SoPlex solver, the latter will call method #generate() in order to
   have a start basis generated. Implementations of method #generate() must
   terminate by \ref soplex::SPxSolver::load() "loading" the generated basis to
   SoPlex. Loaded bases must be nonsingular.
*/
template <class R>
class SPxStarter
{
protected:

   //-------------------------------------
   /**@name Data */
   ///@{
   /// name of the starter
   const char* m_name;
   ///@}

public:

   //-------------------------------------
   /**@name Data */
   ///@{
   /// constructor
   explicit SPxStarter(const char* name)
      : m_name(name)
   {}
   /// copy constructor
   SPxStarter(const SPxStarter& old)
      : m_name(old.m_name)
   {}
   /// assignment operator
   SPxStarter& operator=(const SPxStarter& rhs)
   {
      if(this != &rhs)
      {
         m_name = rhs.m_name;
      }

      return *this;
   }
   /// destructor.
   virtual ~SPxStarter()
   {
      m_name = 0;
   }
   /// clone function for polymorphism
   virtual SPxStarter* clone()const = 0;
   ///@}

   //-------------------------------------
   /**@name Access */
   ///@{
   /// get name of starter.
   virtual const char* getName() const
   {
      return m_name;
   }
   ///@}

   //-------------------------------------
   /**@name Starting */
   ///@{
   /// generates start basis for loaded basis.
   virtual void generate(SPxSolverBase<R>& base) = 0;
   ///@}

   //-------------------------------------
   /**@name Misc */
   ///@{
   /// checks consistency.
   virtual bool isConsistent() const;
   ///@}

private:

   //------------------------------------
   /**@name Blocked */
   ///@{
   /// we have no default constructor.
   SPxStarter();
   ///@}

};
} // namespace soplex

// For general templated functions
#include "spxstarter.hpp"

#endif // _SPXDSTARTER_H_
