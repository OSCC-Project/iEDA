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

/**@file  spxdefaultrt.h
 * @brief Textbook ratio test for SoPlex.
 */
#ifndef _SPXDEFAULTRT_H_
#define _SPXDEFAULTRT_H_


#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxratiotester.h"

namespace soplex
{

/**@brief   Textbook ratio test for SoPlex.
   @ingroup Algo

   Class SPxDefaultRT provides an implementation of the textbook ratio test
   as a derived class of SPxRatioTester. This class is not intended for
   reliably solving LPs (even though it does the job for ``numerically simple''
   LPs). Instead, it should serve as a demonstration of how to write ratio
   tester classes.

   See SPxRatioTester for a class documentation.
*/
template <class R>
class SPxDefaultRT : public SPxRatioTester<R>
{
public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxDefaultRT()
      : SPxRatioTester<R>("Default")
   {}
   /// copy constructor
   SPxDefaultRT(const SPxDefaultRT& old)
      : SPxRatioTester<R>(old)
   {}
   /// assignment operator
   SPxDefaultRT& operator=(const SPxDefaultRT& rhs)
   {
      if(this != &rhs)
      {
         SPxRatioTester<R>::operator=(rhs);
      }

      return *this;
   }
   /// destructor
   virtual ~SPxDefaultRT()
   {}
   /// clone function for polymorphism
   inline virtual SPxRatioTester<R>* clone() const
   {
      return new SPxDefaultRT(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Select enter/leave */
   ///@{
   ///
   virtual int selectLeave(R& val, R, bool);
   ///
   virtual SPxId selectEnter(R& val, int, bool);
};

} // namespace soplex

#include "spxdefaultrt.hpp"

#endif // _SPXDEFAULTRT_H_
