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

/**@file  notimer.h
 * @brief NoTimer class.
 */

#ifndef _NO_TIMER_H_
#define _NO_TIMER_H_

#include "soplex/spxdefines.h"
#include "soplex/timer.h"

namespace soplex
{

class NoTimer : public Timer
{

public:

   //------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   NoTimer()
      : Timer()
   {}
   /// copy constructor
   NoTimer(const NoTimer&)
      : Timer()
   {}
   /// assignment operator
   NoTimer& operator=(const NoTimer&)
   {
      return *this;
   }

   virtual ~NoTimer()
   {}
   ///@}

   //------------------------------------
   /**@name Control */
   ///@{
   /// initialize timer
   virtual void reset()
   {}

   /// start timer
   virtual void start()
   {}

   /// stop timer
   virtual Real stop()
   {
      return 0.0;
   }

   /// return type of timer
   virtual TYPE type()
   {
      return OFF;
   }
   ///@}

   //------------------------------------
   /**@name Access */
   ///@{
   virtual Real time() const
   {
      return 0.0;
   }

   virtual Real lastTime() const
   {
      return 0.0;
   }

   ///@}
};
} // namespace soplex
#endif // _NO_TIMER_H_
