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

/**@file  usertimer.h
 * @brief UserTimer class.
 */

#ifndef _USER_TIMER_H_
#define _USER_TIMER_H_

#include "soplex/spxdefines.h"
#include "soplex/timer.h"

namespace soplex
{

class UserTimer : public Timer
{
private:

   //------------------------------------
   /**@name number of ticks per second */
   ///@{
   static const long ticks_per_sec;  ///< ticks per secound, should be constant
   ///@}

   //------------------------------------
   /**@name Data */
   ///@{
   mutable clock_t uAccount;      ///< user time
   mutable clock_t uTicks;        ///< user ticks

   mutable Real lasttime;
   ///@}

   //------------------------------------
   /**@name Internal helpers */
   ///@{
   /// convert ticks to secounds.
   Real ticks2sec(clock_t ticks) const
   {
      return (Real(ticks) * 1000.0 / Real(ticks_per_sec)) / 1000.0;
   }

   /// get actual user ticks from the system.
   void updateTicks() const;

   ///@}

public:

   //------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   UserTimer()
      : Timer(), uAccount(0), uTicks(0), lasttime(0.0)
   {
      assert(ticks_per_sec > 0);
   }
   /// copy constructor
   UserTimer(const UserTimer& old)
      : Timer(), uAccount(old.uAccount), uTicks(old.uTicks), lasttime(old.lasttime)
   {
      assert(ticks_per_sec > 0);
   }
   /// assignment operator
   UserTimer& operator=(const UserTimer& old)
   {
      assert(ticks_per_sec > 0);
      uAccount = old.uAccount;
      uTicks = old.uTicks;
      lasttime = old.lasttime;
      return *this;
   }

   virtual ~UserTimer()
   {}
   ///@}

   //------------------------------------
   /**@name Control */
   ///@{
   /// initialize timer, set timing accounts to zero.
   virtual void reset()
   {
      status   = RESET;
      uAccount = 0;
      lasttime = 0.0;
   }

   /// start timer, resume accounting user, system and real time.
   virtual void start();

   /// stop timer, return accounted user time.
   virtual Real stop();

   /// return type of timer
   virtual TYPE type()
   {
      return USER_TIME;
   }
   ///@}

   //------------------------------------
   /**@name Access */
   ///@{
   virtual Real time() const;

   virtual Real lastTime() const;
   ///@}
};
} // namespace soplex
#endif // _USER_TIMER_H_
