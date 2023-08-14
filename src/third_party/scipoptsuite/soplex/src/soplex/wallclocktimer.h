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

/**@file  wallclocktimer.h
 * @brief WallclockTimer class.
 */

#ifndef _WALLCLOCK_TIMER_H_
#define _WALLCLOCK_TIMER_H_

#include "soplex/spxdefines.h"
#include "soplex/timer.h"

namespace soplex
{

class WallclockTimer : public Timer
{
private:

   //------------------------------------
   /**@name Data */
   ///@{
   mutable time_t sec;           ///< seconds
   mutable time_t usec;          ///< microseconds

   mutable Real lasttime;
   ///@}

   //------------------------------------
   /**@name Internal helpers */
   ///@{
   /// convert wallclock time to secounds.
   Real wall2sec(time_t s, time_t us) const
   {
      return (Real)s + 0.000001 * (Real)us;
   }

   ///@}

public:

   //------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   WallclockTimer()
      : Timer(), sec(0), usec(0), lasttime(0.0)
   {}
   /// copy constructor
   WallclockTimer(const WallclockTimer& old)
      : Timer(), sec(old.sec), usec(old.usec), lasttime(old.lasttime)
   {}
   /// assignment operator
   WallclockTimer& operator=(const WallclockTimer& old)
   {
      sec = old.sec;
      usec = old.usec;
      lasttime = old.lasttime;
      return *this;
   }

   virtual ~WallclockTimer()
   {}
   ///@}

   //------------------------------------
   /**@name Control */
   ///@{
   /// initialize timer, set timing accounts to zero.
   virtual void reset()
   {
      status   = RESET;
      sec = usec = 0;
      lasttime = 0.0;
   }

   /// start timer, resume accounting user, system and real time.
   virtual void start();

   /// stop timer, return accounted user time.
   virtual Real stop();

   /// return type of timer
   virtual TYPE type()
   {
      return WALLCLOCK_TIME;
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
#endif // _WALLCLOCK_TIMER_H_
