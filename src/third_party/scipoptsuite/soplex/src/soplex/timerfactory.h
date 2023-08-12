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

/**@file  timerfactory.h
 * @brief TimerFactory class.
 */

#ifndef _TIMERFACTORY_H_
#define _TIMERFACTORY_H_

#include "soplex/spxdefines.h"
#include "soplex/spxalloc.h"
#include "soplex/timer.h"
#include "soplex/notimer.h"
#include "soplex/usertimer.h"
#include "soplex/wallclocktimer.h"

namespace soplex
{
/**@class   TimerFactory
   @ingroup Elementary

   @brief Class to create new timers and to switch types of exiting ones
   */

class TimerFactory
{

public:

   /// create timers and allocate memory for them
   static Timer* createTimer(Timer::TYPE ttype)
   {
      Timer* timer = 0;

      switch(ttype)
      {
      case Timer::OFF:
         spx_alloc(timer, sizeof(NoTimer));
         timer = new(timer) NoTimer();
         break;

      case Timer::USER_TIME:
         spx_alloc(timer, sizeof(UserTimer));
         timer = new(timer) UserTimer();
         break;

      case Timer::WALLCLOCK_TIME:
         spx_alloc(timer, sizeof(WallclockTimer));
         timer = new(timer) WallclockTimer();
         break;

      default:
         MSG_ERROR(std::cerr << "wrong timer specified" << std::endl;)
      }

      return timer;
   }

   static Timer* switchTimer(Timer* timer, Timer::TYPE ttype)
   {
      // check whether the type is different from the current one
      if(ttype != timer->type())
      {
         // @todo transfer the old times
         spx_free(timer);
         timer = createTimer(ttype);
      }

      return timer;
   }

};
} // namespace soplex
#endif // _TIMERFACTORY_H_
