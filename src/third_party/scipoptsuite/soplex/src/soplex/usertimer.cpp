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

#include <assert.h>

#if defined(_WIN32) || defined(_WIN64)

#include <time.h>

#else   // !(_WIN32 || _WIN64)

#include <sys/types.h>
#include <sys/times.h>
//#include <sys/param.h>
#include <unistd.h>

#endif  // !(_WIN32 || _WIN64)

#include "soplex/spxdefines.h"
#include "soplex/usertimer.h"

namespace soplex
{
/* determine TIMES_TICKS_PER_SEC for clock ticks delivered by times().
 * (don't use CLOCKS_PER_SEC since this is related to clock() only).
 */
#if defined(CLK_TCK)
#define TIMES_TICKS_PER_SEC CLK_TCK
#elif defined(_SC_CLK_TCK)
#define TIMES_TICKS_PER_SEC sysconf(_SC_CLK_TCK)
#elif defined(HZ)
#define TIMES_TICKS_PER_SEC HZ
#else // !CLK_TCK && !_SC_CLK_TCK && !HZ
#define TIMES_TICKS_PER_SEC 60
#endif // !CLK_TCK && !_SC_CLK_TCK && !HZ

const long UserTimer::ticks_per_sec = long(TIMES_TICKS_PER_SEC);

// get actual user, system and real time from system
void UserTimer::updateTicks() const
{
#if defined(_WIN32) || defined(_WIN64)

   uTicks = clock();

#else   /* !(_WIN32 || _WIN64) */

   struct tms now;
   clock_t    ret = times(&now);

   if(int(ret) == -1)
      now.tms_utime = now.tms_stime = ret = 0;

   uTicks = now.tms_utime;

#endif  /* !(_WIN32 || _WIN64) */
}

// start timer, resume accounting user, system and real time.
void UserTimer::start()
{
   // ignore start request if timer is running
   if(status != RUNNING)
   {
      updateTicks();

      uAccount -= uTicks;
      status    = RUNNING;
   }

   lasttime = 0;
}

// stop timer, return accounted user time.
Real UserTimer::stop()
{
   // status remains unchanged if timer is not running
   if(status == RUNNING)
   {
      updateTicks();

      uAccount += uTicks;
      status    = STOPPED;
   }

   return ticks2sec(uAccount);
}

// get accounted user time.
Real UserTimer::time() const
{
   if(status == RUNNING)
   {
      updateTicks();
      lasttime = ticks2sec(uTicks + uAccount);
   }
   else
   {
      lasttime = ticks2sec(uAccount);
   }

   return lasttime;
}

Real UserTimer::lastTime() const
{
   return lasttime;
}

} // namespace soplex
