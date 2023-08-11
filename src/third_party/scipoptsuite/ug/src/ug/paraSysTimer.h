/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file    paraSysTimer.h
 * @brief   System timer.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_SYS_TIMER_H__
#define __PARA_SYS_TIMER_H__

//-------------------
// include files
//-------------------
#include <cstdlib>
#include "paraTimer.h"

#ifdef __APPLE__
#define BSD
#endif /* Mac OSX */

#ifdef __sun
#define BSD
#endif /* SUN_OS */

#ifdef SUN_OS
#define BSD
#endif /* SUN_OS */

#ifdef SOLARIS
#define SYSV
#endif /* SOLARIS */

#ifdef linux
#define SYSV
#endif /* linux */

#ifdef __linux__
#define SYSV
#endif /* linux */

#ifdef __CYGWIN__ 
#define SYSV
#endif /* linux */

#ifdef BlueGene
#define BSD
#endif

#if !(defined _MSC_VER || defined SYSV || defined BSD )
#error cannot detect timer type!
#endif

#ifdef BSD
#include <sys/time.h>
#include <sys/resource.h>
#endif /* BSD */

#ifdef _MSC_VER
#include <sys/types.h>
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#endif /* _MSC_VER */

#ifdef SYSV
#include <sys/types.h>
#include <sys/times.h>
#include <sys/param.h>
#define TICKS HZ
#endif /* SYSV */

namespace UG
{

///
/// Class ParaSysTimer
///
class ParaSysTimer : public ParaTimer {

public:

   ///
   /// default constructor
   ///
   ParaSysTimer(
         )
   {
   }

   ///
   /// destructor
   ///
   ~ParaSysTimer(
         )
   {
   }

   ///
   /// initialize timer
   ///
   void init(
         ParaComm* paraComm     ///< communicator used
         )
   {
      start();
   }

   ///
   /// get elapsed time
   /// @return elapsed time
   ///
   double getElapsedTime(
         )
   {
      return getRTimeInterval();
   }

   ///
   /// start timer
   ///
   void    start(
         void
         );

   ///
   /// stop timer
   ///
   void    stop(
         void
         );

   ///
   /// get start time
   /// @return start time
   ///
   double  getStartTime(
         void
         );

   ///
   /// get elapsed time from start time
   /// @return elapsed time
   ///
   double  getRTimeInterval(
         void
         );

   ///
   /// get real time between start timne and stop time
   /// @return real time
   ///
   double  getRTime(
         void
         );

   ///
   /// get user time between start timne and stop time
   /// @return user time
   ///
   double  getUTime(
         void
         );

   ///
   /// get system time between start timne and stop time
   /// @return system time
   ///
   double  getSTime(
         void
         );

private:

#ifdef BSD
   struct timeval   stTvTimeStart, stTvTimeStop;
   struct rusage    stRuStart, stRuStop;
#  endif /* BSD */

#ifdef _MSC_VER
   struct _timeb timebStart, timebStop;
   FILETIME ftCreationTime, ftExitTime,
	        ftKernelTimeStart, ftUserTimeStart,
			ftKernelTimeStop,  ftUserTimeStop;
   HANDLE   hCurrentProcess;
#  endif /* _MSC_VER */

#  ifdef SYSV
   long lTimeStart, lTimeStop;
   struct tms stTmsStart, stTmsStop;
/*   long times(); */
#  endif /* SYSV */

};

}

#endif  // __PARA_SYS_TIMER_H__
