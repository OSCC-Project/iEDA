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

/**@file    paraSysTimer.cpp
 * @brief   System timer.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <iostream>
#include "paraSysTimer.h"
using namespace UG;

void
ParaSysTimer::start(
){
#ifdef BSD
   if (gettimeofday(&stTvTimeStart, (struct timezone *)0) < 0) {
      std::cerr << "gettimeofday() error in ParaSysTimer::start" << std::endl;
      exit(1);
   }
   if (getrusage(RUSAGE_SELF, &stRuStart) < 0) {
      std::cerr << "getrusage() error in ParaSysTimer::start" << std::endl;
      exit(1);
   }
#endif /* BSD */

#ifdef _MSC_VER
   _ftime(&timebStart);
   if( ! GetProcessTimes(	
		 GetCurrentProcess(),   // specifies the process of interest
         &ftCreationTime,	// when the process was created
         &ftExitTime,	// when the process exited 
         &ftKernelTimeStart,	// time the process has spent in kernel mode 
         &ftUserTimeStart 	// time the process has spent in user mode 
         ) ){
	  LPVOID lpMsgBuf;
 
      FormatMessage( 
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
      NULL,
      GetLastError(),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
      (LPTSTR) &lpMsgBuf,
      0,
      NULL 
      );

      std::cerr << "GetProcessTimes() error in ParaSysTimer::start errcode = "
		   <<  lpMsgBuf << std::endl;
      exit(1);
   }

#endif /* _MSC_VER */

#ifdef SYSV
   if ((lTimeStart = times(&stTmsStart)) == -1) {
      std::cerr << "times() error inin ParaSysTimer::start" << std::endl;
      exit(1);
   }
#endif /* SYSV */


   return;
}

void
ParaSysTimer::stop(
){
#ifdef BSD
   if (getrusage(RUSAGE_SELF, &stRuStop) < 0) {
      std::cerr << "getrusage() error in ParaSysTimer::stop" << std::endl;
      exit(1);
   }
   if (gettimeofday(&stTvTimeStop, (struct timezone *) 0) < 0) {
      std::cerr << "gettimeofday() error in ParaSysTimer::stop" << std::endl;
      exit(1);
   }
#endif /* BSD */

#ifdef _MSC_VER
   if( ! GetProcessTimes(	
		 GetCurrentProcess(),   // specifies the process of interest
         &ftCreationTime,	// when the process was created
         &ftExitTime,	// when the process exited 
         &ftKernelTimeStop,	// time the process has spent in kernel mode 
         &ftUserTimeStop 	// time the process has spent in user mode 
         ) ){
      std::cerr << "GetProcessTimes() error in ParaSysTimer::start" << std::endl;
      exit(1);
   }
   _ftime(&timebStop);
#endif /* _MSC_VER */

#ifdef SYSV
   if ((lTimeStop = times(&stTmsStop)) == -1) {
      std::cerr << "times() error in ParaSysTimer::stop" << std::endl;
      exit(1);
   }
#endif /* SYSV */

   return;
}

double
ParaSysTimer::getStartTime(
      )
{
#ifdef BSD

   double dStart = ((double) stTvTimeStart.tv_sec) * 1000000.0
                   + stTvTimeStart.tv_usec;
   double dSeconds = dStart/1000000.0;
#endif /* BSD */

#ifdef _MSC_VER
   double dSeconds;
   dSeconds = timebStart.time
           + (double)(timebStart.millitm)/1000;
#endif /* _MSC_VER */

#ifdef SYSV

   double dSeconds = (double)lTimeStart/(double)TICKS;
#endif /* SYSV */

   return dSeconds;
}

double
ParaSysTimer::getRTimeInterval(
){
#ifdef BSD
   struct timeval   stTempTvTimeStop;
   struct rusage    stTempRuStop;

   if (getrusage(RUSAGE_SELF, &stTempRuStop) < 0) {
      std::cerr << "getrusage() error in ParaSysTimer::getRTimeInterval" << std::endl;
      exit(1);
   }
   if (gettimeofday(&stTempTvTimeStop, (struct timezone *) 0) < 0) {
      std::cerr << "gettimeofday() error in ParaSysTimer::getRTimeInterval" << std::endl;
      exit(1);
   }

   double dStart = ((double) stTvTimeStart.tv_sec) * 1000000.0
                   + stTvTimeStart.tv_usec;
   double dStop = ((double) stTempTvTimeStop.tv_sec) * 1000000.0
                   + stTempTvTimeStop.tv_usec;
   double dSeconds = (dStop - dStart)/1000000.0;
#endif /* BSD */

#ifdef _MSC_VER
   struct _timeb timebTempStop;   

   FILETIME ftTempCreationTime, ftTempExitTime,
            ftTempKernelTimeStop,  ftTempUserTimeStop;

   if( ! GetProcessTimes(
                 GetCurrentProcess(),   // specifies the process of interest
         &ftTempCreationTime,       // when the process was created
         &ftTempExitTime,   // when the process exited
         &ftTempKernelTimeStop,     // time the process has spent in kernel mode
         &ftTempUserTimeStop        // time the process has spent in user mode
         ) ){
      std::cerr << "GetProcessTimes() error in ParaSysTimer::start" << std::endl;
      exit(1);
   }
   _ftime(&timebTempStop);

   double dSeconds;
   if ( timebTempStop.millitm - timebStart.millitm >= 0 ){
	   dSeconds = (timebTempStop.time - timebStart.time)
		           + (double)(timebTempStop.millitm - timebStart.millitm)/1000;
   } else {
	   dSeconds = (timebTempStop.time - timebStart.time - 1)
		           + (double)(timebTempStop.millitm + 1000
				              - timebStart.millitm)/1000;
   }
#endif /* _MSC_VER */

#ifdef SYSV
   long lTempTimeStop;
   struct tms stTempTmsStop;

   if ((lTempTimeStop = times(&stTempTmsStop)) == -1) {
      std::cerr << "times() error in ParaSysTimer::stop" << std::endl;
      exit(1);
   }

   double dSeconds = (double)(lTempTimeStop - lTimeStart)/(double)TICKS;
#endif /* SYSV */

   return dSeconds;
}

double
ParaSysTimer::getRTime(
){
#ifdef BSD
   double dStart = ((double) stTvTimeStart.tv_sec) * 1000000.0
                   + stTvTimeStart.tv_usec;
   double dStop = ((double) stTvTimeStop.tv_sec) * 1000000.0
                   + stTvTimeStop.tv_usec;
   double dSeconds = (dStop - dStart)/1000000.0;
#endif /* BSD */

#ifdef _MSC_VER
   double dSeconds;
   if ( timebStop.millitm - timebStart.millitm >= 0 ){
	   dSeconds = (timebStop.time - timebStart.time)
		           + (double)(timebStop.millitm - timebStart.millitm)/1000;
   } else {
	   dSeconds = (timebStop.time - timebStart.time - 1)
		           + (double)(timebStop.millitm + 1000
				              - timebStart.millitm)/1000;
   }
#endif /* _MSC_VER */

#ifdef SYSV
   double dSeconds = (double)(lTimeStop - lTimeStart)/(double)TICKS;
#endif /* SYSV */

   return dSeconds;
}
double
ParaSysTimer::getUTime(
){
#ifdef BSD
   double dStart = ((double) stRuStart.ru_utime.tv_sec) * 1000000.0
                   + stRuStart.ru_utime.tv_usec;
   double dStop = ((double) stRuStop.ru_utime.tv_sec) * 1000000.0
                   + stRuStop.ru_utime.tv_usec;
   double dSeconds = (dStop - dStart)/1000000.0;
#endif /* BSD */

#ifdef _MSC_VER
   double dSeconds;
   __int64  i64Start, i64Stop;
   i64Start = ftUserTimeStart.dwHighDateTime;
   i64Start <<= 32;
   i64Start |= ftUserTimeStart.dwLowDateTime;
   i64Stop = ftUserTimeStop.dwHighDateTime;
   i64Stop <<= 32;
   i64Stop |= ftUserTimeStop.dwLowDateTime;

   dSeconds = (double)( (i64Stop - i64Start) ) / 10000000.0;
#endif /* _MSC_VER */

#ifdef SYSV
   double dSeconds = (double)(stTmsStop.tms_utime - stTmsStart.tms_utime)/
                     (double)TICKS;
#endif /* SYSV */

   return dSeconds;
}

double
ParaSysTimer::getSTime(
){
#ifdef BSD
   double dStart = ((double) stRuStart.ru_stime.tv_sec) * 1000000.0
                   + stRuStart.ru_stime.tv_usec;
   double dStop = ((double)stRuStop.ru_stime.tv_sec) * 1000000.0
                   + stRuStop.ru_stime.tv_usec;
   double dSeconds = (dStop - dStart)/1000000.0;
#endif /* BSD */

#ifdef _MSC_VER
   double dSeconds;
   __int64  i64Start, i64Stop;
   i64Start = ftKernelTimeStart.dwHighDateTime;
   i64Start <<= 32;
   i64Start |= ftKernelTimeStart.dwLowDateTime;
   i64Stop = ftKernelTimeStop.dwHighDateTime;
   i64Stop <<= 32;
   i64Stop |= ftKernelTimeStop.dwLowDateTime;

   dSeconds = (double)( (i64Stop - i64Start) ) / 10000000.0;
#endif /* _MSC_VER */

#ifdef SYSV
   double dSeconds = (double)(stTmsStop.tms_stime - stTmsStart.tms_stime)/
                     (double)TICKS;
#endif /* SYSV */
   return dSeconds;
}
