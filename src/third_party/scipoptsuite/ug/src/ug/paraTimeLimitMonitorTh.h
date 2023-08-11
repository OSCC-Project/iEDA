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

/**@file    paraTimeLimitMonitorTh.h
 * @brief   Time limit monitor thread class.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_TIME_LIMIT_MONITOR_TH_H__
#define __PARA_TIME_LIMIT_MONITOR_TH_H__

#include <algorithm>
#include "paraDef.h"
#ifdef _COMM_PTH
#include "paraCommPth.h"
#endif
#ifdef _COMM_CPP11
#include "paraCommCPP11.h"
#endif

namespace UG
{

///
/// class ParaTimeLimitMonitorTh
///
class ParaTimeLimitMonitorTh
{

protected:

#ifdef _COMM_PTH
   ParaCommPth    *paraComm;                ///< ParaCommunicator used
#endif

#ifdef _COMM_CPP11
   ParaCommCPP11  *paraComm;                ///< ParaCommunicator used
#endif

   double       hardTimeLimit;              ///< hard time limit

public:

   ///
   /// default constructor
   ///
   ParaTimeLimitMonitorTh(
         )
         : paraComm(0),
           hardTimeLimit(0.0)
   {
      THROW_LOGICAL_ERROR1("Default constructor of ParaTimeLimitMonitor is called");
   }


   ///
   /// constructor
   ///
   ParaTimeLimitMonitorTh(
#ifdef _COMM_PTH
       ParaCommPth *comm,      ///< communicator used
#endif
#ifdef _COMM_CPP11
       ParaCommCPP11 *comm,    ///< communicator used
#endif
       double timelimit
       ) : paraComm(comm)
   {
      // hardTimeLimit = timelimit + std::min(60.0, 0.1 * timelimit);   // set hard time limit + 60 seconds longer than time limit
      hardTimeLimit = timelimit + 3;   // set hard time limit + 60 seconds longer than time limit
   }

   ///
   /// destructor
   ///
   virtual ~ParaTimeLimitMonitorTh(
         )
   {
   }

   ///
   /// run this time limit monitor
   ///
   void run(
         )
   {
#ifdef _MSC_VER
     _sleep(static_cast<unsigned int>(hardTimeLimit));
#else
      sleep(static_cast<unsigned int>(hardTimeLimit));
#endif
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, 0, TagHardTimeLimit)
      );
      std::cout << "****** send TagHardTimeLimit message *****" << std::endl;
   }

};

}

#endif // __PARA_TIME_LIMIT_MONITOR_TH_H__

