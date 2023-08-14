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

/**@file    paraTimerMpi.h
 * @brief   ParaTimer extension for MPI timer.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_TIMER_MPI_H__
#define __PARA_TIMER_MPI_H__
#include <mpi.h>
#include "paraComm.h"
#include "paraTimer.h"

namespace UG
{

///
/// class ParaTimerMpi
/// (Timer used in MPI communication)
///
class ParaTimerMpi : public ParaTimer
{

   double startTime;      ///< satrt time

public:

   ///
   /// constructor
   ///
   ParaTimerMpi(
         )
   {
      startTime = MPI_Wtime();
   }

   ///
   /// destructor
   ///
   ~ParaTimerMpi(
         )
   {
   }

   ///
   /// Initialize timer
   ///
   void init(
         ParaComm *comm      ///< communicator used
         );

   ///
   /// get elapsed time
   /// @return elapsed time
   ///
   double getElapsedTime(
         )
   {
      return ( MPI_Wtime() - startTime + offset );
   }

};

}

#endif // __PARA_TIMER_MPI_H__
