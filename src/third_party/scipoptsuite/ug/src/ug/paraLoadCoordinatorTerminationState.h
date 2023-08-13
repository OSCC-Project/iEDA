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

/**@file    paraLoadCoordinatorTerminationState.h
 * @brief   Load coordinator termination state.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_LOADCOORDINATOR_TERMINATION_STATE_H__
#define __PARA_LOADCOORDINATOR_TERMINATION_STATE_H__

#include <string>
#include <cfloat>
#include "paraComm.h"

#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif

namespace UG
{

///
/// Class for LoadCoordinator termination state
/// which contains calculation state in a ParaLoadCoordinator
///
class ParaLoadCoordinatorTerminationState
{
public:

   bool                  isCheckpointState;                   ///< indicate if this state is at checkpoint or not
   int                   rank;                                ///< rank of this ParaLoadCoordinator
   ///
   /// Counters related to this ParaLoadCoordinator
   /// TODO: The numbers should be classified depending on solvers
   ///
   unsigned long long    nWarmStart;                          ///< number of warm starts (restarts)
   unsigned long long    nSent;                               ///< number of ParaTasks sent from LC
   unsigned long long    nReceived;                           ///< number of ParaTasks received from Solvers
   ///
   ///  times of this LoadCoordinator
   ///
   double                idleTime;                            ///< idle time of this LoadCoordinator
   double                runningTime;                         ///< this ParaLoadCoordinator running time

   ///
   /// default constructor
   ///
   ParaLoadCoordinatorTerminationState(
         )
         : isCheckpointState(true),
           rank(0),
           nWarmStart(0),
           nSent(0),
           nReceived(0),
           idleTime(0.0),
           runningTime(0.0)
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaLoadCoordinatorTerminationState(
	        )
   {
   }

   ///
   /// stringfy ParaCalculationState
   /// @return string to show inside of this object
   ///
   virtual std::string toString(
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// write to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out              ///< gzstream for output
         ) = 0;

   ///
   /// read from checkpoint file
   ///
   virtual bool read(
         ParaComm *comm,                       ///< communicator used
         gzstream::igzstream &in               ///< gzstream for input
         ) = 0;

#endif

};

}

#endif // __PARA_LOADCOORDINATOR_TERMINATION_STATE_H__

