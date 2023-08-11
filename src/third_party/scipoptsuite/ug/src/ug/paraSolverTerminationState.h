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

/**@file    paraSolverTerminationState.h
 * @brief   This class contains solver termination state which is transferred form Solver to LC.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_SOLVER_TERMINATION_STATE_H__
#define __PARA_SOLVER_TERMINATION_STATE_H__

#include "paraComm.h"
#include "paraInitiator.h"
#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif

namespace UG
{

///
/// class ParaSolverTerminationState
/// (Solver termination state in a ParaSolver)
///
class ParaSolverTerminationState
{
protected:

   int          interrupted;                         ///< indicate that this solver is interrupted or not.
                                                     ///< 0: not interrupted,
                                                     ///< 1: interrupted,
                                                     ///< 2: checkpoint,
                                                     ///< 3: racing-ramp up
   int          rank;                                ///< rank of this solver
   ///-------------------------------------
   /// Counters related to this ParaSolver
   ///-------------------------------------
   int          nParaTasksReceived;                  ///< number of ParaTasks received in this ParaSolver
   int          nParaTasksSolved;                    ///< number of ParaTasks solved ( received ) in this ParaSolvere
   ///----------------------
   /// times of this solver
   ///----------------------
   double       runningTime;                        ///< this solver running time
   double       idleTimeToFirstParaTask;            ///< idle time to start solving the first ParaTask
   double       idleTimeBetweenParaTasks;           ///< idle time between ParaTasks processing
   double       idleTimeAfterLastParaTask;          ///< idle time after the last ParaTask was solved
   double       idleTimeToWaitNotificationId;       ///< idle time to wait notification Id messages
   double       idleTimeToWaitAckCompletion;        ///< idle time to wait ack completion message
   double       idleTimeToWaitToken;                ///< idle time to wait token
   ///-----------------------------
   ///  times for root task process
   ///-----------------------------
   double       detTime;                            ///< deterministic time, -1: should be non-deterministic

public:

   ///
   /// default constructor
   ///
   ParaSolverTerminationState(
         )
         : interrupted(-1),
           rank(-1),
           nParaTasksReceived(-1),
           nParaTasksSolved(-1),
           runningTime(0.0),
           idleTimeToFirstParaTask(0.0),
           idleTimeBetweenParaTasks(0.0),
           idleTimeAfterLastParaTask(0.0),
           idleTimeToWaitNotificationId(0.0),
           idleTimeToWaitAckCompletion(0.0),
           idleTimeToWaitToken(0.0),
           detTime(-1.0)
   {
   }

   ///
   /// constructor
   ///
   ParaSolverTerminationState(
         int          inInterrupted,                         ///< indicate that this solver is interrupted or not.
                                                             ///< 0: not interrupted,
                                                             ///< 1: interrupted
                                                             ///< 2: checkpoint,
                                                             ///< 3: racing-ramp up
         int          inRank,                                ///< rank of this solver
         int          inNParaTasksReceived,                  ///< number of ParaTasks received in this ParaSolver
         int          inNParaTasksSolved,                    ///< number of ParaTasks solved ( received ) in this ParaSolver
         double       inRunningTime,                         ///< this solver running time
         double       inIdleTimeToFirstParaTask,             ///< idle time to start solving the first ParaTask
         double       inIdleTimeBetweenParaTasks,            ///< idle time between ParaTasks processing
         double       inIddleTimeAfterLastParaTask,          ///< idle time after the last ParaTask was solved
         double       inIdleTimeToWaitNotificationId,        ///< idle time to wait notification Id messages
         double       inIdleTimeToWaitAckCompletion,         ///< idle time to wait ack completion message
         double       inIdleTimeToWaitToken,                 ///< idle time to wait token
         double       inDetTime                              ///< deterministic time, -1: should be non-deterministic
         )
         : interrupted(inInterrupted),
           rank(inRank),
           nParaTasksReceived(inNParaTasksReceived),
           nParaTasksSolved(inNParaTasksSolved),
           runningTime(inRunningTime),
           idleTimeToFirstParaTask(inIdleTimeToFirstParaTask),
           idleTimeBetweenParaTasks(inIdleTimeBetweenParaTasks),
           idleTimeAfterLastParaTask(inIddleTimeAfterLastParaTask),
           idleTimeToWaitNotificationId(inIdleTimeToWaitNotificationId),
           idleTimeToWaitAckCompletion(inIdleTimeToWaitAckCompletion),
           idleTimeToWaitToken(inIdleTimeToWaitToken),
           detTime(inDetTime)
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaSolverTerminationState(
         )
   {
   }

   ///
   /// stringfy ParaSolverTerminationState object
   /// @return string to show inside of ParaSolverTerminationState object
   ///
   virtual std::string toString(
         ParaInitiator *initiator     ///< pointer to ParaInitiator object
         ) = 0;

   ///
   /// getter of interrupted flag
   /// @return true if it is interrupted, false otherwise
   ///
   int getInterruptedMode(
         )
   {
      return interrupted;
   }

   ///
   /// getter of deterministic time
   /// @return deterministic time
   ///
   double getDeterministicTime(
         )
   {
      return detTime;
   }

#ifdef UG_WITH_ZLIB

   ///
   /// write ParaSolverTerminationState to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out      ///< gzstream to output
         ) = 0;

   ///
   /// read ParaSolverTerminationState from checkpoint file
   ///
   virtual bool read(
         ParaComm *comm,               ///< communicator used
         gzstream::igzstream &in       ///< gzstream to input
         ) = 0;

#endif 

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   virtual void send(
         ParaComm *comm,               ///< communicator used
         int destination,              ///< destination rank
         int tag                       ///< TagTerminated
         ) = 0;

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   virtual void receive(
         ParaComm *comm,              ///< communicator used
         int source,                  ///< source rank
         int tag                      ///< TagTerminated
         ) = 0;

};

}

#endif // __PARA_SOLVER_TERMINATION_STATE_H__

