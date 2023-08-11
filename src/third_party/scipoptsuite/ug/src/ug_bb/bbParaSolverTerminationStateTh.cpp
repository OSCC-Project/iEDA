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

/**@file    paraSolverTerminationStateTh.cpp
 * @brief   BbParaSolverTerminationState extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bbParaComm.h"
#include "bbParaSolverTerminationStateTh.h"

using namespace UG;

BbParaSolverTerminationStateTh *
BbParaSolverTerminationStateTh::createDatatype(
      )
{
   return new BbParaSolverTerminationStateTh(
         interrupted,
         rank,
         totalNSolved,
         minNSolved,
         maxNSolved,
         totalNSent,
         totalNImprovedIncumbent,
         nParaTasksReceived,
         nParaTasksSolved,
         nParaNodesSolvedAtRoot,
         nParaNodesSolvedAtPreCheck,
         nTransferredLocalCutsFromSolver,
         minTransferredLocalCutsFromSolver,
         maxTransferredLocalCutsFromSolver,
         nTransferredBendersCutsFromSolver,
         minTransferredBendersCutsFromSolver,
         maxTransferredBendersCutsFromSolver,
         nTotalRestarts,
         minRestarts,
         maxRestarts,
         nTightened,
         nTightenedInt,
         calcTerminationState,
         runningTime,
         idleTimeToFirstParaTask,
         idleTimeBetweenParaTasks,
         idleTimeAfterLastParaTask,
         idleTimeToWaitNotificationId,
         idleTimeToWaitAckCompletion,
         idleTimeToWaitToken,
         totalRootNodeTime,
         minRootNodeTime,
         maxRootNodeTime,
         detTime
         );

}

void
BbParaSolverTerminationStateTh::send(
      ParaComm *comm,
      int destination,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(), ParaSolverTerminationStateType, destination, tag)
   );
}

void
BbParaSolverTerminationStateTh::receive(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaSolverTerminationStateTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaSolverTerminationStateType, source, tag)
   );
   interrupted = received->interrupted;
   rank = received->rank;
   totalNSolved = received->totalNSolved;
   minNSolved = received->minNSolved;
   maxNSolved = received->maxNSolved;
   totalNSent = received->totalNSent;
   totalNImprovedIncumbent = received->totalNImprovedIncumbent;
   nParaTasksReceived = received->nParaTasksReceived;
   nParaTasksSolved = received->nParaTasksSolved;
   nParaNodesSolvedAtRoot = received->nParaNodesSolvedAtRoot;
   nParaNodesSolvedAtPreCheck = received->nParaNodesSolvedAtPreCheck;
   nTransferredLocalCutsFromSolver = received->nTransferredLocalCutsFromSolver;
   minTransferredLocalCutsFromSolver= received->minTransferredLocalCutsFromSolver;
   maxTransferredLocalCutsFromSolver= received->maxTransferredLocalCutsFromSolver;
   nTransferredBendersCutsFromSolver = received->nTransferredBendersCutsFromSolver;
   minTransferredBendersCutsFromSolver= received->minTransferredBendersCutsFromSolver;
   maxTransferredBendersCutsFromSolver= received->maxTransferredBendersCutsFromSolver;
   nTotalRestarts= received->nTotalRestarts;
   minRestarts= received->minRestarts;
   maxRestarts= received->maxRestarts;
   nTightened = received->nTightened;
   nTightenedInt = received->nTightenedInt;
   calcTerminationState = received->calcTerminationState;
   runningTime = received->runningTime;
   idleTimeToFirstParaTask = received->idleTimeToFirstParaTask;
   idleTimeBetweenParaTasks = received->idleTimeBetweenParaTasks;
   idleTimeAfterLastParaTask = received->idleTimeAfterLastParaTask;
   idleTimeToWaitNotificationId = received->idleTimeToWaitNotificationId;
   idleTimeToWaitAckCompletion = received->idleTimeToWaitAckCompletion;
   idleTimeToWaitToken = received->idleTimeToWaitToken;
   totalRootNodeTime = received->totalRootNodeTime;
   minRootNodeTime = received->minRootNodeTime;
   maxRootNodeTime = received->maxRootNodeTime;
   detTime = received->detTime;

   delete received;
}
