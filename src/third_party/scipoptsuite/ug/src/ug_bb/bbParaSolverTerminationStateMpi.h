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

/**@file    paraSolverTerminationStateMpi.h
 * @brief   BbParaSolverTerminationState extension for MIP communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_SOLVER_TERMINATION_STATE_MPI_H__
#define __BB_PARA_SOLVER_TERMINATION_STATE_MPI_H__

#include <mpi.h>
#include "bbParaCommMpi.h"
#include "bbParaSolverTerminationState.h"

namespace UG
{

///
/// class BbParaSolverTerminationStateMpi
/// (Solver termination state in a ParaSolver communicated by MPI)
///
class BbParaSolverTerminationStateMpi : public BbParaSolverTerminationState
{

   ///
   /// create BbParaSolverTerminationStateMpi datatype
   /// @return MPI_Datatype for BbParaSolverTerminationStateMpi
   ///
   MPI_Datatype createDatatype(
         );

public:

   ///
   /// default constructor
   ///
   BbParaSolverTerminationStateMpi(
         )
   {
   }

   ///
   /// constructor
   ///
   BbParaSolverTerminationStateMpi(
         int          inInterrupted,                         ///< indicate that this solver is interrupted or not.
                                                             ///< 0: not interrupted,
                                                             ///< 1: interrupted
                                                             ///< 2: checkpoint,
                                                             ///< 3: racing-ramp up
         int          inRank,                                ///< rank of this solver
         int          inTotalNSolved,                        ///< accumulated number of nodes solved in this ParaSolver
         int          inMinNSolved,                          ///< minimum number of subtree nodes rooted from ParaNode
         int          inMaxNSolved,                          ///< maximum number of subtree nodes rooted from ParaNode
         int          inTotalNSent,                          ///< accumulated number of nodes sent from this ParaSolver
         int          inTotalNImprovedIncumbent,             ///< accumulated number of improvements of incumbent value in this ParaSolver
         int          inNParaNodesReceived,                  ///< number of ParaNodes received in this ParaSolver
         int          inNParaNodesSolved,                    ///< number of ParaNodes solved ( received ) in this ParaSolver
         int          inNParaNodesSolvedAtRoot,              ///< number of ParaNodes solved at root node before sending
         int          inNParaNodesSolvedAtPreCheck,          ///< number of ParaNodes solved at pre-checking of root node solvability
         int          inNTransferredLocalCutsFromSolver,     ///< number of local cuts transferred from this Solver
         int          inMinTransferredLocalCutsFromSolver,   ///< minimum number of local cuts transferred from this Solver
         int          inMaxTransferredLocalCutsFromSolver,   ///< maximum number of local cuts transferred from this Solver
         int          inNTransferredBendersCutsFromSolver,   ///< number of benders cuts transferred from this Solver
         int          inMinTransferredBendersCutsFromSolver, ///< minimum number of benders cuts transferred from this Solver
         int          inMaxTransferredBendersCutsFromSolver, ///< maximum number of benders cuts transferred from this Solver
         int          inNTotalRestarts,                      ///< number of total restarts
         int          inMinRestarts,                         ///< minimum number of restarts
         int          inMaxRestarts,                         ///< maximum number of restarts
         int          inNTightened,                          ///< number of tightened variable bounds during racing stage
         int          inNTightenedInt,                       ///< number of tightened integral variable bounds during racing stage
         int          inCalcTerminationState,                ///< termination sate of a calculation in a Solver
         double       inRunningTime,                         ///< this solver running time
         double       inIdleTimeToFirstParaNode,             ///< idle time to start solving the first ParaNode
         double       inIdleTimeBetweenParaNodes,            ///< idle time between ParaNodes processing
         double       inIdleTimeAfterLastParaNode,           ///< idle time after the last ParaNode was solved
         double       inIdleTimeToWaitNotificationId,        ///< idle time to wait notification Id messages
         double       inIdleTimeToWaitAckCompletion,         ///< idle time to wait ack completion message
         double       inIdleTimeToWaitToken,                 ///< idle time to wait token
         double       inTotalRootNodeTime,                   ///< total time consumed by root node processes
         double       inMinRootNodeTime,                     ///< minimum time consumed by root node processes
         double       inMaxRootNodeTime,                     ///< maximum time consumed by root node processes
         double       inDetTime                              ///< deterministic time, -1: should be non-deterministic
         )
         : BbParaSolverTerminationState( inInterrupted,
                                       inRank,
                                       inTotalNSolved,
                                       inMinNSolved,
                                       inMaxNSolved,
                                       inTotalNSent,
                                       inTotalNImprovedIncumbent,
                                       inNParaNodesReceived,
                                       inNParaNodesSolved,
                                       inNParaNodesSolvedAtRoot,
                                       inNParaNodesSolvedAtPreCheck,
                                       inNTransferredLocalCutsFromSolver,
                                       inMinTransferredLocalCutsFromSolver,
                                       inMaxTransferredLocalCutsFromSolver,
                                       inNTransferredBendersCutsFromSolver,
                                       inMinTransferredBendersCutsFromSolver,
                                       inMaxTransferredBendersCutsFromSolver,
                                       inNTotalRestarts,
                                       inMinRestarts,
                                       inMaxRestarts,
                                       inNTightened,
                                       inNTightenedInt,
                                       inCalcTerminationState,
                                       inRunningTime,
                                       inIdleTimeToFirstParaNode,
                                       inIdleTimeBetweenParaNodes,
                                       inIdleTimeAfterLastParaNode,
                                       inIdleTimeToWaitNotificationId,
                                       inIdleTimeToWaitAckCompletion,
                                       inIdleTimeToWaitToken,
                                       inTotalRootNodeTime,
                                       inMinRootNodeTime,
                                       inMaxRootNodeTime,
                                       inDetTime )
   {
   }

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   void send(
         ParaComm *comm,               ///< communicator used
         int destination,              ///< destination rank
         int tag                       ///< TagTerminated
         );

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   void receive(
         ParaComm *comm,              ///< communicator used
         int source,                  ///< source rank
         int tag                      ///< TagTerminated
         );

};

}

#endif // __BB_PARA_SOLVER_TERMINATION_STATE_MPI_H__

