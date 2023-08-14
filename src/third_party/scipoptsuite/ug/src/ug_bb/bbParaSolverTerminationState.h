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


#ifndef __BB_PARA_SOLVER_TERMINATION_STATE_H__
#define __BB_PARA_SOLVER_TERMINATION_STATE_H__

#include "ug/paraComm.h"
#include "ug/paraInitiator.h"
#include "ug/paraSolverTerminationState.h"
#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif

namespace UG
{

///
/// class BbParaSolverTerminationState
/// (Solver termination state in a ParaSolver)
///
class BbParaSolverTerminationState : public ParaSolverTerminationState
{
protected:
   ///-------------------------------------
   /// Counters related to this ParaSolver
   ///-------------------------------------
   int          totalNSolved;                        ///< accumulated number of nodes solved in this ParaSolver
   int          minNSolved;                          ///< minimum number of subtree nodes rooted from ParaNode
   int          maxNSolved;                          ///< maximum number of subtree nodes rooted from ParaNode
   int          totalNSent;                          ///< accumulated number of nodes sent from this ParaSolver
   int          totalNImprovedIncumbent;             ///< accumulated number of improvements of incumbent value in this ParaSolver
   int          nParaNodesSolvedAtRoot;              ///< number of ParaNodes solved at root node before sending
   int          nParaNodesSolvedAtPreCheck;          ///< number of ParaNodes solved at pre-checking of root node solvability
   int          nTransferredLocalCutsFromSolver;     ///< number of local cuts transferred from this Solver
   int          minTransferredLocalCutsFromSolver;   ///< minimum number of local cuts transferred from this Solver
   int          maxTransferredLocalCutsFromSolver;   ///< maximum number of local cuts transferred from this Solver
   int          nTransferredBendersCutsFromSolver;   ///< number of benders cuts transferred from this Solver
   int          minTransferredBendersCutsFromSolver; ///< minimum number of benders cuts transferred from this Solver
   int          maxTransferredBendersCutsFromSolver; ///< maximum number of benders cuts transferred from this Solver
   int          nTotalRestarts;                      ///< number of total restarts
   int          minRestarts;                         ///< minimum number of restarts
   int          maxRestarts;                         ///< maximum number of restarts
   int          nTightened;                          ///< number of tightened variable bounds during racing stage
   int          nTightenedInt;                       ///< number of tightened integral variable bounds during racing stage
   int          calcTerminationState;                ///< termination sate of a calculation in a Solver
   ///-----------------------------
   ///  times for root node process
   ///-----------------------------
   double       totalRootNodeTime;                  ///< total time consumed by root node processes
   double       minRootNodeTime;                    ///< minimum time consumed by root node processes
   double       maxRootNodeTime;                    ///< maximum time consumed by root node processes

public:

   ///
   /// default constructor
   ///
   BbParaSolverTerminationState(
         )
         : ParaSolverTerminationState(),
           totalNSolved(-1),
           minNSolved(-1),
           maxNSolved(-1),
           totalNSent(-1),
           totalNImprovedIncumbent(-1),
           nParaNodesSolvedAtRoot(-1),
           nParaNodesSolvedAtPreCheck(-1),
           nTransferredLocalCutsFromSolver(0),
           minTransferredLocalCutsFromSolver(0),
           maxTransferredLocalCutsFromSolver(0),
           nTransferredBendersCutsFromSolver(0),
           minTransferredBendersCutsFromSolver(0),
           maxTransferredBendersCutsFromSolver(0),
           nTotalRestarts(0),
           minRestarts(0),
           maxRestarts(0),
           nTightened(0),
           nTightenedInt(0),
           calcTerminationState(CompTerminatedNormally),
           totalRootNodeTime(0.0),
           minRootNodeTime(0.0),
           maxRootNodeTime(0.0)
   {
   }

   ///
   /// constructor
   ///
   BbParaSolverTerminationState(
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
         double       inIddleTimeAfterLastParaNode,          ///< idle time after the last ParaNode was solved
         double       inIdleTimeToWaitNotificationId,        ///< idle time to wait notification Id messages
         double       inIdleTimeToWaitAckCompletion,         ///< idle time to wait ack completion message
         double       inIdleTimeToWaitToken,                 ///< idle time to wait token
         double       inTotalRootNodeTime,                   ///< total time consumed by root node processes
         double       inMinRootNodeTime,                     ///< minimum time consumed by root node processes
         double       inMaxRootNodeTime,                     ///< maximum time consumed by root node processes
         double       inDetTime                              ///< deterministic time, -1: should be non-deterministic
         )
         : ParaSolverTerminationState(inInterrupted, inRank, inNParaNodesReceived, inNParaNodesSolved,
                                      inRunningTime, inIdleTimeToFirstParaNode, inIdleTimeBetweenParaNodes, inIddleTimeAfterLastParaNode,
                                      inIdleTimeToWaitNotificationId, inIdleTimeToWaitAckCompletion, inIdleTimeToWaitToken, inDetTime),
           totalNSolved(inTotalNSolved),
           minNSolved(inMinNSolved),
           maxNSolved(inMaxNSolved),
           totalNSent(inTotalNSent),
           totalNImprovedIncumbent(inTotalNImprovedIncumbent),
           nParaNodesSolvedAtRoot(inNParaNodesSolvedAtRoot),
           nParaNodesSolvedAtPreCheck(inNParaNodesSolvedAtPreCheck),
           nTransferredLocalCutsFromSolver(inNTransferredLocalCutsFromSolver),
           minTransferredLocalCutsFromSolver(inMinTransferredLocalCutsFromSolver),
           maxTransferredLocalCutsFromSolver(inMaxTransferredLocalCutsFromSolver),
           nTransferredBendersCutsFromSolver(inNTransferredBendersCutsFromSolver),
           minTransferredBendersCutsFromSolver(inMinTransferredBendersCutsFromSolver),
           maxTransferredBendersCutsFromSolver(inMaxTransferredBendersCutsFromSolver),
           nTotalRestarts(inNTotalRestarts),
           minRestarts(inMinRestarts),
           maxRestarts(inMaxRestarts),
           nTightened(inNTightened),
           nTightenedInt(inNTightenedInt),
           calcTerminationState(inCalcTerminationState),
           totalRootNodeTime(inTotalRootNodeTime),
           minRootNodeTime(inMinRootNodeTime),
           maxRootNodeTime(inMaxRootNodeTime)
   {
   }

   ///
   /// destructor
   ///
   virtual ~BbParaSolverTerminationState(
         )
   {
   }

   ///
   /// getter of calcTermination state
   /// @return termination sate of a calculation in a Solver
   ///
   int getCalcTerminationState(
         )
   {
      return calcTerminationState;
   }

   ///
   /// stringfy BbParaSolverTerminationState object
   /// @return string to show inside of BbParaSolverTerminationState object
   ///
   std::string toString(
         ParaInitiator *initiator     ///< pointer to ParaInitiator object
         );

#ifdef UG_WITH_ZLIB

   ///
   /// write BbParaSolverTerminationState to checkpoint file
   ///
   void write(
         gzstream::ogzstream &out      ///< gzstream to output
         );

   ///
   /// read BbParaSolverTerminationState from checkpoint file
   ///
   bool read(
         ParaComm *comm,               ///< communicator used
         gzstream::igzstream &in       ///< gzstream to input
         );

#endif 

};

}

#endif // __BB_PARA_SOLVER_TERMINATION_STATE_H__

