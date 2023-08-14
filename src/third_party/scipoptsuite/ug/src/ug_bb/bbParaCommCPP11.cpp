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

/**@file    paraCommCPP11.cpp
 * @brief   ParaComm extension for C++11 thread communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cstring>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "bbParaTagDef.h"
#include "bbParaCommCPP11.h"

using namespace UG;

const char *
BbParaCommCPP11::tagStringTable[] = {
  TAG_STR(TagRetryRampUp),
  TAG_STR(TagGlobalBestDualBoundValueAtWarmStart),
  TAG_STR(TagAnotherNodeRequest),
  TAG_STR(TagNoNodes),
  TAG_STR(TagInCollectingMode),
  TAG_STR(TagCollectAllNodes),
  TAG_STR(TagOutCollectingMode),
  TAG_STR(TagLCBestBoundValue),
  TAG_STR(TagLightWeightRootNodeProcess),
  TAG_STR(TagBreaking),
  TAG_STR(TagGivenGapIsReached),
  TAG_STR(TagAllowToBeInCollectingMode),
  TAG_STR(TagTestDualBoundGain),
  TAG_STR(TagNoTestDualBoundGain),
  TAG_STR(TagNoWaitModeSend),
  TAG_STR(TagRestart),
  TAG_STR(TagLbBoundTightenedIndex),
  TAG_STR(TagLbBoundTightenedBound),
  TAG_STR(TagUbBoundTightenedIndex),
  TAG_STR(TagUbBoundTightenedBound),
  TAG_STR(TagCutOffValue),
  TAG_STR(TagChangeSearchStrategy),
  TAG_STR(TagSolverDiffParamSet),
  TAG_STR(TagKeepRacing),
  TAG_STR(TagTerminateSolvingToRestart),
  TAG_STR(TagSelfSplitFinished),
  TAG_STR(TagNewSubtreeRootNode),
  TAG_STR(TagSubtreeRootNodeStartComputation),
  TAG_STR(TagSubtreeRootNodeToBeRemoved),
  TAG_STR(TagReassignSelfSplitSubtreeRootNode),
  TAG_STR(TagSelfSlpitNodeCalcuationState),
  TAG_STR(TagTermStateForInterruption),
  TAG_STR(TagSelfSplitTermStateForInterruption)
};


bool
BbParaCommCPP11::tagStringTableIsSetUpCoorectly(
      )
{
   if( !UG::ParaCommCPP11::tagStringTableIsSetUpCoorectly() ) return false;
   // std::cout << "size = " << sizeof(tagStringTable)/sizeof(char*)
   //       << ", (N_BB_TH_TAGS - N_TH_TAGS) = " <<  (N_BB_TH_TAGS - N_TH_TAGS) << std::endl;
   return ( sizeof(tagStringTable)/sizeof(char*) == (N_BB_TH_TAGS - N_TH_TAGS) );
}

const char *
BbParaCommCPP11::getTagString(
      int tag                 /// tag to be converted to string
      )
{
   assert( tag >= 0 && tag < N_BB_TH_TAGS );
   if( tag >= 0 && tag < TAG_BB_FIRST )
   {
      return ParaCommCPP11::getTagString(tag);
   }
   else
   {
      return tagStringTable[(tag - TAG_BB_FIRST)];
   }
}


ParaCalculationState *
BbParaCommCPP11::createParaCalculationState(
      )
{
   return new BbParaCalculationStateTh();
}

ParaCalculationState *
BbParaCommCPP11::createParaCalculationState(
               double compTime,                   ///< computation time of this ParaNode
               double rootTime,                   ///< computation time of the root node
               int    nSolved,                    ///< the number of nodes solved
               int    nSent,                      ///< the number of ParaNodes sent
               int    nImprovedIncumbent,         ///< the number of improved solution generated in this ParaSolver
               int    terminationState,           ///< indicate whether if this computation is terminationState or not. 0: no, 1: terminationState
               int    nSolvedWithNoPreprocesses,  ///< number of nodes solved when it is solved with no preprocesses
               int    nSimplexIterRoot,           ///< number of simplex iteration at root node
               double averageSimplexIter,         ///< average number of simplex iteration except root node
               int    nTransferredLocalCuts,      ///< number of local cuts transferred from a ParaNode
               int    minTransferredLocalCuts,    ///< minimum number of local cuts transferred from a ParaNode
               int    maxTransferredLocalCuts,    ///< maximum number of local cuts transferred from a ParaNode
               int    nTransferredBendersCuts,    ///< number of benders cuts transferred from a ParaNode
               int    minTransferredBendersCuts,  ///< minimum number of benders cuts transferred from a ParaNode
               int    maxTransferredBendersCuts,  ///< maximum number of benders cuts transferred from a ParaNode
               int    nRestarts,                  ///< number of restarts
               double minIisum,                   ///< minimum sum of integer infeasibility
               double maxIisum,                   ///< maximum sum of integer infeasibility
               int    minNii,                     ///< minimum number of integer infeasibility
               int    maxNii,                     ///< maximum number of integer infeasibility
               double dualBound,                  ///< final dual bound value
               int    nSelfSplitNodesLeft         ///< number of self-split nodes left
           )
{
   return new BbParaCalculationStateTh(
                  compTime,
                  rootTime,
                  nSolved,
                  nSent,
                  nImprovedIncumbent,
                  terminationState,
                  nSolvedWithNoPreprocesses,
                  nSimplexIterRoot,
                  averageSimplexIter,
                  nTransferredLocalCuts,
                  minTransferredLocalCuts,
                  maxTransferredLocalCuts,
                  nTransferredBendersCuts,
                  minTransferredBendersCuts,
                  maxTransferredBendersCuts,
                  nRestarts,
                  minIisum,
                  maxIisum,
                  minNii,
                  maxNii,
                  dualBound,
                  nSelfSplitNodesLeft
              );
}

ParaTask *
BbParaCommCPP11::createParaTask(
      )
{
   return new BbParaNodeTh();
}

ParaTask *
BbParaCommCPP11::createParaNode(
               TaskId inNodeId,
               TaskId inGeneratorNodeId,
               int inDepth,
               double inDualBoundValue,
               double inOriginalDualBoundValue,
               double inEstimatedValue,
               ParaDiffSubproblem *inDiffSubproblem
            )
{
    return new BbParaNodeTh(
                  inNodeId,
                  inGeneratorNodeId,
                  inDepth,
                  inDualBoundValue,
                  inOriginalDualBoundValue,
                  inEstimatedValue,
                  inDiffSubproblem
              );
}


ParaSolverState *
BbParaCommCPP11::createParaSolverState(
      )
{
   return new BbParaSolverStateTh();
}

ParaSolverState *
BbParaCommCPP11::createParaSolverState(
               int racingStage,
               unsigned int notificationId,
               int lcId,
               int globalSubtreeId,
               long long nodesSolved,
               int nodesLeft,
               double bestDualBoundValue,
               double globalBestPrimalBoundValue,
               double detTime,
               double averageDualBoundGain
           )
{
   return new BbParaSolverStateTh(
                  racingStage,
                  notificationId,
                  lcId,
                  globalSubtreeId,
                  nodesSolved,
                  nodesLeft,
                  bestDualBoundValue,
                  globalBestPrimalBoundValue,
                  detTime,
                  averageDualBoundGain
              );
}

ParaSolverTerminationState *
BbParaCommCPP11::createParaSolverTerminationState(
      )
{
   return new BbParaSolverTerminationStateTh();
}

ParaSolverTerminationState *
BbParaCommCPP11::createParaSolverTerminationState(
               int    interrupted,                ///< indicate that this solver is interrupted or not. 0: not interrupted, 1: interrputed
                                                  ///<                                                  2: checkpoint, 3: racing-ramp up
               int    rank,                       ///< rankLocal of this solver
               int    totalNSolved,               ///< accumulated number of nodes solved in this ParaSolver
               int    minNSolved,                 ///< minimum number of subtree nodes rooted from ParaNode
               int    maxNSolved,                 ///< maximum number of subtree nodes rooted from ParaNode
               int    totalNSent,                 ///< accumulated number of nodes sent from this ParaSolver
               int    totalNImprovedIncumbent,    ///< accumulated number of improvements of incumbent value in this ParaSolver
               int    nParaNodesReceived,         ///< number of ParaNodes received in this ParaSolver
               int    nParaNodesSolved,           ///< number of ParaNodes solved ( received ) in this ParaSolver
               int    nParaNodesSolvedAtRoot,     ///< number of ParaNodes solved at root node before sending
               int    nParaNodesSolvedAtPreCheck, ///< number of ParaNodes solved at pre-checking of root node solvability
               int    nTransferredLocalCutsFromSolver,      ///< number of local cuts transferred from this Solver
               int    minTransferredLocalCutsFromSolver,    ///< minimum number of local cuts transferred from this Solver
               int    maxTransferredLocalCutsFromSolver,    ///< maximum number of local cuts transferred from this Solver
               int    nTransferredBendersCutsFromSolver,    ///< number of local cuts transferred from this Solver
               int    minTransferredBendersCutsFromSolver,  ///< minimum number of local cuts transferred from this Solver
               int    maxTransferredBendersCutsFromSolver,  ///< maximum number of local cuts transferred from this Solver
               int    nTotalRestarts,             ///< number of total restarts
               int    minRestarts,                ///< minimum number of restarts
               int    maxRestarts,                ///< maximum number of restarts
               int    nTightened,                 ///< number of tightened variable bounds during racing stage
               int    nTightenedInt,              ///< number of tightened integral variable bounds during racing stage
               int    calcTerminationState,       ///< termination sate of a calculation in a Solver
               double runningTime,                ///< this solver running time
               double idleTimeToFirstParaNode,    ///< idle time to start solving the first ParaNode
               double idleTimeBetweenParaNodes,   ///< idle time between ParaNodes processing
               double iddleTimeAfterLastParaNode, ///< idle time after the last ParaNode was solved
               double idleTimeToWaitNotificationId,   ///< idle time to wait notification Id messages
               double idleTimeToWaitAckCompletion,    ///< idle time to wait ack completion message
               double idleTimeToWaitToken,        ///< idle time to wait token
               double totalRootNodeTime,          ///< total time consumed by root node processes
               double minRootNodeTime,            ///< minimum time consumed by root node processes
               double maxRootNodeTime,            ///< maximum time consumed by root node processes
               double detTime                     ///< deterministic time, -1: should be non-deterministic
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
                 nParaNodesReceived,
                 nParaNodesSolved,
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
                 idleTimeToFirstParaNode,
                 idleTimeBetweenParaNodes,
                 iddleTimeAfterLastParaNode,
                 idleTimeToWaitNotificationId,
                 idleTimeToWaitAckCompletion,
                 idleTimeToWaitToken,
                 totalRootNodeTime,
                 minRootNodeTime,
                 maxRootNodeTime,
                 detTime
              );
}
