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

/**@file    paraLoadCoordinator.h
 * @brief   Load Coordinator.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_LOADCOORDINATOR_H__
#define __BB_PARA_LOADCOORDINATOR_H__

#include <fstream>
#include <list>
#include <queue>
#include <set>
#include "ug/paraDef.h"
#include "ug/paraComm.h"
#include "ug/paraLoadCoordinator.h"
#include "ug/paraTimer.h"
#include "ug/paraDeterministicTimer.h"
#include "bbParaNodePool.h"
#include "bbParaInitiator.h"
#include "bbParaSolverState.h"
#include "bbParaCalculationState.h"
#include "bbParaNode.h"
#include "bbParaParamSet.h"
#include "bbParaTagDef.h"
#include "bbParaLoadCoordinatorTerminationState.h"
#include "bbParaSolverPool.h"
#include "bbParaSolution.h"
#include "bbParaInstance.h"
#include "bbParaDiffSubproblem.h"
#include "bbParaNodesMerger.h"

#ifdef UG_WITH_UGS
#include "ugs/ugsDef.h"
#include "ugs/ugsParaCommMpi.h"
#endif

namespace UG
{

static const double displayInfOverThisValue = 5.0;          ///< if gap is over this value, Inf is displayed at gap
                                                            ///< TODO: this would move to inherited class

///
/// Class for LoadCoordinator
///
class BbParaLoadCoordinator : public UG::ParaLoadCoordinator
{

protected:

   typedef int(UG::BbParaLoadCoordinator::*BbMessageHandlerFunctionPointer)(int, int);

   bool           initialNodesGenerated;                    ///< indicates that initial nodes have been generated
   int            firstCollectingModeState;                 ///< status of first collecting mode
                                                            ///<   -1 : have not been in collecting mode
                                                            ///<    0 : once in collecting mode
                                                            ///<    1 : collecting mode is terminated once
   bool           isCollectingModeRestarted;                ///< this flag indicate if a collecting mode is restarted or not
   bool           isBreakingFinised;                        ///< indicate that breaking is finished or not
                                                            ///< if bootstrap ramp-up is not specified, this flag should be always true
   int            breakingSolverId;                         ///< all nodes collecting solver Id: -1: no collecting
   int            nReplaceToBetterNode;                     ///< the number of replacing to a better nodes
   int            nNormalSelection;                         ///< number of normal node selection to a random node selection

   bool           winnerSolverNodesCollected;               ///< indicate that all winner solver nodes has been collected

   bool           primalUpdated;                            ///< indicate that primal solution was updated or not
   bool           restartingRacing;                         ///< indicate that racing ramp-up is restarting
   int            nRestartedRacing;                         ///< number of racing stages restarted

   ///
   /// Pools in LoadCorrdinator
   ///
   BbParaNodePool *paraNodePool;                           ///< node pool

   BbParaLoadCoordinatorTerminationState lcts;             ///< LoadCoordinatorTerminationState: counters and times

   ///
   ///  To measure how long does node pool stay in empty situation
   ///
   double          statEmptyNodePoolTime;                   ///< start time that node pool becomes empty. initialized by max double

   ///
   /// racing winner information
   ///
   int                minDepthInWinnerSolverNodes;         ///< minimum depth of open nodes in the winner solver tree
   int                maxDepthInWinnerSolverNodes;         ///< maximum depth of open nodes in the winner solver tree

   ///
   /// for merging nodes
   ///
   bool                merging;                           ///< indicate that merging is processing
   int                 nBoundChangesOfBestNode;           ///< the number of fixed variables of the best node
   /// The followings are used temporary to generate merge nodes info
//   BbParaFixedValue    **varIndexTable;                   ///< variable indices table.
//   BbParaMergeNodeInfo *mergeInfoHead;                    ///< head of BbParaMergeNodeInfo list
//   BbParaMergeNodeInfo *mergeInfoTail;                    ///< tail of BbParaMergeNodeInfo list
   BbParaNodesMerger   *nodesMerger;                     ///< pointer to nodes merger object, which merges nodes

   ///
   /// counter to check if all solvers are terminated or not
   ///
   size_t             nCollectedSolvers;                  ///< number of solvers which open nodes are collected

   double             previousTabularOutputTime;          ///< to keep tabular solving status output time

   double             averageDualBoundGain;               ///< average dual bound gain: could be negative value at restart
   int                nAverageDualBoundGain;              ///< number of nodes whose dual bound gain are counted
   std::deque<double> lastSeveralDualBoundGains;          ///< keep last several dual bound gains
   double             averageLastSeveralDualBoundGains;   ///< average dual bound gains of last several ones

   double             starvingTime;                       ///< start time of starving active solvers
   double             hugeImbalanceTime;                  ///< start time of huge imbalance situation
   BbParaNodePool     *paraNodePoolToRestart;             ///< ParaNode pool to restart in ramp-down phase
   BbParaNodePool     *paraNodePoolBufferToRestart;       ///< ParaNode pool for buffering ParaNodes in huge imbalance situation

   BbParaNodePool     *paraNodePoolBufferToGenerateCPF;   ///< This is  used for GenerateReducedCheckpointFiles
   BbParaNodePool     *paraNodeToKeepCheckpointFileNodes; ///< The first n nodes may always keep in checkpoint file,
                                                          ///< that is, the n nodes are not processed in this run
   BbParaNodePool     *unprocessedParaNodes;              ///< The last n nodes may always keep in checkpoint file,
                                                          ///< that is, the n nodes are not processed in this run

   std::set<int>      *selfSplitFinisedSolvers;           ///< indicate selfSplit finished solvers

   ///
   /// output streams and flags which indicate the output is specified or not
   ///
   bool               outputTabularSolvingStatusFlag;     ///< indicate if solving status in tabular form or not
   std::ofstream      ofsTabularSolvingStatus;            ///< ofstream for solving status in tabular form
   std::ostream       *osTabularSolvingStatus;            ///< ostream for solving status in tabular form to switch output location
   bool               logSubtreeInfoFlag;                 ///< indicate if subtree info. is logged or not
   std::ofstream      ofsLogSubtreeInfo;                  ///< ofstream for subtree info.
   std::ostream       *osLogSubtreeInfo;                  ///< ostram for subtree info. to switch output location

   bool               allCompInfeasibleAfterSolution;     ///< indicate that all computations are infeasible after a feasible solution
   double             minmalDualBoundNormalTermSolvers;   ///< minimal dual bound for normal termination solvers
   bool               warmStartNodeTransferring;          ///< indicate that the first node transferring at warm start (restart)
   bool               hugeImbalance;                      ///< indicate that a huge imbalance in solvers is detected

   bool               isHeaderPrinted;                    ///< indicate if heeader is printed or not
   bool               givenGapIsReached;                  ///< shows if specified gap is reached or not

   BbParaSolution *pendingSolution;                       ///< pending solution during merging


   ///
   /// terminate all solvers
   ///
   void terminateAllSolvers(
         );

   ///
   /// notify retry ramp-up to all solvers
   ///
   virtual void sendRetryRampUpToAllSolvers(
         );

   ///
   /// send interrupt request to all solvers
   ///
   virtual void sendInterruptRequest(
         );


   ///
   /// update incumbent solution
   /// @return true if it is updated
   ///
   virtual bool updateSolution(
         BbParaSolution *                                   ///< pointer to new incumbent solution
         );

   ///
   /// send incumbent value
   ///
   virtual void sendIncumbentValue(
         int receivedRank                                 ///< solver rank which the incumbent value was generated
         );

   ///
   /// send cut off value
   ///
   virtual void sendCutOffValue(
         int receivedRank                                 ///< solver rank which the cut off value was generated
         );

   ///////////////////////
   ///
   /// Message handlers
   ///
   ///////////////////////

   ///
   /// function to process TagTask message
   /// @return always 0 (for extension)
   ///
   virtual int processTagTask(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagTask
         );

   ///
   /// function to process TagSolution message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolution(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSolution
         );

   ///
   /// function to process TagSolverState message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolverState(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSolverState
         );

   ///
   /// function to process TagCompletionOfCalculation message
   /// @return always 0 (for extension)
   ///
   virtual int processTagCompletionOfCalculation(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagCompletionOfCalculation
         );

   ///
   /// function to process TagTermStateForInterruption message
   /// @return always 0 (for extension)
   ///
   virtual int processTagTermStateForInterruption(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagUbBoundTightened
         );

   ///
   /// function to process TagAnotherNodeRequest message
   /// @return always 0 (for extension)
   ///
   virtual int processTagAnotherNodeRequest(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagAnotherNodeRequest
         );

   ///
   /// function to process TagAllowToBeInCollectingMode message
   /// @return always 0 (for extension)
   ///
   virtual int processTagAllowToBeInCollectingMode(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagAllowToBeInCollectingMode
         );

   ///
   /// function to process TagLbBoundTightened message
   /// @return always 0 (for extension)
   ///
   virtual int processTagLbBoundTightened(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagLbBoundTightened
         );

   ///
   /// function to process TagUbBoundTightened message
   /// @return always 0 (for extension)
   ///
   virtual int processTagUbBoundTightened(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagUbBoundTightened
         );

   ///
   /// function to process TagSelfSplitFinished message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSelfSplitFinished(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSelfSplitFinished
         );

   ///
   /// function to process TagNewSubtreeRootNode message
   /// @return always 0 (for extension)
   ///
   virtual int processTagNewSubtreeRootNode(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagNewSubtreeRootNode
         );

   ///
   /// function to process TagSubtreeRootNodeStartComputation message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSubtreeRootNodeStartComputation(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSubtreeRootNodeStartComputation
         );

   ///
   /// function to process TagSubtreeRootNodeToBeRemoved message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSubtreeRootNodeToBeRemoved(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSubtreeRootNodeToBeRemoved
         );

   ///
   /// function to process TagReassignSelfSplitSubtreeRootNode message
   /// @return always 0 (for extension)
   ///
   virtual int processTagReassignSelfSplitSubtreeRootNode(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagReassignSelfSplitSubtreeRootNode
         );

   ///
   /// function to process TagSelfSlpitNodeCalcuationState message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSelfSlpitNodeCalcuationState(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSelfSlpitNodeCalcuationState
         );


   ///
   /// function to process TagSelfSplitTermStateForInterruption message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSelfSplitTermStateForInterruption(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSelfSplitTermStateForInterruption
         );

   ///////////////////////
   ///
   /// message handlers specialized for racing ramp-up
   ///
   ///////////////////////

   ///
   /// function to process TagSolverState message in racing ramp-up stage
   /// @return always 0 (for extension)
   ///
   virtual int processRacingRampUpTagSolverState(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSolverState
         );

   ///
   /// function to process TagCompletionOfCalculation message in racing ramp-up stage
   /// @return always 0 (for extension)
   ///
   virtual int processRacingRampUpTagCompletionOfCalculation(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagCompletionOfCalculation
         );

   ///
   /// run function to start main process
   ///
   virtual void run(
         );

#ifdef UG_WITH_ZLIB

   ///
   /// function to update checkpoint files
   ///
   virtual void updateCheckpointFiles(
         );

   ///
   /// write LoadCorrdinator statistics to checkpoint file
   ///
   virtual void writeLoadCoordinatorStatisticsToCheckpointFile(
         gzstream::ogzstream &loadCoordinatorStatisticsStream, ///< gzstream to checkpoint file
         int nSolverInfo,                                      ///< number of solver info.
         double globalBestDualBoundValue,                      ///< global best dual bound value
         double externalGlobalBestDualBoundValue               ///< external value of the global best dual bound value
         );

   ///
   /// write previous run's statistics information
   ///
   virtual void writePreviousStatisticsInformation(
         );

#endif

   ///
   /// send ParaNodes to idle solvers
   /// @return true, if a ParaNodes is sent
   ///
   virtual bool sendParaTasksToIdleSolvers(
         );

   ///
   /// inactivate racing solver pool
   ///
   virtual void inactivateRacingSolverPool(
         int rank                         ///< winner solver rank
         );

   ///
   /// output tabular solving status header
   ///
   virtual void outputTabularSolvingStatusHeader(
         );

   ///
   /// output solving status in tabular form
   ///
   virtual void outputTabularSolvingStatus(
         char incumbent                   ///< character to show incumbent is obtained '*', if it is not incumbnet ' '
         );

   ///
   /// write subtree info.
   ///
   virtual void writeSubtreeInfo(
         int source,                      ///< solver rank
         ParaCalculationState *calcState  ///< calculation state
         );

#ifdef UG_WITH_ZLIB
   ///
   /// restart in ramp-down phase
   ///
   virtual void restartInRampDownPhase(
         );
#endif

   ///
   /// restart racing
   /// @return true if problem is solved in racing
   ///
   virtual int restartRacing(
         );

   ///
   /// start a new racing
   ///
   virtual void newRacing(
         );

   ///
   /// change search strategy of all solvers to best bound bound search strategy
   ///
   virtual void changeSearchStrategyOfAllSolversToBestBoundSearch(
         );

   ///
   /// change search strategy of all solvers to original search strategy
   ///
   virtual void changeSearchStrategyOfAllSolversToOriginalSearch(
         );

#ifdef UG_WITH_UGS

   ///
   ///  check and read incument solution
   ///
   int checkAndReadIncumbent(
         );

#endif

   ///
   /// check if current stage is in racing or not
   /// @return true, if current stage is in racing
   ///
   virtual bool isRacingStage(
         )
   {
      if( // ( !paraInitiator->getPrefixWarm() ) &&
            runningPhase == RampUpPhase &&
            (paraParams->getIntParamValue(RampUpPhaseProcess) == 1 || paraParams->getIntParamValue(RampUpPhaseProcess) == 2)  
	    &&  racingWinner < 0 )
         return true;
      else
         return false;
   }

   ///
   /// check if Gap reached or not
   /// @return true, if the specified gap is reached
   ///
   virtual bool isGapReached(
         )
   {
      return false;
   }

public:

   ///
   /// constructor
   ///
   BbParaLoadCoordinator(
#ifdef UG_WITH_UGS
         UGS::UgsParaCommMpi *inComUgs,      ///< communicator used for UGS
#endif
         int      inNhanders,                ///< number of valid message handlers
         ParaComm *inComm,                   ///< communicator used
         ParaParamSet *inParaParamSet,       ///< UG parameter set used
         ParaInitiator *paraInitiator,       ///< ParaInitiator for initialization of solving algorithm
         bool *racingSolversExist,           ///< indicate racing solver exits or not
         ParaTimer *paraTimer,               ///< ParaTimer used
         ParaDeterministicTimer *detTimer    ///< DeterministicTimer used
         );

   ///
   /// destructor
   ///
   virtual ~BbParaLoadCoordinator(
         );

#ifdef UG_WITH_ZLIB

   ///
   /// warm start (restart)
   ///
   virtual void warmStart(
         );

#endif

   ///
   /// run for normal ramp-up
   ///
   virtual void run(
         ParaTask *paraNode                              ///< root ParaNode
         );

   ///
   /// run for racing ramp-up
   ///
   virtual void run(
         ParaTask *paraNode,                             ///< root ParaNode
         int nRacingSolvers,                             ///< number of racing solvers
         ParaRacingRampUpParamSet **racingRampUpParams   ///< racing parameters
         );

   ///
   /// set global best incumbent solution
   ///
   void setGlobalBestIncumbentSolution(
         ParaSolution *sol                              ///< incumbent solution to be set
         );

};

}

#endif // __BB_PARA_LOADCOORDINATOR_H__

