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

/**@file    paraSolver.cpp
 * @brief   Base class for solver: Generic parallelized solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cstdlib>
#include <cfloat>
#include <climits>
#include <cassert>
#include <algorithm>
#include "ug/paraComm.h"
#include "ug/paraTask.h"
#include "ug/paraInstance.h"
#include "ug/paraSolution.h"
#include "bbParaSolverTerminationState.h"
#include "bbParaCalculationState.h"
#include "bbParaSolverState.h"
#include "bbParaSolver.h"
#include "bbParaComm.h"
#include "bbParaNodePool.h"

using namespace UG;

BbParaSolver::BbParaSolver(
      int argc,
      char **argv,
      int inNHandlers,
      ParaComm     *comm,
      ParaParamSet *inParaParamSet,
      ParaInstance *inParaInstance,
      ParaDeterministicTimer *inParaDetTimer
      )
      : ParaSolver(argc, argv, inNHandlers, comm, inParaParamSet, inParaInstance, inParaDetTimer),
      globalBestDualBoundValueAtWarmStart(-DBL_MAX),
      globalBestCutOffValue(DBL_MAX),
      lcBestDualBoundValue(-DBL_MAX),
      collectingMode(false),
      aggressiveCollecting(false),
      nSendInCollectingMode(0),
      nCollectOnce(0),
      collectingManyNodes(false),
      collectingInterrupt(false),
      anotherNodeIsRequested(false),
      lightWeightRootNodeComputation(false),
      onceBreak(false),
      rootNodeTime(0.0),
      totalRootNodeTime(0.0),
      minRootNodeTime(DBL_MAX),
      maxRootNodeTime(-DBL_MAX),
      nSolved(0),
      nSent(0),
      nSolvedWithNoPreprocesses(0),
      totalNSolved(0),
      minNSolved(INT_MAX),
      maxNSolved(INT_MIN),
      nTransferredLocalCutsFromSolver(0),
      minTransferredLocalCutsFromSolver(INT_MAX),
      maxTransferredLocalCutsFromSolver(INT_MIN),
      nTransferredBendersCutsFromSolver(0),
      minTransferredBendersCutsFromSolver(INT_MAX),
      maxTransferredBendersCutsFromSolver(INT_MIN),
      nTotalRestarts(0),
      minRestarts(INT_MAX),
      maxRestarts(INT_MIN),
      totalNSent(0),
      totalNImprovedIncumbent(0),
      nParaNodesSolvedAtRoot(0),
      nParaNodesSolvedAtPreCheck(0),
      nSimplexIterRoot(0),
      nTransferredLocalCuts(0),
      minTransferredLocalCuts(INT_MAX),
      maxTransferredLocalCuts(INT_MIN),
      nTransferredBendersCuts(0),
      minTransferredBendersCuts(INT_MAX),
      maxTransferredBendersCuts(INT_MIN),
      nTightened(0),
      nTightenedInt(0),
      minIisum(DBL_MAX),
      maxIisum(0.0),
      minNii(INT_MAX),
      maxNii(0),
      targetBound(-DBL_MAX),
      nTransferLimit(-1),
      nTransferredNodes(-1),
      solverDualBound(-DBL_MAX),
      averageDualBoundGain(0.0),
      enoughGainObtained(true),
      givenGapIsReached(false),
      testDualBoundGain(false),
      noWaitModeSend(false),
      keepRacing(false),
      restartingRacing(false),
      localIncumbentIsChecked(false),
      selfSplitNodePool(0)
{
   /** create timer for this BbParaSolver */

   BbMessageHandlerFunctionPointer *bbMessageHandler = reinterpret_cast<BbMessageHandlerFunctionPointer *>(messageHandler);

   bbMessageHandler[TagTask] = &UG::BbParaSolver::processTagTask;
   bbMessageHandler[TagRetryRampUp] = &UG::BbParaSolver::processTagRetryRampUp;
   bbMessageHandler[TagGlobalBestDualBoundValueAtWarmStart] = &UG::BbParaSolver::processTagGlobalBestDualBoundValueAtWarmStart;
   bbMessageHandler[TagNoNodes] = &UG::BbParaSolver::processTagNoNodes;
   bbMessageHandler[TagInCollectingMode] = &UG::BbParaSolver::processTagInCollectingMode;
   bbMessageHandler[TagCollectAllNodes] = &UG::BbParaSolver::processTagCollectAllNodes;
   bbMessageHandler[TagOutCollectingMode] = &UG::BbParaSolver::processTagOutCollectingMode;
   bbMessageHandler[TagLCBestBoundValue] = &UG::BbParaSolver::processTagLCBestBoundValue;
   bbMessageHandler[TagInterruptRequest] = &UG::BbParaSolver::processTagInterruptRequest;
   bbMessageHandler[TagLightWeightRootNodeProcess] = &UG::BbParaSolver::processTagLightWeightRootNodeProcess;
   bbMessageHandler[TagBreaking] = &UG::BbParaSolver::processTagBreaking;
   bbMessageHandler[TagGivenGapIsReached] = &UG::BbParaSolver::processTagGivenGapIsReached;
   bbMessageHandler[TagTestDualBoundGain] = &UG::BbParaSolver::processTagTestDualBoundGain;
   bbMessageHandler[TagNoTestDualBoundGain] = &UG::BbParaSolver::processTagNoTestDualBoundGain;
   bbMessageHandler[TagNoWaitModeSend] = &UG::BbParaSolver::processTagNoWaitModeSend;
   bbMessageHandler[TagRestart] = &UG::BbParaSolver::processTagRestart;
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      bbMessageHandler[TagLbBoundTightenedIndex] = &UG::BbParaSolver::processTagLbBoundTightened;
      bbMessageHandler[TagUbBoundTightenedIndex] = &UG::BbParaSolver::processTagUbBoundTightened;
   }
   bbMessageHandler[TagCutOffValue] = &UG::BbParaSolver::processTagCutOffValue;
   bbMessageHandler[TagKeepRacing] = &UG::BbParaSolver::processTagKeepRacing;
   bbMessageHandler[TagTerminateSolvingToRestart] = &UG::BbParaSolver::processTagTerminateSolvingToRestart;

   if ( paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 )
   {
      paraParams->setRealParamValue(UG::TimeLimit, -1.0);
   }

   if( isWarmStarted() && paraParams->getIntParamValue(RampUpPhaseProcess) == 3 )
   {
      paraParams->setIntParamValue(RampUpPhaseProcess, 0);      // it should work with normal ramp-up
   }

   if( paraParams->getIntParamValue(RampUpPhaseProcess) == 3 )  // self-split ramp-up
   {
      selfSplitNodePool = new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
   }
}

void
BbParaSolver::sendSolverTerminationState(
      )
{
   double stopTime = paraTimer->getElapsedTime();
   idleTimeAfterLastParaTask = stopTime - previousStopTime - ( idleTimeToWaitToken - previousIdleTimeToWaitToken );
   int interrupted = terminationMode == InterruptedTerminationMode ? 1 : 0;
   int calcTermState = InterruptedTerminationMode ? CompTerminatedByInterruptRequest : CompTerminatedNormally;

   double detTime = -1.0;

   DEF_BB_PARA_COMM(bbParaComm, paraComm);

   BbParaSolverTerminationState *paraSolverTerminationState = dynamic_cast<BbParaSolverTerminationState *>(bbParaComm->createParaSolverTerminationState(
          interrupted,
          paraComm->getRank(),
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
          calcTermState,
          nTightenedInt,
          stopTime,
          idleTimeToFirstParaTask,
          idleTimeBetweenParaTasks,
          idleTimeAfterLastParaTask,
          idleTimeToWaitNotificationId,
          idleTimeToWaitAckCompletion,
          idleTimeToWaitToken,
          totalRootNodeTime,
          minRootNodeTime,
          maxRootNodeTime,
          detTime ));
   paraSolverTerminationState->send(paraComm, 0, TagTerminated);
   delete paraSolverTerminationState;

}

void
BbParaSolver::notifySelfSplitFinished(
      )
{
   // PARA_COMM_CALL(
   //       paraComm->receive( NULL, 0, ParaBYTE, 0, TagSelfSplitFinished )
   //       );
   // Node で
   // 貰ったルートの一つ下
   PARA_COMM_CALL(
         paraComm->send( NULL, 0, ParaBYTE, 0, TagSelfSplitFinished )
         );
   return;
}


int
BbParaSolver::processTagTask(
      int source,
      int tag
      )
{

   DEF_BB_PARA_COMM(bbParaComm, paraComm);

   if( currentTask )
   {
      newTask = bbParaComm->createParaTask();
      newTask->receive(bbParaComm, source);
      if( dynamic_cast<BbParaNode *>(newTask)->getMergingStatus() == 3 )  // This means, the received node is separated
      {
         dynamic_cast<BbParaNode *>(newTask)->setMergingStatus(-1);
      }
   }
   else
   {
      currentTask = bbParaComm->createParaTask();
      currentTask->receive(bbParaComm, source);
      if( dynamic_cast<BbParaNode *>(currentTask)->getMergingStatus() == 3 )  // This means, the received node is separated
      {
         dynamic_cast<BbParaNode *>(currentTask)->setMergingStatus(-1);
      }
   }
   anotherNodeIsRequested = false;
   nParaTasksReceived++;
   if( keepRacing )
   {
      assert(!winnerRacingParams);
   }
   return 0;
}

int
BbParaSolver::processTagTaskReceived(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagTaskReceived )
         );
   return 0;
}

int
BbParaSolver::processTagRampUp(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagRampUp)
         );
   // if( isRacingStage() )
   // {
   //    assert(!racingWinner);
   // }
   rampUp = true;
   // testDualBoundGain = false;
   enoughGainObtained = true;

#ifdef _DEBUG_CHECK_RECEIVE
   std::cout << paraTimer->getElapsedTime() << " Solver" << paraComm->getRank() << " received TagRampUp" << std::endl;
#endif

   if( ( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) && 
       racingWinner )
   {
      return 0;
   }

   // if( racingWinner )   // when the winner solver switches to normal ramp up without collecting nodes, this is necessary
   // {
   //    setOriginalNodeSelectionStrategy();
   // }
   setOriginalNodeSelectionStrategy();
   collectingManyNodes = false;
   collectingMode = false;
   aggressiveCollecting = false;
   nSendInCollectingMode = 0;
   noWaitModeSend = false;
   return 0;
}

int
BbParaSolver::processTagSolution(
      int source,
      int tag
      )
{
   ParaSolution *sol = paraComm->createParaSolution();
   sol->receive(paraComm, source);
   if( EPSLE( sol->getObjectiveFunctionValue(), globalBestIncumbentValue, DEFAULT_NUM_EPSILON ) )
      // sometime it is necessary to save the solution in solver side
      //   updateGlobalBestIncumbentValue( sol->getObjectiveFuntionValue() ) ) //  DO NOT UPDATE!!
      //   The timing of the update depends on solver used
   {
      if( pendingSolution )
      {
         delete pendingSolution;
      }
      pendingSolution = sol;
      pendingIncumbentValue = sol->getObjectiveFunctionValue();
   }
   else
   {
      delete sol;
   }
   return 0;
}

int
BbParaSolver::processTagIncumbentValue(
      int source,
      int tag
      )
{
   double incumbent;
   PARA_COMM_CALL(
         paraComm->receive( &incumbent, 1, ParaDOUBLE, source, TagIncumbentValue)
         );
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      if( incumbent  < globalBestIncumbentValue && incumbent < pendingIncumbentValue )
      {
         pendingIncumbentValue = incumbent;
      }
   }
   else
   {
      updateGlobalBestIncumbentValue(incumbent);
   }

   //
   // should not do as follows: LC does not know which nodes were removed
   //
   // if( selfSplitNodePool )
   // {
   //    selfSplitNodePool->removeBoundedNodes(incumbent);
   // }
   return 0;
}

int
BbParaSolver::processTagNotificationId(
      int source,
      int tag
      )
{
   assert(notificationProcessed);
   unsigned int notificationId;
   PARA_COMM_CALL(
         paraComm->receive( &notificationId, 1, ParaUNSIGNED, source, TagNotificationId)
         );
   if( notificationId == notificationIdGenerator) notificationProcessed = false;
   else {
      THROW_LOGICAL_ERROR4("notificationId received is ", notificationId, ", but generator value is ", notificationIdGenerator);
   }
   return 0;
}

int
BbParaSolver::processTagTerminateRequest(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagTerminateRequest)
         );
// std::cout << "Solver" << paraComm->getRank() << " received exitSolverRequest" << std::endl;
   terminationMode = NormalTerminationMode;
   stayAliveAfterInterrupt = false;
   return 0;
}

int
BbParaSolver::processTagWinnerRacingRampUpParamSet(
      int source,
      int tag
      )
{
   /* received racing parameter set is always winner one,
    * because the initail racing parameter set is broadcasted. */
   winnerRacingParams = paraComm->createParaRacingRampUpParamSet();
   PARA_COMM_CALL(
         winnerRacingParams->receive(paraComm, 0)
         );
   racingInterruptIsRequested = true;
   if( isRacingStage() )
   {
      assert(!racingWinner);
      racingIsInterrupted = true;
   }
#ifdef _DEBUG_CHECK_RECEIVE
   std::cout << paraTimer->getElapsedTime() << " Solver" << paraComm->getRank() << " received racing winner params" << std::endl;
#endif
   return 0;
}

int
BbParaSolver::processTagWinner(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagWinner)
         );
   racingWinner = true;
   assert(!winnerRacingParams);
   winnerRacingParams = racingParams; // LC does not send racing param to winner
   racingParams = 0;      // No racing stage now
   return 0;
}

int
BbParaSolver::processTagToken(
      int source,
      int tag
      )
{
   assert(paraParams->getBoolParamValue(Deterministic));
   int token[2];
   PARA_COMM_CALL(
         paraComm->receive( token, 2, ParaINT, source, TagToken )
         );
   paraComm->setToken(paraComm->getRank(), token);
   return 0;
}

int
BbParaSolver::processTagRetryRampUp(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagRetryRampUp)
         );
   rampUp = false;
   /* RetryRampUp did not work for rmine10 44th run */
   /* Then, added the following four lines */
   collectingMode = false;
   aggressiveCollecting = false;
   nSendInCollectingMode = 0;
   noWaitModeSend = false;
   /* end of the four lines */
   return 0;
}

int
BbParaSolver::processTagGlobalBestDualBoundValueAtWarmStart(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &globalBestDualBoundValueAtWarmStart, 1, ParaDOUBLE, source, TagGlobalBestDualBoundValueAtWarmStart)
         );
   return 0;
}

int
BbParaSolver::processTagNoNodes(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagNoNodes)
         );
   anotherNodeIsRequested = false;
   return 0;
}

int
BbParaSolver::processTagInCollectingMode(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &nSendInCollectingMode, 1, ParaINT, source, TagInCollectingMode)
         );
   if( nSendInCollectingMode < 0 )
   {
      nSendInCollectingMode = ( 0 - nSendInCollectingMode ) - 1;
      aggressiveCollecting = true;
   }
   collectingMode = true;

   if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
   {
      if( selfSplitNodePool->getNumOfNodes() >= 2 ||
            ( selfSplitNodePool->getNumOfNodes() > 0 && getNNodesLeft() > 0 ) )
      {
         BbParaNode *node = selfSplitNodePool->extractNode();
         node->sendSubtreeRootNodeId(paraComm, 0, TagReassignSelfSplitSubtreeRootNode);
         // std::cout << "Trans NODE (inCollecting): " << node->toSimpleString() << std::endl;
         delete node;
         if( selfSplitNodePool->isEmpty() )
         {
            // DO NOT delete below, since it shows that current solving node is generated by self-split ramp-up
//            delete selfSplitNodePool;
//            selfSplitNodePool = 0;
         }
      }
   }

   return 0;
}

int
BbParaSolver::processTagCollectAllNodes(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &nCollectOnce, 1, ParaINT, source, TagCollectAllNodes)
         );
   collectingManyNodes = true;
   noWaitModeSend = true;
   return 0;
}

int
BbParaSolver::processTagOutCollectingMode(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagOutCollectingMode)
         );
   if( terminationMode == NoTerminationMode )
   {
      // if this solver is in Ttermination mode, the following function cannot be called
      setOriginalNodeSelectionStrategy();
   }
   collectingMode = false;
   aggressiveCollecting = false;
   nSendInCollectingMode = 0;
   noWaitModeSend = false;
   return 0;
}

int
BbParaSolver::processTagLCBestBoundValue(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &lcBestDualBoundValue, 1, ParaDOUBLE, source, TagLCBestBoundValue)
         );
   return 0;
}

int
BbParaSolver::processTagInterruptRequest(
      int source,
      int tag
      )
{
// std::cout << paraTimer->getElapsedTime() << " Solver" << paraComm->getRank() << " received TagInterruptRequest" << std::endl;

   int exitSolverRequest;
   PARA_COMM_CALL(
         paraComm->receive( &exitSolverRequest, 1, ParaINT, source, TagInterruptRequest)
         );

   issueInterruptSolve();
   terminationMode = InterruptedTerminationMode;

//   std::cout << "Solver" << paraComm->getRank() << " received TagInterruptRequest with exitSolverRequest = " << exitSolverRequest << std::endl;

   if( exitSolverRequest == 1 )
   {
      collectingInterrupt = true;
      setSendBackAllNodes();
      stayAliveAfterInterrupt = true;
      noWaitModeSend = true;
   }
   else
   {
      if( exitSolverRequest == 2 )
      {
         stayAliveAfterInterrupt = true;
      }
      else
      {
         stayAliveAfterInterrupt = false;
      }
   }

   if( keepRacing )
   {
      stayAliveAfterInterrupt = true;
   }

   if( selfSplitNodePool )
   {
      while( !selfSplitNodePool->isEmpty() )
      {
         BbParaNode *node = selfSplitNodePool->extractNode();
         delete node;
      }
   }

   return 0;
}

int
BbParaSolver::processTagLightWeightRootNodeProcess(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagLightWeightRootNodeProcess)
         );
   lightWeightRootNodeComputation = true;
   setLightWeightRootNodeProcess();
   return 0;
}

int
BbParaSolver::processTagBreaking(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &targetBound, 1, ParaDOUBLE, source, TagBreaking )
         );
   PARA_COMM_CALL(
         paraComm->receive( &nTransferLimit, 1, ParaINT, source, TagBreaking )
         );
   nTransferredNodes = 0;
   collectingManyNodes = true;
   return 0;
}

int
BbParaSolver::processTagGivenGapIsReached(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagGivenGapIsReached )
         );
   issueInterruptSolve();
   stayAliveAfterInterrupt = false;
   givenGapIsReached = true;
   return 0;
}

int
BbParaSolver::processTagTestDualBoundGain(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( &averageDualBoundGain, 1, ParaDOUBLE, source, TagTestDualBoundGain )
         );
   testDualBoundGain = true;
   enoughGainObtained = true;
   return 0;
}

int
BbParaSolver::processTagNoTestDualBoundGain(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagNoTestDualBoundGain )
         );
   testDualBoundGain = false;
   enoughGainObtained = true;
   return 0;
}

int
BbParaSolver::processTagNoWaitModeSend(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagNoWaitModeSend )
         );
   noWaitModeSend = true;
   return 0;
}

int
BbParaSolver::processTagRestart(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagRestart )
         );
   stayAliveAfterInterrupt = false;
   terminationMode = NoTerminationMode;
   rampUp = false;
   return 0;
}

int
BbParaSolver::processTagLbBoundTightened(
      int source,
      int tag
      )
{
   return lbBoundTightened(source, tag);
}

int
BbParaSolver::processTagUbBoundTightened(
      int source,
      int tag
      )
{
   return ubBoundTightened(source, tag);
}

int
BbParaSolver::processTagCutOffValue(
      int source,
      int tag
      )
{
   double cutOffValue;
   PARA_COMM_CALL(
         paraComm->receive( &cutOffValue, 1, ParaDOUBLE, source, TagCutOffValue)
         );
   updateGlobalBestCutOffValue(cutOffValue);
   return 0;
}

int
BbParaSolver::processTagKeepRacing(
      int source,
      int tag
      )
{
   int keep = 0;
   PARA_COMM_CALL(
         paraComm->receive( &keep, 1, ParaINT, source, TagKeepRacing )
         );
   if( keep == 1 )
   {
      keepRacing = true;
   }
   else
   {
      keepRacing = false;
   }
   return 0;
}



int
BbParaSolver::processTagTerminateSolvingToRestart(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagTerminateSolvingToRestart )
         );
   assert(isRacingStage());
   restartingRacing = true;
   racingIsInterrupted = true;
   return 0;
}

void
BbParaSolver::run(
      )
{
   for(;;)
   {
      /***************************************************
       *  Wait a new ParaNode from ParaLoadCoordinator   *
       *  If Termination message is received, then break *
       ***************************************************/
      if( !currentTask )
      {
         if( receiveNewTaskAndReactivate() == false )
         {
            break;
         }
      }

      if( winnerRacingParams ) // winner racing parameter set is received
      {
         if( racingParams )
         {
            delete racingParams;
            racingParams = 0;
         }
         setWinnerRacingParams(winnerRacingParams);
         racingInterruptIsRequested = false;
         // delete winnerRacingParams;
         // winnerRacingParams = 0;
      }

      if( paraParams->getBoolParamValue(SetAllDefaultsAfterRacing) &&
            winnerRacingParams &&
            nParaTasksReceived >= 2 )   // after racing, parameters should be set to default values
      {
         setWinnerRacingParams(0);
         if( winnerRacingParams )
         {
            delete winnerRacingParams;
            winnerRacingParams = 0;
         }
         racingInterruptIsRequested = false;
         racingWinner = false;
      }

      /** set collecting mode */
      collectingMode = false;  /* begin with out-collecting mode: NOTE: LC clear collecting mode for new solver */
      nSendInCollectingMode = 0;
      aggressiveCollecting = false;
      collectingManyNodes = false;
      nCollectOnce = 0;
      resetBreakingInfo();     // set false on collectingManyNodes in the  resetBreakingInfo
      onceBreak = false;
      noWaitModeSend = false;

      /** set start time and ilde times */
      paraTaskStartTime = paraTimer->getElapsedTime();
      if( previousStopTime < 0.0 )
      {
         idleTimeToFirstParaTask = paraTaskStartTime - (idleTimeToWaitToken - previousIdleTimeToWaitToken);
      }
      else
      {
         idleTimeBetweenParaTasks += ( paraTaskStartTime - previousStopTime - ( idleTimeToWaitToken - previousIdleTimeToWaitToken ) );
      }

      /****************************************************
       * create subproblem into target solver environment *
       ***************************************************/
      subproblemFreed = false;
      createSubproblem();
      updatePendingSolution();
      if( globalBestIncumbentSolution )
      {
         tryNewSolution(globalBestIncumbentSolution);
      }
      globalIncumbnetValueUpdateFlag = false;

      /******************
       * start solving  *
       ******************/
      assert(!newTask);
      // std::cout << "R." << paraComm->getRank() << " starts solving" << std::endl;
      solve();
      // std::cout << "R." << paraComm->getRank() << " solved a ParaNode" << std::endl;
      /*****************************************************
       * notify completion of a calculation of a ParaNode  *
       *****************************************************/
      previousStopTime = paraTimer->getElapsedTime();
      double compTime = previousStopTime - paraTaskStartTime;
      previousIdleTimeToWaitToken = idleTimeToWaitToken;

      nSolved += getNNodesSolved();       // In case of self-split, the number of nodes solved needs to be added for root node

      if( paraParams->getBoolParamValue(CheckEffectOfRootNodePreprocesses) && nSolved == 1)
      {
         solveToCheckEffectOfRootNodePreprocesses();
      }

      /****************************************************************************
      * send completion of calculation and update counters and accumulation time  *
      ****************************************************************************/
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         do
         {
            iReceiveMessages();
         } while ( !waitToken(paraComm->getRank()) );
      }
      iReceiveMessages();     /** Before sending completion state, receiving message should be checked.
                               *   When subproblem terminated with no branch, solver lost a timing for receiving new node */

      if( currentTask )       // When a solver is reserved for multi-threaded parallel solver, currentTaks can be deleted by user routine
      {
         if( !selfSplitNodePool )
         {
            // std::cout << "Solver (!selfSplitNodePool) Rank = " << paraComm->getRank() << ", node = " << dynamic_cast<BbParaNode *>(currentTask)->toSimpleString() << std::endl;
            sendCompletionOfCalculation(compTime, TagCompletionOfCalculation, 0);
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               waitAckCompletion();
               // if( hasToken() ) passToken();
               paraDetTimer->update(1.0);
               previousCommTime = paraDetTimer->getElapsedTime();
   #ifdef _DEBUG_DET
               std::cout << previousCommTime << " run2 R." << paraComm->getRank() << ": token passed" << std::endl;
   #endif
               passToken(paraComm->getRank());
            }
         }
         else
         {
            // std::cout << "Solver Rank = " << paraComm->getRank() << ", node = " << dynamic_cast<BbParaNode *>(currentTask)->toSimpleString() << std::endl;
            // dynamic_cast<BbParaNode *>(currentTask)->sendSubtreeRootNodeId(paraComm, 0, TagSubtreeRootNodeToBeRemoved);
            sendCompletionOfCalculation(compTime, TagSelfSlpitNodeCalcuationState, selfSplitNodePool->getNumOfNodes() );
            // std::cout << "*** Rank" << paraComm->getRank() << ", num in selfSplit node pool = " <<  selfSplitNodePool->getNumOfNodes() <<", " << currentTask->toSimpleString() << std::endl;
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               waitAckCompletion();
               // if( hasToken() ) passToken();
               paraDetTimer->update(1.0);
               previousCommTime = paraDetTimer->getElapsedTime();
   #ifdef _DEBUG_DET
               std::cout << previousCommTime << " run2 R." << paraComm->getRank() << ": token passed" << std::endl;
   #endif
               // passToken(paraComm->getRank());
            }
            if( selfSplitNodePool->isEmpty() )
            {
               delete selfSplitNodePool;
               selfSplitNodePool = 0;
               // std::cout << "Rank" << paraComm->getRank() << ", SELFSPLIT NODE POOL NULL" << std::endl;
               if( paraParams->getBoolParamValue(Deterministic) )
               {
                  passToken(paraComm->getRank());
               }
            }
            if( dynamic_cast<BbParaNode *>(currentTask)->isRootTask() )
            {
               notifySelfSplitFinished();
            }
         }
      }

      /*******************************************
      * free solving environment for subproblem  *
      ********************************************/
      if( !subproblemFreed )
      {
         freeSubproblem();
      }

      /* if light wait root node computation is applied, rest it */
      if( lightWeightRootNodeComputation )
      {
         setOriginalRootNodeProcess();
         lightWeightRootNodeComputation = false;
      }

      /**************************
      * update current ParaNode *
      ***************************/
      // assert( currentTask );
      if( currentTask )   // When a solver is reserved for multi-threaded parallel solver, currentTaks can be deleted by user routine
      {
         delete currentTask;
      }
      if( newTask )
      {
         currentTask = newTask;
         newTask = 0;
      }
      else
      {
         currentTask = 0;
      }
      if( terminationMode && !stayAliveAfterInterrupt )
      {
         break;
      }

      if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
      {

         assert(!currentTask);
         while( !currentTask && selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
         {
            currentTask = selfSplitNodePool->extractNode();
            if( dynamic_cast<BbParaNode *>(currentTask)->getInitialDualBoundValue() > globalBestIncumbentValue )
            {
               dynamic_cast<BbParaNode *>(currentTask)->sendSubtreeRootNodeId(paraComm, 0, TagSubtreeRootNodeStartComputation);
               if( selfSplitNodePool->getNumOfNodes() > 0 )
               {
                  sendCompletionOfCalculationWithoutSolving(0.0, TagSelfSlpitNodeCalcuationState, selfSplitNodePool->getNumOfNodes() );
               }
               else
               {
		  // Now self-split node pool becomes empty
                  sendCompletionOfCalculationWithoutSolving(0.0, TagSelfSlpitNodeCalcuationState, 0);
                  delete selfSplitNodePool;
                  selfSplitNodePool = 0;
               }
               delete currentTask;
               currentTask = 0;
               if( paraParams->getBoolParamValue(Deterministic) )
               {
                  waitAckCompletion();
                  // if( hasToken() ) passToken();
                  paraDetTimer->update(1.0);
                  previousCommTime = paraDetTimer->getElapsedTime();
#ifdef _DEBUG_DET
                  std::cout << previousCommTime << " run2 R." << paraComm->getRank() << ": token passed" << std::endl;
#endif
                  // passToken(paraComm->getRank());
               }
            }
         }

         if( paraParams->getBoolParamValue(Deterministic) )
         {
            passToken(paraComm->getRank());
         }

         if( selfSplitNodePool && currentTask )  // NOTE: if selfplitNodePool != NULL, new node could receive in waitAckCompletion in case of deterministic 
         {
            // std::cout << "new NODE: " << currentTask->toSimpleString();
             // if( dynamic_cast<BbParaNode *>(currentTask)->getDiffSubproblem() )
             // {
             //    std::cout
             //    << ", "
             ////     << dynamic_cast<BbParaDiffSubproblem *>(dynamic_cast<BbParaNode *>(currentTask)->getDiffSubproblem())->getNBoundChanges()
             //    << dynamic_cast<BbParaDiffSubproblem *>(dynamic_cast<BbParaNode *>(currentTask)->getDiffSubproblem())->toStringStat();
             // }
             // std::cout << std::endl;
             dynamic_cast<BbParaNode *>(currentTask)->sendSubtreeRootNodeId(paraComm, 0, TagSubtreeRootNodeStartComputation);
             if( selfSplitNodePool->isEmpty() )
             {
                  // DO NOT delete below, since it shows that current solving node is generated by self-split ramp-up
                  //            delete selfSplitNodePool;
                  //            selfSplitNodePool = 0;
             }
             else
             {
                if( collectingMode && selfSplitNodePool->getNumOfNodes() > 0 )
                {
                   BbParaNode *node = selfSplitNodePool->extractNode();
                   node->sendSubtreeRootNodeId(paraComm, 0, TagReassignSelfSplitSubtreeRootNode);
                   // std::cout << "Trans NODE (run): " << node->toSimpleString() << std::endl;
                   // if( dynamic_cast<BbParaNode *>(node)->getDiffSubproblem() )
                   // {
                   //    std::cout
                   //    << ", "
                   ////     << dynamic_cast<BbParaDiffSubproblem *>(dynamic_cast<BbParaNode *>(node)->getDiffSubproblem())->getNBoundChanges()
                   //    << dynamic_cast<BbParaDiffSubproblem *>(dynamic_cast<BbParaNode *>(node)->getDiffSubproblem())->toStringStat();
                   // }
                   // std::cout << std::endl;
                   delete node;
                   if( selfSplitNodePool->isEmpty() )
                   {
                      // DO NOT delete below, since it shows that current solving node is generated by self-split ramp-up
                      //                  delete selfSplitNodePool;
                      //                  selfSplitNodePool = 0;
                   }
                }
             }
         }
      }
   }
   return;
}

bool
BbParaSolver::receiveNewTaskAndReactivate()
{
   for(;;)
   {
      int source;
      int tag;
      int status;
      /*******************************************
       *  waiting for any message form anywhere  *
       *******************************************/
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         do
         {
            iReceiveMessages();
         } while( !waitToken(paraComm->getRank()) );
         iReceiveMessages();
         paraDetTimer->update(1.0);
         previousCommTime = paraDetTimer->getElapsedTime();
#ifdef _DEBUG_DET
         std::cout << previousCommTime << " receiveNewNodeAndReactivate R." << paraComm->getRank() << ": token passed" << std::endl;
#endif
         passToken(paraComm->getRank());
      }
      else
      {
         (void)paraComm->probe(&source, &tag);
         if( messageHandler[tag] )
         {
            status = (this->*messageHandler[tag])(source, tag);
            if( status )
            {
               std::ostringstream s;
               s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
                 << __func__ << ", line = " << __LINE__ << " - "
                 << "process tag = " << tag << std::endl;
               abort();
            }
         }
         else
         {
            THROW_LOGICAL_ERROR3( "No message hander for ", tag, " is not registered" );
         }
      }
      if( currentTask )
      {
         if( paraParams->getBoolParamValue(Deterministic)  )
         {
            previousNotificationTime = paraDetTimer->getElapsedTime();
         }
         else
         {
            previousNotificationTime = paraTimer->getElapsedTime();
         }
         return true;
      }
      if( terminationMode && !stayAliveAfterInterrupt ) break;
   }
   return false;
}

void
BbParaSolver::iReceiveMessages(
      )
{
#ifdef _DEBUG_CHECK_RECEIVE
   static double previousRreceiveCheckTime = DBL_MAX;
   double currentTime = paraTimer->getElapsedTime();
   if( ( currentTime - previousRreceiveCheckTime ) < -1.0 )
   {
      std::cout << currentTime << " Solver" << paraComm->getRank() << " No check receiving message over 500 (sec.) is logging." << std::endl;
   }
   if( ( currentTime - previousRreceiveCheckTime ) > 500.0 )
   {
      std::cout << currentTime << " Solver" << paraComm->getRank() << " did not check receiving message over 500 (sec.)" << std::endl;
      writeSubproblem();
   }
   previousRreceiveCheckTime = currentTime;
#endif
   /************************************************************************
    * This fucntion is called from a callback routine of the target solver *
    * **********************************************************************/
   int source;
   int tag = TagAny;
   int status;
   /************************************
    * check if there are some messages *
    ************************************/
   while( paraComm->iProbe(&source, &tag) )
   {
      if( messageHandler[tag] )
      {
         status = (this->*messageHandler[tag])(source, tag);
         if( status )
         {
            std::ostringstream s;
            s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
              << __func__ << ", line = " << __LINE__ << " - "
              << "process tag = " << tag << std::endl;
            abort();
         }
      }
      else
      {
         if( terminationMode == NormalTerminationMode && tag == TagAckCompletion )
         {
            //
            // receive the notification Id message
            //
            PARA_COMM_CALL(
                  paraComm->receive( NULL, 0, ParaBYTE, 0, TagAckCompletion)
                  );
            continue;
         }
         THROW_LOGICAL_ERROR3( "No message hander for ", tag, " is not registered" );
      }
   }
}

void
BbParaSolver::setRootNodeTime(
      )
{
   rootNodeTime = paraTimer->getElapsedTime() - paraTaskStartTime;
   if( rootNodeTime < minRootNodeTime )
   {
      minRootNodeTime = rootNodeTime;
   }
   if( rootNodeTime > maxRootNodeTime )
   {
      maxRootNodeTime = rootNodeTime;
   }
}

void
BbParaSolver::sendCompletionOfCalculation(
      double compTime,
      int tag,                 ///< message Tag
      int nSelfSplitNodesLeft  ///< number of self-split nodes left
      )
{
   int terminationState = CompTerminatedNormally;
   bool needToResetMaximalDualBound = racingIsInterrupted;

   if( givenGapIsReached )
   {
      if( isRacingStage() )
      {
         terminationState = CompTerminatedInRacingStage;
      }
      else
      {
         terminationState = CompTerminatedByInterruptRequest;
      }
   }
   else if( memoryLimitIsReached )
   {
      terminationState = CompTerminatedByMemoryLimit;
   }
   else
   {
      if( terminationMode == InterruptedTerminationMode )
      {
         terminationState = CompTerminatedByInterruptRequest;
      }
      else
      {
         if( terminationMode == TimeLimitTerminationMode )
         {
            terminationState = CompTerminatedByTimeLimit;
         }
         else
         {
            if ( newTask )
            {
               terminationState = CompTerminatedByAnotherTask;
            }
            else
            {
               if( anotherNodeIsRequested )
               {
                  for(;;)
                  {
                     int source;
                     int ttag;
                     int status;
                     /*******************************************
                      *  waiting for any message from anywhere  *
                      *******************************************/
                     (void)paraComm->probe(&source, &ttag);
                      if( messageHandler[ttag] )
                      {
                         status = (this->*messageHandler[ttag])(source, ttag);
                         if( status )
                         {
                            std::ostringstream s;
                            s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
                              << __func__ << ", line = " << __LINE__ << " - "
                              << "process tag = " << ttag << std::endl;
                            abort();
                         }
                      }
                      else
                      {
                         THROW_LOGICAL_ERROR3( "No message hander for ", ttag, " is not registered" );
                      }
                     if( newTask )
                     {
                        terminationState = CompTerminatedByAnotherTask;
                        break;
                     }
                     if( !anotherNodeIsRequested ) break;
                  }
               }
               if( dynamic_cast<BbParaNode *>(currentTask) )   // may not be BbParaNode
               {
                  if( dynamic_cast<BbParaNode *>(currentTask)->getMergingStatus() == 3 )
                  {
                     terminationState = CompInterruptedInMerging;
                  }
               }
            }

            if( isRacingStage() &&
                  !newTask && !racingWinner)
            {
               if( !racingIsInterrupted && wasTerminatedNormally() )
               {
                  // assert(getNNodesLeft() == 0);   // hard time limit case, this happned.
                  /* CompTerminatedInRacingStage means computation is finished, so terminates all solvers */
                  terminationState = CompTerminatedInRacingStage;
               }
               else
               {
                  /* CompInterruptedInRacingStage means computation is interrupted by termination of the other solvers */
                  terminationState = CompInterruptedInRacingStage;   // even if a racing solver terminated badly, just keep running.
                  racingInterruptIsRequested = false;
                  racingIsInterrupted = false;
               }

            }
            else
            {
               if( !wasTerminatedNormally() )
               {
                  THROW_LOGICAL_ERROR3( "BbParaSolver", paraComm->getRank(), " was terminated abnormally." );
               }
            }
         }
      }

   }

   double averageSimplexIter = 0.0;

   if( nSolved <= 1 )
   {
      /** nSolved > 1 is set within callback routine */
      setRootNodeTime();
      setRootNodeSimplexIter(getSimplexIter());
   }
   else
   {
      averageSimplexIter = static_cast<double>( ( getSimplexIter() - nSimplexIterRoot )/(nSolved - 1) );
   }

   DEF_BB_PARA_COMM(bbParaComm, paraComm);

   if( terminationState == CompTerminatedByInterruptRequest
         && nSolved <= 1    // current solving node is interrupted
         && collectingManyNodes         // collect all nodes
         && nCollectOnce == -1          // collect all nodes
         )
   {
      if( dynamic_cast<BbParaNode *>(currentTask) )   // may not be BbParaNode
      {
         dynamic_cast<BbParaNode *>(currentTask)->send(bbParaComm, 0);    // current solving node has to be collected
      }
   }

   if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
   {
      solverDualBound = std::min(solverDualBound, selfSplitNodePool->getBestDualBoundValue() );
   }

   BbParaCalculationState *paraCalculationState = dynamic_cast<BbParaCalculationState *>(bbParaComm->createParaCalculationState(
         compTime, rootNodeTime, nSolved,nSent, nImprovedIncumbent, terminationState, nSolvedWithNoPreprocesses,
         nSimplexIterRoot, averageSimplexIter,
         nTransferredLocalCuts, minTransferredLocalCuts, maxTransferredLocalCuts,
         nTransferredBendersCuts, minTransferredBendersCuts, maxTransferredBendersCuts,
         getNRestarts(), minIisum, maxIisum, minNii, maxNii, solverDualBound, nSelfSplitNodesLeft ));
   paraCalculationState->send(paraComm, 0, tag);
   delete paraCalculationState;

   /*******************
    * update counters *
    *******************/
   if( nSolved < minNSolved )
   {
      minNSolved = nSolved;
   }
   if( nSolved > maxNSolved )
   {
      maxNSolved = nSolved;
   }
   totalNSolved += nSolved;

   totalNSent += nSent;
   totalNImprovedIncumbent += nImprovedIncumbent;
   nParaTasksSolved++;
   if( nSolved == 1)
   {
      nParaNodesSolvedAtRoot++;
   }

   nTransferredLocalCutsFromSolver += nTransferredLocalCuts;
   if( minTransferredLocalCutsFromSolver > minTransferredLocalCuts  )
   {
      minTransferredLocalCutsFromSolver = minTransferredLocalCuts;
   }
   if( maxTransferredLocalCutsFromSolver < maxTransferredLocalCuts  )
   {
      maxTransferredLocalCutsFromSolver = maxTransferredLocalCuts;
   }

   nTransferredBendersCutsFromSolver += nTransferredBendersCuts;
   if( minTransferredBendersCutsFromSolver > minTransferredBendersCuts  )
   {
      minTransferredBendersCutsFromSolver = minTransferredBendersCuts;
   }
   if( maxTransferredBendersCutsFromSolver < maxTransferredBendersCuts  )
   {
      maxTransferredBendersCutsFromSolver = maxTransferredBendersCuts;
   }

   nTotalRestarts += getNRestarts();
   if( minRestarts > getNRestarts() )
   {
      minRestarts = getNRestarts();
   }
   if( maxRestarts < getNRestarts() )
   {
      maxRestarts = getNRestarts();
   }

   nSolved = 0;
   nSent = 0;
   nImprovedIncumbent = 0;
   nSolvedWithNoPreprocesses = 0;
   nTransferredLocalCuts = 0;
   minTransferredLocalCuts = INT_MAX;
   maxTransferredLocalCuts = INT_MIN;
   nTransferredBendersCuts = 0;
   minTransferredBendersCuts = INT_MAX;
   maxTransferredBendersCuts = INT_MIN;
   /**********************************
   * accumulate total root node time *
   ***********************************/
   totalRootNodeTime += rootNodeTime;
   rootNodeTime = 0.0;

   minIisum = DBL_MAX;
   maxIisum = 0.0;
   minNii = INT_MAX;
   maxNii = 0;

   double detTime = -1.0;
   if( paraParams->getBoolParamValue(Deterministic)  )
   {
      detTime = paraDetTimer->getElapsedTime();
   }
   double stopTime = paraTimer->getElapsedTime();
   if( isRacingStage() )
   {
//      if( newNode )
//      {
//         nParaNodesReceived--;
//      }

      // if( !keepRacing )
      // {
         /** Transfer SolverTermination state during racing ramp-up */
         BbParaSolverTerminationState *paraSolverTerminationState = dynamic_cast<BbParaSolverTerminationState *>(bbParaComm->createParaSolverTerminationState(
            3,    /** interupted flag == 3 means the information for racing ramp-up */
            paraComm->getRank(),
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
            getNTightened(),
            getNTightenedInt(),
            terminationState,
            stopTime,
            idleTimeToFirstParaTask,
            idleTimeBetweenParaTasks,
            0.0,
            idleTimeToWaitNotificationId,
            idleTimeToWaitAckCompletion,
            idleTimeToWaitToken,
            totalRootNodeTime,
            minRootNodeTime,
            maxRootNodeTime,
            detTime
            ));
         assert( tag == TagCompletionOfCalculation );
         paraSolverTerminationState->send(paraComm, 0, TagTermStateForInterruption);
         delete paraSolverTerminationState;
         assert(! racingWinner);
         /** re-initialize all counters, winner counts on current counters */
         minNSolved = INT_MAX;
         maxNSolved = INT_MIN;
         totalNSolved = 0;
         totalNSent = 0;
         totalNImprovedIncumbent = 0;
//         if( newNode )
//            nParaNodesReceived = 1;
//         else
//            nParaNodesReceived = 0;
         nParaTasksSolved = 0;
         nParaNodesSolvedAtRoot = 0;
         nParaNodesSolvedAtPreCheck = 0;
         totalRootNodeTime = 0.0;
         minRootNodeTime = DBL_MAX;
         maxRootNodeTime = -DBL_MAX;

         if( restartingRacing || keepRacing )
         {
            restartRacing();
         }
         else
         {
            terminateRacing();
         }

      // }
   }
   else
   {
      /** Transfer SolverTermination state to save statistic information for checkpoint */
      BbParaSolverTerminationState *paraSolverTerminationState = dynamic_cast<BbParaSolverTerminationState *>(bbParaComm->createParaSolverTerminationState(
    		2,    /** interupted flag == 2 means the information for checkpoint */
         paraComm->getRank(),
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
         getNTightened(),
         getNTightenedInt(),
         terminationState,
         stopTime,
         idleTimeToFirstParaTask,
         idleTimeBetweenParaTasks,
         0.0,
         idleTimeToWaitNotificationId,
         idleTimeToWaitAckCompletion,
         idleTimeToWaitToken,
         totalRootNodeTime,
         minRootNodeTime,
         maxRootNodeTime,
         detTime
         ));
      if( tag == TagCompletionOfCalculation )
      {
         paraSolverTerminationState->send(paraComm, 0, TagTermStateForInterruption);
      }
      else
      {
         paraSolverTerminationState->send(paraComm, 0, TagSelfSplitTermStateForInterruption);
      }
      delete paraSolverTerminationState;
   }
   nSendInCollectingMode = 0;
   if( needToResetMaximalDualBound ) solverDualBound = -DBL_MAX;
   givenGapIsReached = false;
   // keepRacing = false;
}

void
BbParaSolver::sendCompletionOfCalculationWithoutSolving(
      double compTime,
      int tag,                 ///< message Tag
      int nSelfSplitNodesLeft  ///< number of self-split nodes left
      )
{
   int terminationState = CompTerminatedNormally;
   bool needToResetMaximalDualBound = racingIsInterrupted;

   double dualBound = DBL_MAX;

   if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
   {
      dualBound = std::min(solverDualBound, selfSplitNodePool->getBestDualBoundValue() );
   }

   DEF_BB_PARA_COMM(bbParaComm, paraComm);

   BbParaCalculationState *paraCalculationState = dynamic_cast<BbParaCalculationState *>(bbParaComm->createParaCalculationState(
         compTime, 0, 0 , 0, 0, terminationState, 0,
         0, 0.0,
         0, 0, 0,
         0, 0, 0,
         0, 0, 0, 0, 0, dualBound, nSelfSplitNodesLeft ));
   paraCalculationState->send(paraComm, 0, tag);
   delete paraCalculationState;

   /**********************
    * No update counters *
    **********************/

   double detTime = -1.0;
   if( paraParams->getBoolParamValue(Deterministic)  )
   {
      detTime = paraDetTimer->getElapsedTime();
   }
   double stopTime = paraTimer->getElapsedTime();
   /** Transfer SolverTermination state to save statistic information for checkpoint */
   BbParaSolverTerminationState *paraSolverTerminationState = dynamic_cast<BbParaSolverTerminationState *>(bbParaComm->createParaSolverTerminationState(
      2,    /** interupted flag == 2 means the information for checkpoint */
      paraComm->getRank(),
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
      getNTightened(),
      getNTightenedInt(),
      terminationState,
      stopTime,
      idleTimeToFirstParaTask,
      idleTimeBetweenParaTasks,
      0.0,
      idleTimeToWaitNotificationId,
      idleTimeToWaitAckCompletion,
      idleTimeToWaitToken,
      totalRootNodeTime,
      minRootNodeTime,
      maxRootNodeTime,
      detTime
      ));
      paraSolverTerminationState->send(paraComm, 0, TagSelfSplitTermStateForInterruption);
   delete paraSolverTerminationState;
   nSendInCollectingMode = 0;
   if( needToResetMaximalDualBound ) solverDualBound = -DBL_MAX;
   givenGapIsReached = false;
   // keepRacing = false;
}


void
BbParaSolver::sendLocalSolution(
      )
{
   if( localIncumbentSolution && (!notificationProcessed) ) // if solution is sent in notification is processed,
                                                            //  dead lock may be occurred depending of MPI system buffer size
   {
      if( !globalBestIncumbentSolution )
      {
         localIncumbentSolution->send(paraComm, 0);
         globalBestIncumbentSolution = localIncumbentSolution;
      }
      else
      {
         if( EPSLT(localIncumbentSolution->getObjectiveFunctionValue(), globalBestIncumbentSolution->getObjectiveFunctionValue(), eps) )
            // NOTE: globalBestIncumbnetValue may be an objective function value of localIncumbentSolution
         {
            localIncumbentSolution->send(paraComm, 0);
            delete globalBestIncumbentSolution;
            globalBestIncumbentSolution = localIncumbentSolution;
         }
         else
         {
            delete localIncumbentSolution;
         }
      }
      localIncumbentSolution = 0;
   }
}

bool
BbParaSolver::notificationIsNecessary(
      )
{
   // if( paraParams->getBoolParamValue(Deterministic) )
   // {
   //  return true;  // always true, in case of deterministic run
   // }

   if( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
                     paraTimer->getElapsedTime() > ( paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) + paraParams->getRealParamValue(NotificationInterval) + 3.0 ) )  // +3.0 sec. is to issue interupt request
   {
      paraParams->setRealParamValue(NotificationInterval, paraParams->getRealParamValue(NotificationInterval)*1000);
   }

   if( !rampUp )
   {
      if( isRacingStage() )
      {
         if( lcBestDualBoundValue < getDualBoundValue() )
         {
            return true;
         }
         else
         {
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               if( ( paraDetTimer->getElapsedTime() -  previousNotificationTime )
                  > paraParams->getRealParamValue(NotificationInterval) )
               {
                  return true;
               }
               else
               {
                  return false;
               }
            }
            else
            {
               if( ( paraTimer->getElapsedTime() -  previousNotificationTime )
                  > paraParams->getRealParamValue(NotificationInterval) )
               {
                  return true;
               }
               else
               {
                  return false;
               }
            }
         }
      }
      else
      {  // normal ramp-up phase
         return true;
      }
   }
   else
   {
      if( collectingMode || collectingManyNodes )
      {
         if( noWaitModeSend )
         {
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               if(  ( paraDetTimer->getElapsedTime() -  previousNotificationTime )
                     > paraParams->getRealParamValue(NotificationInterval) )
               {
                  return true;
               }
               else
               {
                  return false;
               }
            }
            else
            {
               if( ( paraTimer->getElapsedTime() -  previousNotificationTime )
                     > paraParams->getRealParamValue(NotificationInterval) )
               {
                  return true;
               }
               else
               {
                  return false;
               }
            }
         }
         else
         {
            return true;
         }
      }
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         if(  ( paraDetTimer->getElapsedTime() -  previousNotificationTime )
               > paraParams->getRealParamValue(NotificationInterval) )
         {
            return true;
         }
         else
         {
            return false;
         }
      }
      else
      {
         if( ( paraTimer->getElapsedTime() -  previousNotificationTime )
               > paraParams->getRealParamValue(NotificationInterval) )
         {
            return true;
         }
         else
         {
            return false;
         }
      }
   }
}

void
BbParaSolver::sendSolverState(
      long long nNodesSolved,
      int nNodesLeft,
      double bestDualBoundValue,
      double detTime
      )
{
   if(!notificationProcessed)
   {
      int racingStage = 0;   /** assume not racing stage */
      if( isRacingStage() )
      {
         racingStage = 1;
      }
      double tempGlobalBestPrimalBound = DBL_MAX;
      if( globalBestIncumbentSolution )
      {
         tempGlobalBestPrimalBound = globalBestIncumbentSolution->getObjectiveFunctionValue();
      }

      DEF_BB_PARA_COMM(bbParaComm, paraComm);
      double dualBound = std::min( std::max( bestDualBoundValue,dynamic_cast<BbParaNode *>(currentTask)->getDualBoundValue()), tempGlobalBestPrimalBound );
      if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) )
      {
         nNodesLeft += selfSplitNodePool->getNumOfNodes();
         dualBound = std::min(dualBound,  selfSplitNodePool->getBestDualBoundValue());
      }
      BbParaSolverState *solverState = dynamic_cast<BbParaSolverState *>(bbParaComm->createParaSolverState(
            racingStage,
            ++notificationIdGenerator,
            currentTask->getLcId(), currentTask->getGlobalSubtaskIdInLc(),
            nNodesSolved, nNodesLeft, dualBound,
            tempGlobalBestPrimalBound,
            detTime,
            averageDualBoundGain
      ));
      solverState->send(paraComm, 0, TagSolverState);
      delete solverState;
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         previousNotificationTime = paraDetTimer->getElapsedTime();
      }
      else
      {
         previousNotificationTime = paraTimer->getElapsedTime();
      }
      notificationProcessed = true;
   }
}

int
BbParaSolver::getThresholdValue(
      int nNodes      /**< number of processed nodes, including the focus node */
      )
{
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      return 10;
   }

   double compTime = paraTimer->getElapsedTime() - (idleTimeToFirstParaTask + idleTimeBetweenParaTasks);
   double meanRootNodeTime = ( totalRootNodeTime + rootNodeTime )/(nParaTasksSolved + 1);  // +1 is this ParaNode
   double meanNodeTime = -1.0;             // unrealiable
   if( ( compTime - (totalRootNodeTime + rootNodeTime ) ) > 10.0  // less equal than 10.0 sec. treats as unreliable
         && meanRootNodeTime > 0.0001 ) // meanRootNode time less equal than 0.0001 sec. treats as unreliable
   {
      meanNodeTime = ( compTime - (totalRootNodeTime + rootNodeTime ) )  / (totalNSolved + nNodes - nParaTasksSolved);
                                                                      // compute mean node comp. time except root nodes(nParaNodesSolved+1) and the focus node(-1)
      if( meanNodeTime < 0.0 ) meanNodeTime = 0.0;
   }

   int n;
   if( meanNodeTime < 0.000001 )
   {   // heuristic initial value setting
      n = ( rootNodeTime < 0.01 ) ? 100 : ( ( rootNodeTime < 0.1 ) ? 50 : ( (rootNodeTime < 1.0) ? 10 : 5 ) );
   }
   else
   {
      if( ( meanRootNodeTime / meanNodeTime ) > 5.0 )
      {  // base value at least 5.0
         n = (int) (
               paraParams->getRealParamValue(MultiplierToDetermineThresholdValue)
               * ( meanRootNodeTime / meanNodeTime ) );
         if( n > 100 )
         {
             if( meanNodeTime > 1.0 )
             {
                n = 3;
             }
             else
             {
                n = 100;
             }
         }
      }
      else
      {   // heuristic value setting
         n = ( rootNodeTime < 0.01 ) ? 100 : ( ( rootNodeTime < 0.1 ) ? 50 : ( (rootNodeTime < 1.0) ? 10 : 5 ) );
      }
   }
   n = n * paraParams->getRealParamValue(NoTransferThresholdReductionRatio);
   if( n < 2 ) n = 2;
   return n;
}

void
BbParaSolver::sendParaNode(
      long long n,
      int depth,
      double dualBound,
      double estimateValue,
      ParaDiffSubproblem *diffSubproblem
      )
{
   DEF_BB_PARA_COMM(bbParaComm, paraComm );

   BbParaNode *node = dynamic_cast<BbParaNode *>(bbParaComm->createParaNode(
         TaskId( SubtaskId(currentTask->taskId.subtaskId.lcId,
                           currentTask->taskId.subtaskId.globalSubtaskIdInLc,
                           paraComm->getRank() ),
                           -1),      // sequentail number is not set
         TaskId( currentTask->taskId.subtaskId, n),
         dynamic_cast<BbParaNode *>(currentTask)->getDepth() + depth,
         std::max( dualBound, dynamic_cast<BbParaNode *>(currentTask)->getDualBoundValue() ),
         dualBound,
         estimateValue,
         diffSubproblem
         ));
   double inIdleTime = paraTimer->getElapsedTime();
   node->send(paraComm, 0);
   delete node;
   int tag;
   int status;
   if( nSendInCollectingMode <= 0 || !noWaitModeSend )
   {
      waitingSpecificMessage = true;
      for(;;)
      {
         paraComm->waitSpecTagFromSpecSource(0, TagAny, &tag);
         idleTimeToWaitNotificationId += ( paraTimer->getElapsedTime() - inIdleTime );
         if( tag == TagTaskReceived )
         {
            break;
         }
         status = (this->*messageHandler[tag])(0, tag);
         if( status )
         {
            std::ostringstream s;
            s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
              << __func__ << ", line = " << __LINE__ << " - "
              << "process tag = " << tag << std::endl;
            abort();
         }
         inIdleTime = paraTimer->getElapsedTime();
      }
      idleTimeToWaitNotificationId += ( paraTimer->getElapsedTime() - inIdleTime );
      //
      // receive the NULL message for TagNodeReceived
      //
      status = (this->*messageHandler[tag])(0, tag);
      if( status )
      {
         std::ostringstream s;
         s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
           << __func__ << ", line = " << __LINE__ << " - "
           << "process tag = " << tag << std::endl;
         abort();
      }
      waitingSpecificMessage = false;
   }
   nSent++;
   if( nSendInCollectingMode > 0 ) nSendInCollectingMode--;
   if( nCollectOnce > 0 ) nCollectOnce--;
   if( aggressiveCollecting && nSendInCollectingMode == 0 ) aggressiveCollecting = false;
   if( collectingManyNodes && nCollectOnce == 0 ) collectingManyNodes = false;
   if( isBreaking() ) nTransferredNodes++;
}

void
BbParaSolver::keepParaNode(
      long long n,
      int depth,
      double dualBound,
      double estimateValue,
      ParaDiffSubproblem *diffSubproblem
      )
{
   DEF_BB_PARA_COMM(bbParaComm, paraComm );

   BbParaNode *node = dynamic_cast<BbParaNode *>(bbParaComm->createParaNode(
         TaskId( SubtaskId(currentTask->taskId.subtaskId.lcId,
                           currentTask->taskId.subtaskId.globalSubtaskIdInLc,
                           paraComm->getRank() ),
                           -1),      // sequentail number is not set
         TaskId( currentTask->taskId.subtaskId, n),
         dynamic_cast<BbParaNode *>(currentTask)->getDepth() + depth,
         std::max( dualBound, dynamic_cast<BbParaNode *>(currentTask)->getDualBoundValue() ),
         dualBound,
         estimateValue,
         diffSubproblem
         ));
   // double inIdleTime = paraTimer->getElapsedTime();
   node->sendNewSubtreeRoot(paraComm, 0);
   selfSplitNodePool->insert(node);
}

void
BbParaSolver::sendAnotherNodeRequest(
      double bestDualBoundValue
      )
{
    if( anotherNodeIsRequested ) return;
    if( selfSplitNodePool && (!selfSplitNodePool->isEmpty()) ) return;
    if( givenGapIsReached ) return;
    PARA_COMM_CALL(
          paraComm->send( &bestDualBoundValue, 1, ParaDOUBLE, 0, TagAnotherNodeRequest)
          );
    anotherNodeIsRequested = true;

}

bool
BbParaSolver::updateGlobalBestCutOffValue(
      double newValue
      )
{
    if( newValue < globalBestCutOffValue ){
        globalBestCutOffValue = newValue;
        return true;
    } else {
        return false;
    }
}

void
BbParaSolver::waitMessageIfNecessary(
      )
{
   if( paraParams->getIntParamValue(NotificationSynchronization) == 0 ||
          !rampUp ||
         ( paraParams->getIntParamValue(NotificationSynchronization) == 1 && ( collectingMode || collectingManyNodes ) ) )
   {
      waitNotificationIdMessage();
   }
}

void
BbParaSolver::restartRacing(
      )
{
   assert( !paraInstance );

   freeSubproblem();
   subproblemFreed = true;

   // paraInstance = paraComm->createParaInstance();
   // paraInstance->bcast(paraComm, 0, paraParams->getIntParamValue(InstanceTransferMethod));
   reinitialize();

   if( !keepRacing )
   {
      assert(newTask == 0);
      newTask = paraComm->createParaTask();
      PARA_COMM_CALL(
            newTask->bcast(paraComm, 0)
            );

      nParaTasksReceived++;
   }
   racingIsInterrupted = false;
   restartingRacing = false;
   terminationMode = NoTerminationMode;
}

/*** sendIfImprovedSolutionWasFound routine should be removed in the future **/
bool
BbParaSolver::sendIfImprovedSolutionWasFound(
      ParaSolution *sol
      )
{
   if( !globalBestIncumbentSolution ||
       ( globalBestIncumbentSolution
         && EPSLT( sol->getObjectiveFunctionValue(), globalBestIncumbentSolution->getObjectiveFunctionValue(), eps ) ) )
   {
      globalBestIncumbentValue = sol->getObjectiveFunctionValue();
      globalIncumbnetValueUpdateFlag = true;
      assert( localIncumbentSolution == 0 );
      sol->send(paraComm, 0);
      delete sol;
      //
      // shoul not do as follows. LC does not know which nodes were removed
      //
      // if( selfSplitNodePool )
      // {
      //    selfSplitNodePool->removeBoundedNodes(globalBestIncumbentValue);
      // }
      nImprovedIncumbent++;
      return true;
   }
   else
   {
      delete sol;
      return false;
   }
}

bool
BbParaSolver::saveIfImprovedSolutionWasFound(
      ParaSolution *sol
      )
{
   if( EPSLT( sol->getObjectiveFunctionValue(), globalBestIncumbentValue, eps ) )  // compare to globalBestIncumbentValue
   {                                                                              // no solution sending is accepted!!!
      // globalBestIncumbentValue = sol->getObjectiveFuntionValue(); // should not update globalBestIncumbentValue and flag
      // globalIncumbnetValueUpdateFlag = true;                      // thease are updated only when the value comes from outside
      if( localIncumbentSolution  )
      {
         if( EPSLT( sol->getObjectiveFunctionValue(), localIncumbentSolution->getObjectiveFunctionValue(), eps ) )
         {
            // there is a possibility to be updated several times
            delete localIncumbentSolution;
            localIncumbentSolution = sol;
            nImprovedIncumbent++;
            localIncumbentIsChecked = false;
            return true;
         }
         else
         {
            if( localIncumbentIsChecked )
            {
               delete localIncumbentSolution;
               localIncumbentSolution = 0;
               localIncumbentIsChecked = false;
               return false;
            }
	    else
	    {
               // could check an identical solution 
               localIncumbentIsChecked = true;
	       return true;
	    }
         }
      }
      else
      {
         localIncumbentSolution = sol;
         nImprovedIncumbent++;
         localIncumbentIsChecked = false;
         return true;
      }
   }
   else
   {
      delete sol;
      return false;
   }
}

bool
BbParaSolver::updateGlobalBestIncumbentValue(
      double newValue
      )
{
    if( newValue < globalBestIncumbentValue ){
        globalBestIncumbentValue = newValue;
        globalIncumbnetValueUpdateFlag = true;
        return true;
    } else {
        return false;
    }
}

bool
BbParaSolver::updateGlobalBestIncumbentSolution(
      ParaSolution *sol
      )
{
   if( globalBestIncumbentSolution  )
   {
      if( EPSLE(sol->getObjectiveFunctionValue(), globalBestIncumbentSolution->getObjectiveFunctionValue(),DEFAULT_NUM_EPSILON) )
      {
         if( sol != globalBestIncumbentSolution )  // can have the same pointer address
         {
            delete globalBestIncumbentSolution;
            globalBestIncumbentSolution = sol;
         }
         if( EPSLE(sol->getObjectiveFunctionValue(), globalBestIncumbentValue, DEFAULT_NUM_EPSILON)  )
         {
            globalBestIncumbentValue = sol->getObjectiveFunctionValue();
            globalIncumbnetValueUpdateFlag = true;
         }
         return true;
      }
      else
      {
         return false;
      }
   }
   else
   {
      globalBestIncumbentSolution = sol;
      if( EPSLE(sol->getObjectiveFunctionValue(), globalBestIncumbentValue, DEFAULT_NUM_EPSILON) )
      {
         globalBestIncumbentValue = sol->getObjectiveFunctionValue();
         globalIncumbnetValueUpdateFlag = true;
         // std::cout <<  "R." << paraComm->getRank() << ", pendingIncumbentValue = " << pendingIncumbentValue << std::endl;
      }
      return true;
   }
}

void
BbParaSolver::waitNotificationIdMessage(
      )
{
   double inIdleTime = paraTimer->getElapsedTime();
   int tag;
   int status;
   waitingSpecificMessage = true;
   for(;;)
   {
      paraComm->waitSpecTagFromSpecSource(0, TagAny, &tag);
      idleTimeToWaitNotificationId += ( paraTimer->getElapsedTime() - inIdleTime );
      if( tag == TagNotificationId )
      {
         break;
      }
      status = (this->*messageHandler[tag])(0, tag);
      if( status )
      {
         std::ostringstream s;
         s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
           << __func__ << ", line = " << __LINE__ << " - "
           << "process tag = " << tag << std::endl;
         abort();
      }
      inIdleTime = paraTimer->getElapsedTime();
   }
   idleTimeToWaitNotificationId += ( paraTimer->getElapsedTime() - inIdleTime );
   //
   // receive the notification Id message
   //
   // status = (this->*messageHandler[tag])(0, tag);
   status = processTagNotificationId(0,tag);
   if( status )
   {
      std::ostringstream s;
      s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
        << __func__ << ", line = " << __LINE__ << " - "
        << "process tag = " << tag << std::endl;
      abort();
   }
   waitingSpecificMessage = false;
}

void
BbParaSolver::waitAckCompletion(
      )
{
   double inIdleTime = paraTimer->getElapsedTime();
   int tag;
   int hstatus;
   waitingSpecificMessage = true;
   for(;;)
   {
      paraComm->waitSpecTagFromSpecSource(0, TagAny, &tag);
      idleTimeToWaitAckCompletion += ( paraTimer->getElapsedTime() - inIdleTime );
      if( tag == TagAckCompletion )
      {
         break;
      }
      hstatus = (this->*messageHandler[tag])(0, tag);
      if( hstatus )
      {
         std::ostringstream s;
         s << "[ERROR RETURN form Message Hander]:" <<  __FILE__ <<  "] func = "
           << __func__ << ", line = " << __LINE__ << " - "
           << "process tag = " << tag << std::endl;
         abort();
      }
      if( terminationMode == NormalTerminationMode ) break;
      inIdleTime = paraTimer->getElapsedTime();
   }
   double current = paraTimer->getElapsedTime();
   double idleTime = current - inIdleTime;
   idleTimeToWaitAckCompletion += idleTime;
   if( paraParams->getBoolParamValue(DynamicAdjustNotificationInterval) &&
         current < paraParams->getRealParamValue(CheckpointInterval) &&
         idleTime > 1.0 )
   {
      paraParams->setRealParamValue(NotificationInterval,
            (paraParams->getRealParamValue(NotificationInterval)+idleTime) );
   }
   if( tag == TagAckCompletion )
   {
      //
      // receive the notification Id message
      //
      PARA_COMM_CALL(
            paraComm->receive( NULL, 0, ParaBYTE, 0, TagAckCompletion)
            );
   }
   waitingSpecificMessage = false;
}

