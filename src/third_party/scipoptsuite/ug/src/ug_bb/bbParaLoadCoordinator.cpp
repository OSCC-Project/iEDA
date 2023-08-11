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

/**@file    paraLoadCoordinator.cpp
 * @brief   Load coordinator.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifdef _MSC_VER
#include <functional>
#else
#include <unistd.h>
#endif
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <climits>
#include <algorithm>
#include <iomanip>

#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "ug/paraInitialStat.h"
#include "bbParaLoadCoordinator.h"
#include "bbParaNode.h"
#include "bbParaNodesMerger.h"

using namespace UG;

BbParaLoadCoordinator::BbParaLoadCoordinator(
#ifdef UG_WITH_UGS
      UGS::UgsParaCommMpi *inCommUgs,
#endif
      int inNHandlers,
      ParaComm *inComm,
      ParaParamSet *inParaParamSet,
      ParaInitiator *inParaInitiator,
      bool *inRacingSolversExist,
      ParaTimer *inParaTimer,
      ParaDeterministicTimer *inParaDetTimer
      )
      : ParaLoadCoordinator(
#ifdef UG_WITH_UGS
            inCommUgs,
#endif
            inNHandlers,
            inComm,
            inParaParamSet,
            inParaInitiator,
            inRacingSolversExist,
            inParaTimer,
            inParaDetTimer
            ),
        initialNodesGenerated(false),
        firstCollectingModeState(-1),
        isCollectingModeRestarted(false),
        breakingSolverId(-1),
        nReplaceToBetterNode(0),
        nNormalSelection(-1),
        winnerSolverNodesCollected(false),
        primalUpdated(false),
        restartingRacing(false),
        nRestartedRacing(0),
        statEmptyNodePoolTime(DBL_MAX),
        minDepthInWinnerSolverNodes(INT_MAX),
        maxDepthInWinnerSolverNodes(-1),
        merging(false),
        nBoundChangesOfBestNode(0),
//        varIndexTable(0),
//        mergeInfoHead(0),
//        mergeInfoTail(0),
        nodesMerger(0),
        nCollectedSolvers(0),
        previousTabularOutputTime(0.0),
        averageDualBoundGain(0.0),
        nAverageDualBoundGain(0),
        averageLastSeveralDualBoundGains(0.0),
        starvingTime(-1.0),
        hugeImbalanceTime(-1.0),
        paraNodePoolToRestart(0),
        paraNodePoolBufferToRestart(0),
        paraNodePoolBufferToGenerateCPF(0),
        paraNodeToKeepCheckpointFileNodes(0),
        unprocessedParaNodes(0),
        selfSplitFinisedSolvers(0),
        osTabularSolvingStatus(0),
        osLogSubtreeInfo(0),
        allCompInfeasibleAfterSolution(false),
        minmalDualBoundNormalTermSolvers(DBL_MAX),
        warmStartNodeTransferring(false),
        hugeImbalance(false),
        isHeaderPrinted(false),
        givenGapIsReached(false),
        pendingSolution(0)
{

   BbMessageHandlerFunctionPointer *bbMessageHandler = reinterpret_cast<BbMessageHandlerFunctionPointer *>(messageHandler);

   bbMessageHandler[TagAnotherNodeRequest] = &UG::BbParaLoadCoordinator::processTagAnotherNodeRequest;
   bbMessageHandler[TagAllowToBeInCollectingMode] = &UG::BbParaLoadCoordinator::processTagAllowToBeInCollectingMode;
   bbMessageHandler[TagTermStateForInterruption] = &UG::BbParaLoadCoordinator::processTagTermStateForInterruption;
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      bbMessageHandler[TagLbBoundTightenedIndex] = &UG::BbParaLoadCoordinator::processTagLbBoundTightened;
      bbMessageHandler[TagUbBoundTightenedIndex] = &UG::BbParaLoadCoordinator::processTagUbBoundTightened;
   }
   
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 3 )  // Self-Split ramp-up
   {
      bbMessageHandler[TagSelfSplitFinished] = &UG::BbParaLoadCoordinator::processTagSelfSplitFinished;
      bbMessageHandler[TagNewSubtreeRootNode] = &UG::BbParaLoadCoordinator::processTagNewSubtreeRootNode;
      bbMessageHandler[TagSubtreeRootNodeStartComputation] = &UG::BbParaLoadCoordinator::processTagSubtreeRootNodeStartComputation;
      bbMessageHandler[TagSubtreeRootNodeToBeRemoved] = &UG::BbParaLoadCoordinator::processTagSubtreeRootNodeToBeRemoved;
      bbMessageHandler[TagReassignSelfSplitSubtreeRootNode] = &UG::BbParaLoadCoordinator::processTagReassignSelfSplitSubtreeRootNode;
      bbMessageHandler[TagSelfSlpitNodeCalcuationState] = &UG::BbParaLoadCoordinator::processTagSelfSlpitNodeCalcuationState;
      bbMessageHandler[TagSelfSplitTermStateForInterruption] = &UG::BbParaLoadCoordinator::processTagSelfSplitTermStateForInterruption;
      selfSplitFinisedSolvers = new std::set<int>;
   }

   if(  paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) )
   {
      if( paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) < 1.0 )
      {
         std::cout << "EnhancedCheckpointStartTime mast be greater than 1.0. " << std::endl;
         std::cout << "EnhancedCheckpointStartTime = " <<  paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) << std::endl;
         exit(1);
      }
      if( paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) >= paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) )
      {
         std::cout << "EnhancedCheckpointStartTime mast be less than FinalCheckpointGeneratingTime. " << std::endl;
         std::cout << "EnhancedCheckpointStartTime = " <<  paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) << std::endl;
         std::cout << "FinalCheckpointGeneratingTime = " << paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) << std::endl;
         exit(1);
      }
      if( paraParams->getIntParamValue(UG::FinalCheckpointNSolvers) > (paraComm->getSize() - 1) )
      {
         std::cout << "FinalCheckpointNSolvers mast be less equal than the number of solvers. " << std::endl;
         std::cout << "FinalCheckpointNSolvers = " << paraParams->getIntParamValue(UG::FinalCheckpointNSolvers) << std::endl;
         std::cout << "The number of solvers = " << (paraComm->getSize() - 1) << std::endl;
         exit(1);
      }
   }
   if ( paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 )
   {
      paraParams->setRealParamValue(UG::TimeLimit, -1.0);
      std::cout << "** FinalCheckpointGeneratingTime is specified, then TimeLimit is omitted. ** " << std::endl;
   }

   // disply ramp-up mode
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 0 )
   {
	   std::cout <<"LC is working with normal ramp-up." << std::endl;
   }
   else if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 )
   {
	   std::cout <<"LC is working with racing ramp-up." << std::endl;
   }
   else if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 )
   {
	   std::cout <<"LC is working with racing ramp-up and with rebuilding tree after racing." << std::endl;
   }
   else if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 3 )
   {
      std::cout <<"LC is working with self-split ramp-up." << std::endl;
   }

   /* if initial solution is given, output the primal value */
   if( logSolvingStatusFlag && dynamic_cast<BbParaInitiator *>(paraInitiator)->getGlobalBestIncumbentSolution() )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " LC" << " INITIAL_PRIMAL_VALUE "
      << dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(
            dynamic_cast<BbParaSolution *>(dynamic_cast<BbParaInitiator *>(paraInitiator)->getGlobalBestIncumbentSolution())->getObjectiveFunctionValue()
            ) << std::endl;
   }

   logSubtreeInfoFlag = paraParams->getBoolParamValue(LogSubtreeInfo);
   if( logSubtreeInfoFlag )
   {
      std::ostringstream s;
#ifdef UG_WITH_UGS
      if( commUgs )
      {
         s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << commUgs->getMySolverName() << "_"
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".treelog";
      }
      else
      {
         s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".treelog";
      }
#else
      s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
      << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".treelog";
#endif
      ofsLogSubtreeInfo.open(s.str().c_str(), std::ios::app );
      if( !ofsLogSubtreeInfo )
      {
         std::cout << "Sub tree info. log file cannot open : file name = " << s.str() << std::endl;
         exit(1);
      }
      osLogSubtreeInfo = &ofsLogSubtreeInfo;
   }

   outputTabularSolvingStatusFlag = paraParams->getBoolParamValue(OutputTabularSolvingStatus);
   if( outputTabularSolvingStatusFlag || paraParams->getBoolParamValue(Quiet) )
   {
      if( paraParams->getBoolParamValue(Quiet) )
      {
         osTabularSolvingStatus = &std::cout;
      }
      else
      {
         std::ostringstream s;
#ifdef UG_WITH_UGS
         if( commUgs )
         {
            s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
            << commUgs->getMySolverName() << "_"
            << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << "_T.status";
         }
         else
         {
            s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
            << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << "_T.status";
         }
#else
         s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << "_T.status";
#endif
         ofsTabularSolvingStatus.open(s.str().c_str(), std::ios::app );
         if( !ofsTabularSolvingStatus )
         {
            std::cout << "Tabular solving status file cannot open : file name = " << s.str() << std::endl;
            exit(1);
         }
         osTabularSolvingStatus = &ofsTabularSolvingStatus;
      }
//      if( outputTabularSolvingStatusFlag )
//      {
//         outputTabularSolvingStatusHeader();            /// should not call virutal function in constructor
//      }
   }

   paraSolverPool = new BbParaSolverPoolForMinimization(    // always minimization problem
                    paraParams->getRealParamValue(MultiplierForCollectingMode),
                    paraParams->getRealParamValue(BgapCollectingMode),
                    paraParams->getRealParamValue(MultiplierForBgapCollectingMode),
                    1,                // paraSolver origin rank
                    paraComm, paraParams, paraTimer);

   // always minimization problem
   if( paraParams->getBoolParamValue(CleanUp) )
   {
      paraNodePool = new BbParaNodePoolForCleanUp(paraParams->getRealParamValue(BgapCollectingMode));
   }
   else
   {
      paraNodePool = new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
   }

   if( paraParams->getIntParamValue(NSolverNodesStartBreaking) == 0 ||
         paraParams->getIntParamValue(NStopBreaking) == 0 )
   {
      isBreakingFinised = true;
   }
   else
   {
      isBreakingFinised = false;
   }

   if( paraParams->getBoolParamValue(NChangeIntoCollectingModeNSolvers) )
   {
      paraParams->setIntParamValue(NChangeIntoCollectingMode, std::max((paraSolverPool->getNSolvers()/2), (size_t)1) );
   }

   if( !EPSEQ( paraParams->getRealParamValue(RandomNodeSelectionRatio), 0.0, MINEPSILON ) )
   {
      nNormalSelection = -1;
   }

   if( !paraParams->getBoolParamValue(CollectOnce) || 
       !( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 || paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) )
   { 
      winnerSolverNodesCollected = true;
   }

   if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
   {
      if( !( paraInitiator->getPrefixWarm() &&
            paraParams->getBoolParamValue(UG::MergeNodesAtRestart) &&
            !paraParams->getBoolParamValue(UG::DualBoundGainTest) ) )
      {
         std::cout << "** -w opttion = " << paraInitiator->getPrefixWarm() << " **" << std::endl;
         std::cout << "** MergeNodesAtRestart = " << paraParams->getBoolParamValue(UG::MergeNodesAtRestart) << std::endl;
         std::cout << "** DualBoundGainTest = " << paraParams->getBoolParamValue(UG::DualBoundGainTest) << std::endl;
         std::cout << "** GenerateReducedCheckpointFiles is specified, then this solver shuld be executed with"
               << "-w option and MergeNodesAtRestart = TRUE and DualBoundGainTest = FALSE ** " << std::endl;
         exit(1);
      }
      paraNodePoolBufferToGenerateCPF = new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
   }

   nBoundChangesOfBestNode = paraParams->getIntParamValue(UG::NBoundChangesOfMergeNode);
}

BbParaLoadCoordinator::~BbParaLoadCoordinator(
      )
{
   // std::cout << "In: nTerminated = " << nTerminated << std::endl;
   if( nTerminated != paraSolverPool->getNSolvers() &&
         ( ( paraRacingSolverPool && paraRacingSolverPool->getNumActiveSolvers() > 0 ) ||
	       runningPhase != TerminationPhase ||
               paraSolverPool->getNumActiveSolvers() > 0 ||
               interruptIsRequested
               )
     )
   {
      if( nTerminated != paraSolverPool->getNSolvers() ||
          runningPhase != TerminationPhase )
      {
         terminateAllSolvers();
         runningPhase = TerminationPhase;
      }
      // assert( runningPhase == TerminationPhase );

      int source;
      int tag;

      for(;;)
      {
         /*******************************************
          *  waiting for any message form anywhere  *
          *******************************************/
         double inIdleTime = paraTimer->getElapsedTime();
         (void)paraComm->probe(&source, &tag);
         lcts.idleTime += ( paraTimer->getElapsedTime() - inIdleTime );
         if( messageHandler[tag] )
         {
            int status = (this->*messageHandler[tag])(source, tag);
            if( status )
            {
               std::ostringstream s;
               s << "[ERROR RETURN form termination message handler]:" <<  __FILE__ <<  "] func = "
                 << __func__ << ", line = " << __LINE__ << " - "
                 << "process tag = " << tag << std::endl;
               abort();
            }
         }
         else
         {
            ABORT_LOGICAL_ERROR3( "No message handler for ", tag, " is not registered" );
         }

#ifdef UG_WITH_UGS
         if( commUgs ) checkAndReadIncumbent();
#endif

         // std::cout << "Loop: nTerminated = " << nTerminated << std::endl;
         if( nTerminated == paraSolverPool->getNSolvers() ) break;
      }
   }

//   for( int i = 1; i <= paraSolverPool->getNSolvers(); i++ )
//   {
//      paraComm->send( NULL, 0, ParaBYTE, i, TagTerminated);
//   }

   /** write final solution */
   paraInitiator->writeSolution("Final Solution");

#ifdef UG_WITH_ZLIB
   /* wite final checkpoint file */
   if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
   {
      assert( paraNodePoolBufferToGenerateCPF );
      while ( paraNodePoolBufferToGenerateCPF->getNumOfNodes() > 0 )
      {
         BbParaNode *node = paraNodePoolBufferToGenerateCPF->extractNode();
         paraNodePool->insert(node);
      }
      updateCheckpointFiles();
   }
   else
   {
      if( interruptIsRequested &&
            paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 )
      {
         updateCheckpointFiles();
      }
   }
#endif

   if( paraNodePool || paraNodeToKeepCheckpointFileNodes || unprocessedParaNodes )
   {
      int nKeepingNodes = 0;
      int nNoProcessedNodes = 0;
      if( logTasksTransferFlag )
      {
         if( paraNodeToKeepCheckpointFileNodes )
         {
            assert(!paraNodeToKeepCheckpointFileNodes->isEmpty());
            nKeepingNodes = paraNodeToKeepCheckpointFileNodes->getNumOfNodes();
         }
         if( unprocessedParaNodes )
         {
            nNoProcessedNodes = unprocessedParaNodes->getNumOfNodes();
         }
         *osLogTasksTransfer << std::endl << "BbParaLoadCoordinator: # received = " << lcts.nReceived
         << ", # sent = " << lcts.nSent << ", # sent immediately = " << lcts.nSentBackImmediately << ", # deleted = " << lcts.nDeletedInLc
         << ", # failed to send back = " << lcts.nFailedToSendBack
         << ", Max usage of node pool = " << paraNodePool->getMaxUsageOfPool() + nKeepingNodes + nNoProcessedNodes << std::endl;
         *osLogTasksTransfer << "# sent immediately ( another node ) = " << lcts.nSentBackImmediatelyAnotherNode
         << ", # # failed to send back ( another node ) = " << lcts.nFailedToSendBackAnotherNode << std::endl;
         if( !paraNodePool->isEmpty() ){
            *osLogTasksTransfer << "LoadCoodibator: NodePool in LoadCoordinatator is not empty: "
            << paraNodePool->getNumOfNodes() + nKeepingNodes + nNoProcessedNodes << " nodes remained" << std::endl;
         }
         else if( paraNodeToKeepCheckpointFileNodes || unprocessedParaNodes )
         {
            *osLogTasksTransfer << "LoadCoodibator: NodePool in LoadCoordinatator is not empty: "
            << (nKeepingNodes + nNoProcessedNodes) << " nodes remained" << std::endl;
         }
      }
      lcts.nMaxUsageOfNodePool = paraNodePool->getMaxUsageOfPool() + nKeepingNodes + nNoProcessedNodes;
      lcts.nInitialP = paraParams->getIntParamValue(NChangeIntoCollectingMode);
      if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool) )     // paraSolverPool may not be always BbParaSolverPoolForMinimization
      {
         lcts.mMaxCollectingNodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMMaxCollectingNodes();
      }
      else
      {
         lcts.mMaxCollectingNodes = 0;
      }
      lcts.nNodesInNodePool = paraNodePool->getNumOfNodes() + nKeepingNodes + nNoProcessedNodes;
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   // set final solving status
   if( initialNodesGenerated )
   {
      bbParaInitiator->setFinalSolverStatus(InitialNodesGenerated);
   }
   else
   {
      if( hardTimeLimitIsReached )
      {
         bbParaInitiator->setFinalSolverStatus(HardTimeLimitIsReached);
      }
      else if( memoryLimitIsReached )
      {
         bbParaInitiator->setFinalSolverStatus(MemoryLimitIsReached);
      }
      else if ( givenGapIsReached )
      {
         bbParaInitiator->setFinalSolverStatus(GivenGapIsReached);
      }
      else
      {
         if( interruptedFromControlTerminal ||
               (!racingTermination && computationIsInterrupted ) )
         {
            bbParaInitiator->setFinalSolverStatus(ComputingWasInterrupted);
         }
         else
         {
            if( paraNodeToKeepCheckpointFileNodes )
            {
#ifdef UG_WITH_ZLIB
               updateCheckpointFiles();
#endif
               bbParaInitiator->setFinalSolverStatus(RequestedSubProblemsWereSolved);
            }
            else
            {
               bbParaInitiator->setFinalSolverStatus(ProblemWasSolved);
            }
            if( outputTabularSolvingStatusFlag )
            {
               outputTabularSolvingStatus(' ');       /// this function cannot be overwritten, should turn off outputTabularSolvingStatusFlag, before destruct
            }
         }
      }
   }


   // set number of nodes solved and final dual bound value
   if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
   {
      *osTabularSolvingStatus << "*** This is GenerateReducedCheckpointFiles run ***" << std::endl;
      if( paraNodeToKeepCheckpointFileNodes || unprocessedParaNodes )
      {
         *osTabularSolvingStatus << "*** Current checkpoint-data have " <<
               (paraNodePool->getNumOfNodes() + paraNodeToKeepCheckpointFileNodes->getNumOfNodes() + unprocessedParaNodes->getNumOfNodes())
               << " nodes."
               << " including " << paraNodeToKeepCheckpointFileNodes->getNumOfNodes() + unprocessedParaNodes->getNumOfNodes()
               << " initially keeping nodes by the run-time parameter."
               << std::endl;
      }
      else
      {
         *osTabularSolvingStatus << "*** Current checkpoint-data have " << paraNodePool->getNumOfNodes()
               << " nodes." << std::endl;
      }

   }
   else
   {
      if( initialNodesGenerated )
      {
         if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool) )     // paraSolverPool may not be always BbParaSolverPoolForMinimization
         {
            bbParaInitiator->setNumberOfNodesSolved( std::max(1ULL, dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getTotalNodesSolved()) );
         }
         else
         {
            bbParaInitiator->setNumberOfNodesSolved(1ULL);   // always set 1 : no meaning
         }
         bbParaInitiator->setDualBound(paraNodePool->getBestDualBoundValue());
         lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(paraNodePool->getBestDualBoundValue());
      }
      else
      {
         if( racingTermination )
         {
            if( ( interruptedFromControlTerminal || hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached ) && paraRacingSolverPool )
            {
               bbParaInitiator->setNumberOfNodesSolved(dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesSolvedInBestSolver());
            }
            else
            {
               bbParaInitiator->setNumberOfNodesSolved(nSolvedRacingTermination);
            }
            if( lcts.globalBestDualBoundValue < minmalDualBoundNormalTermSolvers )
            {
               lcts.globalBestDualBoundValue = minmalDualBoundNormalTermSolvers;
            }
            if( bbParaInitiator->getGlobalBestIncumbentSolution() && lcts.globalBestDualBoundValue > bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() )
            {
               lcts.globalBestDualBoundValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
            }
            if( paraNodePool->getNumOfNodes() > 0 )
            {
               lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
            }
            bbParaInitiator->setDualBound(lcts.globalBestDualBoundValue);
            lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
         }
         else
         {
            if( !interruptedFromControlTerminal && !computationIsInterrupted && !hardTimeLimitIsReached && !memoryLimitIsReached && !givenGapIsReached )
            {
               if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool) )     // paraSolverPool may not be always BbParaSolverPoolForMinimization
               {
                  bbParaInitiator->setNumberOfNodesSolved( std::max(1ULL, dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getTotalNodesSolved()) );
               }
               else
               {
                  bbParaInitiator->setNumberOfNodesSolved(1ULL);   // always set 1 : no meaning
               }

               if( bbParaInitiator->getGlobalBestIncumbentSolution() && allCompInfeasibleAfterSolution && (!givenGapIsReached) )
               {
                  lcts.globalBestDualBoundValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
                  if( paraNodePool->getNumOfNodes() > 0 )
                  {
                     lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
                  }
                  bbParaInitiator->setDualBound(lcts.globalBestDualBoundValue);
                  lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
               }
               else
               {
                  if( lcts.globalBestDualBoundValue < minmalDualBoundNormalTermSolvers && !paraNodeToKeepCheckpointFileNodes && !unprocessedParaNodes )
                  {
                     if( bbParaInitiator->getGlobalBestIncumbentSolution() )
                     {
                        lcts.globalBestDualBoundValue = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(),
                              minmalDualBoundNormalTermSolvers );
                     }
                     else
                     {
                        lcts.globalBestDualBoundValue = minmalDualBoundNormalTermSolvers;
                     }
                  }
                  if( paraNodePool->getNumOfNodes() > 0 )
                  {
                     lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
                  }
                  bbParaInitiator->setDualBound(lcts.globalBestDualBoundValue);
                  lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
               }
            }
            else
            {
               if( paraNodePool->getNumOfNodes() > 0 )
               {
                  lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
               }
               if( isRacingStage() && (!hardTimeLimitIsReached) && (!memoryLimitIsReached) && (!givenGapIsReached) )
               {
                  if( paraRacingSolverPool )
                  {
                     bbParaInitiator->setDualBound(lcts.globalBestDualBoundValue);
                     bbParaInitiator->setNumberOfNodesSolved(  dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesSolvedInBestSolver() );
                  }
                  else
                  {
                     ABORT_LOGICAL_ERROR1("Computation is interrupted in racing stage, but no paraRacingSolverPool");
                  }
               }
               else
               {
                  if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool) )     // paraSolverPool may not be always BbParaSolverPoolForMinimization
                  {
                     bbParaInitiator->setNumberOfNodesSolved( std::max(1ULL, dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers()) );
                  }
                  else
                  {
                     bbParaInitiator->setNumberOfNodesSolved(1ULL);   // always set 1 : no meaning
                  }
                  bbParaInitiator->setDualBound(lcts.globalBestDualBoundValue);
               }
               lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
            }
         }
      }

      bbParaInitiator->outputFinalSolverStatistics( osTabularSolvingStatus, paraTimer->getElapsedTime() );
      if( (!racingTermination) &&
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getTotalNodesSolved() != dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers() )
      {
         *osTabularSolvingStatus << "* Warning: the number of nodes (total) including nodes solved by interrupted Solvers is "
               << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers() << std::endl;
      }
   }

   if( racingWinnerParams )
   {
      delete racingWinnerParams;
   }

   if( paraSolverPool && osStatisticsFinalRun  && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool) )     // paraSolverPool may not be always BbParaSolverPoolForMinimization
   {
      lcts.nNodesLeftInAllSolvers = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers();
      *osStatisticsFinalRun << "######### The number of nodes solved in all solvers: "
            << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getTotalNodesSolved();
      if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getTotalNodesSolved() != dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers() )
      {
         *osStatisticsFinalRun << " : "
               << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers();
      }
      *osStatisticsFinalRun << " #########" << std::endl;
   }
   // if( paraSolverPool )  delete paraSolverPool;

   lcts.runningTime = paraTimer->getElapsedTime();
   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << lcts.runningTime << " BbParaLoadCoordinator_TERMINATED" << std::endl;
      if( paraRacingSolverPool )
      {
         *osLogSolvingStatus << lcts.runningTime << " "
               << paraRacingSolverPool->getNumActiveSolvers() << " active racing ramp-up solvers exist." << std::endl;
      }
   }
#ifdef _DEBUG_LB
   std::cout << lcts.runningTime << " BbParaLoadCoordinator_TERMINATED" << std::endl;
   if( paraRacingSolverPool )
   {
      std::cout << lcts.runningTime << " "
            << paraRacingSolverPool->getNumActiveSolvers() << " active racing ramp-up solvers exist." << std::endl;
   }
#endif


   lcts.isCheckpointState = false;
   if( paraParams->getBoolParamValue(StatisticsToStdout) )
   {
      std::cout << lcts.toString() << std::endl;
   }

   if( osStatisticsFinalRun )
   {
      *osStatisticsFinalRun << lcts.toString();
   }

   if( paraRacingSolverPool && osStatisticsFinalRun )
   {
      *osStatisticsFinalRun << "***** <<< NOTE >>> "
            << paraRacingSolverPool->getNumActiveSolvers()
            << " active racing ramp-up solvers exist." << std::endl;
      *osStatisticsFinalRun << dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getStrActiveSolerNumbers() << std::endl;
   }

   if( paraRacingSolverPool )
   {
      delete paraRacingSolverPool;
      paraRacingSolverPool = 0;
      *racingSolversExist = true;
   }

   if( osStatisticsFinalRun )
   {
      osStatisticsFinalRun->flush();
   }

   if( nodesMerger ) delete nodesMerger;

   if( paraNodePool ) delete paraNodePool;
   if( paraNodePoolToRestart ) delete paraNodePoolToRestart;
   if( paraNodePoolBufferToRestart ) delete paraNodePoolBufferToRestart;
   if( paraNodePoolBufferToGenerateCPF ) delete paraNodePoolBufferToGenerateCPF;
   if( paraNodeToKeepCheckpointFileNodes ) delete paraNodeToKeepCheckpointFileNodes;
   if( unprocessedParaNodes ) delete unprocessedParaNodes;

}

void
BbParaLoadCoordinator::terminateAllSolvers(
      )
{
   terminationIssued = true;
   int exitSolverRequest = 0;    // do nothing
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      if( paraSolverPool->isSolverActive(i) ||
            ( paraRacingSolverPool && paraRacingSolverPool->isSolverActive(i) ) )
      {
         if( !paraSolverPool->isInterruptRequested(i) && !paraSolverPool->isTerminateRequested(i) )    /// even if racing solver is running, paraSolverPool flag is used
         {
            PARA_COMM_CALL(
                  paraComm->send( &exitSolverRequest, 1, ParaINT, i, TagInterruptRequest )
            );
            paraSolverPool->interruptRequested(i);
         }
      }
      if( !paraSolverPool->isTerminateRequested(i) )
      {
         PARA_COMM_CALL(
               paraComm->send( NULL, 0, ParaBYTE, i, TagTerminateRequest )
         );
         paraSolverPool->terminateRequested(i);
         if( paraParams->getBoolParamValue(Deterministic) )
         {
            int token[2];
            token[0] = i;
            token[1] = -2;
            PARA_COMM_CALL(
                  paraComm->send( token, 2, ParaINT, token[0], TagToken )
            );
         }
      }
   }
}

int
BbParaLoadCoordinator::processTagTask(
      int source,
      int tag
      )
{

   BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
   paraNode->receive(paraComm, source);

//   std::cout << "S." << source
//        << ", SEND: nBoundChanges = " << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges() << std::endl;

   // std::cout << "processTagTask: " << paraNode->toString() << std::endl;

#ifdef UG_DEBUG_SOLUTION
#ifdef UG_DEBUG_SOLUTION_OPT_PATH
   if( paraNode->getDiffSubproblem() && (!paraNode->getDiffSubproblem()->isOptimalSolIncluded()) )
   {
      delete paraNode;
      PARA_COMM_CALL(
             paraComm->send( NULL, 0, ParaBYTE, source, TagTaskReceived);
             );
      return 0;
   }
#endif
#endif

   // std::cout << paraTimer->getElapsedTime()<< " S." << source << " nodeId = " << paraNode->toSimpleString() << " is recived" << std::endl;

   lcts.nReceived++;

   if( paraNodePoolToRestart )
   {
      delete paraNode;
      return 0;
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( ( bbParaInitiator->getGlobalBestIncumbentSolution() &&
         paraNode->getDualBoundValue() < bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() ) ||
         !( bbParaInitiator->getGlobalBestIncumbentSolution() ) )
   {
      paraNode->setGlobalSubtaskId(paraComm->getRank(), createNewGlobalSubtreeId());
      /** in the case that ParaNode received from LoadCoordinator is not implemented yet */
      ParaTask *ancestorNode = paraSolverPool->getCurrentTask(source);
      paraNode->setAncestor(
            new ParaTaskGenealogicalLocalPtr( ancestorNode->getTaskId(), ancestorNode ));
      ancestorNode->addDescendant(
            new ParaTaskGenealogicalLocalPtr( paraNode->getTaskId(), paraNode) );
      if( hugeImbalance )
      {
         paraNodePoolBufferToRestart->insert(paraNode);
         if( logSolvingStatusFlag )
          {
             // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
             *osLogSolvingStatus << paraTimer->getElapsedTime()
             << " S." << source
             << " |pb< "
             << bbParaInitiator->convertToExternalValue(
                   paraNode->getDualBoundValue() )
             << " "
             << paraNode->toSimpleString();
             if( paraNode->getDiffSubproblem() )
             {
                *osLogSolvingStatus
                 << ", "
                 // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                 << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->toStringStat();
             }
             *osLogSolvingStatus << std::endl;
          }
      }
      else
      {
         paraNodePool->insert(paraNode);
         if( logSolvingStatusFlag )
         {
            // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source
            << " |p< "
            << bbParaInitiator->convertToExternalValue(
                  paraNode->getDualBoundValue() )
            << " "
            << paraNode->toSimpleString();
            if( paraNode->getDiffSubproblem() )
            {
               *osLogSolvingStatus
                << ", "
                // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->toStringStat();
            }
            *osLogSolvingStatus << std::endl;
         }
      }
      if( merging )
      {
         assert(nodesMerger);
         nodesMerger->addNodeToMergeNodeStructs(paraNode);
      }

      if( paraParams->getBoolParamValue(RacingStatBranching) &&
            source == racingWinner &&
            !winnerSolverNodesCollected )
      {
         if( minDepthInWinnerSolverNodes > paraNode->getDepth() )
         {
            minDepthInWinnerSolverNodes = paraNode->getDepth();
         }
      }

      if( !( paraSolverPool->getNumInactiveSolvers() > 0 && sendParaTasksToIdleSolvers() ) )
                                              // In racing stage, some solver could take time for solving a node
                                              // Therefore, some solver could stay idle so long time in paraSolverPool
                                              // However, node collecting has to terminated to sending switch out message
      {
         // paraNodePool->insert(paraNode);
         /** switch-out request might be delayed to reach the target solver.
          * This behavior affects load balancing sometimes.
          *  Therefore, if additional nodes are received, sends switch-out request again.
         //if( runningPhase != RampUpPhase && !(paraSolverPool->isInCollectingMode()) )
         //{
            // paraSolverPool->enforcedSwitchOutCollectingMode(source);
         //}
         ** I don't want to do this. So, I decided to wait message after each notification when LC in collecting mode if necessary */
         // without consideration of keeping nodes in checkpoint file
         double globalBestDualBoundValueLocal =
            std::max (
                  std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                  lcts.globalBestDualBoundValue );
         if( runningPhase != RampUpPhase && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() &&
              ( ( paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
               > ( paraParams->getRealParamValue(MultiplierForCollectingMode) *
                     paraParams->getIntParamValue(NChangeIntoCollectingMode)*dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMCollectingNodes() )
                     ) ||
                    ( paraNodePool->getNumOfNodes()
                                   > ( paraParams->getRealParamValue(MultiplierForCollectingMode) *
                                         paraParams->getIntParamValue(NChangeIntoCollectingMode)*dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMCollectingNodes()*
                                         paraParams->getIntParamValue(LightWeightNodePenartyInCollecting)
                     ) ) ) )
         {
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
            firstCollectingModeState = 1;
            isCollectingModeRestarted = false;
         }
      }
   }
   else
   {
      assert( !( EPSLT( paraNode->getDualBoundValue(), bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), eps) ) );
#ifdef UG_DEBUG_SOLUTION
      if( paraNode->getDiffSubproblem() && paraNode->getDiffSubproblem()->isOptimalSolIncluded() )
      {
         throw "Optimal solution going to be killed.";
      }
#endif
      delete paraNode;
      lcts.nDeletedInLc++;
      if( runningPhase != RampUpPhase && !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode()) &&
            paraSolverPool->getNumInactiveSolvers() > ( paraSolverPool->getNSolvers() * 0.1 )  &&
            paraNodePool->isEmpty() )
      { // inactive solver exists but cannot send a ParaNode to it
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
         if( firstCollectingModeState == -1 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) firstCollectingModeState = 0;
      }
   }

   PARA_COMM_CALL(
          paraComm->send( NULL, 0, ParaBYTE, source, TagTaskReceived);
          );

   return 0;
}

int
BbParaLoadCoordinator::processTagSolution(
      int source,
      int tag
      )
{

   BbParaSolution *sol = dynamic_cast<BbParaSolution *>(paraComm->createParaSolution());
   sol->receive(paraComm, source);

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

#ifdef _DEBUG_DET
   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " S." << source << " R.SOL "
      << bbParaInitiator->convertToExternalValue(
            sol->getObjectiveFunctionValue()
            ) << std::endl;
   }
#endif

   if( updateSolution(sol) ) 
   {
      delete sol;
      if( !(paraParams->getIntParamValue(RampUpPhaseProcess) == 3 && runningPhase == RampUpPhase ) )
      {
         if( bbParaInitiator->canGenerateSpecialCutOffValue() )
         {
            sendCutOffValue(source);
         }
         sendIncumbentValue(source);
      }
      primalUpdated = true;
      allCompInfeasibleAfterSolution = true;
      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " I.SOL "
         << bbParaInitiator->convertToExternalValue(
               bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
               ) << std::endl;
      }
#ifdef _DEBUG_LB
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " I.SOL "
      << paraInitiator->convertToExternalValue(
            paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
            ) << std::endl;
#endif
      /** output tabular solving status */
      if( outputTabularSolvingStatusFlag )
      {
         outputTabularSolvingStatus('*');
      }
#ifdef UG_WITH_ZLIB
      /* Do not have to remove ParaNodes from NodePool. It is checked and removed before sending them */
      /** save incumbent solution */
      char solutionFileNameTemp[256];
      char solutionFileName[256];
      if( paraParams->getBoolParamValue(Checkpoint) && paraComm->getRank() == 0  )
      {
         sprintf(solutionFileNameTemp,"%s%s_after_checkpointing_solution_t.gz",
               paraParams->getStringParamValue(CheckpointFilePath),
               paraInitiator->getParaInstance()->getProbName() );
         paraInitiator->writeCheckpointSolution(std::string(solutionFileNameTemp));
         sprintf(solutionFileName,"%s%s_after_checkpointing_solution.gz",
               paraParams->getStringParamValue(CheckpointFilePath),
               paraInitiator->getParaInstance()->getProbName() );
         if ( rename(solutionFileNameTemp, solutionFileName) )
         {
            std::cout << "after checkpointing solution file cannot be renamed: errno = " << strerror(errno) << std::endl;
            exit(1);
         }
      }
#endif
#ifdef UG_WITH_UGS
      if( commUgs )
      {
         paraInitiator->writeUgsIncumbentSolution(commUgs);
      }
#endif
   }
   else
   {
      delete sol;
   }
   return 0;
}

int
BbParaLoadCoordinator::processTagSolverState(
      int source,
      int tag
      )
{

   double globalBestDualBoundValue = -DBL_MAX;
   double globalBestDualBoundValueLocal = -DBL_MAX;

   BbParaSolverState *solverState = dynamic_cast<BbParaSolverState *>(paraComm->createParaSolverState());
   solverState->receive(paraComm, source, tag);

   if( paraDetTimer
         && paraDetTimer->getElapsedTime() < solverState->getDeterministicTime() )

   {
      paraDetTimer->update( solverState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( solverState->isRacingStage() )
   {
      globalBestDualBoundValueLocal = globalBestDualBoundValue =
               std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(),
                         paraNodePool->getBestDualBoundValue() );
      /** not update paraSolverPool. The solver is inactive in paraSolverPool */
      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " | "
         << bbParaInitiator->convertToExternalValue(
               solverState->getSolverLocalBestDualBoundValue()
               );
         if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
               bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue )
         {
            *osLogSolvingStatus << " ( Inf )";
         }
         else
         {
            *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
         }
         *osLogSolvingStatus << " [ " << solverState->getNNodesLeft() << " ]";
         *osLogSolvingStatus << " ** G.B.: " << bbParaInitiator->convertToExternalValue(globalBestDualBoundValue);
         if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
               bbParaInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
         {
            *osLogSolvingStatus << " ( Inf ) ";
         }
         else
         {
            *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
         }
         *osLogSolvingStatus << "[ " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() << ", " << paraNodePool->getNumOfNodes()
         << " ( " <<  paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
         <<" ) ] ** R" << std::endl;
         // <<" ) ] ** RR " << solverState->getDeterministicTime() << std::endl;  // for debug

      }
#ifdef _DEBUG_LB
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " | "
      << paraInitiator->convertToExternalValue(
            solverState->getSolverLocalBestDualBoundValue()
            );
      if( !paraInitiator->getGlobalBestIncumbentSolution() ||
            paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue )
      {
         std::cout << " ( Inf )";
      }
      else
      {
         std::cout << " ( " << paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
      }
      std::cout << " [ " << solverState->getNNodesLeft() << " ]";
      globalBestDualBoundValue =
               std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() );
      std::cout << " ** G.B.: " << paraInitiator->convertToExternalValue(globalBestDualBoundValue);
      if( !paraInitiator->getGlobalBestIncumbentSolution() ||
            paraInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
      {
         std::cout << " ( Inf ) ";
      }
      else
      {
         std::cout << " ( " << paraInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
      }
      std::cout << "[ " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() << ", " << paraNodePool->getNumOfNodes()
      << " ( " <<  paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
      <<" ) ] ** RR" << std::endl;
#endif
   }
   else
   {
      if( !dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverActive(source) )
      {
         if( runningPhase == TerminationPhase || hardTimeLimitIsReached || givenGapIsReached )
         {
            delete solverState;
            return 0;
         }
      }
      assert( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );
      double solverDualBoundGain = 0.0;
      double sum = 0.0;
      if(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source) == 0
            && solverState->getNNodesSolved() > 0
            && !paraSolverPool->getCurrentTask(source)->isRootTask() )
      {
         solverDualBoundGain = solverState->getSolverLocalBestDualBoundValue() - dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source))->getDualBoundValue();
         if( logSolvingStatusFlag )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " G " <<  paraSolverPool->getCurrentTask(source)->toSimpleString()
            << ", a:" << averageDualBoundGain
            << ", ma:" << averageLastSeveralDualBoundGains
            << ", g:" << solverDualBoundGain;
            if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isDualBounGainTesting(source) )
            {
               *osLogSolvingStatus << ", T";
            }
            *osLogSolvingStatus << ", st:" << solverState->getAverageDualBoundGain()*paraParams->getRealParamValue(DualBoundGainBranchRatio)
            << std::endl;
         }
         lastSeveralDualBoundGains.push_back(solverDualBoundGain);
         for(int i = 0; i < static_cast<int>(lastSeveralDualBoundGains.size()); i++ )
         {
            sum += lastSeveralDualBoundGains[i];
         }
         averageLastSeveralDualBoundGains = sum/lastSeveralDualBoundGains.size();
         averageDualBoundGain =
               averageDualBoundGain * ( static_cast<double>(nAverageDualBoundGain)/(static_cast<double>(nAverageDualBoundGain) + 1.0) )
               + solverDualBoundGain * ( 1.0/(static_cast<double>(nAverageDualBoundGain) + 1.0 ) );
         if( lastSeveralDualBoundGains.size() > 7 )
         {
            lastSeveralDualBoundGains.pop_front();
         }
         nAverageDualBoundGain++;
      }

//      std::cout << "R." << source
//            << ": solverState->getNNodesSolved() = " << solverState->getNNodesSolved()
//            << ", dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved = "
//            << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)
//            << ", solverState->getNNodesLeft() = " << solverState->getNNodesLeft() << std::endl;
//      assert( solverState->getNNodesSolved() >= dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->updateSolverStatus(source,
                                      solverState->getNNodesSolved(),
                                      solverState->getNNodesLeft(),
                                      solverState->getSolverLocalBestDualBoundValue(),
                                      paraNodePool
                                      );

      if( paraNodeToKeepCheckpointFileNodes && paraNodeToKeepCheckpointFileNodes->getNumOfNodes() > 0 )
      {
         globalBestDualBoundValue = paraNodeToKeepCheckpointFileNodes->getBestDualBoundValue();
         globalBestDualBoundValueLocal =
               std::max (
                     std::min(
                           std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(),
                                     paraNodePool->getBestDualBoundValue() ),
                           minmalDualBoundNormalTermSolvers ),
                     lcts.globalBestDualBoundValue );
      }
      else if ( unprocessedParaNodes && unprocessedParaNodes->getNumOfNodes() > 0 )
      {
         globalBestDualBoundValue = globalBestDualBoundValueLocal =
               std::max (
                     std::min(
                           std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                           // minmalDualBoundNormalTermSolvers ),  /// should not be in racing start
                           unprocessedParaNodes->getBestDualBoundValue() ),
                     lcts.globalBestDualBoundValue );
      }
      else
      {
         if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles)  )
         {
            globalBestDualBoundValueLocal = globalBestDualBoundValue =
                  std::min(
                        std::min(
                              std::min( paraNodePool->getBestDualBoundValue(), paraNodePoolBufferToGenerateCPF->getBestDualBoundValue() ),
                              dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() ),
                  lcts.globalBestDualBoundValue );
         }
         else
         {
            if( isRacingStage() || racingTermination )
            {
               globalBestDualBoundValueLocal = globalBestDualBoundValue =
                     std::max (
                           std::min(
                                 std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                                 minmalDualBoundNormalTermSolvers ),
                           lcts.globalBestDualBoundValue );
            }
            else
            {
               globalBestDualBoundValueLocal = globalBestDualBoundValue =
                     std::max (
                           std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                           lcts.globalBestDualBoundValue );
            }

         }
      }

      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " | "
         << bbParaInitiator->convertToExternalValue(
               solverState->getSolverLocalBestDualBoundValue()
               );
         if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
               bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue )
         {
            *osLogSolvingStatus << " ( Inf )";
         }
         else
         {
            *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
         }
         *osLogSolvingStatus << " [ " << solverState->getNNodesLeft() << " ]";
         *osLogSolvingStatus << " ** G.B.: " << bbParaInitiator->convertToExternalValue(globalBestDualBoundValue);
         if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
               bbParaInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
         {
            *osLogSolvingStatus << " ( Inf ) ";
         }
         else
         {
            *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
         }
         *osLogSolvingStatus << "[ " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() << ", " << paraNodePool->getNumOfNodes()
         << " ( " <<  paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
         <<" ) ] **";
         if( runningPhase == RampUpPhase ) *osLogSolvingStatus << " R";
         if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() )
         {
            *osLogSolvingStatus << " C";
            if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverInCollectingMode(source) )
            {
               *osLogSolvingStatus << " 1";
            }
            else
            {
               *osLogSolvingStatus << " 0";
            }

         }
         *osLogSolvingStatus << " " << solverState->getDeterministicTime();   // for debug
         if( paraNodePoolBufferToRestart )
         {
            *osLogSolvingStatus << " " << paraNodePoolBufferToRestart->getNumOfNodes();  // for debug
         }
         *osLogSolvingStatus << std::endl;

         /** log the number of nodes transfer */
         if( paraParams->getIntParamValue(NNodesTransferLogging) > 0 &&
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers() > lcts.nNodesOutputLog )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime() << " = ";
            *osLogSolvingStatus << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers()
                  << " " << (dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() + paraNodePool->getNumOfNodes())
                  << " s " << lcts.nSent << " r " << lcts.nReceived
                  << std::endl;
            lcts.nNodesOutputLog =
                  ( ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers()/paraParams->getIntParamValue(NNodesTransferLogging) )+1)
                  * paraParams->getIntParamValue(NNodesTransferLogging);
         }

         if( paraParams->getRealParamValue(TNodesTransferLogging) > 0.0 &&
               paraTimer->getElapsedTime() > lcts.tNodesOutputLog )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime() << " = ";
            *osLogSolvingStatus << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers()
                  << " " << (dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() + paraNodePool->getNumOfNodes())
                  << " s " << lcts.nSent << " r " << lcts.nReceived
                  << std::endl;
            lcts.tNodesOutputLog = (static_cast<int>(paraTimer->getElapsedTime()/paraParams->getRealParamValue(TNodesTransferLogging)) + 1.0)
                  * paraParams->getRealParamValue(TNodesTransferLogging);
         }
      }

      BbParaNode *node = dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source));
      if( node->getMergeNodeInfo() != 0 && solverState->getNNodesSolved() > 2 )  // I stand on the safety side. we can write "> 1"
      {
         assert(nodesMerger);
         nodesMerger->mergeNodes(node, paraNodePool);
         if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
         {
            // std::cout << "S." << source << " ParaNode is saved to Buffer." << std::endl;
            assert( !node->getMergeNodeInfo() );
            paraNodePoolBufferToGenerateCPF->insert(node);
         }
      }

#ifdef _DEBUG_LB
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " | "
      << paraInitiator->convertToExternalValue(
            solverState->getSolverLocalBestDualBoundValue()
            );
      if( !paraInitiator->getGlobalBestIncumbentSolution() ||
            paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue )
      {
         std::cout << " ( Inf )";
      }
      else
      {
         std::cout << " ( " << paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
      }
      std::cout << " [ " << solverState->getNNodesLeft() << " ]";
      globalBestDualBoundValue =
         std::max (
               std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
               lcts.globalBestDualBoundValue );
      std::cout << " ** G.B.: " << paraInitiator->convertToExternalValue(globalBestDualBoundValue);
      if( !paraInitiator->getGlobalBestIncumbentSolution() ||
            paraInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
      {
         std::cout << " ( Inf ) ";
      }
      else
      {
         std::cout << " ( " << paraInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
      }
      std::cout << "[ " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() << ", " << paraNodePool->getNumOfNodes()
      << " ( " <<  paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
      <<" ) ] **";
      if( runningPhase == RampUpPhase ) std::cout << " R";
      if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() )
      {
         std::cout << " C";
         if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverInCollectingMode(source) )
         {
            std::cout << " 1";
         }
         else
         {
            std::cout << " 0";
         }

      }
      std::cout << std::endl;
#endif

   }

   /** the following should be before noticationId back to the source solver */
   if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) )
   {
      if( bbParaInitiator->getGlobalBestIncumbentSolution() &&
            bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
            < solverState->getGlobalBestPrimalBoundValue() )
      {
         bbParaInitiator->getGlobalBestIncumbentSolution()->send(paraComm, source);
      }
   }

   // if( paraParams->getBoolParamValue(CheckGapInLC) )
   if( !givenGapIsReached )
   {
//       std::cout << "absgap = " <<
//            bbParaInitiator->getAbsgap(globalBestDualBoundValue)  <<
//            ", abs gap value = " << bbParaInitiator->getAbsgapValue() << std::endl;
//       std::cout << "gap = " <<
//            bbParaInitiator->getGap(globalBestDualBoundValue)  <<
//            ", gap value = " << bbParaInitiator->getGapValue() << std::endl;
      if( bbParaInitiator->getAbsgap(globalBestDualBoundValue) < bbParaInitiator->getAbsgapValue() ||
            bbParaInitiator->getGap(globalBestDualBoundValue) < bbParaInitiator->getGapValue() )
      {
         // std::cout << "current dual = "  << paraInitiator->convertToExternalValue(solverState->getSolverLocalBestDualBoundValue()) <<std::endl;
         // std::cout << "current gap = " << paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) <<std::endl;
         for( unsigned int i = 1; i <= paraSolverPool->getNSolvers(); i++ )
         {
            if ( paraSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
            {
               PARA_COMM_CALL(
                     paraComm->send( NULL, 0, ParaBYTE, i, TagGivenGapIsReached )
               );
               paraSolverPool->interruptRequested(i);
            }

         }
         givenGapIsReached = true;
      }
      else
      {
         if( bbParaInitiator->getAbsgap(solverState->getSolverLocalBestDualBoundValue()) < bbParaInitiator->getAbsgapValue() ||
               bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) < bbParaInitiator->getGapValue() )
         {
            PARA_COMM_CALL(
                  paraComm->send( NULL, 0, ParaBYTE, source, TagGivenGapIsReached )
            );
            paraSolverPool->interruptRequested(source);
         }
      }
   }

   double lcBestDualBoundValue = paraNodePool->getBestDualBoundValue();
   PARA_COMM_CALL(
         paraComm->send( &lcBestDualBoundValue, 1, ParaDOUBLE, source, TagLCBestBoundValue)
         );
   unsigned int notificationId = solverState->getNotificaionId();
   PARA_COMM_CALL(
         paraComm->send( &notificationId, 1, ParaUNSIGNED, source, TagNotificationId)
         );

   if( paraParams->getBoolParamValue(InitialNodesGeneration) &&
         (signed)( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() + paraNodePool->getNumOfNodes() ) >= paraParams->getIntParamValue(NumberOfInitialNodes) )
   {
      for(unsigned int i = 1; i <= paraSolverPool->getNSolvers(); i++ )
      {
         int nCollect = -1;
         if( paraSolverPool->isSolverActive(i) )
         {
            PARA_COMM_CALL(
                  paraComm->send( &nCollect, 1, ParaINT, i, TagCollectAllNodes )
            );
         }
      }
      initialNodesGenerated = true;
   }
   else
   {
      if( runningPhase != RampUpPhase  )
      {
         if( paraNodePool->getNumOfGoodNodes(
               globalBestDualBoundValueLocal
               ) > 0 )
         {
            statEmptyNodePoolTime = DBL_MAX;
            isCollectingModeRestarted = false;
         }
         else   // paraNodePool is empty in terms of the number of good nodes
         {
            if( paraSolverPool->getNumInactiveSolvers() > 0 )
            {
               if( !isCollectingModeRestarted )
               {
                  double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
                  if( logSolvingStatusFlag )
                  {
                     *osLogSolvingStatus << paraTimer->getElapsedTime()
                     << " Collecting mode is restarted with " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNLimitCollectingModeSolvers()
                     << std::endl;
                  }
                  if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) isCollectingModeRestarted = true;
               }
            }
            else  // no inactive solvers
            {
               if( isCollectingModeRestarted )
               {
                  statEmptyNodePoolTime = DBL_MAX;  // node pool is  empty, but it becomes empty soon again, it is better to restart collecting mode
                  isCollectingModeRestarted = false;
               }
            }
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               if( ( paraDetTimer->getElapsedTime() - statEmptyNodePoolTime ) < 0 )
               {
                  statEmptyNodePoolTime = paraDetTimer->getElapsedTime();
               }
            }
            else
            {
               if( ( paraTimer->getElapsedTime() - statEmptyNodePoolTime ) < 0 )
               {
                  statEmptyNodePoolTime = paraTimer->getElapsedTime();
               }
            }
         }

         if( ( paraParams->getBoolParamValue(Deterministic) &&
               ( paraDetTimer->getElapsedTime() - statEmptyNodePoolTime ) > paraParams->getRealParamValue(TimeToIncreaseCMS) ) ||
               ( !paraParams->getBoolParamValue(Deterministic) &&
               ( paraTimer->getElapsedTime() - statEmptyNodePoolTime ) > paraParams->getRealParamValue(TimeToIncreaseCMS) ) )
         {
            if( paraNodePool->getNumOfNodes() == 0 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->canIncreaseLimitNLimitCollectingModeSolvers() )
               // ramp-up may collect nodes having not so good nodes. As long as nodes exist, the limit number should not be increased.
            {
               double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->incNLimitCollectingModeSolvers();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " Limit number of collecting mode solvers extends to " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNLimitCollectingModeSolvers()
                  << ", p = " << paraParams->getIntParamValue(UG::NChangeIntoCollectingMode)*dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMCollectingNodes()
                  << std::endl;
                  if( outputTabularSolvingStatusFlag )
                  {
                     *osTabularSolvingStatus <<
                           "Limit number of collecting mode solvers extends to " <<
                           dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNLimitCollectingModeSolvers() <<
                           " after " << paraTimer->getElapsedTime() << " seconds." << std::endl;
                  }
#ifdef _DEBUG_LB
                  std::cout << paraTimer->getElapsedTime()
                        << " Limit number of collecting mode solvers extends to " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNLimitCollectingModeSolvers()
                        << std::endl;
#endif
               }
               // isCollectingModeRestarted = false;
            }
            else   // cannot increase the number of collecting mode solvers
            {
               double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " Collecting mode is restarted with " << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNLimitCollectingModeSolvers()
                  << std::endl;
               }
            }
            if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() )
            {
               isCollectingModeRestarted = true;
               statEmptyNodePoolTime = DBL_MAX;
            }
         }
      }

      if( !( solverState->isRacingStage() ) &&
            runningPhase != RampUpPhase && !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode()) &&
            ( (signed)paraNodePool->getNumOfGoodNodes(
                  globalBestDualBoundValueLocal
                  ) < paraParams->getIntParamValue(NChangeIntoCollectingMode)*dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMCollectingNodes()
                  &&
                  (signed)paraNodePool->getNumOfNodes(
                                    ) < paraParams->getIntParamValue(NChangeIntoCollectingMode)*dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getMCollectingNodes()*
                                    paraParams->getIntParamValue(LightWeightNodePenartyInCollecting)
                  ) )
      {
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
         if( firstCollectingModeState == -1 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) firstCollectingModeState = 0;
      }

      if( !isBreakingFinised )
      {
         if( (!solverState->isRacingStage()) && runningPhase == NormalRunningPhase )
         {
            if( (signed)paraNodePool->getNumOfNodes() > paraParams->getIntParamValue(NStopBreaking) ||
                  (signed)dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() < paraParams->getIntParamValue(NStopBreaking) )
            {
               isBreakingFinised = true;
            }
            else
            {
               if( breakingSolverId == -1 )
               {
                  if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesLeftInBestSolver()
                        > paraParams->getIntParamValue(NSolverNodesStartBreaking) )
                  {
                     breakingSolverId = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getBestSolver();
                     assert( breakingSolverId != -1 );
                     double targetBound = ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue()*
                           paraParams->getRealParamValue(MultiplierForBreakingTargetBound) );
                     int nLimitTransfer = paraParams->getIntParamValue(NTransferLimitForBreaking);
                     PARA_COMM_CALL(
                           paraComm->send( &targetBound, 1, ParaDOUBLE, breakingSolverId, TagBreaking )
                     );
                     PARA_COMM_CALL(
                           paraComm->send( &nLimitTransfer, 1, ParaINT, breakingSolverId, TagBreaking )
                     );
                  }
                  else
                  {
                     if( ( ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue()
                           + paraParams->getRealParamValue(ABgapForSwitchingToBestSolver)*3 ) >
                           solverState->getSolverLocalBestDualBoundValue() ) &&
                                 (signed)dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() >
                                 std::max(paraParams->getIntParamValue(NStopBreaking)*2, (int)paraSolverPool->getNSolvers()*2 ) &&
                         solverState->getNNodesLeft() >  ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers()*0.5 ) )
                     {
                        breakingSolverId = source;
                        double targetBound = ( solverState->getSolverLocalBestDualBoundValue()*
                              paraParams->getRealParamValue(MultiplierForBreakingTargetBound) );
                        int nLimitTransfer = paraParams->getIntParamValue(NTransferLimitForBreaking);
                        PARA_COMM_CALL(
                              paraComm->send( &targetBound, 1, ParaDOUBLE, breakingSolverId, TagBreaking )
                        );
                        PARA_COMM_CALL(
                              paraComm->send( &nLimitTransfer, 1, ParaINT, breakingSolverId, TagBreaking )
                        );
                     }
                  }
               }
            }
         }
      }
      else   // isBootstrapFinised
      {
         if( runningPhase == NormalRunningPhase &&
               (signed)paraNodePool->getNumOfNodes() < paraParams->getIntParamValue(NStopBreaking) &&
               (signed)dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers()
               > std::max(paraParams->getIntParamValue(NStopBreaking)*2, (int)paraSolverPool->getNSolvers()*2 ) )
         {
            // break again. several solvers can be in breaking situation. That is, braking message can be sent to breaking solver
            isBreakingFinised = false;
            breakingSolverId = -1;
         }
      }
   }

   lcts.globalBestDualBoundValue = std::max(lcts.globalBestDualBoundValue, globalBestDualBoundValue);
   lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(globalBestDualBoundValue);

   delete solverState;
   return 0;
}

int
BbParaLoadCoordinator::processTagCompletionOfCalculation(
      int source,
      int tag
      )
{

   BbParaCalculationState *calcState = dynamic_cast<BbParaCalculationState *>(paraComm->createParaCalculationState());
   calcState->receive(paraComm, source, tag);
   if( paraRacingSolverPool && paraRacingSolverPool->isSolverActive(source) ) // racing root node termination
   {
      writeTransferLogInRacing(source, calcState);
   }
   else
   {
      writeTransferLog(source, calcState);
   }

   if( !winnerSolverNodesCollected &&
         racingWinner == source )
   {
      winnerSolverNodesCollected = true;
      if( merging )
      {
         assert(nodesMerger);
         nodesMerger->generateMergeNodesCandidates(paraComm, paraInitiator);   // Anyway,merge nodes candidates have to be generated,
                                                                               // even if running with InitialNodesGeneration
         merging = false;
      }
      if( paraParams->getBoolParamValue(InitialNodesGeneration) &&
           (signed)paraNodePool->getNumOfNodes() >= paraParams->getIntParamValue(NumberOfInitialNodes) )
      {
         initialNodesGenerated = true;
      }
   }

   int calcTerminationState = calcState->getTerminationState();

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( logSolvingStatusFlag )
   {
      switch ( calcTerminationState )
      {
      case CompTerminatedNormally:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " > " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByAnotherTask:
      {
         /** When starts with racing ramp-up, solvers except winner should be terminated in this state */
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_ANOTHER_NODE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByInterruptRequest:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedInRacingStage:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(TERMINATED_IN_RACING_STAGE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompInterruptedInRacingStage:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_IN_RACING_STAGE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompInterruptedInMerging:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_IN_MERGING) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByTimeLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_TIME_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByMemoryLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_MEMORY_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      default:
         THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
      }

      if( paraParams->getBoolParamValue(CheckEffectOfRootNodePreprocesses) &&
            calcState->getNSolvedWithNoPreprocesses() > 0 )
      {
         *osLogSolvingStatus << " SOLVED_AT_ROOT ( DEPTH = "
               << dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source))->getDepth()
               << ", Gap = "
               << bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source))->getDualBoundValue()) * 100
               << "%, TrueGap = "
               << bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source))->getInitialDualBoundValue()) * 100
               << "% ) [ "
               << calcState->getNSolvedWithNoPreprocesses() << " ]";
      }

      *osLogSolvingStatus << ", ct:" << calcState->getCompTime()
            << ", nr:" << calcState->getNRestarts()
            << ", n:" << calcState->getNSolved()
            << ", rt:" << calcState->getRootTime()
            << ", avt:" << calcState->getAverageNodeCompTimeExcpetRoot()
            << std::endl;
   }

#ifdef _DEBUG_LB
   switch ( calcTerminationState )
   {
   case CompTerminatedNormally:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >";
      break;
   }
   case CompTerminatedByAnotherTask:
   {
      /** When starts with racing ramp-up, solvers except winner should be terminated in this state */
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(INTERRUPTED_BY_ANOTHER_NODE)";
      break;
   }
   case CompTerminatedByInterruptRequest:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(INTERRUPTED)";
      break;
   }
   case CompTerminatedInRacingStage:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(TERMINATED_IN_RACING_STAGE)";
      break;
   }
   case CompInterruptedInRacingStage:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(INTERRUPTED_IN_RACING_STAGE)";
      break;
   }
   default:
      THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
   }

   if( paraParams->getBoolParamValue(CheckEffectOfRootNodePreprocesses) &&
         calcState->getNSolvedWithNoPreprocesses() > 0 )
   {
      std::cout << " SOLVED_AT_ROOT ( DEPTH = " << paraSolverPool->getCurrentTask(source)->getDepth()
      << ", Gap = " << paraInitiator->getGap(paraSolverPool->getCurrentTask(source)->getDualBoundValue()) * 100  << "%, TrueGap = "
      << paraInitiator->getGap(paraSolverPool->getCurrentTask(source)->getInitialDualBoundValue()) * 100  << "% ) [ "
      << calcState->getNSolvedWithNoPreprocesses() << " ]";
   }
   std::cout << std::endl;
#endif

   switch ( calcTerminationState )
   {
   case CompTerminatedNormally:
   {
      writeSubtreeInfo(source, calcState);
      BbParaNode *node = dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(source));
      if( node->getMergeNodeInfo() )
      {
         node->setDualBoundValue(bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue()));
         assert(nodesMerger);
         nodesMerger->mergeNodes(node,paraNodePool);
      }
      if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )  // RacingSolverPool is inactivated below
      {
         if( paraNodePoolToRestart )
         {
            BbParaNode *solvingNode = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool);
            if( solvingNode->getAncestor() )
            {
               delete solvingNode;
            }
            else
            {
               paraNodePoolToRestart->insert(solvingNode); // to stand a safety side about timing issue. two branch nodes may be romved.
            }
            return 0;
         }

         if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
               paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
               runningPhase == TerminationPhase )
         {
            // Exceptional case (reqested to a node whoes root node was solving), but it happend.
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
            if( solvingNode->areNodesCollected() )
            {
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " Nodes generated by S." << source << " from " << solvingNode->toSimpleString() << " are collected to LC." << std::endl;
               }
               delete solvingNode;
               nCollectedSolvers++;
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest 1" << std::endl;
            }
            else
            {
               nCollectedSolvers++;
               delete solvingNode;
            }
#ifdef UG_WITH_ZLIB
            if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
                  ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 ) ||
                  ( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 &&
                               paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) ) )
            {
               updateCheckpointFiles();
            }
#endif
         }
         else
         {
            if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverInCollectingMode(source) )
            {
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, calcState->getNSolved(),paraNodePool);
               // reschedule collecting mode
               if( !paraNodePoolBufferToRestart )
               {
                  double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               }
            }
            else
            {
               if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverActive(source) ) // the solver can be inactive for timing issue
               {
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, calcState->getNSolved(),paraNodePool);
               }
            }
         }
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      }
      // std::cout << "Rank" << source
      //     << ", lcts.best = " << lcts.externalGlobalBestDualBoundValue
      //      << ", bound = " << paraInitiator->convertToExternalValue(calcState->getDualBoundValue())
      //     << ", gap = " << std::setprecision(5) << paraInitiator->getGap(calcState->getDualBoundValue())*100 << "%" << std::endl;
      if( !EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, paraInitiator->getEpsilon() ) )
      {
         allCompInfeasibleAfterSolution = false;
         if( EPSLE(lcts.globalBestDualBoundValue, calcState->getDualBoundValue(), paraInitiator->getEpsilon()) &&
               minmalDualBoundNormalTermSolvers > calcState->getDualBoundValue() )
         {
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
            }
            else
            {
               minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
            }
         }
         /*
         if( // maximal dual bound value of terminated solver should be taken.
             // Therefore, the gap value is better than the real value
               calcState->getDualBoundValue() <
               std::min(
                     std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                     minmalDualBoundNormalTermSolvers ) &&
               lcts.globalBestDualBoundValue < calcState->getDualBoundValue() )
         {
            lcts.globalBestDualBoundValue = calcState->getDualBoundValue();
            lcts.externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
            // std::cout << "Updated Rank" << source
            //      << ", lcts.best = " << lcts.externalGlobalBestDualBoundValue
            //      << ", bound = " << paraInitiator->convertToExternalValue(calcState->getDualBoundValue())
            //      << ", gap = " << std::setprecision(5) << paraInitiator->getGap(calcState->getDualBoundValue())*100 << "%" << std::endl;
         }
         */
      }
      // DO NOT send ParaNode here!
      // Send ParaNode after solver termination state is received.
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )
         {
            PARA_COMM_CALL(
                  paraComm->send( NULL, 0, ParaBYTE, source, TagTerminateRequest )
            );
// std::cout << "TagTerminateRequest 1" << std::endl;
            paraSolverPool->terminateRequested(source);
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               int token[2];
               token[0] = source;
               token[1] = -2;
               PARA_COMM_CALL(
                     paraComm->send( token, 2, ParaINT, token[0], TagToken )
               );
            }
         }
      }
      break;
   }
   case CompInterruptedInRacingStage:
   {
      // DO NOT send ParaNode here!
      // Send ParaNode after solver termination state is received and RacingSolver is inactivated.
      // Do not have to update counters of ParaSolverPool.
      break;
   }
   case CompTerminatedByAnotherTask:
   {
      /** in this case the following two numbers should be different */
      /** # Total > # Solved */
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      // dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, calcState->getNSolved(),paraNodePool);  // Keep running to get another task
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )
         {
            PARA_COMM_CALL(
                  paraComm->send( NULL, 0, ParaBYTE, source, TagTerminateRequest )
            );
// std::cout << "TagTerminateRequest 2" << std::endl;
            paraSolverPool->terminateRequested(source);
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               int token[2];
               token[0] = source;
               token[1] = -2;
               PARA_COMM_CALL(
                     paraComm->send( token, 2, ParaINT, token[0], TagToken )
               );
            }
         }
      }
      break;
   }
   case CompTerminatedByInterruptRequest:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      // dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, calcState->getNSolved(),paraNodePool);    // this does not work
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )
         {
            writeSubtreeInfo(source, calcState);
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
            if( solvingNode->areNodesCollected() )
            {
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " Nodes generated by S." << source << " from " << solvingNode->toSimpleString() << " are collected to LC." << std::endl;
               }
               if( calcState->getNSolved() > 1 ||
                     ( calcState->getNSolved() >= 1 && calcState->getNSent() > 0 ) )
               {
                  delete solvingNode;
               }
               else
               {
                  paraNodePool->insert(solvingNode);
               }
               nCollectedSolvers++;
#ifdef UG_WITH_ZLIB
               if( ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 )
               {
                  updateCheckpointFiles();
               }
#endif
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest 2" << std::endl;
            }
            else
            {
               paraNodePool->insert(solvingNode);
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest 3" << std::endl;
            }
         }
         break;
      }
      if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )  // RacingSolverPool is inactivated below
      {
         // paraRacingSolverPool entry is inactivated, when it receives ParaSolverTerminationState message in below.
         BbParaNode *solvingNode = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool);
         if( paraNodePoolToRestart )
         {
            if( solvingNode->getAncestor() )
            {
               delete solvingNode;
            }
            else
            {
               paraNodePoolToRestart->insert(solvingNode);
            }
         }
         else
         {
            paraNodePool->insert(solvingNode);
         }
      }
      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   case CompTerminatedInRacingStage:
   {
      racingTermination = true; // even if interruptIsRequested, 
                                // solver should have been terminated before receiveing it
      if( osStatisticsRacingRampUp )
      {
         *osStatisticsRacingRampUp << "######### Solver Rank = " <<
               source << " is terminated in racing stage #########" << std::endl;
      }
      nSolvedRacingTermination = calcState->getNSolved();

      if( !EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, bbParaInitiator->getEpsilon() ) &&
            EPSEQ( minmalDualBoundNormalTermSolvers, DBL_MAX, bbParaInitiator->getEpsilon() ) )
      {
         if( bbParaInitiator->getGlobalBestIncumbentSolution() )
         {
            minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
         }
         else
         {
            minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
         }
      }
      if( EPSLE(lcts.globalBestDualBoundValue, calcState->getDualBoundValue(), bbParaInitiator->getEpsilon()) &&
            minmalDualBoundNormalTermSolvers < calcState->getDualBoundValue() )
      {
         if( bbParaInitiator->getGlobalBestIncumbentSolution() )
         {
            minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
         }
         else
         {
            minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
         }
      }
      if( (!givenGapIsReached) && bbParaInitiator->getGlobalBestIncumbentSolution() &&
            ( EPSEQ( calcState->getDualBoundValue(), bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), paraInitiator->getEpsilon() ) ||
                  EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, bbParaInitiator->getEpsilon() ) ) )
      {
         lcts.globalBestDualBoundValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
         if( paraNodePool->getNumOfNodes() > 0 )
         {
            lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
         }
         lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
      }
      break;
   }
   case CompInterruptedInMerging:
   {
      BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
      if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
      {
         // std::cout << "S." << source << " is interrupted." << std::endl;
         if( solvingNode->getMergeNodeInfo() )
         {
            assert(nodesMerger);
            nodesMerger->regenerateMergeNodesCandidates(solvingNode, paraComm, paraInitiator);
            paraNodePool->insert(solvingNode);
         }
         // else, node shuld be in paraNodePoolBufferToGenerateCPF.
      }
      else
      {
         assert(solvingNode->getMergeNodeInfo());
         assert(nodesMerger);
         nodesMerger->regenerateMergeNodesCandidates(solvingNode,paraComm, paraInitiator);
         paraNodePool->insert(solvingNode);
      }
      break;
   }
   case CompTerminatedByTimeLimit:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0)  &&
            runningPhase == TerminationPhase )
      {
         if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )
         {
            writeSubtreeInfo(source, calcState);
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
            if( solvingNode->areNodesCollected() )
            {
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " Nodes generated by S." << source << " from " << solvingNode->toSimpleString() << " are collected to LC." << std::endl;
               }
               if( calcState->getNSolved() > 1 ||
                     ( calcState->getNSolved() >= 1 && calcState->getNSent() > 0 ) )
               {
                  delete solvingNode;
               }
               else
               {
                  paraNodePool->insert(solvingNode);
               }
               nCollectedSolvers++;
#ifdef UG_WITH_ZLIB
               if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
                     ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 ) ||
                     ( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 &&
                                  paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) ) )
               {
                  updateCheckpointFiles();
               }
#endif
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest 4" << std::endl;
            }
            else
            {
               paraNodePool->insert(solvingNode);
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest 5" << std::endl;
            }
         }
         break;
      }
      if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )  // RacingSolverPool is inactivated below
      {
         // paraRacingSolverPool entry is inactivated, when it receives ParaSolverTerminationState message in below.
         BbParaNode *solvingNode = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool);
         if( paraNodePoolToRestart )
         {
            if( solvingNode )  // solvingNode can be NULL
            {
               if( solvingNode->getAncestor() )
               {
                  delete solvingNode;
               }
               else
               {
                  paraNodePoolToRestart->insert(solvingNode);
               }
            }
         }
         else
         {
            if( solvingNode )   // solvingNode can be NULL
            {
               paraNodePool->insert(solvingNode);
            }
         }
      }
      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   case CompTerminatedByMemoryLimit:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0)  &&
            runningPhase == TerminationPhase )
      {
         if( paraRacingSolverPool && paraRacingSolverPool->isSolverActive(source) )
         {
            if( dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNumInactiveSolvers() == 0 )
            {
               /// when it is the first solver terminated (this have to be checked very carefully for restart racing (no debug)
               BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->extractNode());
               paraNodePool->insert(solvingNode);
            }
            dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->inactivateSolver(source);
         }
         else
         {
            writeSubtreeInfo(source, calcState);
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
            paraNodePool->insert(solvingNode);

         }
#ifdef UG_WITH_ZLIB
         if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
               ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 ) ||
               ( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 &&
                            paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) ) )
         {
            updateCheckpointFiles();
         }
#endif
      }
      sendInterruptRequest();
// std::cout << "sendInterrruptRequest 6" << std::endl;
      runningPhase = TerminationPhase;
      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   default:
      THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
   }

   if( calcState->getTerminationState() == CompTerminatedByTimeLimit )
   {
      hardTimeLimitIsReached = true;
      // std::cout << "####### Rank " << paraComm->getRank() << " solver terminated with timelimit in solver side. #######" << std::endl;
      // std::cout << "####### Final statistics may be messed up!" << std::endl;
   }

   if( calcState->getTerminationState() == CompTerminatedByMemoryLimit )
   {
      memoryLimitIsReached = true;
   }

   delete calcState;

   return 0;
}

int
BbParaLoadCoordinator::processTagTermStateForInterruption(
      int source,
      int tag
      )
{

   ParaSolverTerminationState *termState = paraComm->createParaSolverTerminationState();

   termState->receive(paraComm, source, TagTermStateForInterruption);

   if( paraDetTimer )
   {
      if( paraDetTimer->getElapsedTime() < termState->getDeterministicTime() )
      {
         paraDetTimer->update( termState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
      }
      // assert( !paraRacingSolverPool );   // can have paraRacingSolverPool, but it would be ok to do as follows
      // std::cout << "paraRacingSolverPool =  " << paraRacingSolverPool << std::endl;
      // std::cout << "paraSolverPool->isTerminateRequested(source) = " << paraSolverPool->isTerminateRequested(source) << std::endl;
      // if( paraRacingSolverPool )
      // {
      //    std::cout << "paraRacingSolverPool->isSolverActive(source) = " << paraRacingSolverPool->isSolverActive(source) << std::endl;
      // }
      if( !paraSolverPool->isTerminateRequested(source) ||
          ( paraRacingSolverPool && paraRacingSolverPool->isSolverActive(source) ) )
      {
         PARA_COMM_CALL(
               paraComm->send( NULL, 0, ParaBYTE, source, TagAckCompletion )
         );
      }
   }

   switch( termState->getInterruptedMode() )
   {
   case 2: /** checkpoint; This is normal termination */
   {
      /** in order to save termination status to check point file, keep this information to solver pool */
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setTermState(source, termState);
      // don't delete termState! it is saved in paraSolverPool
      if( runningPhase != TerminationPhase &&
            !paraNodePoolToRestart &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByTimeLimit &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByMemoryLimit )
      {
         if( paraNodePool->isEmpty() )
         {
            lcts.nFailedToSendBack++;
            if( runningPhase != RampUpPhase && !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode()) )
            {
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               if( firstCollectingModeState == -1 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) firstCollectingModeState = 0;
            }
         }
         else
         {
            if( sendParaTasksToIdleSolvers() )
            {
               lcts.nSentBackImmediately++;
            }
         }
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByTimeLimit )
      {
         hardTimeLimitIsReached = true;
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByMemoryLimit )
      {
         memoryLimitIsReached = true;
      }
      break;
   }
   case 3: /** racing ramp-up */
   {
      if( osStatisticsRacingRampUp )
      {
         *osStatisticsRacingRampUp << termState->toString(paraInitiator);
         osStatisticsRacingRampUp->flush();
      }
      // nTerminated++;      We should not count this, We should always send Term from LC!
      // there is a timming to receive this message after paraRacingSolverPool is removed
      if( paraRacingSolverPool )
      {
         inactivateRacingSolverPool(source);
      }
      /*  Anyway, incumbent value is sent to all Solvers except its generator. In such case, this is not necessary.
      double globalBestIncumbentValue = paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue();
      PARA_COMM_CALL(
            paraComm->send( &globalBestIncumbentValue, 1, ParaDOUBLE, source, TagIncumbentValue )
      );
      */
      if( runningPhase != TerminationPhase &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByTimeLimit &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByMemoryLimit )
      {
         if( paraNodePool->isEmpty() )
         {
            lcts.nFailedToSendBack++;
            if( runningPhase != RampUpPhase && !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode()) )
            {
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               if( firstCollectingModeState == -1 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) firstCollectingModeState = 0;
            }
         }
         else
         {
            if( sendParaTasksToIdleSolvers() )
            {
               lcts.nSentBackImmediately++;
            }
         }
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByTimeLimit )
      {
         hardTimeLimitIsReached = true;
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByMemoryLimit )
      {
         memoryLimitIsReached = true;
      }
      delete termState;
      break;
   }
   default:  /** unexpected mode */
      THROW_LOGICAL_ERROR4("Unexpected termination state received from rank = ", source,
            ", interrupted mode = ", termState->getInterruptedMode());
   }

   if( dynamic_cast<BbParaSolverPool *>(paraSolverPool)->isInterruptRequested(source)
         && !dynamic_cast<BbParaSolverPool *>(paraSolverPool)->getCurrentTask(source) )   // solver is interrupted
   {
      dynamic_cast<BbParaSolverPool *>(paraSolverPool)->inactivateSolver(source, -1, paraNodePool);
   }

   if( racingTermination && dynamic_cast<BbParaSolverPool *>(paraSolverPool)->isTerminateRequested(source) )
   {
      // nTerminated++;
      return 0;
   }

#ifndef _COMM_MPI_WORLD
   if( paraParams->getBoolParamValue(Quiet) && racingTermination )
   {
      if( nTerminated == 0 )
      {
         /** in this case, do not have to wait statistical information from the other solvers */
         // nTerminated = 1;
         terminateAllSolvers();
         // std::cout << "UG_BB nTerminated = " << nTerminated << std::endl;
         delete this;
      }
      else
      {
         // THROW_LOGICAL_ERROR2("unexpated termination received. nTeminated = ", nTerminated);
      }

#ifdef _COMM_PTH
      _exit(0);
#else
      exit(0);
#endif
   }
#endif

   if( source == breakingSolverId )
   {
      breakingSolverId = -1;
      isBreakingFinised = false;
   }
   return 0;
}

int
BbParaLoadCoordinator::processTagAnotherNodeRequest(
      int source,
      int tag
      )
{

   double bestDualBoundValue;
   PARA_COMM_CALL(
         paraComm->receive( &bestDualBoundValue, 1, ParaDOUBLE, source, TagAnotherNodeRequest)
         );

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( paraNodePool->isEmpty() || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->currentSolvingNodehaeDescendant(source) )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, source, TagNoNodes)
            );
      lcts.nFailedToSendBackAnotherNode++;
   }
   else
   {
      BbParaNode *paraNode = 0;
      while( !paraNodePool->isEmpty() )
      {
         paraNode = paraNodePool->extractNode();
         if( !paraNode ) break;
         if( ( bbParaInitiator->getGlobalBestIncumbentSolution() &&
               paraNode->getDualBoundValue() < bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() ) ||
               !( bbParaInitiator->getGlobalBestIncumbentSolution() ) )
         {
            break;
         }
         else
         {
#ifdef UG_DEBUG_SOLUTION
            if( paraNode->getDiffSubproblem() && paraNode->getDiffSubproblem()->isOptimalSolIncluded() )
            {
                throw "Optimal solution going to be killed.";
            }
#endif
            delete paraNode;
            paraNode = 0;
            lcts.nDeletedInLc++;
         }
      }
      if( paraNode )
      {
         if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverActive(source) &&    // can be interrupting
               ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getDualBoundValue(source) - paraNode->getDualBoundValue()) > 0.0 &&
               ( REALABS( ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getDualBoundValue(source) - paraNode->getDualBoundValue() )
                     / std::max( std::fabs(paraNode->getDualBoundValue()), 1.0) ) > paraParams->getRealParamValue(BgapStopSolvingMode) ) )
         {
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->sendSwitchOutCollectingModeIfNecessary(source);
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
            solvingNode->setDualBoundValue(bestDualBoundValue);
            solvingNode->setInitialDualBoundValue(bestDualBoundValue);
            paraNodePool->insert(solvingNode);
            if( solvingNode->getMergeNodeInfo() )
            {
               assert(nodesMerger);
               nodesMerger->mergeNodes(solvingNode, paraNodePool);
            }
            // without consideration of keeping nodes in checkpoint file
            double globalBestDualBoundValueLocal =
               std::max (
                     std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                     lcts.globalBestDualBoundValue );
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->activateSolver(source, paraNode,
                  paraNodePool->getNumOfGoodNodes(globalBestDualBoundValueLocal), averageLastSeveralDualBoundGains);
            // paraNode->send(paraComm, source); // send the node in acitivateSolver
            lcts.nSent++;
            lcts.nSentBackImmediatelyAnotherNode++;
            writeTransferLog(source);
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source << " <(ANOTHER_NODE) "
               << bbParaInitiator->convertToExternalValue(
                     paraNode->getDualBoundValue() );
               if( bbParaInitiator->getGlobalBestIncumbentSolution() )
               {
                  if( bbParaInitiator->getGap(paraNode->getDualBoundValue()) > displayInfOverThisValue )
                  {
                     *osLogSolvingStatus << " ( Inf )";
                  }
                  else
                  {
                     *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(paraNode->getDualBoundValue()) * 100 << "% )";
                  }
               }
               *osLogSolvingStatus << std::endl;
            }
#ifdef _DEBUG_LB
            std::cout << paraTimer->getElapsedTime()
            << " S." << source << " <(ANOTHER_NODE) "
            << paraInitiator->convertToExternalValue(
                  paraNode->getDualBoundValue() );
            if( paraInitiator->getGlobalBestIncumbentSolution() )
            {
               if( paraInitiator->getGap(paraNode->getDualBoundValue()) > displayInfOverThisValue )
               {
                  std::cout << " ( Inf )";
               }
               else
               {
                  std::cout << " ( " << paraInitiator->getGap(paraNode->getDualBoundValue()) * 100 << "% )";
               }
            }
            std::cout << std::endl;
#endif
         }
         else
         {
            paraNodePool->insert(paraNode);
            PARA_COMM_CALL(
                  paraComm->send( NULL, 0, ParaBYTE, source, TagNoNodes)
                  );
            lcts.nFailedToSendBackAnotherNode++;
         }
      }
      else
      {
         PARA_COMM_CALL(
               paraComm->send( NULL, 0, ParaBYTE, source, TagNoNodes)
               );
         lcts.nFailedToSendBackAnotherNode++;
      }
   }
   return 0;
}

int
BbParaLoadCoordinator::processTagAllowToBeInCollectingMode(
      int source,
      int tag
      )
{

   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagAllowToBeInCollectingMode)
         );
   dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setCollectingIsAllowed(source);

   return 0;
}

int
BbParaLoadCoordinator::processTagLbBoundTightened(
      int source,
      int tag
      )
{

   int tightenedIdex;
   double tightenedBound;
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedIdex, 1, ParaINT, source, TagLbBoundTightenedIndex )
         );
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedBound, 1, ParaDOUBLE, source, TagLbBoundTightenedBound )
         );
   if( EPSLT(dynamic_cast<BbParaInitiator *>(paraInitiator)->getTightenedVarLbs(tightenedIdex), tightenedBound, MINEPSILON ) )
   {
	  // std::cout << "From Rank " << source << ": in initiator LB = " << paraInitiator->getTightenedVarLbs(tightenedIdex) << ", rv = " << tightenedBound << std::endl;
      dynamic_cast<BbParaInitiator *>(paraInitiator)->setTightenedVarLbs(tightenedIdex, tightenedBound);
      if( paraRacingSolverPool && ( paraRacingSolverPool->getNumInactiveSolvers() == 0 ) )
      {
         for( size_t i = 1; i <= paraRacingSolverPool->getNumActiveSolvers(); i++ )
         {
            if( static_cast<int>(i) != source )
            {
               PARA_COMM_CALL(
                     paraComm->send( (void *)&tightenedIdex, 1, UG::ParaINT, i, UG::TagLbBoundTightenedIndex )
                     );
               PARA_COMM_CALL(
                     paraComm->send( (void *)&tightenedBound, 1, UG::ParaDOUBLE, i, UG::TagLbBoundTightenedBound )
                     );
            }
         }
         // std::cout << "From Rank " << source << ": broadcast tightened lower bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
      }
   }

   return 0;
}

int
BbParaLoadCoordinator::processTagUbBoundTightened(
      int source,
      int tag
      )
{

   int tightenedIdex;
   double tightenedBound;
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedIdex, 1, ParaINT, source, TagUbBoundTightenedIndex )
         );
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedBound, 1, ParaDOUBLE, source, TagUbBoundTightenedBound )
         );
   if( EPSGT(dynamic_cast<BbParaInitiator *>(paraInitiator)->getTightenedVarUbs(tightenedIdex), tightenedBound, MINEPSILON ) )
   {
	  // std::cout << "From Rank " << source << ": in initiator UB = " << paraInitiator->getTightenedVarUbs(tightenedIdex) << ", rv = " << tightenedBound << std::endl;
      dynamic_cast<BbParaInitiator *>(paraInitiator)->setTightenedVarUbs(tightenedIdex, tightenedBound);
      if( paraRacingSolverPool && ( paraRacingSolverPool->getNumInactiveSolvers() == 0 ) )
      {
         for( size_t i = 1; i <= paraRacingSolverPool->getNumActiveSolvers(); i++ )
         {
            if( static_cast<int>(i) != source )
            {
               PARA_COMM_CALL(
                     paraComm->send( (void *)&tightenedIdex, 1, UG::ParaINT, i, UG::TagUbBoundTightenedIndex )
                     );
               PARA_COMM_CALL(
                     paraComm->send( (void *)&tightenedBound, 1, UG::ParaDOUBLE, i, UG::TagUbBoundTightenedBound )
                     );
            }
         }
         // std::cout << "From Rank " << source << ": broadcast tightened upper bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
      }
   }

   return 0;
}

int
BbParaLoadCoordinator::processTagSelfSplitFinished(
      int source,
      int tag
      )
{

  assert( tag == TagSelfSplitFinished );
  PARA_COMM_CALL(
		 paraComm->receive( NULL, 0, ParaBYTE, source, TagSelfSplitFinished )
		 );
  selfSplitFinisedSolvers->insert(source);
//  std::cout << "***TagSelfSplitFinished*** R." << source
//        << ": dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved = "
//        << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source) << std::endl;
  return 0;
}

int
BbParaLoadCoordinator::processTagNewSubtreeRootNode(
      int source,
      int tag
      )
{

   BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
   paraNode->receiveNewSubtreeRoot(paraComm, source);

//   std::cout << "S." << source
//        << ", KEEP: nBoundChanges = " << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges() << std::endl;

   assert( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );

   if( !hardTimeLimitIsReached && !memoryLimitIsReached && !givenGapIsReached )
   {
      dynamic_cast<BbParaSolverPool *>(paraSolverPool)->addNewSubtreeRootNode(source, paraNode);
   }

   assert( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );

   if( logSolvingStatusFlag )
   {
      BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " S." << source
      << " |< "
      << bbParaInitiator->convertToExternalValue(
            paraNode->getDualBoundValue() )
      << " "
      << paraNode->toSimpleString();
      if( paraNode->getDiffSubproblem() )
      {
         *osLogSolvingStatus
         << ", "
         // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
         << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->toStringStat();
      }
      *osLogSolvingStatus << std::endl;
   }
//   std::cout << "***TagNewSubtreeRootNode*** R." << source
//         << ": dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved = "
//         << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source) << std::endl;
   return 0;
}

int
BbParaLoadCoordinator::processTagSubtreeRootNodeStartComputation(
      int source,
      int tag
      )
{

   BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
   // BbParaNode *nextNode = 0;
   paraNode->receiveSubtreeRootNodeId(paraComm, source, TagSubtreeRootNodeStartComputation);
   // std::cout << "processTagSubtreeRootNodeToBeRemoved" << paraNode->toSimpleString() << std::endl;
 
   if( !hardTimeLimitIsReached && !memoryLimitIsReached && !givenGapIsReached )
   {
      dynamic_cast<BbParaSolverPool *>(paraSolverPool)->makeSubtreeRootNodeCurrent(source, paraNode);
      if( logSolvingStatusFlag )
      {
         // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source
         << " |s "
         << dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPool *>(paraSolverPool)->getCurrentTask(source))->toSimpleString();
         *osLogSolvingStatus << std::endl;
      }
   }

   assert( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );

   return 0;
}

int
BbParaLoadCoordinator::processTagSubtreeRootNodeToBeRemoved(
      int source,
      int tag
      )
{

   BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
   // BbParaNode *nextNode = 0;
   paraNode->receiveSubtreeRootNodeId(paraComm, source, TagSubtreeRootNodeToBeRemoved);
   // std::cout << "processTagSubtreeRootNodeToBeRemoved" << paraNode->toSimpleString() << std::endl;
   
   assert( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );

   if( runningPhase == TerminationPhase &&
         (!dynamic_cast<BbParaSolverPool *>(paraSolverPool)->isSolverActive(source) ) )
   {
      return 0;   // Anyway, the node is going to paraNodePool
   }

   if( !hardTimeLimitIsReached && !memoryLimitIsReached && !givenGapIsReached )
   {
      dynamic_cast<BbParaSolverPool *>(paraSolverPool)->removeSubtreeRootNode(source, paraNode);
   }
   if( logSolvingStatusFlag )
   {
      BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " S." << source
      << " |> "
      << bbParaInitiator->convertToExternalValue(
            paraNode->getDualBoundValue() )
      << " "
      << paraNode->toSimpleString();
      if( paraNode->getDiffSubproblem() )
      {
         *osLogSolvingStatus
         << ", "
         // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
         << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->toStringStat();
      }
      *osLogSolvingStatus << std::endl;
   }

   delete paraNode;

   lcts.nDeletedInLc++;  // this node is deleted without calcutaion

   return 0;
}

int
BbParaLoadCoordinator::processTagReassignSelfSplitSubtreeRootNode(
      int source,
      int tag
      )
{

   BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
   // BbParaNode *nextNode = 0;
   paraNode->receiveSubtreeRootNodeId(paraComm, source, TagReassignSelfSplitSubtreeRootNode);
   // std::cout << "processTagReassignSelfSplitSubtreeRootNode" << paraNode->toSimpleString() << std::endl;

   assert( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached || dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getCurrentTask(source) != 0 );

//   if( runningPhase == TerminationPhase )
//   {
//      return 0;   // Anyway, the node is going to paraNodePool
//   }

   if( !hardTimeLimitIsReached && !memoryLimitIsReached && !givenGapIsReached )
   {
      BbParaNode *reassignedNode = dynamic_cast<BbParaSolverPool *>(paraSolverPool)->extractSelfSplitSubtreeRootNode(source, paraNode);

      if( logSolvingStatusFlag )
      {
         BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source
         << " |>p< "
         << bbParaInitiator->convertToExternalValue(
               reassignedNode->getDualBoundValue() )
         << " "
         << reassignedNode->toSimpleString();
         if( reassignedNode->getDiffSubproblem() )
         {
            *osLogSolvingStatus
             << ", "
             // << dynamic_cast<BbParaDiffSubproblem *>(reassignedNode->getDiffSubproblem())->getNBoundChanges()
             << dynamic_cast<BbParaDiffSubproblem *>(reassignedNode->getDiffSubproblem())->toStringStat();
         }
         *osLogSolvingStatus << std::endl;
      }

      paraNodePool->insert(reassignedNode);

   }

   delete paraNode;

   return 0;
}

int
BbParaLoadCoordinator::processTagSelfSlpitNodeCalcuationState(
      int source,
      int tag
      )
{

   BbParaCalculationState *calcState = dynamic_cast<BbParaCalculationState *>(paraComm->createParaCalculationState());
   calcState->receive(paraComm, source, tag);

   writeTransferLog(source, calcState);

   int calcTerminationState = calcState->getTerminationState();
   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
   BbParaSolverPool *bbParaSolverPool = dynamic_cast<BbParaSolverPool *>(paraSolverPool);
   if( logSolvingStatusFlag )
   {
      switch ( calcTerminationState )
      {
      case CompTerminatedNormally:
      {
         if( bbParaSolverPool->getSelfSplitSubtreeRootNodes(source) )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " |> " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
            << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         }
         else
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " > " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
            << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         }
         break;
      }
      case CompTerminatedByAnotherTask:
      {
         if( bbParaSolverPool->getSelfSplitSubtreeRootNodes(source) )
         {
            /** When starts with racing ramp-up, solvers except winner should be terminated in this state */
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " |>(INTERRUPTED_BY_ANOTHER_NODE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
            << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         }
         else
         {
            /** When starts with racing ramp-up, solvers except winner should be terminated in this state */
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " |>(INTERRUPTED_BY_ANOTHER_NODE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         }
         break;
      }
      case CompTerminatedByInterruptRequest:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
          << " S." << source << " >(INTERRUPTED) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
          << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         break;
      }
      case CompTerminatedByTimeLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_TIME_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
         << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         break;
      }
      case CompTerminatedByMemoryLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_MEMORY_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue())
         << ", " << bbParaSolverPool->getCurrentTask(source)->toSimpleString();
         break;
      }
      default:
         THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
      }
      *osLogSolvingStatus << ", ct:" << calcState->getCompTime()
            << ", nr:" << calcState->getNRestarts()
            << ", n:" << calcState->getNSolved()
            << ", rt:" << calcState->getRootTime()
            << ", avt:" << calcState->getAverageNodeCompTimeExcpetRoot()
            << std::endl;
   }

   switch ( calcTerminationState )
   {
   case CompTerminatedNormally:
   {
      writeSubtreeInfo(source, calcState);
      // std::cout << "*** R." << source << ", calcState->getNSolved() = " << calcState->getNSolved() << std::endl;
      // BbParaNode *node = dynamic_cast<BbParaNode *>(bbParaSolverPool->getCurrentTask(source));
      // assert( (node->next) );    // should always has next. node->next == null should be processed in processTagCompletionOfCalculation
                                    // No: can be null, when a node is reassigned
      assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );  // RacingSolverPool is inactivated below

      assert( !paraNodePoolToRestart );

      bbParaSolverPool->resetCountersInSolver(source, calcState->getNSolved(), calcState->getNSelfSplitNodesLeft(), paraNodePool);
      if( bbParaSolverPool->isSolverInCollectingMode(source) )
      {
         // reschedule collecting mode
         double tempTime = bbParaSolverPool->getSwichOutTime();
         bbParaSolverPool->switchOutCollectingMode();
         bbParaSolverPool->setSwichOutTime(tempTime);
         bbParaSolverPool->switchInCollectingMode(paraNodePool);
      }
      // delete node; // should not delete
      bbParaSolverPool->addTotalNodesSolved(calcState->getNSolved());

      // std::cout << "Rank" << source
      //     << ", lcts.best = " << lcts.externalGlobalBestDualBoundValue
      //      << ", bound = " << paraInitiator->convertToExternalValue(calcState->getDualBoundValue())
      //     << ", gap = " << std::setprecision(5) << paraInitiator->getGap(calcState->getDualBoundValue())*100 << "%" << std::endl;
      if( !EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, paraInitiator->getEpsilon() ) )
      {
         allCompInfeasibleAfterSolution = false;
         if( EPSLE(lcts.globalBestDualBoundValue, calcState->getDualBoundValue(), paraInitiator->getEpsilon()) &&
               minmalDualBoundNormalTermSolvers > calcState->getDualBoundValue() )
         {
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
            }
            else
            {
               minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
            }
         }
      }

      if( !bbParaSolverPool->getSelfSplitSubtreeRootNodes(source) )
      {
         // dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, calcState->getNSolved(),paraNodePool);
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(source, -1, paraNodePool);  // already updated in above
      }
      else
      {
         bbParaSolverPool->deleteCurrentSubtreeRootNode(source);
      }

      break;
   }
   case CompTerminatedByAnotherTask:
   {
      /** in this case the following two numbers should be different */
      /** # Total > # Solved */
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         if( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) )
         {
            PARA_COMM_CALL(
                  paraComm->send( NULL, 0, ParaBYTE, source, TagTerminateRequest )
            );
// std::cout << "TagTerminateRequest 3" << std::endl;
            paraSolverPool->terminateRequested(source);
            if( paraParams->getBoolParamValue(Deterministic) )
            {
               int token[2];
               token[0] = source;
               token[1] = -2;
               PARA_COMM_CALL(
                     paraComm->send( token, 2, ParaINT, token[0], TagToken )
               );
            }
         }
      }
      break;
   }
   case CompTerminatedByInterruptRequest:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );
         writeSubtreeInfo(source, calcState);
         BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
         if( solvingNode->areNodesCollected() )
         {
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " Nodes generated by S." << source << " from " << solvingNode->toSimpleString() << " are collected to LC." << std::endl;
            }
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePool->insert(temp);
               if( logSolvingStatusFlag )
               {
                  // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << source
                  << " p< "
                  << bbParaInitiator->convertToExternalValue(
                        temp->getDualBoundValue() )
                  << " "
                  << temp->toSimpleString();
                  if( temp->getDiffSubproblem() )
                  {
                     *osLogSolvingStatus
                     << ", "
                     // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                     << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
                  }
                  *osLogSolvingStatus << std::endl;
               }
            }
            if( calcState->getNSolved() > 1 ||
                  ( calcState->getNSolved() >= 1 && calcState->getNSent() > 0 ) )
            {
               delete solvingNode;
            }
            else
            {
               paraNodePool->insert(solvingNode);
            }
            nCollectedSolvers++;
#ifdef UG_WITH_ZLIB
            if( ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 )
            {
               updateCheckpointFiles();
            }
#endif
            sendInterruptRequest();
// std::cout << "sendInterrruptRequest 7" << std::endl;
         }
         else
         {
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePool->insert(temp);
               if( logSolvingStatusFlag )
               {
                  // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << source
                  << " p< "
                  << bbParaInitiator->convertToExternalValue(
                        temp->getDualBoundValue() )
                  << " "
                  << temp->toSimpleString();
                  if( temp->getDiffSubproblem() )
                  {
                     *osLogSolvingStatus
                     << ", "
                     // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                     << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
                  }
                  *osLogSolvingStatus << std::endl;
               }
            }
            paraNodePool->insert(solvingNode);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     solvingNode->getDualBoundValue() )
               << " "
               << solvingNode->toSimpleString();
               if( solvingNode->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(solvingNode->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
#ifdef UG_WITH_ZLIB
            updateCheckpointFiles();
#endif
         }
         break;
      }
      assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );  // RacingSolverPool is inactivated below

      // paraRacingSolverPool entry is inactivated, when it receives ParaSolverTerminationState message in below.
      // BbParaNode *solvingNode = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool);
      if( paraNodePoolToRestart )
      {
         assert( !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool))->getAncestor() );
         BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
         while( nodes )
         {
            BbParaNode *temp = nodes;
            nodes = nodes->next;
            temp->next = 0;
            paraNodePoolToRestart->insert(temp);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " rp< "
               << bbParaInitiator->convertToExternalValue(
                     temp->getDualBoundValue() )
               << " "
               << temp->toSimpleString();
               if( temp->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
      }
      else
      {
         BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
         while( nodes )
         {
            BbParaNode *temp = nodes;
            nodes = nodes->next;
            temp->next = 0;
            paraNodePool->insert(temp);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     temp->getDualBoundValue() )
               << " "
               << temp->toSimpleString();
               if( temp->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
      }

      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   case CompTerminatedByTimeLimit:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      hardTimeLimitIsReached = true;
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );
         writeSubtreeInfo(source, calcState);
         BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
         if( solvingNode->areNodesCollected() )
         {
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " Nodes generated by S." << source << " from " << solvingNode->toSimpleString() << " are collected to LC." << std::endl;
            }
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePool->insert(temp);
            }
            if( calcState->getNSolved() > 1 ||
                  ( calcState->getNSolved() >= 1 && calcState->getNSent() > 0 ) )
            {
               delete solvingNode;
            }
            else
            {
               paraNodePool->insert(solvingNode);
            }
            nCollectedSolvers++;
#ifdef UG_WITH_ZLIB
            if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
                  ( nCollectedSolvers % paraParams->getIntParamValue(UG::EnhancedCheckpointInterval) ) == 0 ) ||
                  ( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 &&
                               paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) ) )
            {
               updateCheckpointFiles();
            }
#endif
            sendInterruptRequest();
// std::cout << "sendInterrruptRequest 8" << std::endl;
         }
         else
         {
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePool->insert(temp);
               if( logSolvingStatusFlag )
               {
                  // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << source
                  << " p< "
                  << bbParaInitiator->convertToExternalValue(
                        temp->getDualBoundValue() )
                  << " "
                  << temp->toSimpleString();
                  if( temp->getDiffSubproblem() )
                  {
                     *osLogSolvingStatus
                     << ", "
                     // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                     << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
                  }
                  *osLogSolvingStatus << std::endl;
               }
            }
#ifdef UG_WITH_ZLIB
            updateCheckpointFiles();
#endif
            paraNodePool->insert(solvingNode);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     solvingNode->getDualBoundValue() )
               << " "
               << solvingNode->toSimpleString();
               if( solvingNode->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(solvingNode->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
         break;
      }
      assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );  // RacingSolverPool is inactivated below
      // paraRacingSolverPool entry is inactivated, when it receives ParaSolverTerminationState message in below.
      if( paraNodePoolToRestart )
      {
         if( paraNodePoolToRestart )
         {
            assert( !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool))->getAncestor() );
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePoolToRestart->insert(temp);
               if( logSolvingStatusFlag )
               {
                  // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << source
                  << " rp< "
                  << bbParaInitiator->convertToExternalValue(
                        temp->getDualBoundValue() )
                  << " "
                  << temp->toSimpleString();
                  if( temp->getDiffSubproblem() )
                  {
                     *osLogSolvingStatus
                     << ", "
                     // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                     << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
                  }
                  *osLogSolvingStatus << std::endl;
               }
            }
         }
      }
      else
      {
         BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
         while( nodes )
         {
            BbParaNode *temp = nodes;
            nodes = nodes->next;
            temp->next = 0;
            paraNodePool->insert(temp);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     temp->getDualBoundValue() )
               << " "
               << temp->toSimpleString();
               if( temp->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
      }
      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   case CompTerminatedByMemoryLimit:
   {
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addNumNodesSolved( (calcState->getNSolved() -
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNumOfNodesSolved(source)) );
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->addTotalNodesSolved(calcState->getNSolved());
      memoryLimitIsReached = true;
      if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
            paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
            runningPhase == TerminationPhase )
      {
         assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );
         writeSubtreeInfo(source, calcState);
         BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool));
         BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
         while( nodes )
         {
            BbParaNode *temp = nodes;
            nodes = nodes->next;
            temp->next = 0;
            paraNodePool->insert(temp);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     temp->getDualBoundValue() )
               << " "
               << temp->toSimpleString();
               if( temp->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
#ifdef UG_WITH_ZLIB
         updateCheckpointFiles();
#endif
         paraNodePool->insert(solvingNode);
         if( logSolvingStatusFlag )
         {
            // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source
            << " p< "
            << bbParaInitiator->convertToExternalValue(
                  solvingNode->getDualBoundValue() )
            << " "
            << solvingNode->toSimpleString();
            if( solvingNode->getDiffSubproblem() )
            {
               *osLogSolvingStatus
               << ", "
               // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
               << dynamic_cast<BbParaDiffSubproblem *>(solvingNode->getDiffSubproblem())->toStringStat();
            }
            *osLogSolvingStatus << std::endl;
         }
         sendInterruptRequest();
// std::cout << "sendInterrruptRequest 9" << std::endl;
         runningPhase = TerminationPhase;
         break;
      }
      assert( (!paraRacingSolverPool) || (!paraRacingSolverPool->isSolverActive(source) ) );  // RacingSolverPool is inactivated below
      // paraRacingSolverPool entry is inactivated, when it receives ParaSolverTerminationState message in below.
      if( paraNodePoolToRestart )
      {
         if( paraNodePoolToRestart )
         {
            assert( !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractCurrentNodeAndInactivate(source, paraNodePool))->getAncestor() );
            BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
            while( nodes )
            {
               BbParaNode *temp = nodes;
               nodes = nodes->next;
               temp->next = 0;
               paraNodePoolToRestart->insert(temp);
               if( logSolvingStatusFlag )
               {
                  // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << source
                  << " rp< "
                  << bbParaInitiator->convertToExternalValue(
                        temp->getDualBoundValue() )
                  << " "
                  << temp->toSimpleString();
                  if( temp->getDiffSubproblem() )
                  {
                     *osLogSolvingStatus
                     << ", "
                     // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                     << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
                  }
                  *osLogSolvingStatus << std::endl;
               }
            }
         }
      }
      else
      {
         BbParaNode *nodes = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->extractSelfSplitSubtreeRootNodes(source);
         while( nodes )
         {
            BbParaNode *temp = nodes;
            nodes = nodes->next;
            temp->next = 0;
            paraNodePool->insert(temp);
            if( logSolvingStatusFlag )
            {
               // BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source
               << " p< "
               << bbParaInitiator->convertToExternalValue(
                     temp->getDualBoundValue() )
               << " "
               << temp->toSimpleString();
               if( temp->getDiffSubproblem() )
               {
                  *osLogSolvingStatus
                  << ", "
                  // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges()
                  << dynamic_cast<BbParaDiffSubproblem *>(temp->getDiffSubproblem())->toStringStat();
               }
               *osLogSolvingStatus << std::endl;
            }
         }
      }
      sendInterruptRequest();
// std::cout << "sendInterrruptRequest1 10" << std::endl;
      runningPhase = TerminationPhase;
      //
      // no update lcts.globalBestDualBoundValue and lcts.externalGlobalBestDualBoundValue
      // just use SolerState update
      //
      break;
   }
   default:
      THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
   }

//   if( calcState->getTerminationState() == CompTerminatedByTimeLimit )
//   {
//      hardTimeLimitIsReached = true;
//      // std::cout << "####### Rank " << paraComm->getRank() << " solver terminated with timelimit in solver side. #######" << std::endl;
//      // std::cout << "####### Final statistics may be messed up!" << std::endl;
//   }

   delete calcState;

   return 0;
}

int
BbParaLoadCoordinator::processTagSelfSplitTermStateForInterruption(
      int source,
      int tag
      )
{

   ParaSolverTerminationState *termState = paraComm->createParaSolverTerminationState();

   termState->receive(paraComm, source, TagSelfSplitTermStateForInterruption);

   if( paraDetTimer )
   {
      if( paraDetTimer->getElapsedTime() < termState->getDeterministicTime() )
      {
         paraDetTimer->update( termState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
      }
      assert( !paraRacingSolverPool );
      if( !paraSolverPool->isTerminateRequested(source) )
      {
         PARA_COMM_CALL(
               paraComm->send( NULL, 0, ParaBYTE, source, TagAckCompletion )
         );
      }
   }

   switch( termState->getInterruptedMode() )
   {
   case 2: /** checkpoint; This is normal termination */
   {
      /** in order to save termination status to check point file, keep this information to solver pool */
      dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setTermState(source, termState);
      // don't delete termState! it is saved in paraSolverPool
      if( runningPhase != TerminationPhase &&
            !paraNodePoolToRestart &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByTimeLimit &&
            dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() != CompTerminatedByMemoryLimit )
      {
         if( paraNodePool->isEmpty() )
         {
            lcts.nFailedToSendBack++;
            if( runningPhase != RampUpPhase && !(dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode()) )
            {
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
               if( firstCollectingModeState == -1 && dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isInCollectingMode() ) firstCollectingModeState = 0;
            }
         }
         else
         {
            if( sendParaTasksToIdleSolvers() )
            {
               lcts.nSentBackImmediately++;
            }
         }
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByTimeLimit )
      {
         hardTimeLimitIsReached = true;
      }
      if( dynamic_cast<BbParaSolverTerminationState *>(termState)->getCalcTerminationState() == CompTerminatedByMemoryLimit )
      {
         memoryLimitIsReached = true;
      }
      break;
   }
   default:  /** unexpected mode */
      THROW_LOGICAL_ERROR4("Unexpected termination state received from rank = ", source,
            ", interrupted mode = ", termState->getInterruptedMode());
   }

   return 0;
}

void
BbParaLoadCoordinator::outputTabularSolvingStatusHeader(
      )
{
   // output title line 1
   *osTabularSolvingStatus << std::setw(1) << " ";
   *osTabularSolvingStatus << std::setw(8) << std::right << " ";
   *osTabularSolvingStatus << std::setw(15) << std::right << " ";
   *osTabularSolvingStatus << std::setw(12) << std::right << "Nodes";
   *osTabularSolvingStatus << std::setw(10) << std::right << "Active";
   *osTabularSolvingStatus << std::setw(17) << std::right << " ";
   *osTabularSolvingStatus << std::setw(17) << std::right << " ";
   *osTabularSolvingStatus << std::setw(10) << std::right << " ";
   *osTabularSolvingStatus << std::endl;
   // output title line 2
   *osTabularSolvingStatus << std::setw(1) << " ";
   *osTabularSolvingStatus << std::setw(8) << std::right << "Time";
   *osTabularSolvingStatus << std::setw(15) << std::right << "Nodes";
   *osTabularSolvingStatus << std::setw(12) << std::right << "Left";
   *osTabularSolvingStatus << std::setw(10) << std::right << "Solvers";
   *osTabularSolvingStatus << std::setw(17) << std::right << "Best Integer";
   *osTabularSolvingStatus << std::setw(17) << std::right << "Best Node";
   *osTabularSolvingStatus << std::setw(11) << std::right << "Gap";
   *osTabularSolvingStatus << std::setw(17) << std::right << "Best Node(S)";
   *osTabularSolvingStatus << std::setw(11) << std::right << "Gap(S)";
   *osTabularSolvingStatus << std::endl;

   isHeaderPrinted = true;
}

void
BbParaLoadCoordinator::outputTabularSolvingStatus(
      char incumbent
      )
{

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   *osTabularSolvingStatus << std::setw(1) << incumbent;
   *osTabularSolvingStatus << std::setw(8) << std::right << std::setprecision(0) << std::fixed << paraTimer->getElapsedTime();
   if( // !restarted &&
         ( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ||
               paraParams->getIntParamValue(RampUpPhaseProcess) == 2 )
               && !racingWinnerParams )
   {
      /** racing ramp-up stage now */
      if( !racingTermination && paraRacingSolverPool )
      {
         *osTabularSolvingStatus << std::setw(15) << std::right << dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesSolvedInBestSolver();

         if( paraNodePool->getNumOfNodes() > 0 )
         {
            *osTabularSolvingStatus << std::setw(12) << std::right
                  << (dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesLeftInBestSolver()
                  + paraNodePool->getNumOfNodes());
         }
         else
         {
            *osTabularSolvingStatus << std::setw(12) << std::right << dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesLeftInBestSolver();
         }

         *osTabularSolvingStatus << std::setw(10) << std::right << paraRacingSolverPool->getNumActiveSolvers();
         if( bbParaInitiator->getGlobalBestIncumbentSolution() )
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                  bbParaInitiator->convertToExternalValue(
                        bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue());
         }
         else
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << "-";
         }
         if( dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesSolvedInBestSolver() == 0 )
         {
            if( bbParaInitiator->getGlobalBestIncumbentSolution() != NULL && EPSEQ( lcts.globalBestDualBoundValue, bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), paraInitiator->getEpsilon() ) )
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     lcts.externalGlobalBestDualBoundValue;
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
         }
         else
         {
            if( EPSEQ( lcts.globalBestDualBoundValue,-DBL_MAX, paraInitiator->getEpsilon() ))
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     lcts.externalGlobalBestDualBoundValue;
            }
         }
      }
      else  // One of ParaSolvers terminates in racing stage
      {
         if( nSolvedRacingTermination > 0 )
         {
            *osTabularSolvingStatus << std::setw(15) << std::right << nSolvedRacingTermination;
            *osTabularSolvingStatus << std::setw(12) << std::right << 0;
            *osTabularSolvingStatus << std::setw(10) << std::right << 0;
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     bbParaInitiator->convertToExternalValue(
                           bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue());
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
            // *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            if( EPSEQ( lcts.globalBestDualBoundValue,-DBL_MAX, bbParaInitiator->getEpsilon() ))
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     lcts.externalGlobalBestDualBoundValue;
            }
         }
         else   // should be interrupted
         {
            *osTabularSolvingStatus << std::setw(15) << std::right << nSolvedInInterruptedRacingSolvers;
            *osTabularSolvingStatus << std::setw(12) << std::right << nTasksLeftInInterruptedRacingSolvers;
            *osTabularSolvingStatus << std::setw(10) << std::right << 0;
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     bbParaInitiator->convertToExternalValue(
                           bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue());
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
            *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                  lcts.externalGlobalBestDualBoundValue;
         }
      }
      if( !paraNodeToKeepCheckpointFileNodes &&
            (!bbParaInitiator->getGlobalBestIncumbentSolution() ||
                  bbParaInitiator->getGap(lcts.globalBestDualBoundValue) > displayInfOverThisValue ||
            ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isActive() && paraSolverPool->getNumActiveSolvers() == 0 && paraNodePool->getNumOfNodes() == 0 )
            ) )
      {
         *osTabularSolvingStatus << std::setw(11) << std::right << " -";
      }
      else
      {
         *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
               bbParaInitiator->getGap(lcts.globalBestDualBoundValue) * 100 << "%";
      }
   }
   else
   {
      *osTabularSolvingStatus << std::setw(15) << std::right << dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers();
      if( unprocessedParaNodes )
      {
         *osTabularSolvingStatus << std::setw(12) << std::right << ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers()
               + paraNodePool->getNumOfNodes()
               + unprocessedParaNodes->getNumOfNodes() );
      }
      else if( paraNodeToKeepCheckpointFileNodes )
      {
         *osTabularSolvingStatus << std::setw(12) << std::right << ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers()
               + paraNodePool->getNumOfNodes()
               + paraNodeToKeepCheckpointFileNodes->getNumOfNodes() );
      }
      else
      {
         *osTabularSolvingStatus << std::setw(12) << std::right << ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers()
               + paraNodePool->getNumOfNodes() );
      }
      *osTabularSolvingStatus << std::setw(10) << std::right << paraSolverPool->getNumActiveSolvers();
      if( bbParaInitiator->getGlobalBestIncumbentSolution() )
      {
         *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
               bbParaInitiator->convertToExternalValue(
                     bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue());
      }
      else
      {
         *osTabularSolvingStatus << std::setw(17) << std::right << "-";
      }

      if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles)  )
      {
         // lcts.globalBestDualBoundValue = std::min(std::min( paraNodePool->getBestDualBoundValue(), paraNodePoolForBuffering->getBestDualBoundValue() ),
         //       dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() );
         // lcts.externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue( lcts.globalBestDualBoundValue );

         // *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<  lcts.externalGlobalBestDualBoundValue;
         if( EPSEQ( lcts.globalBestDualBoundValue,-DBL_MAX, bbParaInitiator->getEpsilon() ))
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << "-";
         }
         else
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                  lcts.externalGlobalBestDualBoundValue;
         }
         if(  bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) > displayInfOverThisValue )
         {
            *osTabularSolvingStatus << std::setw(11) << std::right << " -";
         }
         else
         {
            *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                  bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) * 100 << "%";
         }
      }
      else
      {
         if( !paraNodeToKeepCheckpointFileNodes &&
               ( paraSolverPool->getNumActiveSolvers() == 0 &&
               ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesSolvedInSolvers() == 0
                     || paraNodePool->getNumOfNodes() == 0 )
               ) )
         {
            if( (!givenGapIsReached) && bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               lcts.globalBestDualBoundValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
               /*
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     paraInitiator->convertToExternalValue(
                     paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue());
               *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                     paraInitiator->getGap( paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue() ) * 100 << "%";
                     */
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     bbParaInitiator->convertToExternalValue( lcts.globalBestDualBoundValue );
               *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                     bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) * 100 << "%";
            }
            else
            {
               if( EPSEQ( lcts.globalBestDualBoundValue,-DBL_MAX, bbParaInitiator->getEpsilon() ))
               {
                  *osTabularSolvingStatus << std::setw(17) << std::right << "-";
               }
               else
               {
                  *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                        lcts.externalGlobalBestDualBoundValue;
               }
               if(  bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) > displayInfOverThisValue )
               {  
                  *osTabularSolvingStatus << std::setw(11) << std::right << " -";
               }
               else
               {  
                  *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                        bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) * 100 << "%";
               }
            }
            // *osTabularSolvingStatus << std::setw(17) << std::right << "-";
         }
         else
         {
            //*osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
            //      lcts.externalGlobalBestDualBoundValue;
            if( EPSEQ( lcts.globalBestDualBoundValue,-DBL_MAX, bbParaInitiator->getEpsilon() ))
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << "-";
            }
            else
            {
               *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                     lcts.externalGlobalBestDualBoundValue;
            }
            if( paraNodeToKeepCheckpointFileNodes && paraNodeToKeepCheckpointFileNodes->getNumOfNodes() > 0 )
            {
               double globalBestDualBound = paraNodeToKeepCheckpointFileNodes->getBestDualBoundValue();
               if(  bbParaInitiator->getGap( globalBestDualBound ) > displayInfOverThisValue )
               {
                  *osTabularSolvingStatus << std::setw(11) << std::right << " -";
               }
               else
               {
                  *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                        bbParaInitiator->getGap( globalBestDualBound ) * 100 << "%";
               }
            }
            else
            {
               if(  bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) > displayInfOverThisValue )
               {
                  *osTabularSolvingStatus << std::setw(11) << std::right << " -";
               }
               else
               {
                  *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                        bbParaInitiator->getGap( lcts.globalBestDualBoundValue ) * 100 << "%";
               }
            }
         }
         /*
         if( !paraNodeToKeepCheckpointFileNodes &&
               ( !paraInitiator->getGlobalBestIncumbentSolution() ||
               paraInitiator->getGap(lcts.globalBestDualBoundValue) > displayInfOverThisValue ||
               ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isActive() && paraSolverPool->getNumActiveSolvers() == 0 && paraNodePool->getNumOfNodes() == 0 )
               ) )
         {
            *osTabularSolvingStatus << std::setw(10) << std::right << "-";
         }
         else
         {
            *osTabularSolvingStatus << std::setw(9) << std::right << std::setprecision(2) <<
                  paraInitiator->getGap(lcts.globalBestDualBoundValue) * 100 << "%";
         }
         */ 
      }

      if( paraSolverPool->getNumActiveSolvers() > 0 )
      {
         if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() >= -1e+10 )
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << std::setprecision(4) <<
                  bbParaInitiator->convertToExternalValue( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() );
         }
         else
         {
            *osTabularSolvingStatus << std::setw(17) << std::right << "-";
         }
         if(  bbParaInitiator->getGap( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() ) > displayInfOverThisValue )
         {
            *osTabularSolvingStatus << std::setw(11) << std::right << " -";
         }
         else
         {
            *osTabularSolvingStatus << std::setw(10) << std::right << std::setprecision(2) <<
                  bbParaInitiator->getGap( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue() ) * 100 << "%";
         }
      }
      else
      {
         // *osTabularSolvingStatus << std::setw(17) << std::right << "-";
      }

   }
   *osTabularSolvingStatus << std::endl;
}

void
BbParaLoadCoordinator::run(
      )
{

   if( !isHeaderPrinted && outputTabularSolvingStatusFlag )
   {
      outputTabularSolvingStatusHeader();            /// should not call virutal function in constructor
   }

   int source;
   int tag;

   for(;;)
   {
      if( paraSolverPool->getNumActiveSolvers() == 0 )
      {
         if( paraNodePool->isEmpty() )                             // paraNodePool has to be checked
                                                                   // because node cannot send in a parameter settings
         {
            if( runningPhase != TerminationPhase )
            {
               /*
               if( !interruptedFromControlTerminal
                     && !computationIsInterrupted
                     && !hardTimeLimitIsReached
                     && paraInitiator->getGlobalBestIncumbentSolution() )
               {
                  lcts.globalBestDualBoundValue = paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue();
                  lcts.externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
               }
               */
               /* No active solver exists */
               terminateAllSolvers();
               runningPhase = TerminationPhase;
               /*
               if( !racingTermination )
               {
                  lcts.globalBestDualBoundValue = paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue();
                  lcts.externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
               }
               */
            }
            else // runningPhase == TerminationPhase
            {
               if( ( paraRacingSolverPool &&
                     // paraSolverPool->getNumInactiveSolvers() == (paraRacingSolverPool->getNumActiveSolvers() + nTerminated ) ) ||
                     // paraSolverPool->getNumInactiveSolvers() == nTerminated  )
                     paraSolverPool->getNSolvers() == (paraRacingSolverPool->getNumActiveSolvers() + nTerminated ) ) ||
                     paraSolverPool->getNSolvers() == nTerminated  )
               {
                  break;
               }
            }
         }
         else
         {
            if( initialNodesGenerated  )
            {
               if( runningPhase != TerminationPhase )
               {
                  lcts.globalBestDualBoundValue = std::min( paraNodePool->getBestDualBoundValue(), lcts.globalBestDualBoundValue );
                  lcts.externalGlobalBestDualBoundValue = dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(lcts.globalBestDualBoundValue);
#ifdef UG_WITH_ZLIB
                  updateCheckpointFiles();
#endif
                  /* No active solver exists */
                  terminateAllSolvers();
                  runningPhase = TerminationPhase;
               }
               else  // runningPhase == TerminationPhase
               {
                  // if( paraSolverPool->getNumInactiveSolvers() == nTerminated  )
                  if( paraSolverPool->getNSolvers() == nTerminated  )
                  {
                     break;
                  }
               }
            }
         }
         if( ( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
               paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
               runningPhase == TerminationPhase ) )
         {
            break;
         }
      }

      if( paraParams->getIntParamValue(NIdleSolversToTerminate) > 0 &&
            firstCollectingModeState == 1 &&
            (signed)paraSolverPool->getNumInactiveSolvers() >=  paraParams->getIntParamValue(NIdleSolversToTerminate)
            )
      {
         if( runningPhase != TerminationPhase )
         {
            lcts.globalBestDualBoundValue = std::min( paraNodePool->getBestDualBoundValue(), lcts.globalBestDualBoundValue );
            lcts.externalGlobalBestDualBoundValue = dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(lcts.globalBestDualBoundValue);
#ifdef UG_WITH_ZLIB
            updateCheckpointFiles();
#endif
            /* No active solver exists */
            terminateAllSolvers();
            std::cout << "### REACHED TO THE SPECIFIED NUMBER OF IDLER SOLVERS, then EXIT ###" << std::endl;
            exit(1);  // try to terminate all solvers, but do not have to wait until all solvers have terminated.
                      // Basically, this procedure is to kill the ug[*,*].
         } // if already in TerminaitonPhase, just keep on running.
      }

      if( !paraRacingSolverPool && paraSolverPool->getNumActiveSolvers() == 0 )
      {
         if( paraParams->getRealParamValue(TimeLimit) > 0.0 )
         {
             if( hardTimeLimitIsReached || paraTimer->getElapsedTime() >= paraParams->getRealParamValue(TimeLimit) )
             {
                hardTimeLimitIsReached = true;
                break;
             }
             if( givenGapIsReached )
                break;
             // std::cout << "ElapsedTime = " << paraTimer->getElapsedTime() << ", runningPhase = " << static_cast<int>(runningPhase) << std::endl;
             if( paraSolverPool->getNumActiveSolvers() == 0
                 && paraNodePool->isEmpty() 
                 && nTerminated == paraSolverPool->getNSolvers() )
             {
                break;
             }
         }
         else
         {
            break;
         }
      }

      if( racingTermination
            && !paraRacingSolverPool
            && paraSolverPool->getNumActiveSolvers() == 0
            && paraNodePool->getNumOfNodes() <= 1
            && nTerminated == paraSolverPool->getNSolvers() )
      {
         /*
          * special timining problem
          *
          * 1113.58 S.4 I.SOL 0
          * 1113.58 S.3 is the racing winner! Selected strategy 2.
          * 1113.58 S.4 >(TERMINATED_IN_RACING_STAGE)
          *
          */
         break;
      }

      /*******************************************
       *  waiting for any message form anywhere  *
       *******************************************/
      double inIdleTime = paraTimer->getElapsedTime();
      (void)paraComm->probe(&source, &tag);
      lcts.idleTime += ( paraTimer->getElapsedTime() - inIdleTime );
      if( messageHandler[tag] )
      {
         int status = (this->*messageHandler[tag])(source, tag);
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

#ifdef UG_WITH_UGS
      if( commUgs ) checkAndReadIncumbent();
#endif

      /** completion message may delay */
      if( paraRacingSolverPool && paraRacingSolverPool->getNumActiveSolvers() == 0 )
      {
         delete paraRacingSolverPool;
         paraRacingSolverPool = 0;
         if( racingTermination )
         {
            break;
         }
      }

      /** output tabular solving status */
      if( outputTabularSolvingStatusFlag &&
            paraSolverPool->getNumActiveSolvers() != 0 &&
            ( ( ( paraParams->getBoolParamValue(Deterministic) &&
                  paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) ) &&
                  ( paraDetTimer->getElapsedTime() - previousTabularOutputTime ) >
                                 paraParams->getRealParamValue(TabularSolvingStatusInterval) ) ||
            ( ( !paraParams->getBoolParamValue(Deterministic) ||
                  !paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) ) &&
                  ( paraTimer->getElapsedTime() - previousTabularOutputTime ) >
               paraParams->getRealParamValue(TabularSolvingStatusInterval) ) ) )
      {
         outputTabularSolvingStatus(' ');
         if( paraParams->getBoolParamValue(Deterministic) )
         {
            if( paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) )
            {
               previousTabularOutputTime = paraDetTimer->getElapsedTime();
            }
            else
            {
               previousTabularOutputTime = paraTimer->getElapsedTime();
            }
         }
         else
         {
            previousTabularOutputTime = paraTimer->getElapsedTime();
         }
      }

      switch ( runningPhase )
      {
      case RampUpPhase:
      {
         if( selfSplitFinisedSolvers &&  (!hardTimeLimitIsReached) && (!memoryLimitIsReached) && (!givenGapIsReached) &&
           selfSplitFinisedSolvers->size() == (unsigned)(paraComm->getSize() - 1) )  // all solvers have finished self-split
         {
            assert(paraParams->getIntParamValue(RampUpPhaseProcess) == 3);
            sendRampUpToAllSolvers();
            runningPhase = NormalRunningPhase;
            delete selfSplitFinisedSolvers;
            selfSplitFinisedSolvers = 0;
            if( outputTabularSolvingStatusFlag )
            {
               outputTabularSolvingStatus(' ');
            }
            break;
         }
         if( ( racingTermination && paraNodePool->isEmpty() ) ||
               (  paraParams->getRealParamValue(TimeLimit) > 0.0 &&
                 paraTimer->getElapsedTime() > paraParams->getRealParamValue(TimeLimit) ) )
         {
            sendInterruptRequest();
// std::cout << "sendInterrruptRequest1 11" << std::endl;
            runningPhase = TerminationPhase;
         }
         else
         {
            if( paraSolverPool->getNumInactiveSolvers() == 0 )
            {
               // without consideration of keeping nodes in checkpoint file
               double globalBestDualBoundValueLocal =
                  std::max (
                        std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                        lcts.globalBestDualBoundValue );
               if( paraParams->getBoolParamValue(DualBoundGainTest) )
               {
                  if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isActive() &&
                       ( paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
                                       > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode) ||
                        ( paraNodePool->getNumOfNodes()
                                       > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode)*2  &&
                                       paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) ) )
                  {
                     sendRampUpToAllSolvers();
                     runningPhase = NormalRunningPhase;
                  }
               }
               else
               {
                  if( paraParams->getIntParamValue(RampUpPhaseProcess) != 3  )
                  {
                     if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isActive() &&
                          ( (signed)paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
                                          > paraParams->getIntParamValue(NChangeIntoCollectingMode) ||
                           ( (signed)paraNodePool->getNumOfNodes()
                                          > paraParams->getIntParamValue(NChangeIntoCollectingMode) * 2  &&
                                          paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) ) )
                     {
                        sendRampUpToAllSolvers();
                        runningPhase = NormalRunningPhase;
                     }
                  }
               }
            }
            else
            {
               if( winnerSolverNodesCollected && ( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 || paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) )
               {
                  // maybe some solver hard to interrupt in a large scale execution
                  // without consideration of keeping nodes in checkpoint file
                  double globalBestDualBoundValueLocal =
                     std::max (
                           std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                           lcts.globalBestDualBoundValue );
                  if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isActive() &&
                        ( paraNodePool->getNumOfNodes()
                                       > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode)*2  &&
                                       paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) )
                  {
                     sendRampUpToAllSolvers();
                     runningPhase = NormalRunningPhase;
                  }
               }
            }
            (void) sendParaTasksToIdleSolvers();
         }
         break;
      }
      case NormalRunningPhase:
      {
         if( ( racingTermination && paraNodePool->isEmpty() )||
             (  paraParams->getRealParamValue(TimeLimit) > 0.0 &&
               paraTimer->getElapsedTime() > paraParams->getRealParamValue(TimeLimit) ) ||
               ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) &&
                     paraTimer->getElapsedTime() > paraParams->getRealParamValue(UG::EnhancedCheckpointStartTime) ) ||
               ( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 &&
               paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) ) )
         {
            if( !paraSolverPool->isInterruptRequested(source) )
            {
               sendInterruptRequest();
// std::cout << "sendInterrruptRequest1 12" << std::endl;
            }
            runningPhase = TerminationPhase;
         }
         else
         {
            (void) sendParaTasksToIdleSolvers();
         }
         if( isCollectingModeRestarted && paraNodePool->isEmpty() &&
               ( (!paraRacingSolverPool) || ( paraRacingSolverPool && paraRacingSolverPool->getWinner() > 0 ) ) )
         {
            if( ( paraParams->getBoolParamValue(Deterministic) &&
                  ( paraDetTimer->getElapsedTime() - statEmptyNodePoolTime ) > (paraParams->getRealParamValue(TimeToIncreaseCMS)*2) ) ||
                  ( !paraParams->getBoolParamValue(Deterministic) &&
                  ( paraTimer->getElapsedTime() - statEmptyNodePoolTime ) > (paraParams->getRealParamValue(TimeToIncreaseCMS)*2) ) )
            {
               sendRetryRampUpToAllSolvers();
               runningPhase = RampUpPhase;
            }
         }

#ifdef UG_WITH_ZLIB
         if( paraParams->getRealParamValue(RestartInRampDownThresholdTime) > 0.0 )
         {
            if( paraSolverPool->getNumActiveSolvers()
                  < paraSolverPool->getNSolvers()*paraParams->getRealParamValue(RestartInRampDownActiveSolverRatio) )
            {
               if( starvingTime < 0.0 )
               {
                  starvingTime = paraTimer->getElapsedTime();
               }
            }
            else
            {
               starvingTime = -1.0;
            }
            // std::cout << "active solvers:" << paraSolverPool->getNumActiveSolvers() << std::endl;
            if( starvingTime > 0 &&
                  ( paraTimer->getElapsedTime() - starvingTime )
                  > paraParams->getRealParamValue(RestartInRampDownThresholdTime)  &&
                  ( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() + paraNodePool->getNumOfNodes() )
                  > paraSolverPool->getNumActiveSolvers()*10 )
            {
               hugeImbalance = false;
               restartInRampDownPhase();
            }
         }
#endif

         if( paraParams->getRealParamValue(HugeImbalanceThresholdTime) > 0.0 )
         {
            if( paraSolverPool->getNumActiveSolvers()
                  < paraSolverPool->getNSolvers()*paraParams->getRealParamValue(HugeImbalanceActiveSolverRatio) )
            {
               if( hugeImbalanceTime < 0.0 )
               {
                  changeSearchStrategyOfAllSolversToBestBoundSearch();
                  hugeImbalanceTime = paraTimer->getElapsedTime();
               }
            }
            else
            {
               changeSearchStrategyOfAllSolversToOriginalSearch();
               hugeImbalanceTime = -1.0;
            }
            // std::cout << "active solvers:" << paraSolverPool->getNumActiveSolvers() << std::endl;
            if( hugeImbalanceTime > 0 &&
                  ( paraTimer->getElapsedTime() - hugeImbalanceTime )
                  > paraParams->getRealParamValue(HugeImbalanceThresholdTime)  &&
                  dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() > paraSolverPool->getNumActiveSolvers()*100 )
            {
               hugeImbalance = true;
               if( !paraNodePoolBufferToRestart )
               {
                  paraNodePoolBufferToRestart =  new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
               }
               // reschedule collecting mode
               double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
            }
            if( hugeImbalance &&
                  ( paraNodePoolBufferToRestart->getNumOfNodes() > paraSolverPool->getNSolvers() ||
                    dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers() < paraSolverPool->getNumActiveSolvers() * 5 ||
                        paraSolverPool->getNumActiveSolvers() == 0 ) )
            {
               hugeImbalance = false;
               hugeImbalanceTime = -1.0;
               while( !paraNodePoolBufferToRestart->isEmpty() )
               {
                  paraNodePool->insert(paraNodePoolBufferToRestart->extractNode());
               }
               (void) sendParaTasksToIdleSolvers();
               dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(-1.0); // restart collecting
            }
         }

         break;
      }
      case TerminationPhase:
      {
         break;
      }
      default:
      {
         THROW_LOGICAL_ERROR2( "Undefined running phase: ", static_cast<int>(runningPhase) );
      }
      }
#ifdef UG_WITH_ZLIB
      if( paraParams->getBoolParamValue(Checkpoint) &&
            ( paraTimer->getElapsedTime() - previousCheckpointTime )
            > paraParams->getRealParamValue(CheckpointInterval) )
      {
         if( !( interruptIsRequested &&
               paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 ) )
         {
            updateCheckpointFiles();
            previousCheckpointTime = paraTimer->getElapsedTime();
         }
      }
#endif
   }
}

#ifdef UG_WITH_ZLIB
void
BbParaLoadCoordinator::restartInRampDownPhase(
      )
{
   updateCheckpointFiles();
   /** send interrupt request */
   int stayAlive = 0;   // exit!
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
            paraComm->send( &stayAlive, 1, ParaINT, i, TagInterruptRequest )
      );
   }

   /** the pupose of the updateCheckpoitFiles is two
    *  1. Can be killed during restaart, for example, in a case that a solver cannot be intterrupted so long time
    *  2. To update initial dual bound values
    */
   // updateCheckpointFiles();

   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " Interrupt all solvers to restart"
      << std::endl;
      if( outputTabularSolvingStatusFlag )
      {
         *osTabularSolvingStatus <<
               "Interrupt all solvers to restart after "
               << paraTimer->getElapsedTime() << " seconds." << std::endl;
      }
   }

   exit(1);   // Terminate LoadCoordinator. Restart did not work well over 10,000 solvers.

   paraNodePoolToRestart =  new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));

   /** for a timing issue, paraNodePool may not be empty. paraNodes in the pool should be just recived,
    * because paraSolverPool was empty. Then, no check for the ancestors.
    */
   while( !paraNodePool->isEmpty() )
   {
      ParaTask *node = paraNodePool->extractNode();
      delete node;
   }
   while( !paraNodePoolBufferToRestart->isEmpty() )
   {
      ParaTask *node = paraNodePoolBufferToRestart->extractNode();
      delete node;
   }

   /*******************************************
    *  waiting for any message form anywhere  *
    *******************************************/
   for(;;)
   {
      int source;
      int tag;
      double inIdleTime = paraTimer->getElapsedTime();
      (void)paraComm->probe(&source, &tag);
      lcts.idleTime += ( paraTimer->getElapsedTime() - inIdleTime );
      if( messageHandler[tag] )
      {
         int status = (this->*messageHandler[tag])(source, tag);
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

#ifdef UG_WITH_UGS
      if( commUgs ) checkAndReadIncumbent();
#endif

      if( paraSolverPool->getNumActiveSolvers() == 0 )
      {
         break;
      }

   }

   if( !paraNodePool->isEmpty() )
   {
      std::cout << "Logical error occurred during restart in ramp-down phase." << std::endl;
      std::cout << "You can restart from the chakepoint file." << std::endl;
      // exit(1);
      abort();
   }

   while( !paraNodePoolToRestart->isEmpty() )
   {
      BbParaNode *node = dynamic_cast<BbParaNode *>(paraNodePoolToRestart->extractNode());
      node->setDualBoundValue(node->getInitialDualBoundValue());
      paraNodePool->insert(node);
   }

   delete paraNodePoolToRestart;
   paraNodePoolToRestart = 0;

   runningPhase = RampUpPhase;
   /** initialize paraSolerPool */
   dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->reinitToRestart();

   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, i, TagRestart )
      );
   }

   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " Restart"
      << std::endl;
      if( outputTabularSolvingStatusFlag )
      {
         *osTabularSolvingStatus <<
               "Restart after "
               << paraTimer->getElapsedTime() << " seconds." << std::endl;
      }
   }

   (void) sendParaTasksToIdleSolvers();
}
#endif 

bool
BbParaLoadCoordinator::sendParaTasksToIdleSolvers(
      )
{
   if( merging || initialNodesGenerated || hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached ||
       runningPhase == TerminationPhase || hugeImbalance ||
       ( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
                      paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 ) &&
                      runningPhase == TerminationPhase ) ||
         ( !restarted &&
            // paraParamSet->getBoolParamValue(RacingStatBranching) &&
            ( !winnerSolverNodesCollected ||
                  ( paraRacingSolverPool &&
                        paraRacingSolverPool->getNumInactiveSolvers() < paraRacingSolverPool->getNumActiveSolvers() )
            )
         )
       )
   {
      return false;
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   bool sentNode = false;
   while( paraSolverPool->getNumInactiveSolvers() > 0 && !paraNodePool->isEmpty() )
   {
      BbParaNode *paraNode = 0;
      while( !paraNodePool->isEmpty() )
      {
         if( nNormalSelection >= 0 && !warmStartNodeTransferring )
         {
            if( nNormalSelection > static_cast<int>( 1.0 / paraParams->getRealParamValue(RandomNodeSelectionRatio) ) )
            {
               paraNode = paraNodePool->extractNodeRandomly();
            }
            else
            {
               paraNode = paraNodePool->extractNode();
            }
         }
         else
         {
            paraNode = paraNodePool->extractNode();
         }
         if( !paraNode ) break;
         assert( !paraNode->getMergeNodeInfo() ||
               ( paraNode->getMergeNodeInfo() &&
               paraNode->getMergeNodeInfo()->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE &&
               paraNode->getMergeNodeInfo()->mergedTo == 0 ) );

         if( paraParams->getBoolParamValue(UG::GenerateReducedCheckpointFiles) &&
               ( ( !paraNode->getMergeNodeInfo() ) ||
                 ( paraNode->getMergeNodeInfo() &&
                       paraNode->getMergeNodeInfo()->nMergedNodes <= 0 ) ) )
         {
            assert( !paraNode->getMergeNodeInfo() || ( paraNode->getMergeNodeInfo() &&  paraNode->getMergeNodeInfo()->nMergedNodes == 0 ) );
            if( paraNode->getMergeNodeInfo() )
            {
               BbParaMergeNodeInfo *mNode = paraNode->getMergeNodeInfo();
               if( mNode->origDiffSubproblem )
               {
                  paraNode->setDiffSubproblem(mNode->origDiffSubproblem);
                  delete mNode->mergedDiffSubproblem;
                  mNode->mergedDiffSubproblem = 0;
                  mNode->origDiffSubproblem = 0;
               }
               paraNode->setMergeNodeInfo(0);
               paraNode->setMergingStatus(-1);
               assert(nodesMerger);
               nodesMerger->deleteMergeNodeInfo(mNode);
            }
            paraNodePoolBufferToGenerateCPF->insert(paraNode);
            /*
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " node saved to the buffer. Dual bound:"
               << paraInitiator->convertToExternalValue( paraNode->getDualBoundValue() ) << std::endl;
            }
            */
            paraNode = 0;

            continue;
         }

         if( ( bbParaInitiator->getGlobalBestIncumbentSolution() &&
               ( paraNode->getDualBoundValue() < bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() ||
                     ( bbParaInitiator->isObjIntegral() &&
                     static_cast<int>(ceil( paraNode->getDualBoundValue() ) )
                     < static_cast<int>(bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() + MINEPSILON ) )
                     ) ) ||
               !( bbParaInitiator->getGlobalBestIncumbentSolution() ) )
         {
            if( bbParaInitiator->getAbsgap(paraNode->getDualBoundValue() ) > bbParaInitiator->getAbsgapValue() ||
                  bbParaInitiator->getGap(paraNode->getDualBoundValue()) > bbParaInitiator->getGapValue() )
            {
               break;
            }
            else
            {
#ifdef UG_DEBUG_SOLUTION
               if( paraNode->getDiffSubproblem() && paraNode->getDiffSubproblem()->isOptimalSolIncluded() )
               {
                  throw "Optimal solution going to be killed.";
               }
#endif
               delete paraNode;
               paraNode = 0;
               lcts.nDeletedInLc++;
               if( nNormalSelection >= 0 && !warmStartNodeTransferring )
               {
                  if( nNormalSelection > static_cast<int>( 1.0 / paraParams->getRealParamValue(RandomNodeSelectionRatio) ) )
                  {
                     nNormalSelection = 0;
                  }
                  else
                  {
                     nNormalSelection++;
                  }
               }
               /*
               std::cout << "dual bound = " << paraNode->getDualBoundValue() << std::endl;
               std::cout << "agap(dual bound) = " << paraInitiator->getAbsgap(paraNode->getDualBoundValue())
                         << ", agap = " << paraInitiator->getAbsgapValue() << std::endl;
               std::cout << "gap(dual bound) = " << paraInitiator->getGap(paraNode->getDualBoundValue())
                         << ", gap = " << paraInitiator->getGapValue() << std::endl;
               break;
               */
            }
         }
         else
         {
#ifdef UG_DEBUG_SOLUTION
            if( paraNode->getDiffSubproblem() && paraNode->getDiffSubproblem()->isOptimalSolIncluded() )
            {
               throw "Optimal solution going to be killed.";
            }
#endif
            delete paraNode;
            paraNode = 0;
            lcts.nDeletedInLc++;
         }
      }

      if( paraNode )
      {
         if( nNormalSelection >= 0 && !warmStartNodeTransferring )
         {
            if( nNormalSelection > static_cast<int>( 1.0 / paraParams->getRealParamValue(RandomNodeSelectionRatio) ) )
            {
               nNormalSelection = 0;
            }
            else
            {
               nNormalSelection++;
            }
         }

         if( paraNode->getMergeNodeInfo() && paraNode->getMergeNodeInfo()->nMergedNodes == 0 )
         {
            BbParaMergeNodeInfo *mNode = paraNode->getMergeNodeInfo();
            paraNode->setDiffSubproblem(mNode->origDiffSubproblem);
            paraNode->setMergeNodeInfo(0);
            paraNode->setMergingStatus(-1);
            delete mNode->mergedDiffSubproblem;
            mNode->mergedDiffSubproblem = 0;
            mNode->origDiffSubproblem = 0;
            assert(nodesMerger);
            nodesMerger->deleteMergeNodeInfo(mNode);
         }
         if( paraParams->getBoolParamValue(RacingStatBranching) &&
               paraNode->isSameParetntTaskSubtaskIdAs( TaskId() ) &&  //  if parent is the root node
               paraNode->getDiffSubproblem()                          //  paraNode deos not root
               )
         {
            bbParaInitiator->setInitialStatOnDiffSubproblem(
                  minDepthInWinnerSolverNodes, maxDepthInWinnerSolverNodes,
                  dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem()));
         }
         // without consideration of keeping nodes in checkpoint file
         BbParaSolverPoolForMinimization *bbParaSolverPool = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool);
         double globalBestDualBoundValueLocal =
            std::max (
                  std::min( bbParaSolverPool->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                  lcts.globalBestDualBoundValue );
         int destination = bbParaSolverPool->activateSolver(paraNode,
               dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool), (runningPhase==RampUpPhase),
               paraNodePool->getNumOfGoodNodes(globalBestDualBoundValueLocal), averageLastSeveralDualBoundGains );
         if( destination < 0 )
         {
            /** cannot activate */
            paraNodePool->insert(paraNode);
            return sentNode;
         }
         else
         {
            lcts.nSent++;
            writeTransferLog(destination);
            sentNode = true;
            if( runningPhase == RampUpPhase &&
                  paraParams->getIntParamValue(RampUpPhaseProcess) == 0 &&
                  paraParams->getBoolParamValue(CollectOnce) &&
                  // paraSolverPool->getNumActiveSolvers() < paraSolverPool->getNSolvers()/2
                  paraSolverPool->getNumInactiveSolvers() > 0 &&
                  paraSolverPool->getNumActiveSolvers()*2 <
                  ( paraSolverPool->getNSolvers() + paraParams->getIntParamValue(NChangeIntoCollectingMode) )
                  )
            {
               int nCollect = -1;
               PARA_COMM_CALL(
                     paraComm->send( &nCollect, 1, ParaINT, destination, TagCollectAllNodes )
               );
            }
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << destination << " < "
               << bbParaInitiator->convertToExternalValue(
                     paraNode->getDualBoundValue() );
               if( bbParaInitiator->getGlobalBestIncumbentSolution() )
               {
                  if( bbParaInitiator->getGap(paraNode->getDualBoundValue()) > displayInfOverThisValue )
                  {
                     *osLogSolvingStatus << " ( Inf )";
                  }
                  else
                  {
                     *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(paraNode->getDualBoundValue()) * 100 << "% )";
                  }
               }
               if( paraParams->getBoolParamValue(LightWeightRootNodeProcess) &&
                     runningPhase != RampUpPhase && (!paraRacingSolverPool) &&
                     paraSolverPool->getNumInactiveSolvers() >
                         ( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(RatioToApplyLightWeightRootProcess) ) )
               {
                  *osLogSolvingStatus << " L";
               }
               if( paraNode->getMergeNodeInfo() )
               {
                  *osLogSolvingStatus << " M(" << paraNode->getMergeNodeInfo()->nMergedNodes + 1 << ")";
                  if( paraNode->getMergeNodeInfo()->nMergedNodes < 1 )
                  {
                     std::cout << "node id = " << (paraNode->getTaskId()).toString() << std::endl;
                     abort();
                  }
               }
               if( paraNode->getDiffSubproblem() )
               {
                  *osLogSolvingStatus << " " << paraNode->toSimpleString()
                        << ", "
                        // << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->getNBoundChanges();
                        << dynamic_cast<BbParaDiffSubproblem *>(paraNode->getDiffSubproblem())->toStringStat();
               }
               // for debug
               // *osLogSolvingStatus << " " << paraNode->toSimpleString();
               *osLogSolvingStatus << std::endl;
            }
#ifdef _DEBUG_LB
            std::cout << paraTimer->getElapsedTime()
            << " S." << destination << " < "
            << paraInitiator->convertToExternalValue(
                  paraNode->getDualBoundValue() );
            if( paraInitiator->getGlobalBestIncumbentSolution() )
            {
               if( paraInitiator->getGap(paraNode->getDualBoundValue()) > displayInfOverThisValue )
               {
                  std::cout << " ( Inf )";
               }
               else
               {
                  std::cout << " ( " << paraInitiator->getGap(paraNode->getDualBoundValue()) * 100 << "% )";
               }
            }
            if( paraParams->getBoolParamValue(LightWeightRootNodeProcess) &&
                  runningPhase != RampUpPhase && (!paraRacingSolverPool) &&
                  paraSolverPool->getNumInactiveSolvers() >
                     ( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(RatioToApplyLightWeightRootProcess) ) )
            {
               std::cout << " L";
            }
            std::cout << std::endl;
#endif
         }
      }
      else
      {
         break;
      }
   }
   return sentNode;
}

#ifdef UG_WITH_ZLIB
void
BbParaLoadCoordinator::updateCheckpointFiles(
      )
{
   time_t timer;
   char timeStr[30];

   if( paraNodePoolToRestart )
   {
      return;   // Interrupting all solvers;
   }

   if( paraNodePoolBufferToGenerateCPF &&  paraNodePoolBufferToGenerateCPF->getNumOfNodes() > 0 )
   {
      return;   // Collecting nodes.
   }

   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 3 && runningPhase == RampUpPhase )
   {
      return;
   }

   /** get checkpoint time */
   time(&timer);
   /** make checkpoint time string */
#ifdef _MSC_VER
   int bufsize = 256;
   ctime_s(timeStr, bufsize, &timer);
#else
   ctime_r(&timer, timeStr);
#endif
   for( int i = 0; timeStr[i] != '\0' && i < 26; i++ )
   {
      if( timeStr[i] == ' ') timeStr[i] = '_';
      if( timeStr[i] == '\n' ) timeStr[i] = '\0';
   }
   char *newCheckpointTimeStr = &timeStr[4];    // remove a day of the week
   // std::cout << "lstCheckpointTimeStr = " << lastCheckpointTimeStr << std::endl;
   // std::cout << "newCheckpointTimeStr = " << newCheckpointTimeStr << std::endl;
   if( strcmp(newCheckpointTimeStr,lastCheckpointTimeStr) == 0 )
   {
      int l = strlen(newCheckpointTimeStr);
      newCheckpointTimeStr[l] = 'a';
      newCheckpointTimeStr[l+1] = '\0';
   }

   /** save nodes information */
   char nodesFileName[256];
   sprintf(nodesFileName,"%s%s_%s_nodes_LC%d.gz",
         paraParams->getStringParamValue(CheckpointFilePath),
         paraInitiator->getParaInstance()->getProbName(),newCheckpointTimeStr, paraComm->getRank());
   gzstream::ogzstream checkpointNodesStream;
   checkpointNodesStream.open(nodesFileName, std::ios::out | std::ios::binary);
   if( !checkpointNodesStream )
   {
      std::cout << "Checkpoint file for ParaNodes cannot open. file name = " << nodesFileName << std::endl;
      exit(1);
   }
   dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->updateDualBoundsForSavingNodes();
   paraNodePool->updateDualBoundsForSavingNodes();
   int n = 0;
   if( paraNodeToKeepCheckpointFileNodes )
   {
      n += paraNodeToKeepCheckpointFileNodes->writeBbParaNodesToCheckpointFile(checkpointNodesStream);
   }
   n += dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->writeParaNodesToCheckpointFile(checkpointNodesStream);
   n += paraNodePool->writeBbParaNodesToCheckpointFile(checkpointNodesStream);
   if( unprocessedParaNodes && unprocessedParaNodes->getNumOfNodes() > 0 )
   {
      int nUnprocessedNodes = unprocessedParaNodes->writeBbParaNodesToCheckpointFile(checkpointNodesStream);
      if( n <  ( paraParams->getIntParamValue(NEagerToSolveAtRestart)/2 )
            && paraNodePool->getBestDualBoundValue() > unprocessedParaNodes->getBestDualBoundValue() )
      {
         for(int i = n;
               i <= paraParams->getIntParamValue(NEagerToSolveAtRestart)
                     && unprocessedParaNodes->getNumOfNodes() > 0;
               i++ )
         {
            BbParaNode *tempParaNode = unprocessedParaNodes->extractNode();
            paraNodePool->insert(tempParaNode);
         }
      }
      n += nUnprocessedNodes;
   }
   checkpointNodesStream.close();
   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " Checkpoint: " << n << " ParaNodes were saved" <<  std::endl;
   }
#ifdef _DEBUG_LB
   std::cout << paraTimer->getElapsedTime()
   << " Checkpoint: " << n << " ParaNodes were saved" <<  std::endl;
#endif

   if( outputTabularSolvingStatusFlag )
   {
      *osTabularSolvingStatus <<
            "Storing check-point data after " <<
            paraTimer->getElapsedTime() << " seconds. " <<
            n << " nodes were saved." << std::endl;
   }

   /** save incumbent solution */
   char solutionFileName[256];
   if( paraComm->getRank() == 0 )
   {
      sprintf(solutionFileName,"%s%s_%s_solution.gz",
            paraParams->getStringParamValue(CheckpointFilePath),
            paraInitiator->getParaInstance()->getProbName(),newCheckpointTimeStr);
      paraInitiator->writeCheckpointSolution(std::string(solutionFileName));
   }

   /** save Solver statistics */
   char solverStatisticsFileName[256];
   sprintf(solverStatisticsFileName,"%s%s_%s_solverStatistics_LC%d.gz",
         paraParams->getStringParamValue(CheckpointFilePath),
         paraInitiator->getParaInstance()->getProbName(),newCheckpointTimeStr, paraComm->getRank());
   gzstream::ogzstream checkpointSolverStatisticsStream;
   checkpointSolverStatisticsStream.open(solverStatisticsFileName, std::ios::out | std::ios::binary);
   if( !checkpointSolverStatisticsStream )
   {
      std::cout << "Checkpoint file for SolverStatistics cannot open. file name = " << solverStatisticsFileName << std::endl;
      exit(1);
   }
   int nSolverInfo = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->writeSolverStatisticsToCheckpointFile(checkpointSolverStatisticsStream);
   checkpointSolverStatisticsStream.close();

   /** save LoadCoordinator statistics */
   char loadCoordinatorStatisticsFileName[256];
   sprintf(loadCoordinatorStatisticsFileName,"%s%s_%s_loadCoordinatorStatistics_LC%d.gz",
         paraParams->getStringParamValue(CheckpointFilePath),
         paraInitiator->getParaInstance()->getProbName(),newCheckpointTimeStr, paraComm->getRank());
   gzstream::ogzstream loadCoordinatorStatisticsStream;
   loadCoordinatorStatisticsStream.open(loadCoordinatorStatisticsFileName, std::ios::out | std::ios::binary);
   if( !loadCoordinatorStatisticsStream )
   {
      std::cout << "Checkpoint file for SolverStatistics cannot open. file name = " << loadCoordinatorStatisticsFileName << std::endl;
      exit(1);
   }
   // double globalBestDualBoundValue =
   //   std::max (
   //      std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
   //      lcts.globalBestDualBoundValue );
   // double externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue(globalBestDualBoundValue);
   writeLoadCoordinatorStatisticsToCheckpointFile(loadCoordinatorStatisticsStream, nSolverInfo,
         lcts.globalBestDualBoundValue, lcts.externalGlobalBestDualBoundValue );
   loadCoordinatorStatisticsStream.close();

   if( lastCheckpointTimeStr[0] == ' ' )
   {
      /** the first time for checkpointing */
      if( racingWinnerParams )
      {
         /** save racing winner params */
         char racingWinnerParamsName[256];
         sprintf(racingWinnerParamsName,"%s%s_racing_winner_params.gz",
               paraParams->getStringParamValue(CheckpointFilePath),
               paraInitiator->getParaInstance()->getProbName());
         gzstream::ogzstream racingWinnerParamsStream;
         racingWinnerParamsStream.open(racingWinnerParamsName, std::ios::out | std::ios::binary);
         if( !racingWinnerParamsStream )
         {
            std::cout << "Racing winner parameter file cannot open. file name = " << racingWinnerParamsName << std::endl;
            exit(1);
         }
         racingWinnerParams->write(racingWinnerParamsStream);
         racingWinnerParamsStream.close();
      }
   }
   else
   {
      /** remove old check point files */
      sprintf(nodesFileName,"%s%s_%s_nodes_LC%d.gz",
            paraParams->getStringParamValue(CheckpointFilePath),
            paraInitiator->getParaInstance()->getProbName(),lastCheckpointTimeStr, paraComm->getRank());
      if( paraComm->getRank() == 0 )
      {
         sprintf(solutionFileName,"%s%s_%s_solution.gz",
               paraParams->getStringParamValue(CheckpointFilePath),
               paraInitiator->getParaInstance()->getProbName(),lastCheckpointTimeStr);
      }
      sprintf(solverStatisticsFileName,"%s%s_%s_solverStatistics_LC%d.gz",
            paraParams->getStringParamValue(CheckpointFilePath),
            paraInitiator->getParaInstance()->getProbName(),lastCheckpointTimeStr, paraComm->getRank());
      sprintf(loadCoordinatorStatisticsFileName,"%s%s_%s_loadCoordinatorStatistics_LC%d.gz",
            paraParams->getStringParamValue(CheckpointFilePath),
            paraInitiator->getParaInstance()->getProbName(),lastCheckpointTimeStr, paraComm->getRank());
      if( remove(nodesFileName) )
      {
         std::cout << "checkpoint nodes file cannot be removed: errno = " << strerror(errno) << std::endl;
         exit(1);
      }
      if ( remove(solutionFileName) )
      {
         std::cout << "checkpoint solution file cannot be removed: errno = " << strerror(errno) << std::endl;
         exit(1);
      }
      if ( remove(solverStatisticsFileName) )
      {
         std::cout << "checkpoint SolverStatistics file cannot be removed: errno = " << strerror(errno) << std::endl;
         exit(1);
      }
      if ( remove(loadCoordinatorStatisticsFileName) )
      {
         std::cout << "checkpoint LoadCoordinatorStatistics file cannot be removed: errno = " << strerror(errno) << std::endl;
         exit(1);
      }
   }

   char afterCheckpointingSolutionFileName[256];
   sprintf(afterCheckpointingSolutionFileName,"%s%s_after_checkpointing_solution.gz",
         paraParams->getStringParamValue(CheckpointFilePath),
         paraInitiator->getParaInstance()->getProbName() );
   gzstream::igzstream afterCheckpointingSolutionStream;
   afterCheckpointingSolutionStream.open(afterCheckpointingSolutionFileName, std::ios::in | std::ios::binary);
   if( afterCheckpointingSolutionStream  )
   {
      /** afater checkpointing solution file exists */
      afterCheckpointingSolutionStream.close();
      if ( remove(afterCheckpointingSolutionFileName) )
      {
         std::cout << "after checkpointing solution file cannot be removed: errno = " << strerror(errno) << std::endl;
         exit(1);
      }
   }

   /** update last checkpoint time string */
   strcpy(lastCheckpointTimeStr,newCheckpointTimeStr);
}

void
BbParaLoadCoordinator::writeLoadCoordinatorStatisticsToCheckpointFile(
      gzstream::ogzstream &loadCoordinatorStatisticsStream,
      int nSolverInfo,
      double globalBestDualBoundValue,
      double externalGlobalBestDualBoundValue
      )
{
   loadCoordinatorStatisticsStream.write((char *)&nSolverInfo, sizeof(int));
   lcts.isCheckpointState = true;
   if( paraNodeToKeepCheckpointFileNodes )
   {
      lcts.nMaxUsageOfNodePool = paraNodePool->getMaxUsageOfPool() + paraNodeToKeepCheckpointFileNodes->getNumOfNodes();
      lcts.nNodesInNodePool = paraNodePool->getNumOfNodes() + paraNodeToKeepCheckpointFileNodes->getNumOfNodes();
   }
   else
   {
      lcts.nMaxUsageOfNodePool = paraNodePool->getMaxUsageOfPool();
      lcts.nNodesInNodePool = paraNodePool->getNumOfNodes();
   }
   lcts.nNodesLeftInAllSolvers = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getNnodesInSolvers();
   lcts.globalBestDualBoundValue = std::max( globalBestDualBoundValue, lcts.globalBestDualBoundValue );
   lcts.externalGlobalBestDualBoundValue = dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(lcts.globalBestDualBoundValue);
   lcts.runningTime = paraTimer->getElapsedTime();
   if( nodesMerger )
   {
       lcts.addingNodeToMergeStructTime = nodesMerger->getAddingNodeToMergeStructTime();
       lcts.generateMergeNodesCandidatesTime = nodesMerger->getGenerateMergeNodesCandidatesTime();
       lcts.regenerateMergeNodesCandidatesTime = nodesMerger->getRegenerateMergeNodesCandidatesTime();
       lcts.mergeNodeTime = nodesMerger->getMergeNodeTime();
   }
   lcts.write(loadCoordinatorStatisticsStream);
}

void
BbParaLoadCoordinator::warmStart(
      )
{
   ///
   /// check parameter consistency
   ///
   if( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 || paraParams->getIntParamValue(RampUpPhaseProcess) == 2 )
   {
      if( !paraParams->getBoolParamValue(CollectOnce) )
      {
         std::cout << "*** When warm start with racing is specified, you should specify CollectOnce = TRUE ***" << std::endl;
         if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
         {
            std::cout << "*** When warm start with racing is specified, you cannot specify MergeNodesAtRestart = TRUE ***" << std::endl;
         }
         exit(1);
      }
      if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
      {
         std::cout << "*** When warm start with racing is specified, you cannot specify MergeNodesAtRestart = TRUE ***" << std::endl;
         exit(1);
      }
      std::cout << "*** Warm start with racing ramp-up ***" << std::endl;
   }
   else
   {
      if( paraParams->getIntParamValue(RampUpPhaseProcess) == 3 )
      {
         paraParams->setIntParamValue(RampUpPhaseProcess, 0);
      }
      std::cout << "*** Warm start with normal ramp-up ***" << std::endl;
   }


   restarted = true;
   /** write previous statistics information */
   writePreviousStatisticsInformation();

   if( paraParams->getIntParamValue(RampUpPhaseProcess) == 0 )  // if it is not racing ramp-up
   {
      /** try to read racing winner params */
      char racingWinnerParamsName[256];
      sprintf(racingWinnerParamsName,"%s%s_racing_winner_params.gz",
            paraParams->getStringParamValue(CheckpointFilePath),
            paraInitiator->getParaInstance()->getProbName());
      gzstream::igzstream racingWinnerParamsStream;
      racingWinnerParamsStream.open(racingWinnerParamsName, std::ios::in | std::ios::binary);
      if( racingWinnerParamsStream )
      {
         assert(!racingWinnerParams);
         racingWinnerParams = paraComm->createParaRacingRampUpParamSet();
         racingWinnerParams->read(paraComm, racingWinnerParamsStream);
         racingWinnerParamsStream.close();
         for( int i = 1; i < paraComm->getSize(); i++ )
         {
            /** send racing winner params: NOTE: should not broadcast. if we do it, solver routine need to recognize staring process */
            PARA_COMM_CALL(
                  racingWinnerParams->send(paraComm, i)
            );
         }
         std::cout << "*** winner parameter is read from " << racingWinnerParamsName << "***" << std::endl;
      }
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   /** set solution and get internal incumbent value */
   char afterCheckpointingSolutionFileName[256];
   sprintf(afterCheckpointingSolutionFileName,"%s%s_after_checkpointing_solution.gz",
         paraParams->getStringParamValue(CheckpointFilePath),
         paraInitiator->getParaInstance()->getProbName() );
   double incumbentValue = paraInitiator->readSolutionFromCheckpointFile(afterCheckpointingSolutionFileName);
   if( !paraParams->getBoolParamValue(Quiet) )
   {
      bbParaInitiator->writeSolution("[Warm started from "+std::string(bbParaInitiator->getPrefixWarm())+" : the solution from the checkpoint file]");
   }

   int n = 0;
   BbParaNode *paraNode;
   bool onlyBoundChanges = false;
   if( !paraParams->getBoolParamValue(TransferLocalCuts) && !paraParams->getBoolParamValue(TransferConflictCuts) )
   {
      onlyBoundChanges = true;
   }
   if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
   {
      if( nodesMerger ) delete nodesMerger;
      nodesMerger = new BbParaNodesMerger(dynamic_cast<BbParaInstance *>(bbParaInitiator->getParaInstance())->getVarIndexRange(),
            nBoundChangesOfBestNode,paraTimer,dynamic_cast<BbParaInstance *>(bbParaInitiator->getParaInstance()),paraParams);
   }

   BbParaNodePoolForMinimization tempParaNodePool(paraParams->getRealParamValue(BgapCollectingMode));

   for(;;)
   {
      paraNode = bbParaInitiator->readParaNodeFromCheckpointFile(onlyBoundChanges);
      //      paraParamSet->getBoolParamValue(MergingNodeStatusInCheckpointFile) );
      if( paraNode == 0 )
         break;
#ifdef UG_DEBUG_SOLUTION
#ifdef UG_DEBUG_SOLUTION_OPT_PATH
      if( paraNode->getDiffSubproblem() && (!paraNode->getDiffSubproblem()->isOptimalSolIncluded()) )
      {
         delete paraNode;
         paraNode = 0;
         continue;
      }
#endif
#endif 
      n++;
      if( paraNode->getAncestor() )
      {
         std::cout << "Checkpoint node has ancestor: " << paraNode->toString() << std::endl;
         std::cout << "Something wrong for this checkpoint file." << std::endl;
         exit(1);
      }
      paraNode->setDualBoundValue(paraNode->getInitialDualBoundValue());
      // paraNodePool->insert(paraNode);   /** in order to sort ParaNodes, insert paraNodePool once */
      if( paraParams->getBoolParamValue(MergeNodesAtRestart)
            || paraParams->getIntParamValue(NEagerToSolveAtRestart) > 0
            || paraParams->getIntParamValue(NNodesToKeepInCheckpointFile) > 0 )
      {
         tempParaNodePool.insert(paraNode);  /** in order to sort ParaNodes with new dual value, insert it to tempParaNodePool once */
         // addNodeToMergeNodeStructs(paraNode);
      }
      else
      {
         paraNodePool->insert(paraNode);   /** in order to sort ParaNodes, insert paraNodePool once */
      }
   }

   if( paraParams->getIntParamValue(NNodesToKeepInCheckpointFile) > 0 )
   {
      if( paraParams->getIntParamValue(NEagerToSolveAtRestart) > 0 )
      {
         std::cout << "### NNodesToKeepInCheckpointFile is specified to "
               << paraParams->getIntParamValue(NNodesToKeepInCheckpointFile)
               << " and also NEagerToSolveAtRestart is specified to "
               << paraParams->getIntParamValue(NEagerToSolveAtRestart)
               << ". The both values should not greater than 0 together." << std::endl;
         exit(-1);
      }
      if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
      {
         std::cout << "### NNodesToKeepInCheckpointFile is specified to "
               << paraParams->getIntParamValue(NNodesToKeepInCheckpointFile)
               << " and also MergeNodesAtRestart = TRUE is specified. This combination is not allowed." << std::endl;
         exit(-1);
      }
      if( (signed)tempParaNodePool.getNumOfNodes() <= paraParams->getIntParamValue(NNodesToKeepInCheckpointFile) )
      {
         std::cout << "### NNodesToKeepInCheckpointFile is specified to " << paraParams->getIntParamValue(NNodesToKeepInCheckpointFile) <<
               ", but the number of nodes in checkpoint file is " << tempParaNodePool.getNumOfNodes() << ". ###" << std::endl;
         exit(-1);
      }
      if( paraParams->getIntParamValue(RampUpPhaseProcess) != 0  )
      {
         std::cout << "### NEagerToSolveAtRestart is specified to "
               << paraParams->getIntParamValue(NNodesToKeepInCheckpointFile)
               << ", but RampUpPhaseProcess != 0 is specified. This combination is not allowed." << std::endl;
         exit(-1);
      }
      paraNodeToKeepCheckpointFileNodes = new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
      for(int i = 0; i < paraParams->getIntParamValue(NNodesToKeepInCheckpointFile); i++ )
      {
         BbParaNode *tempParaNode = tempParaNodePool.extractNode();
         paraNodeToKeepCheckpointFileNodes->insert(tempParaNode);
      }
      while( tempParaNodePool.getNumOfNodes() > 0 )
      {
         BbParaNode *tempParaNode = tempParaNodePool.extractNode();
         paraNodePool->insert(tempParaNode);
      }
      std::cout << "### NNodesToKeepInCheckpointFile is specified to "
            << paraParams->getIntParamValue(NNodesToKeepInCheckpointFile)
            << " ###" << std::endl;
      std::cout << "### The number of no process nodes = " << paraNodeToKeepCheckpointFileNodes->getNumOfNodes()
            << " ###" << std::endl;
      std::cout << "### The number of nodes will be processed = " << paraNodePool->getNumOfNodes()
            << " ###" << std::endl;
   }

   if( paraParams->getIntParamValue(NEagerToSolveAtRestart) > 0 )
   {
      if( (signed)tempParaNodePool.getNumOfNodes() < paraParams->getIntParamValue(NEagerToSolveAtRestart) )
      {
         std::cout << "### NEagerToSolveAtRestart is specified to " << paraParams->getIntParamValue(NEagerToSolveAtRestart) <<
               ", but the number of nodes in checkpoint file is " << tempParaNodePool.getNumOfNodes() << ". ###" << std::endl;
         exit(-1);
      }
      if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
      {
         std::cout << "### NEagerToSolveAtRestart is specified to "
               << paraParams->getIntParamValue(NEagerToSolveAtRestart)
               << " and also MergeNodesAtRestart = TRUE is specified. This combination is not allowed." << std::endl;
         exit(-1);
      }
      if( paraParams->getIntParamValue(RampUpPhaseProcess) != 0  )
      {
         std::cout << "### NEagerToSolveAtRestart is specified to "
               << paraParams->getIntParamValue(NEagerToSolveAtRestart)
               << ", but RampUpPhaseProcess != 0 is specified. This combination is not allowed." << std::endl;
         exit(-1);
      }
      unprocessedParaNodes = new BbParaNodePoolForMinimization(paraParams->getRealParamValue(BgapCollectingMode));
      for(int i = 0; i < paraParams->getIntParamValue(NEagerToSolveAtRestart); i++ )
      {
         BbParaNode *tempParaNode = tempParaNodePool.extractNode();
         paraNodePool->insert(tempParaNode);
      }
      while( tempParaNodePool.getNumOfNodes() > 0 )
      {
         BbParaNode *tempParaNode = tempParaNodePool.extractNode();
         unprocessedParaNodes->insert(tempParaNode);
      }
      std::cout << "### NEagerToSolveAtRestart is specified to "
            << paraParams->getIntParamValue(NEagerToSolveAtRestart)
            << " ###" << std::endl;
      std::cout << "### The number of no process nodes in the beginning = " << unprocessedParaNodes->getNumOfNodes()
            << " ###" << std::endl;
   }

   // std::cout << "insrt to node pool: " << paraTimer->getElapsedTime() << std::endl;

   if( paraParams->getBoolParamValue(MergeNodesAtRestart) )
   {
      int nMerge = 0;
      BbParaNode *tempNode = 0;
      while( ( tempNode = tempParaNodePool.extractNode() ) )
      {
         if( nBoundChangesOfBestNode < 0 )
         {
            if( tempNode->getDiffSubproblem() )
            {
               nBoundChangesOfBestNode = dynamic_cast<BbParaDiffSubproblem *>(tempNode->getDiffSubproblem())->getNBoundChanges();
            }
            else
            {
               nBoundChangesOfBestNode = 0;
            }
         }
         paraNodePool->insert(tempNode);
         if( paraParams->getIntParamValue(UG::NMergingNodesAtRestart) < 0 ||
               nMerge < paraParams->getIntParamValue(UG::NMergingNodesAtRestart) )
         {
            assert(nodesMerger);
            nodesMerger->addNodeToMergeNodeStructs(tempNode);
            nMerge++;
         }
      }
      assert(nodesMerger);
      nodesMerger->generateMergeNodesCandidates(paraComm, paraInitiator);
   }

   lcts.globalBestDualBoundValue = paraNodePool->getBestDualBoundValue();
   lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);

   // std::cout << "merging finished:" << paraTimer->getElapsedTime() << std::endl;

   if( logSolvingStatusFlag )
   {
      if( bbParaInitiator->getGlobalBestIncumbentSolution() )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " Warm started from "
         << paraInitiator->getPrefixWarm()
         << " : " << n << " ParaNodes read. Current incumbent value = "
         <<  bbParaInitiator->convertToExternalValue(
               bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() )
               << std::endl;
      }
      else
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " Warm started from "
         << paraInitiator->getPrefixWarm()
         << " : " << n << " ParaNodes read. No solution is generated." << std::endl;
      }
   }
#ifdef _DEBUG_LB
   std::cout << paraTimer->getElapsedTime()
   << " Warm started from "
   << paraInitiator->getPrefixWarm()
   << " : " << n << " ParaNodes read. Current incumbent value = "
   <<  paraInitiator->convertToExternalValue(
         paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() )
         << std::endl;
#endif
   runningPhase = RampUpPhase;
   if( paraParams->getIntParamValue(RampUpPhaseProcess) == 0 )
   {
      for( int i = 1; i < paraComm->getSize(); i++ )
      {
         /** send internal incumbent value */
         PARA_COMM_CALL(
               paraComm->send( &incumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
         );
         if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) && bbParaInitiator->getGlobalBestIncumbentSolution() )
         {
            bbParaInitiator->getGlobalBestIncumbentSolution()->send(paraComm, i);
         }
         /** send internal global dual bound value */
         PARA_COMM_CALL(
               paraComm->send( &lcts.globalBestDualBoundValue, 1, ParaDOUBLE, i, TagGlobalBestDualBoundValueAtWarmStart )
         );
      }
      warmStartNodeTransferring = true;
      (void) sendParaTasksToIdleSolvers();
      warmStartNodeTransferring = false;
      if( paraSolverPool->getNumInactiveSolvers() > 0 )
      {
         runningPhase = RampUpPhase;
      }
      else
      {
         // without consideration of keeping nodes in checkpoint file
         double globalBestDualBoundValueLocal =
            std::max (
                  std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
                  lcts.globalBestDualBoundValue );
         if( paraParams->getBoolParamValue(CollectOnce) )
         {
            if( paraParams->getBoolParamValue(DualBoundGainTest) )
            {
               if( ( paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode) ||
                     ( paraNodePool->getNumOfNodes()
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode)*2  &&
                                    paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) ) )
               {
                  sendRampUpToAllSolvers();
                  runningPhase = NormalRunningPhase;
               }
               else
               {
                  runningPhase = RampUpPhase;
               }
            }
            else
            {
               if( ( (signed)paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode) ||
                     ( (signed)paraNodePool->getNumOfNodes()
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode)*2  &&
                                    paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) ) )
               {
                  sendRampUpToAllSolvers();
                  runningPhase = NormalRunningPhase;
               }
               else
               {
                  runningPhase = RampUpPhase;
               }
            }
         }
         else
         {
            if( paraParams->getBoolParamValue(DualBoundGainTest) )
            {
               if( ( paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal )
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode) ||
                     ( paraNodePool->getNumOfNodes()
                                    > paraParams->getIntParamValue(NChangeIntoCollectingMode)*paraParams->getRealParamValue(MultiplierForCollectingMode)*2  &&
                                    paraNodePool->getNumOfGoodNodes( globalBestDualBoundValueLocal ) > 0  ) ) )
               {
                  sendRampUpToAllSolvers();
                  runningPhase = NormalRunningPhase;
               }
               else
               {
                  runningPhase = RampUpPhase;
               }
            }
            else
            {
               sendRampUpToAllSolvers();
               runningPhase = NormalRunningPhase;
            }
         }
      }
      // std::cout << "before run:" << paraTimer->getElapsedTime() << std::endl;
      // exit(1);
      run();
   }
   else
   {
      if( paraParams->getIntParamValue(RampUpPhaseProcess) == 1
            || paraParams->getIntParamValue(RampUpPhaseProcess) == 2 )  // if it isracing ramp-up
      {
         /// racing ramp-up
         assert(!racingWinnerParams);

         paraNode = paraNodePool->extractNode();
         warmStartNodeTransferring = true;
         ParaRacingRampUpParamSet **racingRampUpParams = new ParaRacingRampUpParamSetPtr[paraComm->getSize()];
         paraInitiator->generateRacingRampUpParameterSets( (paraComm->getSize()-1), racingRampUpParams );

         if( paraNodePool->isEmpty() )
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               int noKeep = 0;
               PARA_COMM_CALL(
                     paraComm->send( &noKeep, 1, ParaINT, i, TagKeepRacing)
                     );
               PARA_COMM_CALL(
                     racingRampUpParams[i-1]->send(paraComm, i)
                     );
            }
         }
         else
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               int keep = 1;
               PARA_COMM_CALL(
                     paraComm->send( &keep, 1, ParaINT, i, TagKeepRacing)
                     );
               PARA_COMM_CALL(
                     racingRampUpParams[i-1]->send(paraComm, i)
                     );
            }
         }

         run(paraNode, (paraComm->getSize()-1), racingRampUpParams );

         if( paraNodePool->isEmpty() )
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               /** send internal incumbent value */
               PARA_COMM_CALL(
                     paraComm->send( &incumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
               );
               if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) && bbParaInitiator->getGlobalBestIncumbentSolution() )
               {
                  bbParaInitiator->getGlobalBestIncumbentSolution()->send(paraComm, i);
               }
               /** send internal global dual bound value */
               PARA_COMM_CALL(
                     paraComm->send( &lcts.globalBestDualBoundValue, 1, ParaDOUBLE, i, TagGlobalBestDualBoundValueAtWarmStart )
               );
               if( racingRampUpParams[i-1] ) delete racingRampUpParams[i-1];
            }
         }
         else
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               /** send internal incumbent value */
               PARA_COMM_CALL(
                     paraComm->send( &incumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
               );
               if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) && bbParaInitiator->getGlobalBestIncumbentSolution() )
               {
                  bbParaInitiator->getGlobalBestIncumbentSolution()->send(paraComm, i);
               }
               /** send internal global dual bound value */
               PARA_COMM_CALL(
                     paraComm->send( &lcts.globalBestDualBoundValue, 1, ParaDOUBLE, i, TagGlobalBestDualBoundValueAtWarmStart )
               );
               PARA_COMM_CALL(
                     paraComm->send( NULL, 0, ParaBYTE, i, TagKeepRacing)
                     );
               if( racingRampUpParams[i-1] ) delete racingRampUpParams[i-1];
            }
         }
         delete [] racingRampUpParams;
      }
   }

}

void
BbParaLoadCoordinator::writePreviousStatisticsInformation(
      )
{
   /* read previous LoadCoordinator statistics */
   char loadCoordinatorStatisticsFileName[256];
   sprintf(loadCoordinatorStatisticsFileName,"%s_loadCoordinatorStatistics_LC0.gz", paraInitiator->getPrefixWarm() );
   gzstream::igzstream  loadCoordinatorStatisticsStream;
   loadCoordinatorStatisticsStream.open(loadCoordinatorStatisticsFileName, std::ios::in | std::ios::binary);
   if( !loadCoordinatorStatisticsStream )
   {
      std::cout << "checkpoint LoadCoordinatorStatistics file cannot open: file name = " <<  loadCoordinatorStatisticsFileName << std::endl;
      exit(1);
   }
   int nSolverStatistics;
   loadCoordinatorStatisticsStream.read((char *)&nSolverStatistics, sizeof(int));
   BbParaLoadCoordinatorTerminationState *prevLcts = new BbParaLoadCoordinatorTerminationState();
   if( !prevLcts->read(paraComm, loadCoordinatorStatisticsStream) )
   {
      std::cout << "checkpoint LoadCoordinatorStatistics file cannot read: file name = " <<  loadCoordinatorStatisticsFileName << std::endl;
      exit(1);
   }
   loadCoordinatorStatisticsStream.close();

   /* open Solver statistics file */
   char solverStatisticsFileName[256];
   sprintf(solverStatisticsFileName,"%s_solverStatistics_LC0.gz", paraInitiator->getPrefixWarm() );
   gzstream::igzstream  solverStatisticsStream;
   solverStatisticsStream.open(solverStatisticsFileName, std::ios::in | std::ios::binary);
   if( !solverStatisticsStream )
   {
      std::cout << "checkpoint SolverStatistics file cannot open: file name = " <<  solverStatisticsFileName << std::endl;
      exit(1);
   }

   /* opne output statistics file */
   char previousStatisticsFileName[256];
   sprintf(previousStatisticsFileName, "%s_statistics_w%05lld_LC0",
         paraInitiator->getPrefixWarm(),
         prevLcts->nWarmStart);
   std::ofstream ofsStatistics;
   ofsStatistics.open(previousStatisticsFileName);
   if( !ofsStatistics )
   {
      std::cout << "previous statistics file cannot open : file name = " << previousStatisticsFileName << std::endl;
      exit(1);
   }

   /* read and write solver statistics */
   for( int i = 0; i < nSolverStatistics; i++ )
   {
      ParaSolverTerminationState *psts = paraComm->createParaSolverTerminationState();
      if( !psts->read(paraComm, solverStatisticsStream) )
      {
         std::cout << "checkpoint SolverStatistics file cannot read: file name = " <<  solverStatisticsFileName << std::endl;
         exit(1);
      }
      ofsStatistics << psts->toString(paraInitiator);
      delete psts;
   }

   /* write LoadCoordinator statistics */
   ofsStatistics << prevLcts->toString();

   /* update warm start counter */
   lcts.nWarmStart = prevLcts->nWarmStart + 1;
   lcts.globalBestDualBoundValue = std::max( prevLcts->globalBestDualBoundValue, lcts.globalBestDualBoundValue );
   lcts.externalGlobalBestDualBoundValue = prevLcts->externalGlobalBestDualBoundValue;
   delete prevLcts;

   /* close solver statistics file and output file */
   solverStatisticsStream.close();
   ofsStatistics.close();
}

#endif // End of UG_WITH_ZLIB

void
BbParaLoadCoordinator::run(
      ParaTask *paraNode
      )
{
   if( !isHeaderPrinted && outputTabularSolvingStatusFlag )
   {
      outputTabularSolvingStatusHeader();            /// should not call virutal function in constructor
   }

   assert(!paraRacingSolverPool);
   // without consideration of keeping nodes in checkpoint file
   double globalBestDualBoundValueLocal =
      std::max (
            std::min( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGlobalBestDualBoundValue(), paraNodePool->getBestDualBoundValue() ),
            lcts.globalBestDualBoundValue );

   if( paraParams->getIntParamValue(RampUpPhaseProcess) != 3 )
   {
      int destination = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->activateSolver(
            dynamic_cast<BbParaNode *>(paraNode),
            dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool), (runningPhase==RampUpPhase),
            paraNodePool->getNumOfGoodNodes(globalBestDualBoundValueLocal), averageLastSeveralDualBoundGains );
      lcts.nSent++;
      if( paraParams->getIntParamValue(RampUpPhaseProcess) == 0 &&
            paraParams->getBoolParamValue(CollectOnce)
            )
      {
         int nCollect = -1;
         PARA_COMM_CALL(
               paraComm->send( &nCollect, 1, ParaINT, destination, TagCollectAllNodes )
         );
      }
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         int token[2];
         token[0] = 1;
         token[1] = -1;
         PARA_COMM_CALL(
               paraComm->send( token, 2, ParaINT, token[0], TagToken )
         );
      }
      writeTransferLog(destination);

      BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << destination << " < "
         << bbParaInitiator->convertToExternalValue(
               dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue() );
         if( bbParaInitiator->getGlobalBestIncumbentSolution() )
         {
            if( bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue()) > displayInfOverThisValue )
            {
               *osLogSolvingStatus << " ( Inf )";
            }
            else
            {
               *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue()) * 100 << "% )";
            }
         }
         if( paraParams->getBoolParamValue(LightWeightRootNodeProcess) &&
               runningPhase != RampUpPhase && (!paraRacingSolverPool) &&
               paraSolverPool->getNumInactiveSolvers() >
                  ( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(RatioToApplyLightWeightRootProcess) ) )
         {
            *osLogSolvingStatus << " L";
         }
         *osLogSolvingStatus << std::endl;
      }
#ifdef DEBUG_LB
      std::cout << paraTimer->getElapsedTime()
      << " S." << destination << " > "
      << paraInitiator->convertToExternalValue(
            paraNode->getDualBoundValue() );
      if( paraInitiator->getGlobalBestIncumbentSolution() )
      {
         if( paraInitiator->getGap(paraNode->getDualBoundValue()) > displayInfOverThisValue )
         {
            std::cout << " ( Inf )";
         }
         else
         {
            std::cout << " ( " << paraInitiator->getGap(paraNode->getDualBoundValue()) * 100 << "% )";
         }
      }
      if( paraParams->getBoolParamValue(LightWeightRootNodeProcess) &&
            runningPhase != RampUpPhase && (!paraRacingSolverPool) &&
            paraSolverPool->getNumInactiveSolvers() >
               ( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(RatioToApplyLightWeightRootProcess) ) )
      {
         std::cout << " L";
      }
      std::cout << std::endl;
#endif
   }
   else   // self-split ramp-up
   {
      BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);
      DEF_BB_PARA_COMM(bbParaComm, paraComm);
      delete paraNode;
      for( unsigned int i = 0; i < paraSolverPool->getNSolvers(); i++ )
      {
         paraNode = bbParaComm->createParaNode(
               TaskId(), TaskId(), 0, -DBL_MAX, -DBL_MAX, -DBL_MAX,
               bbParaInitiator->makeRootNodeDiffSubproblem()
               );
         int destination = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->activateSolver(
               dynamic_cast<BbParaNode *>(paraNode),
               dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool), (runningPhase==RampUpPhase),
               paraNodePool->getNumOfGoodNodes(globalBestDualBoundValueLocal), averageLastSeveralDualBoundGains );
         lcts.nSent++;
         writeTransferLog(destination);

         if( logSolvingStatusFlag )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << destination << " < "
            << bbParaInitiator->convertToExternalValue(
                  dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue() );
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               if( bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue()) > displayInfOverThisValue )
               {
                  *osLogSolvingStatus << " ( Inf )";
               }
               else
               {
                  *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue()) * 100 << "% )";
               }
            }
            *osLogSolvingStatus << std::endl;
         }
         if( paraParams->getBoolParamValue(Deterministic) )
          {
             int token[2];
             token[0] = 1;
             token[1] = -1;
             PARA_COMM_CALL(
                   paraComm->send( token, 2, ParaINT, token[0], TagToken )
             );
          }
      }
   }

   run();
}

int
BbParaLoadCoordinator::processRacingRampUpTagSolverState(
      int source,
      int tag
      )
{

   BbParaSolverState *solverState = dynamic_cast<BbParaSolverState *>(paraComm->createParaSolverState());
   solverState->receive(paraComm, source, tag);

#ifdef _DEBUG_DET
   if( paraDetTimer )
   {
      std::cout << "Rank " << source << ": ET = " << paraDetTimer->getElapsedTime() << ", Det time = " << solverState->getDeterministicTime() << std::endl;
   }
#endif

   if( paraDetTimer
         && paraDetTimer->getElapsedTime() < solverState->getDeterministicTime() )

   {
      paraDetTimer->update( solverState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
   }

   assert(solverState->isRacingStage());
   if( !restartingRacing )  // restartingRacing means that LC is terminating racing solvers.
   {
      dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->updateSolverStatus(source,
                                      solverState->getNNodesSolved(),
                                      solverState->getNNodesLeft(),
                                      solverState->getSolverLocalBestDualBoundValue());
      assert( dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNumNodesLeft(source) == solverState->getNNodesLeft() );
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " S." << source << " | "
      << bbParaInitiator->convertToExternalValue(
            solverState->getSolverLocalBestDualBoundValue()
            );
      if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
            bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue
            || solverState->getNNodesLeft() == 0 )
      {
         *osLogSolvingStatus << " ( Inf )";
      }
      else
      {
         *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
      }
      *osLogSolvingStatus << " [ " << solverState->getNNodesLeft() << " ]";
      double globalBestDualBoundValue = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getGlobalBestDualBoundValue();
      if( paraNodePool->getNumOfNodes() > 0 )
      {
         globalBestDualBoundValue = std::min( globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
      }
      *osLogSolvingStatus << " ** G.B.: " << bbParaInitiator->convertToExternalValue(globalBestDualBoundValue);
      if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
            bbParaInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
      {
         *osLogSolvingStatus << " ( Inf ) ";
      }
      else
      {
         *osLogSolvingStatus << " ( " << bbParaInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
      }
      *osLogSolvingStatus << "[ " << dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesLeftInBestSolver()
 //     <<" ] ** RR" << std::endl;
      <<" ] ** RR " << solverState->getDeterministicTime() << std::endl;   // for debug
   }
#ifdef _DEBUG_LB
   std::cout << paraTimer->getElapsedTime()
   << " S." << source << " | "
   << bbParaInitiator->convertToExternalValue(
         solverState->getSolverLocalBestDualBoundValue()
         );
   if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
         bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) > displayInfOverThisValue
         || solverState->getNNodesLeft() == 0 )
   {
      std::cout << " ( Inf )";
   }
   else
   {
      std::cout << " ( " << bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) * 100 << "% )";
   }
   std::cout << " [ " << solverState->getNNodesLeft() << " ]";
   double globalBestDualBoundValue = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getGlobalBestDualBoundValue();
   std::cout << " ** G.B.: " << paraInitiator->convertToExternalValue(globalBestDualBoundValue);
   if( !bbParaInitiator->getGlobalBestIncumbentSolution() ||
         bbParaInitiator->getGap(globalBestDualBoundValue) > displayInfOverThisValue )
   {
      std::cout << " ( Inf ) ";
   }
   else
   {
      std::cout << " ( " << bbParaInitiator->getGap(globalBestDualBoundValue) * 100 << "% ) ";
   }
   std::cout << "[ " << dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesLeftInBestSolver()
   <<" ] ** RR" << std::endl;
#endif

   if( !paraParams->getBoolParamValue(NoUpperBoundTransferInRacing) )
   {
      /** the following should be before noticationId back to the source solver */
      if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) )
      {
         if( bbParaInitiator->getGlobalBestIncumbentSolution() &&
               bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
               < solverState->getGlobalBestPrimalBoundValue() )
         {
            bbParaInitiator->getGlobalBestIncumbentSolution()->send(paraComm, source);
         }
      }
   }

   // if( paraParams->getBoolParamValue(CheckGapInLC) )
   if( !givenGapIsReached )
   {
      if( bbParaInitiator->getAbsgap(solverState->getSolverLocalBestDualBoundValue() ) <
            bbParaInitiator->getAbsgapValue() ||
            bbParaInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) <
            bbParaInitiator->getGapValue()
            )
      {
         for( unsigned int i = 1; i <= paraRacingSolverPool->getNSolvers(); i++ )
         {
            if( paraRacingSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
            {
               PARA_COMM_CALL(
                     paraComm->send( NULL, 0, ParaBYTE, i, TagGivenGapIsReached )
               );
               paraSolverPool->interruptRequested(i);
            }
         }
         // std::cout << "current dual = "  << paraInitiator->convertToExternalValue(solverState->getSolverLocalBestDualBoundValue()) <<std::endl;
         // std::cout << "pool best dual = " << paraInitiator->convertToExternalValue(dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getGlobalBestDualBoundValue()) << std::endl;
         // std::cout << "current gap = " << paraInitiator->getGap(solverState->getSolverLocalBestDualBoundValue()) <<std::endl;
         givenGapIsReached = true;
      }
   }

   double lcBestDualBoundValue = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getGlobalBestDualBoundValue();

   PARA_COMM_CALL(
         paraComm->send( &lcBestDualBoundValue, 1, ParaDOUBLE, source, TagLCBestBoundValue)
         );
   unsigned int notificationId = solverState->getNotificaionId();
   PARA_COMM_CALL(
         paraComm->send( &notificationId, 1, ParaUNSIGNED, source, TagNotificationId)
         );

   if( lcts.globalBestDualBoundValue <  lcBestDualBoundValue )
   {
      if( paraNodePool->getNumOfNodes() > 0 )
      {
         lcts.globalBestDualBoundValue = std::min( lcBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
      }
      else
      {
         lcts.globalBestDualBoundValue = lcBestDualBoundValue;
      }

      lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
   }
   delete solverState;
   return 0;
}

int
BbParaLoadCoordinator::processRacingRampUpTagCompletionOfCalculation(
      int source,
      int tag
      )
{
   BbParaCalculationState *calcState = dynamic_cast<BbParaCalculationState *>(paraComm->createParaCalculationState());
   calcState->receive(paraComm, source, tag);
   writeTransferLogInRacing(source, calcState);

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( logSolvingStatusFlag )
   {
      switch ( calcState->getTerminationState() )
      {
      case CompTerminatedInRacingStage:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(TERMINATED_IN_RACING_STAGE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByInterruptRequest:
      case CompInterruptedInRacingStage:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_TIME_LIMIT or INTERRUPTED_BY_SOME_SOLVER_TERMINATED_IN_RACING_STAGE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByTimeLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_TIME_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      case CompTerminatedByMemoryLimit:
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " S." << source << " >(INTERRUPTED_BY_MEMORY_LIMIT) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
         break;
      }
      default:
         if( warmStartNodeTransferring
               && calcState->getTerminationState() == CompTerminatedByAnotherTask )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << source << " >(INTERRUPTED_BY_ANOTHER_NODE) " << bbParaInitiator->convertToExternalValue(calcState->getDualBoundValue());
            break;
         }
         THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
      }
      *osLogSolvingStatus << ", ct:" << calcState->getCompTime()
            << ", nr:" << calcState->getNRestarts()
            << ", n:" << calcState->getNSolved()
            << ", rt:" << calcState->getRootTime()
            << ", avt:" << calcState->getAverageNodeCompTimeExcpetRoot()
            << std::endl;
   }
#ifdef _DEBUG_LB
   switch ( calcState->getTerminationState() )
   {
   case CompTerminatedInRacingStage:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(TERMINATED_IN_RACING_STAGE)";
      break;
   }
   case CompTerminatedByInterruptRequest:
   case CompInterruptedInRacingStage:
   {
      std::cout << paraTimer->getElapsedTime()
      << " S." << source << " >(INTERRUPTED_BY_TIME_LIMIT or INTERRUPTED_BY_SOME_SOLVER_TERMINATED_IN_RACING_STAGE)";
      break;
   }
   default:
      THROW_LOGICAL_ERROR2("Invalid termination: termination state = ", calcState->getTerminationState() )
   }
   std::cout << std::endl;
#endif

   if( calcState->getTerminationState() == CompTerminatedInRacingStage )
   {
      racingTermination = true; // even if interruptIsRequested, 
                                // solver should have been terminated before receiveing it
      if( osStatisticsRacingRampUp )
      {
         *osStatisticsRacingRampUp << "######### Solver Rank = " <<
               source << " is terminated in racing stage #########" << std::endl;
      }

      if( (dynamic_cast<UG::BbParaInitiator *>(paraInitiator)->getNSolutions() >=
              paraParams->getIntParamValue(OmitTerminationNSolutionsInRacing) )
            )
      {
         nSolvedRacingTermination = calcState->getNSolved();
         if( !EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, bbParaInitiator->getEpsilon() ) &&
               EPSEQ( minmalDualBoundNormalTermSolvers, DBL_MAX, bbParaInitiator->getEpsilon() ) )
         {
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
            }
            else
            {
               minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
            }
         }
         if( EPSLE(lcts.globalBestDualBoundValue, calcState->getDualBoundValue(), bbParaInitiator->getEpsilon()) &&
               minmalDualBoundNormalTermSolvers < calcState->getDualBoundValue() )
         {
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), calcState->getDualBoundValue() );
            }
            else
            {
               minmalDualBoundNormalTermSolvers = calcState->getDualBoundValue();
            }
         }
         if( bbParaInitiator->getGlobalBestIncumbentSolution() && (!givenGapIsReached) &&
               ( EPSEQ( calcState->getDualBoundValue(), bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), paraInitiator->getEpsilon() ) ||
                    EPSEQ( calcState->getDualBoundValue(), -DBL_MAX, paraInitiator->getEpsilon() ) ||
                    EPSEQ( calcState->getDualBoundValue(), DBL_MAX, paraInitiator->getEpsilon() ) ||
                    ( paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) &&
                          // distributed domain propagation could causes the following situation
                          bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() < calcState->getDualBoundValue() ) ||
                     ( calcState->getNSolved() == 0 ) ) )
         {
            lcts.globalBestDualBoundValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
            if( paraNodePool->getNumOfNodes() > 0 )
            {
               lcts.globalBestDualBoundValue = std::min( lcts.globalBestDualBoundValue, paraNodePool->getBestDualBoundValue() );
            }
            lcts.externalGlobalBestDualBoundValue = bbParaInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
            if( bbParaInitiator->getGlobalBestIncumbentSolution() )
            {
               minmalDualBoundNormalTermSolvers = std::min( bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(), lcts.globalBestDualBoundValue );
            }
            else
            {
               minmalDualBoundNormalTermSolvers = lcts.globalBestDualBoundValue;
            }
         }
         /*
         if( paraInitiator->getGlobalBestIncumbentSolution() &&
               paraParamSet->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) &&
               // distributed domain propagation could causes the following situation
               paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue() < calcState->getDualBoundValue() &&
               EPSGE( paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue(), lcts.globalBestDualBoundValue, paraInitiator->getEpsilon() ) )
         {
            lcts.globalBestDualBoundValue = paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFuntionValue();
            lcts.externalGlobalBestDualBoundValue = paraInitiator->convertToExternalValue(lcts.globalBestDualBoundValue);
            minmalDualBoundNormalTermSolvers = lcts.globalBestDualBoundValue;
         }
         */
         assert( paraNodePool->getNumOfNodes() > 0 || !bbParaInitiator->getGlobalBestIncumbentSolution() ||
                 ( bbParaInitiator->getGlobalBestIncumbentSolution() &&
                   EPSEQ(bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(),lcts.globalBestDualBoundValue, paraInitiator->getEpsilon() ) ) );
         if( !warmStartNodeTransferring || ( warmStartNodeTransferring && paraNodePool->isEmpty() ) )
         {
            dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->updateSolverStatus(source,
                  calcState->getNSolved(), 0, minmalDualBoundNormalTermSolvers);
            if( outputTabularSolvingStatusFlag )
            {
               outputTabularSolvingStatus(' ');
            }
         }
      }
   }

   if( calcState->getTerminationState() == CompTerminatedByTimeLimit )
   {
      hardTimeLimitIsReached = true;
      // std::cout << "####### Rank " << paraComm->getRank() << " solver terminated with timelimit in solver side. #######" << std::endl;
      // std::cout << "####### Final statistics may be messed up!" << std::endl;
   }
   if( calcState->getTerminationState() == CompTerminatedByMemoryLimit )
   {
      memoryLimitIsReached = true;
   }

   delete calcState;

   // if( !warmStartNodeTransferring || ( warmStartNodeTransferring && paraNodePool->isEmpty() ) )
   // {
      ParaSolverTerminationState *termState = paraComm->createParaSolverTerminationState();
      termState->receive(paraComm, source, TagTermStateForInterruption);

      if( paraDetTimer )
      {
         if( paraDetTimer->getElapsedTime() < termState->getDeterministicTime() )
         {
            paraDetTimer->update( termState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
         }
         PARA_COMM_CALL(
               paraComm->send( NULL, 0, ParaBYTE, source, TagAckCompletion )
         );
      }

      if( osStatisticsRacingRampUp )
      {
         *osStatisticsRacingRampUp << termState->toString(paraInitiator);
      }

      if( paraParams->getBoolParamValue(StatisticsToStdout) )
      {
         std::cout << termState->toString(paraInitiator) << std::endl;
      }

      delete termState;
   // }
   // else
   // {
   //    if( paraDetTimer )
   //   {
   //      PARA_COMM_CALL(
   //            paraComm->send( NULL, 0, ParaBYTE, source, TagAckCompletion )
   //      );
   //   }
   // }
   // nTerminated++;      We should not count this, We should always send Term from LC!
   inactivateRacingSolverPool(source);

   if( racingTermination &&
         ( (paraParams->getBoolParamValue(OmitInfeasibleTerminationInRacing) &&
          (!bbParaInitiator->getGlobalBestIncumbentSolution()) ) ||
          (dynamic_cast<UG::BbParaInitiator *>(paraInitiator)->getNSolutions() <
           paraParams->getIntParamValue(OmitTerminationNSolutionsInRacing) ) )
         )
   {
      racingTermination = false;
   }

   if( warmStartNodeTransferring && (!paraNodePool->isEmpty()) )
   {
      // racingTermination = false;
      // nTerminated++;
      if( paraRacingSolverPool->getNumActiveSolvers() == 0 )
      {
         newRacing();
      }
   }
   else
   {
#ifndef _COMM_MPI_WORLD
      if( paraParams->getBoolParamValue(Quiet) && racingTermination )
      {
         if( !paraParams->getBoolParamValue(WaitTerminationOfThreads) )
         {
            // nTerminated = 1;
            terminateAllSolvers();
            delete this;
#ifdef _COMM_PTH
            _exit(0);
#else
            exit(0);
#endif
         }
      }
#endif
   }

   return 0;
}


void
BbParaLoadCoordinator::run(
      ParaTask *paraNode,
      int nRacingSolvers,
      ParaRacingRampUpParamSet **racingRampUpParams
      )
{
   if( !isHeaderPrinted && outputTabularSolvingStatusFlag )
   {
      outputTabularSolvingStatusHeader();            /// should not call virutal function in constructor
   }

   racingRampUpMessageHandler = new MessageHandlerFunctionPointer[nHandlers];

   /** register message handlers */
   for( int i = 0; i < nHandlers; i++ )
   {
      racingRampUpMessageHandler[i] = 0;
   }

   BbMessageHandlerFunctionPointer *bbRacingRampUpMessageHandler = reinterpret_cast<BbMessageHandlerFunctionPointer *>(racingRampUpMessageHandler);

   bbRacingRampUpMessageHandler[TagSolution] = &UG::BbParaLoadCoordinator::processTagSolution;
   bbRacingRampUpMessageHandler[TagSolverState] = &UG::BbParaLoadCoordinator::processRacingRampUpTagSolverState;
   bbRacingRampUpMessageHandler[TagCompletionOfCalculation] = &UG::BbParaLoadCoordinator::processRacingRampUpTagCompletionOfCalculation;
   bbRacingRampUpMessageHandler[TagAnotherNodeRequest] = &UG::BbParaLoadCoordinator::processTagAnotherNodeRequest;
   bbRacingRampUpMessageHandler[TagTerminated] = &UG::BbParaLoadCoordinator::processTagTerminated;
   bbRacingRampUpMessageHandler[TagHardTimeLimit] = &UG::BbParaLoadCoordinator::processTagHardTimeLimit;
   // racingRampUpMessageHandler[TagInitialStat] = &UG::BbParaLoadCoordinator::processTagInitialStat;
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      bbRacingRampUpMessageHandler[TagLbBoundTightenedIndex] = &UG::BbParaLoadCoordinator::processTagLbBoundTightened;
      bbRacingRampUpMessageHandler[TagUbBoundTightenedIndex] = &UG::BbParaLoadCoordinator::processTagUbBoundTightened;
   }
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      bbRacingRampUpMessageHandler[TagToken] = &UG::BbParaLoadCoordinator::processTagToken;
   }

   /** creates racing solver pool */
   paraRacingSolverPool = new BbParaRacingSolverPool(
         1,                // paraSolver origin rank
         paraComm, paraParams, paraTimer, paraDetTimer);

   BbParaRacingSolverPool *bbParaRacingSolverPool = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool);

   /** activate racing solver with root node */
   PARA_COMM_CALL(
         paraNode->bcast(paraComm, 0)
         );
   bbParaRacingSolverPool->activate(dynamic_cast<BbParaNode *>(paraNode));
   lcts.nSent++;
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      int token[2];
      token[0] = 1;
      token[1] = -1;
      PARA_COMM_CALL(
            paraComm->send( token, 2, ParaINT, token[0], TagToken )
      );
   }
   if( logTasksTransferFlag )
   {
      for(int i = 1; i < paraComm->getSize(); i++ )
      {
         writeTransferLogInRacing(i);
      }
   }

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   /** output start racing to log file, if it is necessary */
   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " All Solvers starts racing "
      << bbParaInitiator->convertToExternalValue(
            dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue() )
            << ", size of ParaNodePool = " << paraNodePool->getNumOfNodes()
            << std::endl;
   }
#ifdef _DEBUG_LB
   std::cout << paraTimer->getElapsedTime()
   << " All Solvers starts racing "
   << paraInitiator->convertToExternalValue(
         paraNode->getDualBoundValue() )
         << ", size of ParaNodePool = " << paraNodePool->getNumOfNodes()
         << std::endl;
#endif

   int source;
   int tag;

   for(;;)
   {
      /*******************************************
       *  waiting for any message form anywhere  *
       *******************************************/
      double inIdleTime = paraTimer->getElapsedTime();
      (void)paraComm->probe(&source, &tag);
      lcts.idleTime += ( paraTimer->getElapsedTime() - inIdleTime );
      if( racingRampUpMessageHandler[tag] )
      {
         int status = (this->*racingRampUpMessageHandler[tag])(source, tag);
         if( status )
         {
            std::ostringstream s;
            s << "[ERROR RETURN form Racing Ramp-up Message Hander]:" <<  __FILE__ <<  "] func = "
              << __func__ << ", line = " << __LINE__ << " - "
              << "process tag = " << tag << std::endl;
            abort();
         }
      }
      else
      {
         THROW_LOGICAL_ERROR3( "No racing ramp-up message hander for ", tag, " is not registered" );
      }

#ifdef UG_WITH_UGS
      if( commUgs ) checkAndReadIncumbent();
#endif

      /** output tabular solving status */
      if( outputTabularSolvingStatusFlag && // (!racingTermination) &&
            ( ( ( paraParams->getBoolParamValue(Deterministic) &&
                  paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) ) &&
                  ( ( paraDetTimer->getElapsedTime() - previousTabularOutputTime ) >
               paraParams->getRealParamValue(TabularSolvingStatusInterval) ) ) ||
               ( (!paraParams->getBoolParamValue(Deterministic) ||
                     !paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) )  &&
                  ( ( paraTimer->getElapsedTime() - previousTabularOutputTime ) >
               paraParams->getRealParamValue(TabularSolvingStatusInterval) ) ) ) )
      {
         outputTabularSolvingStatus(' ');
         if( paraParams->getBoolParamValue(Deterministic) )
         {
            if( paraParams->getBoolParamValue(DeterministicTabularSolvingStatus) )
            {
               previousTabularOutputTime = paraDetTimer->getElapsedTime();
            }
            else
            {
               previousTabularOutputTime = paraTimer->getElapsedTime();
            }
         }
         else
         {
            previousTabularOutputTime = paraTimer->getElapsedTime();
         }
      }

      if( hardTimeLimitIsReached || memoryLimitIsReached || givenGapIsReached )
          break;

      if( restartingRacing )
      {
         if( paraRacingSolverPool->getNumActiveSolvers() == 0 )
         {
            if( restartRacing() )
            {
               return;  // solved
            }
         }
         continue;
      }

      switch ( runningPhase )
      {
      case RampUpPhase:
      {
         if( !paraRacingSolverPool )
         {
            warmStartNodeTransferring = false;
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->activate();
            run();
            return;
         }
         if( racingTermination )
         {
            if( paraNodePool->isEmpty()
                  && paraRacingSolverPool->getNumActiveSolvers() == 0 )
            {
               warmStartNodeTransferring = false;
               delete paraRacingSolverPool;
               paraRacingSolverPool = 0;
               run();
               return;
            }
            if( paraNodePool->isEmpty()
                  ||  (warmStartNodeTransferring && paraNodePool->getNumOfNodes() > 0 ) )
            {
               if( !paraSolverPool->isInterruptRequested(source) )
               {
                  sendInterruptRequest();
// std::cout << "sendInterrruptRequest1 13" << std::endl;
               }
            }
            break;
         }

         // if( paraParams->getRealParamValue(TimeLimit) > 0.0 &&
         //       paraTimer->getElapsedTime() > paraParams->getRealParamValue(TimeLimit) )
         if( paraParams->getRealParamValue(TimeLimit) > 0.0 && hardTimeLimitIsReached )
         {
            warmStartNodeTransferring = false;
            // hardTimeLimitIsReached = true;
            // sendInterruptRequest();
            // runningPhase = TerminationPhase;     waits until paraRacingSolverPool becomes empty
            break;
         }

         if( paraParams->getRealParamValue(FinalCheckpointGeneratingTime) > 0.0 && hardTimeLimitIsReached )
         //      paraTimer->getElapsedTime() > paraParams->getRealParamValue(FinalCheckpointGeneratingTime) )
         {
            warmStartNodeTransferring = false;
            // hardTimeLimitIsReached = true;
            std::cout << "** Program is still in racing stage. FinalCheckpointGeneratingTime is sppecifid, but the checkpoint files would not be generated." << std::endl;
            // sendInterruptRequest();
            // runningPhase = TerminationPhase;     waits until paraRacingSolverPool becomes empty
            break;
         }

         if( bbParaRacingSolverPool->isWinnerDecided(
               ( bbParaInitiator->getGlobalBestIncumbentSolution() &&
                 EPSLT(bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue(),DBL_MAX, DEFAULT_NUM_EPSILON ) ) ) )
         {
            warmStartNodeTransferring = false;
            if( paraParams->getBoolParamValue(RestartRacing) &&
                  primalUpdated
                  )
            {
               if( !restartingRacing )
               {
                  for(int i = 1; i < paraComm->getSize(); i++)
                  {
                     PARA_COMM_CALL(
                           paraComm->send( NULL, 0, ParaBYTE, i, TagTerminateSolvingToRestart )
                     );
                  }
                  // primalUpdated = false;
                  restartingRacing = true;
               }
            }
            else
            {
               racingWinner = paraRacingSolverPool->getWinner();
               warmStartNodeTransferring = false;
               racingTermination = false;
               assert( racingWinner >0 );
               int numNodesLeft = bbParaRacingSolverPool->getNumNodesLeft(racingWinner);
               BbParaSolverPoolForMinimization *bbParaSolverPool = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool);
               bbParaSolverPool->activateSolver(
                     racingWinner, bbParaRacingSolverPool->extractNode(), numNodesLeft );
               bbParaRacingSolverPool->inactivateSolver(racingWinner);

               // assert(paraRacingSolverPool->getNumActiveSolvers() >= 0);
               PARA_COMM_CALL(
                     paraComm->send( NULL, 0, ParaBYTE, racingWinner, TagWinner )
               );

               if( numNodesLeft >
                    2.0 * paraParams->getIntParamValue(StopRacingNumberOfNodesLeft)*paraParams->getRealParamValue(StopRacingNumberOfNodesLeftMultiplier)
                    ||
                    numNodesLeft <= ( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(ProhibitCollectOnceMultiplier) )
                    )
               {
                  paraParams->setBoolParamValue(CollectOnce,false);
                  paraParams->setBoolParamValue(MergeNodesAtRestart,false);
                  paraParams->setBoolParamValue(RacingStatBranching,false);
                  paraParams->setIntParamValue(RampUpPhaseProcess, 1);
                  winnerSolverNodesCollected = true;
                  std::cout << "Warning: Ramp-Up Phase Process is switched to 1. CollectOnce, MergeNodesAtRestart and RacingStatBranching are switched to FALSE." << std::endl;
                  std::cout << "You should check the following parameter values: StopRacingNumberOfNodesLeft, StopRacingNumberOfNodesLeftMultiplier, ProhibitCollectOnceMultiplier" << std::endl;
               }

               if( numNodesLeft > (signed)paraSolverPool->getNSolvers()
                     ||
                     numNodesLeft > (signed)( paraSolverPool->getNSolvers() * paraParams->getRealParamValue(ProhibitCollectOnceMultiplier) )
                     )
               {
                  if( paraParams->getBoolParamValue(CollectOnce) )
                  {
                     int nCollect = paraParams->getIntParamValue(NCollectOnce);
                     if( nCollect == 0 )
                     {
                        nCollect = ( paraSolverPool->getNSolvers() * 5 );
                     }
                     PARA_COMM_CALL(
                           paraComm->send( &nCollect, 1, ParaINT, racingWinner, TagCollectAllNodes )
                     );
                  }
                  if( paraParams->getIntParamValue(RampUpPhaseProcess) == 2 )
                  {
                     merging = true;
                     if( nodesMerger ) delete nodesMerger;
                     nodesMerger = new BbParaNodesMerger(dynamic_cast<BbParaInstance *>(bbParaInitiator->getParaInstance())->getVarIndexRange(),
                           nBoundChangesOfBestNode,paraTimer,dynamic_cast<BbParaInstance *>(bbParaInitiator->getParaInstance()),paraParams);
                  }
               }
               else
               {
                  winnerSolverNodesCollected = true;   // do not wait until all nodes are collected
               }
               racingWinnerParams = racingRampUpParams[racingWinner - 1];
               // racingWinnerParams->setWinnerRank(racingWinner);
               racingRampUpParams[racingWinner - 1] = 0;
               for(int i = 1; i < paraComm->getSize(); i++)
               {
                  if( racingRampUpParams[i - 1] )
                  {
                     PARA_COMM_CALL(
                           racingWinnerParams->send(paraComm, i)
                           );
                  }
                  int noKeep = 0;
                  PARA_COMM_CALL(
                        paraComm->send( &noKeep, 1, ParaINT,i, TagKeepRacing)
                        );
               }
               /** output winner to log file, if it is necessary */
               if( logSolvingStatusFlag )
               {
                  *osLogSolvingStatus << paraTimer->getElapsedTime()
                  << " S." << racingWinner << " is the racing winner!"
                  << " Selected strategy " << racingWinnerParams->getStrategy()
                  << "." << std::endl;
               }
   #ifdef _DEBUG_LB
               std::cout << paraTimer->getElapsedTime()
               << " S." << racingWinner << " is the racing winner!"
               << " Selected strategy " << racingWinnerParams->getStrategy()
               << "." << std::endl;
   #endif
               if( outputTabularSolvingStatusFlag )
               {
                  *osTabularSolvingStatus <<
                        "Racing ramp-up finished after " <<
                        paraTimer->getElapsedTime() << " seconds." <<
                        " Selected strategy " << racingWinnerParams->getStrategy() <<
                        "." << std::endl;
               }
               // runningPhase = NormalRunningPhase;
               // Keep running as RampUpPhase, but in the run() switching into RampUpPhase in normal running mode
               // delete paraRacingSolverPool;
               run();
               return;
            }
         }

         if( warmStartNodeTransferring
               && paraRacingSolverPool && paraRacingSolverPool->getNumActiveSolvers() == 0
               && paraNodePool->getNumOfNodes() > 0
               )
         {
            newRacing();
         }
         break;
      }
      default:
      {
         THROW_LOGICAL_ERROR2( "Undefined running phase: ", static_cast<int>(runningPhase) );
      }
      }
   }
   return;
}

void
BbParaLoadCoordinator::sendRetryRampUpToAllSolvers(
      )
{
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, i, TagRetryRampUp )
      );
   }
}

void
BbParaLoadCoordinator::sendInterruptRequest(
      )
{
   int exitSolverRequest = 0;    // do nothing
   if( ( paraParams->getBoolParamValue(UG::EnhancedFinalCheckpoint) ||
         paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 ) &&
         (!paraRacingSolverPool) &&
         ( paraParams->getIntParamValue(UG::FinalCheckpointNSolvers) < 0 ||
               ( paraParams->getIntParamValue(UG::FinalCheckpointNSolvers) > 0 &&
                     (signed)nCollectedSolvers < paraParams->getIntParamValue(UG::FinalCheckpointNSolvers) )
         ) )
   {
      if( !interruptIsRequested && outputTabularSolvingStatusFlag )
      {
         *osTabularSolvingStatus <<
                "Start collecting the final check-point data after " <<
                paraTimer->getElapsedTime() << " seconds. " << std::endl;
      }

      if( paraSolverPool->getNumActiveSolvers() > 0 )
      {
         int bestSolverRank = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getGoodSolverSolvingEssentialNode();
         if( bestSolverRank > 0 )   // bestSolverRank exits
         {
            BbParaNode *solvingNode = dynamic_cast<BbParaNode *>(paraSolverPool->getCurrentTask(bestSolverRank));
            assert( !solvingNode->getAncestor() );
            solvingNode->collectsNodes();
            exitSolverRequest = 1;  // collect all nodes
            PARA_COMM_CALL(
                  paraComm->send( &exitSolverRequest, 1, ParaINT, bestSolverRank, TagInterruptRequest )
            );
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << bestSolverRank << " TagInterruptRequest with collecting is sent"
               << std::endl;
            }
         }
         else
         {
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << bestSolverRank << " TagInterruptRequest with collecting could not be sent (No best solvers)"
               << std::endl;
            }
         }
      }
      else
      {
         if( logSolvingStatusFlag )
         {
            *osLogSolvingStatus << paraTimer->getElapsedTime()
            << " S." << 0 << " TagInterruptRequest with collecting could not be sent (No active solvers)"
            << std::endl;
         }
      }
   }
   else
   {
      if( paraParams->getRealParamValue(UG::FinalCheckpointGeneratingTime) > 0.0 &&
         (!paraRacingSolverPool) )
      {
         for( int i = 1; i < paraComm->getSize(); i++ )
         {
            if( paraSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
            {
               PARA_COMM_CALL(
                     paraComm->send( &exitSolverRequest, 1, ParaINT, i, TagInterruptRequest )
               );
               paraSolverPool->interruptRequested(i);
            }
            else
            {
               PARA_COMM_CALL(
                     paraComm->send( NULL, 0, ParaBYTE, i, TagTerminateRequest )
               );
               paraSolverPool->terminateRequested(i);
               if( paraParams->getBoolParamValue(Deterministic) )
               {
                  int token[2];
                  token[0] = i;
                  token[1] = -2;
                  PARA_COMM_CALL(
                        paraComm->send( token, 2, ParaINT, token[0], TagToken )
                  );
               }
               // nTerminated++;
            }
         }
// std::cout << "TagTerminateRequest 4" << std::endl;
      }
      else
      {
         // normal TimeLimit
         if( interruptIsRequested ) return;
         if( paraSolverPool->getNumActiveSolvers() > 0 )
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               if( paraSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
               {
                  PARA_COMM_CALL(
                        paraComm->send( &exitSolverRequest, 1, ParaINT, i, TagInterruptRequest )
                  );
                  paraSolverPool->interruptRequested(i);
               }
               else
               {
                  PARA_COMM_CALL(
                        paraComm->send( NULL, 0, ParaBYTE, i, TagTerminateRequest )
                  );
                  paraSolverPool->terminateRequested(i);
                  if( paraParams->getBoolParamValue(Deterministic) )
                  {
                     int token[2];
                     token[0] = i;
                     token[1] = -2;
                     PARA_COMM_CALL(
                           paraComm->send( token, 2, ParaINT, token[0], TagToken )
                     );
                  }
               }
            }
//    hardTimeLimitIsReached = true;
// std::cout << "TagTerminateRequest 5" << std::endl;
         }
         if( paraRacingSolverPool && paraRacingSolverPool->getNumActiveSolvers() > 0 )
         {
            for( int i = 1; i < paraComm->getSize(); i++ )
            {
               if( paraRacingSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
               {
                  PARA_COMM_CALL(
                        paraComm->send( &exitSolverRequest, 1, ParaINT, i, TagInterruptRequest )
                  );
                  paraSolverPool->interruptRequested(i);
               }
               else
               {
                  PARA_COMM_CALL(
                        paraComm->send( NULL, 0, ParaBYTE, i, TagTerminateRequest )
                  );
                  paraSolverPool->terminateRequested(i);
                  if( paraParams->getBoolParamValue(Deterministic) )
                  {
                     int token[2];
                     token[0] = i;
                     token[1] = -2;
                     PARA_COMM_CALL(
                           paraComm->send( token, 2, ParaINT, token[0], TagToken )
                     );
                  }
               }
            }
//	    hardTimeLimitIsReached = true;
// std::cout << "TagTerminateRequest 6" << std::endl;
// std::cout << "racingTermination = " << racingTermination << std::endl;
         }
      }
   }

   interruptIsRequested = true;
}

bool
BbParaLoadCoordinator::updateSolution(
      BbParaSolution *sol
      )
{

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   if( !bbParaInitiator->getGlobalBestIncumbentSolution() )
      return bbParaInitiator->tryToSetIncumbentSolution(dynamic_cast<BbParaSolution *>(sol->clone(paraComm)),false);
   if( sol->getObjectiveFunctionValue()
         < bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue() )
      return bbParaInitiator->tryToSetIncumbentSolution(dynamic_cast<BbParaSolution *>(sol->clone(paraComm)),false);
   else
      return false;
}

void
BbParaLoadCoordinator::sendIncumbentValue(
      int receivedRank
      )
{

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   double globalBestIncumbentValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
   if( !paraParams->getBoolParamValue(NoUpperBoundTransferInRacing) || !isRacingStage() )
   {
      for( int i = 1; i < paraComm->getSize(); i++ )
      {
         if( i !=  receivedRank )
         {
            PARA_COMM_CALL(
                  paraComm->send( &globalBestIncumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
            );
         }

      }
   }
   lcts.nDeletedInLc += paraNodePool->removeBoundedNodes(globalBestIncumbentValue);
   /*
   if( paraParams->getIntParamValue(RampUpPhaseProcess) == 3 )
   {
      lcts.nDeletedInLc += dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->removeBoundedNodes(globalBestIncumbentValue);
   }
   */
}

void
BbParaLoadCoordinator::sendCutOffValue(
      int receivedRank
      )
{

   BbParaInitiator *bbParaInitiator = dynamic_cast<BbParaInitiator *>(paraInitiator);

   double globalBestCutOffValue = bbParaInitiator->getGlobalBestIncumbentSolution()->getCutOffValue();
   if( !paraParams->getBoolParamValue(NoUpperBoundTransferInRacing) || !isRacingStage() )
   {
      for( int i = 1; i < paraComm->getSize(); i++ )
      {
         if( i !=  receivedRank )
         {
            PARA_COMM_CALL(
                  paraComm->send( &globalBestCutOffValue, 1, ParaDOUBLE, i, TagCutOffValue )
            );
         }

      }
   }
   lcts.nDeletedInLc += paraNodePool->removeBoundedNodes(globalBestCutOffValue);
}

void
BbParaLoadCoordinator::inactivateRacingSolverPool(
      int rank
      )
{
   nSolvedInInterruptedRacingSolvers = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesSolvedInBestSolver();
   nTasksLeftInInterruptedRacingSolvers = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->getNnodesLeftInBestSolver();
   if( paraRacingSolverPool->isSolverActive(rank) )   // if rank is the winner, it should be inactive
   {
      dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->inactivateSolver(rank);
   }
   if( paraSolverPool->isSolverActive(rank) )
   {
      /*
       * special timining problem
       *
       * 1113.58 S.4 I.SOL 0
       * 1113.58 S.3 is the racing winner! Selected strategy 2.
       * 1113.58 S.4 >(TERMINATED_IN_RACING_STAGE)
       *
       */
      if( dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->isSolverInCollectingMode(rank) )
      {
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(rank, -1, paraNodePool);
         // reschedule collecting mode
         if( !paraNodePoolBufferToRestart )
         {
            double tempTime = dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->getSwichOutTime();
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchOutCollectingMode();
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->setSwichOutTime(tempTime);
            dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->switchInCollectingMode(paraNodePool);
         }
      }
      else
      {
         dynamic_cast<BbParaSolverPoolForMinimization *>(paraSolverPool)->inactivateSolver(rank, -1, paraNodePool);
      }
   }

   if ( (!interruptIsRequested) &&
         (!restartingRacing )&&
         paraRacingSolverPool->getNumActiveSolvers() == 0 )
   {
      /** if the computation is interrupted,
       *  paraRacingSolverPool is needed to output statistics */
      if( !warmStartNodeTransferring || ( warmStartNodeTransferring && paraNodePool->isEmpty() ) )
      {
         delete paraRacingSolverPool;
         paraRacingSolverPool = 0;
      }
   }

}

void
BbParaLoadCoordinator::writeSubtreeInfo(
      int source,
      ParaCalculationState *calcState
      )
{
   if( logSubtreeInfoFlag )
   {
     ParaTask *node = paraSolverPool->getCurrentTask(source);
     *osLogSubtreeInfo  << paraTimer->getElapsedTime()
           << ", "
           << source
           << ", "
           << node->toSimpleString()
           << ", "
           << calcState->toSimpleString()
           << std::endl;
   }
}

int
BbParaLoadCoordinator::restartRacing(
      )
{
   // RESTART RACING MAY NOT WORK WITH DETERMINSTIC MODE: NOT DEBUGGED

   if( paraInitiator->reInit(nRestartedRacing) )
   {
      restartingRacing = false;
      return 1;
   }

   ParaInstance *paraInstance = paraInitiator->getParaInstance();
   paraInstance->bcast(paraComm, 0, paraParams->getIntParamValue(InstanceTransferMethod));

   BbParaNode *rootNode = dynamic_cast<BbParaNode *>(dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->extractNode());
   rootNode->setDualBoundValue(-DBL_MAX);
   rootNode->setInitialDualBoundValue(-DBL_MAX);
   rootNode->setEstimatedValue(-DBL_MAX);

   /** recreate racing solver pool */
   delete paraRacingSolverPool;
   paraRacingSolverPool = new BbParaRacingSolverPool(
         1,                // paraSolver origin rank
         paraComm, paraParams, paraTimer, paraDetTimer);

   /** activate racing solver with root node */
   PARA_COMM_CALL(
         rootNode->bcast(paraComm, 0)
         );
   dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool)->activate(rootNode);
   lcts.nSent++;

   nRestartedRacing++;

   if( osStatisticsRacingRampUp )
   {
      *osStatisticsRacingRampUp << "##################################" << std::endl;
      *osStatisticsRacingRampUp << "### Racing restarted " << nRestartedRacing << " times" << std::endl;
      *osStatisticsRacingRampUp << "##################################" << std::endl;
   }

   if( !paraParams->getBoolParamValue(Quiet) )
   {
      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " Racing Ramp-up restarted." << std::endl;
      }
   }

   if( outputTabularSolvingStatusFlag )
   {
      *osTabularSolvingStatus << "Racing Ramp-up restarted." << std::endl;
   }

   primalUpdated = false;
   restartingRacing = false;

   lcts.globalBestDualBoundValue = -DBL_MAX;
   lcts.externalGlobalBestDualBoundValue = -DBL_MAX;
   interruptIsRequested = false;

   return 0;
}

void
BbParaLoadCoordinator::newRacing(
      )
{
   racingTermination = false;
   // assert( nTerminated == paraRacingSolverPool->getNumInactiveSolvers() );
   BbParaNode *paraNode = paraNodePool->extractNode();
   if( paraNodePool->isEmpty() )
   {
      for( int i = 1; i < paraComm->getSize(); i++ )
      {
         int noKeep = 0;
         PARA_COMM_CALL(
               paraComm->send( &noKeep, 1, ParaINT,i, TagKeepRacing)
               );
         PARA_COMM_CALL(
               paraNode->send(paraComm, i)
               );
      }
   }
   else
   {
      for( int i = 1; i < paraComm->getSize(); i++ )
      {
         int keep = 1;
         PARA_COMM_CALL(
               paraComm->send( &keep, 1, ParaINT, i, TagKeepRacing)
               );
//               PARA_COMM_CALL(
//                     paraComm->send( NULL, 0, ParaBYTE,i, TagTerminateSolvingToRestart)
//                     );
         PARA_COMM_CALL(
               paraNode->send(paraComm, i)
               );
      }
   }

   /** activate racing solver with root node */
   BbParaRacingSolverPool *bbParaRacingSolverPool = dynamic_cast<BbParaRacingSolverPool *>(paraRacingSolverPool);
   bbParaRacingSolverPool->reset();
   bbParaRacingSolverPool->activate(paraNode);
   lcts.nSent++;
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      int token[2];
      token[0] = 1;
      token[1] = -1;
      PARA_COMM_CALL(
            paraComm->send( token, 2, ParaINT, token[0], TagToken )
      );
   }
   if( logTasksTransferFlag )
   {
      for(int i = 1; i < paraComm->getSize(); i++ )
      {
         writeTransferLogInRacing(i);
      }
   }

   /** output start racing to log file, if it is necessary */
   if( logSolvingStatusFlag )
   {
      *osLogSolvingStatus << paraTimer->getElapsedTime()
      << " All Solvers starts racing "
      << dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(
            dynamic_cast<BbParaNode *>(paraNode)->getDualBoundValue() )
            << ", size of ParaNodePool = " << paraNodePool->getNumOfNodes()
            << std::endl;
   }
#ifdef _DEBUG_LB
   std::cout << paraTimer->getElapsedTime()
   << " All Solvers starts racing "
   << paraInitiator->convertToExternalValue(
         paraNode->getDualBoundValue() )
         << ", size of ParaNodePool = " << paraNodePool->getNumOfNodes()
         << std::endl;
#endif
   // nTerminated = 0;
   interruptIsRequested = false;
}

void
BbParaLoadCoordinator::changeSearchStrategyOfAllSolversToBestBoundSearch(
      )
{
   /* Not implemented yet
   int bestBoundSearch = 1;
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
         paraComm->send( &bestBoundSearch, 1, ParaINT, i, TagChangeSearchStrategy )
      );
   }
   */
}

void
BbParaLoadCoordinator::changeSearchStrategyOfAllSolversToOriginalSearch(
   )
{
   /* Not implemented yet
   int originalSearch = 0;
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
         paraComm->send( &originalSearch, 1, ParaINT, i, TagChangeSearchStrategy )
      );
   }
   */
}


#ifdef UG_WITH_UGS
int
BbParaLoadCoordinator::checkAndReadIncumbent(
      )
{
   int source = -1;
   int tag = TagAny;

   assert(commUgs);

   while( commUgs->iProbe(&source, &tag) )
   {
      if( source == 0 && tag == UGS::TagUpdateIncumbent )
      {
         if( paraInitiator->readUgsIncumbentSolution(commUgs, source) )
         {
            double globalBestIncumbentValue = paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue();
            if( !paraParams->getBoolParamValue(NoUpperBoundTransferInRacing) || !isRacingStage() )
            {
               if( paraInitiator->canGenerateSpecialCutOffValue() )
               {
                  for( int i = 1; i < paraComm->getSize(); i++ )
                  {
                     sendCutOffValue(i);
                     PARA_COMM_CALL(
                           paraComm->send( &globalBestIncumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
                     );
                  }
               }
               else
               {
                  for( int i = 1; i < paraComm->getSize(); i++ )
                  {
                     PARA_COMM_CALL(
                           paraComm->send( &globalBestIncumbentValue, 1, ParaDOUBLE, i, TagIncumbentValue )
                     );
                  }
               }

            }
            lcts.nDeletedInLc += paraNodePool->removeBoundedNodes(globalBestIncumbentValue);

            primalUpdated = true;
            allCompInfeasibleAfterSolution = true;
            if( logSolvingStatusFlag )
            {
               *osLogSolvingStatus << paraTimer->getElapsedTime()
               << " S." << source << " I.SOL "
               << paraInitiator->convertToExternalValue(
                     paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
                     ) << std::endl;
            }
#ifdef _DEBUG_LB
            std::cout << paraTimer->getElapsedTime()
            << " S." << source << " I.SOL "
            << paraInitiator->convertToExternalValue(
                  paraInitiator->getGlobalBestIncumbentSolution()->getObjectiveFunctionValue()
                  ) << std::endl;
#endif
            /** output tabular solving status */
            if( outputTabularSolvingStatusFlag )
            {
               outputTabularSolvingStatus('*');
            }
#ifdef UG_WITH_ZLIB
            /* Do not have to remove ParaNodes from NodePool. It is checked and removed before sending them */
            /** save incumbent solution */
            char solutionFileNameTemp[256];
            char solutionFileName[256];
            if( paraParams->getBoolParamValue(Checkpoint) && paraComm->getRank() == 0  )
            {
               sprintf(solutionFileNameTemp,"%s%s_after_checkpointing_solution_t.gz",
                     paraParams->getStringParamValue(CheckpointFilePath),
                     paraInitiator->getParaInstance()->getProbName() );
               paraInitiator->writeCheckpointSolution(std::string(solutionFileNameTemp));
               sprintf(solutionFileName,"%s%s_after_checkpointing_solution.gz",
                     paraParams->getStringParamValue(CheckpointFilePath),
                     paraInitiator->getParaInstance()->getProbName() );
               if ( rename(solutionFileNameTemp, solutionFileName) )
               {
                  std::cout << "after checkpointing solution file cannot be renamed: errno = " << strerror(errno) << std::endl;
                  exit(1);
               }
            }
#endif
         }
      }
      else if ( source == 0 && tag == TagTerminated )
      {
         return 1;
      }
      else
      {
         THROW_LOGICAL_ERROR5("Invalid Tag = ", tag, ", from source = ", source, " received.");
      }
   }
   return 0;

}
#endif
