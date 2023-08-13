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
#include <cstdlib>
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
#include "gzstream.h"
#endif

#include "paraLoadCoordinator.h"
#include "paraInitialStat.h"

using namespace UG;

ParaLoadCoordinator::ParaLoadCoordinator(
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
      : nHandlers(inNHandlers),
        messageHandler(0),
        racingRampUpMessageHandler(0),
        globalSubtreeIdGen(0),
        paraParams(inParaParamSet),
        paraInitiator(inParaInitiator),
        racingSolversExist(inRacingSolversExist),
        restarted(false),
        runningPhase(RampUpPhase),
        computationIsInterrupted(false),
        interruptedFromControlTerminal(false),
        hardTimeLimitIsReached(false),
        memoryLimitIsReached(false),
        interruptIsRequested(false),
        paraSolverPool(0),
        paraRacingSolverPool(0),
        nSolvedInInterruptedRacingSolvers(-1),
        nTasksLeftInInterruptedRacingSolvers(-1),
        previousCheckpointTime(0.0),
        eps(MINEPSILON),
        racingWinner(-1),
        racingWinnerParams(0),
        racingTermination(false),
        nSolvedRacingTermination(0),
        nTerminated(0),
        paraTimer(inParaTimer),
        paraDetTimer(inParaDetTimer),
        osLogSolvingStatus(0),
        osLogTasksTransfer(0),
        osStatisticsFinalRun(0),
        osStatisticsRacingRampUp(0),
        pendingSolution(0),
        terminationIssued(false)
{
#ifdef UG_WITH_UGS
   commUgs = inCommUgs;
#endif
   paraComm = inComm;

   ///
   ///  register message handlers
   ///
   messageHandler = new MessageHandlerFunctionPointer[nHandlers];
   for( int i = 0; i < nHandlers; i++ )
   {
      messageHandler[i] = 0;
   }
   messageHandler[TagTask] = &UG::ParaLoadCoordinator::processTagTask;
   messageHandler[TagSolution] = &UG::ParaLoadCoordinator::processTagSolution;
   messageHandler[TagSolverState] = &UG::ParaLoadCoordinator::processTagSolverState;
   messageHandler[TagCompletionOfCalculation] = &UG::ParaLoadCoordinator::processTagCompletionOfCalculation;
   messageHandler[TagTerminated] = &UG::ParaLoadCoordinator::processTagTerminated;
   messageHandler[TagHardTimeLimit] = &UG::ParaLoadCoordinator::processTagHardTimeLimit;
   if( paraParams->getBoolParamValue(Deterministic) )
   {
      messageHandler[TagToken] = &UG::ParaLoadCoordinator::processTagToken;
   }

   ///
   /// set up status log and transfer log
   ///
   logSolvingStatusFlag = paraParams->getBoolParamValue(LogSolvingStatus);
   if( logSolvingStatusFlag )
   {
      std::ostringstream s;
#ifdef UG_WITH_UGS
      if( commUgs )
      {
         s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << commUgs->getMySolverName() << "_"
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".status";
      }
      else
      {
         s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".status";
      }
#else
      s << paraParams->getStringParamValue(LogSolvingStatusFilePath)
      << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".status";
#endif
      ofsLogSolvingStatus.open(s.str().c_str(), std::ios::app );
      if( !ofsLogSolvingStatus )
      {
         std::cout << "Solving status log file cannot open : file name = " << s.str() << std::endl;
         exit(1);
      }
      osLogSolvingStatus = &ofsLogSolvingStatus;
   }

   logTasksTransferFlag = paraParams->getBoolParamValue(LogTasksTransfer);
   if( logTasksTransferFlag )
   {
      std::ostringstream s;
#ifdef UG_WITH_UGS
      if( commUgs )
      {
         s << paraParams->getStringParamValue(LogTasksTransferFilePath)
         << commUgs->getMySolverName() << "_"
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".transfer";
      }
      else
      {
         s << paraParams->getStringParamValue(LogTasksTransferFilePath)
         << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".transfer";
      }
#else
      s << paraParams->getStringParamValue(LogTasksTransferFilePath)
      << paraInitiator->getParaInstance()->getProbName() << "_LC" << paraComm->getRank() << ".transfer";
#endif
      ofsLogTasksTransfer.open(s.str().c_str(), std::ios::app);
      if( !ofsLogTasksTransfer )
      {
         std::cout << "Task transfer log file cannot open : file name = " << s.str() << std::endl;
         exit(1);
      }
      osLogTasksTransfer = &ofsLogTasksTransfer;
   }

   if( !paraParams->getBoolParamValue(Quiet) )
   {
      //
      // open statistic files
      //
      std::ostringstream ssfr;
#ifdef UG_WITH_UGS
      if( commUgs )
      {
         ssfr << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << commUgs->getMySolverName() << "_"
         << paraInitiator->getParaInstance()->getProbName() << "_statistics_final_LC" << paraComm->getRank();
      }
      else
      {
         ssfr << paraParams->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName() << "_statistics_final_LC" << paraComm->getRank();
      }
#else
      ssfr << paraParams->getStringParamValue(LogSolvingStatusFilePath)
      << paraInitiator->getParaInstance()->getProbName() << "_statistics_final_LC" << paraComm->getRank();
#endif
      ofsStatisticsFinalRun.open(ssfr.str().c_str(), std::ios::app);
      if( !ofsStatisticsFinalRun )
      {
         std::cout << "Statistics file for final run cannot open : file name = " << ssfr.str() << std::endl;
         exit(1);
      }
      osStatisticsFinalRun = &ofsStatisticsFinalRun;

//      if( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ||
//            paraParams->getIntParamValue(RampUpPhaseProcess) == 2
//            )  /** racing ramp-up */
//      {
//         std::ostringstream ssrru;
//#ifdef UG_WITH_UGS
//         if( commUgs )
//         {
//            ssrru << paraParams->getStringParamValue(LogSolvingStatusFilePath)
//            << commUgs->getMySolverName() << "_"
//            << paraInitiator->getParaInstance()->getProbName() << "_statistics_racing_LC" << paraComm->getRank();
//         }
//         else
//         {
//            ssrru << paraParams->getStringParamValue(LogSolvingStatusFilePath)
//            << paraInitiator->getParaInstance()->getProbName() << "_statistics_racing_LC" << paraComm->getRank();
//         }
//#else
//         ssrru << paraParams->getStringParamValue(LogSolvingStatusFilePath)
//         << paraInitiator->getParaInstance()->getProbName() << "_statistics_racing_LC" << paraComm->getRank();
//#endif
//         ofsStatisticsRacingRampUp.open(ssrru.str().c_str(), std::ios::app);
//         if( !ofsStatisticsRacingRampUp )
//         {
//            std::cout << "Statistics file for racing ramp-up cannot open : file name = " << ssrru.str() << std::endl;
//            exit(1);
//         }
//         osStatisticsRacingRampUp = &ofsStatisticsRacingRampUp;
//      }
   }

   eps = paraInitiator->getEpsilon();

   lastCheckpointTimeStr[0] = ' ';
   lastCheckpointTimeStr[1] = '\0';

   if( paraParams->getBoolParamValue(Deterministic) )
   {
      assert(paraDetTimer);
   }

}

int
ParaLoadCoordinator::processTagTerminated(
      int source,
      int tag
      )
{
#ifdef _COMM_CPP11
   std::lock_guard<std::mutex> lock(routineMutex);
#endif

   ParaSolverTerminationState *paraSolverTerminationState = paraComm->createParaSolverTerminationState();
   paraSolverTerminationState->receive(paraComm, source, tag);

// std::cout << "TagTerminated received from " << source << ", num active solvers = " << paraSolverPool->getNumActiveSolvers() << std::endl;

   if( paraDetTimer )
   {
      if( paraDetTimer->getElapsedTime() < paraSolverTerminationState->getDeterministicTime() )
      {
         paraDetTimer->update( paraSolverTerminationState->getDeterministicTime() - paraDetTimer->getElapsedTime() );
      }
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, source, TagAckCompletion )
      );
   }

   if( osStatisticsFinalRun )
   {
      *osStatisticsFinalRun << paraSolverTerminationState->toString(paraInitiator);
      osStatisticsFinalRun->flush();
   }
   if( paraParams->getBoolParamValue(StatisticsToStdout) )
   {
      std::cout << paraSolverTerminationState->toString(paraInitiator) << std::endl;
   }

   if( (!racingTermination) && paraSolverTerminationState->getInterruptedMode() == 1 )
   {
      computationIsInterrupted = true;
   }

   paraSolverPool->terminated(source);
   nTerminated++;

   delete paraSolverTerminationState;

   return 0;
}

int
ParaLoadCoordinator::processTagHardTimeLimit(
      int source,
      int tag
      )
{
   PARA_COMM_CALL(
         paraComm->receive( NULL, 0, ParaBYTE, source, TagHardTimeLimit)
         );
   hardTimeLimitIsReached = true;
   return 0;
}

int
ParaLoadCoordinator::processTagToken(
      int source,
      int tag
      )
{

   int token[2];
   PARA_COMM_CALL(
         paraComm->receive( token, 2, ParaINT, source, TagToken)
         );
   if( !paraSolverPool->isTerminated(token[0]) )
   {
      PARA_COMM_CALL(
            paraComm->send( token, 2, ParaINT, token[0], TagToken )
      );
   }
   else
   {
      int startRank = token[0];
      token[0] = ( token[0] % (paraComm->getSize() - 1) )  + 1;
      while( paraSolverPool->isTerminated(token[0]) && token[0] != startRank )
      {
         token[0] = ( token[0] % (paraComm->getSize() - 1) )  + 1;
      }
      if( !paraSolverPool->isTerminated(token[0]) )
      {
         PARA_COMM_CALL(
               paraComm->send( token, 2, ParaINT, token[0], TagToken )
         );
      }
   }

   paraComm->setToken(0, token);    // for debug

   return 0;
}

void
ParaLoadCoordinator::sendRampUpToAllSolvers(
      )
{
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, i, TagRampUp )
      );
   }
}

void
ParaLoadCoordinator::terminateAllSolvers(
      )
{
   terminationIssued = true;
   int exitSolverRequest = 0;    // do nothing
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      if( paraSolverPool->isSolverActive(i) && !paraSolverPool->isInterruptRequested(i) )
      {
         PARA_COMM_CALL(
               paraComm->send( &exitSolverRequest, 1, ParaINT, i, TagInterruptRequest )
         );
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

void
ParaLoadCoordinator::writeTransferLog(
      int rank
      )
{
   // output comp infomation to tree log file
   if( logTasksTransferFlag )
   {
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] ParaTask was sent " << (paraSolverPool->getCurrentTask(rank))->toString() << std::endl;
   }
}

void
ParaLoadCoordinator::writeTransferLog(
      int rank,
      ParaCalculationState *state
      )
{
   // output comp infomation to tree log file
   if( logTasksTransferFlag )
   {
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] Solved " << (paraSolverPool->getCurrentTask(rank))->toString() << std::endl;
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] " << state->toString() << std::endl;
   }
}

void
ParaLoadCoordinator::writeTransferLogInRacing(
      int rank
      )
{
   // output comp infomation to tree log file
   if( logTasksTransferFlag )
   {
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] ParaTask was sent " << (paraRacingSolverPool->getCurrentTask(rank))->toString() << std::endl;
   }
}

void
ParaLoadCoordinator::writeTransferLogInRacing(
      int rank,
      ParaCalculationState *state
      )
{
   // output comp infomation to tree log file
   if( logTasksTransferFlag )
   {
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] Solved " << (paraRacingSolverPool->getCurrentTask(rank))->toString() << std::endl;
      *osLogTasksTransfer << "[Solver-ID: " << rank
      << "] " << state->toString() << std::endl;
   }
}

void
ParaLoadCoordinator::sendTagToAllSolvers(
      const int tag
      )
{
   for( int i = 1; i < paraComm->getSize(); i++ )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, i, tag )
      );
   }
}
