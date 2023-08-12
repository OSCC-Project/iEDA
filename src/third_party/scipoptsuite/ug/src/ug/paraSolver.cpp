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
#include <cstring>
#include <algorithm>
#include "paraComm.h"
#include "paraTask.h"
#include "paraInstance.h"
#include "paraSolver.h"
#include "paraSolution.h"
#include "paraSolverTerminationState.h"
#include "paraCalculationState.h"
#include "paraSolverState.h"
#ifdef _COMM_PTH
#include "paraCommPth.h"
#endif
#ifdef _COMM_CPP11
#include "paraCommCPP11.h"
#endif

using namespace UG;

ParaSolver::ParaSolver(
      int argc,
      char **argv,
      int  inNHandlers,
      ParaComm     *comm,
      ParaParamSet *inParaParamSet,
      ParaInstance *inParaInstance,
      ParaDeterministicTimer *inParaDetTimer
      )
      : nHandlers(inNHandlers),
        messageHandler(0),
        notificationIdGenerator(0),
        paraComm(comm),
        paraParams(inParaParamSet),
        racingParams(0),
        winnerRacingParams(0),
        paraDetTimer(inParaDetTimer),
        globalBestIncumbentValue(DBL_MAX),
        globalBestIncumbentSolution(0),
        localIncumbentSolution(0),
        pendingSolution(0),
        pendingIncumbentValue(DBL_MAX),
        paraInstance(inParaInstance),
        currentTask(0),
        newTask(0),
        terminationMode(NoTerminationMode),
        warmStarted(false),
        rampUp(false),
        racingInterruptIsRequested(false),
        racingIsInterrupted(false),
        racingWinner(false),
        waitingSpecificMessage(false),
        memoryLimitIsReached(false),
        previousNotificationTime(0.0),
        paraTaskStartTime(0.0),
        previousStopTime(-DBL_MAX),
        idleTimeToFirstParaTask(0.0),
        idleTimeBetweenParaTasks(0.0),
        idleTimeAfterLastParaTask(0.0),
        idleTimeToWaitNotificationId(0.0),
        idleTimeToWaitAckCompletion(0.0), idleTimeToWaitToken(0.0), previousIdleTimeToWaitToken(0.0), offsetTimeToWaitToken(0.0),
        nImprovedIncumbent(0),
        nParaTasksReceived(0),
        nParaTasksSolved(0),
        updatePendingSolutionIsProceeding(false),
        globalIncumbnetValueUpdateFlag(false),
        notificationProcessed(false),
        eps(MINEPSILON),
        previousCommTime(0.0),
        subproblemFreed(false),
        stayAliveAfterInterrupt(false)
{
   if( paraComm )
   {
      /** create timer for this ParaSolver */
      paraTimer = paraComm->createParaTimer();
      paraTimer->init(paraComm);
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         assert(paraDetTimer);
      }

      messageHandler = new MessageHandlerFunctionPointer[nHandlers];
      ///
      ///  register message handlers
      ///
      for( int i = 0; i < nHandlers; i++ )
      {
         messageHandler[i] = 0;
      }
      messageHandler[TagTaskReceived] = &UG::ParaSolver::processTagTaskReceived;
      messageHandler[TagRampUp] = &UG::ParaSolver::processTagRampUp;
      messageHandler[TagSolution] = &UG::ParaSolver::processTagSolution;
      messageHandler[TagIncumbentValue] = &UG::ParaSolver::processTagIncumbentValue;
      messageHandler[TagNotificationId] = &UG::ParaSolver::processTagNotificationId;
      messageHandler[TagTerminateRequest] = &UG::ParaSolver::processTagTerminateRequest;
      messageHandler[TagRacingRampUpParamSet] = &UG::ParaSolver::processTagWinnerRacingRampUpParamSet;
      messageHandler[TagWinner] = &UG::ParaSolver::processTagWinner;
      messageHandler[TagToken] = &UG::ParaSolver::processTagToken;

      offsetTimeToWaitToken = ( paraParams->getRealParamValue(NotificationInterval) / (paraComm->getSize() - 1) )
            * (paraComm->getRank() - 1);

      for(int i = 0; i < argc; i++ )
      {
         if( strcmp(argv[i], "-w") == 0 )
         {
            warmStarted = true;
         }
      }
   }
}

ParaSolver::~ParaSolver(
      )
{
   if(racingParams) delete racingParams;
   if(winnerRacingParams) delete winnerRacingParams;
   if( globalBestIncumbentSolution ) delete globalBestIncumbentSolution;
   if(localIncumbentSolution) delete localIncumbentSolution;
   if(currentTask) delete currentTask;
   if(newTask) delete newTask;

   delete paraTimer;

   if( paraInstance ) delete paraInstance;

   delete [] messageHandler;
}
