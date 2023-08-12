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


#ifndef __PARA_LOADCOORDINATOR_H__
#define __PARA_LOADCOORDINATOR_H__

#include <fstream>
#include <list>
#include <queue>
#include <mutex>
#include "paraDef.h"
#include "paraComm.h"
#include "paraCalculationState.h"
#include "paraTask.h"
#include "paraSolverState.h"
#include "paraSolverPool.h"
#include "paraInitiator.h"
#include "paraTimer.h"
#include "paraDeterministicTimer.h"

#ifdef UG_WITH_UGS
#include "ugs/ugsDef.h"
#include "ugs/ugsParaCommMpi.h"
#endif

namespace UG
{

///
/// running phase definition
///
enum RunningPhase
{
   RampUpPhase,             ///< ramp-up phase
   NormalRunningPhase,      ///< normal running phase (primary phase)
   TerminationPhase         ///< termination phase, includes interrupting phase
};

///
/// Class for LoadCoordinator
///
class ParaLoadCoordinator
{

protected:

   typedef int(UG::ParaLoadCoordinator::*MessageHandlerFunctionPointer)(int, int);

   int nHandlers;                                                 ///< number of valid handlers
   MessageHandlerFunctionPointer   *messageHandler;               ///< message handlers table for primary phase
   MessageHandlerFunctionPointer   *racingRampUpMessageHandler;   ///< message handlers table for racing stage

   int            globalSubtreeIdGen;                             ///< global subtree id generator
#ifdef UG_WITH_UGS
   UGS::UgsParaCommMpi *commUgs;                                  ///< communicator used for UGS: None zero means LC is running under UGS */
#endif
   ParaComm       *paraComm;                                      ///< communicator used
   ParaParamSet   *paraParams;                                    ///< UG parameter set
   ParaInitiator  *paraInitiator;                                 ///< initiator
   bool           *racingSolversExist;                            ///< indicate if racing solver exits or not, true: exists
   bool           restarted;                                      ///< indicates that this run is restarted from checkpoint files
   RunningPhase   runningPhase;                                   ///< status of LoadCoordinator

   bool           computationIsInterrupted;                       ///< indicate that current computation is interrupted or not
   bool           interruptedFromControlTerminal;                 ///< interrupted from control terminal
   bool           hardTimeLimitIsReached;                         ///< indicate that hard time limit is reached or not
   bool           memoryLimitIsReached;                           ///< indicate if memory limit is reached or not in a solver, when base solver has memory management feature
   bool           interruptIsRequested;                           ///< indicate that all solver interrupt message is requested or not

   ///
   /// Pools in LoadCorrdinator
   ///
   ParaSolverPool  *paraSolverPool;                               ///< solver pool
   ParaRacingSolverPool *paraRacingSolverPool;                    ///< racing solver pool

   long long       nSolvedInInterruptedRacingSolvers;             ///< number of tasks solved of the winner solver in the racing solvers
   long long       nTasksLeftInInterruptedRacingSolvers;          ///< number of of tasks remains of the the winner solver in the racing solvers

   ///
   ///  For checkpoint
   ///
   double          previousCheckpointTime;                        ///< previous checkpoint time
   char            lastCheckpointTimeStr[26];                     ///< lastCheckpointTimeStr[0] == ' ' means no checkpoint

   ///
   ///  epsilon
   ///
   double            eps;                                         ///< absolute values smaller than this are considered zero  */

   ///
   /// racing winner information
   ///
   int                racingWinner;                               ///< racing winner, -1: not determined
   ParaRacingRampUpParamSet *racingWinnerParams;                  ///< racing winner parameter set

   ///
   /// racing termination information
   ///
   bool               racingTermination;                          ///< racing termination flag, true: if a racing solver solved the problem
   int                nSolvedRacingTermination;                   ///< number of tasks solved at the racing termination solver

   ///
   /// counter to check if all solvers are terminated or not
   ///
   size_t             nTerminated;                                ///< number of terminated Solvers

   ///
   ///  Timers for LoadCoordinator
   ///
   ParaTimer              *paraTimer;                             ///< normal timer used
   ParaDeterministicTimer *paraDetTimer;                          ///< deterministic timer used in case of deterministic mode
                                                                  ///< this timer need to be created in case of deterministic mode

   ///
   /// output streams and flags which indicate the output is specified or not
   ///
   bool               logSolvingStatusFlag;                       ///< indicate if solving status is logged or not
   std::ofstream      ofsLogSolvingStatus;                        ///< ofstream for solving status
   std::ostream       *osLogSolvingStatus;                        ///< ostram for solving status to switch output location
   bool               logTasksTransferFlag;                       ///< indicate if task transfer info. is logged or not
   std::ofstream      ofsLogTasksTransfer;                        ///< ofstream for task transfer info.
   std::ostream       *osLogTasksTransfer;                        ///< ostream for task transfer info. to switch output location
   std::ofstream      ofsStatisticsFinalRun;                      ///< ofstream for statistics of the final run
   std::ostream       *osStatisticsFinalRun;                      ///< ostream for statistics of the final run
   std::ofstream      ofsStatisticsRacingRampUp;                  ///< ofstream for statistics for racing solvers
   std::ostream       *osStatisticsRacingRampUp;                  ///< ostream for statistics for racing solvers to switch output location

   ParaSolution *pendingSolution;                                 ///< pending solution during merging
   bool terminationIssued;                                        ///< indicate termination request is issued
   std::mutex routineMutex;                                       ///< used to exclusive control of routines

   ///
   /// write transfer log
   ///
   virtual void writeTransferLog(
         int rank,                                        ///< solver rank
         ParaCalculationState *state                      ///< calculation status
         );
   ///
   /// write transfer log
   ///
   virtual void writeTransferLog(
         int rank                                        ///< solver rank
         );

   ///
   /// write transfer log in racing
   ///
   virtual void writeTransferLogInRacing(
         int rank,                                       ///< solver rank
         ParaCalculationState *state                     ///< calculation status
         );

   ///
   /// write transfer log in racing
   ///
   virtual void writeTransferLogInRacing(
         int rank                                        ///< solver rank
         );

   ///
   /// notify ramp-up to all solvers
   ///
   virtual void sendRampUpToAllSolvers(
         );

   ///
   /// notify retry ramp-up to all solvers (Maybe, this remove from this base class)
   ///
   virtual void sendRetryRampUpToAllSolvers(
         )
   {
   }

   ///
   /// send interrupt request to all solvers
   ///
   virtual void sendInterruptRequest(
        ) = 0;

   ///
   /// terminate all solvers
   ///
   virtual void terminateAllSolvers(
         );

   ///
   /// create a new global subtree Id
   /// @return global subtree id generated
   ///
   int createNewGlobalSubtreeId(
         )
   {
      return ++globalSubtreeIdGen;
   }

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
         ) = 0;

   ///
   /// function to process TagSolution message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolution(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSolution
         ) = 0;

   ///
   /// function to process TagSolverState message
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolverState(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagSolverState
         ) = 0;

   ///
   /// function to process TagCompletionOfCalculation message
   /// @return always 0 (for extension)
   ///
   virtual int processTagCompletionOfCalculation(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagCompletionOfCalculation
         ) = 0;

   ///
   /// function to process TagTerminated message
   /// @return always 0 (for extension)
   ///
   virtual int processTagTerminated(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagTerminated
         );

   ///
   /// function to process TagHardTimeLimit message
   /// @return always 0 (for extension)
   ///
   virtual int processTagHardTimeLimit(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagHardTimeLimit
         );

   ///
   /// function to process TagToken message
   /// @return always 0 (for extension)
   ///
   virtual int processTagToken(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagToken
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
         ) = 0;

   ///
   /// function to process TagCompletionOfCalculation message in racing ramp-up stage
   /// @return always 0 (for extension)
   ///
   virtual int processRacingRampUpTagCompletionOfCalculation(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagCompletionOfCalculation
         ) = 0;

   ///
   /// check if current stage is in racing or not
   /// @return true, if current stage is in racing
   ///
//   bool isRacingStage(
//         )
//   {
//      if( // ( !paraInitiator->getPrefixWarm() ) &&
//            runningPhase == RampUpPhase &&
//            paraParams->getIntParamValue(RampUpPhaseProcess) > 0 &&  racingWinner < 0 )
//         return true;
//      else
//         return false;
//   }

   ///
   /// send specified tag to all solvers
   ///
   void sendTagToAllSolvers(
         const int tag                    ///< tag which is sent to all solvers
         );

#ifdef UG_WITH_UGS

   ///
   ///  check and read incument solution
   ///
   // int checkAndReadIncumbent(
   //      );

#endif

   ///
   /// run function to start main process
   ///
   virtual void run(
         ) = 0;

   ///
   /// send ParaTasks to idle solvers
   /// @return true, if a ParaTasks is sent
   ///
   virtual bool sendParaTasksToIdleSolvers(
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// function to update checkpoint files
   ///
   virtual void updateCheckpointFiles(
         ) = 0;

#endif

public:

   ///
   /// constructor
   ///
   ParaLoadCoordinator(
#ifdef UG_WITH_UGS
         UGS::UgsParaCommMpi *inComUgs,      ///< communicator used for UGS
#endif
         int nHandlers,                      ///< number of handlers
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
   virtual ~ParaLoadCoordinator(
         )
   {
      /// destructor should be implemented appropriately in a derived class of ParaLoadCoordinator
      if( paraSolverPool ) delete paraSolverPool;
      if( paraRacingSolverPool ) delete paraRacingSolverPool;

      if( messageHandler ) delete [] messageHandler;
      if( racingRampUpMessageHandler ) delete [] racingRampUpMessageHandler;
   }

   ///
   /// interrupt from out side
   ///
   virtual void interrupt(
         )
   {
      interruptedFromControlTerminal = true;
      sendInterruptRequest();
   }

#ifdef UG_WITH_ZLIB

   ///
   /// warm start (restart)
   ///
   virtual void warmStart(
         )
   {
      // if user want to support warm start (restart), user need to implement this
   }

#endif

   ///
   /// run for normal ramp-up
   ///
   virtual void run(
         ParaTask *paraTask                              ///< root ParaTask
         )
   {
   }

   ///
   /// run for racing ramp-up
   ///
   virtual void run(
         ParaTask *paraTask,                             ///< root ParaTask
         int nRacingSolvers,                             ///< number of racing solvers
         ParaRacingRampUpParamSet **racingRampUpParams   ///< racing parameters
         )
   {
   }

   ///
   /// execute UG parallel solver totally solver dependent way
   ///
   virtual void parallelDispatch(
         )
   {
      run();
   }

};

}

#endif // __PARA_LOADCOORDINATOR_H__

