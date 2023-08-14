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

/**@file    paraSolver.h
 * @brief   Base class for solver: Generic parallelized solver.
 * @author  Yuji Shinano
 *
 *
 */


/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_SOLVER_H__
#define __PARA_SOLVER_H__

#include "paraDef.h"
#include "paraComm.h"
#include "paraParamSet.h"
#include "paraRacingRampUpParamSet.h"
#include "paraTask.h"
#include "paraTimer.h"
#include "paraDeterministicTimer.h"
#include "paraSolution.h"
// #ifdef _MSC_VER
// #include "pthread.h"
// #endif

#define ENFORCED_THRESHOLD 5

namespace UG
{

///
/// termination mode
///
static const int NoTerminationMode          = 0;
static const int NormalTerminationMode      = 1;
static const int InterruptedTerminationMode = 2;
static const int TimeLimitTerminationMode   = 3;
///----------------------------------------------
static const int UgBaseTerminationModeLast  = 3;

///
/// class ParaSolver
///
class ParaSolver
{

protected:

   typedef int(ParaSolver::*MessageHandlerFunctionPointer)(int, int);

   int nHandlers;                                        ///< number of valid message handlers
   MessageHandlerFunctionPointer *messageHandler;     ///< table for message handlers

   unsigned int notificationIdGenerator;

   ParaComm     *paraComm;                            ///< ParaCommunicator object
   ParaParamSet *paraParams;                          ///< ParaParamSet object
   ParaRacingRampUpParamSet *racingParams;            ///< ParaRacingRampUpParamSet object.
                                                      ///< This is also a flag to indicate running with racing ramp-up
   ParaRacingRampUpParamSet *winnerRacingParams;      ///< Winner ParaRacingRampUpParamSet object

   ParaTimer    *paraTimer;                           ///< timer for this ParaSolver
   ParaDeterministicTimer *paraDetTimer;              ///< deterministic timer for this ParaSolver

   double       globalBestIncumbentValue;             ///< global best incumbent value
   ParaSolution *globalBestIncumbentSolution;         ///< global best solution. However, this is not always feasible for the current sub-MIP
   ParaSolution *localIncumbentSolution;              ///< incumbent solution generated in local solver
   ParaSolution *pendingSolution;                     ///< solution which is pending to update in case of deterministic runs
   double        pendingIncumbentValue;               ///< incumbent value which is pending to update in case of deterministic runs
   ParaInstance *paraInstance;                        ///< root problem instance
   ParaTask     *currentTask;                         ///< solving task
   ParaTask     *newTask;                             ///< new task to solve
   int          terminationMode;                      ///< indicate that termination mode
                                                      ///<     0: no termination mode
                                                      ///<     1: normal termination mode
                                                      ///<     2: interrupted termination
   bool         warmStarted;                          ///< indicate whether if system is warm started or not
   bool         rampUp;                               ///< indicate whether if ramp-up phase is finished or not: true - finish
   bool         racingInterruptIsRequested;           ///< indicate a racing interrupt is requested
   bool         racingIsInterrupted;                  ///< indicate whether if racing phases is interrupted or not: true - interrupted
   bool         racingWinner;                         ///< indicate racing ramp-up winner or not: true - winner
   bool         waitingSpecificMessage;               ///< indicate that this solver is waiting for a specific message
   bool         memoryLimitIsReached;                 ///< indicate if memory limit is reached or not, when base solver has memory management feature

   double       previousNotificationTime;             ///< previous notification time
   double       paraTaskStartTime;                    ///< start time of current ParaTask

   ///
   ///  Idle Times
   ///
   double       previousStopTime;                     ///< previous stop solving time of this Solver: For measurement
   double       idleTimeToFirstParaTask;              ///< idle time to start solving the first ParaTask
   double       idleTimeBetweenParaTasks;             ///< idle time between ParaTasks processing
   double       idleTimeAfterLastParaTask;            ///< idle time after the last ParaTask was solved
   double       idleTimeToWaitNotificationId;         ///< idle time to wait a message within collecting mode
   double       idleTimeToWaitAckCompletion;          ///< idle time to wait acknowledgment of completion
   double       idleTimeToWaitToken;                  ///< idle time to wait token
   double       previousIdleTimeToWaitToken;          ///< previous idle time to wait token
   double       offsetTimeToWaitToken;                ///< offset time to wait token

   ///
   ///  Counters related to the current ParaTask
   ///
   int          nImprovedIncumbent;                   ///< number of improvements of incumbent value

   ///
   ///  Counters related to this ParaSolver
   ///
   int          nParaTasksReceived;                   ///< number of ParaTasks received in this ParaSolver
   int          nParaTasksSolved;                     ///< number of ParaTasks solved ( received ) in this ParaSolver

   bool         updatePendingSolutionIsProceeding;    ///< update pending solution process is proceeding
   bool         globalIncumbnetValueUpdateFlag;       ///< indicate that global incumbent value is updated in iReceiveMessages() routine
   bool         notificationProcessed;                ///< if true, notification is issued but not receive the corresponding LCB

   double       eps;                                  ///< absolute values smaller than this are considered zero
                                                      ///< esp should be set in the constructor of the derived class of ParaSolver

   double       previousCommTime;                     ///< previous communication time for deterministic execution

   bool subproblemFreed;                              ///< indicate that subproblem is already freed or not

   bool stayAliveAfterInterrupt;                      ///< indicate that stay alive this solver after interrupt request

   ///-------------------
   /// Message handlers
   ///-------------------

   ///
   /// process TagTask
   /// @return always 0 (for extension)
   ///
   virtual int processTagTask(
         int source,    ///< source rank
         int tag        ///< TagTask
         ) = 0;

   ///
   /// process TagTaskReceived
   /// @return always 0 (for extension)
   ///
   virtual int processTagTaskReceived(
         int source,    ///< source rank
         int tag        ///< TagTaskReceived
         ) = 0;

   ///
   /// process TagRampUp
   /// @return always 0 (for extension)
   ///
   virtual int processTagRampUp(
         int source,    ///< source rank
         int tag        ///< TagRampUp
         )
   {
      std::cout << "*** virtual function ParaSolver::processTagRampUp is called ***" << std::endl;
      return 0;
   }

   ///
   /// process TagSolution
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolution(
         int source,    ///< source rank
         int tag        ///< TagSolution
         ) = 0;

   ///
   /// process TagIncumbentValue
   /// @return always 0 (for extension)
   ///
   virtual int processTagIncumbentValue(
        int source,    ///< source rank
        int tag        ///< TagIncumbentValue
        )
   {
      std::cout << "*** virtual function ParaSolver::processTagIncumbentValue is called ***" << std::endl;
      return 0;
   }

   ///
   /// process TagNotificationId
   /// @return always 0 (for extension)
   ///
   virtual int processTagNotificationId(
         int source,    ///< source rank
         int tag        ///< TagNotificationId
         ) = 0;

   ///
   /// process TagTerminateRequest
   /// @return always 0 (for extension)
   ///
   virtual int processTagTerminateRequest(
         int source,    ///< source rank
         int tag        ///< TagTerminateRequest
         ) = 0;

   ///
   /// process TagInterruptRequest
   /// @return always 0 (for extension)
   ///
   virtual int processTagInterruptRequest(
         int source,    ///< source rank
         int tag        ///< TagInterruptRequest
         ) = 0;

   ///
   /// process TagWinnerRacingRampUpParamSet
   /// @return always 0 (for extension)
   ///
   virtual int processTagWinnerRacingRampUpParamSet(
         int source,    ///< source rank
         int tag        ///< TagWinnerRacingRampUpParamSet
         )
   {
      std::cout << "*** virtual function ParaSolver::processTagWinnerRacingRampUpParamSet is called ***" << std::endl;
      return 0;
   }

   ///
   /// process TagWinner
   /// @return always 0 (for extension)
   ///
   virtual int processTagWinner(
         int source,    ///< source rank
         int tag        ///< TagWinner
         )
   {
      std::cout << "*** virtual function ParaSolver::processTagWinner is called ***" << std::endl;
      return 0;
   }

   ///
   /// process TagToken
   /// @return always 0 (for extension)
   ///
   virtual int processTagToken(
         int source,    ///< source rank
         int tag        ///< TagToken
         )
   {
      std::cout << "*** virtual function ParaSolver::processTagToken is called ***" << std::endl;
      return 0;
   }

   ///
   /// wait for receiving a new task and reactivate solver
   /// @return true if a new task is received, false esle
   ///
   virtual bool receiveNewTaskAndReactivate(
         ) = 0;

   ///
   /// wait notification id message to synchronized with LoadCoordinator
   ///
   virtual void waitNotificationIdMessage(
         ) = 0;

   ///
   /// wait ack completion to synchronized with LoadCoordinator
   ///
   virtual void waitAckCompletion(
         ) = 0;

   ///
   /// restart racing
   ///
//   void restartRacing(
//        );

   ///
   /// send completion of calculation
   ///
   virtual void sendCompletionOfCalculation(
         double stopTime       ///< stopping time
         ) = 0;

   ///
   /// update global best incumbent solution
   /// @return true if the best incumbent solution was updated, false otherwise
   ///
   virtual bool updateGlobalBestIncumbentSolution(
        ParaSolution *sol     ///< pointer to new solution object
        )
   {
      std::cout << "*** virtual function ParaSolver::updateGlobalBestIncumbentSolution is called ***" << std::endl;
      return false;
   }

   ///
   /// update global best incumbent value
   /// @return true if the best incumbent value was updated, false otherwise
   ///
   virtual bool updateGlobalBestIncumbentValue(
        double newValue       ///< new incumbent value
        )
   {
      std::cout << "*** virtual function ParaSolver::updateGlobalBestIncumbentValue is called ***" << std::endl;
      return false;
   }

   ///
   /// set racing parameters
   ///
   virtual void setRacingParams(
         ParaRacingRampUpParamSet *racingParms,   ///< pointer to racing parameter set object
         bool winnerParam                         ///< indicate if the parameter set is winner one
         )
   {
      std::cout << "*** virtual function ParaSolver::setRacingParams is called ***" << std::endl;
   }

   ///
   /// set winner racing parameters
   ///
   virtual void setWinnerRacingParams(
         ParaRacingRampUpParamSet *racingParms    ///< pointer to winner racing parameter set object
         )
   {
      std::cout << "*** virtual function ParaSolver::setWinnerRacingParams is called ***" << std::endl;
   }

   ///
   /// create subproblem
   ///
   virtual void createSubproblem(
         )
   {
      std::cout << "*** virtual function ParaSolver::createSubproblem is called ***" << std::endl;
   }

   ///
   /// free subproblem
   ///
   virtual void freeSubproblem(
         )
   {
      std::cout << "*** virtual function ParaSolver::freeSubproblem is called ***" << std::endl;
   }

   ///
   /// solve (sub)problem
   ///
   virtual void solve(
         ) = 0;

   ///
   /// re-initialized instance
   ///
   virtual void reinitialize(
         )
   {
      std::cout << "*** virtual function ParaSolver::reinitializeInstance is called ***" << std::endl;
   }

public:

   ///
   /// constructor
   ///
   ParaSolver(
         )
   {
      THROW_LOGICAL_ERROR1("Default constructor of ParaSolver is called");
   }

   ///
   /// constructor
   ///
   ParaSolver(
         int argc,                          ///< number of arguments
         char **argv,                       ///< array of arguments
         int nHandlers,                     ///< number of valid message handlers
         ParaComm *comm,                    ///< communicator used
         ParaParamSet *inParaParamSet,      ///< pointer to ParaParamSet object
         ParaInstance *paraInstance,        ///< pointer to ParaInstance object
         ParaDeterministicTimer *detTimer   ///< pointer to deterministic timer object
         );

   ///
   /// destructor
   ///
   virtual ~ParaSolver(
         );

   ///
   /// get paraParaComm
   /// @return communicator used
   ///
   ParaComm  *getParaComm(
         )
   {
      return paraComm;
   }

   ///
   /// check if current execution is warm start (restart) or not
   /// @return true if the execution is warm start (restart), false otherwise
   ///
   bool isWarmStarted(
         )
   {
      return warmStarted;
   }

   ///
   /// run this Solver
   ///
   virtual void run(
         ) = 0;

   ///
   /// run this Solver with ParaTask object
   ///
   virtual void run(
         ParaTask *paraTask    ///< pointer to ParaTask object
         )
   {
      currentTask = paraTask;
      run();
   }

   ///
   /// run this solver with racing parameters
   ///
   virtual void run(
         ParaRacingRampUpParamSet *inRacingRampUpParamSet    ///< pointer to ParaRacingRampUpParamSet object
         )
   {
      std::cout << "*** virtual function ParaSolver::run(araRacingRampUpParamSet *) is called ***" << std::endl;

      // Example code is below:
      //----------------------------------------------------------------
      //      ParaTask *rootTask = paraComm->createParaTask();
      //      PARA_COMM_CALL(
      //            rootTask->bcast(paraComm, 0)
      //            );
      //      nParaTasksReceived++;
      //      racingParams = inRacingRampUpParamSet;
      //      setRacingParams(racingParams, false);
      //      if( paraParams->getBoolParamValue(Deterministic) )
      //      {
      //         do
      //         {
      //            iReceiveMessages();
      //         } while( !waitToken(paraComm->getRank()) );
      //      }
      //      iReceiveMessages();   // Feasible solution may be received.
      //      if( paraParams->getBoolParamValue(Deterministic) )
      //      {
      //         passToken(paraComm->getRank());
      //      }
      //      run( rootTask );

   }

   ///
   /// the following functions may be called from callback routines of the target Solver
   ///

   ///
   /// get elapsed time of task solving
   /// @return elapsed time
   ///
   double getElapsedTimeOfTaskSolving(
         )
   {
      return (paraTimer->getElapsedTime() - paraTaskStartTime);
   }

   ///
   /// non-blocking receive messages
   ///
   virtual void iReceiveMessages(
         ) = 0;

   ///
   /// check if this solver is ramp-up or not
   /// @return true if ramp-upped, false otherwise
   ///
   bool isRampUp(
         )
   {
      return rampUp;
   }

   ///
   /// check if this solver is in racing ramp-up or not
   /// @return true if this solver is in racing ramp-up, false otherwise
   ///
//   bool isRacingRampUp(
//         )
//   {
//      return ( ( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ) ||
//            ( paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) );
//   }

   ///
   /// check if this solver is the racing winner or not
   /// @return true if this solver is the racing winner, false otherwise
   ///
   bool isRacingWinner(
         )
   {
      return racingWinner;
   }

   ///
   /// send improved solution if it was found in this Solver
   ///
   virtual bool sendIfImprovedSolutionWasFound(
         ParaSolution *sol       ///< solution found in this Solver
         )
   {
      std::cout << "*** virtual function ParaSolver::sendIfImprovedSolutionWasFound is called ***" << std::endl;
      return false;
   }

   ///
   /// save improved solution if it was found in this Solver
   ///
   virtual bool saveIfImprovedSolutionWasFound(
         ParaSolution *sol       ///< solution found in this Solver
         )
   {
      std::cout << "*** virtual function ParaSolver::saveIfImprovedSolutionWasFound is called ***" << std::endl;
      return false;
   }

   ///
   /// send solution found in this Solver
   ///
   virtual void sendLocalSolution(
         )
   {
      std::cout << "*** virtual function ParaSolver::sendLocalSolution is called ***" << std::endl;
   }

   ///
   /// check if a notification message needs to send or not
   /// TODO: function name should be isNotificationNecessary
   /// @return true if the notification message needs to send, false otherwise
   ///
   virtual bool notificationIsNecessary(
         ) = 0;

   ///
   /// check if Solver is in interrupting phase or not
   /// @return true if Solver is in interrupting phase, false otherwise
   ///
   bool isInterrupting(
         )
   {
      return ( terminationMode == InterruptedTerminationMode );
   }

   ///
   /// check if termination was requested or not
   /// @return true if termination was requested, false otherwise
   ///
   bool isTerminationRequested(
         )
   {
      return  ( terminationMode == NormalTerminationMode );
   }

   ///
   /// check if a new ParaTask was received or not
   /// @return true if a new ParaTask was received, false otherwise
   ///
   bool newParaTaskExists(
         )
   {
      return (newTask != 0);
   }

   ///
   /// check if Solver is in notification process or not
   /// TODO: function name should be changed
   /// @return true if Solver is in notification process, false otherwise
   ///
   bool getNotificaionProcessed(
         )
   {
      return notificationProcessed;
   }

   ///
   /// get current ParaTask object
   /// @return pointer to ParaTask object
   ///
   ParaTask *getCurrentTask(
         )
   {
      return currentTask;
   }

   ///
   /// get ParaInstance object
   /// @return pointer to ParaInstance object
   ///
   ParaInstance *getParaInstance(
         )
   {
      return paraInstance;
   }

   ///
   /// get ParaParamSet object
   /// @return pointer to ParaParamSet object
   ///
   ParaParamSet *getParaParamSet(
         )
   {
      return paraParams;
   }

   ///
   /// get rank of this Solver
   /// @return rank of this Solver
   ///
   virtual int getRank(
         )
   {
      return paraComm->getRank();
   }

   ///
   /// wait a notification id message if it is needed to synchronize with LoadCoordinaor
   ///
   virtual void waitMessageIfNecessary(
         ) = 0;

   ///
   /// check if Solver is in racing stage or not
   /// @return true if Solver is in racing stage, false otherwise
   ///
//   bool isRacingStage(
//         )
//   {
//      return (racingParams &&
//            (paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ||
//             paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) );
//   }

   ///
   /// terminate racing stage
   ///
   void terminateRacing()
   {
      assert(racingParams);
      delete racingParams;
      racingParams = 0;
      racingInterruptIsRequested = false;
      racingIsInterrupted = false;    // rampUp message might have been received before terminate racing
                                      // Then, this flag should be set false
   }

   ///
   /// get global best incumbent solution
   /// @return pointer to ParaSolution object
   ///
   ParaSolution *getGlobalBestIncumbentSolution(
         )
   {
      return globalBestIncumbentSolution;
   }

   ///
   /// check if Solver is waiting for a specific message or not
   /// @return true if Solver is waiting for a specific message, false otherwise
   ///
   bool isWaitingForSpecificMessage(
         )
   {
      return waitingSpecificMessage;
   }

   ///
   /// wait token for deterministic mode
   /// @return true when token is received, false otherwise
   ///
   virtual bool waitToken(
         int rank     ///< rank of this Solver
         )
   {
      std::cout << "*** virtual function ParaSolver::waitToken is called ***" << std::endl;
      return false;
//      bool result;
//      double startTimeToWaitToken = paraTimer->getElapsedTime();
//      result = paraComm->waitToken(rank);
//      idleTimeToWaitToken += (paraTimer->getElapsedTime() - startTimeToWaitToken);
//      return result;
   }

   ///
   /// pass token to the next process
   ///
   virtual void passToken(
         int rank     ///< rank of this Solver
         )
   {
      std::cout << "*** virtual function ParaSolver::passToken is called ***" << std::endl;
//      paraComm->passToken(rank);
   }

   ///
   /// get deterministic timer object
   /// @return pointer to deterministic timer object
   ///
   ParaDeterministicTimer *getDeterministicTimer(
         )
   {
      return paraDetTimer;
   }

   ///
   /// get offset time to wait token
   /// @return offset time
   ///
   double getOffsetTimeToWaitToken(
         )
   {
      return offsetTimeToWaitToken;
   }

   ///
   /// update pending solution
   /// @note there is a case that solution cannot updated immediately for some timing issue
   ///
   virtual void updatePendingSolution(
         )
   {
      if( updatePendingSolutionIsProceeding == false )
      {
         updatePendingSolutionIsProceeding = true;
         if( !pendingSolution )
         {
            updateGlobalBestIncumbentValue(pendingIncumbentValue);
            pendingIncumbentValue = DBL_MAX;
            updatePendingSolutionIsProceeding = false;
            return;
         }
         if( updateGlobalBestIncumbentSolution(pendingSolution) )
         {
            tryNewSolution(pendingSolution);
         }
         else
         {
            delete pendingSolution;
         }
         pendingIncumbentValue = DBL_MAX;
         pendingSolution = 0;
         updatePendingSolutionIsProceeding = false;
      }
   }

   ///
   /// check if Solver was terminated normally or not
   /// @return true if Solver was terminated normally, false otherwise
   ///
   virtual bool wasTerminatedNormally(
         ) = 0;

   ///
   /// write current task problem
   /// (this method is always useful for debugging, so we should implement this method)
   ///
   virtual void writeCurrentTaskProblem(
         const std::string& filename       ///< file name to write
         ) = 0;

   ///
   /// try to enter solution to base solver environment
   ///
   virtual void tryNewSolution(
         ParaSolution *sol                ///< solution to be enterred
         ) = 0;

   ///
   /// write subproblem
   ///
   virtual void writeSubproblem(
         ) = 0;

   ///
   /// set previous communication time for deterministic execution
   ///
   void setPreviousCommTime(
         double detTime           ///< deterministic time
         )
   {
      previousCommTime = detTime;
   }

   ///
   /// get previous communication time for deterministic execution
   /// @return previous communication time in deterministic time
   ///
   double getPreviousCommTime(
         )
   {
      return previousCommTime;
   }

   ///
   /// set termination mode
   ///
   void setTerminationMode(
         int tm                   ///< terminiation mode to be set
         )
   {
      terminationMode = tm;
   }

   ///
   /// get termination mode
   /// @return termination mode
   ///
   int getTerminationMode(
         )
   {
      return terminationMode;
   }

};

}

#endif // __PARA_SOLVER_H__
