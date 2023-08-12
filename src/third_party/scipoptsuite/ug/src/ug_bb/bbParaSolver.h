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


#ifndef __BB_PARA_SOLVER_H__
#define __BB_PARA_SOLVER_H__

#include "ug/paraDef.h"
#include "ug/paraComm.h"
#include "ug/paraRacingRampUpParamSet.h"
#include "ug/paraTimer.h"
#include "ug/paraDeterministicTimer.h"
#include "ug/paraSolution.h"
#include "ug/paraSolver.h"
#include "bbParaTagDef.h"
#include "bbParaParamSet.h"
#include "bbParaNode.h"
#include "bbParaNodePool.h"
// #ifdef _MSC_VER
// #include "pthread.h"
// #endif

#define ENFORCED_THRESHOLD 5

namespace UG
{

///
/// class BbParaSolver
///
class BbParaSolver : public ParaSolver
{

protected:

   typedef int(BbParaSolver::*BbMessageHandlerFunctionPointer)(int, int);

   double       globalBestDualBoundValueAtWarmStart;  ///< global best dual bound value which is set when system warm starts
   double       globalBestCutOffValue;                ///< global best cut off value
   double       lcBestDualBoundValue;                 ///< LoadCoordinator best dual bound value
   bool         collectingMode;                       ///< indicate whether if this solver is in collecting mode or not
   bool         aggressiveCollecting;                 ///< indicate that if this solver has two nodes, this solver sends one to LC
   int          nSendInCollectingMode;                ///< number of nodes need to send in collecting mode
   int          nCollectOnce;                         ///< number of nodes need to collect once
   bool         collectingManyNodes;                  ///< indicate that many nodes collecting is requested by LC
   bool         collectingInterrupt;                  ///< when the solver is interrupted, all nodes are collected to LC
   bool         anotherNodeIsRequested;               ///< indicate that another node is requested or not
   bool         lightWeightRootNodeComputation;       ///< indicate that fast root node computation is required

   bool         onceBreak;                            ///< indicate that the sub-MIP is broken down once

   ///
   ///  Times
   ///
   double       rootNodeTime;                         ///< root node process time of current ParaNode
   double       totalRootNodeTime;                    ///< accumulated root node process time solved by this solver so far
   double       minRootNodeTime;                      ///< minimum time consumed by root node process
   double       maxRootNodeTime;                      ///< maximum time consumed by root node process

   ///
   ///  Counters related to the current ParaNode
   ///
   int          nSolved;                              ///< number of nodes solved, that is, number of subtree nodes rooted from ParaNode
   int          nSent;                                ///< number of ParaNodes sent from this subtree rooted from the current ParaNode
   int          nSolvedWithNoPreprocesses;            ///< number of nodes solved when it is solved with no preprocesses

   ///
   ///  Counters related to this BbParaSolver
   ///
   int          totalNSolved;                         ///< accumulated number of nodes solved in this BbParaSolver
   int          minNSolved;                           ///< minimum number of subtree nodes rooted from ParaNode
   int          maxNSolved;                           ///< maximum number of subtree nodes rooted from ParaNode
   int          nTransferredLocalCutsFromSolver;      ///< number of local cuts transferred from this Solver
   int          minTransferredLocalCutsFromSolver;    ///< minimum number of local cuts transferred from this Solver
   int          maxTransferredLocalCutsFromSolver;    ///< maximum number of local cuts transferred from this Solver
   int          nTransferredBendersCutsFromSolver;    ///< number of benders cuts transferred from this Solver
   int          minTransferredBendersCutsFromSolver;  ///< minimum number of benders cuts transferred from this Solver
   int          maxTransferredBendersCutsFromSolver;  ///< maximum number of benders cuts transferred from this Solver
   int          nTotalRestarts;                       ///< number of total restarts
   int          minRestarts;                          ///< minimum number of restarts
   int          maxRestarts;                          ///< maximum number of restarts
   int          totalNSent;                           ///< accumulated number of nodes sent from this BbParaSolver
   int          totalNImprovedIncumbent;              ///< accumulated number of improvements of incumbent value in this BbParaSolver
   int          nParaNodesSolvedAtRoot;               ///< number of ParaNodes solved at root node
   int          nParaNodesSolvedAtPreCheck;           ///< number of ParaNodes solved at pre-checking of root node solvability

   int          nSimplexIterRoot;                     ///< number of simplex iteration at root node
   int          nTransferredLocalCuts;                ///< number of local cuts (including conflict cuts) transferred from a ParaNode
   int          minTransferredLocalCuts;              ///< minimum number of local cuts (including conflict  cuts) transferred from a ParaNode
   int          maxTransferredLocalCuts;              ///< maximum number of local cuts (including conflict  cuts) transferred from a ParaNode
   int          nTransferredBendersCuts;              ///< number of benders cuts transferred from a ParaNode
   int          minTransferredBendersCuts;            ///< minimum number of benders cuts transferred from a ParaNode
   int          maxTransferredBendersCuts;            ///< maximum number of benders cuts transferred from a ParaNode
   int          nTightened;                           ///< the number of tightened variable bounds in racing
   int          nTightenedInt;                        ///< the number of tightened integral variable bounds in racing
   double       minIisum;                             ///< minimum sum of integer infeasibility
   double       maxIisum;                             ///< maximum sum of integer infeasibility
   int          minNii;                               ///< minimum number of integer infeasibility
   int          maxNii;                               ///< maximum number of integer infeasibility

   double       targetBound;                          ///< target bound value for breaking
   int          nTransferLimit;                       ///< limit number of transferring nodes for breaking
   int          nTransferredNodes;                    ///< keep track number of transferred nodes for breaking

   double       solverDualBound;                      ///< dual bound value achieved for a subproblem

   double       averageDualBoundGain;                 ///< average dual bound gain
   bool         enoughGainObtained;                   ///< indicate that the root node process improved dual bound enough or not

   bool         givenGapIsReached;                    ///< indicate that the given gap is reached or not
   bool         testDualBoundGain;                    ///< indicate that the dual bound gain needs to test or not
   bool         noWaitModeSend;                       ///< indicate that no wait mode sending is applied
   bool         keepRacing;                           ///< indicate if Solver needs to do racing ramp-up repeatedly in case of warm start
   bool         restartingRacing;                     ///< indicate that this solver is restarting racing
   bool         localIncumbentIsChecked;              ///< indicate if a local incumbent solution is checked or not

   ///
   /// Pool in Solver
   ///
   BbParaNodePool *selfSplitNodePool;                 ///< node pool for self-split subtree root nodes

   ///-------------------
   /// Message handlers
   ///-------------------

   ///
   /// process TagNode
   /// @return always 0 (for extension)
   ///
   virtual int processTagTask(
         int source,    ///< source rank
         int tag        ///< TagNode
        );

   ///
   /// process TagTaskReceived
   /// @return always 0 (for extension)
   ///
   virtual int processTagTaskReceived(
         int source,    ///< source rank
         int tag        ///< TagTaskReceived
         );

   ///
   /// process TagRampUp
   /// @return always 0 (for extension)
   ///
   virtual int processTagRampUp(
         int source,    ///< source rank
         int tag        ///< TagRampUp
         );

   ///
   /// process TagSolution
   /// @return always 0 (for extension)
   ///
   virtual int processTagSolution(
         int source,    ///< source rank
         int tag        ///< TagSolution
         );

   ///
   /// process TagIncumbentValue
   /// @return always 0 (for extension)
   ///
   virtual int processTagIncumbentValue(
        int source,    ///< source rank
        int tag        ///< TagIncumbentValue
        );

   ///
   /// process TagNotificationId
   /// @return always 0 (for extension)
   ///
   virtual int processTagNotificationId(
         int source,    ///< source rank
         int tag        ///< TagNotificationId
         );

   ///
   /// process TagTerminateRequest
   /// @return always 0 (for extension)
   ///
   virtual int processTagTerminateRequest(
         int source,    ///< source rank
         int tag        ///< TagTerminateRequest
         );

   ///
   /// process TagInterruptRequest
   /// @return always 0 (for extension)
   ///
   virtual int processTagInterruptRequest(
         int source,    ///< source rank
         int tag        ///< TagInterruptRequest
         );

   ///
   /// process TagWinnerRacingRampUpParamSet
   /// @return always 0 (for extension)
   ///
   virtual int processTagWinnerRacingRampUpParamSet(
         int source,    ///< source rank
         int tag        ///< TagWinnerRacingRampUpParamSet
         );

   ///
   /// process TagWinner
   /// @return always 0 (for extension)
   ///
   virtual int processTagWinner(
         int source,    ///< source rank
         int tag        ///< TagWinner
         );

   ///
   /// process TagToken
   /// @return always 0 (for extension)
   ///
   virtual int processTagToken(
         int source,    ///< source rank
         int tag        ///< TagToken
         );

   ///
   /// process TagRetryRampUp
   /// @return always 0 (for extension)
   ///
   virtual int processTagRetryRampUp(
         int source,    ///< source rank
         int tag        ///< TagRetryRampUp
         );

   ///
   /// process TagGlobalBestDualBoundValueAtWarmStart
   /// @return always 0 (for extension)
   ///
   virtual int processTagGlobalBestDualBoundValueAtWarmStart(
         int source,    ///< source rank
         int tag        ///< TagGlobalBestDualBoundValueAtWarmStart
         );

   ///
   /// process TagNoNodes
   /// @return always 0 (for extension)
   ///
   virtual int processTagNoNodes(
         int source,    ///< source rank
         int tag        ///< TagNoNodes
         );

   ///
   /// process TagInCollectingMode
   /// @return always 0 (for extension)
   ///
   virtual int processTagInCollectingMode(
         int source,    ///< source rank
         int tag        ///< TagInCollectingMode
         );

   ///
   /// process TagCollectAllNodes
   /// @return always 0 (for extension)
   ///
   virtual int processTagCollectAllNodes(
         int source,    ///< source rank
         int tag        ///< TagCollectAllNodes
         );

   ///
   /// process TagOutCollectingMode
   /// @return always 0 (for extension)
   ///
   virtual int processTagOutCollectingMode(
         int source,    ///< source rank
         int tag        ///< TagOutCollectingMode
         );

   ///
   /// process TagLCBestBoundValue
   /// @return always 0 (for extension)
   ///
   virtual int processTagLCBestBoundValue(
         int source,    ///< source rank
         int tag        ///< TagLCBestBoundValue
         );

   ///
   /// process TagLightWeightRootNodeProcess
   /// @return always 0 (for extension)
   ///
   virtual int processTagLightWeightRootNodeProcess(
         int source,    ///< source rank
         int tag        ///< TagLightWeightRootNodeProcess
         );

   ///
   /// process TagBreaking
   /// @return always 0 (for extension)
   ///
   virtual int processTagBreaking(
         int source,    ///< source rank
         int tag        ///< TagBreaking
         );

   ///
   /// process TagGivenGapIsReached
   /// @return always 0 (for extension)
   ///
   virtual int processTagGivenGapIsReached(
         int source,    ///< source rank
         int tag        ///< TagGivenGapIsReached
         );

   ///
   /// process TagTestDualBoundGain
   /// @return always 0 (for extension)
   ///
   virtual int processTagTestDualBoundGain(
         int source,    ///< source rank
         int tag        ///< TagTestDualBoundGain
         );

   ///
   /// process TagNoTestDualBoundGain
   /// @return always 0 (for extension)
   ///
   virtual int processTagNoTestDualBoundGain(
         int source,    ///< source rank
         int tag        ///< TagNoTestDualBoundGain
         );

   ///
   /// process TagNoWaitModeSend
   /// @return always 0 (for extension)
   ///
   virtual int processTagNoWaitModeSend(
         int source,    ///< source rank
         int tag        ///< TagNoWaitModeSend
         );

   ///
   /// process TagRestart
   /// @return always 0 (for extension)
   ///
   virtual int processTagRestart(
         int source,    ///< source rank
         int tag        ///< TagRestart
         );

   ///
   /// process TagLbBoundTightened
   /// @return always 0 (for extension)
   ///
   virtual int processTagLbBoundTightened(
         int source,    ///< source rank
         int tag        ///< TagLbBoundTightened
         );

   ///
   /// process TagUbBoundTightened
   /// @return always 0 (for extension)
   ///
   virtual int processTagUbBoundTightened(
         int source,    ///< source rank
         int tag        ///< TagUbBoundTightened
         );

   ///
   /// process TagCutOffValue
   /// @return always 0 (for extension)
   ///
   virtual int processTagCutOffValue(
         int source,    ///< source rank
         int tag        ///< TagCutOffValue
         );

   ///
   /// process TagKeepRacing
   /// @return always 0 (for extension)
   ///
   virtual int processTagKeepRacing(
         int source,    ///< source rank
         int tag        ///< TagChangeSearchStrategy
         );

   ///
   /// process TagTerminateSolvingToRestart
   /// @return always 0 (for extension)
   ///
   virtual int processTagTerminateSolvingToRestart(
        int source,    ///< source rank
        int tag        ///< TagTerminateSolvingToRestart
       );

   ///
   /// wait for receiving a new node and reactivate solver
   /// @return true if a new node is received, false esle
   ///
   virtual bool receiveNewTaskAndReactivate(
         );

   ///
   /// send completion of calculation
   ///
   virtual void sendCompletionOfCalculation(
         double stopTime       ///< stopping time
         )
   {
      std::cerr << "********** BbParaSolver does not use this function. **********" << std::endl;
      abort();
   }

   ///
   /// send completion of calculation with arguments
   ///
   virtual void sendCompletionOfCalculation(
         double stopTime,         ///< stopping time
         int tag,                 ///< message Tag
         int nSelfSplitNodesLeft  ///< number of self-split nodes left
         );

   ///
   /// send completion of calculation with arguments
   ///
   virtual void sendCompletionOfCalculationWithoutSolving(
         double stopTime,         ///< stopping time
         int tag,                 ///< message Tag
         int nSelfSplitNodesLeft  ///< number of self-split nodes left
         );

   ///
   /// update global best cutoff value
   /// @return true if the global best cutoff value was updated, false otherwise
   ///
   virtual bool updateGlobalBestCutOffValue(
         double newValue       ///< new cutoff value
         );

   ///
   /// set racing parameters
   ///
   virtual void setRacingParams(
         ParaRacingRampUpParamSet *racingParms,   ///< pointer to racing parameter set object
         bool winnerParam                         ///< indicate if the parameter set is winner one
         ) = 0;

   ///
   /// set winner racing parameters
   ///
   virtual void setWinnerRacingParams(
         ParaRacingRampUpParamSet *racingParms    ///< pointer to winner racing parameter set object
         ) = 0;

   ///
   /// create subproblem
   ///
   virtual void createSubproblem(
         ) = 0;

   ///
   /// free subproblem
   ///
   virtual void freeSubproblem(
         ) = 0;

   ///
   /// solve (sub)problem
   ///
   virtual void solve(
         ) = 0;

   ///
   /// get number of nodes solved
   /// @return the number of nodes solved
   ///
   virtual long long getNNodesSolved(
         ) = 0;

   ///
   /// get number of nodes left
   /// @return the number of nodes left
   ///
   virtual int getNNodesLeft(
         ) = 0;

   ///
   /// get dual bound value
   /// @return dual bound value
   ///
   virtual double getDualBoundValue(
         ) = 0;

   ///
   /// set original node selection strategy
   ///
   virtual void setOriginalNodeSelectionStrategy(
         ) = 0;

   ///
   /// solve to check effect of root node preprocesses
   ///
   virtual void solveToCheckEffectOfRootNodePreprocesses(
         )
   {
      /// set nSolvedWithNoPreprocesses
   }

   ///
   /// lower bound of variable tightened
   /// @return always 0 (for extension)
   ///
   virtual int lbBoundTightened(
         int source,      ///< source rank
         int tag          ///< TagLbBoundTightened
         )
   {
      return 0;
   }

   ///
   /// upper bound of variable tightened
   /// @return always 0 (for extension)
   ///
   virtual int ubBoundTightened(
         int source,      ///< source rank
         int tag          ///< TagUbBoundTightened
         )
   {
      return 0;
   }

   ///
   ///  get number of tightened variables during racing
   ///
   virtual int getNTightened(
         )
   {
      return 0;
   }

   ///
   ///  get number of tightened integral variables during racing
   ///
   virtual int getNTightenedInt(
         )
   {
      return 0;
   }

   ///
   /// change search strategy
   ///
   virtual void changeSearchStrategy(
         int searchStrategy  ///< searchStrategy == 0: original search, 1: best bound search
         )
   {
   }

   ///
   /// send Solver termination state
   ///
   virtual void sendSolverTerminationState(
         );

   ///
   /// notify Self-Split finished
   ///
   virtual void notifySelfSplitFinished(
         );

   ///
   /// restart racing
   ///
   virtual void restartRacing(
        );

   ///
   /// update global best incumbent solution
   /// @return true if the best incumbent solution was updated, false otherwise
   ///
   virtual bool updateGlobalBestIncumbentSolution(
        ParaSolution *sol     ///< pointer to new solution object
        );

   ///
   /// update global best incumbent value
   /// @return true if the best incumbent value was updated, false otherwise
   ///
   virtual bool updateGlobalBestIncumbentValue(
        double newValue       ///< new incumbent value
        );

public:

   ///
   /// constructor
   ///
   BbParaSolver(
         )
   {
      THROW_LOGICAL_ERROR1("Default constructor of BbParaSolver is called");
   }

   ///
   /// constructor
   ///
   BbParaSolver(
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
   virtual ~BbParaSolver(
         )
   {
//      int source, tag;
//      (void)paraComm->probe(&source, &tag);
//      (void)paraComm->receive( NULL, 0, ParaBYTE, source, TagTerminated);
   }

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
   /// run this Solver
   ///
   using ParaSolver::run;
   virtual void run(
         );

   ///
   /// run this Solver with ParaNode object
   ///
//   virtual void run(
//         ParaTask *paraNode    ///< pointer to ParaNode object
//         )
//   {
//      currentTask = paraNode;
//      run();
//   }

   ///
   /// run this solver with racing parameters
   ///
   virtual void run(
         ParaRacingRampUpParamSet *inRacingRampUpParamSet    ///< pointer to ParaRacingRampUpParamSet object
         )
   {
      ParaTask *rootNode = paraComm->createParaTask();
      PARA_COMM_CALL(
            rootNode->bcast(paraComm, 0)
            );
      nParaTasksReceived++;
      racingParams = inRacingRampUpParamSet;
      setRacingParams(racingParams, false);
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         do
         {
            iReceiveMessages();
         } while( !waitToken(paraComm->getRank()) );
      }
      iReceiveMessages();   // Feasible solution may be received.
      if( paraParams->getBoolParamValue(Deterministic) )
      {
         passToken(paraComm->getRank());
      }
      ParaSolver::run( rootNode );
   }

   ///
   /// the following functions may be called from callback routines of the target Solver
   ///

   ///
   /// get elapsed time of node solving
   /// @return elapsed time
   ///
   double getElapsedTimeOfNodeSolving(
         )
   {
      return (paraTimer->getElapsedTime() - paraTaskStartTime);
   }

   ///
   /// get global best dual bound value at warm start (restart)
   /// @return global best dual bound value
   ///
   double getGlobalBestDualBoundValueAtWarmStart(
         )
   {
      return globalBestDualBoundValueAtWarmStart;
   }

   ///
   /// get LoadCorrdinator best dual bound value
   /// @return LoadCoordinator best dual bound value
   ///
   double getLcBestDualBoundValue(
         )
   {
      return lcBestDualBoundValue;
   }

   ///
   /// get number of nodes to stop solving. This number is not used to decide stop solving.
   /// It is used a part of conditions.
   /// @return number of nodes to stop solving
   ///
   int getNStopSolvingMode(
         )
   {
      return dynamic_cast<BbParaParamSet *>(paraParams)->getIntParamValue(NStopSolvingMode);
   }

   ///
   /// get time to stop solving. This value is not used to decide stop solving.
   /// It is used a part of conditions.
   /// @return time to stop solving
   ///
   double getTimeStopSolvingMode(
         )
   {
      return paraParams->getRealParamValue(TimeStopSolvingMode);
   }

   ///
   /// get root node computing time
   /// @return root node computing time
   ///
   double getRootNodeTime(
         )
   {
      return rootNodeTime;
   }

   ///
   /// get bound gap for stop solving. This value is not used to decide stop solving.
   /// It is used a part of conditions.
   /// @return gap value
   ///
   double getBoundGapForStopSolving(
         )
   {
      return paraParams->getRealParamValue(BgapStopSolvingMode);
   }

   ///
   /// get bound gap for collecting mode
   /// @return gap value
   ///
   double getBoundGapForCollectingMode(
         )
   {
      return paraParams->getRealParamValue(BgapCollectingMode);
   }

   ///
   /// non-blocking receive messages
   ///
   virtual void iReceiveMessages(
         );

   ///
   /// check if global incumbent value is updated or not
   /// @return true if global incumbent value is updated
   ///
   bool isGlobalIncumbentUpdated(
         )
   {
      return globalIncumbnetValueUpdateFlag;
   }

   ///
   /// set global incumbent value is reflected
   ///
   void globalIncumbnetValueIsReflected(
         )
   {
      globalIncumbnetValueUpdateFlag = false;
   }

   ///
   /// check if racing interrupt was requested or not
   /// @return true if racing interrupt was requested, false otherwise
   ///
   bool isRacingInterruptRequested(
         )
   {
      return ( racingInterruptIsRequested || restartingRacing );
   }

   ///
   /// check if collecting interrupt (interrupt with collecting all nodes) is requested or not
   /// @return true if collecting interrupt was requested, false otherwise
   ///
   bool isCollecingInterrupt(
         )
   {
      return collectingInterrupt;
   }

   ///
   /// set root node computing time
   ///
   virtual void setRootNodeTime(
         );

   ///
   /// send solution found in this Solver
   ///
   virtual void sendLocalSolution(
         );

   ///
   /// check if a notification message needs to send or not
   /// @return true if the notification message needs to send, false otherwise
   ///
   virtual bool notificationIsNecessary(
         );

   ///
   /// send Solver state to LoadCoordinator
   ///
   virtual void sendSolverState(
         long long nNodesSolved,
         int nNodesLeft,
         double bestDualBoundValue,
         double detTime
         );

   ///
   /// check if a new ParaNode was received or not
   /// @return true if a new ParaNode was received, false otherwise
   ///
   bool newParaNodeExists(
         )
   {
      return (newTask != 0);
   }

   ///
   /// check if Solver is in collecting mode or not
   /// @return true if Solver is in collecting mode, false otherwise
   ///
   bool isInCollectingMode(
         )
   {
      return ( collectingMode || collectingManyNodes );
   }

   ///
   /// check if Solver is in aggressive collecting mode or not
   /// @return true if Solver is in aggressive collecting mode, false otherwise
   ///
   bool isAggressiveCollecting(
         )
   {
      return aggressiveCollecting;
   }

   ///
   /// check if many nodes collection was requested or not
   /// @return true if many nodes collection was requested, false otherwise
   ///
   bool isManyNodesCollectionRequested(
         )
   {
      return collectingManyNodes;
   }

   ///
   /// get threshold value to send ParaNodes to LoadCoordinator
   /// @return the number of ParaNodes
   ///
   virtual int  getThresholdValue(
         int nNodes      ///< number of processed nodes, including the focus node
         );

   ///
   /// send a branch-and-bound node as ParaNode to LoadCoordinator
   ///
   virtual void sendParaNode(
         long long n,                        ///< branch-and-bound node number in this Solver
         int depth,                          ///< depth of branch-and-bound node in this Solver
         double dualBound,                   ///< dual bound value of branch-and-bound node
         double estimateValue,               ///< estimate value of branch-and-bound node
         ParaDiffSubproblem *diffSubproblem  ///< difference between the root branch-and-bound node and transferred one
         );

   ///
   /// keep a branch-and-bound node as ParaNode to LoadCoordinator
   ///
   virtual void keepParaNode(
         long long n,                        ///< branch-and-bound node number in this Solver
         int depth,                          ///< depth of branch-and-bound node in this Solver
         double dualBound,                   ///< dual bound value of branch-and-bound node
         double estimateValue,               ///< estimate value of branch-and-bound node
         ParaDiffSubproblem *diffSubproblem  ///< difference between the root branch-and-bound node and transferred one
         );

   ///
   /// send another node request
   ///
   virtual void sendAnotherNodeRequest(
         double bestDualBoundValue          ///< best dual bound value in this Solver
         );

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
   /// get global best incumbent value
   /// @return global best incumbent value
   ///
   double getGlobalBestIncumbentValue(
         )
   {
      return globalBestIncumbentValue;
   }

   ///
   /// get current ParaNode object
   /// @return pointer to ParaNode object
   ///
   ParaTask *getCurrentNode(
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
   /// count ParaNode solved at root node in pre-check
   ///
   void countInPrecheckSolvedParaNodes(
         )
   {
      nParaNodesSolvedAtPreCheck++;
   }

   ///
   /// wait a notification id message if it is needed to synchronize with LoadCoordinaor
   ///
   virtual void waitMessageIfNecessary(
         );

   ///
   /// get number of ParaNodes already sent in a collecting mode
   /// @return the number of ParaNodes sent
   ///
   int getNSendInCollectingMode(
         )
   {
      return nSendInCollectingMode;
   }

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
//   void terminateRacing()
//   {
//      assert(racingParams);
//      delete racingParams;
//      racingParams = 0;
//      racingInterruptIsRequested = false;
//      racingIsInterrupted = false;    // rampUp message might have been received before terminate racing
//                                      // Then, this flag should be set false
//   }

   ///
   /// get global best incumbent solution
   /// @return pointer to ParaSolution object
   ///
//   ParaSolution *getGlobalBestIncumbentSolution(
//         )
//   {
//      return globalBestIncumbentSolution;
//   }

   ///
   /// check if Solver is waiting for a specific message or not
   /// @return true if Solver is waiting for a specific message, false otherwise
   ///
//   bool isWaitingForSpecificMessage(
//         )
//   {
//      return waitingSpecificMessage;
//   }

   ///
   /// check if Solver is in breaking mode
   /// @return true if Solver is in breaking mode, false otherwise
   ///
   bool isBreaking(
         )
   {
      return (nTransferLimit > 0);
   }

   ///
   /// get target bound for breaking
   /// @return target bound value
   ///
   double getTargetBound(
         )
   {
      return targetBound;
   }

   ///
   /// check if the number of ParaNodes sent is reached to transfer limit specified
   /// @return true if the number of ParaNodes sent is reached to the limit, false otherwise
   ///
   bool isTransferLimitReached(
         )
   {
      return (nTransferredNodes >= nTransferLimit);
   }

   ///
   /// reset breaking information
   ///
   void resetBreakingInfo(
         )
   {
      targetBound = -DBL_MAX;
      nTransferLimit = -1;
      nTransferredNodes = -1;
      collectingManyNodes = false;
   }

   ///
   /// check if once breaking procedure worked or not
   /// @return true if the breaking procedure worked, false otherwise
   ///
   bool isOnceBreak(
         )
   {
      return onceBreak;
   }

   ///
   /// set once braking procedure worked
   ///
   void setOnceBreak(
         )
   {
      nCollectOnce = -1;
      collectingManyNodes = true;
      onceBreak = true;
   }

   ///
   /// check if aggressive presolving is specified
   /// @return true if aggressive presolving is specified, false otherwise
   ///
   bool isAggressivePresolvingSpecified(
         )
   {
      return ( paraParams->getIntParamValue(AggressivePresolveDepth) >= 0 );
   }

   ///
   /// get depth to apply aggressive presolving
   /// @return depth to apply aggressive presolving
   ///
   int getAggresivePresolvingDepth(
         )
   {
      return paraParams->getIntParamValue(AggressivePresolveDepth);
   }

   ///
   /// get depth to stop aggressive presolving
   /// @return depth to stop aggressive presolving
   ///
   int getAggresivePresolvingStopDepth(
         )
   {
      return paraParams->getIntParamValue(AggressivePresolveStopDepth);
   }

   ///
   /// get depth of sub-MIP root node in global search tree
   /// @return depth fo sub-MIP root
   ///
   int getSubMipDepth(
         )
   {
      return dynamic_cast<BbParaNode *>(currentTask)->getDepth();
   }

   ///
   /// set counter and flag to indicate that all nodes are sent to LoadCooordinator
   ///
   void setSendBackAllNodes(
         )
   {
      nCollectOnce = -1;          // collect all
      collectingManyNodes = true;
   }

   ///
   /// check if Solver is sending all nodes to LoadCoordinaor or not
   /// @return true if Solver is sending all nodes, false otherwise
   ///
   bool isCollectingAllNodes(
         )
   {
      return( collectingManyNodes && (nCollectOnce < 0) );
   }

   ///
   /// get big dual gap subtree handling strategy
   /// @return big dual gap subtree handling strategy
   ///
   int getBigDualGapSubtreeHandlingStrategy(
         )
   {
      return paraParams->getIntParamValue(BigDualGapSubtreeHandling);
   }

   ///
   /// check if given gap is reached or not
   /// @return true if given gap is reached, false otherwise
   ///
   bool isGivenGapReached(
         )
   {
      return givenGapIsReached;
   }

   ///
   /// check if iterative break down is applied or not
   /// @return true if iterative break down is applied, false otherwise
   ///
   bool isIterativeBreakDownApplied(
         )
   {
      return paraParams->getBoolParamValue(IterativeBreakDown);
   }

   ///
   /// set sum and number of integer infeasibility
   ///
   void setII(
         double sum,    ///< sum of integer infeasibility
         int count      ///< number of integer infeasibility
         )
   {
      if( minIisum > sum ) minIisum = sum;
      if( maxIisum < sum ) maxIisum = sum;
      if( minNii > count ) minNii = count;
      if( maxNii < count ) maxNii = count;
   }

   ///
   /// set number of simplex iteration at root node
   ///
   void setRootNodeSimplexIter(
         int iter
         )
   {
      nSimplexIterRoot = iter;
   }

   ///
   /// wait token for deterministic mode
   /// @return true when token is received, false otherwise
   ///
   bool waitToken(
         int rank     ///< rank of this Solver
         )
   {
      bool result;
      double startTimeToWaitToken = paraTimer->getElapsedTime();
      result = paraComm->waitToken(rank);
      idleTimeToWaitToken += (paraTimer->getElapsedTime() - startTimeToWaitToken);
      return result;
   }

   ///
   /// pass token to the next process
   ///
   void passToken(
         int rank     ///< rank of this Solver
         )
   {
      paraComm->passToken(rank);
   }

   ///
   /// get current solving node merging status
   /// @return merging status
   ///
   int getCurrentSolivingNodeMergingStatus(
         )
   {
      return dynamic_cast<BbParaNode *>(currentTask)->getMergingStatus();
   }

   ///
   /// get initial dual bound of current solving node
   /// @return initial dual bound value
   ///
   double getCurrentSolvingNodeInitialDualBound(
         )
   {
      return dynamic_cast<BbParaNode *>(currentTask)->getInitialDualBoundValue();
   }

   ///
   /// get average dual bound gain
   /// @return average dual bound gaine
   ///
   double getAverageDualBoundGain(
         )
   {
      return averageDualBoundGain;
   }

   ///
   /// set dual bound gain is not enough
   ///
   void setNotEnoughGain(
         )
   {
      enoughGainObtained = false;
   }

   ///
   /// check if dual bound gains enough or not
   /// @return true if dual bound gains enough, false otherwise
   ///
   bool isEnoughGainObtained(
         )
   {
      return enoughGainObtained;
   }

   ///
   /// check if dual bound gain needs to be tested or not
   /// @return true if dual bound gain needs to be tested, false otherwise
   ///
   bool isDualBoundGainTestNeeded(
         )
   {
      return testDualBoundGain;
   }

   ///
   /// check if this solver is in racing ramp-up or not
   /// @return true if this solver is in racing ramp-up, false otherwise
   ///
   bool isRacingRampUp(
         )
   {
      return ( ( paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ) ||
            ( paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) );
   }

   ///
   /// check if Solver is in racing stage or not
   /// @return true if Solver is in racing stage, false otherwise
   ///
   bool isRacingStage(
         )
   {
      return (racingParams &&
            (paraParams->getIntParamValue(RampUpPhaseProcess) == 1 ||
             paraParams->getIntParamValue(RampUpPhaseProcess) == 2 ) );
   }

   ///
   /// check if Solver was terminated normally or not
   /// @return true if Solver was terminated normally, false otherwise
   ///
   virtual bool wasTerminatedNormally(
         ) = 0;

   ///
   /// write current node problem
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
   /// set light weight root node process
   ///
   virtual void setLightWeightRootNodeProcess(
         )
   {
      std::cout << "*** virtual function BbParaSolver::setLightWeightRootNodeProcess is called ***" << std::endl;
   }

   ///
   /// set original root node process
   ///
   virtual void setOriginalRootNodeProcess(
         )
   {
      std::cout << "*** virtual function BbParaSolver::setOriginalRootNodeProcess is called ***" << std::endl;
   }

   ///
   /// write subproblem
   ///
   virtual void writeSubproblem(
         ) = 0;

   ///
   /// get number of simplex iterations
   ///
   virtual long long getSimplexIter(
         ) = 0;

   ///
   /// get number of restarts
   /// (Derived class for SCIP should override this function)
   /// @return number of restarts
   ///
   virtual int getNRestarts(
         )
   {
      return 0;
   }

   ///
   /// check if base solver can generate special cut off value or not
   /// @return true if base solver can generate special cut off value, false otherwise
   ///
   virtual bool canGenerateSpecialCutOffValue(
         )
   {
      return false;
   }

   ///
   /// get cut off value
   /// @return cut off value
   ///
   double getCutOffValue(
         )
   {
      return globalBestCutOffValue;
   }

   ///
   /// update number of transferred local cuts
   ///
   void updateNTransferredLocalCuts(
         int n              ///< number of transferred local cuts to be added
         )
   {
      nTransferredLocalCuts += n;
      if( minTransferredLocalCuts > n )
      {
         minTransferredLocalCuts = n;
      }
      if( maxTransferredLocalCuts < n )
      {
         maxTransferredLocalCuts = n;
      }
   }

   ///
   /// update number of transferred benders cuts
   ///
   void updateNTransferredBendersCuts(
         int n              ///< number of transferred benders cuts to be added
         )
   {
      nTransferredBendersCuts += n;
      if( minTransferredBendersCuts > n )
      {
         minTransferredBendersCuts = n;
      }
      if( maxTransferredBendersCuts < n )
      {
         maxTransferredBendersCuts = n;
      }
   }

   ///
   /// check if another node is requested or not
   /// @return true if another node is requested, false otherwise
   ///
   bool isAnotherNodeIsRequested(
         )
   {
      return anotherNodeIsRequested;
   }

   ///
   /// get pending incumbent value
   /// @return pending incumbent value
   ///
   double getPendingIncumbentValue(
         )
   {
      return pendingIncumbentValue;
   }

   ///
   /// set keep racing value
   ///
   void setKeepRacing(
         bool value
         )
   {
      keepRacing = value;
   }

   ///
   /// get the number of nodes in slef-split node pool
   /// @return the number of self-split nodes left
   ///
   int getSelfSplitNodesLeft(
         )
   {
      return selfSplitNodePool->getNumOfNodes();
   }

   ///
   /// send improved solution if it was found in this Solver
   ///
   virtual bool sendIfImprovedSolutionWasFound(
         ParaSolution *sol       ///< solution found in this Solver
         );

   ///
   /// save improved solution if it was found in this Solver
   ///
   virtual bool saveIfImprovedSolutionWasFound(
         ParaSolution *sol       ///< solution found in this Solver
         );

   ///
   /// wait notification id message to synchronized with LoadCoordinator
   ///
   virtual void waitNotificationIdMessage(
         );

   ///
   /// wait ack completion to synchronized with LoadCoordinator
   ///
   virtual void waitAckCompletion(
         );

   ///
   /// issue interrupt to solve
   ///
   virtual void issueInterruptSolve(
         )
   {
   }

};

}

#endif // __BB_PARA_SOLVER_H__
