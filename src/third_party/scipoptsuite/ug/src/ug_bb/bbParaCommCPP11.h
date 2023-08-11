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

/**@file    paraCommCPP11.h
 * @brief   ParaComm extension for C++11 thread communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_COMM_CPP11_H__
#define __BB_PARA_COMM_CPP11_H__

#include <stdexcept>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cassert>
#include "bbParaNodeTh.h"
#include "bbParaCalculationStateTh.h"
#include "bbParaSolverStateTh.h"
#include "bbParaSolverTerminationStateTh.h"
#include "ug/paraCommCPP11.h"

namespace UG
{

///
/// bb user defined transfer data types
///
static const int BB_USER_TYPE_FIRST             = UG_USER_TYPE_LAST + 1;
static const int ParaSolverDiffParamType        = BB_USER_TYPE_FIRST + 0;
static const int BB_USER_TYPE_LAST              = BB_USER_TYPE_FIRST + 0;

class BbParaCalculationState;
class BbParaParamSet;
class BbParaSolverState;
class BbParaSolverTerminationState;
class BbParaInstance;
class BbParaDiffSubproblem;
class BbParaSolution;
class BbParaInitialStat;
class BbParaRacingRampUpParamSet;
class BbParaNode;

///
/// Communicator object for C++11 thread communications
///
class BbParaCommCPP11 : public ParaCommCPP11
{
protected:

   static const char          *tagStringTable[];               ///< tag name string table

   std::mutex                 applicationLockMutex;            ///< mutex for applications

   ///
   /// check if tag string table (for debugging) set up correctly
   /// @return true if tag string table is set up correctly, false otherwise
   ///
   virtual bool tagStringTableIsSetUpCoorectly(
         );

   ///
   /// get Tag string for debugging
   /// @return string which shows Tag
   ///
   virtual const char *getTagString(
         int tag                 /// tag to be converted to string
         );

public:

   ///
   /// constructor of ParaComCPP11
   ///
   BbParaCommCPP11(
         )
         : ParaCommCPP11()
   {
   }

   ///
   /// destructor of this communicator
   ///
   virtual ~BbParaCommCPP11(
         )
   {
   }

   ///
   /// lock UG application to synchronize with other threads
   ///
   void lockApp(
         )
   {
      applicationLockMutex.lock();
   }

   ///
   /// unlock UG application to synchronize with other threads
   ///
   void unlockApp(
         )
   {
      applicationLockMutex.unlock();
   }

   ///
   /// create ParaCalculationState object by default constructor
   /// @return pointer to ParaCalculationState object
   ///
   virtual ParaCalculationState *createParaCalculationState(
         );

   ///
   /// create ParaCalculationState object
   /// @return pointer to ParaCalculationState object
   ///
   virtual ParaCalculationState *createParaCalculationState(
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
           );

   ///
   /// create ParaNode object by default constructor
   /// @return pointer to ParaNode object
   ///
   virtual ParaTask *createParaTask(
         );

   ///
   /// create ParaNode object
   /// @return pointer to ParaNode object
   ///
   virtual ParaTask *createParaNode(
               TaskId inNodeId,                       ///< node id
               TaskId inGeneratorNodeId,              ///< generator node id
               int inDepth,                           ///< node depth in global search tree
               double inDualBoundValue,               ///< dual bound value
               double inOriginalDualBoundValue,       ///< original dual bound value
               double inEstimatedValue,               ///< estimated value
               ParaDiffSubproblem *inDiffSubproblem   ///< diff subproblem info. from the original problem
            );

   ///
   /// create ParaSolverState object by default constructor
   /// @return pointer to ParaSolverState object
   ///
   virtual ParaSolverState *createParaSolverState(
         );

   ///
   /// create ParaSolverState object
   /// @return pointer to ParaSolverState object
   ///
   virtual ParaSolverState *createParaSolverState(
               int racingStage,                    ///< if this value is 1, solver is in racing stage
               unsigned int notificationId,        ///< id for this notification
               int lcId,                           ///< lc id of current ParaNode
               int globalSubtreeId,                ///< global subtree id of current ParaNode
               long long nodesSolved,              ///< number of nodes solved
               int nodesLeft,                      ///< number of remaining nodes
               double bestDualBoundValue,          ///< best dual bound value in that of remaining nodes
               double globalBestPrimalBoundValue,  ///< global best primal bound value
               double detTime,                     ///< deterministic time, -1: should be non-deterministic
               double averageDualBoundGain         ///< average dual bound gain received
           );

   ///
   /// create ParaSolverTerminationState object by default constructor
   /// @return pointer to ParaSolverTerminationState object
   ///
   virtual ParaSolverTerminationState *createParaSolverTerminationState(
         );

   ///
   /// create ParaSolverTerminationState object
   /// @return pointer to ParaSolverTerminationState object
   ///
   virtual ParaSolverTerminationState *createParaSolverTerminationState(
               int    interrupted,                         ///< indicate that this solver is interrupted or not. 0: not interrupted, 1: interrupted
                                                           ///<                                                  2: checkpoint, 3: racing-ramp up
               int    rank,                                ///< rank of this solver
               int    totalNSolved,                        ///< accumulated number of nodes solved in this ParaSolver
               int    minNSolved,                          ///< minimum number of subtree nodes rooted from ParaNode
               int    maxNSolved,                          ///< maximum number of subtree nodes rooted from ParaNode
               int    totalNSent,                          ///< accumulated number of nodes sent from this ParaSolver
               int    totalNImprovedIncumbent,             ///< accumulated number of improvements of incumbent value in this ParaSolver
               int    nParaNodesReceived,                  ///< number of ParaNodes received in this ParaSolver
               int    nParaNodesSolved,                    ///< number of ParaNodes solved ( received ) in this ParaSolver
               int    nParaNodesSolvedAtRoot,              ///< number of ParaNodes solved at root node before sending
               int    nParaNodesSolvedAtPreCheck,          ///< number of ParaNodes solved at pre-checking of root node solvability
               int    nTransferredLocalCutsFromSolver,     ///< number of local cuts transferred from this Solver
               int    minTransferredLocalCutsFromSolver,   ///< minimum number of local cuts transferred from this Solver
               int    maxTransferredLocalCutsFromSolver,   ///< maximum number of local cuts transferred from this Solver
               int    nTransferredBendersCutsFromSolver,   ///< number of benders cuts transferred from this Solver
               int    minTransferredBendersCutsFromSolver, ///< minimum number of benders cuts transferred from this Solver
               int    maxTransferredBendersCutsFromSolver, ///< maximum number of benders cuts transferred from this Solver
               int    nTotalRestarts,                      ///< number of total restarts
               int    minRestarts,                         ///< minimum number of restarts
               int    maxRestarts,                         ///< maximum number of restarts
               int    nTightened,                          ///< number of tightened variable bounds during racing stage
               int    nTightenedInt,                       ///< number of tightened integral variable bounds during racing stage
               int    calcTerminationState,                ///< termination sate of a calculation in a Solver
               double runningTime,                         ///< this solver running time
               double idleTimeToFirstParaNode,             ///< idle time to start solving the first ParaNode
               double idleTimeBetweenParaNodes,            ///< idle time between ParaNodes processing
               double iddleTimeAfterLastParaNode,          ///< idle time after the last ParaNode was solved
               double idleTimeToWaitNotificationId,        ///< idle time to wait notification Id messages
               double idleTimeToWaitAckCompletion,         ///< idle time to wait ack completion message
               double idleTimeToWaitToken,                 ///< idle time to wait token
               double totalRootNodeTime,                   ///< total time consumed by root node processes
               double minRootNodeTime,                     ///< minimum time consumed by root node processes
               double maxRootNodeTime,                     ///< maximum time consumed by root node processes
               double detTime                              ///< deterministic time, -1: should be non-deterministic
            );

   ///
   /// create ParaRacingRampUpParamSet object by default constructor
   /// @return pointer to ParaRacingRampUpParamSet object
   ///
   virtual ParaRacingRampUpParamSet* createParaRacingRampUpParamSet(
           )
   {
      THROW_LOGICAL_ERROR1("*** createParaRacingRampUpParamSet() is called in BbParaComm class ***");
   }

};

#define DEF_BB_PARA_COMM( para_comm, comm ) UG::BbParaCommCPP11 *para_comm = dynamic_cast< UG::BbParaCommCPP11* >(comm)

}

#endif  // __BB_PARA_COMM_CPP11_H__
