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

/**@file    paraSolverPool.h
 * @brief   Solver pool.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_SOLVER_POOL_H__
#define __PARA_SOLVER_POOL_H__
#include <cstdlib>
#include <map>
#include "paraTask.h"
#include "paraTimer.h"
#include "paraRacingRampUpParamSet.h"
#include "paraSolverTerminationState.h"
#include "paraDeterministicTimer.h"

namespace UG
{

class ParaSolverPoolElement;
typedef ParaSolverPoolElement * ParaSolverPoolElementPtr;

enum SolverStatus {Inactive, Racing, RacingEvaluation, Active, Reserved, Dead, InterruptRequested, TerminateRequested, Terminated};

#define SOLVER_POOL_INDEX( rank )   ( rank - originRank )

///
/// class ParaSolverPool
/// (Solver Pool base class)
///
class ParaSolverPool {

protected:

   int                                  originRank;                  ///< origin rank of Solvers managed by this Solver pool
   std::size_t                          nSolvers;                    ///< number of Solvers
   ParaComm                             *paraComm;                   ///< communicator
   ParaParamSet                         *paraParams;                 ///< runtime parameters for parallelization
   ParaTimer                            *paraTimer;                  ///< timer

public:

   ///
   /// constructor
   ///
   ParaSolverPool(
         int inOriginRank,            ///< origin rank of Solvers managed by this Solver pool
         ParaComm *inParaComm,        ///< communicator used
         ParaParamSet *inParaParams,  ///< paraParamSet used
         ParaTimer *inParaTimer       ///< timer used
         )
         : originRank(inOriginRank),
           paraComm(inParaComm),
           paraParams(inParaParams),
           paraTimer(inParaTimer)
   {
      nSolvers = paraComm->getSize() - inOriginRank;
   }

   ///
   ///  destructor
   ///
   virtual ~ParaSolverPool(
         )
   {
   }

   ///
   /// get number of Solvers in this Solver pool
   /// @return number of Solvers
   ///
   std::size_t getNSolvers(
         )
   {
      return nSolvers;
   }

   ///
   /// get number of active Solvers
   /// @return number of active Solvers
   ///
   virtual std::size_t getNumActiveSolvers(
         ) = 0;

   ///
   /// get number of inactive Solvers
   /// @return number of inactive Solvers
   ///
   virtual std::size_t getNumInactiveSolvers(
         ) = 0;

   ///
   /// check if the Solver specified by rank is active or not
   /// @return true if the Solver is active, false otherwise
   ///
   virtual bool isSolverActive(
         int rank     ///< rank of the Solver to be checked
         ) = 0;

   ///
   /// get current solving ParaTask in the Solver specified by rank
   /// @return pointer to ParaTask object
   ///
   virtual ParaTask *getCurrentTask(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// set the Solver specified by rank is interrupt requested
   ///
   virtual void interruptRequested(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// check if the Solver specified by rank is interrupt requested or not
   /// @return return true if the Solver is interrupt requested, false otherwise
   ///
   virtual bool isInterruptRequested(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// set the Solver specified by rank is terminate requested
   ///
   virtual void terminateRequested(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// check if the Solver specified by rank is terminate requested or not
   /// @return return true if the Solver is terminate requested, false otherwise
   ///
   virtual bool isTerminateRequested(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// set the Solver specified by rank is terminated
   ///
   virtual void terminated(
         int rank     ///< rank of the Solver
         ) = 0;

   ///
   /// check if the Solver specified by rank is terminated or not
   /// @return return true if the Solver is terminated, false otherwise
   ///
   virtual bool isTerminated(
         int rank     ///< rank of the Solver
         ) = 0;

};

///
/// class ParaRacingSolverPool
/// (Racing Solver Pool)
///
class ParaRacingSolverPool
{

protected:

   int                       winnerRank;               ///< winner rank of racing ramp-up, -1: not decided yet
   int                       originRank;               ///< origin rank of Solvers managed by this Solver pool
   int                       nSolvers;                 ///< number of Solvers
   ParaComm                  *paraComm;                ///< communicator
   ParaParamSet              *paraParams;              ///< runtime parameters for parallelization
   ParaTimer                 *paraTimer;               ///< timer used
   ParaDeterministicTimer    *paraDetTimer;            ///< deterministic timer used

public:

   ///
   /// constructor
   ///
   ParaRacingSolverPool(
         int inOriginRank,                          ///< origin rank of Solvers managed by this Solver pool
         ParaComm *inParaComm,                      ///< communicator used
         ParaParamSet *inParaParams,                ///< paraParamSet used
         ParaTimer    *inParaTimer,                 ///< timer used
         ParaDeterministicTimer *inParaDetTimer     ///< deterministic timer used
         )
         : winnerRank(-1),
           originRank(inOriginRank),
           paraComm(inParaComm),
           paraParams(inParaParams),
	        paraTimer(inParaTimer),
	        paraDetTimer(inParaDetTimer)
   {
       nSolvers = paraComm->getSize() - inOriginRank;
   }

   ///
   /// destructor
   ///
   virtual ~ParaRacingSolverPool(
         )
   {
   }

   ///
   /// get root ParaTask object of the Solver specified
   ///
   virtual ParaTask *getCurrentTask(
         int rank          ///< rank of the Solver
         ) = 0;

   ///
   /// get number of active Solvers
   /// @return number of active Solvers
   ///
   virtual std::size_t getNumActiveSolvers(
         ) = 0;


   ///
   /// get number of inactive Solvers
   /// @return number of inactive Solvers
   ///
   virtual std::size_t getNumInactiveSolvers(
         ) = 0;

   ///
   /// check if the specified Solver is active or not
   /// @return true if the specified Solver is active, false otherwise
   ///
   virtual bool isSolverActive(
         int rank        ///< rank of the Solver
         ) = 0;

   ///
   /// get winner Solver rank
   /// @return rank of the winner Solver
   ///
   int getWinner(
         )
   {
      // assert( winnerRank > 0 );
      return winnerRank;   // -1 means that winner is not decided
   }

   ///
   /// get number of Solvers in this Solver pool
   /// @return number of Solvers
   ///
   std::size_t getNSolvers(
         )
   {
      return nSolvers;
   }

};

}

#endif // __PARA_SOLVER_POOL_H__

