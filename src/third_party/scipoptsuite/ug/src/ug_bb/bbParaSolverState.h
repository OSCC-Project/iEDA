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

/**@file    paraSolverState.h
 * @brief   This class has solver state to be transferred.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_SOLVER_STATE_H__
#define __BB_PARA_SOLVER_STATE_H__

#include <cfloat>
#include "ug/paraComm.h"
#include "ug/paraSolverState.h"

namespace UG
{

///
/// class BbParaSolverState
/// (ParaSolver state object for notification message)
///
class BbParaSolverState : public ParaSolverState
{

protected:

//   int          racingStage;                ///< if this value is 1, solver is in racing stage
//   unsigned int notificationId;             ///< id for this notification
//   int          lcId;                       ///< lc id of current ParaNode
//   int          globalSubtreeIdInLc;        ///< global subtree id of current ParaNode
   long long    nNodesSolved;               ///< number of nodes solved
   int          nNodesLeft;                 ///< number of remaining nodes
   double       bestDualBoundValue;         ///< best dual bound value in that of remaining nodes
   double       globalBestPrimalBoundValue; ///< global best primal bound value
//   double       detTime;                    ///< deterministic time, -1: should be non-deterministic
   double       averageDualBoundGain;       ///< average dual bound gain received

public:

   ///
   /// default constructor
   ///
   BbParaSolverState(
         )
         : ParaSolverState(),
           nNodesSolved(0),
           nNodesLeft(-1),
           bestDualBoundValue(0.0),
           globalBestPrimalBoundValue(DBL_MAX),
           // detTime(-1.0),
           averageDualBoundGain(0.0)
   {
   }

   ///
   /// constructor
   ///
   BbParaSolverState(
         int inRacingStage,                    ///< indicate if Solver is in racing stage or not
         unsigned int inNotificationId,        ///< id for this notification
         int inLcId,                           ///< lc id of current ParaNode
         int inGlobalSubtreeId,                ///< global subtree id of current ParaNode
         long long inNodesSolved,              ///< number of nodes solved
         int inNodesLeft,                      ///< number of remaining nodes
         double inBestDualBoundValue,          ///< best dual bound value in that of remaining nodes
         double inGlobalBestPrimalBoundValue,  ///< global best primal bound value
         double inDetTime,                     ///< deterministic time, -1: should be non-deterministic
         double inAverageDualBoundGain         ///< average dual bound gain received
         )
         : ParaSolverState(inRacingStage, inNotificationId, inLcId, inGlobalSubtreeId, inDetTime),
           nNodesSolved(inNodesSolved),
           nNodesLeft(inNodesLeft),
           bestDualBoundValue(inBestDualBoundValue),
           globalBestPrimalBoundValue(inGlobalBestPrimalBoundValue),
           averageDualBoundGain(inAverageDualBoundGain)
   {
   }

   ///
   /// destractor
   ///
   virtual ~BbParaSolverState(
         )
   {
   }

   ///
   /// getter of isRacingStage
   /// @return true if the Solver notified this message is in racing stage, false otherwise
   ///
   bool isRacingStage(
         )
   {
      return (racingStage == 1);
   }

   ///
   /// getter of notification id
   /// @return notification id
   ///
   unsigned int getNotificaionId(
         )
   {
      return notificationId;
   }

   ///
   /// getter of LoadCoordintor id
   /// @return LoadCoordinator id
   ///
   int getLcId(
         )
   {
      return lcId;
   }

   ///
   /// getter of global subtree id
   /// @return global subtree id
   ///
   int getGlobalSubtreeId(
         )
   {
      return globalSubtreeIdInLc;
   }

   ///
   /// gettter of best dual bound value
   /// @return best dual bound value
   ///
   double getSolverLocalBestDualBoundValue(
         )
   {
      return bestDualBoundValue;
   }

   ///
   /// get global best primal bound value that the notification Solver has
   /// @return global best primal bound value
   ///
   double getGlobalBestPrimalBoundValue(
         )
   {
      return globalBestPrimalBoundValue;
   }

   ///
   /// getter of number of nodes solved by the notification Solver
   /// @return number of nodes solved
   ///
   long long getNNodesSolved(
         )
   {
      return nNodesSolved;
   }


   ///
   /// getter of number of nodes left by the notification Solver
   /// @return number of nodes left
   ///
   int getNNodesLeft(
         )
   {
      return nNodesLeft;
   }

   ///
   /// getter of deterministic time
   /// @return deterministic time
   ///
   double getDeterministicTime(
         )
   {
      return detTime;
   }

   ///
   /// getter of average dual bound gain received
   /// @return average dual bound gain
   ///
   double getAverageDualBoundGain(
         )
   {
      return averageDualBoundGain;
   }

   ///
   /// stringfy BbParaSolverState
   /// @return string to show inside of BbParaSolverState
   ///
   std::string toString(
         )
   {
      std::ostringstream s;
      s << "racingStage = " << racingStage << ", notificationId = " << notificationId << ": ";
      s << "[" << lcId << ":" << globalSubtreeIdInLc << "]"
      << " Best dual bound value = " << bestDualBoundValue
      << " number of nodes solved = " << nNodesSolved
      << ", number of nodes left = " << nNodesLeft;
      return s.str();
   }

};

}

#endif // __BB_PARA_SOLVER_STATE_H__
