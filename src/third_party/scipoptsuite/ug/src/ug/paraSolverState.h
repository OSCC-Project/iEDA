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


#ifndef __PARA_SOLVER_STATE_H__
#define __PARA_SOLVER_STATE_H__

#include <cfloat>
#include "paraComm.h"

namespace UG
{

///
/// class ParaSolverState
/// (ParaSolver state object for notification message)
///
class ParaSolverState
{

protected:

   int          racingStage;                ///< if this value is 1, solver is in racing stage
   unsigned int notificationId;             ///< id for this notification
   int          lcId;                       ///< lc id of current ParaTask
   int          globalSubtreeIdInLc;        ///< global subtree id of current ParaTask
   double       detTime;                    ///< deterministic time, -1: should be non-deterministic

public:

   ///
   /// default constructor
   ///
   ParaSolverState(
         )
         : racingStage(0),
           notificationId(0),
           lcId(-1),
           globalSubtreeIdInLc(-1),
           detTime(-1.0)
   {
   }

   ///
   /// copy constructor
   ///
   ParaSolverState(
         const ParaSolverState& paraSolverState
         )
         : racingStage(paraSolverState.racingStage),
           notificationId(paraSolverState.notificationId),
           lcId(paraSolverState.lcId),
           globalSubtreeIdInLc(paraSolverState.globalSubtreeIdInLc),
           detTime(paraSolverState.detTime)
   {
   }

   ///
   /// constructor
   ///
   ParaSolverState(
         int          inRacingStage,               ///< if this value is 1, solver is in racing stage
         unsigned int inNotificationId,            ///< id for this notification
         int          inLcId,                      ///< lc id of current ParaTask
         int          inGlobalSubtreeIdInLc,       ///< global subtree id of current ParaTask
         double       inDetTime                    ///< deterministic time, -1: should be non-deterministic
         )
         : racingStage(inRacingStage),
           notificationId(inNotificationId),
           lcId(inLcId),
           globalSubtreeIdInLc(inGlobalSubtreeIdInLc),
           detTime(inDetTime)
   {
   }

   ///
   /// destractor
   ///
   virtual ~ParaSolverState(
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
   /// getter of deterministic time
   /// @return deterministic time
   ///
   double getDeterministicTime(
         )
   {
      return detTime;
   }

   ///
   /// stringfy ParaSolverState
   /// @return string to show inside of ParaSolverState
   ///
   virtual std::string toString(
         ) = 0;

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   virtual void send(
         ParaComm *comm,    ///< communicator used
         int destination,   ///< destination rank
         int tag            ///< TagSolverState
         ) = 0;

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   virtual void receive(
         ParaComm *comm,   ///< communicator used
         int source,       ///< source rank
         int tag           ///< TagSolverState
         ) = 0;

};

}

#endif // __PARA_SOLVER_STATE_H__
