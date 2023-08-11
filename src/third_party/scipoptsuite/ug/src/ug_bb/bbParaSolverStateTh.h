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

/**@file    paraSolverStateTh.h
 * @brief   BbParaSolverState extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_SOLVER_STATE_TH_H__
#define __BB_PARA_SOLVER_STATE_TH_H__

#include "bbParaSolverState.h"

namespace UG
{

///
/// class BbParaSolverStateTh
/// (ParaSolver state object for notification message by thread communication)
///
class BbParaSolverStateTh : public BbParaSolverState
{
   ///
   /// create BbParaSolverStateTh datatype
   /// @return BbParaSolverStateTh dataype
   ///
   BbParaSolverStateTh *createDatatype(
         );

public:

   ///
   /// default constructor
   ///
   BbParaSolverStateTh(
         )
   {
   }

   ///
   /// constructor
   ///
   BbParaSolverStateTh(
         int          inRacingStage,                  ///< indicate if Solver is in racing stage or not
         unsigned int inNotificationId,               ///< id for this notification
         int          inLcId,                         ///< lc id of current ParaNode
         int          inGlobalSubtreeId,              ///< global subtree id of current ParaNode
         long long    inNodesSolved,                  ///< number of nodes solved
         int          inNodesLeft,                    ///< number of remaining nodes
         double       inBestDualBoundValue,           ///< best dual bound value in that of remaining nodes
         double       inGlobalBestPrimalBoundValue,   ///< global best primal bound value
         double       inDetTime,                      ///< deterministic time, -1: should be non-deterministic
         double       inAverageDualBoundGain          ///< average dual bound gain received
         )
         : BbParaSolverState(inRacingStage, inNotificationId, inLcId, inGlobalSubtreeId,
                           inNodesSolved, inNodesLeft, inBestDualBoundValue, inGlobalBestPrimalBoundValue, inDetTime, inAverageDualBoundGain)
   {
   }

   ///
   /// destractor
   ///
   ~BbParaSolverStateTh(
         )
   {
   }

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   void send(
         ParaComm *comm,        ///< communicator used
         int destination,       ///< destination rank
         int tag                ///< TagSolverState
         );

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   void receive(
         ParaComm *comm,        ///< communicator used
         int source,            ///< source rank
         int tag                ///< TagSolverState
         );

};

#define DEF_PARA_SOLVER_STATE( para_state, state ) BbParaSolverStateTh *para_state = dynamic_cast< BbParaSolverStateTh* >(state)

}

#endif // __BB_PARA_SOLVER_STATE_TH_H__

