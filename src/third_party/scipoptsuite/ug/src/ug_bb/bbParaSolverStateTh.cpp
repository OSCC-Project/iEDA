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

/**@file    paraSolverStateTh.cpp
 * @brief   BbParaSolverState extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bbParaComm.h"
#include "bbParaSolverStateTh.h"

using namespace UG;

BbParaSolverStateTh *
BbParaSolverStateTh::createDatatype(
      )
{
   return new BbParaSolverStateTh(
         racingStage,
         notificationId,
         lcId,
         globalSubtreeIdInLc,
         nNodesSolved,
         nNodesLeft,
         bestDualBoundValue,
         globalBestPrimalBoundValue,
         detTime,
         averageDualBoundGain
         );
}

void
BbParaSolverStateTh::send(
      ParaComm *comm,
      int destination,
      int tag
      )
{
   assert(nNodesLeft >= 0);
   assert(bestDualBoundValue >= -1e+10);
   DEF_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(), ParaSolverStateType, destination, tag)
   );
}

void
BbParaSolverStateTh::receive(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaSolverStateTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaSolverStateType, source, tag)
   );

   racingStage = received->racingStage;
   notificationId = received->notificationId;
   lcId = received->lcId;
   globalSubtreeIdInLc = received->globalSubtreeIdInLc;
   nNodesSolved = received->nNodesSolved;
   nNodesLeft = received->nNodesLeft;
   bestDualBoundValue = received->bestDualBoundValue;
   globalBestPrimalBoundValue = received->globalBestPrimalBoundValue;
   detTime = received->detTime;
   averageDualBoundGain = received->averageDualBoundGain;

   delete received;

}
