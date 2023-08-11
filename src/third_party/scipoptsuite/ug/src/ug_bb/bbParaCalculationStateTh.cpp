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

/**@file    paraCalculationStateTh.cpp
 * @brief   CalcutationStte object extension for threads communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bbParaComm.h"
#include "bbParaCalculationStateTh.h"

using namespace UG;

BbParaCalculationStateTh*
BbParaCalculationStateTh::createDatatype(
      )
{
   return new BbParaCalculationStateTh(
         compTime,
         rootTime,
         nSolved,
         nSent,
         nImprovedIncumbent,
         terminationState,
         nSolvedWithNoPreprocesses,
         nSimplexIterRoot,
         averageSimplexIter,
         nTransferredLocalCuts,
         minTransferredLocalCuts,
         maxTransferredLocalCuts,
         nTransferredBendersCuts,
         minTransferredBendersCuts,
         maxTransferredBendersCuts,
         nRestarts,
         minIisum,
         maxIisum,
         minNii,
         maxNii,
         dualBound,
         nSelfSplitNodesLeft
         );
}

void
BbParaCalculationStateTh::send(
      ParaComm *comm,
      int destination,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend(createDatatype(), ParaCalculationStateType, destination, tag)
   );
}

void
BbParaCalculationStateTh::receive(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaCalculationStateTh *received;

   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaCalculationStateType, source, tag)
   );

   compTime = received->compTime;
   rootTime = received->rootTime;
   nSolved = received->nSolved;
   nSent = received->nSent;
   nImprovedIncumbent = received->nImprovedIncumbent;
   terminationState = received->terminationState;
   nSolvedWithNoPreprocesses = received->nSolvedWithNoPreprocesses;
   nSimplexIterRoot = received->nSimplexIterRoot;
   averageSimplexIter = received->averageSimplexIter;
   nTransferredLocalCuts = received->nTransferredLocalCuts;
   minTransferredLocalCuts = received->nTransferredLocalCuts;
   maxTransferredLocalCuts = received->maxTransferredLocalCuts;
   nTransferredBendersCuts = received->nTransferredBendersCuts;
   minTransferredBendersCuts = received->nTransferredBendersCuts;
   maxTransferredBendersCuts = received->maxTransferredBendersCuts;
   nRestarts = received->nRestarts;
   minIisum = received->minIisum;
   maxIisum = received->maxIisum;
   minNii = received->minNii;
   maxNii = received->maxNii;
   dualBound = received->dualBound;
   nSelfSplitNodesLeft = received->nSelfSplitNodesLeft;

   delete received;

}
