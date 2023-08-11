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

/**@file    scipParaInitialStatTh.cpp
 * @brief   ScipParaInitialStat extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaCommTh.h"
#include "scipParaInitialStatTh.h"

using namespace ParaSCIP;

/** create ScipParaInitialStatTh */
UG::ParaInitialStat *
ScipParaInitialStatTh::createDatatype(
	UG::ParaComm *comm
      )
{
   return clone(comm);
}

/** send solution data to the rank */
void
ScipParaInitialStatTh::send(UG::ParaComm *comm, int destination)
{
   DEF_SCIP_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(comm), ParaInitialStatType, destination, TagInitialStat)
   );
}

/** receive solution data from the source rank */
void
ScipParaInitialStatTh::receive(UG::ParaComm *comm, int source)
{
   DEF_SCIP_PARA_COMM( commTh, comm);

   ScipParaInitialStatTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaInitialStatType, source, TagInitialStat)
   );

   maxDepth = received->maxDepth;
   maxTotalDepth = received->maxTotalDepth;
   nVarBranchStatsDown = received->nVarBranchStatsDown;
   nVarBranchStatsUp = received->nVarBranchStatsUp;
   idxLBranchStatsVarsDown = new int[nVarBranchStatsDown];
   nVarBranchingDown = new int[nVarBranchStatsDown];
   idxLBranchStatsVarsUp = new int[nVarBranchStatsUp];
   nVarBranchingUp = new int[nVarBranchStatsUp];
   downpscost = new SCIP_Real[nVarBranchStatsDown];
   downvsids = new SCIP_Real[nVarBranchStatsDown];
   downconflen = new SCIP_Real[nVarBranchStatsDown];
   downinfer = new SCIP_Real[nVarBranchStatsDown];
   downcutoff = new SCIP_Real[nVarBranchStatsDown];
   uppscost = new SCIP_Real[nVarBranchStatsUp];
   upvsids = new SCIP_Real[nVarBranchStatsUp];
   upconflen = new SCIP_Real[nVarBranchStatsUp];
   upinfer = new SCIP_Real[nVarBranchStatsUp];
   upcutoff  = new SCIP_Real[nVarBranchStatsUp];
   for( int i = 0; i < nVarBranchStatsDown; i++ )
   {
      idxLBranchStatsVarsDown[i] = received->idxLBranchStatsVarsDown[i];
      nVarBranchingDown[i] = received->nVarBranchingDown[i];
      downpscost[i] = received->downpscost[i];
      downvsids[i] = received->downvsids[i];
      downconflen[i] = received->downconflen[i];
      downinfer[i] = received->downinfer[i];
      downcutoff[i] = received->downcutoff[i];
   }
   for( int i = 0; i < nVarBranchStatsUp; i++ )
   {
      idxLBranchStatsVarsUp[i] = received->idxLBranchStatsVarsUp[i];
      nVarBranchingUp[i] = received->nVarBranchingUp[i];
      uppscost[i] = received->uppscost[i];
      upvsids[i] = received->upvsids[i];
      upconflen[i] = received->upconflen[i];
      upinfer[i] = received->upinfer[i];
      upcutoff[i] = received->upcutoff[i];
   }
   delete received;

}
