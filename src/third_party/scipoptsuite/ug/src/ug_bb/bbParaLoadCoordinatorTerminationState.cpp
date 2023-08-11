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

/**@file    paraLoadCoordinatorTerminationState.cpp
 * @brief   Load coordinator termination state.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <sstream>
#include "bbParaLoadCoordinatorTerminationState.h"

using namespace UG;

/** stringfy ParaCalculationState */
std::string
BbParaLoadCoordinatorTerminationState::toString(
      )
{
   std::ostringstream s;
   if( isCheckpointState )
   {
      s << "######### LoadCoordinator Rank = " << rank << " is at checkpoint. #########" << std::endl;
   }
   else
   {
      s << "######### LoadCoordinator Rank = " << rank << " is terminated. #########" << std::endl;
   }

   s << "#=== The number of ParaNodes received = " << nReceived << std::endl;
   s << "#=== The number of ParaNodes sent = " << nSent << std::endl;
   s << "#=== ( # sent back immediately = " << nSentBackImmediately << ", # failed to send back = " << nFailedToSendBack << " )" << std::endl;
   s << "#=== ( # sent back immediately ( another node ) = " << nSentBackImmediatelyAnotherNode
     << ", # failed to send back ( another node ) = " << nFailedToSendBackAnotherNode << " )" << std::endl;
   s << "#=== The number of ParaNodes deleted in LoadCoordinator = " << (nDeletedInLc + nDeletedByMerging)
     << " ( by merging: " << nDeletedByMerging << " )" << std::endl;
   s << "#=== Maximum usage of node pool = " << nMaxUsageOfNodePool << ", initial p = " << nInitialP << ", maximum multiplier = " << mMaxCollectingNodes << std::endl;
   if( nNodesInNodePool )
   {
      s << "#=== LoadCoodibator NodePool is not empty. "
         <<  nNodesInNodePool  << " nodes remained." << std::endl;
   }
   if( nNodesLeftInAllSolvers )
   {
      s << "#=== Solvers have nodes. "
        << nNodesLeftInAllSolvers << " nodes left in Solvers." << std::endl;
   }
   if( isCheckpointState )
   {
      s << "#=== Current global best dual bound value = " <<  externalGlobalBestDualBoundValue
        << "( internal value = " << globalBestDualBoundValue << " )" << std::endl;
      s << "#=== Idle time to checkpoint of this LoadCoordinator  = " << idleTime << std::endl;
      s << "#=== Elapsed time to checkpoint of this LoadCoordinator  = " << runningTime << std::endl;
   }
   else
   {
      s << "#=== Idle time to terminate this LoadCoordinator  = " << idleTime << std::endl;
      s << "#=== Elapsed time to terminate this LoadCoordinator  = " << runningTime << std::endl;
   }
   s << "#=== Time used for merging nodes: add = " << addingNodeToMergeStructTime
         << ", gen. = " << generateMergeNodesCandidatesTime
         << ", regen. = " << regenerateMergeNodesCandidatesTime
         << ", merge = " << mergeNodeTime << std::endl;
   return s.str();
}

#ifdef UG_WITH_ZLIB
void
BbParaLoadCoordinatorTerminationState::write(
      gzstream::ogzstream &out
      )
{
   out.write((char *)&isCheckpointState, sizeof(bool));
   out.write((char *)&rank, sizeof(int));
   out.write((char *)&nWarmStart, sizeof(unsigned long long));
   out.write((char *)&nSent, sizeof(unsigned long long));
   out.write((char *)&nSentBackImmediately, sizeof(unsigned long long));
   out.write((char *)&nSentBackImmediatelyAnotherNode, sizeof(unsigned long long));
   out.write((char *)&nReceived, sizeof(unsigned long long));
   out.write((char *)&nDeletedInLc, sizeof(unsigned long long));
   out.write((char *)&nFailedToSendBack, sizeof(unsigned long long));
   out.write((char *)&nFailedToSendBackAnotherNode, sizeof(unsigned long long));
   out.write((char *)&nMaxUsageOfNodePool, sizeof(unsigned long long));
   out.write((char *)&nNodesInNodePool, sizeof(unsigned long long));
   out.write((char *)&nNodesLeftInAllSolvers, sizeof(unsigned long long));
   out.write((char *)&globalBestDualBoundValue, sizeof(double));
   out.write((char *)&externalGlobalBestDualBoundValue, sizeof(double));
   out.write((char *)&idleTime, sizeof(double));
   out.write((char *)&runningTime, sizeof(double));
   out.write((char *)&addingNodeToMergeStructTime, sizeof(double));
   out.write((char *)&generateMergeNodesCandidatesTime, sizeof(double));
   out.write((char *)&regenerateMergeNodesCandidatesTime, sizeof(double));
   out.write((char *)&mergeNodeTime, sizeof(double));
}

bool
BbParaLoadCoordinatorTerminationState::read(
      ParaComm *comm,
      gzstream::igzstream &in
      )
{
   in.read((char *)&isCheckpointState, sizeof(bool));
   if( in.eof() ) return false;
   in.read((char *)&rank, sizeof(int));
   in.read((char *)&nWarmStart, sizeof(unsigned long long));
   in.read((char *)&nSent, sizeof(unsigned long long));
   in.read((char *)&nSentBackImmediately, sizeof(unsigned long long));
   in.read((char *)&nSentBackImmediatelyAnotherNode, sizeof(unsigned long long));
   in.read((char *)&nReceived, sizeof(unsigned long long));
   in.read((char *)&nDeletedInLc, sizeof(unsigned long long));
   in.read((char *)&nFailedToSendBack, sizeof(unsigned long long));
   in.read((char *)&nFailedToSendBackAnotherNode, sizeof(unsigned long long));
   in.read((char *)&nMaxUsageOfNodePool, sizeof(unsigned long long));
   in.read((char *)&nNodesInNodePool, sizeof(unsigned long long));
   in.read((char *)&nNodesLeftInAllSolvers, sizeof(unsigned long long));
   in.read((char *)&globalBestDualBoundValue, sizeof(double));
   in.read((char *)&externalGlobalBestDualBoundValue, sizeof(double));
   in.read((char *)&idleTime, sizeof(double));
   in.read((char *)&runningTime, sizeof(double));
   in.read((char *)&addingNodeToMergeStructTime, sizeof(double));
   in.read((char *)&generateMergeNodesCandidatesTime, sizeof(double));
   in.read((char *)&regenerateMergeNodesCandidatesTime, sizeof(double));
   in.read((char *)&mergeNodeTime, sizeof(double));
   return true;
}

#endif
