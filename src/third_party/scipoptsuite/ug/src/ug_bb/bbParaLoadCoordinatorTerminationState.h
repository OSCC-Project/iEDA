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

/**@file    paraLoadCoordinatorTerminationState.h
 * @brief   Load coordinator termination state.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_LOADCOORDINATOR_TERMINATION_STATE_H__
#define __BB_PARA_LOADCOORDINATOR_TERMINATION_STATE_H__

#include <string>
#include <cfloat>
#include "ug/paraComm.h"
#include "ug/paraLoadCoordinatorTerminationState.h"

#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif

namespace UG
{

///
/// Class for LoadCoordinator termination state
/// which contains calculation state in a ParaLoadCoordinator
///
class BbParaLoadCoordinatorTerminationState : public ParaLoadCoordinatorTerminationState
{
public:

   // bool                  isCheckpointState;                   ///< indicate if this state is at checkpoint or not
   // int                   rank;                                ///< rank of this ParaLoadCoordinator
   ///
   /// Counters related to this ParaLoadCoordinator
   /// TODO: The numbers should be classified depending on solvers
   ///
   unsigned long long    nSentBackImmediately;                ///< number of ParaNodes sent back immediately from LC
   unsigned long long    nSentBackImmediatelyAnotherNode;     ///< number of ParaNodes sent back immediately after AnotherNode request from LC
   unsigned long long    nDeletedInLc;                        ///< number of ParaNodes deleted in LC
   unsigned long long    nDeletedByMerging;                   ///< number of ParaNodes deleted by merging
   unsigned long long    nFailedToSendBack;                   ///< number of ParaNodes failed to send back
   unsigned long long    nFailedToSendBackAnotherNode;        ///< number of ParaNodes failed to send back after AnotherNode request
   unsigned long long    nMaxUsageOfNodePool;                 ///< maximum number of ParaNodes in ParaNodePool
   unsigned long long    nInitialP;                           ///< initial p value, which indicates the number of good ParaNodes try to keep in LC
   unsigned long long    mMaxCollectingNodes;                 ///< maximum multiplier for the number of collecting nodes
   unsigned long long    nNodesInNodePool;                    ///< number of nodes in ParaNodePool
   unsigned long long    nNodesLeftInAllSolvers;              ///< number of nodes left in all Solvers
   unsigned long long    nNodesOutputLog;                     ///< count for next logging of the number of transferred ParaNodes
   double                tNodesOutputLog;                     ///< keep time for next logging of the number of transferred ParaNodes
   ///
   ///  current dual bound value
   ///
   double                globalBestDualBoundValue;            ///< global best dual bound value (internal value)
   double                externalGlobalBestDualBoundValue;    ///< global best dual bound value (external value)
   ///
   ///  times of this LoadCoordinator
   ///
   double                idleTime;                            ///< idle time of this LoadCoordinator
   double                runningTime;                         ///< this ParaLoadCoordinator running time
   ///
   /// times used for merging
   ///
   double                addingNodeToMergeStructTime;         ///< time when a ParaNode is added to merge struct
   double                generateMergeNodesCandidatesTime;    ///< time when merge ParaNode candidates are generated
   double                regenerateMergeNodesCandidatesTime;  ///< time when merge ParaNode candidates are regenerated
   double                mergeNodeTime;                       ///< time when ParaNode is merged


   ///
   /// default constructor
   ///
   BbParaLoadCoordinatorTerminationState(
         )
         : ParaLoadCoordinatorTerminationState(),
           nSentBackImmediately(0),
           nSentBackImmediatelyAnotherNode(0),
           nDeletedInLc(0),
           nDeletedByMerging(0),
           nFailedToSendBack(0),
           nFailedToSendBackAnotherNode(0),
           nMaxUsageOfNodePool(0),
           nInitialP(0),
           mMaxCollectingNodes(0),
           nNodesInNodePool(0),
           nNodesLeftInAllSolvers(0),
           nNodesOutputLog(0),
           tNodesOutputLog(0.0),
           globalBestDualBoundValue(-DBL_MAX),
           externalGlobalBestDualBoundValue(-DBL_MAX),
           idleTime(0.0),
           runningTime(0.0),
           addingNodeToMergeStructTime(0.0),
           generateMergeNodesCandidatesTime(0.0),
           regenerateMergeNodesCandidatesTime(0.0),
           mergeNodeTime(0.0)
   {
   }

   ///
   /// destructor
   ///
   virtual ~BbParaLoadCoordinatorTerminationState(
	        )
   {
   }

   ///
   /// stringfy ParaCalculationState
   /// @return string to show inside of this object
   ///
   std::string toString(
         );

#ifdef UG_WITH_ZLIB

   ///
   /// write to checkpoint file
   ///
   void write(
         gzstream::ogzstream &out              ///< gzstream for output
         );

   ///
   /// read from checkpoint file
   ///
   bool read(
         ParaComm *comm,                       ///< communicator used
         gzstream::igzstream &in               ///< gzstream for input
         );

#endif

};

}

#endif // __BB_PARA_LOADCOORDINATOR_TERMINATION_STATE_H__

