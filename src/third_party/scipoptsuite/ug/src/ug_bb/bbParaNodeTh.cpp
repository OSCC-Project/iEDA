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

/**@file    paraNodeTh.cpp
 * @brief   BbParaNode extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bbParaComm.h"
#include "bbParaNodeTh.h"

using namespace UG;

BbParaNodeTh *
BbParaNodeTh::createDatatype(
    ParaComm *comm
      )
{
   return clone(comm);
}

int
BbParaNodeTh::bcast(
      ParaComm *comm,
      int root
      )
{
   DEF_PARA_COMM( commTh, comm);

   if( commTh->getRank() == root )
   {
      for( int i = 0; i < commTh->getSize(); i++ )
      {
         if( i != root )
         {
            BbParaNodeTh *sent;
            sent = createDatatype(comm);
            assert(!(sent->mergeNodeInfo));
            sent->mergeNodeInfo = 0;
            PARA_COMM_CALL(
               commTh->uTypeSend((void *)sent, ParaTaskType, i, TagTask)
            );
         }
      }
   }
   else
   {
      BbParaNodeTh *received;
      PARA_COMM_CALL(
         commTh->uTypeReceive((void **)&received, ParaTaskType, root, TagTask)
      );
      taskId = received->taskId;
      generatorTaskId = received->generatorTaskId;
      depth = received->depth;
      dualBoundValue = received->dualBoundValue;
      initialDualBoundValue = received->initialDualBoundValue;
      estimatedValue = received->estimatedValue;
      diffSubproblemInfo = received->diffSubproblemInfo;
      if( diffSubproblemInfo )
      {
         diffSubproblem = received->diffSubproblem->clone(commTh);
      }
      basisInfo = received->basisInfo;
      mergingStatus = received->mergingStatus;
      delete received;
   }
   return 0;
}

int
BbParaNodeTh::send(
      ParaComm *comm,
      int destination
      )
{
    DEF_PARA_COMM( commTh, comm);

    BbParaNodeTh *sent;
    sent = createDatatype(comm);
    assert(!(sent->mergeNodeInfo));
    sent->mergeNodeInfo = 0;
    PARA_COMM_CALL(
       commTh->uTypeSend((void *)sent, ParaTaskType, destination, TagTask)
    );

   return 0;
}

int
BbParaNodeTh::sendNewSubtreeRoot(
      ParaComm *comm,
      int destination
      )
{

    assert( !this->isRootTask() );
    DEF_PARA_COMM( commTh, comm);

    BbParaNodeTh *sent;
    sent = createDatatype(comm);
    assert(!(sent->mergeNodeInfo));
    sent->mergeNodeInfo = 0;
    PARA_COMM_CALL(
       commTh->uTypeSend((void *)sent, ParaTaskType, destination, TagNewSubtreeRootNode)
    );

   return 0;
}

int
BbParaNodeTh::sendSubtreeRootNodeId(
      ParaComm *comm,
      int destination,              ///< destination rank
      int tag
      )
{
    DEF_PARA_COMM( commTh, comm);

    BbParaNodeTh *sent;
    sent = createDatatype(comm);
    assert(!(sent->mergeNodeInfo));
    sent->mergeNodeInfo = 0;
    PARA_COMM_CALL(
       commTh->uTypeSend((void *)sent, ParaTaskType, destination, tag)
    );

   return 0;
}

//int
//BbParaNodeTh::sendReassignSelfSplitSubtreeRoot(
//      ParaComm *comm,
//      int destination               ///< destination rank
//      )
//{
//    DEF_PARA_COMM( commTh, comm);
//
//    BbParaNodeTh *sent;
//    sent = createDatatype(comm);
//    assert(!(sent->mergeNodeInfo));
//    sent->mergeNodeInfo = 0;
//    PARA_COMM_CALL(
//       commTh->uTypeSend((void *)sent, ParaTaskType, destination, TagReassignSelfSplitSubtreeRootNode)
//    );
//
//   return 0;
//}


int
BbParaNodeTh::receive(
      ParaComm *comm,
      int source
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaNodeTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaTaskType, source, TagTask)
   );
   taskId = received->taskId;
   generatorTaskId = received->generatorTaskId;
   depth = received->depth;
   dualBoundValue = received->dualBoundValue;
   initialDualBoundValue = received->initialDualBoundValue;
   estimatedValue = received->estimatedValue;
   diffSubproblemInfo = received->diffSubproblemInfo;
   if( diffSubproblemInfo )
   {
      diffSubproblem = received->diffSubproblem->clone(commTh);
   }
   basisInfo = received->basisInfo;
   mergingStatus = received->mergingStatus;
   delete received;

   return 0;
}

int
BbParaNodeTh::receiveNewSubtreeRoot(
      ParaComm *comm,
      int source
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaNodeTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaTaskType, source, TagNewSubtreeRootNode)
   );
   taskId = received->taskId;
   generatorTaskId = received->generatorTaskId;
   depth = received->depth;
   dualBoundValue = received->dualBoundValue;
   initialDualBoundValue = received->initialDualBoundValue;
   estimatedValue = received->estimatedValue;
   diffSubproblemInfo = received->diffSubproblemInfo;
   if( diffSubproblemInfo )
   {
      diffSubproblem = received->diffSubproblem->clone(commTh);
   }
   basisInfo = received->basisInfo;
   mergingStatus = received->mergingStatus;
   delete received;

   return 0;
}

int
BbParaNodeTh::receiveSubtreeRootNodeId(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commTh, comm);

   BbParaNodeTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaTaskType, source, tag)
   );
   taskId = received->taskId;
   generatorTaskId = received->generatorTaskId;
   delete received;

   return 0;
}

//int
//BbParaNodeTh::receiveReassignSelfSplitSubtreeRoot(
//      ParaComm *comm,
//      int source
//      )
//{
//   DEF_PARA_COMM( commTh, comm);
//
//   BbParaNodeTh *received;
//   PARA_COMM_CALL(
//      commTh->uTypeReceive((void **)&received, ParaTaskType, source, TagReassignSelfSplitSubtreeRootNode)
//   );
//   taskId = received->taskId;
//   generatorTaskId = received->generatorTaskId;
//   delete received;
//
//   return 0;
//}
