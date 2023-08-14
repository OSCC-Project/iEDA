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

/**@file    paraNodeMpi.h
 * @brief   BbParaNode extension for MIP communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_NODE_MPI_H__
#define __BB_PARA_NODE_MPI_H__

#include <mpi.h>
#include <iostream>
#include <fstream>
#include "bbParaCommMpi.h"
#include "bbParaNode.h"

namespace UG
{

///
/// class BbParaNodeMpi
///
class BbParaNodeMpi : public BbParaNode
{
   ///
   /// create BbParaNode datatype
   ///
   MPI_Datatype createDatatype(
         );

   ///
   /// create BbParaNode datatype
   ///
   MPI_Datatype createDatatypeForNodeId(
         );

public :

   ///
   /// default constructor
   ///
   BbParaNodeMpi(
         )
   {
   }

   ///
   /// constructor
   ///
   BbParaNodeMpi(
         TaskId inNodeId,                       ///< node id
         TaskId inGeneratorNodeId,              ///< generator node id
         int inDepth,                           ///< depth in global search tree
         double inDualBoundValue,               ///< dual bound value
         double inOriginalDualBoundValue,       ///< original dual bound value when the node is generated
         double inEstimatedValue,               ///< estimated value
         ParaDiffSubproblem *inDiffSubproblem   ///< pointer to ParaDiffSubproblem object
         )
         : BbParaNode(inNodeId, inGeneratorNodeId, inDepth, inDualBoundValue, inOriginalDualBoundValue, inEstimatedValue, inDiffSubproblem)
   {
   }

   ///
   /// destructor
   ///
   ~BbParaNodeMpi(
         )
   {
   }

   ///
   /// clone this BbParaNodeMpi
   /// @return pointer to cloned BbParaNodeMpi object
   ///
   BbParaNodeMpi *clone(
         ParaComm *comm         ///< communicator used
         )
   {
      if( diffSubproblem )
      {
         return ( new
            BbParaNodeMpi(taskId, generatorTaskId, depth, dualBoundValue, initialDualBoundValue,
                  initialDualBoundValue,diffSubproblem->clone(comm) ) );
      }
      else
      {
         return ( new
            BbParaNodeMpi(taskId, generatorTaskId, depth, dualBoundValue, initialDualBoundValue,
                  initialDualBoundValue, 0 ) );
      }
   }

   ///
   /// broadcast this object
   /// @return always 0 (for future extensions)
   ///
   int bcast(
         ParaComm *comm,         ///< communicator used
         int root                ///< root rank of broadcast
         );

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   int send(
         ParaComm *comm,         ///< communicator used
         int destination         ///< destination rank
         );

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   int receive(
         ParaComm *comm,         ///< communicator used
         int source              ///< source rank
         );

   ///
   /// send new subtree root node
   /// @return always 0 (for future extensions)
   ///
   int sendNewSubtreeRoot(
         ParaComm *comm,        ///< communicator used
         int destination        ///< destination rank
         );

   ///
   /// send subtree root to be removed
   /// @return always 0 (for future extensions)
   ///
   int sendSubtreeRootNodeId(
         ParaComm *comm,               ///< communicator used
         int destination,              ///< destination rank
         int tag                       ///< tag of message
         );

   ///
   /// send subtree root to be reassigned
   /// @return always 0 (for future extensions)
   ///
   int sendReassignSelfSplitSubtreeRoot(
         ParaComm *comm,               ///< communicator used
         int destination               ///< destination rank
         );

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   int receiveNewSubtreeRoot(
         ParaComm *comm,        ///< communicator used
         int source             ///< source rank
      );

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   int receiveSubtreeRootNodeId(
         ParaComm *comm,                ///< communicator used
         int source,                    ///< source rank
         int tag                        ///< tag of message
      );

   ///
   /// receive this object node Id
   /// @return always 0 (for future extensions)
   ///
   int receiveReassignSelfSplitSubtreeRoot(
         ParaComm *comm,                ///< communicator used
         int source                      ///< source rank
         );
  
};

}

#endif // __BB_PARA_NODE_MPI_H__
