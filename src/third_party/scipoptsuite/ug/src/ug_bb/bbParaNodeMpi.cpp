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

/**@file    paraNodeMpi.cpp
 * @brief   BbParaNode extension for MIP communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <mpi.h>
#include "bbParaNodeMpi.h"

using namespace UG;

MPI_Datatype
BbParaNodeMpi::createDatatype(
      )
{
   const int nBlocks = 15;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   for( int i = 0; i < nBlocks; i++ ){
      blockLengths[i] = 1;
      types[i] = MPI_INT;
   }

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.lcId, &startAddress )
   );
   displacements[0] = 0;

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.globalSubtaskIdInLc, &address )
   );
   displacements[1] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.solverId, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &taskId.seqNum, &address )
   );
   displacements[3] = address - startAddress;
#ifdef _ALIBABA
   types[3] = MPI_LONG;
#else
   types[3] = MPI_LONG_LONG;
#endif

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.lcId, &address )
   );
   displacements[4] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.globalSubtaskIdInLc, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.solverId, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.seqNum, &address )
   );
   displacements[7] = address - startAddress;
#ifdef _ALIBABA
   types[7] = MPI_LONG;
#else
   types[7] = MPI_LONG_LONG;
#endif

   MPI_CALL(
      MPI_Get_address( &depth, &address )
   );
   displacements[8] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &dualBoundValue, &address )
   );
   displacements[9] = address - startAddress;
   types[9] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &initialDualBoundValue, &address )
   );
   displacements[10] = address - startAddress;
   types[10] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &estimatedValue, &address )
   );
   displacements[11] = address - startAddress;
   types[11] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &diffSubproblemInfo, &address )
   );
   displacements[12] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &basisInfo, &address )
   );
   displacements[13] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &mergingStatus, &address )
   );
   displacements[14] = address - startAddress;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

MPI_Datatype
BbParaNodeMpi::createDatatypeForNodeId(
      )
{
   const int nBlocks = 8;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   for( int i = 0; i < nBlocks; i++ ){
      blockLengths[i] = 1;
      types[i] = MPI_INT;
   }

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.lcId, &startAddress )
   );
   displacements[0] = 0;

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.globalSubtaskIdInLc, &address )
   );
   displacements[1] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &taskId.subtaskId.solverId, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &taskId.seqNum, &address )
   );
   displacements[3] = address - startAddress;
#ifdef _ALIBABA
   types[3] = MPI_LONG;
#else
   types[3] = MPI_LONG_LONG;
#endif

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.lcId, &address )
   );
   displacements[4] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.globalSubtaskIdInLc, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.subtaskId.solverId, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &generatorTaskId.seqNum, &address )
   );
   displacements[7] = address - startAddress;
#ifdef _ALIBABA
   types[7] = MPI_LONG;
#else
   types[7] = MPI_LONG_LONG;
#endif

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}


int
BbParaNodeMpi::bcast(
      ParaComm *comm,
      int root
      )
{
    DEF_PARA_COMM( commMpi, comm);

    MPI_Datatype datatype;
    datatype = createDatatype();
    MPI_CALL(
       MPI_Type_commit( &datatype )
    );
    PARA_COMM_CALL(
       commMpi->ubcast(&taskId.subtaskId.lcId, 1, datatype, root)
    );
    MPI_CALL(
       MPI_Type_free( &datatype )
    );

   // root node does not have diffSubproblem
   if( diffSubproblemInfo )
   {
      if( commMpi->getRank() != root )
      {
         diffSubproblem = commMpi->createParaDiffSubproblem();
      }
      diffSubproblem->bcast(commMpi, root);
   }
   return 0;
}

int
BbParaNodeMpi::send(
      ParaComm *comm,
      int destination
      )
{
    DEF_PARA_COMM( commMpi, comm);

    MPI_Datatype datatype;
    datatype = createDatatype();
    MPI_CALL(
       MPI_Type_commit( &datatype )
    );
    PARA_COMM_CALL(
       commMpi->usend(&taskId.subtaskId.lcId, 1, datatype, destination, TagTask)
    );
    MPI_CALL(
       MPI_Type_free( &datatype )
    );
   // root node does not have diffSubproblem
   if( diffSubproblemInfo ) diffSubproblem->send(commMpi, destination);
   return 0;
}

int
BbParaNodeMpi::sendNewSubtreeRoot(
      ParaComm *comm,
      int destination
      )
{
    DEF_PARA_COMM( commMpi, comm);

    MPI_Datatype datatype;
    datatype = createDatatype();
    MPI_CALL(
       MPI_Type_commit( &datatype )
    );
    PARA_COMM_CALL(
       commMpi->usend(&taskId.subtaskId.lcId, 1, datatype, destination, TagNewSubtreeRootNode)
    );
    MPI_CALL(
       MPI_Type_free( &datatype )
    );
   // root node does not have diffSubproblem
   if( diffSubproblemInfo ) diffSubproblem->send(commMpi, destination);
   return 0;
}

int
BbParaNodeMpi::sendSubtreeRootNodeId(
      ParaComm *comm,
      int destination,               ///< destination rank
      int tag
      )
{
    DEF_PARA_COMM( commMpi, comm);

    MPI_Datatype datatype;
    datatype = createDatatypeForNodeId();
    MPI_CALL(
       MPI_Type_commit( &datatype )
    );
    PARA_COMM_CALL(
       commMpi->usend(&taskId.subtaskId.lcId, 1, datatype, destination, tag)
    );
    MPI_CALL(
       MPI_Type_free( &datatype )
    );
   return 0;
}

//int
//BbParaNodeMpi::sendReassignSelfSplitSubtreeRoot(
//      ParaComm *comm,
//      int destination               ///< destination rank
//      )
//{
//    DEF_PARA_COMM( commMpi, comm);
//
//    MPI_Datatype datatype;
//    datatype = createDatatypeForNodeId();
//    MPI_CALL(
//       MPI_Type_commit( &datatype )
//    );
//    PARA_COMM_CALL(
//       commMpi->usend(&taskId.subtaskId.lcId, 1, datatype, destination, TagReassignSelfSplitSubtreeRootNode)
//    );
//    MPI_CALL(
//       MPI_Type_free( &datatype )
//    );
//   return 0;
//}

int
BbParaNodeMpi::receive(ParaComm *comm, int source){
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&taskId.subtaskId.lcId, 1, datatype, source, TagTask)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );

   if( diffSubproblemInfo )
   {
      diffSubproblem = commMpi->createParaDiffSubproblem();
      diffSubproblem->receive(commMpi, source);
   }

   return 0;
}

int
BbParaNodeMpi::receiveNewSubtreeRoot(
      ParaComm *comm,
      int source
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&taskId.subtaskId.lcId, 1, datatype, source, TagNewSubtreeRootNode)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );

   if( diffSubproblemInfo )
   {
      diffSubproblem = commMpi->createParaDiffSubproblem();
      diffSubproblem->receive(commMpi, source);
   }

   return 0;
}

int
BbParaNodeMpi::receiveSubtreeRootNodeId(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatypeForNodeId();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&taskId.subtaskId.lcId, 1, datatype, source, tag);
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
 
   return 0;
}


//int
//BbParaNodeMpi::receiveReassignSelfSplitSubtreeRoot(
//      ParaComm *comm,
//      int source
//      )
//{
//   DEF_PARA_COMM( commMpi, comm);
//
//   MPI_Datatype datatype;
//   datatype = createDatatypeForNodeId();
//   MPI_CALL(
//      MPI_Type_commit( &datatype )
//   );
//   PARA_COMM_CALL(
//      commMpi->ureceive(&taskId.subtaskId.lcId, 1, datatype, source, TagReassignSelfSplitSubtreeRootNode);
//   );
//   MPI_CALL(
//      MPI_Type_free( &datatype )
//   );
//
//   return 0;
//}
