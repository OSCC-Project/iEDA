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

/**@file    paraSolverTerminationStateMpi.cpp
 * @brief   BbParaSolverTerminationState extension for MIP communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "ug/paraComm.h"
#include "bbParaSolverTerminationStateMpi.h"

using namespace UG;

MPI_Datatype
BbParaSolverTerminationStateMpi::createDatatype(){

   const int nBlocks = 34;

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
      MPI_Get_address( &interrupted, &startAddress )
   );
   displacements[0] = 0;

   MPI_CALL(
      MPI_Get_address( &rank, &address )
   );
   displacements[1] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &totalNSolved, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &minNSolved, &address )
   );
   displacements[3] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &maxNSolved, &address )
   );
   displacements[4] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &totalNSent, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &totalNImprovedIncumbent, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nParaTasksReceived, &address )
   );
   displacements[7] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nParaTasksSolved, &address )
   );
   displacements[8] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nParaNodesSolvedAtRoot, &address )
   );
   displacements[9] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nParaNodesSolvedAtPreCheck, &address )
   );
   displacements[10] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nTransferredLocalCutsFromSolver, &address )
   );
   displacements[11] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &minTransferredLocalCutsFromSolver, &address )
   );
   displacements[12] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &maxTransferredLocalCutsFromSolver, &address )
   );
   displacements[13] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nTransferredBendersCutsFromSolver, &address )
   );
   displacements[14] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &minTransferredBendersCutsFromSolver, &address )
   );
   displacements[15] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &maxTransferredBendersCutsFromSolver, &address )
   );
   displacements[16] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nTotalRestarts, &address )
   );
   displacements[17] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &minRestarts, &address )
   );
   displacements[18] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &maxRestarts, &address )
   );
   displacements[19] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nTightened, &address )
   );
   displacements[20] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nTightenedInt, &address )
   );
   displacements[21] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &calcTerminationState, &address )
   );
   displacements[22] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &runningTime, &address )
   );
   displacements[23] = address - startAddress;
   types[23] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeToFirstParaTask, &address )
   );
   displacements[24] = address - startAddress;
   types[24] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeBetweenParaTasks, &address )
   );
   displacements[25] = address - startAddress;
   types[25] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeAfterLastParaTask, &address )
   );
   displacements[26] = address - startAddress;
   types[26] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeToWaitNotificationId, &address )
   );
   displacements[27] = address - startAddress;
   types[27] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeToWaitAckCompletion, &address )
   );
   displacements[28] = address - startAddress;
   types[28] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &idleTimeToWaitToken, &address )
   );
   displacements[29] = address - startAddress;
   types[29] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &totalRootNodeTime, &address )
   );
   displacements[30] = address - startAddress;
   types[30] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &minRootNodeTime, &address )
   );
   displacements[31] = address - startAddress;
   types[31] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &maxRootNodeTime, &address )
   );
   displacements[32] = address - startAddress;
   types[32] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &detTime, &address )
   );
   displacements[33] = address - startAddress;
   types[33] = MPI_DOUBLE;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;

}

void
BbParaSolverTerminationStateMpi::send(
      ParaComm *comm,
      int destination,
      int tag
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&interrupted, 1, datatype, destination, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}

void
BbParaSolverTerminationStateMpi::receive(
      ParaComm *comm,
      int source,
      int tag
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&interrupted, 1, datatype, source, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}
