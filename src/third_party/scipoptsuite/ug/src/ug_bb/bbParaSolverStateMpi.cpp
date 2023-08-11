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

/**@file    paraSolverStateMpi.cpp
 * @brief   BbParaSolverState extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "bbParaSolverStateMpi.h"

using namespace UG;

MPI_Datatype
BbParaSolverStateMpi::createDatatype(
      )
{

   const int nBlocks = 10;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   for( int i = 0; i < nBlocks; i++ )
   {
      blockLengths[i] = 1;
      types[i] = MPI_INT;
   }

   MPI_CALL(
      MPI_Get_address( &racingStage, &startAddress )
   );
   displacements[0] = 0;

   MPI_CALL(
      MPI_Get_address( &notificationId, &address )
   );
   displacements[1] = address - startAddress;
   types[1] = MPI_UNSIGNED;

   MPI_CALL(
      MPI_Get_address( &lcId, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &globalSubtreeIdInLc, &address )
   );
   displacements[3] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nNodesSolved, &address )
   );
   displacements[4] = address - startAddress;
#ifdef _ALIBABA
   types[4] = MPI_LONG;
#else
   types[4] = MPI_LONG_LONG;
#endif

   MPI_CALL(
      MPI_Get_address( &nNodesLeft, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &bestDualBoundValue, &address )
   );
   displacements[6] = address - startAddress;
   types[6] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &globalBestPrimalBoundValue, &address )
   );
   displacements[7] = address - startAddress;
   types[7] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &detTime, &address )
   );
   displacements[8] = address - startAddress;
   types[8] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &averageDualBoundGain, &address )
   );
   displacements[9] = address - startAddress;
   types[9] = MPI_DOUBLE;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

void
BbParaSolverStateMpi::send(
      ParaComm *comm,
      int destination,
      int tag
      )
{
   assert(nNodesLeft >= 0);
   assert(bestDualBoundValue >= -1e+10);
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&racingStage, 1, datatype, destination, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}

void
BbParaSolverStateMpi::receive(
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
      commMpi->ureceive(&racingStage, 1, datatype, source, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}
