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

/**@file    paraCalculationStateMpi.cpp
 * @brief   CalcutationStte object extension for MPI communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "bbParaCalculationStateMpi.h"

using namespace UG;

///
/// create MPI datatype of this object
/// @return
///
MPI_Datatype
BbParaCalculationStateMpi::createDatatype(
      )
{

   const int nBlocks = 22;

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
      MPI_Get_address( &compTime, &startAddress )
   );
   displacements[0] = 0;
   types[0] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &rootTime, &address )
   );
   displacements[1] = address - startAddress;
   types[1] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &nSolved, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSent, &address )
   );
   displacements[3] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nImprovedIncumbent, &address )
   );
   displacements[4] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &terminationState, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSolvedWithNoPreprocesses, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSimplexIterRoot, &address )
   );
   displacements[7] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &averageSimplexIter, &address )
   );
   displacements[8] = address - startAddress;
   types[8] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &nTransferredLocalCuts, &address )
   );
   displacements[9] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &minTransferredLocalCuts, &address )
   );
   displacements[10] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &maxTransferredLocalCuts, &address )
   );
   displacements[11] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nTransferredBendersCuts, &address )
   );
   displacements[12] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &minTransferredBendersCuts, &address )
   );
   displacements[13] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &maxTransferredBendersCuts, &address )
   );
   displacements[14] = address - startAddress;


   MPI_CALL(
      MPI_Get_address( &nRestarts, &address )
   );
   displacements[15] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &minIisum, &address )
   );
   displacements[16] = address - startAddress;
   types[16] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &maxIisum, &address )
   );
   displacements[17] = address - startAddress;
   types[17] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &minNii, &address )
   );
   displacements[18] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &maxNii, &address )
   );
   displacements[19] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &dualBound, &address )
   );
   displacements[20] = address - startAddress;
   types[20] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &nSelfSplitNodesLeft, &address )
   );
   displacements[21] = address - startAddress;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;

}

///
/// send this object to destination
///
void
BbParaCalculationStateMpi::send(
      ParaComm *comm,       /**< communicator used to send this object */
      int destination,      /**< destination rank to send */
      int tag               /**< tag to show this object */
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&compTime, 1, datatype, destination, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}

///
/// receive this object from source
///
void
BbParaCalculationStateMpi::receive(
      ParaComm *comm,      /**< communicator used to receive this object */
      int source,          /**< source rank to receive this object */
      int tag              /**< tag to show this object */
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype;
   datatype = createDatatype();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&compTime, 1, datatype, source, tag)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
}
