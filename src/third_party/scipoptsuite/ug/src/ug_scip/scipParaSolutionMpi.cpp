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

/**@file    scipParaSolutionMpi.cpp
 * @brief   ScipParaSolution extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <mpi.h>
#include "scipParaCommMpi.h"
#include "scipParaSolutionMpi.h"

using namespace UG;
using namespace ParaSCIP;

/** create clone of this object */
ScipParaSolutionMpi *
ScipParaSolutionMpi::clone(ParaComm *comm)
{
   return( new ScipParaSolutionMpi(objectiveFunctionValue, nVars, indicesAmongSolvers, values));
}

/** create ScipDiffSubproblemPreDatatype */
MPI_Datatype
ScipParaSolutionMpi::createPreDatatype(
      )
{
   const int nBlocks = 2;
   MPI_Datatype preDatatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   for( int i = 0; i < nBlocks; i++ ){
       blockLengths[i] = 1;
   }

   MPI_CALL(
      MPI_Get_address( &objectiveFunctionValue, &startAddress )
   );
   displacements[0] = 0;
   types[0] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &nVars, &address )
   );
   displacements[1] = address - startAddress;
   types[1] = MPI_INT;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &preDatatype)
         );

   return preDatatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaSolutionMpi::createDatatype(
      bool memAllocNecessary
      )
{
   const int nBlocks = 2;

   MPI_Datatype datatype = MPI_DATATYPE_NULL;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   if( nVars )
   {
      if( memAllocNecessary )
      {
         indicesAmongSolvers = new int[nVars];
         values = new SCIP_Real[nVars];
      }

      MPI_CALL(
         MPI_Get_address( indicesAmongSolvers, &startAddress )
      );
      displacements[0] = 0;
      blockLengths[0] = nVars;
      types[0] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( values, &address )
      );
      displacements[1] = address - startAddress;
      blockLengths[1] = nVars;
      types[1] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
      );
   }
   return datatype;
}

/** send solution data to the rank */
void
ScipParaSolutionMpi::bcast(ParaComm *comm, int root)
{
   DEF_PARA_COMM( commMpi, comm);
   MPI_Datatype preDatatype;
   preDatatype = createPreDatatype();
   MPI_CALL(
      MPI_Type_commit( &preDatatype )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&objectiveFunctionValue, 1, preDatatype, root)
   );
   MPI_CALL(
      MPI_Type_free( &preDatatype )
   );

   if( nVars ){
      MPI_Datatype datatype;
      if( comm->getRank() == root )
      {
         datatype = createDatatype(false);
      }
      else
      {
         datatype = createDatatype(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(indicesAmongSolvers, 1, datatype, root)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );
   }
}


/** send solution data to the rank */
void
ScipParaSolutionMpi::send(ParaComm *comm, int destination)
{
   DEF_PARA_COMM( commMpi, comm);
   MPI_Datatype preDatatype;
   preDatatype = createPreDatatype();
   MPI_CALL(
      MPI_Type_commit( &preDatatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&objectiveFunctionValue, 1, preDatatype, destination, TagSolution)
   );
   MPI_CALL(
      MPI_Type_free( &preDatatype )
   );

   if( nVars ){
      MPI_Datatype datatype;
      datatype = createDatatype(false);
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
            commMpi->usend(indicesAmongSolvers, 1, datatype, destination, TagSolution1)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );
   }
}

/** receive solution data from the source rank */
void
ScipParaSolutionMpi::receive(ParaComm *comm, int source)
{
   DEF_PARA_COMM( commMpi, comm);
   MPI_Datatype preDatatype;
   preDatatype = createPreDatatype();
   MPI_CALL(
      MPI_Type_commit( &preDatatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&objectiveFunctionValue, 1, preDatatype, source, TagSolution)
   );
   MPI_CALL(
      MPI_Type_free( &preDatatype )
   );

   if( nVars ){
      MPI_Datatype datatype;
      datatype = createDatatype(true);
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(indicesAmongSolvers, 1, datatype, source, TagSolution1)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );
   }
}
