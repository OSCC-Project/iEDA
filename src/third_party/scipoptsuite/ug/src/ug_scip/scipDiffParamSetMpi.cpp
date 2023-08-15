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

/**@file    scipDiffParamSetMpi.cpp
 * @brief   ScipDiffParamSet extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <string.h>
#include <cassert>
#include "scip/scip.h"
#include "scipParaTagDef.h"
#include "ug/paraCommMpi.h"
#include "scipDiffParamSetMpi.h"

using namespace UG;
using namespace ParaSCIP;

/** create scipDiffParamSetPreType */
MPI_Datatype
ScipDiffParamSetMpi::createDatatype1(
      )
{
   MPI_Datatype datatype;

   int blockLengthsPre[13];
   MPI_Aint displacementsPre[13];
   MPI_Datatype typesPre[13];

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   for( int i = 0; i < 13; i++ ){
       blockLengthsPre[i] = 1;
       typesPre[i] = MPI_INT;
   }

   MPI_CALL(
      MPI_Get_address( &numBoolParams, &startAddress )
   );
   displacementsPre[0] = 0;
   MPI_CALL(
      MPI_Get_address( &boolParamNamesSize, &address )
   );
   displacementsPre[1] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &numIntParams, &address )
   );
   displacementsPre[2] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &intParamNamesSize, &address )
   );
   displacementsPre[3] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &numLongintParams, &address )
   );
   displacementsPre[4] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &longintParamNamesSize, &address )
   );
   displacementsPre[5] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &numRealParams, &address )
   );
   displacementsPre[6] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &realParamNamesSize, &address )
   );
   displacementsPre[7] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &numCharParams, &address )
   );
   displacementsPre[8] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &charParamNamesSize, &address )
   );
   displacementsPre[9] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &numStringParams, &address )
   );
   displacementsPre[10] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &stringParamNamesSize, &address )
   );
   displacementsPre[11] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &stringParamValuesSize, &address )
   );
   displacementsPre[12] = address - startAddress;

   MPI_CALL(
         MPI_Type_create_struct(13, blockLengthsPre, displacementsPre, typesPre, &datatype)
         );

   return datatype;

}

/** create scipDiffParamSetType */
MPI_Datatype
ScipDiffParamSetMpi::createDatatype2(
      bool memAllocNecessary
      )
{
   MPI_Datatype datatype;

   int blockLengths[13];
   MPI_Aint displacements[13];
   MPI_Datatype types[13];

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   if( memAllocNecessary ){
      allocateMemoty();
   }

   int nBlocks = 0;

   /** this is dummy */
   MPI_CALL(
      MPI_Get_address( &numBoolParams, &startAddress )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = 0;
   types[nBlocks++] = MPI_INT;

   if( boolParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( boolParamNames, &address )
      );
      blockLengths[nBlocks] = boolParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( numBoolParams > 0 )
   {
      MPI_CALL(
         MPI_Get_address( boolParamValues, &address )
      );
      blockLengths[nBlocks] = numBoolParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_UNSIGNED;
   }


   if( intParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( intParamNames, &address )
      );
      blockLengths[nBlocks] = intParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }


   if( numIntParams > 0 )
   {
      MPI_CALL(
         MPI_Get_address( intParamValues, &address )
      );
      blockLengths[nBlocks] = numIntParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_INT;
   }

   if( longintParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( longintParamNames, &address )
      );
      blockLengths[nBlocks] = longintParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( numLongintParams > 0 )
   {
      MPI_CALL(
         MPI_Get_address( longintParamValues, &address )
      );
      blockLengths[nBlocks] = numLongintParams;
      displacements[nBlocks] = address - startAddress;
   #ifdef _ALIBABA
      types[nBlocks++] = MPI_LONG;
   #else
      types[nBlocks++] = MPI_LONG_LONG;
   #endif
   }

   if( realParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( realParamNames, &address )
      );
      blockLengths[nBlocks] = realParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( numRealParams > 0 )
   {
      MPI_CALL(
         MPI_Get_address( realParamValues, &address )
      );
      blockLengths[nBlocks] = numRealParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_DOUBLE;
   }

   if( charParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( charParamNames, &address )
      );
      blockLengths[nBlocks] = charParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( numCharParams > 0 )
   {
      MPI_CALL(
         MPI_Get_address( charParamValues, &address )
      );
      blockLengths[nBlocks] = numCharParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( stringParamNamesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( stringParamNames, &address )
      );
      blockLengths[nBlocks] = stringParamNamesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }

   if( stringParamValuesSize > 0 )
   {
      MPI_CALL(
         MPI_Get_address( stringParamValues, &address )
      );
      blockLengths[nBlocks] = stringParamValuesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks++] = MPI_CHAR;
   }


   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

/** send solution data to the rank */
int
ScipDiffParamSetMpi::bcast(
      ParaComm *comm,
      int root
      )
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatype = createDatatype1();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&numBoolParams, 1, datatype, root)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );

   if( comm->getRank() == root )
   {
      datatype = createDatatype2(false);
   }
   else
   {
      datatype = createDatatype2(true);
   }
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&numBoolParams, 1, datatype, root)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
   return 0;
}

/** send solution data to the rank */
int
ScipDiffParamSetMpi::send(
      ParaComm *comm,
      int dest
      )
{
   DEF_PARA_COMM( commMpi, comm);
   MPI_Datatype datatype = createDatatype1();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&numBoolParams, 1, datatype, dest, UG::TagSolverDiffParamSet)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );

   datatype = createDatatype2(false);
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->usend(&numBoolParams, 1, datatype, dest, TagSolverDiffParamSet1)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
   // std::cout << "Send Rank " << comm->getRank() << ": " << toString() << std::endl; 
   return 0;
}

 /** receive solution data from the source rank */
int
ScipDiffParamSetMpi::receive(
       ParaComm *comm,
       int source
       )
{
   DEF_PARA_COMM( commMpi, comm);
   MPI_Datatype datatype = createDatatype1();
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&numBoolParams, 1, datatype, source, UG::TagSolverDiffParamSet)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );

   datatype = createDatatype2(true);
   MPI_CALL(
      MPI_Type_commit( &datatype )
   );
   PARA_COMM_CALL(
       commMpi->ureceive(&numBoolParams, 1, datatype, source, TagSolverDiffParamSet1)
   );
   MPI_CALL(
      MPI_Type_free( &datatype )
   );
   // std::cout << "Recieve Rank " << comm->getRank() << ": " << toString() << std::endl; 
   return 0;
}
