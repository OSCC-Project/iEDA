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

/**@file    scipParaInstanceMpi.cpp
 * @brief   ScipParaInstance extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cstring>
#include "scipParaCommMpi.h"
#include "scipParaInstanceMpi.h"

using namespace UG;
using namespace ParaSCIP;

/** create ScipInstancePrePreDatatype */
MPI_Datatype
ScipParaInstanceMpi::createDatatype1(
      )
{
   const int nBlocks = 21;
   MPI_Datatype prePreDatatype;

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
      MPI_Get_address( &lProbName, &startAddress )
   );
   displacements[0] = 0;

   MPI_CALL(
      MPI_Get_address( &nCopies, &address )
   );
   displacements[1] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &origObjSense, &address )
   );
   displacements[2] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &objScale, &address )
   );
   displacements[3] = address - startAddress;
   types[3] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &objOffset, &address )
   );
   displacements[4] = address - startAddress;
   types[4] = MPI_DOUBLE;

   MPI_CALL(
      MPI_Get_address( &nVars, &address )
   );
   displacements[5] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &varIndexRange, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &lVarNames, &address )
   );
   displacements[7] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nConss, &address )
   );
   displacements[8] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &lConsNames, &address )
   );
   displacements[9] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nLinearConss, &address )
   );
   displacements[10] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSetppcConss, &address )
   );
   displacements[11] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nLogicorConss, &address )
   );
   displacements[12] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nKnapsackConss, &address )
   );
   displacements[13] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nVarboundConss, &address )
   );
   displacements[14] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nVarBoundDisjunctionConss, &address )
   );
   displacements[15] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSos1Conss, &address )
   );
   displacements[16] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nSos2Conss, &address )
   );
   displacements[17] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &nAggregatedConss, &address )
   );
   displacements[18] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &lAggregatedVarNames, &address )
   );
   displacements[19] = address - startAddress;

   MPI_CALL(
      MPI_Get_address( &lAggregatedConsNames, &address )
   );
   displacements[20] = address - startAddress;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &prePreDatatype)
         );

   return prePreDatatype;
}

void
ScipParaInstanceMpi::allocateMemoryForDatatype2(
      )
{
   if( !lProbName )  THROW_LOGICAL_ERROR1("No problem name");
   probName = new char[lProbName+1];
   if( nVars )
   {
      varLbs = new SCIP_Real[nVars];
      varUbs = new SCIP_Real[nVars];
      objCoefs = new SCIP_Real[nVars];
      varTypes = new int[nVars];
      if(lVarNames) varNames = new char[lVarNames];
      posVarNames = new int[nVars];
   }
   if( nConss )
   {
      if( lConsNames ) consNames = new char[lConsNames];
      posConsNames = new int[nConss];
   }
   if( nLinearConss )
   {
      idxLinearConsNames = new int[nLinearConss];
      linearLhss = new SCIP_Real[nLinearConss];
      linearRhss = new SCIP_Real[nLinearConss];
      nLinearCoefs = new int[nLinearConss];
   }
   if( nSetppcConss )
   {
      idxSetppcConsNames = new int[nSetppcConss];
      nIdxSetppcVars = new int[nSetppcConss];
      setppcTypes = new int[nSetppcConss];
   }
   if( nLogicorConss )
   {
      idxLogicorConsNames = new int[nLogicorConss];
      nIdxLogicorVars = new int[nLogicorConss];
   }
   if( nKnapsackConss )
   {
      idxKnapsackConsNames = new int[nKnapsackConss];
      capacities = new SCIP_Longint[nKnapsackConss];
      nLKnapsackCoefs = new int[nKnapsackConss];
   }
   if( nVarboundConss )
   {
      idxVarboundConsNames = new int[nVarboundConss];
      varboundLhss = new SCIP_Real[nVarboundConss];
      varboundRhss = new SCIP_Real[nVarboundConss];
      idxVarboundCoefVar1s = new int[nVarboundConss];
      varboundCoef2s = new SCIP_Real[nVarboundConss];
      idxVarboundCoefVar2s = new int[nVarboundConss];
   }
   if( nVarBoundDisjunctionConss ){
      idxBoundDisjunctionConsNames = new int[nVarBoundDisjunctionConss];
      nVarsBoundDisjunction = new int[nVarBoundDisjunctionConss];

   }
   if( nSos1Conss )
   {
      idxSos1ConsNames = new int[nSos1Conss];
      nSos1Coefs = new int[nSos1Conss];
   }
   if( nSos2Conss )
   {
      idxSos2ConsNames = new int[nSos2Conss];
      nSos2Coefs = new int[nSos2Conss];
   }
   if( nAggregatedConss )
   {
      if( lAggregatedVarNames ) aggregatedVarNames = new char[lAggregatedVarNames];
      posAggregatedVarNames = new int[nAggregatedConss];
      aggregatedConsNames = new char[lAggregatedConsNames];
      posAggregatedConsNames = new int[nAggregatedConss];
      if( nAggregatedConss ) aggregatedLhsAndLhss = new SCIP_Real[nAggregatedConss];
      nAggregatedCoefs = new int [nAggregatedConss];
   }
}

/** create ScipInstancePreDatatype */
MPI_Datatype
ScipParaInstanceMpi::createDatatype2(
      bool memAllocNecessary
      )
{
   const int nBlocks = 37;
   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   if( memAllocNecessary )
   {
      allocateMemoryForDatatype2();
   }

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   int n = 0;

   MPI_CALL(
      MPI_Get_address( probName, &startAddress )
   );
   displacements[n] = 0;
   blockLengths[n] = lProbName + 1;
   types[n++] = MPI_CHAR;

   if( nVars )
   {
      MPI_CALL(
         MPI_Get_address( varLbs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVars;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( varUbs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVars;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( objCoefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVars;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( varTypes, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVars;
      types[n++] = MPI_INT;

      if( lVarNames )
      {
         MPI_CALL(
            MPI_Get_address( varNames, &address )
         );
         displacements[n] =  address - startAddress;
         blockLengths[n] = lVarNames;
         types[n++] = MPI_CHAR;
      }

      MPI_CALL(
         MPI_Get_address( posVarNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVars;
      types[n++] = MPI_INT;
   }

   if( nConss )
   {
      if( lConsNames )
      {
         MPI_CALL(
            MPI_Get_address( consNames, &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = lConsNames;
         types[n++] = MPI_CHAR;
      }
      MPI_CALL(
         MPI_Get_address( posConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nConss;
      types[n++] = MPI_INT;
   }

   if( nLinearConss )
   {
      MPI_CALL(
         MPI_Get_address( idxLinearConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLinearConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( linearLhss, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLinearConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( linearRhss, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLinearConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( nLinearCoefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLinearConss;
      types[n++] = MPI_INT;
   }

   if( nSetppcConss )
   {
      MPI_CALL(
         MPI_Get_address( idxSetppcConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSetppcConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( nIdxSetppcVars, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSetppcConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( setppcTypes, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSetppcConss;
      types[n++] = MPI_INT;
   }

   if( nLogicorConss )
   {
      MPI_CALL(
         MPI_Get_address( idxLogicorConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLogicorConss;
      types[n++] = MPI_INT;
      MPI_CALL(
         MPI_Get_address( nIdxLogicorVars, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nLogicorConss;
      types[n++] = MPI_INT;
   }

   if( nKnapsackConss )
   {
      MPI_CALL(
         MPI_Get_address( idxKnapsackConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nKnapsackConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( capacities, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nKnapsackConss;
#ifdef _ALIBABA
      types[n++] = MPI_LONG;
#else
      types[n++] = MPI_LONG_LONG;
#endif

      MPI_CALL(
         MPI_Get_address( nLKnapsackCoefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nKnapsackConss;
      types[n++] = MPI_INT;
   }

   if( nVarboundConss )
   {
      MPI_CALL(
         MPI_Get_address( idxVarboundConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( varboundLhss, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( varboundRhss, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( idxVarboundCoefVar1s, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( varboundCoef2s, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( idxVarboundCoefVar2s, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarboundConss;
      types[n++] = MPI_INT;
   }

   if( nVarBoundDisjunctionConss )
   {
      MPI_CALL(
         MPI_Get_address( idxBoundDisjunctionConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarBoundDisjunctionConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( nVarsBoundDisjunction, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nVarBoundDisjunctionConss;
      types[n++] = MPI_INT;
   }

   if( nSos1Conss )
   {
      MPI_CALL(
         MPI_Get_address( idxSos1ConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSos1Conss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( nSos1Coefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSos1Conss;
      types[n++] = MPI_INT;
   }

   if( nSos2Conss )
   {
      MPI_CALL(
         MPI_Get_address( idxSos2ConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSos2Conss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( nSos2Coefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nSos2Conss;
      types[n++] = MPI_INT;
   }

   if( nAggregatedConss )
   {
      if( lAggregatedVarNames )
      {
         MPI_CALL(
            MPI_Get_address( aggregatedVarNames, &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = lAggregatedVarNames;
         types[n++] = MPI_CHAR;
      }

      MPI_CALL(
         MPI_Get_address( posAggregatedVarNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nAggregatedConss;
      types[n++] = MPI_INT;

      if( lAggregatedConsNames )
      {
         MPI_CALL(
            MPI_Get_address( aggregatedConsNames, &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = lAggregatedConsNames;
         types[n++] = MPI_CHAR;
      }

      MPI_CALL(
         MPI_Get_address( posAggregatedConsNames, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nAggregatedConss;
      types[n++] = MPI_INT;

      MPI_CALL(
         MPI_Get_address( aggregatedLhsAndLhss, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nAggregatedConss;
      types[n++] = MPI_DOUBLE;

      MPI_CALL(
         MPI_Get_address( nAggregatedCoefs, &address )
      );
      displacements[n] = address - startAddress;
      blockLengths[n] = nAggregatedConss;
      types[n++] = MPI_INT;
   }

   MPI_CALL(
         MPI_Type_create_struct(n, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

void
ScipParaInstanceMpi::allocateMemoryForDatatype3(
      )
{
   if( nLinearConss )
   {
      linearCoefs = new SCIP_Real*[nLinearConss];
      idxLinearCoefsVars = new int*[nLinearConss];
      for(int i = 0; i < nLinearConss; i++ )
      {
         linearCoefs[i] = new  SCIP_Real[nLinearCoefs[i]];
         idxLinearCoefsVars[i] = new int[nLinearCoefs[i]];
      }
   }
   if( nSetppcConss )
   {
      idxSetppcVars = new int*[nSetppcConss];
      for(int i = 0; i < nSetppcConss; i++ )
      {
         idxSetppcVars[i] = new int[nIdxSetppcVars[i]];
      }
   }
   if( nLogicorConss )
   {
      idxLogicorVars = new int*[nLogicorConss];
      for( int i = 0; i < nLogicorConss; i++ )
      {
         idxLogicorVars[i] = new int[nIdxLogicorVars[i]];
      }
   }
   if( nKnapsackConss )
   {
      knapsackCoefs = new SCIP_Longint*[nKnapsackConss];
      idxKnapsackCoefsVars = new int*[nKnapsackConss];
      for( int i = 0; i < nKnapsackConss; i++ )
      {
         knapsackCoefs[i] = new SCIP_Longint[nLKnapsackCoefs[i]];
         idxKnapsackCoefsVars[i] = new int[nLKnapsackCoefs[i]];
      }
   }
   if( nVarBoundDisjunctionConss )
   {
      idxVarBoundDisjunction = new int*[nVarBoundDisjunctionConss];
      boundTypesBoundDisjunction = new SCIP_BOUNDTYPE*[nVarBoundDisjunctionConss];
      boundsBoundDisjunction = new SCIP_Real*[nVarBoundDisjunctionConss];
      for( int i = 0; i < nVarBoundDisjunctionConss; i++ )
      {
         idxVarBoundDisjunction[i] = new int[nVarsBoundDisjunction[i]];
         boundTypesBoundDisjunction[i] = new SCIP_BOUNDTYPE[nVarsBoundDisjunction[i]];
         boundsBoundDisjunction[i] = new SCIP_Real[nVarsBoundDisjunction[i]];
      }
   }
   if( nSos1Conss )
   {
      sos1Coefs = new SCIP_Real*[nSos1Conss];
      idxSos1CoefsVars = new int*[nSos1Conss];
      for( int i = 0; i < nSos1Conss; i++ )
      {
         sos1Coefs[i] = new SCIP_Real[nSos1Coefs[i]];
         idxSos1CoefsVars[i] = new int[nSos1Coefs[i]];
      }
   }
   if( nSos2Conss )
   {
      sos2Coefs = new SCIP_Real*[nSos2Conss];
      idxSos2CoefsVars = new int*[nSos2Conss];
      for( int i = 0; i < nSos1Conss; i++ )
      {
         sos2Coefs[i] = new SCIP_Real[nSos2Coefs[i]];
         idxSos2CoefsVars[i] = new int[nSos2Coefs[i]];
      }
   }
   if( nAggregatedConss )
   {
      aggregatedCoefs = new SCIP_Real*[nAggregatedConss];
      idxAggregatedCoefsVars = new int*[nAggregatedConss];
      for( int i = 0; i < nAggregatedConss; i++ )
      {
         aggregatedCoefs[i] = new SCIP_Real[nAggregatedCoefs[i]];
         idxAggregatedCoefsVars[i] = new int[nAggregatedCoefs[i]];
      }
   }
}

/** create ScipInstanceDatatype */
MPI_Datatype
ScipParaInstanceMpi::createDatatype3(
      bool memAllocNecessary
      )
{
   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   if( memAllocNecessary )
   {
      allocateMemoryForDatatype3();
   }

   int nArrays = 1 + 2*nLinearConss + nSetppcConss + nLogicorConss + 2*nKnapsackConss + 3*nVarBoundDisjunctionConss + 2*nSos1Conss + 2*nSos2Conss + 2*nAggregatedConss;
   int *blockLengths = new int[nArrays];
   MPI_Aint *displacements = new MPI_Aint[nArrays];
   MPI_Datatype *types = new MPI_Datatype[nArrays];

   int n = 0;

   MPI_CALL(
      MPI_Get_address( &dummyToKeepStartPos, &startAddress )
   );
   displacements[n] = 0;
   blockLengths[n] = 1;
   types[n++] = MPI_INT;

   if( nLinearConss )
   {
      for(int i = 0; i <  nLinearConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( linearCoefs[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nLinearCoefs[i];
         types[n++] = MPI_DOUBLE;
         MPI_CALL(
            MPI_Get_address( idxLinearCoefsVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nLinearCoefs[i];
         types[n++] = MPI_INT;
      }
   }

   if( nSetppcConss )
   {
      for( int i = 0; i < nSetppcConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( idxSetppcVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nIdxSetppcVars[i];
         types[n++] = MPI_INT;
      }
   }

   if( nLogicorConss )
   {
      for( int i = 0; i < nLogicorConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( idxLogicorVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nIdxLogicorVars[i];
         types[n++] = MPI_INT;
      }
   }
   if( nKnapsackConss )
   {
      for( int i = 0; i < nKnapsackConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( knapsackCoefs[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nLKnapsackCoefs[i];
#ifdef _ALIBABA
         types[n++] = MPI_LONG;
#else
         types[n++] = MPI_LONG_LONG;
#endif
         MPI_CALL(
            MPI_Get_address( idxKnapsackCoefsVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nLKnapsackCoefs[i];
         types[n++] = MPI_INT;
      }
   }
   if( nVarBoundDisjunctionConss )
   {
      for( int i = 0; i < nVarBoundDisjunctionConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( idxVarBoundDisjunction[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nVarsBoundDisjunction[i];
         types[n++] = MPI_INT;
         MPI_CALL(
            MPI_Get_address( boundTypesBoundDisjunction[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nVarsBoundDisjunction[i];
         types[n++] = MPI_INT;
         MPI_CALL(
            MPI_Get_address( boundsBoundDisjunction[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nVarsBoundDisjunction[i];
         types[n++] = MPI_DOUBLE;
      }
   }
   if( nSos1Conss )
   {
      for( int i = 0; i < nSos1Conss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( sos1Coefs[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nSos1Coefs[i];
         types[n++] = MPI_DOUBLE;
         MPI_CALL(
            MPI_Get_address( idxSos1CoefsVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nSos1Coefs[i];
         types[n++] = MPI_INT;
      }
   }
   if( nSos2Conss )
   {
      for( int i = 0; i < nSos1Conss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( sos2Coefs[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nSos2Coefs[i];
         types[n++] = MPI_DOUBLE;
         MPI_CALL(
            MPI_Get_address( idxSos2CoefsVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nSos2Coefs[i];
         types[n++] = MPI_INT;
      }
   }
   if( nAggregatedConss )
   {
      for( int i = 0; i < nAggregatedConss; i++ )
      {
         MPI_CALL(
            MPI_Get_address( aggregatedCoefs[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nAggregatedCoefs[i];
         types[n++] = MPI_DOUBLE;
         MPI_CALL(
            MPI_Get_address( idxAggregatedCoefsVars[i], &address )
         );
         displacements[n] = address - startAddress;
         blockLengths[n] = nAggregatedCoefs[i];
         types[n++] = MPI_INT;
      }
   }

   assert(n == nArrays);

   MPI_CALL(
         MPI_Type_create_struct(n, blockLengths, displacements, types, &datatype)
         );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

int
ScipParaInstanceMpi::bcast(
      ParaComm *comm,
      int root,
      int method
      )
{
   DEF_PARA_COMM( commMpi, comm);

   switch ( method )
   {
   case 0 :
   {
      MPI_Datatype datatype = createDatatype1();
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
         commMpi->ubcast(&lProbName, 1, datatype, root)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );

      if( commMpi->getRank() == root )
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
         commMpi->ubcast(probName, 1, datatype, root)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );

      if( commMpi->getRank() == root )
      {
         datatype = createDatatype3(false);
      }
      else
      {
         datatype = createDatatype3(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
         commMpi->ubcast(&dummyToKeepStartPos, 1, datatype, root)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );
      break;
   }
   case 1:
   case 2:
   {
      MPI_Datatype datatype = createDatatype1();
      MPI_CALL(
         MPI_Type_commit( &datatype )
      );
      PARA_COMM_CALL(
         commMpi->ubcast(&lProbName, 1, datatype, root)
      );
      MPI_CALL(
         MPI_Type_free( &datatype )
      );
      if( fileName )
      {
         char *probNameFromFileName;
         char *temp = new char[strlen(fileName)+1];
         (void) strcpy(temp, fileName);
         SCIPsplitFilename(temp, NULL, &probNameFromFileName, NULL, NULL);
         probName = new char[strlen(probNameFromFileName)+1];
         (void) strcpy(probName, probNameFromFileName);
         delete [] temp;
         assert(static_cast<unsigned int>(lProbName) == strlen(probName));
      }
      break;
   }
   default:
      THROW_LOGICAL_ERROR1("Undefined instance transfer method");
   }

   return 0;

}
