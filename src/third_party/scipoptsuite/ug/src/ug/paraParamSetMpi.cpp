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

/**@file    paraParamSetMpi.cpp
 * @brief   ParaParamSet extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "paraCommMpi.h"
#include "paraParamSetMpi.h"
#include <cstring>

using namespace UG;

/** allocate memory for transfer */
void
ParaParamSetMpi::allocateMemory(
      )
{
   if( getNumBoolParams() )
   {
      // std::cout << "getNumBoolParams() = " << getNumBoolParams() << std::endl;
      boolParams = new int[getNumBoolParams()];
      boolParamValues = new char[getNumBoolParams()];
   }
   if( getNumIntParams() )
   {
      // std::cout << "getNumIntParams()  = " << getNumIntParams()  << std::endl;
      intParams = new int[getNumIntParams()];
      intParamValues = new int[getNumIntParams()];
   }
   if( getNumLongintParams() )
   {
      // std::cout << "getNumLongintParams() = " << getNumLongintParams() << std::endl;
      longintParams = new int[getNumLongintParams()];
      longintParamValues = new long long[getNumLongintParams()];
   }
   if( getNumRealParams() )
   {
      // std::cout << "getNumRealParams() = " << getNumRealParams() << std::endl;
      realParams = new int[getNumRealParams()];
      realParamValues = new double[getNumRealParams()];
   }
   if( getNumCharParams() )
   {
      // std::cout << "getNumCharParams() = " << getNumCharParams() << std::endl;
      charParams = new int[getNumCharParams()];
      charParamValues = new char[getNumCharParams()];
   }
   if( getNumStringParams() )
   {
      // std::cout << "getNumStringParams() = " << getNumStringParams() << std::endl;
      int allStringParamValusesSize = 0;
      stringParams = new int[getNumStringParams()];
      for( size_t i = ParaParamsStringFirst; i < nParaParams; i++ )
      {
         if( paraParams[i]->getType() == ParaParamTypeString )
         {
            ParaParamString *paraParamString = dynamic_cast< ParaParamString * >(paraParams[i]);
            if( !paraParamString->isDefaultValue() )
            {
               allStringParamValusesSize += (std::strlen(paraParamString->getValue()) + 1);
            }
         }
      }
      stringParamValues = new char[allStringParamValusesSize];
   }
}

/** free memory for transfer */
void
ParaParamSetMpi::freeMemory(
      )
{
   if( getNumBoolParams() )
   {
      delete [] boolParams;
      delete [] boolParamValues;
   }
   if( getNumIntParams() )
   {
      delete [] intParams;
      delete [] intParamValues;
   }
   if( getNumLongintParams() )
   {
      delete [] longintParams;
      delete [] longintParamValues;
   }
   if( getNumRealParams() )
   {
      delete [] realParams;
      delete [] realParamValues;
   }
   if( getNumCharParams() )
   {
      delete [] charParams;
      delete [] charParamValues;
   }
   if( getNumStringParams() )
   {
      delete [] stringParams;
      delete [] stringParamValues;
   }
}


/** constructor with scip */
void
ParaParamSetMpi::createDiffParams(
      )
{

   nBoolParams = 0;
   nIntParams = 0;
   nLongintParams = 0;
   nRealParams = 0;
   nCharParams = 0;
   nStringParams = 0;
   stringParamValuesSize = 0;
   int pos = 0;

   for( size_t i = 0; i < nParaParams; i++ )
   {
      switch ( paraParams[i]->getType() )
      {
         case ParaParamTypeBool:
         {
            ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool * >(paraParams[i]);
            if( !paraParamBool->isDefaultValue() )
            {
               boolParams[nBoolParams] = i;
               if( paraParamBool->getValue() )
               {
                  boolParamValues[nBoolParams] = 'T';
               }
               else
               {
                  boolParamValues[nBoolParams] = 'F';
               }
               nBoolParams++;
            }
            break;
         }
         case ParaParamTypeInt:
         {
            ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt * >(paraParams[i]);
            if( !paraParamInt->isDefaultValue() )
            {
               intParams[nIntParams] = i;
               intParamValues[nIntParams] = paraParamInt->getValue();
               nIntParams++;
            }
            break;
         }
         case ParaParamTypeLongint:
         {
            ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint * >(paraParams[i]);
            if( !paraParamLongint->isDefaultValue() )
            {
               longintParams[nLongintParams] = i;
               intParamValues[nLongintParams] = paraParamLongint->getValue();
               nLongintParams++;
            }
            break;
         }
         case ParaParamTypeReal:
         {
            ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal * >(paraParams[i]);
            if( !paraParamReal->isDefaultValue() )
            {
               realParams[nRealParams] = i;
               realParamValues[nRealParams] = paraParamReal->getValue();
               nRealParams++;
            }
            break;
         }
         case ParaParamTypeChar:
         {
            ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar * >(paraParams[i]);
            if( !paraParamChar->isDefaultValue() )
            {
               charParams[nCharParams] = i;
               charParamValues[nCharParams] = paraParamChar->getValue();
               nCharParams++;
            }
            break;
         }
         case ParaParamTypeString:
         {
            ParaParamString *paraParamString = dynamic_cast< ParaParamString * >(paraParams[i]);
            if( !paraParamString->isDefaultValue() )
            {
               stringParams[nStringParams] = i;
               std::strcpy( &stringParamValues[pos], paraParamString->getValue() );
               int len = (std::strlen(paraParamString->getValue()) + 1);
               pos += len;
               stringParamValuesSize += len;
               nStringParams++;
            }
            break;
         }
         default:
         {
            THROW_LOGICAL_ERROR1("Unknown parameter type");
         }
      }
   }
}

/** set these parameter values in scip environment */
void
ParaParamSetMpi::setDiffParams(
      )
{
   if ( nBoolParams )
   {
      for( int i = 0; i < nBoolParams; i++ )
      {
         ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool * >(paraParams[boolParams[i]]);
         if( boolParamValues[i] == 'T')
         {
            paraParamBool->setValue(true);
         }
         else
         {
            paraParamBool->setValue(false);
         }
      }
   }
   if ( nIntParams )
   {
      for( int i = 0; i < nIntParams; i++ )
      {
         ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt * >(paraParams[intParams[i]]);
         paraParamInt->setValue(intParamValues[i]);
      }
   }
   if ( nLongintParams )
   {
      for( int i = 0; i < nLongintParams; i++ )
      {
         ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint * >(paraParams[longintParams[i]]);
         paraParamLongint->setValue(longintParamValues[i]);
      }
   }
   if ( nRealParams )
   {
      for( int i = 0; i < nRealParams; i++ )
      {
         ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal * >(paraParams[realParams[i]]);
         paraParamReal->setValue(realParamValues[i]);
      }
   }
   if ( nCharParams )
   {
      for( int i = 0; i < nCharParams; i++ )
      {
         ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar * >(paraParams[charParams[i]]);
         paraParamChar->setValue(charParamValues[i]);
      }
   }
   int pos = 0;
   if ( nStringParams )
   {
      for( int i = 0; i < nStringParams; i++ )
      {
         ParaParamString *paraParamString = dynamic_cast< ParaParamString * >(paraParams[stringParams[i]]);
         char *value = new char[std::strlen(&stringParamValues[pos]) + 1];
         std::strcpy(value, &stringParamValues[pos]);
         paraParamString->setValue(value);
         pos += ( std::strlen(&stringParamValues[pos]) + 1 );
      }
   }
}

/** create Datatype1 */
MPI_Datatype
ParaParamSetMpi::createDatatype1(
      )
{
   const int nBlocks = 7;

   MPI_Datatype datatype;

   int blockLengths[nBlocks];
   MPI_Aint displacements[nBlocks];
   MPI_Datatype types[nBlocks];

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   for( int i = 0; i < nBlocks; i++ )
   {
       blockLengths[i] = 1;
       types[i] = MPI_INT;
   }

   MPI_CALL(
      MPI_Get_address( &nBoolParams, &startAddress )
   );
   displacements[0] = 0;
   MPI_CALL(
      MPI_Get_address( &nIntParams, &address )
   );
   displacements[1] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nLongintParams, &address )
   );
   displacements[2] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nRealParams, &address )
   );
   displacements[3] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nCharParams, &address )
   );
   displacements[4] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &nStringParams, &address )
   );
   displacements[5] = address - startAddress;
   MPI_CALL(
      MPI_Get_address( &stringParamValuesSize, &address )
   );
   displacements[6] = address - startAddress;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;

}

/** create Datatype2 */
MPI_Datatype
ParaParamSetMpi::createDatatype2(
      bool reallocateStringParamValues
      )
{
   const int nMaxBlocks = 13;

   MPI_Datatype datatype;

   int blockLengths[nMaxBlocks];
   MPI_Aint displacements[nMaxBlocks];
   MPI_Datatype types[nMaxBlocks];

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nBlocks = 0;

   MPI_CALL(
      MPI_Get_address( &nBoolParams, &startAddress )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = 0;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   if( nBoolParams )
   {
      MPI_CALL(
         MPI_Get_address( boolParams, &address )
      );
      blockLengths[nBlocks] = nBoolParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;

      MPI_CALL(
         MPI_Get_address( boolParamValues, &address )
      );
      blockLengths[nBlocks] = nBoolParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_CHAR;
      nBlocks++;
   }

   if( nIntParams )
   {
      MPI_CALL(
         MPI_Get_address( intParams, &address )
      );
      blockLengths[nBlocks] = nIntParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;

      MPI_CALL(
         MPI_Get_address( intParamValues, &address )
      );
      blockLengths[nBlocks] = nIntParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;
   }

   if( nLongintParams )
   {
      MPI_CALL(
         MPI_Get_address( longintParams, &address )
      );
      blockLengths[nBlocks] = nLongintParams;
      displacements[nBlocks] = address - startAddress;
#ifdef _ALIBABA
      types[nBlocks] = MPI_LONG;
#else
      types[nBlocks] = MPI_LONG_LONG;
#endif
      nBlocks++;
      MPI_CALL(
         MPI_Get_address( longintParamValues, &address )
      );
      blockLengths[nBlocks] = nLongintParams;
      displacements[nBlocks] = address - startAddress;
#ifdef _ALIBABA
      types[nBlocks] = MPI_LONG;
#else
      types[nBlocks] = MPI_LONG_LONG;
#endif
      nBlocks++;
   }

   if( nRealParams )
   {
      MPI_CALL(
         MPI_Get_address( realParams, &address )
      );
      blockLengths[nBlocks] = nRealParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;
      MPI_CALL(
         MPI_Get_address( realParamValues, &address )
      );
      blockLengths[nBlocks] = nRealParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_DOUBLE;

      nBlocks++;
   }

   if( nCharParams )
   {
      MPI_CALL(
         MPI_Get_address( charParams, &address )
      );
      blockLengths[nBlocks] = nCharParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;
      MPI_CALL(
         MPI_Get_address( charParamValues, &address )
      );
      blockLengths[nBlocks] = nCharParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_CHAR;
      nBlocks++;
   }

   if( nStringParams )
   {
      if( reallocateStringParamValues )
      {
         delete[] stringParamValues;
         stringParamValues = new char[stringParamValuesSize];
      }
      MPI_CALL(
         MPI_Get_address( stringParams, &address )
      );
      blockLengths[nBlocks] = nStringParams;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_INT;
      nBlocks++;
      MPI_CALL(
         MPI_Get_address( stringParamValues, &address )
      );
      blockLengths[nBlocks] = stringParamValuesSize;
      displacements[nBlocks] = address - startAddress;
      types[nBlocks] = MPI_CHAR;
      nBlocks++;
   }

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

int
ParaParamSetMpi::bcast(ParaComm *comm, int root)
{

   DEF_PARA_COMM( commMpi, comm);

   allocateMemory();
   if( commMpi->getRank() == root )
   {
      createDiffParams();
   }

   MPI_Datatype datatype1;
   datatype1 = createDatatype1();
   MPI_CALL(
      MPI_Type_commit( &datatype1 )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&nBoolParams, 1, datatype1, root)
   );
   MPI_CALL(
      MPI_Type_free( &datatype1 )
   );

   MPI_Datatype datatype2;
   if( commMpi->getRank() == root )
   {
      datatype2 = createDatatype2(false);
   }
   else
   {
      datatype2 = createDatatype2(true);
   }
   MPI_CALL(
      MPI_Type_commit( &datatype2 )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&nBoolParams, 1, datatype2, root)  // nBoolParams sending twice, to fix start point
   );
   MPI_CALL(
      MPI_Type_free( &datatype2 )
   );
   if( commMpi->getRank() != root )
   {
      setDiffParams();
   }
   freeMemory();

   return 0;

}
