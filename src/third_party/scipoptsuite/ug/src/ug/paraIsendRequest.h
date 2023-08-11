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

/**@file    ParaIsendRequest.h
 * @brief   iSend request data structure
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef __PARA_ISEND_REQUEST_H__
#define __PARA_ISEND_REQUEST_H__
#include <cassert>
#include "mpi.h"
#include "paraCommMpi.h"

namespace UG
{

class ParaIsendRequest
{
public:
   enum DataType
   {
      ParaCHAR,
      ParaSHORT,
      ParaINT,
      ParaLONG,
      ParaLONG_LONG,
      ParaSIGNED_CHAR,
      ParaUNSIGNED_CHAR,
      ParaUNSIGNED_SHORT,
      ParaUNSIGNED,
      ParaUNSIGNED_LONG,
      ParaUNSIGNED_LONG_LONG,
      ParaFLOAT,
      ParaDOUBLE,
      ParaLONG_DOUBLE,
      ParaBOOL,
      ParaBYTE
   } dataType;
   MPI_Request *req;
   union ObjectPointer {
      char *pChar;
      short *pShort;
      int *pInt;
      long *pLong;
      long long *pLongLong;
      signed char *pSignedChar;
      unsigned char *pUnsignedChar;
      unsigned short *pUnsignedShort;
      unsigned *pUnsigned;
      unsigned long *pUnsignedLong;
      unsigned long long *pUnsignedLongLong;
      float *pFloat;
      double *pDouble;
      long double *pLongDouble;
      bool *pBool;
      char *pByte;    // must be always NULL
   } objectPointer;

   ///
   /// Constructor
   ///
   ParaIsendRequest(
         ) : dataType(ParaCHAR), req(0)    /// ParaCHAR type is dummy and req = 0 means that message is not basic type
   {
   }

   ///
   /// Constructor
   ///
   ParaIsendRequest(
         DataType inDataType,
         MPI_Request *inReq,
         void *inObjectPointer
         ) : dataType(inDataType), req(inReq)
   {
      switch ( dataType )
      {
      case ParaCHAR:
      {
         objectPointer.pChar = reinterpret_cast<char *>(inObjectPointer);
         break;
      }
      case ParaSHORT:
      {
         objectPointer.pShort = reinterpret_cast<short *>(inObjectPointer);
         break;
      }
      case ParaINT:
      {
         objectPointer.pInt = reinterpret_cast<int *>(inObjectPointer);
         break;
      }
      case ParaLONG:
      {
         objectPointer.pLong = reinterpret_cast<long *>(inObjectPointer);
         break;
      }
      case ParaLONG_LONG:
      {
         objectPointer.pLongLong = reinterpret_cast<long long *>(inObjectPointer);
         break;
      }
      case ParaSIGNED_CHAR:
      {
         objectPointer.pSignedChar = reinterpret_cast<signed char *>(inObjectPointer);
         break;
      }
      case ParaUNSIGNED_CHAR:
      {
         objectPointer.pUnsignedChar = reinterpret_cast<unsigned char *>(inObjectPointer);
         break;
      }
      case ParaUNSIGNED_SHORT:
      {
         objectPointer.pUnsignedShort = reinterpret_cast<unsigned short *>(inObjectPointer);
         break;
      }
      case ParaUNSIGNED:
      {
         objectPointer.pUnsigned = reinterpret_cast<unsigned *>(inObjectPointer);
         break;
      }
      case ParaUNSIGNED_LONG:
      {
         objectPointer.pUnsignedLong = reinterpret_cast<unsigned long *>(inObjectPointer);
         break;
      }
      case ParaUNSIGNED_LONG_LONG:
      {
         objectPointer.pUnsignedLongLong = reinterpret_cast<unsigned long long *>(inObjectPointer);
         break;
      }
      case ParaFLOAT:
      {
         objectPointer.pFloat = reinterpret_cast<float *>(inObjectPointer);
         break;
      }
      case ParaDOUBLE:
      {
         objectPointer.pDouble = reinterpret_cast<double *>(inObjectPointer);
         break;
      }
      case ParaLONG_DOUBLE:
      {
         objectPointer.pLongDouble = reinterpret_cast<long double *>(inObjectPointer);
         break;
      }
      case ParaBOOL:
      {
         objectPointer.pBool = reinterpret_cast<bool *>(inObjectPointer);
         break;
      }
      case ParaBYTE:
      {
         objectPointer.pByte = reinterpret_cast<char *>(inObjectPointer);
         break;
      }
      default:
      {
         std::cerr << "Invalid dataType = " << dataType << std::endl;
         abort();
      }
      }
   }

   ///
   /// deconstructor
   /// delete the object after sending, however, only paraTask is not deleted
   /// because the object is saved in Pool after sending.
   ///
   virtual ~ParaIsendRequest(
         )
   {
      if( req  ) // req is a kind of switch to show if the Isend is for basic datatype or not
      {
         delete req;
         switch ( dataType )
         {
         case ParaCHAR:
         {
            if( objectPointer.pChar ) delete [] objectPointer.pChar;
            break;
         }
         case ParaSHORT:
         {
            if( objectPointer.pShort ) delete [] objectPointer.pShort;
            break;
         }
         case ParaINT:
         {
            if( objectPointer.pInt ) delete [] objectPointer.pInt;
            break;
         }
         case ParaLONG:
         {
            if( objectPointer.pLong ) delete [] objectPointer.pLong;
            break;
         }
         case ParaLONG_LONG:
         {
            if( objectPointer.pLongLong ) delete [] objectPointer.pLongLong;
            break;
         }
         case ParaSIGNED_CHAR:
         {
            if( objectPointer.pSignedChar ) delete [] objectPointer.pSignedChar;
            break;
         }
         case ParaUNSIGNED_CHAR:
         {
            if( objectPointer.pUnsignedChar ) delete [] objectPointer.pUnsignedChar;
            break;
         }
         case ParaUNSIGNED_SHORT:
         {
            if( objectPointer.pUnsignedShort ) delete [] objectPointer.pUnsignedShort;
            break;
         }
         case ParaUNSIGNED:
         {
            if( objectPointer.pUnsigned ) delete [] objectPointer.pUnsigned;
            break;
         }
         case ParaUNSIGNED_LONG:
         {
            if( objectPointer.pUnsignedLong ) delete [] objectPointer.pUnsignedLong;
            break;
         }
         case ParaUNSIGNED_LONG_LONG:
         {
            if( objectPointer.pUnsignedLongLong ) delete [] objectPointer.pUnsignedLongLong;
            break;
         }
         case ParaFLOAT:
         {
            if( objectPointer.pFloat ) delete [] objectPointer.pFloat;
            break;
         }
         case ParaDOUBLE:
         {
            if( objectPointer.pDouble ) delete [] objectPointer.pDouble;
            break;
         }
         case ParaLONG_DOUBLE:
         {
            if( objectPointer.pLongDouble ) delete [] objectPointer.pLongDouble;
            break;
         }
         case ParaBOOL:
         {
            if( objectPointer.pBool ) delete [] objectPointer.pBool;
            break;
         }
         case ParaBYTE:
         {
            assert( !objectPointer.pByte );
            break;
         }
         default:
         {
            std::cerr << "Invalid dataType = " << dataType << std::endl;
            abort();
         }
         }
      }
   }

   ///
   /// getter of pointer of object buffer
   ///
   virtual void* buffer(
         )
   {
      switch ( dataType )
      {
      case ParaCHAR:
      {
         return objectPointer.pChar;
      }
      case ParaSHORT:
      {
         return objectPointer.pShort;
      }
      case ParaINT:
      {
         return objectPointer.pInt;
      }
      case ParaLONG:
      {
         return objectPointer.pLong;
      }
      case ParaLONG_LONG:
      {
         return objectPointer.pLongLong;
      }
      case ParaSIGNED_CHAR:
      {
         return objectPointer.pSignedChar;
      }
      case ParaUNSIGNED_CHAR:
      {
         return objectPointer.pUnsignedChar;
      }
      case ParaUNSIGNED_SHORT:
      {
         return objectPointer.pUnsignedShort;
      }
      case ParaUNSIGNED:
      {
         return objectPointer.pUnsigned;
      }
      case ParaUNSIGNED_LONG:
      {
         return objectPointer.pUnsignedLong;
      }
      case ParaUNSIGNED_LONG_LONG:
      {
         return objectPointer.pUnsignedLongLong;
      }
      case ParaFLOAT:
      {
         return objectPointer.pFloat;
      }
      case ParaDOUBLE:
      {
         return objectPointer.pDouble;
      }
      case ParaLONG_DOUBLE:
      {
         return objectPointer.pLongDouble;
      }
      case ParaBOOL:
      {
         return objectPointer.pBool;
      }
      case ParaBYTE:
      {
         return objectPointer.pByte;
      }
      default:
      {
         std::cerr << "Invalid dataType = " << dataType << std::endl;
         abort();
      }
      }
   }

   virtual bool test(
         )
   {
      assert( req );
      int flag = 0;
      MPI_CALL(
         MPI_Test(req, &flag, MPI_STATUS_IGNORE)
      );
      if( flag ) return true;
      else return false;
   }

   virtual void wait(
         )
   {
      assert( req );
      MPI_CALL(
         MPI_Wait(req, MPI_STATUS_IGNORE)
      );
   }
};

}

#endif // __PARA_ISEND_REQUEST_H__


