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

/**@file    paraParamSetMpi.h
 * @brief   ParaParamSet extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_PARAM_SET_MPI_H__
#define __PARA_PARAM_SET_MPI_H__
#include <mpi.h>
#include "paraCommMpi.h"
#include "paraParamSet.h"

namespace UG
{

///
/// class ParaParamSetMpi
/// @note This class is to transfer all parameters which are NOT default values
///
class ParaParamSetMpi : public ParaParamSet {

   int          nBoolParams;              ///< the number of bool parameters
   int          *boolParams;              ///< boolean parameter ids
   char         *boolParamValues;         ///< boolean parameter values

   int          nIntParams;               ///< the number of int parameters
   int          *intParams;               ///< int parameter ids
   int          *intParamValues;          ///< int parameter values

   int          nLongintParams;           ///< the number of long int parameters
   int          *longintParams;           ///< long int parameter ids
   long long    *longintParamValues;      ///< long int parameter values

   int          nRealParams;              ///< the number of real parameters
   int          *realParams;              ///< real parameter ids
   double       *realParamValues;         ///< real parameter values

   int          nCharParams;              ///< the number of char parameters
   int          *charParams;              ///< char parameter ids
   char         *charParamValues;         ///< char parameter values

   int          nStringParams;            ///< the number of string parameters
   int          *stringParams;            ///< string parameter ids
   int          stringParamValuesSize;    ///< size of stringParameterValues area
   char         *stringParamValues;       ///< string parameter values: values are concatenated

   ///
   /// allocate temporary memory for transfer
   ///
   void allocateMemory(
         );

   ///
   /// free allocated temporary memory for transfer
   ///
   void freeMemory(
         );

   ///
   /// create non default parameters for transfer
   ///
   void createDiffParams(
         );

   ///
   /// set non default parameters transferred
   ///
   void setDiffParams(
         );

   ///
   /// create ParaParamSetDatatype1
   /// @return ParaParamSetDatatype1 created
   ///
   MPI_Datatype createDatatype1(
         );

   ///
   /// create ParaParamSetDatatype2
   /// @return ParaParamSetDatatype2 created
   ///
   MPI_Datatype createDatatype2(
         bool reallocateStringPramsValue
         );

public:

   ///
   /// constructor
   ///
   ParaParamSetMpi(
         )
         : nBoolParams(0),
           boolParams(0),
           boolParamValues(0),
           nIntParams(0),
           intParams(0),
           intParamValues(0),
           nLongintParams(0),
           longintParams(0),
           longintParamValues(0),
           nRealParams(0),
           realParams(0),
           realParamValues(0),
           nCharParams(0),
           charParams(0),
           charParamValues(0),
           nStringParams(0),
           stringParams(0),
           stringParamValuesSize(0),
           stringParamValues(0)
   {
   }

   ///
   /// constructor
   ///
   ParaParamSetMpi(
         int inNParaParams
         )
         : ParaParamSet(inNParaParams),
           nBoolParams(0),
           boolParams(0),
           boolParamValues(0),
           nIntParams(0),
           intParams(0),
           intParamValues(0),
           nLongintParams(0),
           longintParams(0),
           longintParamValues(0),
           nRealParams(0),
           realParams(0),
           realParamValues(0),
           nCharParams(0),
           charParams(0),
           charParamValues(0),
           nStringParams(0),
           stringParams(0),
           stringParamValuesSize(0),
           stringParamValues(0)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamSetMpi(
         )
   {
   }

   ///
   /// broadcast ParaParams
   /// @return always 0 (for future extensions)
   ///
   int bcast(
         ParaComm *comm,     ///< communicator used
         int root            ///< root rank for broadcast
         );

};

}

#endif  // __PARA_PARAM_SET_MPI_H__
