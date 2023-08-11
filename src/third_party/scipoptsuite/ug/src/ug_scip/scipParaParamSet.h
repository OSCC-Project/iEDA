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

/**@file    paraParamSet.h
 * @brief   Parameter set for UG framework.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_PARAM_SET_H__
#define __SCIP_PARA_PARAM_SET_H__
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include "ug_bb/bbParaComm.h"
#include "ug_bb/bbParaParamSet.h"

namespace ParaSCIP
{

#define SCIP_PRESOLVIG_MEMORY_FACTOR 1.5  /// This could be changed SCIP version used
#define SCIP_FIXED_MEMORY_FACTOR 0.1      /// This could be changed SCIP version used
#define SCIP_MEMORY_COPY_FACTOR 0.15      /// This could be changed SCIP version used

///
///  Bool parameters
///
static const int ScipParaParamsFirst                  = UG::BbParaParamsLast + 1;
static const int ScipParaParamsBoolFirst              = ScipParaParamsFirst;
//-------------------------------------------------------------------------
static const int RootNodeSolvabilityCheck            = ScipParaParamsBoolFirst + 0;
static const int CustomizedToSharedMemory            = ScipParaParamsBoolFirst + 1;
static const int LocalBranching                      = ScipParaParamsBoolFirst + 2;
//-------------------------------------------------------------------------
static const int ScipParaParamsBoolLast              = ScipParaParamsBoolFirst + 2;
static const int ScipParaParamsBoolN                 = ScipParaParamsBoolLast - ScipParaParamsBoolFirst + 1;
///
/// Int parameters
///
static const int ScipParaParamsIntFirst              = ScipParaParamsBoolLast  + 1;
//-------------------------------------------------------------------------
static const int AddDualBoundCons                    = ScipParaParamsIntFirst + 0;
//-------------------------------------------------------------------------
static const int ScipParaParamsIntLast               = ScipParaParamsIntFirst + 0;
static const int ScipParaParamsIntN                  = ScipParaParamsIntLast - ScipParaParamsIntFirst + 1;
///
/// Longint parameters
///
static const int ScipParaParamsLongintFirst          = ScipParaParamsIntLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int ScipParaParamsLongintLast           = ScipParaParamsLongintFirst - 1;  // No params -1
static const int ScipParaParamsLongintN              = ScipParaParamsLongintLast - ScipParaParamsLongintFirst + 1;
///
/// Real parameters
///
static const int ScipParaParamsRealFirst              = ScipParaParamsLongintLast + 1;
//-------------------------------------------------------------------------
static const int MemoryLimit                          = ScipParaParamsRealFirst + 0;
//-------------------------------------------------------------------------
static const int ScipParaParamsRealLast                = ScipParaParamsRealFirst + 0;
static const int ScipParaParamsRealN                   = ScipParaParamsRealLast - ScipParaParamsRealFirst + 1;
///
/// Char parameters
///
static const int ScipParaParamsCharFirst               = ScipParaParamsRealLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int ScipParaParamsCharLast                = ScipParaParamsCharFirst - 1;   // No params -1
static const int ScipParaParamsCharN                   = ScipParaParamsCharLast - ScipParaParamsCharFirst + 1;
///
/// String parameters
///
static const int ScipParaParamsStringFirst             = ScipParaParamsCharLast      +1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int ScipParaParamsStringLast            = ScipParaParamsStringFirst - 1;  // No params -1
static const int ScipParaParamsStringN               = ScipParaParamsStringLast - ScipParaParamsStringFirst + 1;
static const int ScipParaParamsLast                  = ScipParaParamsStringLast;
static const int ScipParaParamsSize                  = ScipParaParamsLast + 1;


class ParaComm;
///
/// class BbParaParamSet
///
class ScipParaParamSet : public UG::BbParaParamSet
{

public:

   ///
   /// constructor
   ///
   ScipParaParamSet(
         );
//
//         : UG::BbParaParamSet(ScipParaParamsSize)
//   {
//   }

   ///
   /// destructor
   ///
   virtual ~ScipParaParamSet(
         )
   {
   }

   ///
   /// get number of bool parameters
   /// @return size of parameter table
   ///
   size_t getNumBoolParams(
         )
   {
      return (UG::ParaParamsBoolN + UG::BbParaParamsBoolN + ScipParaParamsBoolN);
   }

   ///
   /// get number of int parameters
   /// @return size of parameter table
   ///
   size_t getNumIntParams(
         )
   {
      return (UG::ParaParamsIntN + UG::BbParaParamsIntN + ScipParaParamsIntN);
   }

   ///
   /// get number of longint parameters
   /// @return size of parameter table
   ///
   size_t getNumLongintParams(
         )
   {
      return (UG::ParaParamsLongintN + UG::BbParaParamsLongintN + ScipParaParamsLongintN);
   }

   ///
   /// get number of real parameters
   /// @return size of parameter table
   ///
   size_t getNumRealParams(
         )
   {
      return (UG::ParaParamsRealN + UG::BbParaParamsRealN + ScipParaParamsRealN);
   }

   ///
   /// get number of char parameters
   /// @return size of parameter table
   ///
   size_t getNumCharParams(
         )
   {
      return (UG::ParaParamsCharN + UG::BbParaParamsCharN + ScipParaParamsCharN);
   }

   ///
   /// get number of string parameters
   /// @return size of parameter table
   ///
   size_t getNumStringParams(
         )
   {
      return (UG::ParaParamsStringN + UG::BbParaParamsStringN + ScipParaParamsStringN);
   }

   ///
   /// initialize ParaParamSet object
   ///
//   void init(
//         );


};

}

#endif  // __SCIP_PARA_PARAM_SET_H__
