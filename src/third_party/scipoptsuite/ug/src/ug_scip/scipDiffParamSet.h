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

/**@file    scipDiffParamSet.h
 * @brief   SCIP parameter set to be transferred ( Only keep difference between default settings ).
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_DIFF_PARAM_SET_H__
#define __SCIP_DIFF_PARAM_SET_H__
#include <cstring>
#include "ug/paraDef.h"
#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "ug/paraComm.h"
#include "scip/scip.h"

namespace ParaSCIP
{

/** ScipDiffParamSet class */
class ScipDiffParamSet {
protected:
   int          numBoolParams;            /**< the number of bool parameters */
   size_t       boolParamNamesSize;       /**< size of boolParameterNames area */
   char         *boolParamNames;          /**< boolean parameter names: names are concatenated */
   unsigned int *boolParamValues;         /**< boolean parameter values */

   int          numIntParams;             /**< the number of int parameters */
   size_t       intParamNamesSize;        /**< size of intParameterNames area */
   char         *intParamNames;           /**< int parameter names: names are concatenated */
   int          *intParamValues;          /**< int parameter values */

   int          numLongintParams;         /**< the number of longint parameters */
   size_t       longintParamNamesSize;    /**< size of longintParameterNames area */
   char         *longintParamNames;       /**< longint parameter names: names are concatenated */
   long long    *longintParamValues;      /**< longint parameter values */

   int          numRealParams;            /**< the number of real parameters */
   size_t       realParamNamesSize;       /**< size of realParameterNames area */
   char         *realParamNames;          /**< real parameter names: names are concatenated */
   double       *realParamValues;         /**< real parameter values */

   int          numCharParams;            /**< the number of char parameters */
   size_t       charParamNamesSize;       /**< size of charParameterNames area */
   char         *charParamNames;          /**< char parameter names: names are concatenated */
   char         *charParamValues;         /**< char parameter values */

   int          numStringParams;          /**< the number of string parameters */
   size_t       stringParamNamesSize;      /**< size of stringParameterNames area */
   char         *stringParamNames;        /**< string parameter names: names are concatenated */
   size_t       stringParamValuesSize;    /**< size of stringParameterValues area */
   char         *stringParamValues;       /**< string parameter values: values are concatenated */

   /** allocate memory for names and values */
   void allocateMemoty();

public:
   /** constructor */
   ScipDiffParamSet(
         )
         : numBoolParams(0), boolParamNamesSize(0), boolParamNames(0), boolParamValues(0),
         numIntParams(0), intParamNamesSize(0), intParamNames(0), intParamValues(0),
         numLongintParams(0), longintParamNamesSize(0), longintParamNames(0), longintParamValues(0),
         numRealParams(0), realParamNamesSize(0), realParamNames(0), realParamValues(0),
         numCharParams(0), charParamNamesSize(0), charParamNames(0), charParamValues(0),
         numStringParams(0), stringParamNamesSize(0), stringParamNames(0), stringParamValuesSize(0), stringParamValues(0)
	  {
   }

   /** constructor with scip */
   ScipDiffParamSet(
         SCIP *scip
         );

   /** destructor */
   virtual ~ScipDiffParamSet(
         )
   {
      if( boolParamNames )     delete[] boolParamNames;
      if( boolParamValues )    delete[] boolParamValues;
      if( intParamNames )      delete[] intParamNames;
      if( intParamValues )     delete[] intParamValues;
      if( longintParamNames )  delete[] longintParamNames;
      if( longintParamValues ) delete[] longintParamValues;
      if( realParamNames )     delete[] realParamNames;
      if( realParamValues )    delete[] realParamValues;
      if( charParamNames )     delete[] charParamNames;
      if( charParamValues )    delete[] charParamValues;
      if( stringParamNames )   delete[] stringParamNames;
      if( stringParamValues )  delete[] stringParamValues;
   }

   /** set these parameter values in scip environment */
   void setParametersInScip(
         SCIP *scip
         );

   /** get number of different parameters between their default values */
   int nDiffParams(
        )
   {
      return (numBoolParams + numIntParams + numLongintParams + numRealParams + numCharParams + numStringParams);
   }

   /** check if this parameter setting contains the argument name real parameter or not **/
   /** NOTE: this function is not tested */
   bool doesContainRealParam(char *string)
   {
      char *paramName = realParamNames;
      for( int i = 0; i < numRealParams; i++ )
      {
         if( std::strcmp(paramName, string) == 0 )
         {
            return true;
         }
         paramName += std::strlen(paramName) + 1;
      }
      return false;
   }

   /** stringfy DiffParamSet */
   std::string toString();

  /** broadcast scipDiffParamSet */
  virtual int bcast(UG::ParaComm *comm, int root) = 0;

  /** end scipDiffParamSet to the rank */
  virtual int send(UG::ParaComm *comm, int destination) = 0;

  /** receive scipDiffParamSet from the source rank */
  virtual int receive(UG::ParaComm *comm, int source) = 0;

#ifdef UG_WITH_ZLIB
  /** write ScipDiffParamSet */
  void write(gzstream::ogzstream &out);

  /** read ScipDiffParamSet */
  bool read(UG::ParaComm *comm, gzstream::igzstream &in);
#endif

};

}

#endif // _SCIP_DIFF_PARAM_SET_H__
