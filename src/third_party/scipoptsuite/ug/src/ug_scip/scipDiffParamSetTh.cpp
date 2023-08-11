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

/**@file    scipDiffParamSetTh.cpp
 * @brief   ScipDiffParamSet extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cstring>
#include <cassert>
#include "scip/scip.h"
#include "scipDiffParamSetTh.h"

using namespace UG;
using namespace ParaSCIP;

/** create scipDiffParamSetPreType */
ScipDiffParamSetTh *
ScipDiffParamSetTh::createDatatype(
      )
{
   return clone();
}

/** create clone */
ScipDiffParamSetTh *
ScipDiffParamSetTh::clone(
      )
{
   ScipDiffParamSetTh *newParam = new ScipDiffParamSetTh();

   newParam->numBoolParams = numBoolParams;
   newParam->boolParamNamesSize = boolParamNamesSize;
   newParam->numIntParams = numIntParams;
   newParam->intParamNamesSize = intParamNamesSize;
   newParam->numLongintParams = numLongintParams;
   newParam->longintParamNamesSize = longintParamNamesSize;
   newParam->numRealParams = numRealParams;
   newParam->realParamNamesSize = realParamNamesSize;
   newParam->numCharParams = numCharParams;
   newParam->charParamNamesSize = charParamNamesSize;
   newParam->numStringParams = numStringParams;
   newParam->stringParamNamesSize = stringParamNamesSize;
   newParam->stringParamValuesSize = stringParamValuesSize;

   newParam->allocateMemoty();

   char *cloneParamName;
   char *paramName;

   /** copy boolean parameter values */
   cloneParamName = newParam->boolParamNames;
   paramName = boolParamNames;
   for(int i = 0; i < newParam->numBoolParams; i++ )
   {
      newParam->boolParamValues[i] = boolParamValues[i];
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy int parameter values */
   cloneParamName = newParam->intParamNames;
   paramName = intParamNames;
   for(int i = 0; i < newParam->numIntParams; i++ )
   {
      newParam->intParamValues[i] = intParamValues[i];
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy longint parameter values */
   cloneParamName = newParam->longintParamNames;
   paramName = longintParamNames;
   for(int i = 0; i < newParam->numLongintParams; i++ )
   {
      newParam->longintParamValues[i] = longintParamValues[i];
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy real parameter values */
   cloneParamName = newParam->realParamNames;
   paramName = realParamNames;
   for(int i = 0; i < newParam->numRealParams; i++ )
   {
      newParam->realParamValues[i] = realParamValues[i];
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy char parameter values */
   cloneParamName = newParam->charParamNames;
   paramName = charParamNames;
   for(int i = 0; i < newParam->numCharParams; i++ )
   {
      newParam->charParamValues[i] = charParamValues[i];
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy string parameter values */
   char *cloneParamValue = newParam->stringParamValues;
   char *paramValue = stringParamValues;
   cloneParamName = newParam->stringParamNames;
   paramName = stringParamNames;
   for(int i = 0; i < newParam->numStringParams; i++ )
   {
      strcpy(cloneParamValue, paramValue);
      cloneParamValue += strlen(cloneParamValue) + 1;
      paramValue += strlen(paramValue) + 1;
      strcpy(cloneParamName, paramName);
      cloneParamName += strlen(cloneParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   return newParam;
}

void
ScipDiffParamSetTh::setValues(
      ScipDiffParamSetTh *from
      )
{

   numBoolParams = from->numBoolParams;
   boolParamNamesSize = from->boolParamNamesSize;
   numIntParams = from->numIntParams;
   intParamNamesSize = from->intParamNamesSize;
   numLongintParams = from->numLongintParams;
   longintParamNamesSize = from->longintParamNamesSize;
   numRealParams = from->numRealParams;
   realParamNamesSize = from->realParamNamesSize;
   numCharParams = from->numCharParams;
   charParamNamesSize = from->charParamNamesSize;
   numStringParams = from->numStringParams;
   stringParamNamesSize = from->stringParamNamesSize;
   stringParamValuesSize = from->stringParamValuesSize;

   allocateMemoty();

   char *paramName;
   char *fromParamName;

   /** copy boolean parameter values */
   paramName = boolParamNames;
   fromParamName = from->boolParamNames;
   for(int i = -0; i < from->numBoolParams; i++ )
   {
      boolParamValues[i] = from->boolParamValues[i];
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy int parameter values */
   paramName = intParamNames;
   fromParamName = from->intParamNames;
   for(int i = -0; i < from->numIntParams; i++ )
   {
      intParamValues[i] = from->intParamValues[i];
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy longint parameter values */
   paramName = longintParamNames;
   fromParamName = from->longintParamNames;
   for(int i = -0; i < from->numLongintParams; i++ )
   {
      longintParamValues[i] = from->longintParamValues[i];
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy real parameter values */
   paramName = realParamNames;
   fromParamName = from->realParamNames;
   for(int i = -0; i < from->numRealParams; i++ )
   {
      realParamValues[i] = from->realParamValues[i];
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy char parameter values */
   paramName = charParamNames;
   fromParamName = from->charParamNames;
   for(int i = -0; i < from->numCharParams; i++ )
   {
      charParamValues[i] = from->charParamValues[i];
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }

   /** copy string parameter values */
   char *paramValue = stringParamValues;
   char *fromParamValue = from->stringParamValues;
   paramName = stringParamNames;
   fromParamName = from->stringParamNames;
   for(int i = -0; i < from->numStringParams; i++ )
   {
      strcpy(paramValue, fromParamValue);
      fromParamValue += strlen(fromParamValue) + 1;
      paramValue += strlen(paramValue) + 1;
      strcpy(paramName, fromParamName);
      fromParamName += strlen(fromParamName) + 1;
      paramName += strlen(paramName) + 1;
   }
}

/** send solution data to the rank */
int
ScipDiffParamSetTh::bcast(
      ParaComm *comm,
      int root
      )
{
   DEF_PARA_COMM( commTh, comm);

   if( commTh->getRank() == root )
   {
      for( int i = 0; i < commTh->getSize(); i++ )
      {
         if( i != root )
         {
            ScipDiffParamSetTh *sent;
            sent = createDatatype();
            PARA_COMM_CALL(
               commTh->uTypeSend((void *)sent, UG::ParaSolverDiffParamType, i, UG::TagSolverDiffParamSet)
            );
         }
      }
   }
   else
   {
      ScipDiffParamSetTh *received;
      PARA_COMM_CALL(
         commTh->uTypeReceive((void **)&received, UG::ParaSolverDiffParamType, root, UG::TagSolverDiffParamSet)
      );
      setValues(received);
      delete received;
   }

   return 0;
}

/** send solution data to the rank */
int
ScipDiffParamSetTh::send(
      ParaComm *comm,
      int dest
      )
{
   DEF_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(), UG::ParaSolverDiffParamType, dest, UG::TagSolverDiffParamSet)
   );

   return 0;
}

 /** receive solution data from the source rank */
int
ScipDiffParamSetTh::receive(
       ParaComm *comm,
       int source
       )
 {
   DEF_PARA_COMM( commTh, comm);

   ScipDiffParamSetTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, UG::ParaSolverDiffParamType, source, UG::TagSolverDiffParamSet)
   );
   setValues(received);
   delete received;

   return 0;
 }
