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

/**@file    scipParaSolutionTh.cpp
 * @brief   ScipParaSolution extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaSolutionTh.h"

using namespace UG;
using namespace ParaSCIP;

/** create clone of this object */
ScipParaSolutionTh *
ScipParaSolutionTh::clone(UG::ParaComm *comm)
{
   return( new ScipParaSolutionTh(objectiveFunctionValue, nVars, indicesAmongSolvers, values));
}

/** create ScipDiffSubproblemPreDatatype */
ScipParaSolutionTh *
ScipParaSolutionTh::createDatatype(
      UG::ParaComm *comm
      )
{
   return clone(comm);
}

/** send solution data to the rank */
void
ScipParaSolutionTh::bcast(ParaComm *comm, int root)
{
   DEF_PARA_COMM( commTh, comm);

   if( commTh->getRank() == root )
   {
      for( int i = 0; i < commTh->getSize(); i++ )
      {
         if( i != root )
         {
            PARA_COMM_CALL(
               commTh->uTypeSend((void *)createDatatype(comm), ParaSolutionType, i, TagSolution)
            );
         }
      }
   }
   else
   {
      ScipParaSolutionTh *received;
      PARA_COMM_CALL(
         commTh->uTypeReceive((void **)&received, ParaSolutionType, root, TagSolution)
      );

      objectiveFunctionValue = received->objectiveFunctionValue;
      nVars = received->nVars;
      indicesAmongSolvers = new int[nVars];
      values = new SCIP_Real[nVars];
      for( int i = 0; i < nVars; i++ )
      {
         indicesAmongSolvers[i] = received->indicesAmongSolvers[i];
         values[i] = received->values[i];
      }
      delete received;
   }
}

/** send solution data to the rank */
void
ScipParaSolutionTh::send(ParaComm *comm, int destination)
{
   DEF_PARA_COMM( commTh, comm);
   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(comm), ParaSolutionType, destination, TagSolution)
   );
}

/** receive solution data from the source rank */
void
ScipParaSolutionTh::receive(ParaComm *comm, int source)
{
   DEF_PARA_COMM( commTh, comm);

   ScipParaSolutionTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, ParaSolutionType, source, TagSolution)
   );

   objectiveFunctionValue = received->objectiveFunctionValue;
   nVars = received->nVars;
   indicesAmongSolvers = new int[nVars];
   values = new SCIP_Real[nVars];
   for( int i = 0; i < nVars; i++ )
   {
      indicesAmongSolvers[i] = received->indicesAmongSolvers[i];
      values[i] = received->values[i];
   }
   delete received;

}
