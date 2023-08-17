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

/**@file    scipParaSolutionMpi.h
 * @brief   ScipParaSolution extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_SOLUTION_MPI_H__
#define __SCIP_PARA_SOLUTION_MPI_H__

#include <mpi.h>
#include "scipParaTagDef.h"
#include "scipParaSolution.h"
#include "scipParaSolver.h"

namespace ParaSCIP
{

/** ScipSolution class */
class ScipParaSolutionMpi : public ScipParaSolution
{
   /** create scipSolutionPreDatatype */
   MPI_Datatype createPreDatatype();
   /** create scipSolutionDatatype */
   MPI_Datatype createDatatype(bool memAllocNecessary);

public:

   /** default constructor */
   ScipParaSolutionMpi(
	      )
   {
   }

   /** constructor */
   ScipParaSolutionMpi(
         ScipParaSolver *solver,
         SCIP_Real      objval,
         int            inNvars,
         SCIP_VAR **    vars,
         SCIP_Real *    vals
         )
        : ScipParaSolution(solver, objval, inNvars, vars, vals){}

   /** constructor */
   ScipParaSolutionMpi(
         double inObjectiveFunctionValue,
         int inNVars,                       /**< number of variables */
         int *inIndicesAmongSolvers,        /**< array of variable indices ( probindex )  */
         SCIP_Real *inValues                /**< array of bounds which the branchings     */
         ): ScipParaSolution(inObjectiveFunctionValue, inNVars, inIndicesAmongSolvers, inValues) {}

   /** destructor */
   ~ScipParaSolutionMpi(
         )
   {
   }

   /** create clone of this object */
   ScipParaSolutionMpi *clone(UG::ParaComm *comm);

   /** broadcast solution data to from the root rank */
   void bcast(UG::ParaComm *comm, int root);

   /** send solution data to the rank */
   void send(UG::ParaComm *comm, int destination);

   /** receive solution data from the source rank */
   void receive(UG::ParaComm *comm, int source);

};

typedef ScipParaSolutionMpi *ScipParaSolutionMpiPtr;

}

#endif // __SCIP_PARA_SOLUTION_MPI_H__

