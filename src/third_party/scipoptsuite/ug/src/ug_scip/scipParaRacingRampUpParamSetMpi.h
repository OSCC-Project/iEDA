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

/**@file    scipParaRacingRampUpParamSetMpi.h
 * @brief   ScipParaRacingRampUpParamSet extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_MPI_H__
#define __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_MPI_H__

#include <mpi.h>
#include "ug/paraCommMpi.h"
#include "scipParaTagDef.h"
#include "scipParaRacingRampUpParamSet.h"

namespace ParaSCIP
{

/** The difference between instance and subproblem: this is base class */
class ScipParaRacingRampUpParamSetMpi : public ScipParaRacingRampUpParamSet
{

   /** create ScipParaRacingRampUpParamSet datatype */
   MPI_Datatype createDatatype();

public:
   /** default constructor */
   ScipParaRacingRampUpParamSetMpi(
         )
   {
   }

   /** Constructor */
   ScipParaRacingRampUpParamSetMpi(
         int inTerminationCriteria,
         int inNNodesLeft,
         double inTimeLimit,
         int inScipRacingParamSeed,
         int inPermuteProbSeed,
         int inGenerateBranchOrderSeed,
         ScipDiffParamSet *inScipDiffParamSet
         ) : ScipParaRacingRampUpParamSet(inTerminationCriteria, inNNodesLeft, inTimeLimit,
         inScipRacingParamSeed,inPermuteProbSeed, inGenerateBranchOrderSeed, inScipDiffParamSet)
   {
   }

   /** destructor */
   ~ScipParaRacingRampUpParamSetMpi(
         )
   {
   }

   int send(UG::ParaComm *comm, int dest);
   int receive(UG::ParaComm *comm, int source);
};

}

#endif    // __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_MPI_H__

