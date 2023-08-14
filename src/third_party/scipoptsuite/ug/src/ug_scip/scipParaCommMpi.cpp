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

/**@file    scipParaCommMpi.cpp
 * @brief   SCIP ParaComm extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaCommMpi.h"
#include "scipParaInstanceMpi.h"
#include "scipParaDiffSubproblemMpi.h"
#include "scipParaSolutionMpi.h"
#include "scipParaInitialStatMpi.h"
#include "scipParaRacingRampUpParamSetMpi.h"
#include "scipDiffParamSetMpi.h"
#include "scipParaParamSet.h"

using namespace ParaSCIP;


const char *
ScipParaCommMpi::tagStringTable[] = {
      TAG_STR(TagInitialStat1),
      TAG_STR(TagSolverDiffParamSet1),
      TAG_STR(TagDiffSubproblem1),
      TAG_STR(TagDiffSubproblem2),
      TAG_STR(TagDiffSubproblem3),
      TAG_STR(TagDiffSubproblem4),
      TAG_STR(TagDiffSubproblem5),
      TAG_STR(TagDiffSubproblem6),
      TAG_STR(TagDiffSubproblem7),
      TAG_STR(TagDiffSubproblem8),
      TAG_STR(TagDiffSubproblem9),
      TAG_STR(TagDiffSubproblem10),
      TAG_STR(TagDiffSubproblem11),
      TAG_STR(TagDiffSubproblem12),
      TAG_STR(TagDiffSubproblem13),
      TAG_STR(TagDiffSubproblem14),
      TAG_STR(TagSolution1)
};

bool
ScipParaCommMpi::tagStringTableIsSetUpCoorectly(
      )
{
   return ( sizeof(tagStringTable)/sizeof(char*) == (N_SCIP_MPI_TAGS - UG::N_BB_MPI_TAGS) );
}

const char *
ScipParaCommMpi::getTagString(
      int tag                 /// tag to be converted to string
      )
{
   assert( tag >= 0 && tag < N_SCIP_MPI_TAGS );
   if( tag >= 0 && tag < TAG_SCIP_FIRST )
   {
      return BbParaCommMpi::getTagString(tag);
   }
   else
   {
      return tagStringTable[(tag - TAG_SCIP_FIRST)];
   }
}

/*******************************************************************************
* transfer object factory
*******************************************************************************/
UG::ParaDiffSubproblem *
ScipParaCommMpi::createParaDiffSubproblem(
    )
{ 
    return new ScipParaDiffSubproblemMpi(); 
}

UG::ParaInitialStat* 
ScipParaCommMpi::createParaInitialStat(
    )
{ 
    return new ScipParaInitialStatMpi(); 
}

UG::ParaRacingRampUpParamSet* 
ScipParaCommMpi::createParaRacingRampUpParamSet(
    )
{ 
    return new ScipParaRacingRampUpParamSetMpi(); 
}

UG::ParaInstance*
ScipParaCommMpi::createParaInstance(
    )
{ 
    return new ScipParaInstanceMpi(); 
}

UG::ParaSolution*
ScipParaCommMpi::createParaSolution(
    )
{ 
    return new ScipParaSolutionMpi(); 
}

ScipParaInstance*
ScipParaCommMpi::createScipParaInstance(
    SCIP *scip, 
    int method
    )
{
    return new ScipParaInstanceMpi(scip, method);
}

ScipParaSolution*
ScipParaCommMpi::createScipParaSolution(
    ScipParaSolver *solver,
    SCIP_Real objval, 
    int inNvars, 
    SCIP_VAR ** vars, 
    SCIP_Real *vals
    )
{
    return new ScipParaSolutionMpi(solver, objval, inNvars, vars, vals);
}

UG::ParaParamSet*
ScipParaCommMpi::createParaParamSet(
    )
{
    return new ScipParaParamSet();
}

ScipParaSolution*
ScipParaCommMpi::createScipParaSolution(
    SCIP_Real objval, 
    int inNvars, 
    int *inIndicesAmongSolvers,
    SCIP_Real *vals
    )
{
    return new ScipParaSolutionMpi(objval, inNvars, inIndicesAmongSolvers, vals);
}

ScipParaDiffSubproblem*
ScipParaCommMpi::createScipParaDiffSubproblem(
         SCIP *scip,
         ScipParaSolver *scipParaSolver,
         int nNewBranchVars,
         SCIP_VAR **newBranchVars,
         SCIP_Real *newBranchBounds,
         SCIP_BOUNDTYPE *newBoundTypes,
         int nAddedConss,
         SCIP_CONS **addedConss
         )
{
    return new ScipParaDiffSubproblemMpi(
         scip,
         scipParaSolver,
         nNewBranchVars,
         newBranchVars,
         newBranchBounds,
         newBoundTypes,
         nAddedConss,
         addedConss
         );
}

ScipParaInitialStat*
ScipParaCommMpi::createScipParaInitialStat(
         SCIP *scip
         )
{
    return new ScipParaInitialStatMpi(
         scip
         );
}

ScipParaInitialStat*
ScipParaCommMpi::createScipParaInitialStat(
            int inMaxDepth,
            int inMaxTotalDepth,
            int inNVarBranchStatsDown,
            int inNVarBranchStatsUp,
            int *inIdxLBranchStatsVarsDown,
            int *inNVarBranchingDown,
            int *inIdxLBranchStatsVarsUp,
            int *inNVarBranchingUp,
            SCIP_Real *inDownpscost,
            SCIP_Real *inDownvsids,
            SCIP_Real *inDownconflen,
            SCIP_Real *inDowninfer,
            SCIP_Real *inDowncutoff,
            SCIP_Real *inUppscost,
            SCIP_Real *inUpvsids,
            SCIP_Real *inUpconflen,
            SCIP_Real *inUpinfer,
            SCIP_Real *inUpcutoff
         )
{
    return new ScipParaInitialStatMpi(
            inMaxDepth,
            inMaxTotalDepth,
            inNVarBranchStatsDown,
            inNVarBranchStatsUp,
            inIdxLBranchStatsVarsDown,
            inNVarBranchingDown,
            inIdxLBranchStatsVarsUp,
            inNVarBranchingUp,
            inDownpscost,
            inDownvsids,
            inDownconflen,
            inDowninfer,
            inDowncutoff,
            inUppscost,
            inUpvsids,
            inUpconflen,
            inUpinfer,
            inUpcutoff
         );
}

ScipParaRacingRampUpParamSet *
ScipParaCommMpi::createScipParaRacingRampUpParamSet(
         int inTerminationCriteria,
         int inNNodesLeft,
         double inTimeLimit,
         int inScipRacingParamSeed,
         int inPermuteProbSeed,
         int inGenerateBranchOrderSeed,
         ScipDiffParamSet *inScipDiffParamSet
         )
{
    return new ScipParaRacingRampUpParamSetMpi(
               inTerminationCriteria,
               inNNodesLeft,
               inTimeLimit,
               inScipRacingParamSeed,
               inPermuteProbSeed,
               inGenerateBranchOrderSeed,
               inScipDiffParamSet
               );
}

ScipDiffParamSet *
ScipParaCommMpi::createScipDiffParamSet()
{
    return new ScipDiffParamSetMpi();
}

ScipDiffParamSet *
ScipParaCommMpi::createScipDiffParamSet(
        SCIP *scip
        )
{
    return new ScipDiffParamSetMpi(scip);
}

