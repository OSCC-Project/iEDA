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

/**@file    scipParaCommTh.cpp
 * @brief   SCIP ParaComm extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaCommTh.h"
#include "scipParaInstanceTh.h"
#include "scipParaDiffSubproblemTh.h"
#include "scipParaSolutionTh.h"
#include "scipParaInitialStatTh.h"
#include "scipParaRacingRampUpParamSetTh.h"
#include "scipDiffParamSetTh.h"
#include "scipParaParamSet.h"
#include "scipParaInitialStat.h"

using namespace ParaSCIP;

const char *
ScipParaCommTh::tagStringTable[] = {
  TAG_STR(TagInitialStat)
};

bool
ScipParaCommTh::tagStringTableIsSetUpCoorectly(
      )
{
#ifdef _COMM_CPP11
   if( !UG::BbParaCommCPP11::tagStringTableIsSetUpCoorectly() ) return false;
#else
   if( !UG::BbParaCommPth::tagStringTableIsSetUpCoorectly() ) return false;
#endif
   // std::cout << "size = " << sizeof(tagStringTable)/sizeof(char*)
    //      << ", (N_SCIP_TH_TAGS - UG::N_BB_TH_TAGS) = " <<  (N_SCIP_TH_TAGS - UG::N_BB_TH_TAGS) << std::endl;
   return ( sizeof(tagStringTable)/sizeof(char*) == (N_SCIP_TH_TAGS - UG::N_BB_TH_TAGS) );
}

const char *
ScipParaCommTh::getTagString(
      int tag                 /// tag to be converted to string
      )
{
   assert( tag >= 0 && tag < N_SCIP_TH_TAGS );
   if( tag >= 0 && tag < TAG_SCIP_FIRST )
   {
#ifdef _COMM_CPP11
      return BbParaCommCPP11::getTagString(tag);
#else
      return BbParaCommPth::getTagString(tag);
#endif
   }
   else
   {
      return tagStringTable[(tag - TAG_SCIP_FIRST)];
   }
}

ScipParaCommTh::~ScipParaCommTh(
      )
{
#ifdef _COMM_CPP11
   std::lock_guard<std::mutex> lock(rankLockMutex);
#else
   UG::LOCK_RAII(&rankLock);
#endif
   for(int i = 0; i < (comSize + 1); i++)
   {
      UG::MessageQueueElement *elem = messageQueueTable[i]->extarctElement(&sentMessage[i]);
      while( elem )
      {
         if( elem->getData() )
         {
            if( !freeStandardTypes(elem) )
            {
               switch( elem->getDataTypeId())
               {
               case UG::ParaSolverDiffParamType:
               {
                  delete reinterpret_cast<ScipDiffParamSet *>(elem->getData());
                  break;
               }
               case ParaInitialStatType:
               {
                  delete reinterpret_cast<ScipParaInitialStat *>(elem->getData());
                  break;
               }
               default:
               {
                  ABORT_LOGICAL_ERROR2("Requested type is not implemented. Type = ", elem->getDataTypeId() );
               }
               }
            }
         }
         delete elem;
         elem = messageQueueTable[i]->extarctElement(&sentMessage[i]);
      }
   }
}

/*******************************************************************************
* transfer object factory
*******************************************************************************/
UG::ParaDiffSubproblem *
ScipParaCommTh::createParaDiffSubproblem(
    )
{ 
    return new ScipParaDiffSubproblemTh(); 
}

UG::ParaInitialStat* 
ScipParaCommTh::createParaInitialStat(
    )
{ 
    return new ScipParaInitialStatTh(); 
}

UG::ParaRacingRampUpParamSet* 
ScipParaCommTh::createParaRacingRampUpParamSet(
    )
{ 
    return new ScipParaRacingRampUpParamSetTh(); 
}

UG::ParaInstance*
ScipParaCommTh::createParaInstance(
    )
{ 
    return new ScipParaInstanceTh(); 
}

UG::ParaSolution*
ScipParaCommTh::createParaSolution(
    )
{ 
    return new ScipParaSolutionTh(); 
}

UG::ParaParamSet*
ScipParaCommTh::createParaParamSet(
    )
{
    return new ScipParaParamSet();
}


ScipParaInstance*
ScipParaCommTh::createScipParaInstance(
    SCIP *scip, 
    int method
    )
{
    return new ScipParaInstanceTh(scip, method);
}

ScipParaSolution*
ScipParaCommTh::createScipParaSolution(
    ScipParaSolver *solver,
    SCIP_Real objval, 
    int inNvars, 
    SCIP_VAR ** vars, 
    SCIP_Real *vals
    )
{
    return new ScipParaSolutionTh(solver, objval, inNvars, vars, vals);
}

ScipParaSolution*
ScipParaCommTh::createScipParaSolution(
    SCIP_Real objval, 
    int inNvars, 
    int *inIndicesAmongSolvers,
    SCIP_Real *vals
    )
{
    return new ScipParaSolutionTh(objval, inNvars, inIndicesAmongSolvers, vals);
}

ScipParaDiffSubproblem*
ScipParaCommTh::createScipParaDiffSubproblem(
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
    return new ScipParaDiffSubproblemTh(
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
ScipParaCommTh::createScipParaInitialStat(
         SCIP *scip
         )
{
    return new ScipParaInitialStatTh(
         scip
         );
}

ScipParaInitialStat*
ScipParaCommTh::createScipParaInitialStat(
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
    return new ScipParaInitialStatTh(
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
ScipParaCommTh::createScipParaRacingRampUpParamSet(
         int inTerminationCriteria,
         int inNNodesLeft,
         double inTimeLimit,
         int inScipRacingParamSeed,
         int inPermuteProbSeed,
         int inGenerateBranchOrderSeed,
         ScipDiffParamSet *inScipDiffParamSet
         )
{
    return new ScipParaRacingRampUpParamSetTh(
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
ScipParaCommTh::createScipDiffParamSet()
{
    return new ScipDiffParamSetTh();
}

ScipDiffParamSet *
ScipParaCommTh::createScipDiffParamSet(
        SCIP *scip
        )
{
    return new ScipDiffParamSetTh(scip);
}
