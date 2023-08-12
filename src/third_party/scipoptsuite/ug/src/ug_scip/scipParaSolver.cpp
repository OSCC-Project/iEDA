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
/* $Id: scipParaSolver.cpp,v 1.46 2014/04/29 20:12:52 bzfshina Exp $ */

/**@file    $RCSfile: scipParaSolver.cpp,v $
 * @brief   ParaSolver extension for SCIP: Parallelized solver implementation for SCIP.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cfloat>
#include <cstring>
#include <cstdlib>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <typeinfo>
#include <string>
#include <sstream>
#include "ug/paraInitialStat.h"
#include "ug_bb/bbParaComm.h"
#include "ug_bb/bbParaNode.h"
#include "ug_bb/bbParaInstance.h"
#include "ug_bb/bbParaSolver.h"
#include "ug_bb/bbParaSolution.h"
#include "ug_bb/bbParaSolverTerminationState.h"
#include "objscip/objscip.h"
#include "scipParaTagDef.h"
#include "scipParaParamSet.h"
#include "scipParaObjMessageHdlr.h"
#include "scipParaObjCommPointHdlr.h"
#include "scipParaObjLimitUpdator.h"
#include "scipParaObjProp.h"
#include "scipParaObjBranchRule.h"
#include "scipParaInitialStat.h"
#include "scipParaRacingRampUpParamSet.h"
#include "scipParaObjNodesel.h"
#include "scipParaObjSelfSplitNodesel.h"
#include "scip/scip.h"
#ifdef UG_DEBUG_SOLUTION
#ifndef WITH_DEBUG_SOLUTION
#define WITH_DEBUG_SOLUTION
#endif
#include "scip/debug.h"
#include "scip/struct_scip.h"
#include "scip/struct_set.h"
#endif
// #include "scip/scipdefplugins.h"

using namespace ParaSCIP;

#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
extern long long virtualMemUsedAtLc;
extern double memoryLimitOfSolverSCIP;
#endif

extern void
setUserPlugins(UG::ParaInstance *instance);
extern void
setUserPlugins(UG::ParaSolver *solver);

/*
 * Callback methods of conflict handler
 */
#define CONFLICTHDLR_NAME      "conflictCollector"
#define CONFLICTHDLR_DESC      "conflict handler to collect conflicts"
#define CONFLICTHDLR_PRIORITY  +100000000
static
SCIP_DECL_CONFLICTEXEC(conflictExecCollector)
{  /*lint --e{715}*/
   SCIP_VAR** vars;
   SCIP_Real* vals;
   SCIP_Real lhs;
   int i;

   assert(conflicthdlr != NULL);
   assert(strcmp(SCIPconflicthdlrGetName(conflicthdlr), CONFLICTHDLR_NAME) == 0);
   assert(bdchginfos != NULL || nbdchginfos == 0);
   assert(result != NULL);

   /* don't process already resolved conflicts */
   if( resolved )
   {
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   *result = SCIP_DIDNOTFIND;

   /* create array of variables and coefficients: sum_{i \in P} x_i - sum_{i \in N} x_i >= 1 - |N| */
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, nbdchginfos) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vals, nbdchginfos) );
   lhs = 1.0;
   for( i = 0; i < nbdchginfos; ++i )
   {
      assert(bdchginfos != NULL);

      vars[i] = SCIPbdchginfoGetVar(bdchginfos[i]);

      /* we can only treat binary variables */
      /**@todo extend linear conflict constraints to some non-binary cases */
      if( !SCIPvarIsBinary(vars[i]) )
         break;

      /* check whether the variable is fixed to zero (P) or one (N) in the conflict set */
      if( SCIPbdchginfoGetNewbound(bdchginfos[i]) < 0.5 )
         vals[i] = 1.0;
      else
      {
         vals[i] = -1.0;
         lhs -= 1.0;
      }
   }

   if( i == nbdchginfos )
   {
      ScipParaSolver *scipParaSolver = reinterpret_cast<ScipParaSolver *>(SCIPconflicthdlrGetData(conflicthdlr));
      std::list<LocalNodeInfoPtr> *conflictConsList = scipParaSolver->getConflictConsList();
      LocalNodeInfo *localNodeInfo = new LocalNodeInfo;
      localNodeInfo->linearRhs = SCIPinfinity(scip);
      localNodeInfo->nLinearCoefs = nbdchginfos;
      localNodeInfo->idxLinearCoefsVars = new int[nbdchginfos];
      localNodeInfo->linearCoefs = new double[nbdchginfos];
      for( i = 0; i < nbdchginfos; ++i )
      {
         SCIP_VAR *transformVar = vars[i];
         SCIP_Real scalar = vals[i];
         SCIP_Real constant = 0.0;
         if( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) ==  SCIP_INVALIDDATA )
            break;
         // assert(transformVar != NULL);
         if( transformVar )
         {
            lhs -= constant;
            if( scipParaSolver->isOriginalIndeciesMap() )
            {
               localNodeInfo->idxLinearCoefsVars[i] = scipParaSolver->getOriginalIndex(SCIPvarGetIndex(transformVar));
            }
            else
            {
               localNodeInfo->idxLinearCoefsVars[i] = SCIPvarGetIndex(transformVar);
            }
            localNodeInfo->linearCoefs[i] = scalar;
         }
         else
         {
            break;
         }
      }
      if( i == nbdchginfos )
      {
         localNodeInfo->linearLhs = lhs;
         conflictConsList->push_back(localNodeInfo);
      }
      else
      {
         delete [] localNodeInfo->idxLinearCoefsVars;
         delete [] localNodeInfo->linearCoefs;
         delete localNodeInfo;
      }
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &vals);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

void
ScipParaSolver::setWinnerRacingParams(
      UG::ParaRacingRampUpParamSet *inRacingParams   /**< winner solver pramset */
      )
{
   if( !userPlugins )
   {
      SCIP_CALL_ABORT( SCIPresetParams(scip) );
   }

   if( paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing) )
   {
      SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
      SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
      SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
      if( inRacingParams )
      {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
         ScipParaRacingRampUpParamSet *scipRacingParams = dynamic_cast< ScipParaRacingRampUpParamSet * >(inRacingParams);
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "misc/permutationseed", scipRacingParams->getPermuteProbSeed()) );
#endif
      }
   }
   else
   {
      ScipParaRacingRampUpParamSet *scipRacingParams = dynamic_cast< ScipParaRacingRampUpParamSet * >(inRacingParams);
      setRacingParams(scipRacingParams, true);
   }

#if SCIP_VERSION >= 320
   setBakSettings();
#endif
 
}

void
ScipParaSolver::setRacingParams(
      UG::ParaRacingRampUpParamSet *inRacingParams,
      bool winnerParam
      )
{
   ScipParaRacingRampUpParamSet *scipRacingParams = dynamic_cast< ScipParaRacingRampUpParamSet * >(inRacingParams);

   if( !winnerParam && !userPlugins )
   {
      SCIP_CALL_ABORT( SCIPresetParams(scip) );
   }

   if ( std::string(paraParams->getStringParamValue(UG::RacingParamsDirPath)) != std::string("") )
   {
      assert( scipRacingParams->getScipDiffParamSet() );
      if( !winnerParam )
      {
         scipRacingParams->getScipDiffParamSet()->setParametersInScip(scip);
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/randomseedshift", scipRacingParams->getScipRacingParamSeed()) );
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/permutationseed", scipRacingParams->getPermuteProbSeed()) );
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/lpseed", scipRacingParams->getPermuteProbSeed()) );
         // int tempInt = 0;
         // SCIP_CALL_ABORT( SCIPgetIntParam(scip, "lp/solvefreq", &tempInt) );
         // std::cout << "R." << paraComm->getRank() <<  " lp/solvefreq = " << tempInt << std::endl;
      }
   }
   else
   {
      if( paraParams->getBoolParamValue(UG::ProvingRun) )
      {
         if( !winnerParam && scipRacingParams->getScipDiffParamSet() )
         {
            scipRacingParams->getScipDiffParamSet()->setParametersInScip(scip);
         }
      }
      else
      {
         int nHeuristics;
         if( paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing ))
         {
            nHeuristics = scipRacingParams->getScipRacingParamSeed() % 2;
         }
         else
         {
            nHeuristics = scipRacingParams->getScipRacingParamSeed() % 4;
         }
         int nPresolving = (scipRacingParams->getScipRacingParamSeed()/4) % 4;
         int nSeparating = (scipRacingParams->getScipRacingParamSeed()/(4*4)) % 4;

         switch( nHeuristics )
         {
            case 0:
            {
               SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
               break;
            }
            case 1:
            {
               SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
               break;
            }
            case 2:
            {
               SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_FAST, TRUE) );
               break;
            }
            case 3:
            {
               SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE) );
               break;
            }
            default:
               THROW_LOGICAL_ERROR1("invalid nHeuristics");
         }

         switch( nPresolving )
         {
            case 0:
            {
               SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
               break;
            }
            case 1:
            {
#ifdef _COMM_PTH
               if( paraParams->getBoolParamValue(ParaSCIP::CustomizedToSharedMemory ) )
               {
                  SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
               }
               else
               {
                  SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
               }
#else
               SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
#endif
               break;
            }
            case 2:
            {
               SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_FAST, TRUE) );
               break;
            }
            case 3:
            {
               SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE) );
               break;
            }
            default:
               THROW_LOGICAL_ERROR1("invalid nPresolving");
         }

         switch( nSeparating )
         {
            case 0:
            {
               SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
               break;
            }
            case 1:
            {
#ifdef _COMM_PTH
               if( paraParams->getBoolParamValue(ParaSCIP::CustomizedToSharedMemory ) )
               {
                  SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
               }
               else
               {
                  if( paraParams->getBoolParamValue(UG::NoAggressiveSeparatorInRacing) )
                  {
                     SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
                     SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
                  }
                  else
                  {
                     SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
                  }
               }
#else
               if( paraParams->getBoolParamValue(UG::NoAggressiveSeparatorInRacing) )
               {
                  SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
                  SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
               }
               else
               {
                  SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_AGGRESSIVE, TRUE) );
               }
#endif
               break;
            }
            case 2:
            {
               SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_FAST, TRUE) );
               break;
            }
            case 3:
            {
               SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_OFF, TRUE) );
               break;
            }
            default:
               THROW_LOGICAL_ERROR1("invalid nSeparating");
         }
      }

      assert(SCIPgetStage(scip) <= SCIP_STAGE_TRANSFORMED);
      // make sure that the permutation works on transformed problem
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "misc/permutationseed", scipRacingParams->getPermuteProbSeed()) );
#endif

      if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
            paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
            paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
      {
         dropSettingsForVariableBoundsExchnage();
      }

      if( !winnerParam && scipRacingParams->getPermuteProbSeed() >= 64 )  // after all parameters tested, random branchig variable selection
      {
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "branching/random/maxdepth", 2) );
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "branching/random/priority", 100000) );
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "branching/random/seed", scipRacingParams->getGenerateBranchOrderSeed()) );
      }
   }

   if( winnerParam && (!paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing)) )
   {
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/randomseedshift", scipRacingParams->getScipRacingParamSeed()) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/permutationseed", scipRacingParams->getPermuteProbSeed()) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "randomization/lpseed", scipRacingParams->getPermuteProbSeed()) );
      if( scipRacingParams->getScipDiffParamSet() )
      {
         scipRacingParams->getScipDiffParamSet()->setParametersInScip(scip);
      }
   }

#if SCIP_VERSION >= 320
   setBakSettings();
#endif

   // writeSubproblem();

}

void
ScipParaSolver::createSubproblem(
      )
{
   assert(currentTask);

   UG::BbParaNode *bbCurrentNode = dynamic_cast<UG::BbParaNode *>(currentTask);

#ifdef UG_DEBUG_SOLUTION
   if( scip->set->debugsoldata == NULL )
   {
      SCIP_CALL_ABORT( SCIPdebugSolDataCreate(&((scip->set)->debugsoldata)));
      // SCIPdebugSetMainscipset(scip->set);
   }
#endif

   /** set instance specific parameters */
   // if( currentNode->isRootNode() && !(paraParams->getBoolParamValue(UseRootNodeCuts)) )
   // Probably, in order to avoid root node settings twice. Once root nodes is solved in LC
   // when UseRootNodeCuts is specified. However, for racing ramp-up, this is too bad.
   //  So, I changed the specification. The root node parameter settings is applied twice now

   // Do not reset here, because racing parameters might be set already.
   // SCIP_CALL_ABORT( SCIPresetParams(scip) );

   /** set original node selection strategy */
   // setOriginalNodeSelectionStrategy();
   commPointHdlr->resetCommPointHdlr();
   nodesel->reset();
   if( bbCurrentNode->isRootTask() )
   {
      if ( std::string(paraParams->getStringParamValue(UG::RacingParamsDirPath)) == std::string("") )
      {
         scipDiffParamSetRoot->setParametersInScip(scip);
      }
      if( paraParams->getBoolParamValue(UG::NoSolverPresolvingAtRoot) ||
            ( paraParams->getBoolParamValue(UG::NoSolverPresolvingAtRootDefaultSet) &&
                  isRacingStage() &&
                  paraComm->getRank() == 1 )    // rank 1 should be all default
            )
      {
         SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE) );
      }
   }
   else
   {
      if( paraParams->getBoolParamValue(UG::NoSolverPresolvingAtRoot) )
      {
         SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );
      }
      if( paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing) )
      {
         if( scipDiffParamSet )   // this may not be necessary , check  setWinnerRacingParams(0)
         {
            scipDiffParamSet->setParametersInScip(scip);
         }
      }

      /*
      if( paraParams->getBoolParamValue(UG::ControlCollectingModeOnSolverSide) )
      {
         int maxrestarts;
         SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/maxrestarts", &maxrestarts) );
         if( maxrestarts < 0 )
         {
            std::cerr << "presolving/maxrestarts >= 0 when you specify ControlCollectingModeOnSolverSide = TRUE."
                  << std::endl;
            exit(1);
         }
      }
      else
      {
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrestarts", 0 ) );
      }
      */
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrestarts", getOriginalMaxRestart()) );
   }

   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "misc/usesymmetry", 0 ) );  // Symmetry handling technique is explicitly turn off in Solever for this version (SCIP 5.0)

#if SCIP_VERSION >= 320
   setBakSettings();
#endif

   double dualBoundValue = bbCurrentNode->getDualBoundValue();

   ScipParaDiffSubproblem *scipParaDiffSubproblem = dynamic_cast< ScipParaDiffSubproblem* >(currentTask->getDiffSubproblem());

   SCIP_VAR **orgVars = SCIPgetOrigVars(scip);  // variables are indexed by index
   int nOrg = SCIPgetNOrigVars(scip);       // the number of original variables
   if( scipParaDiffSubproblem )
   {
      if( mapToProbIndecies )
      {
         assert( mapToSolverLocalIndecies );
         for(int v = 0; v <  scipParaDiffSubproblem->getNBoundChanges(); v++)
         {
            assert(mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]] >= 0);
            if( mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]] < nOrg )
            {
               if( scipParaDiffSubproblem->getBoundType(v) == SCIP_BOUNDTYPE_LOWER )
               {
                  SCIP_CALL_ABORT(
                        SCIPchgVarLbGlobal(
                              scip,
                              orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]],
                              scipParaDiffSubproblem->getBranchBound(v) )
                        );
                  if( scipParaDiffSubproblem->getIndex(v) < nOrgVars )
                  {
                     assert(SCIPisEQ(scip,SCIPvarGetLbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]]),scipParaDiffSubproblem->getBranchBound(v)));
                     assert(SCIPisLE(scip,SCIPvarGetLbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]]),SCIPvarGetUbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]])));
                  }
               }
               else if (scipParaDiffSubproblem->getBoundType(v) == SCIP_BOUNDTYPE_UPPER)
               {
                  SCIP_CALL_ABORT(SCIPchgVarUbGlobal(
                        scip,
                        orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]],
                        scipParaDiffSubproblem->getBranchBound(v) )
                  );
                  if( scipParaDiffSubproblem->getIndex(v) < nOrgVars )
                  {
                     assert(SCIPisEQ(scip,SCIPvarGetUbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]]),scipParaDiffSubproblem->getBranchBound(v)));
                     assert(SCIPisLE(scip,SCIPvarGetLbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]]),SCIPvarGetUbGlobal(orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]]])));
                  }
               }
               else
               {
                  THROW_LOGICAL_ERROR2("Invalid bound type: type = ", static_cast<int>(scipParaDiffSubproblem->getBoundType(v))) ;
               }
            }
            else
            {
               std::cout << "fixing branching variable index = " << mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIndex(v)]] << " is omitted!" << std::endl;
            }
         }
      }
      else
      {
         for(int v = 0; v <  scipParaDiffSubproblem->getNBoundChanges(); v++)
         {
            if( scipParaDiffSubproblem->getIndex(v) < nOrg )
            {
               if( scipParaDiffSubproblem->getBoundType(v) == SCIP_BOUNDTYPE_LOWER )
               {
                  SCIP_CALL_ABORT(
                        SCIPchgVarLbGlobal(
                              scip,
                              orgVars[scipParaDiffSubproblem->getIndex(v)],
                              scipParaDiffSubproblem->getBranchBound(v) )
                        );
                  if( scipParaDiffSubproblem->getIndex(v) < nOrgVars )
                  {
                     assert(SCIPisEQ(scip,SCIPvarGetLbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)]),scipParaDiffSubproblem->getBranchBound(v)));
                     assert(SCIPisLE(scip,SCIPvarGetLbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)]),SCIPvarGetUbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)])));
                  }
               }
               else if (scipParaDiffSubproblem->getBoundType(v) == SCIP_BOUNDTYPE_UPPER)
               {
                  SCIP_CALL_ABORT(SCIPchgVarUbGlobal(
                        scip,
                        orgVars[scipParaDiffSubproblem->getIndex(v)],
                        scipParaDiffSubproblem->getBranchBound(v) )
                  );
                  if( scipParaDiffSubproblem->getIndex(v) < nOrgVars )
                  {
                     assert(SCIPisEQ(scip,SCIPvarGetUbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)]),scipParaDiffSubproblem->getBranchBound(v)));
                     assert(SCIPisLE(scip,SCIPvarGetLbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)]),SCIPvarGetUbGlobal(orgVars[scipParaDiffSubproblem->getIndex(v)])));
                  }
               }
               else
               {
                  THROW_LOGICAL_ERROR2("Invalid bound type: type = ", static_cast<int>(scipParaDiffSubproblem->getBoundType(v))) ;
               }
            }
            else
            {
               std::cout << "fixing branching variable index = " << scipParaDiffSubproblem->getIndex(v) << " is omitted!" << std::endl;
            }
         }
      }

      if( scipParaDiffSubproblem->getNBranchConsLinearConss() > 0 ||
            scipParaDiffSubproblem->getNBranchConsSetppcConss() > 0 ||
            scipParaDiffSubproblem->getNLinearConss() > 0 ||
            scipParaDiffSubproblem->getNBendersLinearConss() > 0 ||
            scipParaDiffSubproblem->getNBoundDisjunctions() )
      {
         assert(addedConss == 0);
         addedConss = new SCIP_CONS*[scipParaDiffSubproblem->getNBranchConsLinearConss()
                                     + scipParaDiffSubproblem->getNBranchConsSetppcConss()
                                     + scipParaDiffSubproblem->getNLinearConss()
                                     + scipParaDiffSubproblem->getNBendersLinearConss()
                                     + scipParaDiffSubproblem->getNBoundDisjunctions()];
      }

      SCIP_CONS* cons;
      char consname[SCIP_MAXSTRLEN];

      int c = 0;
      for(; c < scipParaDiffSubproblem->getNBranchConsLinearConss() ; c++ )
      {
         SCIP_VAR** vars;
         SCIP_Real* vals;
         int nVars = scipParaDiffSubproblem->getBranchConsNLinearCoefs(c);

         /* create array of variables and coefficients */
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nVars) );
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals, nVars) );

         if( mapToProbIndecies )
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getBranchConsLinearIdxCoefsVars(c,v)]]];
               vals[v] = scipParaDiffSubproblem->getBranchConsLinearCoefs(c,v);
            }
         }
         else
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[scipParaDiffSubproblem->getBranchConsLinearIdxCoefsVars(c,v)];
               vals[v] = scipParaDiffSubproblem->getBranchConsLinearCoefs(c,v);
            }
         }

         /* create a constraint */
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s", scipParaDiffSubproblem->getBranchConsLinearConsNames(c));
         SCIP_CALL_ABORT( SCIPcreateConsLinear(scip, &cons, consname, nVars, vars, vals,
               scipParaDiffSubproblem->getBranchConsLinearLhs(c), scipParaDiffSubproblem->getBranchConsLinearRhs(c),
               TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE) );
         /** only a constraint whose "enforce is TRUE can be written in transformed problem */

         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         assert(cons);
         addedConss[c] = cons;
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         /* free temporary memory */
         SCIPfreeBufferArray(scip, &vals);
         SCIPfreeBufferArray(scip, &vars);
      }

      int i = 0;
      for(; c < (scipParaDiffSubproblem->getNBranchConsLinearConss()
            + scipParaDiffSubproblem->getNBranchConsSetppcConss()) ; c++ )
      {
         SCIP_VAR** vars;

         int nVars = scipParaDiffSubproblem->getBranchConsSetppcNVars(i);
         /* create array of variables, types and bounds */
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nVars) );

         if( mapToProbIndecies )
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getBranchConsSetppcVars(i,v)]]];
            }
         }
         else
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[scipParaDiffSubproblem->getBranchConsSetppcVars(i,v)];
            }
         }

         /* create a constraint */
         assert( scipParaDiffSubproblem->getBranchConsSetppcType(i) == SCIP_SETPPCTYPE_PARTITIONING ); // currently, only this should be used
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s", scipParaDiffSubproblem->getBranchConsSetppcConsNames(i));
         if( scipParaDiffSubproblem->getBranchConsSetppcType(i) == SCIP_SETPPCTYPE_PARTITIONING )
         {
            SCIP_CALL_ABORT( SCIPcreateConsSetpart(scip, &cons, consname, nVars, vars,
                  TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE ) );
         } else if ( scipParaDiffSubproblem->getBranchConsSetppcType(i) == SCIP_SETPPCTYPE_PACKING )
         {
            SCIP_CALL_ABORT( SCIPcreateConsSetpack(scip, &cons, consname, nVars, vars,
                  TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE ) );
         } else if ( scipParaDiffSubproblem->getBranchConsSetppcType(i) == SCIP_SETPPCTYPE_COVERING )
         {
            SCIP_CALL_ABORT( SCIPcreateConsSetcover(scip, &cons, consname, nVars, vars,
                  TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE ) );
         } else {
            THROW_LOGICAL_ERROR2("Unknown setppc constraint is received: type = ", scipParaDiffSubproblem->getBranchConsSetppcType(i));
         }

         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         assert(cons);
         addedConss[c] = cons;
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         /* free temporary memory */
         SCIPfreeBufferArray(scip, &vars);
         i++;
      }

      i = 0;
      for(; c < (scipParaDiffSubproblem->getNBranchConsLinearConss()
            + scipParaDiffSubproblem->getNBranchConsSetppcConss()
            + scipParaDiffSubproblem->getNLinearConss()) ; c++ )
      {
         SCIP_VAR** vars;
         SCIP_Real* vals;
         int nVars = scipParaDiffSubproblem->getNLinearCoefs(i);

         /* create array of variables and coefficients */
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nVars) );
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals, nVars) );

         if( mapToProbIndecies )
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIdxLinearCoefsVars(i,v)]]];
               vals[v] = scipParaDiffSubproblem->getLinearCoefs(i,v);
            }
         }
         else
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[scipParaDiffSubproblem->getIdxLinearCoefsVars(i,v)];
               vals[v] = scipParaDiffSubproblem->getLinearCoefs(i,v);
            }
         }

         /* create a constraint */
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "cli%d", i);
         SCIP_CALL_ABORT( SCIPcreateConsLinear(scip, &cons, consname, nVars, vars, vals,
               scipParaDiffSubproblem->getLinearLhs(i), scipParaDiffSubproblem->getLinearRhs(i),
               TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE) );
         /** only a constraint whose "enforce is TRUE can be written in transformed problem */

         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         assert(cons);
         addedConss[c] = cons;
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         /* free temporary memory */
         SCIPfreeBufferArray(scip, &vals);
         SCIPfreeBufferArray(scip, &vars);
         i++;
      }

      i = 0;
      for(; c < (scipParaDiffSubproblem->getNBranchConsLinearConss()
            + scipParaDiffSubproblem->getNBranchConsSetppcConss()
            + scipParaDiffSubproblem->getNLinearConss()
			+ scipParaDiffSubproblem->getNBendersLinearConss()) ; c++ )
      {
         SCIP_VAR** vars;
         SCIP_Real* vals;
         int nVars = scipParaDiffSubproblem->getNBendersLinearCoefs(i);

         /* create array of variables and coefficients */
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nVars) );
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals, nVars) );

         if( mapToProbIndecies )
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIdxBendersLinearCoefsVars(i,v)]]];
               vals[v] = scipParaDiffSubproblem->getBendersLinearCoefs(i,v);
            }
         }
         else
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[scipParaDiffSubproblem->getIdxBendersLinearCoefsVars(i,v)];
               vals[v] = scipParaDiffSubproblem->getBendersLinearCoefs(i,v);
            }
         }

         /* create a constraint */
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "cli%d", i);
         SCIP_CALL_ABORT( SCIPcreateConsLinear(scip, &cons, consname, nVars, vars, vals,
               scipParaDiffSubproblem->getBendersLinearLhs(i), scipParaDiffSubproblem->getBendersLinearRhs(i),
               TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE) );
         /** only a constraint whose "enforce is TRUE can be written in transformed problem */

         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         assert(cons);
         addedConss[c] = cons;
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         /* free temporary memory */
         SCIPfreeBufferArray(scip, &vals);
         SCIPfreeBufferArray(scip, &vars);
         i++;
      }

      i = 0;
      for(; c < (scipParaDiffSubproblem->getNBranchConsLinearConss()
            + scipParaDiffSubproblem->getNBranchConsSetppcConss()
            + scipParaDiffSubproblem->getNLinearConss()
			+ scipParaDiffSubproblem->getNBendersLinearConss()
            + scipParaDiffSubproblem->getNBoundDisjunctions()) ; c++ )
      {
         SCIP_VAR** vars;
         SCIP_BOUNDTYPE  *types;
         SCIP_Real* bounds;
         int nVars = scipParaDiffSubproblem->getNVarsBoundDisjunction(i);
         /* create array of variables, types and bounds */
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nVars) );
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &types, nVars) );
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &bounds, nVars) );

         if( mapToProbIndecies )
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIdxBoundDisjunctionVars(i,v)]]];
               types[v] = scipParaDiffSubproblem->getBoundTypesBoundDisjunction(i,v);
               bounds[v] = scipParaDiffSubproblem->getBoundsBoundDisjunction(i,v);
            }
         }
         else
         {
            for( int v = 0; v < nVars; ++v )
            {
               vars[v] = orgVars[scipParaDiffSubproblem->getIdxBoundDisjunctionVars(i,v)];
               types[v] = scipParaDiffSubproblem->getBoundTypesBoundDisjunction(i,v);
               bounds[v] = scipParaDiffSubproblem->getBoundsBoundDisjunction(i,v);
            }
         }

         /* create a constraint */
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "bdj%d", i);
         SCIP_CALL_ABORT( SCIPcreateConsBounddisjunction(scip, &cons, consname, nVars, vars, types, bounds,
               scipParaDiffSubproblem->getFlagBoundDisjunctionInitial(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionSeparate(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionEnforce(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionCheck(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionPropagate(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionLocal(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionModifiable(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionDynamic(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionRemovable(i),
               scipParaDiffSubproblem->getFlagBoundDisjunctionStickingatnode(i) ) );
         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         assert(cons);
         addedConss[c] = cons;
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         /* free temporary memory */
         SCIPfreeBufferArray(scip, &bounds);
         SCIPfreeBufferArray(scip, &types);
         SCIPfreeBufferArray(scip, &vars);
         i++;
      }
      nAddedConss = c;
   }


   int addingConsParam = getParaParamSet()->getIntParamValue(AddDualBoundCons);
   addedDualCons = 0;
   if( addingConsParam != 0 )
   {
      if( ( addingConsParam == 1 && !SCIPisGT(scip, bbCurrentNode->getDualBoundValue(), bbCurrentNode->getInitialDualBoundValue()) )
            || addingConsParam == 2 || addingConsParam == 3 )
      {
         SCIP_CONS* cons;
         int nvars = SCIPgetNVars(scip);
         SCIP_VAR **vars = SCIPgetVars(scip);
         SCIP_Real* vals = new SCIP_Real[nvars];
         for(int v = 0; v < nvars; ++v )
         {
            vals[v] = SCIPvarGetObj(vars[v]);
         }
         SCIP_CALL_ABORT( SCIPcreateConsLinear(scip, &cons, "objective", nvars, vars, vals, dualBoundValue, SCIPinfinity(scip),
               TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE) );
         assert( SCIPisEQ( scip, SCIPgetTransObjscale(scip), 1.0 ) );
         assert( SCIPisZero( scip, SCIPgetTransObjoffset(scip) ) );

         /** try to automatically convert a linear constraint into a more specific and more specialized constraint */

         /* add constraint to SCIP */
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
         addedDualCons = cons;
      }
   }

   if( userPlugins && scipParaDiffSubproblem )
   {
      userPlugins->newSubproblem(scip, scipParaDiffSubproblem->getBranchLinearConss(), scipParaDiffSubproblem->getBranchSetppcConss());
   }

   if( !paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing) && winnerRacingParams )
   {
         setWinnerRacingParams(winnerRacingParams);   // winner parameters are set, again
         // std::cout << winnerRacingParams->toString() << std::endl;
   }

   /// do not save solutions to original problem space
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/transsolsorig", FALSE) );

   if( SCIPgetStage(scip) == SCIP_STAGE_PROBLEM)
   {
      SCIP_CALL_ABORT( SCIPtransformProb(scip));
   }

   if( scipParaDiffSubproblem && scipParaDiffSubproblem->getNVarBranchStats() > 0 )
   {
      orgVars = SCIPgetOrigVars(scip);       /* original problem's variables              */
      if( mapToProbIndecies )
      {
         for( int i = 0; i < scipParaDiffSubproblem->getNVarBranchStats(); i++ )
         {
            SCIP_CALL_ABORT( SCIPinitVarBranchStats(scip, orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIdxLBranchStatsVars(i)]]],
                  scipParaDiffSubproblem->getDownpscost(i),
                  scipParaDiffSubproblem->getUppscost(i),
                  scipParaDiffSubproblem->getDownvsids(i),
                  scipParaDiffSubproblem->getUpvsids(i),
                  scipParaDiffSubproblem->getDownconflen(i),
                  scipParaDiffSubproblem->getUpconflen(i),
                  scipParaDiffSubproblem->getDowninfer(i),
                  scipParaDiffSubproblem->getUpinfer(i),
                  scipParaDiffSubproblem->getDowncutoff(i),
                  scipParaDiffSubproblem->getUpcutoff(i)
                  )
            );
         }
      }
      else
      {
         for( int i = 0; i < scipParaDiffSubproblem->getNVarBranchStats(); i++ )
         {
            SCIP_CALL_ABORT( SCIPinitVarBranchStats(scip, orgVars[scipParaDiffSubproblem->getIdxLBranchStatsVars(i)],
                  scipParaDiffSubproblem->getDownpscost(i),
                  scipParaDiffSubproblem->getUppscost(i),
                  scipParaDiffSubproblem->getDownvsids(i),
                  scipParaDiffSubproblem->getUpvsids(i),
                  scipParaDiffSubproblem->getDownconflen(i),
                  scipParaDiffSubproblem->getUpconflen(i),
                  scipParaDiffSubproblem->getDowninfer(i),
                  scipParaDiffSubproblem->getUpinfer(i),
                  scipParaDiffSubproblem->getDowncutoff(i),
                  scipParaDiffSubproblem->getUpcutoff(i)
                  )
            );
         }
      }
   }

#if SCIP_VERSION >= 312
   // std::cout << " VERSION >= 320 " << std::endl;

   if( scipParaDiffSubproblem && scipParaDiffSubproblem->getNVarValueVars() > 0 )
   {
      orgVars = SCIPgetOrigVars(scip);       /* original problem's variables              */
      if( mapToProbIndecies )
      {
         for( int i = 0; i < scipParaDiffSubproblem->getNVarValueVars(); i++ )
         {
            for( int j = 0; j < scipParaDiffSubproblem->getNVarValueValues(i); j++ )
            {
               SCIP_CALL_ABORT( SCIPinitVarValueBranchStats(scip, orgVars[mapToProbIndecies[mapToSolverLocalIndecies[scipParaDiffSubproblem->getIdxLBranchStatsVars(i)]]],
                     scipParaDiffSubproblem->getVarValue(i,j),
                     scipParaDiffSubproblem->getVarValueDownvsids(i,j),
                     scipParaDiffSubproblem->getVarVlaueUpvsids(i,j),
                     scipParaDiffSubproblem->getVarValueDownconflen(i,j),
                     scipParaDiffSubproblem->getVarValueUpconflen(i,j),
                     scipParaDiffSubproblem->getVarValueDowninfer(i,j),
                     scipParaDiffSubproblem->getVarValueUpinfer(i,j),
                     scipParaDiffSubproblem->getVarValueDowncutoff(i,j),
                     scipParaDiffSubproblem->getVarValueUpcutoff(i,j)
                     )
               );
               /*
               std::cout << mapToOriginalIndecies[scipParaDiffSubproblem->getIdxLBranchStatsVars(i)]
                         << ", "
                         << scipParaDiffSubproblem->getVarValue(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueDownvsids(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarVlaueUpvsids(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueDownconflen(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueUpconflen(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueDowninfer(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueUpinfer(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueDowncutoff(i,j)
                         << ", "
                         << scipParaDiffSubproblem->getVarValueUpcutoff(i,j)
                         << std::endl;
                         */
            }
         }
      }
      else
      {
         for( int i = 0; i < scipParaDiffSubproblem->getNVarValueVars(); i++ )
         {
            for( int j = 0; j < scipParaDiffSubproblem->getNVarValueValues(i); j++ )
            {
               SCIP_CALL_ABORT( SCIPinitVarValueBranchStats(scip, orgVars[scipParaDiffSubproblem->getIdxLBranchStatsVars(i)],
                     scipParaDiffSubproblem->getVarValue(i,j),
                     scipParaDiffSubproblem->getVarValueDownvsids(i,j),
                     scipParaDiffSubproblem->getVarVlaueUpvsids(i,j),
                     scipParaDiffSubproblem->getVarValueDownconflen(i,j),
                     scipParaDiffSubproblem->getVarValueUpconflen(i,j),
                     scipParaDiffSubproblem->getVarValueDowninfer(i,j),
                     scipParaDiffSubproblem->getVarValueUpinfer(i,j),
                     scipParaDiffSubproblem->getVarValueDowncutoff(i,j),
                     scipParaDiffSubproblem->getVarValueUpcutoff(i,j)
                     )
               );
            }
         }
      }
      /** for debug *********************
      int nvars;                                ** number of variables
      int nbinvars;                             ** number of binary variables
      int nintvars;                             ** number of integer variables
      SCIP_VAR** vars;                          ** transformed problem's variables
      SCIP_CALL_ABORT( SCIPgetVarsData(scip, &vars, &nvars, &nbinvars, &nintvars, NULL, NULL) );
      int ngenvars = nbinvars+nintvars;
      int nOrgVarst = 0;

      if( ngenvars > 0 )
      {
         std::cout << "R." << paraComm->getRank()  << ", ngenvars = " << ngenvars << std::endl;;
         for( int i = 0; i < ngenvars; i++ )
         {
            assert( SCIPvarGetType(vars[i]) == SCIP_VARTYPE_BINARY || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_INTEGER );

            SCIP_VAR *transformVar = vars[i];
            SCIP_Real scalar = 1.0;
            SCIP_Real constant = 0.0;
            SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) );
            assert(transformVar != NULL);

            if( transformVar )  // The variable in the transformed space
            {
               SCIP_VALUEHISTORY* valuehistory = SCIPvarGetValuehistory(vars[i]);
               if( valuehistory != NULL )
               {
                  nOrgVarst++;
               }
               else
               {
                  std::cout << "R." << paraComm->getRank()  << ", no history for var i = " << i << std::endl;
                  std::cout << "R." << paraComm->getRank()  << ", org = " << SCIPvarGetValuehistory(transformVar) << std::endl;
               }
            }
            else
            {
               std::cout  << "R." << paraComm->getRank()  << ", no transfrom var i = " << i << std::endl;;
            }
         }
      }
      if( nOrgVarst == 0 )
      {
         std::cout << "Failed to set Var Value stat. R." << paraComm->getRank() << std::endl;;
         abort();
      }
      else
      {
         std::cout << "Set " << nOrgVarst << " Var Value stat. R." << paraComm->getRank() << std::endl;;
      }
      *** end of debug */
   }
#endif

//   if( userPlugins && scipParaDiffSubproblem)
//   {
//      userPlugins->newSubproblem(scip, scipParaDiffSubproblem->getBranchLinearConss(), scipParaDiffSubproblem->getBranchSetppcConss());
//   }

}

void
ScipParaSolver::freeSubproblem(
      )
{
   if( isRacingStage() && paraParams->getBoolParamValue(UG::RacingStatBranching) && !restartingRacing )
   {
	  DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
      ScipParaInitialStat *initialStat = scipParaComm->createScipParaInitialStat(scip);
      initialStat->send(paraComm, 0);
      delete initialStat;
   }
   SCIP_CALL_ABORT( SCIPfreeTransform(scip) );

   int c = 0;
   if( currentTask->getDiffSubproblem() )
   {
      ScipParaDiffSubproblem *scipParaDiffSubproblem = dynamic_cast< ScipParaDiffSubproblem* >(currentTask->getDiffSubproblem());
      if( scipParaDiffSubproblem->getNLinearConss() > 0 )
      {
         for(; c < scipParaDiffSubproblem->getNLinearConss() ; c++ )
         {
            if( !SCIPconsIsDeleted(addedConss[c]) )
            {
               SCIP_CALL_ABORT( SCIPdelCons(scip, addedConss[c]) );
            }	
         }
      }
   }

   if( addedConss )
   {
      for(; c < nAddedConss; c++ )
      {
         if( !SCIPconsIsDeleted(addedConss[c]) )
         {
            SCIP_CALL_ABORT( SCIPdelCons(scip, addedConss[c]) );
         }
      }
      delete [] addedConss;
      addedConss = 0;
   }

   if( addedDualCons )
   {
      SCIP_CALL_ABORT( SCIPdelCons(scip, addedDualCons) );
      addedDualCons = 0;
   }

   SCIP_VAR **orgVars = SCIPgetOrigVars(scip);  // variables are indexed by index
   // Taking into account multi-aggregate vars.
   // int n = SCIPgetNOrigVars(scip);       // the number of original variables
   // assert( n == nOrgVars );
   assert( nOrgVarsInSolvers == SCIPgetNOrigVars(scip));
   // for( int v = 0; v < n; v++ )
   for( int v = 0; v < nOrgVars; v++ )
   {
      SCIP_CALL_ABORT( SCIPchgVarLbGlobal( scip,orgVars[v], orgVarLbs[v] ) );
      SCIP_CALL_ABORT( SCIPchgVarUbGlobal( scip,orgVars[v], orgVarUbs[v] ) );
   }

   if( racingWinner )
   {
      nTightened = getNTightened();
      nTightenedInt = getNTightenedInt();
      // std::cout << "Winner: nTightened = " << nTightened << ", nTightenedInt = " << nTightenedInt << std::endl;
   }

   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      recoverOriginalSettings();
   }

}

void
ScipParaSolver::saveImprovedSolution(
      )
{
   DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
   SCIP_SOL *sol = SCIPgetBestSol(scip);
   int nVars = SCIPgetNOrigVars(scip);
   SCIP_VAR **vars = SCIPgetOrigVars(scip);
   SCIP_Real *vals = new SCIP_Real[nVars];
   SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
   if( isCopyIncreasedVariables() )
   {
      SCIP_VAR **varsInOrig = new SCIP_VAR*[nVars];
      SCIP_Real *valsInOrig = new SCIP_Real[nVars]();
      int nVarsInOrig = 0;
      for( int i = 0; i < nVars; i++ )
      {
         if( getOriginalIndex(SCIPvarGetIndex(vars[i])) >= 0 )
         {
            varsInOrig[nVarsInOrig] = vars[i];
            valsInOrig[nVarsInOrig] = vals[i];
            nVarsInOrig++;
         }
      }
      saveIfImprovedSolutionWasFound(
            scipParaComm->createScipParaSolution(
                  this,
                  SCIPgetSolOrigObj(scip, sol),
                  nVarsInOrig,
                  varsInOrig,
                  valsInOrig
                  )
            );
      delete [] varsInOrig;
      delete [] valsInOrig;
   }
   else
   {
      saveIfImprovedSolutionWasFound(
               scipParaComm->createScipParaSolution(
                     this,
                     SCIPgetSolOrigObj(scip, sol),
                     nVars,
                     vars,
                     vals
                     )
      );
   }
   delete [] vals;
}

void
ScipParaSolver::solve(
      )
{

   if( paraParams->getBoolParamValue(UG::ControlCollectingModeOnSolverSide) )
   {
      prohibitCollectingMode();
   }

   // if( paraParams->getBoolParamValue(UG::CheckGapInLC) )
   // {
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/gap", 0.0 ) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/absgap", 0.0 ) );
   // }

   // if( paraParams->getBoolParamValue(UG::CheckFeasibilityInLC) )
   // {
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/feastol", (orgFeastol/10.0) ) );
      if( SCIP_APIVERSION < 61 )
      {
         SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/lpfeastol", (orgLpfeastol/10.0) ) );
      }
      /*
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/feastol", (orgFeastol/100.0) ) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/lpfeastol", (orgLpfeastol/100.0) ) );
      */
   // }

   /** solve */
   if( paraParams->getBoolParamDefaultValue(UG::TransferConflictCuts) )
   {
      assert(conflictConsList->size() == 0);
   }

   /* don't catch control+c */
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );

   /** set cutoff value */
#ifdef UG_DEBUG_SOLUTION
   const double limit = DBL_MAX;
   if( limit < globalBestIncumbentValue )
   {
      SCIP_CALL_ABORT( SCIPsetObjlimit(scip, limit) );
   }
   else
   {
      SCIP_CALL_ABORT( SCIPsetObjlimit(scip, globalBestIncumbentValue) );
   }
   // if( ( !currentNode->getDiffSubproblem() ) ||
   //       currentNode->getDiffSubproblem()->isOptimalSolIncluded() )
   // {
   //    writeSubproblem();
   // }
   writeSubproblem();
#else
   /** set cutoff value */
   SCIP_CALL_ABORT( SCIPsetObjlimit(scip, globalBestIncumbentValue) );
#endif
   nPreviousNodesLeft = 0;

#ifdef _DEBUG_DET
   writeSubproblem();
#endif

   // if( paraComm->getRank() == 1) checkVarsAndIndex("***before solve***",scip);

   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "misc/usesymmetry", 0 ) );  // Symmetry handling technique is explicitly turn off in Solever for this version (SCIP 5.0)

   SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) ); // always wall clock time, racing may change clocktype
   // writeSubproblem();

#ifdef UG_DEBUG_SOLUTION
   // assert( paraNode == currentNode );
   if( ( !currentTask->getDiffSubproblem() ) ||
         currentTask->getDiffSubproblem()->isOptimalSolIncluded() )
   {
      SCIPdebugSolEnable(scip);
      std::cout << "R." << paraComm->getRank() << ": enable debug" << std::endl;
      assert( SCIPdebugSolIsEnabled(scip) == TRUE );
   }
   else
   {
      SCIPdebugSolDisable(scip);
      std::cout << "R." << paraComm->getRank() << ": disable debug" << std::endl;
      assert( SCIPdebugSolIsEnabled(scip) == FALSE );
   }
#endif

#if (SCIP_VERSION >= 700)
   if( paraParams->getBoolParamValue(UG::AllowTreeSearchRestart) == false )
   {
      if( !isRacingStage() )
      {
         SCIP_CALL_ABORT( SCIPsetCharParam(scip, "estimation/restarts/restartpolicy", 'n' ) );
      }
   }
#endif

   if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
   {
      double timeRemains =  std::max(0.0, (paraParams->getRealParamValue(UG::TimeLimit) - paraTimer->getElapsedTime()) ); 
      SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) );         // to confirm this for the scip environment actually to work
      SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
   }

#if SCIP_APIVERSION >= 101
   if( SCIPfindPresol(scip, "milp") != NULL )
   {
      int nmilpthreads = 0;
      SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/milp/threads", &nmilpthreads) );
      assert( nmilpthreads == 1 );
   }
#endif

   terminationMode = UG::CompTerminatedNormally; // assume that it terminates normally

   // writeSubproblem();
   SCIP_RETCODE ret = SCIPsolve(scip);
   if( ret != SCIP_OKAY )
   {
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIPprintError(ret, NULL);
#else
      SCIPprintError(ret);
#endif
      writeSubproblem();
      if( userPlugins )
      {
         userPlugins->writeSubproblem(scip);
      }
#ifdef _MSC_VER
      _sleep(10);
#else
      sleep(10);
#endif
      std::cout << "ret = " << (int)ret << std::endl;
      THROW_LOGICAL_ERROR1("SCIP terminate with NO SCIP_OKAY");
   }

   // Notification message has to complete
   if( notificationProcessed )
   {
      waitNotificationIdMessage();
   }

   // Then, solver status should be checked
   SCIP_STATUS status = SCIPgetStatus(scip);

#if SCIP_APIVERSION >= 101
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) ==  3 &&   // self-split ramp-up
         currentTask->isRootTask()  // the following should do only for the self-split ramp-up procedure
         )
   {
      if( status == SCIP_STATUS_OPTIMAL )
      {
         saveImprovedSolution();
      }
      if( selfSplitNodesel->inSampling() && status == SCIP_STATUS_OPTIMAL )   // it is still sampling. This means the poroblem was solved
      {
         if( paraComm->getRank() == 1 )
         {
            sendLocalSolution();
         }
         return;
      }
      else
      {
         sendLocalSolution();               // solution was not sent during sampling to generate the same search tree
      }
   }
#endif

//   if( status == SCIP_STATUS_OPTIMAL )   // when sub-MIP is solved at root node, the solution may not be saved
   if( SCIPgetNSols(scip) > 0 )
   {
      saveImprovedSolution();
      sendLocalSolution();
   }
//   else
//   {
#ifdef UG_DEBUG_SOLUTION
   if( status != SCIP_STATUS_OPTIMAL )
   {
      if( SCIPdebugSolIsEnabled(scip) && 
            currentTask->getMergingStatus() != 3 &&
            !( isRacingInterruptRequested() ||
                  ( isRacingWinner() && isCollectingAllNodes() ) ) )
      {
         std::cout << "R" << paraComm->getRank() << " solver lost optimal solution." << std::endl;
         throw "Optimal solution lost!";
      }
   }
#endif
//      if( status == SCIP_STATUS_MEMLIMIT  )
//      {
//         std::cout << "Warning: SCIP was interrupted because the memory limit was reached" << std::endl;
//         abort();
//      }
//   }

// std::cout << "R" << paraComm->getRank() << " status = " << (int)status << std::endl;

   if( status == SCIP_STATUS_OPTIMAL ||
         status == SCIP_STATUS_GAPLIMIT  )

   {
      solverDualBound = SCIPgetDualbound(scip);
      // if( status == SCIP_STATUS_OPTIMAL )
      // {
         // current best solution may not be accepted in LC
         // solverDualBound = std::max(solverDualBound, getGlobalBestIncumbentValue() );
         // should not do above, since globalBestIncumbentValue might not be updated
      // }

      /*
      double dualBound = SCIPgetDualbound(scip);
      if( isRacingStage() )
      {
         if( dualBound > solverDualBound )
         {
            solverDualBound = dualBound;
         }
      }
      else
      {
         solverDualBound = dualBound;
         // maximalDualBound = std::max(dualBound, currentNode->getDualBoundValue());
         if( EPSEQ( maximalDualBound, -DBL_MAX, eps ) )
         {
            maximalDualBound = std::max(dualBound, currentNode->getDualBoundValue());
         }
         else
         {
            if( !SCIPisInfinity(scip, -dualBound) &&
                  dualBound > currentNode->getDualBoundValue() &&
                  dualBound < maximalDualBound )
            {
               maximalDualBound = dualBound;
            }
            else
            {
               if( maximalDualBound < currentNode->getDualBoundValue() )
               {
                  maximalDualBound = currentNode->getDualBoundValue();
               }
            }
         }
      }
      */
   }
   else if( status == SCIP_STATUS_INFEASIBLE  )
   {
      if( EPSEQ(globalBestIncumbentValue, DBL_MAX, eps ) )
      {
         solverDualBound = SCIPgetDualbound(scip);
      }
      else
      {
         solverDualBound = globalBestIncumbentValue;
      }
   }
   else
   {
      if( status == SCIP_STATUS_NODELIMIT )
      {
         throw "SCIP terminated with SCIP_STATUS_NODELIMIT";
      }
      else if( status == SCIP_STATUS_TOTALNODELIMIT )
      {
         throw "SCIP terminated with SCIP_STATUS_TOTALNODELIMIT";
      }
      else if( status == SCIP_STATUS_STALLNODELIMIT )
      {
         throw "SCIP terminated with SCIP_STATUS_STALLNODELIMIT";
      }
      else if( status == SCIP_STATUS_TIMELIMIT )
      {
         // throw "SCIP terminated with SCIP_STATUS_TIMELIMIT";
         solverDualBound = SCIPgetDualbound(scip);
         setTerminationMode(UG::TimeLimitTerminationMode);
         if( isRacingStage() )
         {
            racingIsInterrupted = true;
         }
      }
      else if( status == SCIP_STATUS_MEMLIMIT )
      {
         memoryLimitIsReached = true;
      }
      else if( status == SCIP_STATUS_SOLLIMIT )
      {
         throw "SCIP terminated with SCIP_STATUS_SOLLIMIT";
      }
      else if( status == SCIP_STATUS_BESTSOLLIMIT )
      {
         throw "SCIP terminated with SCIP_STATUS_BESTSOLLIMIT";
      }
      else if( status == SCIP_STATUS_USERINTERRUPT &&  SCIPisObjIntegral(scip) )
      {
         if( SCIPfeasCeil(scip, dynamic_cast<UG::BbParaNode *>(getCurrentNode())->getDualBoundValue()) >= getGlobalBestIncumbentValue() )
         {
            solverDualBound = SCIPfeasCeil(scip, dynamic_cast<UG::BbParaNode *>(getCurrentNode())->getDualBoundValue());
         }
         else
         {
            solverDualBound = std::max(dynamic_cast<UG::BbParaNode *>(getCurrentNode())->getDualBoundValue(), SCIPgetDualbound(scip));
         }
      }
      else if( status == SCIP_STATUS_USERINTERRUPT )
      {
         solverDualBound = std::max(dynamic_cast<UG::BbParaNode *>(getCurrentNode())->getDualBoundValue(), SCIPgetDualbound(scip));
      }
      else
      {
         solverDualBound = -DBL_MAX;
      }
      /*
      if( maximalDualBound < currentNode->getDualBoundValue() )
      {
         maximalDualBound = currentNode->getDualBoundValue();
      }
      */
   }

   /*
   double dualBound = SCIPgetDualbound(scip);
   if( isRacingStage() )
   {
      if( dualBound > maximalDualBound )
      {
         maximalDualBound = dualBound;
      }
   }
   else
   {
      if( !SCIPisInfinity(scip, -dualBound) &&
            dualBound < maximalDualBound )
      {
         maximalDualBound = dualBound;
      }
   }
   */


   if( conflictConsList && conflictConsList->size() > 0 )
   {
      int nConfilcts = conflictConsList->size();
      for(int i = 0; i < nConfilcts; i++ )
      {
         assert(!conflictConsList->empty());
         LocalNodeInfo *info= conflictConsList->front();
         conflictConsList->pop_front();
         if( info->linearCoefs ) delete[] info->linearCoefs;
         if( info->idxLinearCoefsVars ) delete[] info->idxLinearCoefsVars;
         delete info;
      }
   }

#if SCIP_APIVERSION >= 101
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) ==  3 &&   // self-split ramp-up
         currentTask->isRootTask() )
   {
      int numnodesels = SCIPgetNNodesels( scip );
      SCIP_NODESEL** nodesels = SCIPgetNodesels( scip );
      int i;
      for( i = 0; i < numnodesels; ++i )
      {
         std::string nodeselname(SCIPnodeselGetName(nodesels[i]));
         if( std::string(nodeselname) == std::string("ScipParaObjSelfSplitNodeSel") )
         {
            break;
         }
      }
      assert( i != numnodesels );
      SCIP_CALL_ABORT( SCIPsetNodeselStdPriority(scip, nodesels[i], -INT_MAX/4 ) );
   }
#endif

}

long long
ScipParaSolver::getNNodesSolved(
      )
{
   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING || SCIPgetStage(scip) == SCIP_STAGE_SOLVED  )
   {
      return SCIPgetNTotalNodes(scip);
   }
   else
   {
      return 0;
   }
}

int
ScipParaSolver::getNNodesLeft(
      )
{
   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING || SCIPgetStage(scip) == SCIP_STAGE_SOLVED  )
   {
      return SCIPgetNNodesLeft(scip);
   }
   else
   {
      if( SCIPgetStage(scip) >= SCIP_STAGE_PRESOLVING && SCIPgetStage(scip) <= SCIP_STAGE_INITSOLVE )
      {
         return 1;
      }
      else
      {
         return 0;
      }
   }
}

double
ScipParaSolver::getDualBoundValue(
      )
{
   if( SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING || SCIPgetStage(scip) == SCIP_STAGE_INITSOLVE )
   {
      return dynamic_cast<UG::BbParaNode *>(currentTask)->getDualBoundValue();
   }
   else
   {
      return SCIPgetDualbound(scip);
   }
}

ScipParaSolver::ScipParaSolver(
      int argc,
      char **argv,
      UG::ParaComm     *comm,
      UG::ParaParamSet *inParaParamSet,
      UG::ParaInstance *inParaInstance,
      UG::ParaDeterministicTimer *inDetTimer
      ) : BbParaSolver(argc, argv, N_SCIP_TAGS, comm, inParaParamSet, inParaInstance, inDetTimer),
      messagehdlr(0),
      logfile(0),
      originalParamSet(0),
      conflictConsList(0),
      userPlugins(0),
      commPointHdlr(0),
      nodesel(0),
#if SCIP_APIVERSION >= 101
      selfSplitNodesel(0),
#endif
      scipPropagator(0),
      interruptMsgMonitor(0),
      nPreviousNodesLeft(0),
      originalPriority(0),
      nOrgVars(0),
      nOrgVarsInSolvers(0),
      orgVarLbs(0),
      orgVarUbs(0),
      tightenedVarLbs(0),
      tightenedVarUbs(0),
      mapToOriginalIndecies(0),
      mapToSolverLocalIndecies(0),
      mapToProbIndecies(0),
      // stuffingMaxrounds(0),
      // domcolMaxrounds(0),
      // dualcompMaxrounds(0),
      // dualinferMaxrounds(0),
      // dualaggMaxrounds(0),
      // abspowerDualpresolve(0),
      // andDualpresolving(0),
      // cumulativeDualpresolve(0),
      // knapsackDualpresolving(0),
      // linearDualpresolving(0),
      // setppcDualpresolving(0),
      // logicorDualpresolving(0),
      miscAllowdualreds(0),
      nAddedConss(0),
      addedConss(0),
      addedDualCons(0),
      settingsNameLC(0),
      fiberSCIP(false),
      quiet(false),
      collectingModeIsProhibited(false),
      problemFileName(0),
      orgFeastol(0.0),
      orgLpfeastol(0.0),
      copyIncreasedVariables(false)
{

   // ScipMessageHandlerFunctionPointer *scipMessageHandler = reinterpret_cast<ScipMessageHandlerFunctionPointer *>(messageHandler);
   // no additional message handlers

   char* logname = NULL;

   ScipParaInstance *scipParaInstance = dynamic_cast< ScipParaInstance *>(paraInstance);

   /* Initialize the SCIP environment */
   /*********
    * Setup *
    *********/
   /* initialize SCIP */
   SCIP_CALL_ABORT( SCIPcreate(&scip) );
   SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) ); // always wall clock time
   if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
   {
      double timeRemains =  std::max(0.0, (paraParams->getRealParamValue(UG::TimeLimit) - paraTimer->getElapsedTime() + 3.0) );  // 3.0: timming issue
      SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
   }
   /** set user plugins */
   ::setUserPlugins(this);
   if( paraParams->getIntParamValue(UG::InstanceTransferMethod) != 2 )
   {
      /* include default SCIP plugins */
      SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );
      /** include user plugins */
      includeUserPlugins(scip);
   }
   /* include communication point handler */
   ScipParaObjLimitUpdator *updator = new ScipParaObjLimitUpdator(scip,this);
   commPointHdlr = new ScipParaObjCommPointHdlr(paraComm, this, updator);
   SCIP_CALL_ABORT( SCIPincludeObjEventhdlr(scip, commPointHdlr, TRUE) );
   SCIP_CALL_ABORT( SCIPincludeObjHeur(scip, updator, TRUE) );

   /* include propagator */
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      scipPropagator = new ScipParaObjProp(paraComm, this);
      SCIP_CALL_ABORT( SCIPincludeObjProp(scip, scipPropagator, TRUE) );
      saveOriginalSettings();
      dropSettingsForVariableBoundsExchnage();
   }

   /* include node selector */
   nodesel = new ScipParaObjNodesel(this);
   SCIP_CALL_ABORT( SCIPincludeObjNodesel(scip, nodesel, TRUE) );
#if SCIP_APIVERSION >= 101
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) ==  3 )   // self-split ramp-up
   {
      selfSplitNodesel = new ScipParaObjSelfSplitNodesel(
            paraComm->getRank() - 1,
            paraComm->getSize() - 1,
            paraParams->getIntParamValue(UG::SelfSplitTreeDepth),
            paraComm,
            this,
            scip
            );
      SCIP_CALL_ABORT( SCIPincludeObjNodesel(scip, selfSplitNodesel, TRUE) );
   }
#endif
   /* include branch rule plugins */
   SCIP_CALL_ABORT( SCIPincludeObjBranchrule(scip, new ScipParaObjBranchRule(this), TRUE) );

   if( inParaParamSet->getBoolParamValue(UG::TransferConflictCuts) )
   {
      conflictConsList = new std::list<LocalNodeInfoPtr>;
      SCIP_CONFLICTHDLRDATA *conflictHdrData = reinterpret_cast< SCIP_CONFLICTHDLRDATA * >(this);
      /* create conflict handler to collects conflicts */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
      SCIP_CALL_ABORT( SCIPincludeConflicthdlr(scip, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            NULL, NULL, NULL, NULL, NULL, NULL, conflictExecCollector, conflictHdrData) );
#else
      SCIP_CALL_ABORT( SCIPincludeConflicthdlrBasic(scip, NULL, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            conflictExecCollector, conflictHdrData) );
#endif
   }

   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/feastol", &orgFeastol ) );
   if( SCIP_APIVERSION < 61 )
   {
      SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/lpfeastol", &orgLpfeastol ) );
   }

   /********************
    * Parse parameters *
    ********************/
   for( int i = 3; i < argc; ++i )   /** the first argument is runtime parameter file for ParaSCIP */
   {
      if( strcmp(argv[i], "-l") == 0 )
      {
         i++;
         if( i < argc )
            logname = argv[i];
         else
         {
            THROW_LOGICAL_ERROR1("missing log filename after parameter '-l'");
         }
      }
      else if( strcmp(argv[i], "-q") == 0 )
         quiet = true;
      // other arguments are omitted in Solver
   }

   /***********************************
    * create log file message handler *
    ***********************************/
   if( paraParams->getBoolParamValue(UG::Quiet) )
   {
      // SCIP_CALL_ABORT( SCIPsetMessagehdlr(NULL) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, NULL, TRUE, FALSE), TRUE) );
      SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
      SCIPsetMessagehdlrQuiet(scip, TRUE);
#endif
   }
   else
   {
      if( logname != NULL || quiet  )
      {
         if( logname != NULL )
         {
            std::ostringstream os;
            os << logname << comm->getRank();
            logfile = fopen(os.str().c_str(), "a"); // append to log file */
            if( logfile == NULL )
            {
               THROW_LOGICAL_ERROR3("cannot open log file <", logname, "> for writing");
            }
         }
         SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, logfile, quiet, FALSE), TRUE) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(scip, messagehdlr) );
         SCIP_CALL_ABORT( SCIPmessagehdlrRelease(&messagehdlr));
#endif
      }
   }

   DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
   scipDiffParamSetRoot = scipParaComm->createScipDiffParamSet();
   scipDiffParamSetRoot->bcast(comm, 0);    /** this bcast is sent as SolverInitializationMessage */
   scipDiffParamSet = scipParaComm->createScipDiffParamSet();
   scipDiffParamSet->bcast(comm, 0);    /** this bcast is sent as SolverInitializationMessage */
   int tempIsWarmStarted;
   comm->bcast(&tempIsWarmStarted, 1, UG::ParaINT, 0);
   warmStarted = (tempIsWarmStarted == 1);
   comm->bcast(&globalBestIncumbentValue, 1, UG::ParaDOUBLE, 0);
   if( paraParams->getBoolParamValue( UG::DistributeBestPrimalSolution ) )
   {
      int solutionExists = 0;
      comm->bcast(&solutionExists, 1, UG::ParaINT, 0);
      if( solutionExists )
      {
         globalBestIncumbentSolution = paraComm->createParaSolution();
         globalBestIncumbentSolution->bcast(comm, 0);
      }
   }

   /** set parameters for SCIP: this values are reseted before solving */
   /* move to the place after problem has been created because the parameters may be changed.
   scipDiffParamSet->setParametersInScip(scip);
   SCIP_Real epsilon;
   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon));
   eps = epsilon;
   */ 

   char *isolname = 0;
   for( int i = 3; i < argc; ++i )   /** the first argument is runtime parameter file for ParaSCIP */
                                     /** the second argument is problem file name */
   {
      if( strcmp(argv[i], "-sl") == 0 )
      {
         i++;
         if( i < argc )
         {
            settingsNameLC = argv[i];
            break;
         }
         else
         {
            std::cerr << "missing settings filename after parameter '-sl'" << std::endl;
            exit(1);
         }
      }
      else if ( strcmp(argv[i], "-isol") == 0 )
      {
         i++;
         if( i < argc )
         {
            isolname = argv[i];
          }
          else
          {
             std::cerr << "missing settings filename after parameter '-isol'" << std::endl;
             exit(1);
          }
      }
   }

   /** create problem */
   scipParaInstance->createProblem(scip, 
                                   paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                   paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                   paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                   scipDiffParamSetRoot,
                                   scipDiffParamSet,
                                   settingsNameLC,
                                   isolname
                                  );

   if( SCIPgetStage(scip) == SCIP_STAGE_INIT &&  paraParams->getIntParamValue(UG::InstanceTransferMethod) == 2)
   {
      delete paraComm;
      exit(0);    // the problem should be solved in LC
   }

   /** set parameters for SCIP: this values are reseted before solving */
   scipDiffParamSet->setParametersInScip(scip);
   setOriginalMaxRestart();
   SCIP_Real epsilon;
   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon));
   eps = epsilon;

   saveOrgProblemBounds();
   if( paraParams->getBoolParamValue(UG::CheckEffectOfRootNodePreprocesses) )
   {
      /* initialize SCIP to check root solvability */
      SCIP_CALL_ABORT( SCIPcreate(&scipToCheckEffectOfRootNodeProcesses) );
      /* include default SCIP plugins */
      SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scipToCheckEffectOfRootNodeProcesses) );
      /* include scipParaConshdlr plugins */
      scipParaInstance->createProblem(scipToCheckEffectOfRootNodeProcesses, 
                                      paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                      paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                      paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                      scipDiffParamSetRoot,
                                      scipDiffParamSet,
                                      settingsNameLC,
                                      isolname
                                     );
      scipDiffParamSet->setParametersInScip(scipToCheckEffectOfRootNodeProcesses);
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrestarts", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrounds", 0) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/linear/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/and/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/logicor/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/setppc/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "propagating/probing/maxprerounds", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/feaspump/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/rens/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxcutsroot", 100) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxroundsroot", 5) );
   }
   delete paraInstance;
   paraInstance = 0;

   SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)) );
//   if( paraComm->getRank() == 1 )
//   {
//      std::cout << "*** set memory limit to " << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) << " for each SCIP ***" << std::endl;
//   }

   /** save original priority of changing node selector */
   assert(scip);
   saveOriginalPriority();

   if( !paraParams->getBoolParamValue(UG::Deterministic) )
   {
      interruptMsgMonitor = new ScipParaInterruptMsgMonitor(scipParaComm, this);
      // interruptMsgMonitorThread = std::thread( runInterruptMsgMonitorThread, interruptMsgMonitor );
      std::thread t( runInterruptMsgMonitorThread, interruptMsgMonitor );
      t.detach();
   }
}

ScipParaSolver::ScipParaSolver(
      int argc,
      char **argv,
      UG::ParaComm     *comm,
      UG::ParaParamSet *inParaParamSet,
      UG::ParaInstance *inParaInstance,
      UG::ParaDeterministicTimer *inDetTimer,
      double timeOffset,
      bool thread
      ) : BbParaSolver(argc, argv, N_SCIP_TAGS, comm, inParaParamSet, inParaInstance, inDetTimer),
      messagehdlr(0),
      logfile(0),
      originalParamSet(0),
      conflictConsList(0),
      userPlugins(0),
      commPointHdlr(0),
      nodesel(0),
#if SCIP_APIVERSION >= 101
      selfSplitNodesel(0),
#endif
      scipPropagator(0),
      interruptMsgMonitor(0),
      originalPriority(0),
      orgMaxRestart(0),
      nOrgVars(0),
      nOrgVarsInSolvers(0),
      orgVarLbs(0),
      orgVarUbs(0),
      tightenedVarLbs(0),
      tightenedVarUbs(0),
      mapToOriginalIndecies(0),
      mapToSolverLocalIndecies(0),
      mapToProbIndecies(0),
      // stuffingMaxrounds(0),
      // domcolMaxrounds(0),
      // dualcompMaxrounds(0),
      // dualinferMaxrounds(0),
      // dualaggMaxrounds(0),
      // abspowerDualpresolve(0),
      // andDualpresolving(0),
      // cumulativeDualpresolve(0),
      // knapsackDualpresolving(0),
      // linearDualpresolving(0),
      // setppcDualpresolving(0),
      // logicorDualpresolving(0),
      miscAllowdualreds(0),
      nAddedConss(0),
      addedConss(0),
      addedDualCons(0),
      settingsNameLC(0),
      fiberSCIP(true),
      quiet(false),
      collectingModeIsProhibited(false),
      problemFileName(0),
      orgFeastol(0.0),
      orgLpfeastol(0.0),
      copyIncreasedVariables(false)
{
   assert(thread);  // This is a constructor for threads parallel version

   // ScipMessageHandlerFunctionPointer *scipMessageHandler = reinterpret_cast<ScipMessageHandlerFunctionPointer *>(messageHandler);
   // no additional message handlers

   char* logname = NULL;

   ScipParaInstance *scipParaInstance = dynamic_cast<ScipParaInstance *>(paraInstance);

   paraTimer->setOffset(timeOffset);

   /* Initialize the SCIP environment */
   /*********
    * Setup *
    *********/
   if( paraParams->getIntParamValue(UG::InstanceTransferMethod) == 0 )
   {
      /* copy SCIP environment */
      scip = scipParaInstance->getScip();
      SCIP_CALL_ABORT( SCIPresetParams(scip) );  // if LC parameter settings are applied,
                                                 // it is necessary to reset them
   }
   else
   {
      SCIP_CALL_ABORT( SCIPcreate(&scip) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) ); // always wall clock time
      if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
      {
         double timeRemains =  std::max( 0.0, (paraParams->getRealParamValue(UG::TimeLimit) - paraTimer->getElapsedTime() + 3.0) );  // 3.0: timming issue
         SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
      }
      SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );
      /** user include plugins */
      includeUserPlugins(scip);
   }

   /* include communication point handler */
   ScipParaObjLimitUpdator *updator = new ScipParaObjLimitUpdator(scip,this);
   commPointHdlr = new ScipParaObjCommPointHdlr(paraComm, this, updator);
   SCIP_CALL_ABORT( SCIPincludeObjEventhdlr(scip, commPointHdlr, TRUE) );
   SCIP_CALL_ABORT( SCIPincludeObjHeur(scip, updator, TRUE) );

   /* include propagator */
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      scipPropagator = new ScipParaObjProp(paraComm, this);
      SCIP_CALL_ABORT( SCIPincludeObjProp(scip, scipPropagator, TRUE) );
      saveOriginalSettings();
      dropSettingsForVariableBoundsExchnage();
   }

   /* include node selector */
   nodesel = new ScipParaObjNodesel(this);
   SCIP_CALL_ABORT( SCIPincludeObjNodesel(scip, nodesel, TRUE) );
#if SCIP_APIVERSION >= 101
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) ==  3 )   // self-split ramp-up
   {
      selfSplitNodesel = new ScipParaObjSelfSplitNodesel(
            paraComm->getRank() - 1,
            paraComm->getSize() - 1,
            paraParams->getIntParamValue(UG::SelfSplitTreeDepth),
            paraComm,
            this,
            scip
            );
      SCIP_CALL_ABORT( SCIPincludeObjNodesel(scip, selfSplitNodesel, TRUE) );
   }
#endif
   /* include branch rule plugins */
   SCIP_CALL_ABORT( SCIPincludeObjBranchrule(scip, new ScipParaObjBranchRule(this), TRUE) );

   if( inParaParamSet->getBoolParamValue(UG::TransferConflictCuts) )
   {
      conflictConsList = new std::list<LocalNodeInfoPtr>;
      SCIP_CONFLICTHDLRDATA *conflictHdrData = reinterpret_cast< SCIP_CONFLICTHDLRDATA * >(this);
      /* create conflict handler to collects conflicts */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
      SCIP_CALL_ABORT( SCIPincludeConflicthdlr(scip, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            NULL, NULL, NULL, NULL, NULL, NULL, conflictExecCollector, conflictHdrData) );
#else
      SCIP_CALL_ABORT( SCIPincludeConflicthdlrBasic(scip, NULL, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            conflictExecCollector, conflictHdrData) );
#endif
   }

   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/feastol", &orgFeastol ) );
   if( SCIP_APIVERSION < 61 )
   {
      SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/lpfeastol", &orgLpfeastol ) );
   }

   /********************
    * Parse parameters *
    ********************/
   for( int i = 3; i < argc; ++i )   /** the first argument is runtime parameter file for ParaSCIP */
   {
      if( strcmp(argv[i], "-l") == 0 )
      {
         i++;
         if( i < argc )
            logname = argv[i];
         else
         {
            THROW_LOGICAL_ERROR1("missing log filename after parameter '-l'");
         }
      }
      else if( strcmp(argv[i], "-q") == 0 )
         quiet = true;
      // other arguments are omitted in Solver
   }

   /***********************************
    * create log file message handler *
    ***********************************/
   if( paraParams->getBoolParamValue(UG::Quiet) )
   {
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, NULL, TRUE, FALSE), TRUE) );
      SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
      SCIPsetMessagehdlrQuiet(scip, TRUE );
#endif
   }
   else
   {
      if( logname != NULL || quiet )
      {
         paraComm->lockApp();    // if solver runs as thread, this lock is necessary
         if( logname != NULL )
         {
            std::ostringstream os;
            os << logname << comm->getRank();
            logfile = fopen(os.str().c_str(), "a"); // append to log file */
            if( logfile == NULL )
            {
               THROW_LOGICAL_ERROR3("cannot open log file <", logname, "> for writing");
            }
         }
         SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, logfile, quiet, TRUE), TRUE) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(scip, messagehdlr) );
         SCIP_CALL_ABORT( SCIPmessagehdlrRelease(&messagehdlr));
#endif
         paraComm->unlockApp();
      }
   }

   DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
   scipDiffParamSetRoot = scipParaComm->createScipDiffParamSet();
   scipDiffParamSetRoot->bcast(comm, 0);    /** this bcast is sent as SolverInitializationMessage */
   scipDiffParamSet = scipParaComm->createScipDiffParamSet();
   scipDiffParamSet->bcast(comm, 0);    /** this bcast is sent as SolverInitializationMessage */
   int tempIsWarmStarted;
   comm->bcast(&tempIsWarmStarted, 1, UG::ParaINT, 0);
   warmStarted = (tempIsWarmStarted == 1);
   comm->bcast(&globalBestIncumbentValue, 1, UG::ParaDOUBLE, 0);

   if( paraParams->getBoolParamValue(UG::NoUpperBoundTransferInRacing) )
   {
      int solutionExists = 0;
      paraComm->bcast(&solutionExists, 1, UG::ParaINT, 0);
   }
   else
   {
      if( paraParams->getBoolParamValue( UG::DistributeBestPrimalSolution ) )
      {
         int solutionExists = 0;
         comm->bcast(&solutionExists, 1, UG::ParaINT, 0);
         if( solutionExists )
         {
            globalBestIncumbentSolution = paraComm->createParaSolution();
            globalBestIncumbentSolution->bcast(comm, 0);
         }
      }
   }

   /** set parameters for SCIP: this values are reseted before solving */
   /* move to the place after problem has been created because the parameters may be changed.
   scipDiffParamSet->setParametersInScip(scip);
   SCIP_Real epsilon;
   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon));
   eps = epsilon;
   */ 

   char *isolname = 0;
   for( int i = 3; i < argc; ++i )   /** the first argument is runtime parameter file for ParaSCIP */
                                     /** the second argument is problem file name */
   {
      if( strcmp(argv[i], "-sl") == 0 )
      {
         i++;
         if( i < argc )
         {
            settingsNameLC = argv[i];
            break;
         }
         else
         {
            std::cerr << "missing settings filename after parameter '-sl'" << std::endl;
            exit(1);
         }
      }
      else if ( strcmp(argv[i], "-isol") == 0 )
      {
         i++;
         if( i < argc )
         {
            isolname = argv[i];
          }
          else
          {
             std::cerr << "missing settings filename after parameter '-isol'" << std::endl;
             exit(1);
          }
      }
   }

   /** create problem */
   scipParaInstance->createProblem(scip, 
                                   paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                   paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                   paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                   scipDiffParamSetRoot,
                                   scipDiffParamSet,
                                   settingsNameLC,
                                   isolname
                                  );

   /** set parameters for SCIP: this values are reseted before solving */
   scipDiffParamSet->setParametersInScip(scip);
   SCIP_Real epsilon;
   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon));
   eps = epsilon;

   saveOrgProblemBounds();
   if( paraParams->getBoolParamValue(UG::CheckEffectOfRootNodePreprocesses) )
   {
      /* initialize SCIP to check root solvability */
      SCIP_CALL_ABORT( SCIPcreate(&scipToCheckEffectOfRootNodeProcesses) );
      /* include default SCIP plugins */
      SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scipToCheckEffectOfRootNodeProcesses) );
      /* include scipParaConshdlr plugins */
      scipParaInstance->createProblem(scipToCheckEffectOfRootNodeProcesses, 
                                      paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                      paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                      paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                      scipDiffParamSetRoot,
                                      scipDiffParamSet,
                                      settingsNameLC,
                                      isolname
                                     );
      scipDiffParamSet->setParametersInScip(scipToCheckEffectOfRootNodeProcesses);
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrestarts", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrounds", 0) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/linear/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/and/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/logicor/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/setppc/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "propagating/probing/maxprerounds", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/feaspump/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/rens/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxcutsroot", 100) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxroundsroot", 5) );
   }
   delete paraInstance;
   paraInstance = 0;

#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
   assert( memoryLimitOfSolverSCIP > 0.0 );
   SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", memoryLimitOfSolverSCIP) );
#else
   SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", (dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit))/(paraComm->getSize()*SCIP_MEMORY_COPY_FACTOR))); // LC has SCIP env.
#endif
//   if( paraComm->getRank() == 1 )
//   {
//      std::cout << "*** set memory limit to " << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)/(paraComm->getSize()*SCIP_MEMORY_COPY_FACTOR) << " for each SCIP ***" << std::endl;
//   }

   /** save original priority of changing node selector */
   saveOriginalPriority();

   if( !paraParams->getBoolParamValue(UG::Deterministic) )
   {
      interruptMsgMonitor = new ScipParaInterruptMsgMonitor(scipParaComm, this);
      interruptMsgMonitorThread = std::thread( runInterruptMsgMonitorThread, interruptMsgMonitor );
      //std::thread t( runInterruptMsgMonitorThread, interruptMsgMonitor );
      //t.detach();
   }
}

ScipParaSolver::~ScipParaSolver(
      )
{
   if( interruptMsgMonitor )
   {
      interruptMsgMonitor->terminate();
   }

   /** delete scip diff param sets */
   if( scipDiffParamSetRoot ) delete scipDiffParamSetRoot;
   if( scipDiffParamSet ) delete scipDiffParamSet;
   if( userPlugins ) delete userPlugins;

   /** reset message handler */
   // message handler is mangaed within scip. It is freed at SCIPfree
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
   if( messagehdlr )
   {
      SCIP_CALL_ABORT( SCIPsetDefaultMessagehdlr() );
      SCIP_CALL_ABORT( SCIPfreeObjMessagehdlr(&messagehdlr) );
   }
#endif

   /* free SCIP environment */
   if( paraParams->getBoolParamValue(UG::CheckEffectOfRootNodePreprocesses) )
   {
      SCIP_CALL_ABORT( SCIPfree(&scipToCheckEffectOfRootNodeProcesses) );
   }
   SCIP_CALL_ABORT( SCIPfree(&scip) );

   /** close log file */
   if( logfile ) fclose(logfile);

   if( conflictConsList && conflictConsList->size() > 0 )
   {
      int nConfilcts = conflictConsList->size();
      for(int i = 0; i < nConfilcts; i++ )
      {
         assert(!conflictConsList->empty());
         LocalNodeInfo *info= conflictConsList->front();
         conflictConsList->pop_front();
         if( info->linearCoefs ) delete[] info->linearCoefs;
         if( info->idxLinearCoefsVars ) delete[] info->idxLinearCoefsVars;
         delete info;
      }
   }

   if( conflictConsList ) delete conflictConsList;

   if( orgVarLbs ) delete [] orgVarLbs;
   if( orgVarUbs ) delete [] orgVarUbs;
   if( tightenedVarLbs ) delete [] tightenedVarLbs;
   if( tightenedVarUbs ) delete [] tightenedVarUbs;
   if( mapToOriginalIndecies ) delete[] mapToOriginalIndecies;
   if( mapToSolverLocalIndecies ) delete[] mapToSolverLocalIndecies;
   if( mapToProbIndecies) delete [] mapToProbIndecies;
   if( addedConss ) delete [] addedConss;

   // DON'T FREE problemFileName

   ///
   /// The following code was in Solver base class before
   ///
   if( terminationMode != UG::InterruptedTerminationMode 
      && terminationMode != UG::TimeLimitTerminationMode )
   {
      iReceiveMessages();
   }

   if( paraParams->getBoolParamValue(UG::Deterministic) )
   {
      for(;;)
      {
         // if( !( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 && paraParams->getRealParamValue(UG::TimeLimit) <= paraTimer->getElapsedTime() ) )
         // {
            while( !waitToken(paraComm->getRank()) )
            {
               iReceiveMessages();
               // if( ( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 
               //       && paraParams->getRealParamValue(UG::TimeLimit) <= paraTimer->getElapsedTime() ) )
               // {
               //    break;
               // }
	    }
            // paraDetTimer->update(1.0);
            // previousCommTime = paraDetTimer->getElapsedTime();
              if( paraComm->passTermToken(paraComm->getRank()) ) break;
         // }
      }
   }

   double stopTime = paraTimer->getElapsedTime();
   idleTimeAfterLastParaTask = stopTime - previousStopTime - ( idleTimeToWaitToken - previousIdleTimeToWaitToken );
   int interrupted = terminationMode == UG::InterruptedTerminationMode ? 1 : 0;
   int calcTermState = terminationMode == UG::InterruptedTerminationMode ? UG::CompTerminatedByInterruptRequest : UG::CompTerminatedNormally;

   double detTime = -1.0;
   if( paraDetTimer )
   {
      detTime = paraDetTimer->getElapsedTime();
   }

   DEF_BB_PARA_COMM(bbParaComm, paraComm);

   UG::BbParaSolverTerminationState *paraSolverTerminationState = dynamic_cast<UG::BbParaSolverTerminationState *>(bbParaComm->createParaSolverTerminationState(
          interrupted,
          paraComm->getRank(),
          totalNSolved,
          minNSolved,
          maxNSolved,
          totalNSent,
          totalNImprovedIncumbent,
          nParaTasksReceived,
          nParaTasksSolved,
          nParaNodesSolvedAtRoot,
          nParaNodesSolvedAtPreCheck,
          nTransferredLocalCutsFromSolver,
          minTransferredLocalCutsFromSolver,
          maxTransferredLocalCutsFromSolver,
          nTransferredBendersCutsFromSolver,
          minTransferredBendersCutsFromSolver,
          maxTransferredBendersCutsFromSolver,
          nTotalRestarts,
          minRestarts,
          maxRestarts,
          nTightened,
          nTightenedInt,
          calcTermState,
          stopTime,
          idleTimeToFirstParaTask,
          idleTimeBetweenParaTasks,
          idleTimeAfterLastParaTask,
          idleTimeToWaitNotificationId,
          idleTimeToWaitAckCompletion,
          idleTimeToWaitToken,
          totalRootNodeTime,
          minRootNodeTime,
          maxRootNodeTime,
          detTime ));
   paraSolverTerminationState->send(paraComm, 0, UG::TagTerminated);

   delete paraSolverTerminationState;

   if( interruptMsgMonitor )
   {
      interruptMsgMonitorThread.join();
   }

}

void
ScipParaSolver::writeCurrentTaskProblem(
      const std::string& filename
      )
{
   FILE *file = fopen(filename.c_str(),"a");
   if( !file )
   {
      std::cout << "file : " << filename << "cannot open." << std::endl;
      abort();
   }
   SCIP_CALL_ABORT( SCIPprintTransProblem(scip, file, "cip", FALSE) );
}

void
ScipParaSolver::solveToCheckEffectOfRootNodePreprocesses(
      )
{
   SCIP *backupScip = scip;
   scip = scipToCheckEffectOfRootNodeProcesses;
   createSubproblem();
   /** set cutoff value */
   SCIP_CALL_ABORT( SCIPsetObjlimit(scip, globalBestIncumbentValue) );
   /** solve */
   SCIP_CALL_ABORT( SCIPsolve(scip) );
   nSolvedWithNoPreprocesses = SCIPgetNNodes(scip);
   freeSubproblem();
   scip = backupScip;
}

void
ScipParaSolver::tryNewSolution(
      UG::ParaSolution *sol
      )
{

   if( SCIPgetStage(scip) <= SCIP_STAGE_TRANSFORMING || SCIPgetStage(scip) >= SCIP_STAGE_SOLVED  )
   {
      THROW_LOGICAL_ERROR1("invalid tyrNewSolution");
   }

   ScipParaSolution *tempSol = dynamic_cast< ScipParaSolution * >(sol);
   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */

   if( isCopyIncreasedVariables() )
   {
      return;
      //
      // It would be good to install the solution to SCIP as the PartialSolution.
      // However, currently SCIP does not support this.
      // In the future, this might be better to change.
      //
      // if( SCIPcreatePartialSol(scip, &newsol, 0) != SCIP_OKAY )  // SCIP_CALL_ABORT ????
      // {
      //    return;
      // }
   }
   else
   {
      if( SCIPcreateOrigSol(scip, &newsol, 0) != SCIP_OKAY )  // SCIP_CALL_ABORT ????
      {
         return;
      }
   }


   SCIP_VAR** vars = SCIPgetOrigVars(scip);
   SCIP_Real* vals = new SCIP_Real[tempSol->getNVars()]();
   int i;
   for(i = 0; i < tempSol->getNVars(); i++ )
   {
      int probindex = tempSol->indexAmongSolvers(i);
      if( mapToProbIndecies )
      {
         probindex = mapToProbIndecies[tempSol->indexAmongSolvers(i)];
      }
      // assert( probindex >= 0 );
      if( probindex >= 0 )
      {
         vals[probindex] = tempSol->getValues()[i];
      }
      // /* skip inactive varibales */
      // if( probindex < 0 )
      //   continue;
      // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[probindex], tempSol->getValues()[i]) );
   }
   SCIP_CALL_ABORT( SCIPsetSolVals(scip, newsol, tempSol->getNVars(), vars, vals) );
   delete [] vals;
   // if( i != tempSol->getNVars() )
   // {
   //   /** the given solution should be generated in original space,
   //    * therefore the solution values cannot use for ParaSCIP
   //    */
   //   SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
   //   // delete tempSol;  // this case, DO NOT DELETE tempSol.
   //   return;
   // }

#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
   if( SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED || SCIPgetStage(scip) == SCIP_STAGE_PRESOLVED  )
#else
   if( SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED ||
		   SCIPgetStage(scip) == SCIP_STAGE_PRESOLVED ||
		   SCIPgetStage(scip) == SCIP_STAGE_INITPRESOLVE )
#endif
   {
      SCIP_Bool success;
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
      // std::cout << "Rank " << paraComm->getRank() << ": success = " << success << std::endl;
      // if( !success ) abort();
   }
   else
   {
      SCIP_CALL_ABORT( SCIPheurPassSolTrySol(scip, SCIPfindHeur(scip, "trysol"), newsol) );
      SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
   }
}

void
ScipParaSolver::setLightWeightRootNodeProcess(
      )
{
   lightWeightRootNodeComputation = true;
   if( !originalParamSet )
   {
      DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
      originalParamSet = scipParaComm->createScipDiffParamSet(scip);;
   }
   SCIP_CALL_ABORT( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_FAST, TRUE) );
   SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_FAST, TRUE) );
   SCIP_CALL_ABORT( SCIPsetSeparating(scip, SCIP_PARAMSETTING_FAST, TRUE) );
}

void
ScipParaSolver::setOriginalRootNodeProcess(
      )
{
   assert(originalParamSet);
   originalParamSet->setParametersInScip(scip);
   lightWeightRootNodeComputation = false;

#if SCIP_VERSION >= 320
   setBakSettings();
#endif

}

// static int id = 0;

void
ScipParaSolver::writeSubproblem(
      )
{
   std::string subcipprefix("SolverCip");
   std::string subcipfilename;
   std::ostringstream oss;
   oss << subcipprefix << paraComm->getRank();
   // oss << subcipprefix << paraComm->getRank() << "." << id++;;
   subcipfilename = oss.str();
   subcipfilename += ".cip";
   SCIP_CALL_ABORT( SCIPwriteOrigProblem(scip, subcipfilename.c_str(), "cip", FALSE) );
#ifdef UG_DEBUG_SOLUTION
   if( ( !currentTask->getDiffSubproblem() ) ||
         currentTask->getDiffSubproblem()->isOptimalSolIncluded() )
   {
      std::cout << "** " << subcipfilename << " contains optimal solution." << std::endl;
   }
   else
   {
      std::cout << "** " << subcipfilename << " does NOT contain optimal solution." << std::endl;
   }
#endif
   subcipfilename = oss.str();
   subcipfilename += "-t.cip";
   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL_ABORT( SCIPwriteTransProblem(scip, subcipfilename.c_str(), "cip", FALSE) );
   }
   char name[SCIP_MAXSTRLEN];
   (void)SCIPsnprintf(name, SCIP_MAXSTRLEN, "SolverCip%d.set", paraComm->getRank());
   SCIP_CALL_ABORT( SCIPwriteParams(scip, name, TRUE, FALSE) );
}

void
ScipParaSolver::saveOrgProblemBounds(
      )
{
   assert(paraInstance);
   ScipParaInstance *scipParaInstance = dynamic_cast<ScipParaInstance*>(paraInstance);
   nOrgVars = scipParaInstance->getNVars();           // number of original variables in LC
   nOrgVarsInSolvers = SCIPgetNOrigVars(scip);    // maybe increaded
   assert( nOrgVarsInSolvers == scipParaInstance->getVarIndexRange() );
   // if( nOrgVars == 0 ) nOrgVars = nOrgVarsInSolver;
   // nOrgVars = nOrgVarsInSolver;
   // assert( nOrgVars <= paraInstance->getVarIndexRange() );  // when copy generated additional variables, this does not hold
   orgVarLbs = new SCIP_Real[nOrgVarsInSolvers];
   orgVarUbs = new SCIP_Real[nOrgVarsInSolvers];
   if( scipParaInstance->isOriginalIndeciesMap() )
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
            paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
            paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
      {
         tightenedVarLbs = new double[nOrgVarsInSolvers];
         tightenedVarUbs = new double[nOrgVarsInSolvers];
         // for(int v = 0; v < paraInstance->getVarIndexRange(); v++)
         // for( int v = 0; v < nOrgVarsInSolvers ; v++ )
         for( int v = 0; v < nOrgVars ; v++ )
         {
            // int orgIndex = scipParaInstance->getOrigProbIndex(SCIPvarGetIndex(vars[v]));
            // assert(orgIndex >= 0);
            tightenedVarLbs[v] = orgVarLbs[v] = scipParaInstance->getVarLb(v);
            tightenedVarUbs[v] = orgVarUbs[v] = scipParaInstance->getVarUb(v);
            // std::cout << scipParaInstance->getVarName(v) << " orgVarLbs[" << v << "] = " << orgVarLbs[v] << std::endl;
            // std::cout << scipParaInstance->getVarName(v) << " orgVarUbs[" << v << "] = " << orgVarUbs[v] << std::endl;
         }
      }
      else
      {
         // for(int v = 0; v < paraInstance->getVarIndexRange(); v++)
         // for( int v = 0; v < nOrgVarsInSolvers ; v++ )
         for( int v = 0; v < nOrgVars; v++ )
         {
            // int orgIndex = scipParaInstance->getOrigProbIndex(SCIPvarGetIndex(vars[v]));
            // assert(orgIndex >= 0);
            orgVarLbs[v] = scipParaInstance->getVarLb(v);
            orgVarUbs[v] = scipParaInstance->getVarUb(v);
            // std::cout << scipParaInstance->getVarName(v) << " orgVarLbs[" << v << "] = " << orgVarLbs[v] << std::endl;
            // std::cout << scipParaInstance->getVarName(v) << " orgVarUbs[" << v << "] = " << orgVarUbs[v] << std::endl;
         }
      }
      assert( mapToOriginalIndecies == 0 && mapToProbIndecies == 0 );
      mapToOriginalIndecies =  scipParaInstance->extractOrigProbIndexMap();
      mapToSolverLocalIndecies = scipParaInstance->extractSolverLocalIndexMap();
      mapToProbIndecies = new int[SCIPgetNTotalVars(scip)]; // need to allocate enough for SCIPvarGetIndex(scip)
      for( int i = 0; i < SCIPgetNTotalVars(scip); i++ )
      {
         mapToProbIndecies[i] = -1;
      }
      for( int i = 0; i < nOrgVarsInSolvers; i++ )
      {
         mapToProbIndecies[SCIPvarGetIndex(vars[i])] = i;
      }
      if ( dynamic_cast<ScipParaInstance *>(paraInstance)->isCopyIncreasedVariables() )
      {
         copyIncreasedVariables = true;
      }
   }
   else
   {
      if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
            paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
            paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
      {
         tightenedVarLbs = new double[nOrgVars];
         tightenedVarUbs = new double[nOrgVars];
         // for(int v = 0; v < paraInstance->getVarIndexRange(); v++)
         // for( int v = 0; v < nOrgVarsInSolvers; v++ )
         for( int v = 0; v < nOrgVars; v++ )
         {
            tightenedVarLbs[v] = orgVarLbs[v] = scipParaInstance->getVarLb(v);
            tightenedVarUbs[v] = orgVarUbs[v] = scipParaInstance->getVarUb(v);
            // std::cout << scipParaInstance->getVarName(v) << " orgVarLbs[" << v << "] = " << orgVarLbs[v] << std::endl;
            // std::cout << scipParaInstance->getVarName(v) << " orgVarUbs[" << v << "] = " << orgVarUbs[v] << std::endl;
         }
      }
      else
      {
         // for(int v = 0; v < paraInstance->getVarIndexRange(); v++)
         // for( int v = 0; v < nOrgVarsInSolvers; v++ )
         for( int v = 0; v < nOrgVars; v++ )
         {
            orgVarLbs[v] = scipParaInstance->getVarLb(v);
            orgVarUbs[v] = scipParaInstance->getVarUb(v);
            // std::cout << scipParaInstance->getVarName(v) << " orgVarLbs[" << v << "] = " << orgVarLbs[v] << std::endl;
            // std::cout << scipParaInstance->getVarName(v) << " orgVarUbs[" << v << "] = " << orgVarUbs[v] << std::endl;
         }
      }
   }
}

void
ScipParaSolver::reinitialize(
      )
{
   /*************************************
   ** This function does not work well **
   **************************************/
   /****************************
   ** reset original instance  *
   *****************************/
   /* Reinitialize the SCIP environment */
   /* free SCIP environment */
   if( paraParams->getBoolParamValue(UG::CheckEffectOfRootNodePreprocesses) )
   {
      SCIP_CALL_ABORT( SCIPfree(&scipToCheckEffectOfRootNodeProcesses) );
   }
   SCIP_CALL_ABORT( SCIPfree(&scip) );
   if( orgVarLbs ) delete [] orgVarLbs;
   if( orgVarUbs ) delete [] orgVarUbs;
   /*********
    * Setup *
    *********/
   if( fiberSCIP )
   {
      paraInstance = paraComm->createParaInstance();
      // setUserPlugins(paraInstance);           //  instance data should not be read from original data file
      paraInstance->bcast(paraComm, 0, paraParams->getIntParamValue(UG::InstanceTransferMethod));
      /* copy SCIP environment */
      ScipParaInstance *scipParaInstance = dynamic_cast<ScipParaInstance*>(paraInstance);
      scip = scipParaInstance->getScip();
      SCIP_CALL_ABORT( SCIPresetParams(scip) );  // if LC parameter settings are applied,
                                                 // it is necessary to reset them
   }
   else
   {
      paraInstance = paraComm->createParaInstance();
      ::setUserPlugins(paraInstance);
      ScipParaInstance *scipParaInstance = dynamic_cast<ScipParaInstance*>(paraInstance);
      scipParaInstance->setFileName(problemFileName);
      paraInstance->bcast(paraComm, 0, paraParams->getIntParamValue(UG::InstanceTransferMethod));
      /* initialize SCIP */
      SCIP_CALL_ABORT( SCIPcreate(&scip) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) ); // always wall clock time
      if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
      {
         double timeRemains =  std::max( 0.0, (paraParams->getRealParamValue(UG::TimeLimit) - paraTimer->getElapsedTime() + 3.0) );  // 3.0: timming issue
         SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
      }
      /* include default SCIP plugins */
      SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );
      /** user include plugins */
      includeUserPlugins(scip);
   }

   /* include communication point handler */
   ScipParaObjLimitUpdator *updator = new ScipParaObjLimitUpdator(scip,this);
   commPointHdlr = new ScipParaObjCommPointHdlr(paraComm, this, updator);
   SCIP_CALL_ABORT( SCIPincludeObjEventhdlr(scip, commPointHdlr, TRUE) );
   SCIP_CALL_ABORT( SCIPincludeObjHeur(scip, updator, TRUE) );

   /* include propagator */
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      scipPropagator = new ScipParaObjProp(paraComm, this);
      SCIP_CALL_ABORT( SCIPincludeObjProp(scip, scipPropagator, TRUE) );
      saveOriginalSettings();
      dropSettingsForVariableBoundsExchnage();
   }

   /* include node selector */
   nodesel = new ScipParaObjNodesel(this);
   SCIP_CALL_ABORT( SCIPincludeObjNodesel(scip, nodesel, TRUE) );
   /* include branch rule plugins */
   SCIP_CALL_ABORT( SCIPincludeObjBranchrule(scip, new ScipParaObjBranchRule(this), TRUE) );

   if( paraParams->getBoolParamValue(UG::TransferConflictCuts) )
   {
      assert(conflictConsList);
      delete conflictConsList;
      conflictConsList = new std::list<LocalNodeInfoPtr>;
      SCIP_CONFLICTHDLRDATA *conflictHdrData = reinterpret_cast< SCIP_CONFLICTHDLRDATA * >(this);
      /* create conflict handler to collects conflicts */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
      SCIP_CALL_ABORT( SCIPincludeConflicthdlr(scip, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            NULL, NULL, NULL, NULL, NULL, NULL, conflictExecCollector, conflictHdrData) );
#else
      SCIP_CALL_ABORT( SCIPincludeConflicthdlrBasic(scip, NULL, CONFLICTHDLR_NAME, CONFLICTHDLR_DESC, CONFLICTHDLR_PRIORITY,
            conflictExecCollector, conflictHdrData) );
#endif
   }

   if( paraParams->getBoolParamValue(UG::Quiet) )
   {
      // SCIP_CALL_ABORT( SCIPsetMessagehdlr(NULL) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, NULL, TRUE, FALSE), TRUE) );
      SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
      SCIPsetMessagehdlrQuiet(scip, TRUE);
#endif
   }
   else
   {
      if( logfile != NULL || quiet  )
      {
         SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, new ScipParaObjMessageHdlr(paraComm, logfile, quiet, FALSE), TRUE) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(scip, messagehdlr) );
         SCIP_CALL_ABORT( SCIPmessagehdlrRelease(&messagehdlr));
#endif
      }
   }

   ScipParaInstance *scipParaInstance = dynamic_cast<ScipParaInstance*>(paraInstance);
   scipParaInstance->createProblem(scip,
                                   paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                   paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                   paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                   scipDiffParamSetRoot,
                                   scipDiffParamSet,
                                   settingsNameLC,
                                   NULL
                                  );
   saveOrgProblemBounds();
   if( paraParams->getBoolParamValue(UG::CheckEffectOfRootNodePreprocesses) )
   {
      scipParaInstance->createProblem(scipToCheckEffectOfRootNodeProcesses,
                                      paraParams->getIntParamValue(UG::InstanceTransferMethod),
                                      paraParams->getBoolParamValue(UG::NoPreprocessingInLC),
                                      paraParams->getBoolParamValue(UG::UseRootNodeCuts),
                                      scipDiffParamSetRoot,
                                      scipDiffParamSet,
                                      settingsNameLC,
                                      NULL
                                     );
      scipDiffParamSet->setParametersInScip(scipToCheckEffectOfRootNodeProcesses);
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrestarts", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "presolving/maxrounds", 0) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/linear/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/and/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/logicor/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scipToCheckEffectOfRootNodeProcesses, "constraints/setppc/presolpairwise", FALSE) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "propagating/probing/maxprerounds", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/feaspump/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "heuristics/rens/freq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxcutsroot", 100) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scipToCheckEffectOfRootNodeProcesses, "separating/maxroundsroot", 5) );
   }
   delete paraInstance;
   paraInstance = 0;
}

void
ScipParaSolver::setOriginalNodeSelectionStrategy(
      )
{
   if( SCIPgetStage(scip) != SCIP_STAGE_SOLVING )
   {
      if( (!paraParams->getBoolParamValue(UG::SetAllDefaultsAfterRacing)) && winnerRacingParams )
      {
         // setWinnerRacingParams(winnerRacingParams);   // winner parameters are set, again
         // std::cout << winnerRacingParams->toString() << std::endl;
      }
      else
      {
         scipDiffParamSet->setParametersInScip(scip);
      }
   }
   commPointHdlr->setOriginalNodeSelectionStrategy();

#if SCIP_VERSION >= 320
   if( currentTask )
   {
      setBakSettings();
   }
#endif
}

void
ScipParaSolver::setBakSettings(
      )
{
   char *bakFileName = NULL;
   SCIP_CALL_ABORT( SCIPgetStringParam(scip,"visual/bakfilename", &bakFileName) );
   if( strcmp(bakFileName,"-") != 0 )
   {
      std::ostringstream os;
      os << bakFileName << "_" << paraComm->getRank();
      SCIP_CALL_ABORT( SCIPsetStringParam(scip,"visual/bakfilename", os.str().c_str() ) );
      SCIP_CALL_ABORT( SCIPsetStringParam(scip,"visual/baknodeprefix", ((currentTask->getTaskId()).toString()+":").c_str() ) );
      SCIP_CALL_ABORT( SCIPsetStringParam(scip,"visual/bakrootinfo", (currentTask->getGeneratorTaskId()).toString().c_str() ) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip,"visual/baktimeoffset", paraTimer->getElapsedTime() ) );
   }
}

int
ScipParaSolver::lbBoundTightened(
      int source,
      int tag
      )
{
   int tightenedIdex;
   double tightenedBound;
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedIdex, 1, UG::ParaINT, source, UG::TagLbBoundTightenedIndex )
         );
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedBound, 1, UG::ParaDOUBLE, source, UG::TagLbBoundTightenedBound )
         );

   if( terminationMode != UG::NoTerminationMode )
   {
      return 0;
   }

   if( mapToProbIndecies )
   {
      assert( mapToSolverLocalIndecies );
      assert(mapToProbIndecies[mapToSolverLocalIndecies[tightenedIdex]] >= 0);
      tightenedIdex = mapToProbIndecies[mapToSolverLocalIndecies[tightenedIdex]];
   }
   assert( SCIPisLE(scip,orgVarLbs[tightenedIdex], tightenedBound) &&
         SCIPisGE(scip,orgVarUbs[tightenedIdex], tightenedBound)  );
   // std::cout << "Rank " << paraComm->getRank() << ": receive tightened lower bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
   SCIP_Var* var = SCIPvarGetTransVar(SCIPgetOrigVars(scip)[tightenedIdex]);
   if( var && SCIPisLT(scip,tightenedVarLbs[tightenedIdex], tightenedBound) && SCIPvarGetStatus(var) != SCIP_VARSTATUS_MULTAGGR )
   {
      // std::cout << "Solver Lb = " << tightenedVarLbs[tightenedIdex]
      // << ", Rank " << paraComm->getRank() << ": try to tighten lower bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
      scipPropagator->addBoundChange(scip, SCIP_BOUNDTYPE_LOWER, tightenedIdex, tightenedBound );
      tightenedVarLbs[tightenedIdex] = tightenedBound;

   }
   
   return 0;
}

int
ScipParaSolver::ubBoundTightened(
      int source,
      int tag
      )
{
   int tightenedIdex;
   double tightenedBound;
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedIdex, 1, UG::ParaINT, source, UG::TagUbBoundTightenedIndex )
         );
   PARA_COMM_CALL(
         paraComm->receive( (void *)&tightenedBound, 1, UG::ParaDOUBLE, source, UG::TagUbBoundTightenedBound )
         );

   if( terminationMode != UG::NoTerminationMode )
   {
      return 0;
   }

   if( mapToProbIndecies )
   {
      assert( mapToSolverLocalIndecies );
      assert(mapToProbIndecies[mapToSolverLocalIndecies[tightenedIdex]] >= 0);
      tightenedIdex = mapToProbIndecies[mapToSolverLocalIndecies[tightenedIdex]];
   }
   assert( SCIPisLE(scip,orgVarLbs[tightenedIdex], tightenedBound) &&
         SCIPisGE(scip,orgVarUbs[tightenedIdex], tightenedBound)  );
   // std::cout << "Rank " << paraComm->getRank() << ": receive tightened upper bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
   SCIP_Var* var = SCIPvarGetTransVar(SCIPgetOrigVars(scip)[tightenedIdex]);
   if( var && SCIPisGT(scip,tightenedVarUbs[tightenedIdex], tightenedBound) && SCIPvarGetStatus(var) != SCIP_VARSTATUS_MULTAGGR )
   {
      // std::cout << "Solver Ub = " << tightenedVarUbs[tightenedIdex]
      // << ", Rank " << paraComm->getRank() << ": try to tighten upper bond. idx = " << tightenedIdex << ", bound = " << tightenedBound << std::endl;
      scipPropagator->addBoundChange(scip, SCIP_BOUNDTYPE_UPPER, tightenedIdex, tightenedBound );
      tightenedVarUbs[tightenedIdex] = tightenedBound;
   }
  
   return 0;
}

/** get number of tightened variables during racing */
int
ScipParaSolver::getNTightened(
      )
{
   if( scipPropagator )
   {
      return scipPropagator->getNtightened();
   }
   else
   {
      if( nTightened > 0 )
      {
         return nTightened;
      }
      else
      {
         return 0;
      }
   }
}

/** get number of tightened integral variables during racing */
int
ScipParaSolver::getNTightenedInt(
      )
{
   if( scipPropagator )
   {
      return scipPropagator->getNtightenedInt();
   }
   else
   {
      if( nTightened > 0 )
      {
         return nTightenedInt;
      }
      else
      {
         return 0;
      }
   }
}

void
ScipParaSolver::saveOriginalSettings(
      )
{
   /* presolvers */
#if 0
   SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/stuffing/maxrounds", &stuffingMaxrounds) );
   SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/domcol/maxrounds", &domcolMaxrounds) );
#if  ( SCIP_VERSION >= 322 || (SCIP_VERSION == 321 && SCIP_SUBVERSION >= 2) )
   SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/dualcomp/maxrounds", &dualcompMaxrounds) );  /*TODO: ok? */
#endif
   SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/dualinfer/maxrounds", &dualinferMaxrounds) ); /*TODO: probably fine */
   // SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/dualagg/maxrounds", &dualaggMaxrounds ) ); // TODO: seems to have no copy callback */
   /* constraint handlers */
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/abspower/dualpresolve", &abspowerDualpresolve) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/and/dualpresolving", &andDualpresolving) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/cumulative/dualpresolve", &cumulativeDualpresolve) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/knapsack/dualpresolving", &knapsackDualpresolving) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/dualpresolving", &linearDualpresolving) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/setppc/dualpresolving", &setppcDualpresolving) );
   SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/logicor/dualpresolving", &logicorDualpresolving) );
#endif
   if ( isRacingStage() && paraComm->getRank() != 1 )
   {
#if SCIP_APIVERSION > 34
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "misc/allowstrongdualreds", &miscAllowdualreds) );
#else
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "misc/allowdualreds", &miscAllowdualreds) );
#endif
   }
}

void
ScipParaSolver::dropSettingsForVariableBoundsExchnage(
      )
{
   /* presolvers */
#if 0
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/stuffing/maxrounds", 0) );
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/domcol/maxrounds", 0) );
#if  ( SCIP_VERSION >= 322 || (SCIP_VERSION == 321 && SCIP_SUBVERSION >= 2) )
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualcomp/maxrounds", 0) );  /*TODO: ok? */
#endif
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualinfer/maxrounds", 0) ); /*TODO: probably fine */
   // SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualagg/maxrounds", 0) );  // TODO: seems to have no copy callback */
   /* constraint handlers */
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/abspower/dualpresolve", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/and/dualpresolving", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/cumulative/dualpresolve", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/knapsack/dualpresolving", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/dualpresolving", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/setppc/dualpresolving", FALSE) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/logicor/dualpresolving", FALSE) );
#endif
   if ( isRacingStage() && paraComm->getRank() != 1 )
   {
#if SCIP_APIVERSION > 34
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/allowstrongdualreds", FALSE) );
#else
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/allowdualreds", FALSE) );
#endif
   }
}

void
ScipParaSolver::recoverOriginalSettings(
      )
{
   /* presolvers */
#if 0
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/stuffing/maxrounds", stuffingMaxrounds) );
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/domcol/maxrounds", domcolMaxrounds) );
#if  ( SCIP_VERSION >= 322 || (SCIP_VERSION == 321 && SCIP_SUBVERSION >= 2) )
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualcomp/maxrounds", dualcompMaxrounds) );  /*TODO: ok? */
#endif
   SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualinfer/maxrounds", dualinferMaxrounds) ); /*TODO: probably fine */
   // SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/dualagg/maxrounds", dualaggMaxrounds ) ); // TODO: seems to have no copy callback */
   /* constraint handlers */
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/setppc/dualpresolving", setppcDualpresolving) );
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/logicor/dualpresolving", logicorDualpresolving) );
#endif
   if ( (paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 )     // may not be in racing statge
        && paraComm->getRank() != 1 )
   {
#if SCIP_APIVERSION > 34
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/allowstrongdualreds", miscAllowdualreds) );
#else
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/allowdualreds", miscAllowdualreds) );
#endif
   }
}

void 
ScipParaSolver::issueInterruptSolve()
{
   SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
   commPointHdlr->issueInterrupt();
}

bool 
ScipParaSolver::isInterrupting()
{
   return commPointHdlr->isInterrupting();
}
