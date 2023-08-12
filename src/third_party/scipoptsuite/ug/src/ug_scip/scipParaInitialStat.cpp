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

/**@file    scipParaInitialStat.cpp
 * @brief   ParaInitialStat extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <iostream>
#include <cassert>
#include "scipParaComm.h"
#include "scipParaInitialStat.h"

using namespace UG;
using namespace ParaSCIP;

/** create clone of this object */
UG::ParaInitialStat *
ScipParaInitialStat::clone(
    UG::ParaComm *comm
		)
{
   int newMaxDepth = maxDepth;
   int newMaxTotalDepth = maxTotalDepth;
   int newNVarBranchStatsDown = nVarBranchStatsDown;
   int newNVarBranchStatsUp = nVarBranchStatsUp;
   int *newIdxLBranchStatsVarsDown = new int[nVarBranchStatsDown];
   int *newNVarBranchingDown = new int[nVarBranchStatsDown];
   int *newIdxLBranchStatsVarsUp = new int[nVarBranchStatsUp];
   int *newNVarBranchingUp = new int[nVarBranchStatsUp];
   SCIP_Real *newDownpscost = new SCIP_Real[nVarBranchStatsDown];
   SCIP_Real *newDownvsids = new SCIP_Real[nVarBranchStatsDown];
   SCIP_Real *newDownconflen = new SCIP_Real[nVarBranchStatsDown];
   SCIP_Real *newDowninfer = new SCIP_Real[nVarBranchStatsDown];
   SCIP_Real *newDowncutoff = new SCIP_Real[nVarBranchStatsDown];
   SCIP_Real *newUppscost = new SCIP_Real[nVarBranchStatsUp];
   SCIP_Real *newUpvsids = new SCIP_Real[nVarBranchStatsUp];
   SCIP_Real *newUpconflen = new SCIP_Real[nVarBranchStatsUp];
   SCIP_Real *newUpinfer = new SCIP_Real[nVarBranchStatsUp];
   SCIP_Real *newUpcutoff  = new SCIP_Real[nVarBranchStatsUp];
   for( int i = 0; i < nVarBranchStatsDown; i++ )
   {
      newIdxLBranchStatsVarsDown[i] = idxLBranchStatsVarsDown[i];
      newNVarBranchingDown[i] = nVarBranchingDown[i];
      newDownpscost[i] = downpscost[i];
      newDownvsids[i] = downvsids[i];
      newDownconflen[i] = downconflen[i];
      newDowninfer[i] = downinfer[i];
      newDowncutoff[i] = downcutoff[i];
   }
   for( int i = 0; i < nVarBranchStatsUp; i++ )
   {
      newIdxLBranchStatsVarsUp[i] = idxLBranchStatsVarsUp[i];
      newNVarBranchingUp[i] = nVarBranchingUp[i];
      newUppscost[i] = uppscost[i];
      newUpvsids[i] = upvsids[i];
      newUpconflen[i] = upconflen[i];
      newUpinfer[i] = upinfer[i];
      newUpcutoff[i] = upcutoff[i];
   }
   DEF_SCIP_PARA_COMM( scipParaComm, comm);
   return (
         scipParaComm->createScipParaInitialStat(newMaxDepth, newMaxTotalDepth, newNVarBranchStatsDown, newNVarBranchStatsUp,
                                 newIdxLBranchStatsVarsDown, newNVarBranchingDown, newIdxLBranchStatsVarsUp, newNVarBranchingUp,
                                 newDownpscost, newDownvsids, newDownconflen, newDowninfer, newDowncutoff,
                                 newUppscost, newUpvsids, newUpconflen, newUpinfer, newUpcutoff)
   );
}

ScipParaInitialStat::ScipParaInitialStat(
      SCIP *scip
      ) :
      maxDepth(0),
      maxTotalDepth(0),
      nVarBranchStatsDown(0),
      nVarBranchStatsUp(0),
      idxLBranchStatsVarsDown(0),
      nVarBranchingDown(0),
      idxLBranchStatsVarsUp(0),
      nVarBranchingUp(0),
      downpscost(0),
      downvsids(0),
      downconflen(0),
      downinfer(0),
      downcutoff(0),
      uppscost(0),
      upvsids(0),
      upconflen(0),
      upinfer(0),
      upcutoff(0)
{
   maxDepth = SCIPgetMaxDepth(scip);
   maxTotalDepth = SCIPgetMaxTotalDepth(scip);
   nVarBranchStatsDown = 0;
   nVarBranchStatsUp = 0;

   int nvars;                                /* number of variables                           */
   int nbinvars;                             /* number of binary variables                    */
   int nintvars;                             /* number of integer variables                   */
   SCIP_VAR** vars;                          /* transformed problem's variables               */
   SCIP_CALL_ABORT( SCIPgetVarsData(scip, &vars, &nvars, &nbinvars, &nintvars, NULL, NULL) );
   int ngenvars = nbinvars+nintvars;

   /** count downward vars and upward vars */
   for( int i = 0; i < ngenvars; ++i )
   {
      assert( SCIPvarGetType(vars[i]) == SCIP_VARTYPE_BINARY || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_INTEGER );
      SCIP_VAR *transformVar = vars[i];
      SCIP_Real scalar = 1.0;
      SCIP_Real constant = 0.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) );
      assert(transformVar != NULL);
      if( scalar > 0.0 )
      {
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS) > 0 )
         {
            nVarBranchStatsDown++;
         }
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS) > 0 )
         {
            nVarBranchStatsUp++;
         }
      }
      else
      {
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS) > 0 )
         {
            nVarBranchStatsUp++;
         }
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS) > 0 )
         {
            nVarBranchStatsDown++;
         }
      }
   }

   /** allocate memory */
   idxLBranchStatsVarsDown = new int[nVarBranchStatsDown];
   nVarBranchingDown = new int[nVarBranchStatsDown];
   idxLBranchStatsVarsUp = new int[nVarBranchStatsUp];
   nVarBranchingUp = new int[nVarBranchStatsUp];
   downpscost = new SCIP_Real[nVarBranchStatsDown];
   downvsids = new SCIP_Real[nVarBranchStatsDown];
   downconflen = new SCIP_Real[nVarBranchStatsDown];
   downinfer = new SCIP_Real[nVarBranchStatsDown];
   downcutoff = new SCIP_Real[nVarBranchStatsDown];
   uppscost = new SCIP_Real[nVarBranchStatsUp];
   upvsids = new SCIP_Real[nVarBranchStatsUp];
   upconflen = new SCIP_Real[nVarBranchStatsUp];
   upinfer = new SCIP_Real[nVarBranchStatsUp];
   upcutoff = new SCIP_Real[nVarBranchStatsUp];

   int nDown = 0;
   int nUp = 0;
   for( int i = 0; i < ngenvars; ++i )
   {
      assert( SCIPvarGetType(vars[i]) == SCIP_VARTYPE_BINARY || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_INTEGER );
      SCIP_VAR *transformVar = vars[i];
      SCIP_Real scalar = 1.0;
      SCIP_Real constant = 0.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) );
      assert(transformVar != NULL);
      if( scalar > 0.0 )
      {
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS) > 0 )
         {
            idxLBranchStatsVarsDown[nDown] = SCIPvarGetIndex(transformVar);
            nVarBranchingDown[nDown] = SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS);
            downpscost[nDown] = SCIPgetVarPseudocost(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downvsids[nDown] = SCIPgetVarVSIDS(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downconflen[nDown] = SCIPgetVarAvgConflictlength(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downinfer[nDown] = SCIPgetVarAvgInferences(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downcutoff[nDown] = SCIPgetVarAvgCutoffs(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            nDown++;
         }
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS) > 0 )
         {
            idxLBranchStatsVarsUp[nUp] = SCIPvarGetIndex(transformVar);
            nVarBranchingUp[nUp] = SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS);
            uppscost[nUp] = SCIPgetVarPseudocost(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upvsids[nUp] = SCIPgetVarVSIDS(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upconflen[nUp] = SCIPgetVarAvgConflictlength(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upinfer[nUp] = SCIPgetVarAvgInferences(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upcutoff[nUp] = SCIPgetVarAvgCutoffs(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            nUp++;
         }
      }
      else
      {
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS) > 0 )
         {
            idxLBranchStatsVarsUp[nUp] = SCIPvarGetIndex(transformVar);
            nVarBranchingUp[nUp] = SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS);
            uppscost[nUp] = SCIPgetVarPseudocost(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upvsids[nUp] = SCIPgetVarVSIDS(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upconflen[nUp] = SCIPgetVarAvgConflictlength(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upinfer[nUp] = SCIPgetVarAvgInferences(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            upcutoff[nUp] = SCIPgetVarAvgCutoffs(scip, vars[i], SCIP_BRANCHDIR_UPWARDS);
            nUp++;
         }
         if( SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_UPWARDS) > 0 )
         {
            idxLBranchStatsVarsDown[nDown] = SCIPvarGetIndex(transformVar);
            nVarBranchingDown[nDown] = SCIPvarGetNBranchings(transformVar, SCIP_BRANCHDIR_DOWNWARDS);
            downpscost[nDown] = SCIPgetVarPseudocost(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downvsids[nDown] = SCIPgetVarVSIDS(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downconflen[nDown] = SCIPgetVarAvgConflictlength(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downinfer[nDown] = SCIPgetVarAvgInferences(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            downcutoff[nDown] = SCIPgetVarAvgCutoffs(scip, vars[i], SCIP_BRANCHDIR_DOWNWARDS);
            nDown++;
         }
      }
   }
   assert( nVarBranchStatsDown == nDown && nVarBranchStatsUp == nUp );
}

void
ScipParaInitialStat::accumulateOn(
      SCIP *scip
      )
{
   SCIP_VAR **vars = SCIPgetVars(scip);
   for(int n = 0; n < nVarBranchStatsDown; n++ )
   {
      for( int i = 0; i < nVarBranchingDown[n]; i++ )
      {
         assert( SCIPvarGetProbindex(vars[idxLBranchStatsVarsDown[n]]) == idxLBranchStatsVarsDown[n]);
         SCIP_CALL_ABORT( SCIPinitVarBranchStats(scip,vars[idxLBranchStatsVarsDown[n]],downpscost[n],0.0,downvsids[n],0.0,downconflen[n],0.0,downinfer[n],0.0,downcutoff[n],0.0) );
      }
   }
   for(int n = 0; n < nVarBranchStatsUp; n++ )
   {
      for( int i = 0; i < nVarBranchingUp[n]; i++ )
      {
         assert( SCIPvarGetProbindex(vars[idxLBranchStatsVarsUp[n]]) == idxLBranchStatsVarsUp[n]);
         SCIP_CALL_ABORT( SCIPinitVarBranchStats(scip,vars[idxLBranchStatsVarsUp[n]],0.0,uppscost[n],0.0,upvsids[n],0.0,upconflen[n],0.0,upinfer[n],0.0,upcutoff[n]) );
      }
   }
}
