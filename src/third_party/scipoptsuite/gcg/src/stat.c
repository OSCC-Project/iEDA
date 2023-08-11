/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
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
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * @file   stat.c
 * @brief  Some printing methods for statistics
 * @author Alexander Gross
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scip.h"
#include "stat.h"
#include "scip_misc.h"
#include "pub_decomp.h"
#include "cons_decomp.h"
#include "struct_detector.h"
#include "pub_gcgvar.h"
#include "pricer_gcg.h"
#include "gcg.h"
#include "relax_gcg.h"


/** prints information about the best decomposition*/
SCIP_RETCODE GCGwriteDecompositionData(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   DEC_DECOMP* decomposition;

   DEC_DETECTOR* detector;
   DEC_DECTYPE type;
   const char* typeName;

   int i;
   int nblocks;
   int nlinkingconss;
   int nlinkingvars;
   int* nvarsinblocks;
   int* nconssinblocks;

   assert(scip != NULL);

   decomposition = DECgetBestDecomp(scip, TRUE);
   type = DECdecompGetType(decomposition);
   typeName = DECgetStrType(type);

   detector = DECdecompGetDetector(decomposition);

   nblocks = DECdecompGetNBlocks(decomposition);

   nvarsinblocks = DECdecompGetNSubscipvars(decomposition);
   nconssinblocks = DECdecompGetNSubscipconss(decomposition);

   nlinkingvars = DECdecompGetNLinkingvars(decomposition);
   nlinkingconss = DECdecompGetNLinkingconss(decomposition);

   /* print information about decomposition type and number of blocks, vars, linking vars and cons */
   SCIPinfoMessage(scip, NULL, "Decomposition:\n");
   SCIPinfoMessage(scip, NULL, "Decomposition Type: %s \n", typeName);

   SCIPinfoMessage(scip, NULL, "Decomposition Detector: %s\n", detector == NULL ? "reader": detector->name);
   SCIPinfoMessage(scip, NULL, "Number of Blocks: %d \n", nblocks);
   SCIPinfoMessage(scip, NULL, "Number of LinkingVars: %d\n", nlinkingvars);
   SCIPinfoMessage(scip, NULL, "Number of LinkingCons: %d\n", nlinkingconss);

   /* print number of variables and constraints per block */
   SCIPinfoMessage(scip, NULL, "Block Information\n");
   SCIPinfoMessage(scip, NULL, "no.:\t\t#Vars\t\t#Constraints\n");
   for( i = 0; i < nblocks; i++ )
   {
      SCIPinfoMessage(scip, NULL, "%d:\t\t%d\t\t%d\n", i, nvarsinblocks[i], nconssinblocks[i]);
   }

   DECdecompFree(scip, &decomposition);

   return SCIP_OKAY;
}

/** prints additional solving statistics */
SCIP_RETCODE GCGwriteSolvingDetails(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CLOCK* rootnodetime;

   assert(scip != NULL);

   rootnodetime = GCGgetRootNodeTime(scip);
   SCIPinfoMessage(scip, NULL, "Solving Details    :\n");
   SCIPinfoMessage(scip, NULL, "  time in root node: %10.2f\n", SCIPgetClockTime(scip, rootnodetime));

   return SCIP_OKAY;
}

/** prints information about the creation of the Vars*/
SCIP_RETCODE GCGwriteVarCreationDetails(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_VAR** vars;
   SCIP_SOL* sol;
   SCIP_Real solvingtime;
   SCIP_Longint nnodes;

   int nvars, i, n;
   SCIP_Longint* createnodestat;
   int nodes[2];         /* Wurzel Knoten und nicht wurzelknoten  */
   SCIP_Longint createtimestat[10];
   int createiterstat[10];
   int m;

   assert(scip != NULL);

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);
   nnodes = SCIPgetNNodes(scip);
   sol = SCIPgetBestSol(scip);

   solvingtime = SCIPgetSolvingTime(scip);
   assert(nnodes < INT_MAX);
   SCIP_CALL( SCIPallocBufferArray(scip, &createnodestat, (int)nnodes) ); /* lld doesn't work here */

   SCIPinfoMessage(scip, NULL, "AddedVarDetails:\n");

   for( i = 0; i < 10; i++ )
   {
      createtimestat[i] = 0;
      createiterstat[i] = 0;
   }

   nodes[0] = 0;
   nodes[1] = 0;

   SCIPinfoMessage(scip, NULL, "VAR: name\tnode\ttime\titer\trootredcostcall\tredcost\tgap\tsolval\trootlpsolval\n");
   for( i = 0; i < nvars; i++ )
   {
      SCIP_Real redcost;
      SCIP_Real gap;
      SCIP_Longint  node;
      SCIP_Real time;
      SCIP_Longint iteration;
      SCIP_Longint rootredcostcall;
      SCIP_Real rootlpsolval;

      node = GCGgetCreationNode(vars[i]);
      time = GCGgetCreationTime(vars[i]);
      iteration = GCGgetIteration(vars[i]);
      redcost = GCGgetRedcost(vars[i]);
      gap = GCGgetVarGap(vars[i]);
      rootredcostcall = GCGgetRootRedcostCall(vars[i]);

      rootlpsolval = NAN;

#ifdef SCIP_STATISTIC
      rootlpsolval = SCIPgetSolVal(scip, GCGmasterGetRootLPSol(scip), vars[i]);
#endif
      SCIPinfoMessage(scip, NULL, "VAR: <%s>\t%lld\t%f\t%lld\t%lld\t%f\t%f\t%f\t%f\n", SCIPvarGetName(vars[i]), node, time,
         iteration, rootredcostcall, redcost, gap, SCIPgetSolVal(scip, sol, vars[i]), rootlpsolval);

      if( SCIPisEQ(scip, SCIPgetSolVal(scip, sol, vars[i]), 0.0) )
      {
         continue;
      }
      else
      {
         SCIPdebugMessage("var <%s> has sol value %f (%lld, %f)\n", SCIPvarGetName(vars[i]),
            SCIPgetSolVal(scip, sol, vars[i]), node, time);
      }

      n = (int)(100 * time / solvingtime) % 10;
      m = (int)(100 * iteration / SCIPgetNLPIterations(scip)) % 10;
      createiterstat[n]++;
      createtimestat[m]++;

      if( node == 1 )
      {
         nodes[0]++;
      }
      else
      {
         nodes[1]++;
      }
   }

   SCIPinfoMessage(scip, NULL, "Root node:\tAdded Vars %d\n", nodes[0]);
   SCIPinfoMessage(scip, NULL, "Leftover nodes:\tAdded Vars %d\n", nodes[1]);

   for( i = 0; i < 10; i++ )
   {
      SCIPinfoMessage(scip, NULL, "Time %d-%d%%: Vars: %lld \n", 10 * i, 10 * (i + 1), createtimestat[i]);
   }

   for( i = 0; i < 10; i++ )
   {
      SCIPinfoMessage(scip, NULL, "Iter %d-%d%%: Vars: %d \n", 10 * i, 10 * (i + 1), createiterstat[i]);
   }

   SCIPfreeBufferArray(scip, &createnodestat);

   return SCIP_OKAY;
}
