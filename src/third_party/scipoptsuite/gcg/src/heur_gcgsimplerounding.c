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

/**@file   heur_gcgsimplerounding.c
 * @brief  simple and fast LP rounding heuristic
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgsimplerounding.h"
#include "gcg.h"


#define HEUR_NAME             "gcgsimplerounding"
#define HEUR_DESC             "simple and fast LP rounding heuristic on original variables"
#define HEUR_DISPCHAR         'r'
#define HEUR_PRIORITY         0
#define HEUR_FREQ             1
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE
#define HEUR_USESSUBSCIP      FALSE


/* locally defined heuristic data */
struct SCIP_HeurData
{
   SCIP_SOL*             sol;                /**< working solution */
   SCIP_Longint          lastlp;             /**< last LP number where the heuristic was applied */
   int                   nroundablevars;     /**< number of variables that can be rounded (-1 if not yet calculated) */
};




/*
 * Callback methods
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
#define heurCopyGcgsimplerounding NULL

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
#define heurFreeGcgsimplerounding NULL


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitGcgsimplerounding) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(SCIPheurGetData(heur) == NULL);

   /* create heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );
   heurdata->lastlp = -1;
   heurdata->nroundablevars = -1;
   SCIPheurSetData(heur, heurdata);

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitGcgsimplerounding) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* free heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolGcgsimplerounding)
{
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   heurdata->lastlp = -1;

   return SCIP_OKAY;
}


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolGcgsimplerounding NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecGcgsimplerounding) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP* masterprob;
   SCIP_HEURDATA* heurdata;
   SCIP_SOL* sol;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   SCIP_Longint nlps;
   int nlpcands;
   int c;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(result != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   *result = SCIP_DIDNOTRUN;

   /* do not execute the heuristic on invalid relaxation solutions
    * (which is the case if the node has been cut off)
    */
   if( !SCIPisRelaxSolValid(scip) )
   {
      SCIPdebugMessage("skipping GCG simple rounding: invalid relaxation solution\n");
      return SCIP_OKAY;
   }

   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetStage(masterprob) > SCIP_STAGE_SOLVING || SCIPgetLPSolstat(masterprob) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* on our first call, calculate the number of roundable variables */
   if( heurdata->nroundablevars == -1 )
   {
      SCIP_VAR** vars;
      int nvars;
      int nroundablevars;
      int i;

      vars = SCIPgetVars(scip);
      nvars = SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip);
      nroundablevars = 0;
      for( i = 0; i < nvars; ++i )
      {
         if( SCIPvarMayRoundDown(vars[i]) || SCIPvarMayRoundUp(vars[i]) )
            nroundablevars++;
      }
      heurdata->nroundablevars = nroundablevars;
   }

   /* don't call heuristic if there are no roundable variables */
   if( heurdata->nroundablevars == 0 )
      return SCIP_OKAY;

   /* don't call heuristic, if we have already processed the current LP solution */
   nlps = SCIPgetNLPs(masterprob);
   if( nlps == heurdata->lastlp )
      return SCIP_OKAY;
   heurdata->lastlp = nlps;

   /* get fractional variables, that should be integral */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &lpcands, &lpcandssol, NULL, &nlpcands, NULL, NULL, NULL, NULL) );

   /* only call heuristic, if LP solution is fractional */
   if( nlpcands == 0 )
      return SCIP_OKAY;

   /* don't call heuristic, if there are more fractional variables than roundable ones */
   if( nlpcands > heurdata->nroundablevars )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   SCIPdebugMessage("executing GCG simple rounding heuristic: %d fractionals\n", nlpcands);

   /* get the working solution from heuristic's local data */
   sol = heurdata->sol;
   assert(sol != NULL);

   /* copy the current LP solution to the working solution */
   SCIP_CALL( SCIPlinkRelaxSol(scip, sol) );

   /* round all roundable fractional columns in the corresponding direction as long as no unroundable column was found */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;
      SCIP_Real oldsolval;
      SCIP_Real newsolval;
      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;

      oldsolval = lpcandssol[c];
      assert(!SCIPisFeasIntegral(scip, oldsolval));
      var = lpcands[c];
      assert(SCIPvarGetStatus(var) == SCIP_VARSTATUS_COLUMN);
      mayrounddown = SCIPvarMayRoundDown(var);
      mayroundup = SCIPvarMayRoundUp(var);
      SCIPdebugMessage("GCG simple rounding heuristic: var <%s>, val=%g, rounddown=%u, roundup=%u\n",
         SCIPvarGetName(var), oldsolval, mayrounddown, mayroundup);

      /* choose rounding direction */
      if( mayrounddown && mayroundup )
      {
         /* we can round in both directions: round in objective function direction */
         if( SCIPvarGetObj(var) >= 0.0 )
            newsolval = SCIPfeasFloor(scip, oldsolval);
         else
            newsolval = SCIPfeasCeil(scip, oldsolval);
      }
      else if( mayrounddown )
         newsolval = SCIPfeasFloor(scip, oldsolval);
      else if( mayroundup )
         newsolval = SCIPfeasCeil(scip, oldsolval);
      else
         break;

      /* store new solution value */
      SCIP_CALL( SCIPsetSolVal(scip, sol, var, newsolval) );
   }

   /* check, if rounding was successful */
   if( c == nlpcands )
   {
      SCIP_Bool stored;

      /* check solution for feasibility, and add it to solution store if possible
       * neither integrality nor feasibility of LP rows has to be checked, because all fractional
       * variables were already moved in feasible direction to the next integer
       */
      SCIP_CALL( SCIPtrySol(scip, sol, FALSE, FALSE, FALSE, FALSE, FALSE, &stored) );

      if( stored )
      {
#ifdef SCIP_DEBUG
         SCIPdebugMessage("found feasible rounded solution:\n");
         SCIPprintSol(scip, sol, NULL, FALSE);
#endif
         *result = SCIP_FOUNDSOL;
      }
   }

   return SCIP_OKAY;
}




/*
 * heuristic specific interface methods
 */

/** creates the GCG simple rounding heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurGcgsimplerounding(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   /* include heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyGcgsimplerounding, heurFreeGcgsimplerounding, heurInitGcgsimplerounding, heurExitGcgsimplerounding,
         heurInitsolGcgsimplerounding, heurExitsolGcgsimplerounding, heurExecGcgsimplerounding,
         NULL) );

   return SCIP_OKAY;
}

