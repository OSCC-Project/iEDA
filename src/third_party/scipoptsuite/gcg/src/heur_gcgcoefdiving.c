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

/**@file   heur_gcgcoefdiving.c
 * @brief  LP diving heuristic that chooses fixings w.r.t. the matrix coefficients
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgcoefdiving.h"
#include "heur_origdiving.h"
#include "gcg.h"


#define HEUR_NAME             "gcgcoefdiving"
#define HEUR_DESC             "LP diving heuristic that chooses fixings w.r.t. the matrix coefficients"
#define HEUR_DISPCHAR         'c'
#define HEUR_PRIORITY         -1001000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          1
#define HEUR_MAXDEPTH         -1


/*
 * Default diving rule specific parameter settings
 */

#define DEFAULT_USEMASTERLOCKS    FALSE      /**< calculate the number of locks w.r.t. the master LP? */


/* locally defined diving heuristic data */
struct GCG_DivingData
{
   SCIP_Bool             usemasterlocks;     /**< calculate the number of locks w.r.t. the master LP? */
};


/*
 * local methods
 */

/** check whether an original variable and a master variable belong to the same block */
static
SCIP_Bool areVarsInSameBlock(
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_VAR*             mastervar           /**< master variable */
   )
{
   int origblock;
   int masterblock;

   /* get the blocks the variables belong to */
   origblock = GCGvarGetBlock(origvar);
   masterblock = GCGvarGetBlock(mastervar);

   /* the original variable is a linking variable:
    * check whether the master variable is either its direct copy
    * or in one of its blocks
    */
   if( GCGoriginalVarIsLinking(origvar) )
   {
      assert(origblock == -2);
      if( masterblock == -1 )
      {
         SCIP_VAR** mastervars;

         mastervars = GCGoriginalVarGetMastervars(origvar);

         return mastervars[0] == mastervar;
      }
      else
      {
         assert(masterblock >= 0);
         return GCGisLinkingVarInBlock(origvar, masterblock);
      }
   }
   /* the original variable was directly copied to the master problem:
    * check whether the master variable is its copy
    */
   else if( origblock == -1 )
   {
      SCIP_VAR** mastervars;

      mastervars = GCGoriginalVarGetMastervars(origvar);
      assert(GCGoriginalVarGetNMastervars(origvar) == 1);

      return mastervars[0] == mastervar;
   }
   /* the original variable belongs to exactly one block */
   else
   {
      assert(origblock >= 0);
      return origblock == masterblock;
   }
}

/** get the number of down-locks for an original variable w.r.t. the master problem */
static
SCIP_RETCODE getNLocksDown(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< original variable to get locks for */
   int*                  nlocksdown          /**< pointer to store number of down-locks */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nmastervars;
   int norigmastervars;
   SCIP_Real roundval;

   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   /* get master variables in which the original variable appears */
   origmastervars = GCGoriginalVarGetMastervars(var);
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasFloor(scip, SCIPgetRelaxSolVal(scip, var));
   *nlocksdown = 0;

   /* calculate locks = the sum of down-locks of all master variables
    * which would have to be fixed to zero if the original variable were rounded down
    */
   if( SCIPisFeasNegative(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i] )
            && !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, mastervars[i])) )
            *nlocksdown += SCIPvarGetNLocksDown(mastervars[i]);
      }
      for( i = 0; i < norigmastervars; ++i )
         if( !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, origmastervars[i]) )
            && SCIPisFeasLE(masterprob, origmastervals[i], roundval) )
            *nlocksdown -= SCIPvarGetNLocksDown(origmastervars[i]);
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
         if( !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, origmastervars[i]) )
            && SCIPisFeasGT(masterprob, origmastervals[i], roundval) )
            *nlocksdown += SCIPvarGetNLocksDown(origmastervars[i]);
   }
   assert(*nlocksdown >= 0);

   return SCIP_OKAY;
}

/** get the number of up-locks for an original variable w.r.t. the master problem */
static
SCIP_RETCODE getNLocksUp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< original variable to get locks for */
   int*                  nlocksup            /**< pointer to store number of up-locks */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nmastervars;
   int norigmastervars;
   SCIP_Real roundval;

   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   /* get master variables in which the original variable appears */
   origmastervars = GCGoriginalVarGetMastervars(var);
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasCeil(scip, SCIPgetRelaxSolVal(scip, var));
   *nlocksup = 0;

   /* calculate locks = the sum of down-locks of all master variables
    * which would have to be fixed to zero if the original variable were rounded up
    */
   if( SCIPisFeasPositive(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i] )
            && !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, mastervars[i])) )
            *nlocksup += SCIPvarGetNLocksDown(mastervars[i]);
      }
      for( i = 0; i < norigmastervars; ++i )
         if( !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, origmastervars[i]) )
            && SCIPisFeasGE(masterprob, origmastervals[i], roundval) )
            *nlocksup -= SCIPvarGetNLocksDown(origmastervars[i]);
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
         if( !SCIPisFeasZero(masterprob, SCIPgetSolVal(masterprob, NULL, origmastervars[i]) )
            && SCIPisFeasLT(masterprob, origmastervals[i], roundval) )
            *nlocksup += SCIPvarGetNLocksDown(origmastervars[i]);
   }
   assert(*nlocksup >= 0);

   return SCIP_OKAY;
}


/*
 * Callback methods
 */

/** destructor of diving heuristic to free user data (called when GCG is exiting) */
static
GCG_DECL_DIVINGFREE(heurFreeGcgcoefdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* free diving rule specific data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);
   SCIPfreeMemory(scip, &divingdata);
   GCGheurSetDivingDataOrig(heur, NULL);

   return SCIP_OKAY;
}


/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. locking numbers:
 * - prefer variables that may not be rounded without destroying LP feasibility:
 *   - of these variables, round variable with least number of locks in corresponding direction
 * - if all remaining fractional variables may be rounded without destroying LP feasibility:
 *   - round variable with least number of locks in opposite of its feasible rounding direction
 * - binary variables are preferred
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarGcgcoefdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   int nlpcands;
   SCIP_Bool bestcandmayrounddown;
   SCIP_Bool bestcandmayroundup;
   int bestnviolrows;             /* number of violated rows for best candidate */
   SCIP_Real bestcandfrac;        /* fractionality of best candidate */
   int c;

   /* check preconditions */
   assert(scip != NULL);
   assert(heur != NULL);
   assert(bestcand != NULL);
   assert(bestcandmayround != NULL);
   assert(bestcandroundup != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* get fractional variables that should be integral */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &lpcands, &lpcandssol, NULL, &nlpcands, NULL, NULL, NULL, NULL) );
   assert(lpcands != NULL);
   assert(lpcandssol != NULL);

   bestcandmayrounddown = TRUE;
   bestcandmayroundup = TRUE;
   bestnviolrows = INT_MAX;
   bestcandfrac = SCIP_INVALID;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;

      int nlocksdown;
      int nlocksup;
      int nviolrows;

      int i;

      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;
      SCIP_Bool roundup;
      SCIP_Real frac;

      var = lpcands[c];
      frac = lpcandssol[c] - SCIPfloor(scip, lpcandssol[c]);

      /* if the variable is on the tabu list, do not choose it */
      for( i = 0; i < tabulistsize; ++i )
         if( tabulist[i] == var )
            break;
      if( i < tabulistsize )
         continue;

      if( divingdata->usemasterlocks )
      {
         SCIP_CALL( getNLocksDown(scip, var, &nlocksdown) );
         SCIP_CALL( getNLocksUp(scip, var, &nlocksup) );
         mayrounddown = nlocksdown == 0;
         mayroundup = nlocksup == 0;
      }
      else
      {
         nlocksdown = SCIPvarGetNLocksDown(var);
         nlocksup = SCIPvarGetNLocksUp(var);
         mayrounddown = SCIPvarMayRoundDown(var);
         mayroundup = SCIPvarMayRoundUp(var);
      }

      if( mayrounddown || mayroundup )
      {
         /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
         if( bestcandmayrounddown || bestcandmayroundup )
         {
            /* choose rounding direction:
             * - if variable may be rounded in both directions, round corresponding to the fractionality
             * - otherwise, round in the infeasible direction, because feasible direction is tried by rounding
             *   the current fractional solution
             */
            if( mayrounddown && mayroundup )
               roundup = (frac > 0.5);
            else
               roundup = mayrounddown;

            if( roundup )
            {
               frac = 1.0 - frac;
               nviolrows = nlocksup;
            }
            else
               nviolrows = nlocksdown;

            /* penalize too small fractions */
            if( frac < 0.01 )
               nviolrows *= 100;

            /* prefer decisions on binary variables */
            if( !SCIPvarIsBinary(var) )
               nviolrows *= 100;

            /* check, if candidate is new best candidate */
            assert(0.0 < frac && frac < 1.0);
            if( nviolrows + frac < bestnviolrows + bestcandfrac )
            {
               *bestcand = var;
               bestnviolrows = nviolrows;
               bestcandfrac = frac;
               bestcandmayrounddown = mayrounddown;
               bestcandmayroundup = mayroundup;
               *bestcandroundup = roundup;
            }
         }
      }
      else
      {
         /* the candidate may not be rounded */
         roundup = (nlocksdown > nlocksup || (nlocksdown == nlocksup && frac > 0.5));
         if( roundup )
         {
            nviolrows = nlocksup;
            frac = 1.0 - frac;
         }
         else
            nviolrows = nlocksdown;

         /* penalize too small fractions */
         if( frac < 0.01 )
            nviolrows *= 100;

         /* prefer decisions on binary variables */
         if( !SCIPvarIsBinary(var) )
            nviolrows *= 100;

         /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
         assert(0.0 < frac && frac < 1.0);
         if( bestcandmayrounddown || bestcandmayroundup || nviolrows + frac < bestnviolrows + bestcandfrac )
         {
            *bestcand = var;
            bestnviolrows = nviolrows;
            bestcandfrac = frac;
            bestcandmayrounddown = FALSE;
            bestcandmayroundup = FALSE;
            *bestcandroundup = roundup;
         }
         assert(bestcandfrac < SCIP_INVALID);
      }
   }

   *bestcandmayround = bestcandmayroundup || bestcandmayrounddown;

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the gcgcoefdiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurGcgcoefdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEUR* heur;
   GCG_DIVINGDATA* divingdata;

   /* create gcgcoefdiving primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &divingdata) );

   /* include diving heuristic */
   SCIP_CALL( GCGincludeDivingHeurOrig(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, heurFreeGcgcoefdiving, NULL, NULL, NULL, NULL, NULL, NULL,
         heurSelectVarGcgcoefdiving, divingdata) );

   assert(heur != NULL);

   /* add gcgcoefdiving specific parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/usemasterlocks",
         "calculate the number of locks w.r.t. the master LP?",
         &divingdata->usemasterlocks, TRUE, DEFAULT_USEMASTERLOCKS, NULL, NULL) );

   return SCIP_OKAY;
}

