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

/**@file   heur_gcgfracdiving.c
 * @brief  LP diving heuristic that chooses fixings w.r.t. the fractionalities
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgfracdiving.h"
#include "heur_origdiving.h"
#include "gcg.h"


#define HEUR_NAME             "gcgfracdiving"
#define HEUR_DESC             "LP diving heuristic that chooses fixings w.r.t. the fractionalities"
#define HEUR_DISPCHAR         'f'
#define HEUR_PRIORITY         -1003000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          3
#define HEUR_MAXDEPTH         -1


/*
 * Default diving rule specific parameter settings
 */

#define DEFAULT_USEMASTERFRACS    FALSE      /**< calculate the fractionalities w.r.t. the master LP? */


/* locally defined diving heuristic data */
struct GCG_DivingData
{
   SCIP_Bool             usemasterfracs;     /**< calculate the fractionalities w.r.t. the master LP? */
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

/** get the 'down' fractionality of an original variable w.r.t. the master problem;
 *  this is the sum of the fractionalities of the master variables
 *  which would have to be fixed to zero if the original variable were rounded down
 */
static
SCIP_RETCODE getMasterDownFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< original variable to get fractionality for */
   SCIP_Real*            frac                /**< pointer to store fractionality */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nmastervars;
   int norigmastervars;
   SCIP_Real roundval;
   SCIP_Real masterlpval;

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
   *frac = 0.0;

   /* calculate sum of fractionalities over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasNegative(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i]) )
         {
            masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
            *frac += SCIPfeasFrac(masterprob, masterlpval);
         }
      }
      for( i = 0; i < norigmastervars; ++i )
      {
         masterlpval = SCIPgetSolVal(masterprob, NULL, origmastervars[i]);
         if( SCIPisFeasLE(masterprob, origmastervals[i], roundval) )
            *frac -= SCIPfeasFrac(masterprob, masterlpval);
      }
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
      {
         masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
         if( SCIPisFeasGT(masterprob, origmastervals[i], roundval) )
            *frac += SCIPfeasFrac(masterprob, masterlpval);
      }
   }

   return SCIP_OKAY;
}

/** get the 'up' fractionality of an original variable w.r.t. the master problem;
 *  this is the sum of the fractionalities of the master variables
 *  which would have to be fixed to zero if the original variable were rounded up
 */
static
SCIP_RETCODE getMasterUpFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< original variable to get fractionality for */
   SCIP_Real*            frac                /**< pointer to store fractionality */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nmastervars;
   int norigmastervars;
   SCIP_Real roundval;
   SCIP_Real masterlpval;

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
   *frac = 0.0;

   /* calculate sum of fractionalities over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasPositive(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i]) )
         {
            masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
            *frac += SCIPfeasFrac(masterprob, masterlpval);
         }
      }
      for( i = 0; i < norigmastervars; ++i )
      {
         masterlpval = SCIPgetSolVal(masterprob, NULL, origmastervars[i]);
         if( SCIPisFeasGE(masterprob, origmastervals[i], roundval) )
            *frac -= SCIPfeasFrac(masterprob, masterlpval);
      }
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
      {
         masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
         if( SCIPisFeasLT(masterprob, origmastervals[i], roundval) )
            *frac += SCIPfeasFrac(masterprob, masterlpval);
      }
   }

   return SCIP_OKAY;
}


/*
 * Callback methods
 */

/** destructor of diving heuristic to free user data (called when GCG is exiting) */
static
GCG_DECL_DIVINGFREE(heurFreeGcgfracdiving) /*lint --e{715}*/
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
 * finds best candidate variable w.r.t. fractionality:
 * - prefer variables that may not be rounded without destroying LP feasibility:
 *   - of these variables, round least fractional variable in corresponding direction
 * - if all remaining fractional variables may be rounded without destroying LP feasibility:
 *   - round variable with least increasing objective value
 * - binary variables are preferred
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarGcgfracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   int nlpcands;
   SCIP_Real bestobjgain;
   SCIP_Real bestfrac;                       /* fractionality of best candidate */
   SCIP_Bool bestcandmayrounddown;
   SCIP_Bool bestcandmayroundup;
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
   bestobjgain = SCIPinfinity(scip);
   bestfrac = SCIP_INVALID;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;
      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;
      SCIP_Bool roundup;
      SCIP_Real frac;
      SCIP_Real origfrac;
      SCIP_Real downfrac;
      SCIP_Real upfrac;
      SCIP_Real obj;

      int i;

      var = lpcands[c];

      /* if the variable is on the tabu list, do not choose it */
       for( i = 0; i < tabulistsize; ++i )
          if( tabulist[i] == var )
             break;
       if( i < tabulistsize )
          continue;

      mayrounddown = SCIPvarMayRoundDown(var);
      mayroundup = SCIPvarMayRoundUp(var);
      SCIP_CALL( getMasterDownFrac(scip, var, &downfrac) );
      SCIP_CALL( getMasterUpFrac(scip, var, &upfrac) );
      origfrac = lpcandssol[c] - SCIPfloor(scip, lpcandssol[c]);
      obj = SCIPvarGetObj(var);

      if( mayrounddown || mayroundup )
      {
         /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
         if( bestcandmayrounddown || bestcandmayroundup )
         {
            SCIP_Real objgain;

            /* choose rounding direction:
             * - if variable may be rounded in both directions, round corresponding to the fractionality
             * - otherwise, round in the infeasible direction, because feasible direction is tried by rounding
             *   the current fractional solution
             */
            if( mayrounddown && mayroundup )
               roundup = divingdata->usemasterfracs ? (upfrac < downfrac) : (origfrac > 0.5);
            else
               roundup = mayrounddown;

            if( roundup )
            {
               origfrac = 1.0 - origfrac;
               objgain = origfrac*obj;
            }
            else
               objgain = -origfrac*obj;

            if( divingdata->usemasterfracs )
               frac = MIN(downfrac, upfrac);
            else
               frac = origfrac;

            /* penalize too small fractions */
            if( frac < 0.01 )
               objgain *= 1000.0;

            /* prefer decisions on binary variables */
            if( !SCIPvarIsBinary(var) )
               objgain *= 1000.0;

            /* check, if candidate is new best candidate */
            if( SCIPisLT(scip, objgain, bestobjgain) || (SCIPisEQ(scip, objgain, bestobjgain) && frac < bestfrac) )
            {
               *bestcand = var;
               bestobjgain = objgain;
               bestfrac = frac;
               bestcandmayrounddown = mayrounddown;
               bestcandmayroundup = mayroundup;
               *bestcandroundup = roundup;
            }
         }
      }
      else
      {
         /* the candidate may not be rounded */
         if( divingdata->usemasterfracs )
         {
            if( downfrac < upfrac )
            {
               roundup = FALSE;
               frac = downfrac;
            }
            else
            {
               roundup = TRUE;
               frac = upfrac;
            }
         }
         else
         {
            if( origfrac < 0.5 )
            {
               roundup = FALSE;
               frac = origfrac;
            }
            else
            {
               roundup = TRUE;
               frac = 1.0 - origfrac;
            }
         }

         /* penalize too small fractions */
         if( frac < 0.01 )
            frac += 10.0;

         /* prefer decisions on binary variables */
         if( !SCIPvarIsBinary(var) )
            frac *= 1000.0;

         /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
         if( bestcandmayrounddown || bestcandmayroundup || frac < bestfrac )
         {
            *bestcand = var;
            bestfrac = frac;
            bestcandmayrounddown = FALSE;
            bestcandmayroundup = FALSE;
            *bestcandroundup = roundup;
         }
         assert(bestfrac < SCIP_INVALID);
      }
   }

   *bestcandmayround = bestcandmayroundup || bestcandmayrounddown;

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the gcgfracdiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurGcgfracdiving(
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
         HEUR_MAXDEPTH, heurFreeGcgfracdiving, NULL, NULL, NULL, NULL, NULL, NULL,
         heurSelectVarGcgfracdiving, divingdata) );

   assert(heur != NULL);

   /* add gcgfracdiving specific parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/usemasterfracs",
         "calculate the fractionalities w.r.t. the master LP?",
         &divingdata->usemasterfracs, TRUE, DEFAULT_USEMASTERFRACS, NULL, NULL) );

   return SCIP_OKAY;
}

