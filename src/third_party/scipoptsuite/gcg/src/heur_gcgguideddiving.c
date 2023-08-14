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

/**@file   heur_gcgguideddiving.c
 * @brief  LP diving heuristic that chooses fixings in direction of incumbent solutions
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgguideddiving.h"
#include "heur_origdiving.h"
#include "gcg.h"


#define HEUR_NAME             "gcgguideddiving"
#define HEUR_DESC             "LP diving heuristic that chooses fixings in direction of incumbent solutions"
#define HEUR_DISPCHAR         'g'
#define HEUR_PRIORITY         -1007000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          7
#define HEUR_MAXDEPTH         -1


/*
 * Default diving rule specific parameter settings
 */

#define DEFAULT_USEMASTERFRACS    FALSE      /**< calculate the fractionalities w.r.t. the master LP? */


/* locally defined diving heuristic data */
struct GCG_DivingData
{
   SCIP_Bool             usemasterfracs;     /**< calculate the fractionalities w.r.t. the master LP? */
   SCIP_SOL*             bestsol;            /**< best known feasible solution before the diving loop */
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
GCG_DECL_DIVINGFREE(heurFreeGcgguideddiving) /*lint --e{715}*/
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


/** execution initialization method of diving heuristic (called when execution of diving heuristic is about to begin) */
static
GCG_DECL_DIVINGINITEXEC(heurInitexecGcgguideddiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* store a copy of the best solution */
   SCIP_CALL( SCIPcreateSolCopy(scip, &divingdata->bestsol, SCIPgetBestSol(scip)) );

   return SCIP_OKAY;
}


/** execution deinitialization method of diving heuristic (called when execution data is freed) */
static
GCG_DECL_DIVINGEXITEXEC(heurExitexecGcgguideddiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* free copied best solution */
   SCIP_CALL( SCIPfreeSol(scip, &divingdata->bestsol) );

   return SCIP_OKAY;
}


/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. the incumbent solution:
 * - prefer variables that may not be rounded without destroying LP feasibility:
 *   - of these variables, round a variable to its value in direction of incumbent solution, and choose the
 *     variable that is closest to its rounded value
 * - if all remaining fractional variables may be rounded without destroying LP feasibility:
 *   - round variable in direction that destroys LP feasibility (other direction is checked by SCIProundSol())
 *   - round variable with least increasing objective value
 * - binary variables are prefered
 * - variables in a minimal cover or variables that are also fractional in an optimal LP solution might
 *   also be prefered if a corresponding parameter is set
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarGcgguideddiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   SCIP_Real* lpcandsfrac;
   int nlpcands;
   SCIP_Real bestobjgain;
   SCIP_Real bestscore;                       /* fractionality of best candidate */
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
   assert(divingdata->bestsol != NULL);

   /* get fractional variables that should be integral */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, NULL, NULL, NULL, NULL) );
   assert(lpcands != NULL);
   assert(lpcandsfrac != NULL);
   assert(lpcandssol != NULL);

   bestcandmayrounddown = TRUE;
   bestcandmayroundup = TRUE;
   bestobjgain = SCIPinfinity(scip);
   bestscore = SCIP_INVALID;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;
      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;
      SCIP_Bool roundup;
      SCIP_Real frac;
      SCIP_Real obj;
      SCIP_Real score;
      SCIP_Real solval;
      SCIP_Real bestsolval;

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
      solval = lpcandssol[c];
      frac = lpcandsfrac[c];
      obj = SCIPvarGetObj(var);
      bestsolval = SCIPgetSolVal(scip, divingdata->bestsol, var);

      /* select default rounding direction */
      roundup = (solval < bestsolval);

      if( mayrounddown || mayroundup )
      {
         /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
         if( bestcandmayrounddown || bestcandmayroundup )
         {
            SCIP_Real objgain;

            /* choose rounding direction:
             * - if variable may be rounded in both directions, round corresponding to its value in incumbent solution
             * - otherwise, round in the infeasible direction, because feasible direction is tried by rounding
             *   the current fractional solution with SCIProundSol()
             */
            if( !mayrounddown || !mayroundup )
               roundup = mayrounddown;

            if( roundup )
            {
               frac = 1.0 - frac;
               objgain = frac*obj;
            }
            else
               objgain = -frac*obj;

            /* penalize too small fractions */
            if( frac < 0.01 )
               objgain *= 1000.0;

            /* prefer decisions on binary variables */
            if( !SCIPvarIsBinary(var) )
               objgain *= 1000.0;

            if( divingdata->usemasterfracs )
            {
               SCIP_Real downfrac;
               SCIP_Real upfrac;

               SCIP_CALL( getMasterDownFrac(scip, var, &downfrac) );
               SCIP_CALL( getMasterUpFrac(scip, var, &upfrac) );
               score = roundup ? upfrac : downfrac;
            }
            else
               score = frac;

            /* check, if candidate is new best candidate */
            if( SCIPisLT(scip, objgain, bestobjgain) || (SCIPisEQ(scip, objgain, bestobjgain) && score < bestscore) )
            {
               *bestcand = var;
               bestobjgain = objgain;
               bestscore = score;
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
            SCIP_Real downfrac;
            SCIP_Real upfrac;

            SCIP_CALL( getMasterDownFrac(scip, var, &downfrac) );
            SCIP_CALL( getMasterUpFrac(scip, var, &upfrac) );
            score = roundup ? upfrac : downfrac;
         }
         else
            score = roundup? 1.0 - frac : frac;

         /* penalize too small fractions */
         if( score < 0.01 )
            score += 10.0;

         /* prefer decisions on binary variables */
         if( !SCIPvarIsBinary(var) )
            score *= 1000.0;

         /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
         if( bestcandmayrounddown || bestcandmayroundup || score < bestscore )
         {
            *bestcand = var;
            bestscore = score;
            bestcandmayrounddown = FALSE;
            bestcandmayroundup = FALSE;
            *bestcandroundup = roundup;
         }
      }
   }

   *bestcandmayround = bestcandmayroundup || bestcandmayrounddown;

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the gcgguideddiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurGcgguideddiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEUR* heur;
   GCG_DIVINGDATA* divingdata;

   /* create gcgguideddiving primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &divingdata) );

   /* include diving heuristic */
   SCIP_CALL( GCGincludeDivingHeurOrig(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, heurFreeGcgguideddiving, NULL, NULL, NULL, NULL, heurInitexecGcgguideddiving,
         heurExitexecGcgguideddiving, heurSelectVarGcgguideddiving, divingdata) );

   assert(heur != NULL);

   /* add gcgguideddiving specific parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/usemasterfracs",
         "calculate the fractionalities w.r.t. the master LP?",
         &divingdata->usemasterfracs, TRUE, DEFAULT_USEMASTERFRACS, NULL, NULL) );

   return SCIP_OKAY;
}

