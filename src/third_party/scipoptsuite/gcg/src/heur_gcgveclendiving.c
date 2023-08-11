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

/**@file   heur_gcgveclendiving.c
 * @brief  LP diving heuristic that rounds variables with long column vectors
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgveclendiving.h"
#include "heur_origdiving.h"
#include "gcg.h"


#define HEUR_NAME             "gcgveclendiving"
#define HEUR_DESC             "LP diving heuristic that rounds variables with long column vectors"
#define HEUR_DISPCHAR         'v'
#define HEUR_PRIORITY         -1003100
#define HEUR_FREQ             10
#define HEUR_FREQOFS          4
#define HEUR_MAXDEPTH         -1


/*
 * Default diving rule specific parameter settings
 */

#define DEFAULT_USEMASTERSCORES   FALSE      /**< calculate vector length scores w.r.t. the master LP? */


/* locally defined diving heuristic data */
struct GCG_DivingData
{
   SCIP_Bool             usemasterscores;    /**< calculate vector length scores w.r.t. the master LP? */
   SCIP_Real*            masterscores;       /**< vector length based scores for the master variables */
};


/*
 * local methods
 */

/** for a variable, calculate the vector length score w.r.t. the original problem */
static
SCIP_RETCODE calculateScoreOrig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to calculate score for */
   SCIP_Real             frac,               /**< the variable's fractionality in the current LP solution */
   SCIP_Real*            score,              /**< pointer to return the score */
   SCIP_Bool*            roundup             /**< pointer to return whether the variable is rounded up */
   )
{
   SCIP_Real obj;
   SCIP_Real objdelta;
   int colveclen;

   obj = SCIPvarGetObj(var);
   *roundup = (obj >= 0.0);
   objdelta = (*roundup ? (1.0-frac)*obj : -frac * obj);
   assert(objdelta >= 0.0);

   colveclen = (SCIPvarGetStatus(var) == SCIP_VARSTATUS_COLUMN ? SCIPcolGetNNonz(SCIPvarGetCol(var)) : 0);

   /* smaller score is better */
   *score = (objdelta + SCIPsumepsilon(scip))/((SCIP_Real)colveclen+1.0);

   /* prefer decisions on binary variables */
   if( SCIPvarGetType(var) != SCIP_VARTYPE_BINARY )
      *score *= 1000.0;

   return SCIP_OKAY;
}

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

/** get the 'down' score of an original variable w.r.t. the master problem;
 *  this is the sum of the vector length scores of the master variables
 *  which would have to be fixed to zero if the original variable were rounded down
 */
static
SCIP_RETCODE getMasterDownScore(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_DIVINGDATA*       divingdata,         /**< diving heuristic data */
   SCIP_VAR*             var,                /**< original variable to get fractionality for */
   SCIP_Real*            score               /**< pointer to store fractionality */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
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
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasFloor(scip, SCIPgetRelaxSolVal(scip, var));
   *score = 0.0;

   /* calculate sum of scores over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasNegative(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
         if( areVarsInSameBlock(var, mastervars[i]) )
            *score += divingdata->masterscores[i];
      for( i = 0; i < norigmastervars; ++i )
         if( SCIPisFeasLE(masterprob, origmastervals[i], roundval) )
            *score -= divingdata->masterscores[i];
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
         if( SCIPisFeasGT(masterprob, origmastervals[i], roundval) )
            *score += divingdata->masterscores[i];
   }

   return SCIP_OKAY;
}

/** get the 'up' score of an original variable w.r.t. the master problem;
 *  this is the sum of the scores of the master variables
 *  which would have to be fixed to zero if the original variable were rounded up
 */
static
SCIP_RETCODE getMasterUpScore(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_DIVINGDATA*       divingdata,         /**< diving heuristic data */
   SCIP_VAR*             var,                /**< original variable to get fractionality for */
   SCIP_Real*            score               /**< pointer to store fractionality */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
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
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasCeil(scip, SCIPgetRelaxSolVal(scip, var));
   *score = 0.0;

   /* calculate sum of scores over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasPositive(masterprob, roundval) )
   {
      for( i = 0; i < nmastervars; ++i )
         if( areVarsInSameBlock(var, mastervars[i]) )
            *score += divingdata->masterscores[i];
      for( i = 0; i < norigmastervars; ++i )
         if( SCIPisFeasGE(masterprob, origmastervals[i], roundval) )
            *score -= divingdata->masterscores[i];
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
         if( SCIPisFeasLT(masterprob, origmastervals[i], roundval) )
            *score += divingdata->masterscores[i];
   }

   return SCIP_OKAY;
}

/** for a variable, calculate the vector length score w.r.t. the master problem */
static
SCIP_RETCODE calculateScoreMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_DIVINGDATA*       divingdata,         /**< diving heuristic data */
   SCIP_VAR*             var,                /**< variable to calculate score for */
   SCIP_Real*            score,              /**< pointer to return the score */
   SCIP_Bool*            roundup             /**< pointer to return whether the variable is rounded up */
   )
{
   SCIP_Real downscore;
   SCIP_Real upscore;

   /* calculate scores for rounding down or up */
   SCIP_CALL( getMasterDownScore(scip, divingdata, var, &downscore) );
   SCIP_CALL( getMasterUpScore(scip, divingdata, var, &upscore) );

   *score = MIN(downscore, upscore);
   *roundup = upscore <= downscore;

   /* prefer decisions on binary variables */
   if( SCIPvarGetType(var) != SCIP_VARTYPE_BINARY )
      *score *= 1000.0;

   return SCIP_OKAY;
}


/*
 * Callback methods
 */


/** destructor of diving heuristic to free user data (called when GCG is exiting) */
static
GCG_DECL_DIVINGFREE(heurFreeGcgveclendiving) /*lint --e{715}*/
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
GCG_DECL_DIVINGINITEXEC(heurInitexecGcgveclendiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP* masterprob;
   SCIP_SOL* masterlpsol;
   SCIP_VAR** mastervars;
   int nmastervars;

   int i;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* do not collect vector length scores on master variables if not used */
   if( !divingdata->usemasterscores )
      return SCIP_OKAY;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variables */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &divingdata->masterscores, nmastervars) );
   SCIP_CALL( SCIPcreateSol(masterprob, &masterlpsol, NULL) );

   /* get master LP solution */
   SCIP_CALL( SCIPlinkLPSol(masterprob, masterlpsol) );

   /* for each master variable, calculate a score */
   for( i = 0; i < nmastervars; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_Real objdelta;
      int colveclen;

      mastervar = mastervars[i];
      objdelta = SCIPfeasFrac(masterprob, SCIPgetSolVal(masterprob, masterlpsol, mastervar)) * SCIPvarGetObj(mastervar);
      objdelta = ABS(objdelta);
      colveclen = (SCIPvarGetStatus(mastervar) == SCIP_VARSTATUS_COLUMN ? SCIPcolGetNNonz(SCIPvarGetCol(mastervar)) : 0);
      divingdata->masterscores[i] = (objdelta + SCIPsumepsilon(scip))/((SCIP_Real)colveclen+1.0);
   }

   /* free memory */
   SCIP_CALL( SCIPfreeSol(masterprob, &masterlpsol) );

   return SCIP_OKAY;
}


/** execution deinitialization method of diving heuristic (called when execution data is freed) */
static
GCG_DECL_DIVINGEXITEXEC(heurExitexecGcgveclendiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* memory needs to to be freed if vector length scores on master variables were not used */
   if( !divingdata->usemasterscores )
      return SCIP_OKAY;

   /* free memory */
   SCIPfreeBufferArray(scip, &divingdata->masterscores);

   return SCIP_OKAY;
}


/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. vector length:
 * - round variables in direction where objective value gets worse; for zero objective coefficient, round upwards
 * - round variable with least objective value deficit per row the variable appears in
 *   (we want to "fix" as many rows as possible with the least damage to the objective function)
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarGcgveclendiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   int nlpcands;
   SCIP_Real bestscore;
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

   bestscore = SCIP_REAL_MAX;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_Real score;
      SCIP_Bool roundup;

      int i;

      /* if the variable is on the tabu list, do not choose it */
       for( i = 0; i < tabulistsize; ++i )
          if( tabulist[i] == lpcands[c] )
             break;
       if( i < tabulistsize )
          continue;

      /* calculate score */
      if( divingdata->usemasterscores )
      {
         SCIP_CALL( calculateScoreMaster(scip, divingdata, lpcands[c], &score, &roundup) );
      }
      else
      {
         SCIP_CALL( calculateScoreOrig(scip, lpcands[c], lpcandssol[c] - SCIPfloor(scip, lpcandssol[c]), &score, &roundup) );
      }

      /* check whether the variable is roundable */
      *bestcandmayround = *bestcandmayround && (SCIPvarMayRoundDown(lpcands[c]) || SCIPvarMayRoundUp(lpcands[c]));

      /* check, if candidate is new best candidate */
      if( score < bestscore )
      {
         *bestcand = lpcands[c];
         bestscore = score;
         *bestcandroundup = roundup;
      }
   }

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the gcgveclendiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurGcgveclendiving(
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
         HEUR_MAXDEPTH, heurFreeGcgveclendiving, NULL, NULL, NULL, NULL, heurInitexecGcgveclendiving,
         heurExitexecGcgveclendiving, heurSelectVarGcgveclendiving, divingdata) );

   assert(heur != NULL);

   /* add gcgveclendiving specific parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/usemasterscores",
         "calculate vector length scores w.r.t. the master LP?",
         &divingdata->usemasterscores, TRUE, DEFAULT_USEMASTERSCORES, NULL, NULL) );

   return SCIP_OKAY;
}

