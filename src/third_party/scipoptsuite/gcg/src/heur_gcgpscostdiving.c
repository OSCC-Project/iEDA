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

/**@file   heur_gcgpscostdiving.c
 * @brief  LP diving heuristic that chooses fixings w.r.t. the pseudo cost values
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgpscostdiving.h"
#include "heur_origdiving.h"
#include "gcg.h"


#define HEUR_NAME             "gcgpscostdiving"
#define HEUR_DESC             "LP diving heuristic that chooses fixings w.r.t. the pseudo cost values"
#define HEUR_DISPCHAR         'p'
#define HEUR_PRIORITY         -1002000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          2
#define HEUR_MAXDEPTH         -1


/*
 * Default diving rule specific parameter settings
 */

#define DEFAULT_USEMASTERPSCOSTS  FALSE      /**< shall pseudocosts be calculated w.r.t. the master problem? */


/* locally defined diving heuristic data */
struct GCG_DivingData
{
   SCIP_Bool             usemasterpscosts;   /**< shall pseudocosts be calculated w.r.t. the master problem? */
   SCIP_SOL*             rootsol;            /**< relaxation solution at the root node */
   SCIP_Bool             firstrun;           /**< is the heuristic running for the first time? */
   SCIP_Real*            masterpscosts;      /**< pseudocosts of the master variables */
};


/*
 * local methods
 */

/** get relaxation solution of root node (in original variables) */
static
SCIP_RETCODE getRootRelaxSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            rootsol             /**< pointer to store root relaxation solution */
   )
{
   SCIP* masterprob;
   SCIP_SOL* masterrootsol;
   SCIP_VAR** mastervars;
   int nmastervars;
   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* allocate memory for master root LP solution */
   SCIP_CALL( SCIPcreateSol(masterprob, &masterrootsol, NULL) );

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* store root LP values in working master solution */
   for( i = 0; i < nmastervars; i++ )
      SCIP_CALL( SCIPsetSolVal(masterprob, masterrootsol, mastervars[i], SCIPvarGetRootSol(mastervars[i])) );

   /* calculate original root LP solution */
   SCIP_CALL( GCGtransformMastersolToOrigsol(scip, masterrootsol, rootsol) );

   /* free memory */
   SCIP_CALL( SCIPfreeSol(masterprob, &masterrootsol) );

   return SCIP_OKAY;
}

/** calculate pseudocosts for the master variables */
static
SCIP_RETCODE calcMasterPscosts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real**           masterpscosts       /**< pointer to store the array of master pseudocosts */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   int nbinvars;
   int nintvars;

   int i;
   int j;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variable data */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, NULL, &nbinvars, &nintvars, NULL, NULL) );

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, masterpscosts, nbinvars + nintvars) );

   /* calculate pseudocosts */
   for( i = 0; i < nbinvars + nintvars; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_VAR** masterorigvars;
      SCIP_Real* masterorigvals;
      int nmasterorigvars;

      mastervar = mastervars[i];
      masterorigvars = GCGmasterVarGetOrigvars(mastervar);
      masterorigvals = GCGmasterVarGetOrigvals(mastervar);
      nmasterorigvars = GCGmasterVarGetNOrigvars(mastervar);

      (*masterpscosts)[i] = 0.0;
      for( j = 0; j < nmasterorigvars; ++j )
      {
         SCIP_VAR* origvar;

         origvar = masterorigvars[j];
         if( !SCIPvarIsBinary(origvar) && !SCIPvarIsIntegral(origvar) )
            continue;
         (*masterpscosts)[i] += SCIPgetVarPseudocostVal(scip, origvar, 0.0-masterorigvals[j]);
      }
   }

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

/** calculates the down-pseudocost for a given original variable w.r.t. the master variables in which it is contained */
static
SCIP_RETCODE calcPscostDownMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real*            masterpscosts,      /**< master variable pseudocosts */
   SCIP_Real*            pscostdown          /**< pointer to store the pseudocost value */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nbinmastervars;
   int nintmastervars;
   int norigmastervars;
   SCIP_Real roundval;
   SCIP_Real masterlpval;
   int idx;

   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, NULL, &nbinmastervars, &nintmastervars, NULL, NULL) );

   /* get master variables in which the original variable appears */
   origmastervars = GCGoriginalVarGetMastervars(var);
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasFloor(scip, SCIPgetRelaxSolVal(scip, var));
   *pscostdown = 0.0;

   /* calculate sum of pseudocosts over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasNegative(masterprob, roundval) )
   {
      for( i = 0; i < nbinmastervars + nintmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i]) )
         {
            masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
            *pscostdown += masterpscosts[i] * SCIPfeasFrac(masterprob, masterlpval);
         }
      }
      for( i = 0; i < norigmastervars; ++i )
      {
         idx = SCIPvarGetProbindex(origmastervars[i]);
         masterlpval = SCIPgetSolVal(masterprob, NULL, origmastervars[i]);
         if( (SCIPvarIsBinary(origmastervars[i]) || SCIPvarIsIntegral(origmastervars[i]) )
            && SCIPisFeasLE(masterprob, origmastervals[i], roundval) )
            *pscostdown -= masterpscosts[idx] * SCIPfeasFrac(masterprob, masterlpval);
      }
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
      {
         idx = SCIPvarGetProbindex(origmastervars[i]);
         masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
         if( (SCIPvarIsBinary(origmastervars[i]) || SCIPvarIsIntegral(origmastervars[i]) )
            && SCIPisFeasGT(masterprob, origmastervals[i], roundval) )
            *pscostdown += masterpscosts[idx] * SCIPfeasFrac(masterprob, masterlpval);
      }
   }

   return SCIP_OKAY;
}

/** calculates the up-pseudocost for a given original variable w.r.t. the master variables in which it is contained */
static
SCIP_RETCODE calcPscostUpMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real*            masterpscosts,      /**< master variable pseudocosts */
   SCIP_Real*            pscostup            /**< pointer to store the pseudocost value */
   )
{
   SCIP* masterprob;
   SCIP_VAR** mastervars;
   SCIP_VAR** origmastervars;
   SCIP_Real* origmastervals;
   int nbinmastervars;
   int nintmastervars;
   int norigmastervars;
   SCIP_Real roundval;
   SCIP_Real masterlpval;
   int idx;

   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, NULL, &nbinmastervars, &nintmastervars, NULL, NULL) );

   /* get master variables in which the original variable appears */
   origmastervars = GCGoriginalVarGetMastervars(var);
   origmastervals = GCGoriginalVarGetMastervals(var);
   norigmastervars = GCGoriginalVarGetNMastervars(var);

   roundval = SCIPfeasCeil(scip, SCIPgetRelaxSolVal(scip, var));
   *pscostup = 0.0;

   /* calculate sum of pseudocosts over all master variables
    * which would violate the new original variable bound
    */
   if( SCIPisFeasPositive(masterprob, roundval) )
   {
      for( i = 0; i < nbinmastervars + nintmastervars; ++i )
      {
         if( areVarsInSameBlock(var, mastervars[i]) )
         {
            masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
            *pscostup += masterpscosts[i] * SCIPfeasFrac(masterprob, masterlpval);
         }
      }
      for( i = 0; i < norigmastervars; ++i )
      {
         idx = SCIPvarGetProbindex(origmastervars[i]);
         masterlpval = SCIPgetSolVal(masterprob, NULL, origmastervars[i]);
         if( (SCIPvarIsBinary(origmastervars[i]) || SCIPvarIsIntegral(origmastervars[i]) )
            && SCIPisFeasGE(masterprob, origmastervals[i], roundval) )
            *pscostup -= masterpscosts[idx] * SCIPfeasFrac(masterprob, masterlpval);
      }
   }
   else
   {
      for( i = 0; i < norigmastervars; ++i )
      {
         idx = SCIPvarGetProbindex(origmastervars[i]);
         masterlpval = SCIPgetSolVal(masterprob, NULL, mastervars[i]);
         if( (SCIPvarIsBinary(origmastervars[i]) || SCIPvarIsIntegral(origmastervars[i]) )
            && SCIPisFeasLT(masterprob, origmastervals[i], roundval) )
            *pscostup += masterpscosts[idx] * SCIPfeasFrac(masterprob, masterlpval);
      }
   }

   return SCIP_OKAY;
}

/** calculates the pseudocost score for a given variable w.r.t. a given solution value and a given rounding direction */
static
SCIP_RETCODE calcPscostQuot(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_DIVINGDATA*       divingdata,         /**< diving data */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             primsol,            /**< primal solution of variable */
   SCIP_Real             rootsolval,         /**< root relaxation solution of variable */
   SCIP_Real             frac,               /**< fractionality of variable */
   int                   rounddir,           /**< -1: round down, +1: round up, 0: select due to pseudo cost values */
   SCIP_Real*            pscostquot,         /**< pointer to store pseudo cost quotient */
   SCIP_Bool*            roundup             /**< pointer to store whether the variable should be rounded up */
   )
{
   SCIP_Real pscostdown;
   SCIP_Real pscostup;

   assert(pscostquot != NULL);
   assert(roundup != NULL);
   assert(SCIPisEQ(scip, frac, primsol - SCIPfeasFloor(scip, primsol)));

   /* bound fractions to not prefer variables that are nearly integral */
   frac = MAX(frac, 0.1);
   frac = MIN(frac, 0.9);

   /* get pseudo cost quotient */
   if( divingdata->usemasterpscosts )
   {
      SCIP_CALL( calcPscostDownMaster(scip, var, divingdata->masterpscosts, &pscostdown) );
      SCIP_CALL( calcPscostUpMaster(scip, var, divingdata->masterpscosts, &pscostup) );
   }
   else
   {
      pscostdown = SCIPgetVarPseudocostVal(scip, var, 0.0-frac);
      pscostup = SCIPgetVarPseudocostVal(scip, var, 1.0-frac);
   }
   SCIPdebugMessage("Pseudocosts of variable %s: %g down, %g up\n", SCIPvarGetName(var), pscostdown, pscostup);
   assert(!SCIPisNegative(scip, pscostdown) && !SCIPisNegative(scip,pscostup));

   /* choose rounding direction */
   if( rounddir == -1 )
      *roundup = FALSE;
   else if( rounddir == +1 )
      *roundup = TRUE;
   else if( primsol < rootsolval - 0.4 )
      *roundup = FALSE;
   else if( primsol > rootsolval + 0.4 )
      *roundup = TRUE;
   else if( frac < 0.3 )
      *roundup = FALSE;
   else if( frac > 0.7 )
      *roundup = TRUE;
   else if( pscostdown < pscostup )
      *roundup = FALSE;
   else
      *roundup = TRUE;

   /* calculate pseudo cost quotient */
   if( *roundup )
      *pscostquot = sqrt(frac) * (1.0+pscostdown) / (1.0+pscostup);
   else
      *pscostquot = sqrt(1.0-frac) * (1.0+pscostup) / (1.0+pscostdown);

   /* prefer decisions on binary variables */
   if( SCIPvarIsBinary(var) )
      (*pscostquot) *= 1000.0;

   return SCIP_OKAY;
}


/*
 * Callback methods
 */

/** destructor of diving heuristic to free user data (called when GCG is exiting) */
static
GCG_DECL_DIVINGFREE(heurFreeGcgpscostdiving) /*lint --e{715}*/
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


/** initialization method of diving heuristic (called after problem was transformed) */
static
GCG_DECL_DIVINGINIT(heurInitGcgpscostdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* initialize data */
   divingdata->firstrun = TRUE;
   divingdata->rootsol = NULL;
   divingdata->masterpscosts = NULL;

   return SCIP_OKAY;
}


/** deinitialization method of diving heuristic (called before transformed problem is freed) */
static
GCG_DECL_DIVINGEXIT(heurExitGcgpscostdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   assert(divingdata->firstrun == TRUE || divingdata->rootsol != NULL);

   /* free root relaxation solution */
   if( divingdata->rootsol != NULL )
      SCIP_CALL( SCIPfreeSol(scip, &divingdata->rootsol) );

   return SCIP_OKAY;
}


/** execution initialization method of diving heuristic (called when execution of diving heuristic is about to begin) */
static
GCG_DECL_DIVINGINITEXEC(heurInitexecGcgpscostdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* if the heuristic is running for the first time, the root relaxation solution needs to be stored */
   if( divingdata->firstrun )
   {
      assert(divingdata->rootsol == NULL);
      SCIP_CALL( getRootRelaxSol(scip, &divingdata->rootsol) );
      assert(divingdata->rootsol != NULL);
      divingdata->firstrun = FALSE;
   }

   SCIP_CALL( calcMasterPscosts(scip, &divingdata->masterpscosts) );

   return SCIP_OKAY;
}


/** execution deinitialization method of diving heuristic (called when execution data is freed) */
static
GCG_DECL_DIVINGEXITEXEC(heurExitexecGcgpscostdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;

   assert(heur != NULL);

   /* get diving data */
   divingdata = GCGheurGetDivingDataOrig(heur);
   assert(divingdata != NULL);

   /* free memory */
   SCIPfreeBufferArray(scip, &divingdata->masterpscosts);

   return SCIP_OKAY;
}


/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. pseudo costs:
 * - prefer variables that may not be rounded without destroying LP feasibility:
 *   - of these variables, round variable with largest rel. difference of pseudo cost values in corresponding
 *     direction
 * - if all remaining fractional variables may be rounded without destroying LP feasibility:
 *   - round variable in the objective value direction
 * - binary variables are preferred
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarGcgpscostdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   GCG_DIVINGDATA* divingdata;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   int nlpcands;
   SCIP_Bool bestcandmayrounddown;
   SCIP_Bool bestcandmayroundup;
   SCIP_Real bestpscostquot;
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
   assert(divingdata->rootsol != NULL);

   /* get fractional variables that should be integral */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &lpcands, &lpcandssol, NULL, &nlpcands, NULL, NULL, NULL, NULL) );
   assert(lpcands != NULL);
   assert(lpcandssol != NULL);

   bestcandmayrounddown = TRUE;
   bestcandmayroundup = TRUE;
   bestpscostquot = -1.0;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;
      SCIP_Real primsol;

      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;
      SCIP_Bool roundup;
      SCIP_Real frac;
      SCIP_Real pscostquot;
      SCIP_Real rootsolval;

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
      primsol = lpcandssol[c];
      rootsolval = SCIPgetSolVal(scip, divingdata->rootsol, var);
      frac = lpcandssol[c] - SCIPfloor(scip, lpcandssol[c]);

      if( mayrounddown || mayroundup )
      {
         /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
         if( bestcandmayrounddown || bestcandmayroundup )
         {
            /* choose rounding direction:
             * - if variable may be rounded in both directions, round corresponding to the pseudo cost values
             * - otherwise, round in the infeasible direction, because feasible direction is tried by rounding
             *   the current fractional solution
             */
            roundup = FALSE;
            if( mayrounddown && mayroundup )
            {
               SCIP_CALL( calcPscostQuot(scip, divingdata, var, primsol, rootsolval, frac, 0, &pscostquot, &roundup) );
            }
            else if( mayrounddown )
            {
               SCIP_CALL( calcPscostQuot(scip, divingdata, var, primsol, rootsolval, frac, +1, &pscostquot, &roundup) );
            }
            else
            {
               SCIP_CALL( calcPscostQuot(scip, divingdata, var, primsol, rootsolval, frac, -1, &pscostquot, &roundup) );
            }

            /* check, if candidate is new best candidate */
            if( pscostquot > bestpscostquot )
            {
               *bestcand = var;
               bestpscostquot = pscostquot;
               bestcandmayrounddown = mayrounddown;
               bestcandmayroundup = mayroundup;
               *bestcandroundup = roundup;
            }
         }
      }
      else
      {
         /* the candidate may not be rounded: calculate pseudo cost quotient and preferred direction */
         SCIP_CALL( calcPscostQuot(scip, divingdata, var, primsol, rootsolval, frac, 0, &pscostquot, &roundup) );

         /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
         if( bestcandmayrounddown || bestcandmayroundup || pscostquot > bestpscostquot )
         {
            *bestcand = var;
            bestpscostquot = pscostquot;
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

/** creates the gcgpscostdiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurGcgpscostdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEUR* heur;
   GCG_DIVINGDATA* divingdata;

   /* create gcgpscostdiving primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &divingdata) );

   /* include diving heuristic */
   SCIP_CALL( GCGincludeDivingHeurOrig(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, heurFreeGcgpscostdiving, heurInitGcgpscostdiving, heurExitGcgpscostdiving, NULL, NULL,
         heurInitexecGcgpscostdiving, heurExitexecGcgpscostdiving, heurSelectVarGcgpscostdiving, divingdata) );

   assert(heur != NULL);

   /* add gcgpscostdiving specific parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/usemasterpscosts",
         "shall pseudocosts be calculated w.r.t. the master problem?",
         &divingdata->usemasterpscosts, TRUE, DEFAULT_USEMASTERPSCOSTS, NULL, NULL) );

   return SCIP_OKAY;
}

