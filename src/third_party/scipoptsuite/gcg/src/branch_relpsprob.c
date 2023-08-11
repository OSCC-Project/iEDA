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

/**@file   branch_relpsprob.c
 * @ingroup BRANCHINGRULES
 * @brief  generalized reliable pseudo costs branching rule
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Jens Schulz
 * @author Gerald Gamrath
 *
 * - probing is executed until depth 10 and afterwards with stepsize 5
 *   by that all pseudocost scores and inference informations are updated
 *   otherwise the variable with best score is branched on
 * - NEW! probing is done according to reliability values per candidate depending on tree size and probing rounds
 * - the node is reevaluated immediately if MAXBDCHGS occur during probing
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */
#include <assert.h>
#include <string.h>

#include "branch_relpsprob.h"
#include "relax_gcg.h"
#include "cons_integralorig.h"
#include "pricer_gcg.h"
#include "gcg.h"

#include "scip/nodesel_estimate.h"
#include "scip/nodesel_hybridestim.h"
#include "scip/nodesel_restartdfs.h"
#include "scip/branch_allfullstrong.h"
#include "scip/branch_fullstrong.h"
#include "scip/branch_inference.h"
#include "scip/branch_mostinf.h"
#include "scip/branch_leastinf.h"
#include "scip/branch_pscost.h"
#include "scip/branch_random.h"
#include "scip/branch_relpscost.h"
#include "scip/nodesel_bfs.h"
#include "scip/nodesel_dfs.h"


#define BRANCHRULE_NAME          "relpsprob"
#define BRANCHRULE_DESC          "generalized reliability branching using probing"
#define BRANCHRULE_PRIORITY      -100
#define BRANCHRULE_MAXDEPTH      -1
#define BRANCHRULE_MAXBOUNDDIST  1.0

#define DEFAULT_CONFLICTWEIGHT   0.01   /**< weight in score calculations for conflict score */
#define DEFAULT_CONFLENGTHWEIGHT 0.0001 /**< weight in score calculations for conflict length score*/
#define DEFAULT_INFERENCEWEIGHT  0.1    /**< weight in score calculations for inference score */
#define DEFAULT_CUTOFFWEIGHT     0.0001 /**< weight in score calculations for cutoff score */
#define DEFAULT_PSCOSTWEIGHT     1.0    /**< weight in score calculations for pseudo cost score */
#define DEFAULT_MINRELIABLE      1.0    /**< minimal value for minimum pseudo cost size to regard pseudo cost value as reliable */
#define DEFAULT_MAXRELIABLE      8.0    /**< maximal value for minimum pseudo cost size to regard pseudo cost value as reliable */
#define DEFAULT_ITERQUOT         0.5    /**< maximal fraction of branching LP iterations compared to normal iters */
#define DEFAULT_ITEROFS     100000      /**< additional number of allowed LP iterations */
#define DEFAULT_MAXLOOKAHEAD     8      /**< maximal number of further variables evaluated without better score */
#define DEFAULT_INITCAND       100      /**< maximal number of candidates initialized with strong branching per node */
#define DEFAULT_MAXBDCHGS       20      /**< maximal number of bound tightenings before the node is immediately reevaluated (-1: unlimited) */
#define DEFAULT_MINBDCHGS        1      /**< minimal number of bound tightenings before the node is reevaluated */
#define DEFAULT_USELP            TRUE   /**< shall the lp be solved during probing? */
#define DEFAULT_RELIABILITY      0.8    /**< reliability value for probing */

#define HASHSIZE_VARS            131101 /**< minimal size of hash table in bdchgdata */


/** branching rule data */
struct SCIP_BranchruleData
{
   SCIP_Real             conflictweight;     /**< weight in score calculations for conflict score */
   SCIP_Real             conflengthweight;   /**< weight in score calculations for conflict length score */
   SCIP_Real             inferenceweight;    /**< weight in score calculations for inference score */
   SCIP_Real             cutoffweight;       /**< weight in score calculations for cutoff score */
   SCIP_Real             pscostweight;       /**< weight in score calculations for pseudo cost score */
   SCIP_Real             minreliable;        /**< minimal value for minimum pseudo cost size to regard pseudo cost value as reliable */
   SCIP_Real             maxreliable;        /**< maximal value for minimum pseudo cost size to regard pseudo cost value as reliable */
   SCIP_Real             iterquot;           /**< maximal fraction of branching LP iterations compared to normal iters */
   SCIP_Longint          nlpiterations;      /**< total number of used LP iterations */
   int                   iterofs;            /**< additional number of allowed LP iterations */
   int                   maxlookahead;       /**< maximal number of further variables evaluated without better score */
   int                   initcand;           /**< maximal number of candidates initialized with strong branching per node */
   int                   maxbdchgs;          /**< maximal number of bound tightenings before the node is immediately reevaluated (-1: unlimited) */
   int                   minbdchgs;          /**< minimal number of bound tightenings before bound changes are applied */
   SCIP_Bool             uselp;              /**< shall the lp be solved during probing? */
   int                   nprobingnodes;      /**< counter to store the total number of probing nodes */
   int                   ninfprobings;       /**< counter to store the number of probings which led to an infeasible branch */
   SCIP_Real             reliability;        /**< reliability value for branching variables */
   int                   nbranchings;        /**< counter to store the total number of nodes that are branched */
   int                   nresolvesminbdchgs; /**< counter to store how often node is reevaluated due to min bdchgs */
   int                   nresolvesinfcands;   /**< counter to store how often node is reevaluated since candidate with inf branch is chosen */
   int                   nprobings;          /**< counter to store the total number of probings that were performed */
   SCIP_HASHMAP*         varhashmap;         /**< hash storing variables; image is position in following arrays */
   int*                  nvarbranchings;     /**< array to store number of branchings per variable */
   int*                  nvarprobings;       /**< array to store number of probings per variable */
   int                   nvars;              /**< number of variables that are in hashmap */
   int                   maxvars;            /**< capacity of nvarbranchings and nvarprobings */
};

/** data for pending bound changes */
struct BdchgData
{
   SCIP_HASHMAP*         varhashmap;         /**< hash storing variables; image is position in lbchgs-array */
   SCIP_Real*            lbchgs;             /**< array containing lower bounds per variable */
   SCIP_Real*            ubchgs;             /**< array containing upper bounds per variable */
   SCIP_Bool*            infroundings;       /**< array to store for each var if some rounding is infeasible */
   int                   nvars;              /**< number of variables that are considered so far */
};
typedef struct BdchgData BDCHGDATA;


/*
 * local methods
 */

/** creates bound change data structure:
 * all variables are put into a hashmap and arrays containig current lower and upper bounds are created
 */
static
SCIP_RETCODE createBdchgData(
   SCIP*                 scip,               /**< SCIP data structure */
   BDCHGDATA**           bdchgdata,          /**< bound change data to be allocated */
   SCIP_VAR**            vars,               /**< array of variables to be watched */
   int                   nvars               /**< number of variables to be watched */
   )
{

   int i;

   assert(scip != NULL);
   assert(*bdchgdata == NULL);

   /* get memory for bound change data structure */
   SCIP_CALL( SCIPallocBuffer(scip, bdchgdata) );

   /* create hash map */
   SCIP_CALL( SCIPhashmapCreate(&(*bdchgdata)->varhashmap, SCIPblkmem(scip), HASHSIZE_VARS) );

   /* get all variables */
   SCIP_CALL( SCIPallocBufferArray(scip, &(*bdchgdata)->lbchgs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &(*bdchgdata)->ubchgs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &(*bdchgdata)->infroundings, nvars) );
   (*bdchgdata)->nvars = nvars;

   /* store current local bounds and capture variables */
   for( i = 0; i < nvars; ++i )
   {
      SCIP_CALL( SCIPhashmapInsert((*bdchgdata)->varhashmap, vars[i], (void*) (size_t)i) );

      (*bdchgdata)->lbchgs[i] = SCIPfeasCeil(scip, SCIPvarGetLbLocal(vars[i]));
      (*bdchgdata)->ubchgs[i] = SCIPfeasFloor(scip, SCIPvarGetUbLocal(vars[i]));
      (*bdchgdata)->infroundings[i] = FALSE;
   }

   return SCIP_OKAY;
}

/** method to free bound change data strucutre */
static
SCIP_RETCODE freeBdchgData(
   SCIP*                 scip,               /**< SCIP data structure */
   BDCHGDATA*            bdchgdata           /**< bound change data to be allocated */
   )
{

   assert(scip != NULL);
   assert(bdchgdata != NULL);

   /* free arrays & hashmap */
   SCIPfreeBufferArray(scip, &bdchgdata->infroundings);
   SCIPfreeBufferArray(scip, &bdchgdata->ubchgs);
   SCIPfreeBufferArray(scip, &bdchgdata->lbchgs);

   SCIPhashmapFree(&(bdchgdata->varhashmap));

   /* free memory for bound change data structure */
   SCIPfreeBuffer(scip, &bdchgdata);

   return SCIP_OKAY;
}


/** adds given variable and bound change to hashmap and bound change arrays */
static
SCIP_RETCODE addBdchg(
   SCIP*                 scip,               /**< SCIP data structure */
   BDCHGDATA*            bdchgdata,          /**< structure to keep bound chage data */
   SCIP_VAR*             var,                /**< variable to store bound change */
   SCIP_Real             newbound,           /**< new bound for given variable */
   SCIP_BOUNDTYPE        boundtype,          /**< lower or upper bound change */
   SCIP_Bool             infrounding,        /**< is the bdchg valid due to an infeasible rounding of the given var */
   int*                  nbdchgs,            /**< total number of bound changes occured so far */
   SCIP_Bool*            infeasible          /**< pointer to store whether bound change makes the node infeasible */
   )
{
   int nvars;
   int pos;

   assert(scip != NULL);
   assert(bdchgdata != NULL);
   assert(bdchgdata->varhashmap != NULL);
   assert(bdchgdata->lbchgs != NULL);
   assert(bdchgdata->ubchgs != NULL);
   assert(var != NULL);

   nvars = bdchgdata->nvars;

   if( infeasible != NULL )
      *infeasible = FALSE;

   /* if variable is not in hashmap insert it and increase array sizes */
   if( !SCIPhashmapExists(bdchgdata->varhashmap, var) )
   {
      SCIP_CALL( SCIPhashmapInsert(bdchgdata->varhashmap, var, (void*) (size_t)nvars) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &bdchgdata->lbchgs, nvars + 1) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &bdchgdata->ubchgs, nvars + 1) );

      bdchgdata->lbchgs[nvars] = SCIPfeasCeil(scip, SCIPvarGetLbLocal(var));
      bdchgdata->ubchgs[nvars] = SCIPfeasFloor(scip, SCIPvarGetUbLocal(var));
      (bdchgdata->nvars)++;

      assert(SCIPhashmapExists(bdchgdata->varhashmap, var)
         && (int)(size_t) SCIPhashmapGetImage(bdchgdata->varhashmap, var) == nvars); /*lint !e507*/

   }

   /* get position of this variable */
   pos = (int)(size_t) SCIPhashmapGetImage(bdchgdata->varhashmap, var); /*lint !e507*/

   if( infrounding )
   {
      bdchgdata->infroundings[pos] = TRUE;
   }

   /* update bounds if necessary */
   if( boundtype == SCIP_BOUNDTYPE_LOWER )
   {
      if( bdchgdata->lbchgs[pos] < newbound )
      {
         bdchgdata->lbchgs[pos] = newbound;
         (*nbdchgs)++;
      }

      if( (infeasible != NULL) && (newbound > bdchgdata->ubchgs[pos]) )
      {
         *infeasible = TRUE;
      }

   }
   else
   {
      if( newbound < bdchgdata->ubchgs[pos] )
      {
         bdchgdata->ubchgs[pos] = newbound;
         (*nbdchgs)++;
      }
      if( (infeasible != NULL) && (newbound < bdchgdata->lbchgs[pos]) )
      {
         *infeasible = TRUE;
      }
   }

   return SCIP_OKAY;
}



/** applies bound changes stored in bound change arrays */
static
SCIP_RETCODE applyBdchgs(
   SCIP*                 scip,               /**< SCIP data structure */
   BDCHGDATA*            bdchgdata,          /**< structure containing bound changes for almost all variables */
   SCIP_NODE*            node                /**< node for which bound changes are applied, NULL for curnode */
   )
{
   SCIP_VAR** vars;

   int nintvars;
   int nbinvars;
   int nvars;
   int nbdchgs;
   int i;

   assert(scip != NULL);
   assert(bdchgdata != NULL);

   SCIPdebugMessage("apply bound changes\n");

   nbdchgs = 0;

   /* get all variables */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, NULL, &nbinvars, &nintvars, NULL, NULL) );
   nvars = nbinvars + nintvars;
   assert(vars != NULL);

   /* get variable image in hashmap and update bounds if better ones found  */
   for( i = 0; i < nvars; ++i )
   {
      if( SCIPhashmapExists(bdchgdata->varhashmap, vars[i]) )
      {
         int pos;
         pos = (int)(size_t)SCIPhashmapGetImage(bdchgdata->varhashmap, vars[i]); /*lint !e507*/

         /* update lower bounds */
         if( SCIPisFeasGT(scip, (bdchgdata->lbchgs)[pos], SCIPvarGetLbLocal(vars[i])) )
         {
            SCIPdebugMessage("branch_relpsprob: update lower bound of <%s> from %g to %g\n",
               SCIPvarGetName(vars[i]), SCIPvarGetLbLocal(vars[i]), (bdchgdata->lbchgs)[pos]);
            SCIP_CALL( SCIPchgVarLbNode(scip, node, vars[i], (bdchgdata->lbchgs)[pos]) );
            ++nbdchgs;
         }
         /* update upper bounds */
         if( SCIPisFeasLT(scip, (bdchgdata->ubchgs)[pos], SCIPvarGetUbLocal(vars[i])) )
         {
            SCIPdebugMessage("branch_relpsprob: update upper bound of <%s> from %g to %g\n",
               SCIPvarGetName(vars[i]), SCIPvarGetUbLocal(vars[i]), (bdchgdata->ubchgs)[pos]);

            SCIP_CALL( SCIPchgVarUbNode(scip, node, vars[i], (bdchgdata->ubchgs)[pos]) );
            ++nbdchgs;
         }
      }
   }

   SCIPdebugMessage("applied %d bound changes\n", nbdchgs);
   return SCIP_OKAY;
}


/** calculates an overall score value for the given individual score values */
static
SCIP_Real calcScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULEDATA*  branchruledata,     /**< branching rule data */
   SCIP_Real             conflictscore,      /**< conflict score of current variable */
   SCIP_Real             avgconflictscore,   /**< average conflict score */
   SCIP_Real             conflengthscore,    /**< conflict length score of current variable */
   SCIP_Real             avgconflengthscore, /**< average conflict length score */
   SCIP_Real             inferencescore,     /**< inference score of current variable */
   SCIP_Real             avginferencescore,  /**< average inference score */
   SCIP_Real             cutoffscore,        /**< cutoff score of current variable */
   SCIP_Real             avgcutoffscore,     /**< average cutoff score */
   SCIP_Real             pscostscore,        /**< pscost score of current variable */
   SCIP_Real             avgpscostscore,     /**< average pscost score */
   SCIP_Real             frac                /**< fractional value of variable in current solution */
   )
{
   SCIP_Real score;

   assert(branchruledata != NULL);
   /*    assert(0.0 < frac && frac < 1.0); */

   score = branchruledata->conflictweight * (1.0 - 1.0/(1.0+conflictscore/avgconflictscore))
      + branchruledata->conflengthweight * (1.0 - 1.0/(1.0+conflengthscore/avgconflengthscore))
      + branchruledata->inferenceweight * (1.0 - 1.0/(1.0+inferencescore/avginferencescore))
      + branchruledata->cutoffweight * (1.0 - 1.0/(1.0+cutoffscore/avgcutoffscore))
      + branchruledata->pscostweight * (1.0 - 1.0/(1.0+pscostscore/avgpscostscore));

   /* values close to integral are possible and are adjusted to small non-zero values */
   if( frac < 0.00000001 || frac > 0.999999 )
      frac = 0.0001;
   if( MIN(frac, 1.0 - frac) < 10.0*SCIPfeastol(scip) )
      score *= 1e-6;

   return score;
}


/* calculates variable bounds for an up-branch and a down-branch, supposig a LP or pseudo solution is given */
static
SCIP_RETCODE calculateBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             branchvar,          /**< branching variable */
   SCIP_Real*            downlb,             /**< lower bound of variable in down branch */
   SCIP_Real*            downub,             /**< upper bound of variable in down branch */
   SCIP_Real*            uplb,               /**< lower bound of variable in up branch */
   SCIP_Real*            upub                /**< upper bound of variable in up branch */
   )
{
   SCIP_Real varsol;
   SCIP_Real lbdown;
   SCIP_Real ubdown;
   SCIP_Real lbup;
   SCIP_Real ubup;

   SCIP_Real lblocal;
   SCIP_Real ublocal;

   assert(scip != NULL);
   assert(branchvar != NULL);

   varsol = SCIPgetVarSol(scip, branchvar);

   lblocal = SCIPfeasCeil(scip, SCIPvarGetLbLocal(branchvar));
   ublocal = SCIPfeasFloor(scip, SCIPvarGetUbLocal(branchvar));

   /* calculate bounds in down branch */
   lbdown = lblocal;

   /* in down branch: new upper bound is at most local upper bound - 1 */
   ubdown = SCIPfeasFloor(scip, varsol) ;
   if( SCIPisEQ(scip, ubdown, ublocal) )
      ubdown -= 1.0;

   assert(lbdown <= ubdown);

   /* calculate bounds in up branch */
   ubup = ublocal;

   /* in right branch: new lower bound is at least local lower bound + 1 */
   lbup = SCIPfeasCeil(scip, varsol);
   if( SCIPisEQ(scip, lbup, lblocal) )
      lbup += 1.0;

   assert(SCIPisLE(scip, lbup, ubup));

   /* ensure that both branches partition the domain */
   if( SCIPisEQ(scip, lbup, ubdown) )
   {
      SCIP_Real middle = SCIPfloor(scip, lblocal/2 + ublocal/2);

      if( SCIPisLE(scip, lbup, middle) )
         ubdown -= 1.0;
      else
         lbup += 1.0;
   }

   /* ensure a real partition of the domain */
   assert(SCIPisLT(scip, ubdown, lbup));
   assert(SCIPisLE(scip, lbdown, ubdown));
   assert(SCIPisLE(scip, lbup, ubup));

   /* set return values */
   if( downlb != NULL )
      *downlb = lbdown;
   if( downub != NULL )
      *downub = ubdown;
   if( uplb != NULL )
      *uplb = lbup;
   if( upub != NULL )
      *upub = ubup;

   return SCIP_OKAY;
}


/** applies probing of a single variable in the given direction, and stores evaluation in given arrays */
static
SCIP_RETCODE applyProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< problem variables */
   int                   nvars,              /**< number of problem variables */
   SCIP_VAR*             probingvar,         /**< variable to perform probing on */
   SCIP_Bool             probingdir,         /**< value to fix probing variable to */
   SCIP_Bool             solvelp,            /**< value to decide whether pricing loop shall be performed */
   SCIP_Longint*         nlpiterations,      /**< pointer to store the number of used LP iterations */
   SCIP_Real*            proplbs,            /**< array to store lower bounds after full propagation */
   SCIP_Real*            propubs,            /**< array to store upper bounds after full propagation */
   SCIP_Real*            lpobjvalue,         /**< pointer to store the lp obj value if lp was solved */
   SCIP_Bool*            lpsolved,           /**< pointer to store whether the lp was solved */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing direction is infeasible */
   )
{
   SCIP* masterscip;

   /* SCIP_Real varsol; */
   SCIP_Real leftlbprobing;
   SCIP_Real leftubprobing;
   SCIP_Real rightlbprobing;
   SCIP_Real rightubprobing;

   leftubprobing = -1.0;
   leftlbprobing = -1.0;
   rightlbprobing = -1.0;
   rightubprobing = -1.0;

   assert(proplbs != NULL);
   assert(propubs != NULL);
   assert(cutoff != NULL);
   assert(SCIPvarGetLbLocal(probingvar) - 0.5 < SCIPvarGetUbLocal(probingvar));
   assert(SCIPisFeasIntegral(scip, SCIPvarGetLbLocal(probingvar)));
   assert(SCIPisFeasIntegral(scip, SCIPvarGetUbLocal(probingvar)));

   assert(!solvelp || (lpsolved!=NULL && lpobjvalue!=NULL && lperror!=NULL));

   /* get SCIP data structure of master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   /* varsol = SCIPgetRelaxSolVal(scip, probingvar); */

   if( probingdir == FALSE )
   {

      SCIP_CALL( calculateBounds(scip, probingvar,
            &leftlbprobing, &leftubprobing, NULL, NULL) );
   }
   else
   {
      SCIP_CALL( calculateBounds(scip, probingvar,
            NULL, NULL, &rightlbprobing, &rightubprobing) );
   }

   SCIPdebugMessage("applying probing on variable <%s> == %u [%g,%g] (nlocks=%d/%d, impls=%d/%d, clqs=%d/%d)\n",
      SCIPvarGetName(probingvar), probingdir,
      probingdir ? rightlbprobing : leftlbprobing, probingdir ? rightubprobing : leftubprobing,
      SCIPvarGetNLocksDown(probingvar), SCIPvarGetNLocksUp(probingvar),
      SCIPvarGetNImpls(probingvar, FALSE), SCIPvarGetNImpls(probingvar, TRUE),
      SCIPvarGetNCliques(probingvar, FALSE), SCIPvarGetNCliques(probingvar, TRUE));

   /* start probing mode */
   SCIP_CALL( GCGrelaxStartProbing(scip, NULL) );
   SCIP_CALL( GCGrelaxNewProbingnodeOrig(scip) );

   *lpsolved = FALSE;
   *lperror = FALSE;

   /* change variable bounds for the probing directions*/
   if( probingdir == FALSE )
   {
      SCIP_CALL( SCIPchgVarUbProbing(scip, probingvar, leftubprobing) );
   }
   else
   {
      SCIP_CALL( SCIPchgVarLbProbing(scip, probingvar, rightlbprobing) );
   }

   /* apply propagation */
   if( !(*cutoff) )
   {
      SCIP_CALL( SCIPpropagateProbing(scip, -1, cutoff, NULL) ); /** @todo use maxproprounds */
   }

   /* evaluate propagation */
   if( !(*cutoff) )
   {
      int i;

      for( i = 0; i < nvars; ++i )
      {
         proplbs[i] = SCIPvarGetLbLocal(vars[i]);
         propubs[i] = SCIPvarGetUbLocal(vars[i]);
      }
   }

   /* if parameter is set, we want to use the outcome of the LP relaxation */
   if( !(*cutoff) && solvelp )
   {
      *nlpiterations -= SCIPgetNLPIterations(masterscip);

      /** @todo handle the feasible result */
      SCIP_CALL( GCGrelaxNewProbingnodeMaster(scip) );
      SCIP_CALL( GCGrelaxPerformProbingWithPricing(scip, -1, nlpiterations, NULL,
            lpobjvalue, lpsolved, lperror, cutoff) );
   }

   /* exit probing mode */
   SCIP_CALL( GCGrelaxEndProbing(scip) );

   SCIPdebugMessage("probing results in cutoff/lpsolved/lpobj: %s / %s / %g\n",
      *cutoff?"cutoff":"no cutoff", *lpsolved?"lpsolved":"lp not solved", *lpobjvalue);

   return SCIP_OKAY;
}


/** gets generalized strong branching information on problem variable */
static
SCIP_RETCODE getVarProbingbranch(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             probingvar,         /**< variable to get strong probing branching values for */
   BDCHGDATA*            bdchgdata,          /**< structure containing bound changes for almost all variables */
   SCIP_Bool             solvelp,            /**< value to decide whether pricing loop shall be performed */
   SCIP_Longint*         nlpiterations,      /**< pointert to stroe the number of used LP iterations */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< pointer to store whether the downwards branch is infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< pointer to store whether the upwards branch is infeasible, or NULL */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   int*                  nbdchgs             /**< pointer to store number of total bound changes */
   )
{
   /* data for variables and bdchg arrays */
   SCIP_VAR** probvars;
   SCIP_VAR** vars;
   int nvars;
   int nintvars;
   int nbinvars;

   SCIP_Real* leftproplbs;
   SCIP_Real* leftpropubs;
   SCIP_Real* rightproplbs;
   SCIP_Real* rightpropubs;

   SCIP_Real leftlpbound;
   SCIP_Real rightlpbound;
   SCIP_Bool leftlpsolved;
   SCIP_Bool rightlpsolved;
   SCIP_Bool leftlperror;
   SCIP_Bool rightlperror;
   SCIP_Bool leftcutoff;
   SCIP_Bool rightcutoff;
   SCIP_Bool cutoff;
   int i;
   int j;

   assert(lperror != NULL);

   if( downvalid != NULL )
      *downvalid = FALSE;
   if( upvalid != NULL )
      *upvalid = FALSE;
   if( downinf != NULL )
      *downinf = FALSE;
   if( upinf != NULL )
      *upinf = FALSE;

   if( SCIPisStopped(scip) )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL,
         "   (%.1fs) probing aborted: solving stopped\n", SCIPgetSolvingTime(scip));
      return SCIP_OKAY;
   }

   /* get all variables to store branching deductions of variable bounds */
   /* get all variables and store them in array 'vars' */
   SCIP_CALL( SCIPgetVarsData(scip, &probvars, NULL, &nbinvars, &nintvars, NULL, NULL) );
   nvars = nbinvars + nintvars; /* continuous variables are not considered here */

   SCIP_CALL( SCIPduplicateBufferArray(scip, &vars, probvars, nvars) );

   /* capture variables to make sure the variables are not deleted */
   for( i = 0; i < nvars; ++i )
   {
      SCIP_CALL( SCIPcaptureVar(scip, vars[i]) );
   }

   /* get temporary memory for storing probing results */
   SCIP_CALL( SCIPallocBufferArray(scip, &leftproplbs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &leftpropubs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &rightproplbs, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &rightpropubs, nvars) );

   /* for each binary variable, probe fixing the variable to left and right */
   cutoff = FALSE;
   leftcutoff = FALSE;
   rightcutoff = FALSE;

   /* better assume we don't have an error (fixes clang warning)*/
   leftlperror = FALSE;
   rightlperror = FALSE;

   /* better assume we don't have solved the lp (fixes clang warning)*/
   leftlpsolved = FALSE;
   rightlpsolved = FALSE;


   /* left branch: apply probing for setting ub to LP solution value  */
   SCIP_CALL( applyProbing(scip, vars, nvars, probingvar, FALSE, solvelp, nlpiterations,
         leftproplbs, leftpropubs,
         &leftlpbound, &leftlpsolved, &leftlperror, &leftcutoff) );

   if( leftcutoff )
   {
      SCIP_Real newbound;

      SCIP_CALL( calculateBounds(scip, probingvar, NULL, NULL, &newbound, NULL) );

      /* lower bound can be updated */
      SCIPdebugMessage("change lower bound of probing variable <%s> from %g to %g, nlocks=(%d/%d)\n",
         SCIPvarGetName(probingvar), SCIPvarGetLbLocal(probingvar), newbound,
         SCIPvarGetNLocksDown(probingvar), SCIPvarGetNLocksUp(probingvar));

      SCIP_CALL( addBdchg(scip, bdchgdata, probingvar, newbound, SCIP_BOUNDTYPE_LOWER, TRUE, nbdchgs, &cutoff) );
   }

   if( !cutoff )
   {
      /* right branch: apply probing for setting lb to LP solution value  */
      SCIP_CALL( applyProbing(scip, vars, nvars, probingvar, TRUE, solvelp, nlpiterations,
            rightproplbs, rightpropubs,
            &rightlpbound, &rightlpsolved, &rightlperror, &rightcutoff) );

      if( rightcutoff )
      {
         SCIP_Real newbound;

         SCIP_CALL( calculateBounds(scip, probingvar, NULL, &newbound, NULL, NULL) );

         /* upper bound can be updated */
         SCIPdebugMessage("change probing variable <%s> upper bound from %g to %g, nlocks=(%d/%d)\n",
            SCIPvarGetName(probingvar), SCIPvarGetUbLocal(probingvar), newbound,
            SCIPvarGetNLocksDown(probingvar), SCIPvarGetNLocksUp(probingvar));

         SCIP_CALL( addBdchg(scip, bdchgdata, probingvar, newbound, SCIP_BOUNDTYPE_UPPER, TRUE, nbdchgs, &cutoff) );
      }
   }

   /* set return value of lperror */
   cutoff = cutoff || (leftcutoff && rightcutoff);
   *lperror = leftlperror || rightlperror;


   /* analyze probing deductions */

   /* 1. dualbounds */
   if( leftlpsolved )
      *down = leftlpbound;
   if( rightlpsolved )
      *up = rightlpbound; /*lint !e644*/

   /* 2. update bounds */
   for( j = 0; j < nvars && !cutoff; ++j )
   {
      SCIP_Real newlb;
      SCIP_Real newub;

      if( vars[j] == probingvar )
         continue;

      /* new bounds of the variable is the union of the propagated bounds of the left and right case */
      newlb = MIN(leftproplbs[j], rightproplbs[j]);
      newub = MAX(leftpropubs[j], rightpropubs[j]);

      /* check for fixed variables */
      if( SCIPisFeasEQ(scip, newlb, newub) )
      {
         /* in both probings, variable j is deduced to a fixed value */
         SCIP_CALL( addBdchg(scip, bdchgdata, vars[j], newlb, SCIP_BOUNDTYPE_LOWER, FALSE, nbdchgs, &cutoff) );
         SCIP_CALL( addBdchg(scip, bdchgdata, vars[j], newub, SCIP_BOUNDTYPE_UPPER, FALSE, nbdchgs, &cutoff) );
         continue;
      }
      else
      {
         SCIP_Real oldlb;
         SCIP_Real oldub;

         assert(SCIPvarGetType(vars[j]) == SCIP_VARTYPE_BINARY || SCIPvarGetType(vars[j]) == SCIP_VARTYPE_INTEGER);

         /* check for bound tightenings */
         oldlb = SCIPvarGetLbLocal(vars[j]);
         oldub = SCIPvarGetUbLocal(vars[j]);
         if( SCIPisLbBetter(scip, newlb, oldlb, oldub) )
         {
            /* in both probings, variable j is deduced to be at least newlb: tighten lower bound */
            SCIP_CALL( addBdchg(scip, bdchgdata, vars[j], newlb, SCIP_BOUNDTYPE_LOWER, FALSE, nbdchgs, &cutoff) );
         }
         if( SCIPisUbBetter(scip, newub, oldlb, oldub) && !cutoff )
         {
            /* in both probings, variable j is deduced to be at most newub: tighten upper bound */
            SCIP_CALL( addBdchg(scip, bdchgdata, vars[j], newub, SCIP_BOUNDTYPE_UPPER, FALSE, nbdchgs, &cutoff) );
         }

      }

   } /* end check for deductions */

   /* set correct return values */
   if( down != NULL && leftlpsolved )
      *down = leftlpbound;
   if( up != NULL && rightlpsolved )
      *up = rightlpbound;
   if( downvalid != NULL && leftlpsolved )
      *downvalid = TRUE;
   if( downvalid != NULL && !leftlpsolved )
      *downvalid = FALSE;
   if( upvalid != NULL && rightlpsolved )
      *upvalid = TRUE;
   if( upvalid != NULL && !rightlpsolved )
      *upvalid = FALSE;
   if( downinf != NULL )
      *downinf = leftcutoff;
   if( upinf != NULL )
      *upinf = rightcutoff;

   if( cutoff )
   {
      if( downinf != NULL )
         *downinf = cutoff;
      if( upinf != NULL )
         *upinf = cutoff;
   }

   /* free temporary memory */
   SCIPfreeBufferArray(scip, &rightpropubs);
   SCIPfreeBufferArray(scip, &rightproplbs);
   SCIPfreeBufferArray(scip, &leftpropubs);
   SCIPfreeBufferArray(scip, &leftproplbs);

   /* release variables */
   for( i = 0; i < nvars; ++i )
   {
      SCIP_CALL( SCIPreleaseVar(scip, &vars[i]) );
   }
   SCIPfreeBufferArray(scip, &vars);


   return SCIP_OKAY;
}




/** adds branching candidates to branchruledata to collect infos about it */
static
SCIP_RETCODE addBranchcandsToData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR**            branchcands,        /**< branching candidates */
   int                   nbranchcands        /**< number of branching candidates */
   )
{

   SCIP_BRANCHRULEDATA* branchruledata;
   int j;


   /* get branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   if( branchruledata->nvars == 0 )
   {     /* no variables known before, reinitialized hashmap and variable info storage */

      /* create hash map */
      assert(branchruledata->varhashmap == NULL);
      SCIP_CALL( SCIPhashmapCreate(&(branchruledata->varhashmap), SCIPblkmem(scip), HASHSIZE_VARS) );

      branchruledata->maxvars = SCIPcalcMemGrowSize(scip, nbranchcands);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->nvarprobings, branchruledata->maxvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->nvarbranchings, branchruledata->maxvars) );
      branchruledata->nvars = nbranchcands;

      /* store each variable in hashmap and initialize array entries */
      for( j = 0; j < nbranchcands; ++j )
      {
         SCIP_CALL( SCIPhashmapInsert(branchruledata->varhashmap, branchcands[j], (void*) (size_t)j) );
         branchruledata->nvarprobings[j] = 0;
         branchruledata->nvarbranchings[j] = 0;
      }
   }
   else  /* possibly new variables need to be added */
   {

      /* if var is not in hashmap, insert it */
      for( j = 0; j < nbranchcands; ++j )
      {
         SCIP_VAR* var;
         int nvars;

         var = branchcands[j];
         assert(var != NULL);
         nvars = branchruledata->nvars;

         /* if variable is not in hashmap insert it and increase array sizes */
         if( !SCIPhashmapExists(branchruledata->varhashmap, var) )
         {
            int newsize = SCIPcalcMemGrowSize(scip, nvars + 1);
            SCIP_CALL( SCIPhashmapInsert(branchruledata->varhashmap, var, (void*) (size_t)nvars) );
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &branchruledata->nvarprobings, branchruledata->maxvars,
               newsize) );
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &branchruledata->nvarbranchings, branchruledata->maxvars,
               newsize) );
            branchruledata->maxvars = newsize;

            branchruledata->nvarprobings[nvars] = 0;
            branchruledata->nvarbranchings[nvars] = 0;

            assert(SCIPhashmapExists(branchruledata->varhashmap, var)
               && (int)(size_t) SCIPhashmapGetImage(branchruledata->varhashmap, var) == nvars); /*lint !e507*/

            ++(branchruledata->nvars);
         }

      }
   }

   return SCIP_OKAY;
}

/** increases number of branchings that took place on the given variable */
static
SCIP_RETCODE incNVarBranchings(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR*             var                 /**< variable to increase number of branchings on */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;
   int pos;

   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(var != NULL);

   /* get branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   assert(SCIPhashmapExists(branchruledata->varhashmap, var) );

   pos = (int)(size_t) SCIPhashmapGetImage(branchruledata->varhashmap, var); /*lint !e507*/
   (branchruledata->nvarbranchings[pos])++;

   (branchruledata->nbranchings)++;

   return SCIP_OKAY;
}

/** increases number of probings that took place on the given variable */
static
SCIP_RETCODE incNVarProbings(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR*             var                 /**< variable to increase number of branchings on */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;
   int pos;

   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(var != NULL);

   /* get branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   assert(SCIPhashmapExists(branchruledata->varhashmap, var) );

   pos = (int)(size_t) SCIPhashmapGetImage(branchruledata->varhashmap, var); /*lint !e507*/
   (branchruledata->nvarprobings[pos])++;

   (branchruledata->nprobings)++;

   return SCIP_OKAY;
}


/** execute generalized reliability pseudo cost probing branching */
static
SCIP_RETCODE execRelpsprob(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR**            branchcands,        /**< branching candidates */
   SCIP_Real*            branchcandssol,     /**< solution value for the branching candidates */
   int                   nbranchcands,       /**< number of branching candidates */
   int                   nvars,              /**< number of variables to be watched be bdchgdata */
   SCIP_RESULT*          result,             /**< pointer to the result of the execution */
   SCIP_VAR**            branchvar           /**< pointer to the variable to branch on */
   )
{
   SCIP* masterscip;
   SCIP_BRANCHRULEDATA* branchruledata;
   BDCHGDATA* bdchgdata;
   SCIP_Real lpobjval;
#ifndef NDEBUG
   SCIP_Real cutoffbound;
#endif
   SCIP_Real provedbound;
#ifdef SCIP_DEBUG
   SCIP_Bool bestisstrongbranch = FALSE;
#endif
   int bestcand = -1;

   *result = SCIP_DIDNOTRUN;

   SCIPdebugMessage("execrelpsprob method called\n relpsprob\n relpsprob\n relpsprob\n relpsprob\n relpsprob\n relpsprob\n relpsprob\n relpsprob\n");

   /* get SCIP pointer of master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   /* get branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   /* add all branching candidates into branchruledata if not yet inserted */
   SCIP_CALL( addBranchcandsToData(scip, branchrule, branchcands, nbranchcands) );

   bdchgdata = NULL;
   /* create data structure for bound change infos */
   SCIP_CALL( createBdchgData(scip, &bdchgdata, branchcands, nvars) );
   assert(bdchgdata != NULL);

   /* get current LP objective bound of the local sub problem and global cutoff bound */
   lpobjval = SCIPgetLocalLowerbound(scip);
#ifndef NDEBUG
   cutoffbound = SCIPgetCutoffbound(scip);
#endif
   provedbound = lpobjval;

   if( nbranchcands == 1 )
   {
      /* only one candidate: nothing has to be done */
      bestcand = 0;
   }
   else
   {
      SCIP_Real* initcandscores;
      int* initcands;
      int maxninitcands;
      int nbdchgs;
      SCIP_Real avgconflictscore;
      SCIP_Real avgconflengthscore;
      SCIP_Real avginferencescore;
      SCIP_Real avgcutoffscore;
      SCIP_Real avgpscostscore;
      SCIP_Real bestpsscore;
      SCIP_Real bestsbscore;
      SCIP_Real bestuninitsbscore;
      SCIP_Real bestsbfracscore;
      SCIP_Real bestsbdomainscore;
      int ninfprobings;
      int maxbdchgs;
      int bestpscand;
      int bestsbcand;
      int i;
      int j;
      int c;
      int ninitcands = 0;

      /* get average conflict, inference, and pseudocost scores */
      avgconflictscore = SCIPgetAvgConflictScore(scip);
      avgconflictscore = MAX(avgconflictscore, 0.1);
      avgconflengthscore = SCIPgetAvgConflictlengthScore(scip);
      avgconflengthscore = MAX(avgconflengthscore, 0.1);
      avginferencescore = SCIPgetAvgInferenceScore(scip);
      avginferencescore = MAX(avginferencescore, 0.1);
      avgcutoffscore = SCIPgetAvgCutoffScore(scip);
      avgcutoffscore = MAX(avgcutoffscore, 0.1);
      avgpscostscore = SCIPgetAvgPseudocostScore(scip);
      avgpscostscore = MAX(avgpscostscore, 0.1);

      /* get maximal number of candidates to initialize with strong branching; if the current solutions is not basic,
       * we cannot apply the simplex algorithm and therefore don't initialize any candidates
       */
      maxninitcands = MIN(nbranchcands, branchruledata->initcand);

      if( !SCIPisLPSolBasic(masterscip) )
      {
         maxninitcands = 0;
         SCIPdebugMessage("solution is not basic\n");
      }

      SCIPdebugMessage("maxninitcands = %d\n", maxninitcands);

      /* get buffer for storing the unreliable candidates */
      SCIP_CALL( SCIPallocBufferArray(scip, &initcands, maxninitcands+1) ); /* allocate one additional slot for convenience */
      SCIP_CALL( SCIPallocBufferArray(scip, &initcandscores, maxninitcands+1) );

      /* initialize bound change arrays */
      nbdchgs = 0;
      maxbdchgs = branchruledata->maxbdchgs;

      ninfprobings = 0;


      /* search for the best pseudo cost candidate, while remembering unreliable candidates in a sorted buffer */
      bestpscand = -1;
      bestpsscore = -SCIPinfinity(scip);
      for( c = 0; c < nbranchcands; ++c )
      {
         SCIP_Real conflictscore;
         SCIP_Real conflengthscore;
         SCIP_Real inferencescore;
         SCIP_Real cutoffscore;
         SCIP_Real pscostscore;
         SCIP_Real score;

         assert(branchcands[c] != NULL);

         /* get conflict, inference, cutoff, and pseudo cost scores for candidate */
         conflictscore = SCIPgetVarConflictScore(scip, branchcands[c]);
         conflengthscore = SCIPgetVarConflictlengthScore(scip, branchcands[c]);
         inferencescore = SCIPgetVarAvgInferenceScore(scip, branchcands[c]);
         cutoffscore = SCIPgetVarAvgCutoffScore(scip, branchcands[c]);
         pscostscore = SCIPgetVarPseudocostScore(scip, branchcands[c], branchcandssol[c]);


         /* combine the four score values */
         score = calcScore(scip, branchruledata, conflictscore, avgconflictscore, conflengthscore, avgconflengthscore,
            inferencescore, avginferencescore, cutoffscore, avgcutoffscore, pscostscore, avgpscostscore,
            branchcandssol[c] - SCIPfloor(scip, branchcandssol[c]));

         /* pseudo cost of variable is not reliable: insert candidate in initcands buffer */
         for( j = ninitcands; j > 0 && score > initcandscores[j-1]; --j )
         {
            initcands[j] = initcands[j-1];
            initcandscores[j] = initcandscores[j-1];
         }

         initcands[j] = c;
         initcandscores[j] = score;
         ninitcands++;
         ninitcands = MIN(ninitcands, maxninitcands);
      }

      /* initialize unreliable candidates with probing,
       * search best strong branching candidate
       */
      SCIPdebugMessage("ninitcands = %d\n", ninitcands);

      bestsbcand = -1;
      bestsbscore = -SCIPinfinity(scip);
      bestsbfracscore = -SCIPinfinity(scip);
      bestsbdomainscore = -SCIPinfinity(scip);
      for( i = 0; i < ninitcands; ++i )
      {
         SCIP_Real down;
         SCIP_Real up;
         SCIP_Real downgain;
         SCIP_Real upgain;
         SCIP_Bool downvalid;
         SCIP_Bool upvalid;
         SCIP_Bool lperror;
         SCIP_Bool downinf;
         SCIP_Bool upinf;

         lperror = FALSE;
         up = 0.;
         down = 0.;

         /* get candidate number to initialize */
         c = initcands[i];

         SCIPdebugMessage("init pseudo cost (%g/%g) of <%s> with bounds [%g,%g] at %g (score:%g)\n",
            SCIPgetVarPseudocostCountCurrentRun(scip, branchcands[c], SCIP_BRANCHDIR_DOWNWARDS),
            SCIPgetVarPseudocostCountCurrentRun(scip, branchcands[c], SCIP_BRANCHDIR_UPWARDS),
            SCIPvarGetName(branchcands[c]), SCIPvarGetLbLocal(branchcands[c]), SCIPvarGetUbLocal(branchcands[c]),
            branchcandssol[c], initcandscores[i]);

         /* try branching on this variable (propagation + lp solving (pricing) ) */
         SCIP_CALL( getVarProbingbranch(scip, branchcands[c], bdchgdata, branchruledata->uselp, &branchruledata->nlpiterations,
               &down, &up, &downvalid, &upvalid, &downinf, &upinf, &lperror, &nbdchgs) );

         branchruledata->nprobingnodes++;
         branchruledata->nprobingnodes++;
         SCIP_CALL( incNVarProbings(scip, branchrule, branchcands[c]) );

         /* check for an error in strong branching */
         if( lperror )
         {
            if( !SCIPisStopped(scip) )
            {
               SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL,
                  "(node %"SCIP_LONGINT_FORMAT") error in strong branching call for variable <%s> with solution %g\n",
                  SCIPgetNNodes(scip), SCIPvarGetName(branchcands[c]), branchcandssol[c]);
            }
            break;
         }


         if( SCIPisStopped(scip) )
         {
            break;
         }

         /* check if there are infeasible roundings */
         if( downinf && upinf )
         {
            /* both roundings are infeasible -> node is infeasible */
            SCIPdebugMessage(" -> variable <%s> is infeasible in both directions\n",
               SCIPvarGetName(branchcands[c]));

            *result = SCIP_CUTOFF;
            break; /* terminate initialization loop, because node is infeasible */
         }


         /* evaluate strong branching */
         down = MAX(down, lpobjval);
         up = MAX(up, lpobjval);
         downgain = down - lpobjval;
         upgain = up - lpobjval;
         assert(!downvalid || downinf == SCIPisGE(scip, down, cutoffbound));
         assert(!upvalid || upinf == SCIPisGE(scip, up, cutoffbound));

         /* the minimal lower bound of both children is a proved lower bound of the current subtree */
         if( downvalid && upvalid )
         {
            SCIP_Real minbound;

            minbound = MIN(down, up);
            provedbound = MAX(provedbound, minbound);
         }


         /* terminate initialization loop, if enough roundings are performed */
         if( maxbdchgs >= 0 && nbdchgs >= maxbdchgs )
            break;

         /* case one rounding is infeasible is regarded in method SCIPgetVarProbingbranch */
         if( downinf || upinf )
         {
            branchruledata->ninfprobings++;
            ninfprobings++;
         }

         /* if both roundings are valid, update scores */
         if( !downinf && !upinf )
         {
            SCIP_Real frac;
            SCIP_Real conflictscore;
            SCIP_Real conflengthscore;
            SCIP_Real inferencescore;
            SCIP_Real cutoffscore;
            SCIP_Real pscostscore;
            SCIP_Real score;

            frac = branchcandssol[c] - SCIPfloor(scip, branchcandssol[c]);

            /* check for a better score */
            conflictscore = SCIPgetVarConflictScore(scip, branchcands[c]);
            conflengthscore = SCIPgetVarConflictlengthScore(scip, branchcands[c]);
            inferencescore = SCIPgetVarAvgInferenceScore(scip, branchcands[c]);
            cutoffscore = SCIPgetVarAvgCutoffScore(scip, branchcands[c]);
            pscostscore = SCIPgetBranchScore(scip, branchcands[c], downgain, upgain);
            score = calcScore(scip, branchruledata, conflictscore, avgconflictscore, conflengthscore, avgconflengthscore,
               inferencescore, avginferencescore, cutoffscore, avgcutoffscore, pscostscore, avgpscostscore, frac);

            if( SCIPisSumGE(scip, score, bestsbscore) )
            {
               SCIP_Real fracscore;
               SCIP_Real domainscore;

               fracscore = MIN(frac, 1.0 - frac);
               domainscore = -(SCIPvarGetUbLocal(branchcands[c]) - SCIPvarGetLbLocal(branchcands[c]));
               if( SCIPisSumGT(scip, score, bestsbscore )
                  || SCIPisSumGT(scip, fracscore, bestsbfracscore)
                  || (SCIPisSumGE(scip, fracscore, bestsbfracscore) && domainscore > bestsbdomainscore) )
               {
                  bestsbcand = c;
                  bestsbscore = score;
                  bestsbfracscore = fracscore;
                  bestsbdomainscore = domainscore;
               }
            }

            /* update pseudo cost values */
            assert(!SCIPisFeasNegative(scip, frac));
            SCIP_CALL( SCIPupdateVarPseudocost(scip, branchcands[c], 0.0-frac, downgain, 1.0) );
            SCIP_CALL( SCIPupdateVarPseudocost(scip, branchcands[c], 1.0-frac, upgain, 1.0) );

            SCIPdebugMessage(" -> variable <%s> (solval=%g, down=%g (%+g), up=%g (%+g), score=%g/ %g/%g %g/%g -> %g)\n",
               SCIPvarGetName(branchcands[c]), branchcandssol[c], down, downgain, up, upgain,
               pscostscore, conflictscore, conflengthscore, inferencescore, cutoffscore,  score);

         }
      }
#ifdef SCIP_DEBUG
      if( bestsbcand >= 0 )
      {
         SCIPdebugMessage(" -> best: <%s> (%g / %g / %g)\n",
               SCIPvarGetName(branchcands[bestsbcand]), bestsbscore, bestsbfracscore, bestsbdomainscore);
      }
#endif
      if( bestsbcand >= 0 )
      {
         SCIPdebugMessage(" -> best: <%s> (%g / %g / %g)\n",
            SCIPvarGetName(branchcands[bestsbcand]), bestsbscore, bestsbfracscore, bestsbdomainscore);
      }

      /* get the score of the best uninitialized strong branching candidate */
      if( i < ninitcands )
         bestuninitsbscore = initcandscores[i];
      else
         bestuninitsbscore = -SCIPinfinity(scip);

      /* if the best pseudo cost candidate is better than the best uninitialized strong branching candidate,
       * compare it to the best initialized strong branching candidate
       */
      if( bestpsscore > bestuninitsbscore && SCIPisSumGT(scip, bestpsscore, bestsbscore) )
      {
         bestcand = bestpscand;
#ifdef SCIP_DEBUG
         bestisstrongbranch = FALSE;
#endif
      }
      else if( bestsbcand >= 0 )
      {
         bestcand = bestsbcand;
#ifdef SCIP_DEBUG
         bestisstrongbranch = TRUE;
#endif
      }
      else
      {
         /* no candidate was initialized, and the best score is the one of the first candidate in the initialization
          * queue
          */
         assert(ninitcands >= 1);
         bestcand = initcands[0];
#ifdef SCIP_DEBUG
         bestisstrongbranch = FALSE;
#endif
      }

      /* apply domain reductions */
      if( (nbdchgs >= branchruledata->minbdchgs || ninfprobings >= 5 )
         && *result != SCIP_CUTOFF && !SCIPisStopped(scip) )
      {
         SCIP_CALL( applyBdchgs(scip, bdchgdata, SCIPgetCurrentNode(scip)) );
         branchruledata->nresolvesminbdchgs++;
         *result = SCIP_REDUCEDDOM; /* why was this commented out?? */
      }

      /* free buffer for the unreliable candidates */
      SCIPfreeBufferArray(scip, &initcandscores);
      SCIPfreeBufferArray(scip, &initcands);
   }

   /* if no domain could be reduced, create the branching */
   if( *result != SCIP_CUTOFF && *result != SCIP_REDUCEDDOM
      && *result != SCIP_CONSADDED && !SCIPisStopped(scip) )
   {
      assert(*result == SCIP_DIDNOTRUN);
      assert(0 <= bestcand && bestcand < nbranchcands);
      assert(SCIPisLT(scip, provedbound, cutoffbound));

#ifdef SCIP_DEBUG
      SCIPdebugMessage(" -> best: <%s> (strongbranch = %ud)\n", SCIPvarGetName(branchcands[bestcand]), bestisstrongbranch);
#endif
      *branchvar = branchcands[bestcand];
      SCIP_CALL( incNVarBranchings(scip, branchrule, *branchvar) );
   }

   /* free data structure for bound change infos */
   SCIP_CALL( freeBdchgData(scip, bdchgdata) );

   return SCIP_OKAY;
}


/*
 * Callback methods
 */

/** destructor of branching rule to free user data (called when SCIP is exiting) */
static
SCIP_DECL_BRANCHFREE(branchFreeRelpsprob)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   /* free branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);

   SCIPdebugMessage("**needed in total %d probing nodes\n", branchruledata->nprobingnodes);

   SCIPfreeMemory(scip, &branchruledata);
   SCIPbranchruleSetData(branchrule, NULL);

   return SCIP_OKAY;
}


/** solving process initialization method of branching rule (called when branch and bound process is about to begin) */
static
SCIP_DECL_BRANCHINITSOL(branchInitsolRelpsprob)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   /* free branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);

   branchruledata->nprobingnodes = 0;
   branchruledata->nlpiterations = 0;

   branchruledata->nprobings = 0;
   branchruledata->nbranchings = 0;
   branchruledata->ninfprobings = 0;
   branchruledata->nresolvesminbdchgs = 0;
   branchruledata->nresolvesinfcands = 0;

   branchruledata->varhashmap = NULL;
   branchruledata->nvarbranchings = NULL;
   branchruledata->nvarprobings = NULL;
   branchruledata->nvars = 0;
   branchruledata->maxvars = 0;

   return SCIP_OKAY;
}

/** solving process deinitialization method of branching rule (called before branch and bound process data is freed) */
static
SCIP_DECL_BRANCHEXITSOL(branchExitsolRelpsprob)
{  /*lint --e{715}*/
   SCIP_BRANCHRULEDATA* branchruledata;

   /* free branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);

   SCIPdebugMessage("**in total: nprobings = %d; part of it are ninfprobings = %d\n",
      branchruledata->nprobings, branchruledata->ninfprobings );

   SCIPdebugMessage("**nbranchings = %d, nresolvesinfcands = %d, nresolvesminbdchgs = %d\n",
      branchruledata->nbranchings, branchruledata->nresolvesinfcands, branchruledata->nresolvesminbdchgs );


   /* free arrays for variables & hashmap */
   SCIPfreeBlockMemoryArrayNull(scip, &branchruledata->nvarprobings, branchruledata->maxvars);
   SCIPfreeBlockMemoryArrayNull(scip, &branchruledata->nvarbranchings, branchruledata->maxvars);
   branchruledata->nvars = 0;

   if( branchruledata->varhashmap != NULL )
   {
      SCIPhashmapFree(&(branchruledata->varhashmap));
   }

   return SCIP_OKAY;
}


/*
 * branching specific interface methods
 */

/** creates the reliable pseudo cost braching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleRelpsprob(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP* origscip;
   SCIP_BRANCHRULE* branchrule;
   SCIP_BRANCHRULEDATA* branchruledata;

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* create relpsprob branching rule data */
   SCIP_CALL( SCIPallocMemory(scip, &branchruledata) );

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchruleBasic(scip, &branchrule, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY,
            BRANCHRULE_MAXDEPTH, BRANCHRULE_MAXBOUNDDIST, branchruledata) );
   assert(branchrule != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetBranchruleFree(scip, branchrule, branchFreeRelpsprob) );
   SCIP_CALL( SCIPsetBranchruleInitsol(scip, branchrule, branchInitsolRelpsprob) );
   SCIP_CALL( SCIPsetBranchruleExitsol(scip, branchrule, branchExitsolRelpsprob) );

   /* relpsprob branching rule parameters */
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/conflictweight",
         "weight in score calculations for conflict score",
         &branchruledata->conflictweight, TRUE, DEFAULT_CONFLICTWEIGHT, SCIP_REAL_MIN, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/conflictlengthweight",
         "weight in score calculations for conflict length score",
         &branchruledata->conflengthweight, TRUE, DEFAULT_CONFLENGTHWEIGHT, SCIP_REAL_MIN, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/inferenceweight",
         "weight in score calculations for inference score",
         &branchruledata->inferenceweight, TRUE, DEFAULT_INFERENCEWEIGHT, SCIP_REAL_MIN, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/cutoffweight",
         "weight in score calculations for cutoff score",
         &branchruledata->cutoffweight, TRUE, DEFAULT_CUTOFFWEIGHT, SCIP_REAL_MIN, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/pscostweight",
         "weight in score calculations for pseudo cost score",
         &branchruledata->pscostweight, TRUE, DEFAULT_PSCOSTWEIGHT, SCIP_REAL_MIN, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/minreliable",
         "minimal value for minimum pseudo cost size to regard pseudo cost value as reliable",
         &branchruledata->minreliable, TRUE, DEFAULT_MINRELIABLE, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/maxreliable",
         "maximal value for minimum pseudo cost size to regard pseudo cost value as reliable",
         &branchruledata->maxreliable, TRUE, DEFAULT_MAXRELIABLE, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/iterquot",
         "maximal fraction of branching LP iterations compared to node relaxation LP iterations",
         &branchruledata->iterquot, FALSE, DEFAULT_ITERQUOT, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(origscip,
         "branching/relpsprob/iterofs",
         "additional number of allowed LP iterations",
         &branchruledata->iterofs, FALSE, DEFAULT_ITEROFS, 0, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(origscip,
         "branching/relpsprob/maxlookahead",
         "maximal number of further variables evaluated without better score",
         &branchruledata->maxlookahead, TRUE, DEFAULT_MAXLOOKAHEAD, 1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(origscip,
         "branching/relpsprob/initcand",
         "maximal number of candidates initialized with strong branching per node",
         &branchruledata->initcand, FALSE, DEFAULT_INITCAND, 0, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(origscip,
         "branching/relpsprob/maxbdchgs",
         "maximal number of bound tightenings before the node is immediately reevaluated (-1: unlimited)",
         &branchruledata->maxbdchgs, TRUE, DEFAULT_MAXBDCHGS, -1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(origscip,
         "branching/relpsprob/minbdchgs",
         "minimal number of bound tightenings before bound changes are applied",
         &branchruledata->minbdchgs, TRUE, DEFAULT_MINBDCHGS, 1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(origscip,
         "branching/relpsprob/uselp",
         "shall the LP be solved during probing? (TRUE)",
         &branchruledata->uselp, FALSE, DEFAULT_USELP, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(origscip,
         "branching/relpsprob/reliability",
         "reliability value for probing",
         &branchruledata->reliability, FALSE, DEFAULT_RELIABILITY, 0.0, 1.0, NULL, NULL) );

   /* notify cons_integralorig about the original variable branching rule */
   SCIP_CALL( GCGconsIntegralorigAddBranchrule(scip, branchrule) );

   return SCIP_OKAY;
}

/** execution reliability pseudo cost probing branching with the given branching candidates */
SCIP_RETCODE SCIPgetRelpsprobBranchVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            branchcands,        /**< brancing candidates */
   SCIP_Real*            branchcandssol,     /**< solution value for the branching candidates */
   int                   nbranchcands,       /**< number of branching candidates */
   int                   nvars,              /**< number of variables to be watched by bdchgdata */
   SCIP_RESULT*          result,             /**< pointer to the result of the execution */
   SCIP_VAR**            branchvar           /**< pointer to the variable to branch on */
   )
{
   SCIP_BRANCHRULE* branchrule;
   SCIP* origscip;

   assert(scip != NULL);
   assert(result != NULL);

   /* find branching rule */
   branchrule = SCIPfindBranchrule(scip, BRANCHRULE_NAME);
   assert(branchrule != NULL);
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* execute branching rule */
   SCIP_CALL( execRelpsprob(origscip, branchrule, branchcands, branchcandssol, nbranchcands, nvars, result, branchvar) );

   return SCIP_OKAY;
}
