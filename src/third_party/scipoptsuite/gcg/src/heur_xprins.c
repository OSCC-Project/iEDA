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

/**@file   heur_xprins.c
 * @brief  Extreme Point RINS
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "heur_xprins.h"
#include "gcg.h"

#include "scip/scip.h"
#include "scip/misc.h"
#include "scip/scipdefplugins.h"


#define HEUR_NAME             "xprins"
#define HEUR_DESC             "Extreme Point RINS"
#define HEUR_DISPCHAR         'Y'
#define HEUR_PRIORITY         -1100600
#define HEUR_FREQ             0
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE
#define HEUR_USESSUBSCIP      TRUE

#define DEFAULT_EQUALITYRATE  0.5           /**< minimum percentage of coincidence of relaxation and extreme pts     */
#define DEFAULT_MAXNODES      1000LL        /**< maximum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINIMPROVE    0.01          /**< factor by which xprins should at least improve the incumbent        */
#define DEFAULT_MINNODES      200LL         /**< minimum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINFIXINGRATE 0.4           /**< minimum percentage of integer variables that have to be fixed       */
#define DEFAULT_NODESOFS      200LL         /**< number of nodes added to the contingent of the total nodes          */
#define DEFAULT_NODESQUOT     0.1           /**< subproblem nodes in relation to nodes of the original problem       */
#define DEFAULT_NUSEDPTS      -1            /**< number of extreme pts per block that will be taken into account
                                             * (-1: all; 0: all which contribute to current relaxation solution)
                                             */
#define DEFAULT_RANDOMIZATION FALSE         /**< should the choice which sols to take be randomized?                 */
#define DEFAULT_COPYCUTS      TRUE          /**< if DEFAULT_USELPROWS is FALSE, then should all active cuts from the cutpool
                                             * of the original scip be copied to constraints of the subscip
                                             */
#define DEFAULT_RANDSEED         7           /**< initial random seed                                                  */



/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   SCIP_Real             equalityrate;       /**< minimum percentage of coincidence of relaxation and extreme pts   */
   SCIP_Longint          maxnodes;           /**< maximum number of nodes to regard in the subproblem               */
   SCIP_Longint          minnodes;           /**< minimum number of nodes to regard in the subproblem               */
   SCIP_Longint          nodesofs;           /**< number of nodes added to the contingent of the total nodes        */
   SCIP_Longint          usednodes;          /**< nodes already used by xprins in earlier calls                     */
   SCIP_Real             nodesquot;          /**< subproblem nodes in relation to nodes of the original problem     */
   int                   nusedpts;           /**< number of extreme pts per block that will be taken into account
                                              *   (-1: all; 0: all which contribute to current relaxation solution)
                                              */
   unsigned int          nfailures;          /**< number of failures since last successful call                     */
   SCIP_Longint          nextnodenumber;     /**< number of BnB nodes at which crossover should be called next      */
   SCIP_Real             minfixingrate;      /**< minimum percentage of integer variables that have to be fixed     */
   SCIP_Real             minimprove;         /**< factor by which xprins should at least improve the incumbent      */
   SCIP_Bool             randomization;      /**< should the choice which sols to take be randomized?               */
   SCIP_Bool             copycuts;           /**< if uselprows == FALSE, should all active cuts from cutpool be copied
                                              *   to constraints in subproblem?
                                              */
   SCIP_RANDNUMGEN*      randnumgen;         /**< random number generator                                           */

#ifdef SCIP_STATISTIC
   SCIP_Longint          nfixfails;          /**< number of abortions due to a bad fixing rate                      */
   SCIP_Real             avgfixrate;         /**< average rate of variables that are fixed                          */
   SCIP_Real             avgzerorate;        /**< average rate of fixed variables that are zero                     */
   SCIP_Longint          totalsols;          /**< total number of subSCIP solutions (including those which have not
                                              *   been added)
                                              */
   SCIP_Real             subsciptime;        /**< total subSCIP solving time in seconds                             */
   SCIP_Real             bestprimalbd;       /**< objective value of best solution found by this heuristic          */
#endif
};




/*
 * Local methods
 */


/** for each block, select extreme points (represented by mastervars) to be compared to the relaxation solution */
static
SCIP_RETCODE selectExtremePoints(
   SCIP*                 scip,               /**< original SCIP data structure                                    */
   SCIP_HEURDATA*        heurdata,           /**< primal heuristic data                                           */
   int*                  selection,          /**< indices of selected extreme points                              */
   int*                  nactualpts,         /**< number of points per block that have actually been selected     */
   SCIP_Bool*            success             /**< pointer to store whether the process was successful             */
   )
{
   SCIP* masterprob;
   int nblocks;

   SCIP_VAR** mastervars;
   SCIP_Real* mastervals;
   int nmastervars;

   int nusedpts;
   int block;
#ifndef NDEBUG
   int nidentblocks;
#endif
   int* blocknrs;
   int* identblock;
   SCIP_Real* blockvalue;
   SCIP_Real value;
   SCIP_Real* selvalue;

   int i;
   int j;

   /* check preconditions */
   assert(scip != NULL);
   assert(heurdata != NULL);
   assert(selection != NULL);
   assert(success != NULL);

   assert(heurdata->nusedpts >= 0);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get number of blocks */
   nblocks = GCGgetNPricingprobs(scip);

   /* get variables of the master problem */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* get master LP solution values */
   SCIP_CALL( SCIPallocBufferArray(scip, &mastervals, nmastervars) );
   SCIP_CALL( SCIPgetSolVals(masterprob, NULL, nmastervars, mastervars, mastervals) );

   /* get number of extreme points per block */
   nusedpts = heurdata->nusedpts;

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &selvalue, nblocks * nusedpts) );
   SCIP_CALL( SCIPallocBufferArray(scip, &blocknrs, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &blockvalue, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &identblock, nblocks) );

   /* initialize the block values for the pricing problems */
   for( i = 0; i < nblocks; i++ )
   {
      blockvalue[i] = 0.0;
      blocknrs[i] = 0;
      identblock[i] = i;
      nactualpts[i] = 0;
   }

   *success = FALSE;

   /* loop over all given master variables;
    * this loop treats master variables that have value one or greater
    * (in particular important if blocks are represented by others)
    */
   for( i = 0; i < nmastervars; i++ )
   {
      SCIP_VAR* mastervar;

      mastervar = mastervars[i];
      assert(GCGvarIsMaster(mastervar));

      /* get block information and solution value */
      block = GCGvarGetBlock(mastervar);
      value = SCIPgetSolVal(masterprob, NULL, mastervar);

      /** @todo handle infinite master solution values */
      assert(!SCIPisInfinity(scip, value));

      /* ignore irrelevant extreme points */
      if( SCIPisFeasZero(scip, value) )
         continue;

      /* ignore rays
       * @todo do it smarter */
      if( GCGmasterVarIsRay(mastervar) )
         continue;

      /* variables belonging to no block are not treated here */
      if( block == -1 )
         continue;

      /* get number of blocks that are identical to this block */
      assert(block >= 0);
#ifndef NDEBUG
      nidentblocks = GCGgetNIdenticalBlocks(scip, block);
#endif

      while( SCIPisFeasGE(scip, mastervals[i], 1.0) )
      {
         /* insert the extreme point in the selection (should be the only point for this block) */
         j = identblock[block] * nusedpts;
         assert(selection[j] == -1);
         assert(nactualpts[identblock[block]] == 0);

         nactualpts[identblock[block]] = 1;
         selection[j] = i;
         selvalue[j] = 1.0;

         mastervals[i] = mastervals[i] - 1.0;
         blocknrs[block]++;

         /* search the next block to be considered */
         for( j = identblock[block] + 1; j < nblocks; ++j )
            if( GCGgetBlockRepresentative(scip, j) == block )
            {
               identblock[block] = j;
               break;
            }

#ifndef NDEBUG
         assert(blocknrs[block] >= nidentblocks || j < nblocks);
#endif
      }
   }

   /* loop over all given master variables */
   for( i = 0; i < nmastervars; i++ )
   {
      SCIP_VAR* mastervar;

      mastervar = mastervars[i];
      assert(GCGvarIsMaster(mastervar));

      /* get block information and solution value */
      block = GCGvarGetBlock(mastervar);
      value = SCIPgetSolVal(masterprob, NULL, mastervar);

      /** @todo handle infinite master solution values */
      assert(!SCIPisInfinity(scip, value));

      /* ignore irrelevant extreme points */
      if( SCIPisFeasZero(scip, value) )
         continue;

      /* ignore rays */
      if( GCGmasterVarIsRay(mastervar) )
         continue;

      /* variables belonging to no block are not treated here */
      if( block == -1 )
         continue;

      /* get number of blocks that are identical to this block */
      assert(block >= 0);
#ifndef NDEBUG
      nidentblocks = GCGgetNIdenticalBlocks(scip, block);
#endif

      assert(SCIPisFeasGE(scip, mastervals[i], 0.0) && SCIPisFeasLT(scip, mastervals[i], 1.0));

      while( SCIPisFeasPositive(scip, mastervals[i]) )
      {
         value = MIN(mastervals[i], 1.0 - blockvalue[block]);

         /* check if the extreme point is good enough to be inserted in the selection
          * by looking for a position where it may be inserted
          */
         for( j = (identblock[block] * nusedpts) + nactualpts[identblock[block]];
            j > identblock[block] * nusedpts && SCIPisGT(scip, value, selvalue[j]); --j )
         {
            if( j < (identblock[block] + 1) * nusedpts )
            {
               selection[j] = selection[j-1];
               selvalue[j] = selvalue[j-1];
            }
         }
         if( j < (identblock[block] * nusedpts) + nusedpts )
         {
            selection[j] = i;
            selvalue[j] = value;

            if( nactualpts[identblock[block]] < nusedpts )
               ++nactualpts[identblock[block]];
         }

         mastervals[i] = mastervals[i] - value;
         if( SCIPisFeasZero(scip, mastervals[i]) )
            mastervals[i] = 0.0;
         blockvalue[block] += value;

         /* if the value assigned to the block is equal to 1, this block is full and we consider the next block */
         if( SCIPisFeasGE(scip, blockvalue[block], 1.0) )
         {
            blockvalue[block] = 0.0;
            blocknrs[block]++;

            /* search the next identical block to be considered */
            for( j = identblock[block] + 1; j < nblocks; ++j )
               if( GCGgetBlockRepresentative(scip, j) == block )
               {
                  identblock[block] = j;
                  break;
               }

#ifndef NDEBUG
            assert(blocknrs[block] >= nidentblocks || j < nblocks);
#endif
         }
      }
   }

   *success = TRUE;

   /* free memory */
   SCIPfreeBufferArray(scip, &identblock);
   SCIPfreeBufferArray(scip, &blockvalue);
   SCIPfreeBufferArray(scip, &blocknrs);
   SCIPfreeBufferArray(scip, &selvalue);
   SCIPfreeBufferArray(scip, &mastervals);

   return SCIP_OKAY;
}


/** select extreme points (represented by mastervars) randomly */
static
SCIP_RETCODE selectExtremePointsRandomized(
   SCIP*                 scip,               /**< original SCIP data structure                                    */
   SCIP_HEURDATA*        heurdata,           /**< primal heuristic data                                           */
   int*                  selection,          /**< indices of selected extreme points                              */
   int*                  nactualpts,         /**< number of points per block that have actually been selected     */
   SCIP_Bool*            success             /**< pointer to store whether the process was successful             */
   )
{
   SCIP* masterprob;
   int nblocks;

   SCIP_VAR** mastervars;
   int nmastervars;

   int nusedpts;         /* number of extreme points per block to be chosen        */
   int* npts;            /* for each block, the number of available extreme points */
   int* blockpts;        /* all points of a block which to be considered           */
   SCIP_Real* ptvals;    /* solution values of extreme points in master problem    */

   int i;
   int j;

   /* check preconditions */
   assert(scip != NULL);
   assert(heurdata != NULL);
   assert(selection != NULL);
   assert(success != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* get number of blocks */
   nblocks = GCGgetNPricingprobs(scip);

   /* get variables of the master problem */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* get number of extreme points per block */
   nusedpts = heurdata->nusedpts;

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &npts, nblocks) );

   *success = TRUE;

   /* check whether we have enough points per block to perform a randomization */
   for( i = 0; i < nblocks; ++i )
      npts[i] = 0;
   for( i = 0; i < nmastervars; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_Real solval;
      int block;

      mastervar = mastervars[i];
      solval = SCIPgetSolVal(masterprob, NULL, mastervar);
      block = GCGvarGetBlock(mastervar);

      if( block >= 0 && !SCIPisFeasZero(scip, solval) )
         ++npts[block];
   }
   for( i = 0; i < nblocks; ++i )
      if( GCGisPricingprobRelevant(scip, i) && npts[i] <= nusedpts )
               *success = FALSE;

   /* do not randomize if there are not enough points available */
   if( !*success )
   {
      SCIPdebugMessage(" -> not enough extreme points available for randomization.\n");

      /* free memory */
      SCIPfreeBufferArray(scip, &npts);

      return SCIP_OKAY;
   }

   *success = FALSE;

   /* perform randomization: for each block, select a set of extreme points to be considered */
   for( i = 0; i < nblocks; ++i )
   {
      int blockrep;
      int lastpt;           /* the worst extreme point possible to choose             */

      int k;


      SCIP_CALL( SCIPallocBufferArray(scip, &blockpts, npts[i]) );
      SCIP_CALL( SCIPallocBufferArray(scip, &ptvals, npts[i]) );

      /* get representative of this block */
      blockrep = GCGgetBlockRepresentative(scip, i);
      assert(blockrep >= 0 && blockrep <= i);

      /* get all relevant extreme points for this block */
      k = 0;
      for( j = 0; j < nmastervars; ++j )
      {
         SCIP_VAR* mastervar;
         SCIP_Real solval;
         int block;

         mastervar = mastervars[j];
         solval = SCIPgetSolVal(masterprob, NULL, mastervar);
         block = GCGvarGetBlock(mastervar);

         if( block == blockrep && !SCIPisFeasZero(scip, solval) )
         {
            assert(k < npts[blockrep]);
            blockpts[k] = j;
            ++k;
         }
      }
      assert(k == npts[blockrep]);

      /* sort the extreme points */
      SCIPsortRealInt(ptvals, blockpts, npts[blockrep]);
      lastpt = npts[blockrep];

      /* perform a random selection for this block */
      for( k = 0; k < nusedpts; ++k )
      {
         int idx;
         int selidx;

         idx = SCIPrandomGetInt(heurdata->randnumgen, nusedpts-k-1, lastpt-1);
         selidx = i * nusedpts + k;
         selection[selidx] = blockpts[idx];
         lastpt = idx;
      }

      nactualpts[i] = nusedpts;

      SCIPfreeBufferArray(scip, &ptvals);
      SCIPfreeBufferArray(scip, &blockpts);
   }

   *success = TRUE;

   /* free memory */
   SCIPfreeBufferArray(scip, &npts);

   return SCIP_OKAY;
}

/*
 * count extreme points per block to be considered;
 * this is only done when no selection of extreme points has been made!
 */
static
SCIP_RETCODE countExtremePoints(
   SCIP*                 scip,               /**< original SCIP data structure                                     */
   int*                  selection,          /**< selected extreme points the heuristic will use, or NULL          */
   int                   nusedpts,           /**< number of extreme points per block to be considered, or 0, or -1 */
   int*                  nactualpts          /**< number of points per block that have actually been selected      */
   )
{
   SCIP* masterprob;                         /* master problem                         */
   SCIP_VAR** mastervars;                    /* master variables                       */
   int nmastervars;                          /* number of master variables             */
   int nblocks;

   int i;

   assert(nusedpts == 0 || nusedpts == -1);
   assert(selection == NULL);

   /* get master problem and its variables */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   nblocks = GCGgetNPricingprobs(scip);

   for( i = 0; i < nblocks; ++i )
      nactualpts[i] = 0;
   for( i = 0; i < nmastervars; ++i )
   {
      SCIP_VAR* mastervar;
      int block;

      mastervar = mastervars[i];
      assert(mastervar != NULL);
      block = GCGvarGetBlock(mastervar);
      if( block >= 0 && (nusedpts == -1 || !SCIPisZero(scip, SCIPgetSolVal(masterprob, NULL, mastervar))) )
         ++nactualpts[block];
   }
   /* We have only counted for the relevant blocks so far,
    * so it is still necessary to set the numbers of extreme points for the other blocks
    */
   for( i = 0; i < nblocks; ++i )
   {
      int blockrep = GCGgetBlockRepresentative(scip, i);
      assert(blockrep >= 0 && blockrep <= i);

      nactualpts[i] = nactualpts[blockrep];
   }

   return SCIP_OKAY;
}



/** initialize the subSCIP instance: copy SCIP to subSCIP, set the parameters */
static
SCIP_RETCODE setupSubproblem(
   SCIP*                 scip,               /**< original SCIP data structure                                  */
   SCIP*                 subscip,            /**< SCIP data structure for the subproblem                        */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                               */
   SCIP_HEURDATA*        heurdata,           /**< primal heuristic data                                         */
   SCIP_Longint          nstallnodes,        /**< node limit for subproblem                                     */
   SCIP_Real             timelimit,          /**< time limit for subproblem                                     */
   SCIP_Real             memorylimit         /**< memory limit for subproblem                                   */
   )
{
   SCIP_VAR** vars;
   int nvars;

   int i;

   char probname[SCIP_MAXSTRLEN];
   SCIP_HASHMAP* varmapfw;                   /* mapping of SCIP variables to subSCIP variables      */
   SCIP_Bool valid;

   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPhashmapCreate(&varmapfw, SCIPblkmem(subscip), nvars) );

   /* copy the SCIP instance to the subSCIP */

   /* copy all plugins */
   SCIP_CALL( SCIPincludeDefaultPlugins(subscip) );

   /* get name of the original problem and add the string "_extremeptsub" */
   (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_extremeptsub", SCIPgetProbName(scip));

   /* create the subproblem */
   SCIP_CALL( SCIPcreateProb(subscip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

   /* copy all variables */
   SCIP_CALL( SCIPcopyVars(scip, subscip, varmapfw, NULL, NULL, NULL, 0, TRUE) );

   /* copy all constraints */
   valid = FALSE;
   SCIP_CALL( SCIPcopyConss(scip, subscip, varmapfw, NULL, TRUE, FALSE, &valid) );
   if( heurdata->copycuts )
   {
      /* copies all active cuts from cutpool of sourcescip to linear constraints in targetscip */
      SCIP_CALL( SCIPcopyCuts(scip, subscip, varmapfw, NULL, TRUE, NULL) );
   }
   SCIPdebugMessage("Copying the SCIP constraints was %s complete.\n", valid ? "" : "not ");

   /* get the subproblem variables */
   for( i = 0; i < nvars; i++ )
      subvars[i] = (SCIP_VAR*) SCIPhashmapGetImage(varmapfw, vars[i]);

   /* free hash map */
   SCIPhashmapFree(&varmapfw);

   /* setup parameters of subSCIP */
   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );

#ifdef SCIP_DEBUG
   /* for debugging RENS, enable MIP output */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 5) );
   SCIP_CALL( SCIPsetIntParam(subscip, "display/freq", 100000000) );
#else
   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );
#endif

   /* set limits for the subproblem */
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", nstallnodes) );
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/stallnodes", MAX(10, nstallnodes/10)) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   /* forbid recursive call of heuristics and separators solving subMIPs */
   SCIP_CALL( SCIPsetSubscipsOff(subscip, TRUE) );

   /* disable cutting plane separation */
   SCIP_CALL( SCIPsetSeparating(subscip, SCIP_PARAMSETTING_OFF, TRUE) );

   /* disable expensive presolving */
   SCIP_CALL( SCIPsetPresolving(subscip, SCIP_PARAMSETTING_FAST, TRUE) );

   /* use best estimate node selection */
   if( SCIPfindNodesel(subscip, "estimate") != NULL && !SCIPisParamFixed(subscip, "nodeselection/estimate/stdpriority") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "nodeselection/estimate/stdpriority", INT_MAX/4) );
   }

   /* use inference branching */
   if( SCIPfindBranchrule(subscip, "inference") != NULL && !SCIPisParamFixed(subscip, "branching/inference/priority") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "branching/inference/priority", INT_MAX/4) );
   }

   /* disable conflict analysis */
   if( !SCIPisParamFixed(subscip, "conflict/enable") )
   {
      SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/enable", FALSE) );
   }

   /* if there is already a solution, add an objective cutoff */
   if( SCIPgetNSols(scip) > 0 )
   {
      SCIP_Real cutoff;                         /* objective cutoff for the subproblem                 */
      SCIP_Real upperbound;

      assert( !SCIPisInfinity(scip,SCIPgetUpperbound(scip)) );

      upperbound = SCIPgetUpperbound(scip) - SCIPsumepsilon(scip);
      if( !SCIPisInfinity(scip,-1.0*SCIPgetLowerbound(scip)) )
      {
         cutoff = (1-heurdata->minimprove)*SCIPgetUpperbound(scip) + heurdata->minimprove*SCIPgetLowerbound(scip);
      }
      else
      {
         if( SCIPgetUpperbound ( scip ) >= 0 )
            cutoff = ( 1 - heurdata->minimprove ) * SCIPgetUpperbound ( scip );
         else
            cutoff = ( 1 + heurdata->minimprove ) * SCIPgetUpperbound ( scip );
      }
      cutoff = MIN(upperbound, cutoff );
      SCIP_CALL( SCIPsetObjlimit(subscip, cutoff) );
   }

   return SCIP_OKAY;
}


/**
 * compare an extreme point (represented by a master variable) to the relaxation solution
 */
static
void compareOneExtremePoint(
   SCIP*                 scip,               /**< original SCIP data structure                                  */
   SCIP_VAR*             mastervar,          /**< master variable representing the extreme point                */
   int                   solblock,           /**< block in which the relaxation solution should be compared, or -1 if all blocks of the extreme point should be considered */
   int*                  neqpts,             /**< for each original variable, count how many extreme points share its relaxation solution value */
   SCIP_Bool*            zeroblocks          /**< for each block, the information whether it would be fixed entirely to zero */
   )
{
   int block;                                /* representative block the master variable belongs to   */
   int nblocks;
   SCIP_VAR** origvars;                      /* original variables of the extreme point               */
   SCIP_Real* origvals;                      /* values of the original variables in the extreme point */
   int norigvars;                            /* number of original variables of the extreme point     */

   int i;
   int j;
   int k;

   assert(GCGvarIsMaster(mastervar));

   block = GCGvarGetBlock(mastervar);
   assert(block >= 0);

   nblocks = GCGgetNPricingprobs(scip);

   /* get the actual extreme point */
   origvars = GCGmasterVarGetOrigvars(mastervar);
   origvals = GCGmasterVarGetOrigvals(mastervar);
   norigvars = GCGmasterVarGetNOrigvars(mastervar);

   /* compare each extreme point value to the corresponding relaxation solution value */
   for( i = 0; i < norigvars; ++i )
   {
      SCIP_VAR* pricingvar;
      SCIP_VAR** pricingorigvars;
      int npricingorigvars;

      if( SCIPvarGetType(origvars[i]) > SCIP_VARTYPE_INTEGER )
         continue;

      /* get the corresponding pricing variable;
       * needed to obtain original variables corresponding to the current one from other blocks
       * (see below)
       */
      if( GCGoriginalVarIsLinking(origvars[i]) )
      {
         SCIP_VAR** linkingpricingvars = GCGlinkingVarGetPricingVars(origvars[i]);
         pricingvar = linkingpricingvars[block];
      }
      else
         pricingvar = GCGoriginalVarGetPricingVar(origvars[i]);

      assert(pricingvar != NULL);
      assert(GCGvarIsPricing(pricingvar));

      /* get all original variables corresponding to the current one;
       * this is necessary since in case of identical, aggregated blocks,
       * the extreme point may belong to multiple blocks
       */
      pricingorigvars = GCGpricingVarGetOrigvars(pricingvar);
      npricingorigvars = GCGpricingVarGetNOrigvars(pricingvar);
      assert(pricingorigvars != NULL);
      assert(npricingorigvars >= 0);

      for( j = 0; j < npricingorigvars; ++j )
      {
         int origblock;
         int idx;
         SCIP_Real solval;

         origblock = GCGvarGetBlock(pricingorigvars[j]);
         assert(origblock != -1);

         if( solblock != -1 &&
            ((origblock != -2 && origblock != solblock) || (origblock == -2 && !GCGisLinkingVarInBlock(pricingorigvars[j], solblock))) )
            continue;

         idx = SCIPvarGetProbindex(pricingorigvars[j]);
         assert(SCIPvarGetType(pricingorigvars[j]) <= SCIP_VARTYPE_INTEGER);
         solval = SCIPgetRelaxSolVal(scip, pricingorigvars[j]);

         /* If a relaxation solution value is zero, we assumed that it is zero in all extreme points,
          * so we need to decrease the counter if this is not the case
          */
         if( SCIPisZero(scip, solval) )
         {
            if( !SCIPisZero(scip, origvals[i]) )
               --neqpts[idx];
         }
         else
         {
            if( SCIPisEQ(scip, solval, origvals[i]) )
               ++neqpts[idx];

            /* The block will not be entirely fixed to zero, since the variable has nonzero relaxation solution value */
            if( origblock != -2 )
               zeroblocks[origblock] = FALSE;
            else
            {
               /* For a linking variable, get all blocks it appears in */
               SCIP_VAR** linkingpricingvars = GCGlinkingVarGetPricingVars(pricingorigvars[j]);
               for( k = 0; k < nblocks; ++k )
                  if( linkingpricingvars[k] != NULL )
                     zeroblocks[k] = FALSE;
            }
         }
      }
   }
}

/**
 * compare all selected extreme points to the relaxation solution
 */
static
SCIP_RETCODE compareExtremePointsToRelaxSol(
   SCIP*                 scip,               /**< original SCIP data structure                                     */
   int*                  selection,          /**< selected extreme points the heuristic will use, or NULL          */
   int                   nusedpts,           /**< number of extreme points per block to be considered, or 0, or -1 */
   int*                  neqpts,             /**< for each original variable, count how many extreme points share its relaxation solution value */
   SCIP_Bool*            zeroblocks          /**< for each block, the information whether it would be fixed entirely to zero */
   )
{
   SCIP* masterprob;                         /* master problem                         */
   SCIP_VAR** mastervars;                    /* master variables                       */
   int nmastervars;                          /* number of master variables             */
   int nblocks;

   int i;
   int j;

   /* get master problem and its variables */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   nblocks = GCGgetNPricingprobs(scip);

   if( nusedpts <= 0 )
   {
      assert(nusedpts == 0 || nusedpts == -1);

      for( i = 0; i < nmastervars; ++i )
      {
         SCIP_VAR* mastervar;

         /* get master variable */
         mastervar = mastervars[i];

         /* ignore master variables which are just copies from original variables and no extreme points */
         if( GCGvarGetBlock(mastervar) == -1 )
            continue;

         /* ignore master variable if it is zero and only the nonzeroes should be considered */
         if( nusedpts == 0 && SCIPisZero(scip, SCIPgetSolVal(masterprob, NULL, mastervar)) )
            continue;

         compareOneExtremePoint(scip, mastervar, -1, neqpts, zeroblocks);
      }
   }
   else
   {
      assert(selection != NULL);

      for( i = 0; i < nblocks; ++i )
      {
         /* compare the relaxation solution to the selected extreme points */
         for( j = 0; j < nusedpts; ++j )
         {
            int selidx = i * nusedpts + j;
            if( selection[selidx] != -1 )
            {
               SCIP_VAR* mastervar;

               /* get master variable */
               mastervar = mastervars[selection[selidx]];
               assert(mastervar != NULL);
               assert(GCGvarGetBlock(mastervar) == GCGgetBlockRepresentative(scip, i));

               compareOneExtremePoint(scip, mastervar, i, neqpts, zeroblocks);
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** fix variables; for each variable, we evaluate the percentage of extreme points in which it has the same value
 *  as in the relaxation solution and fix it if the percentage exceeds a certain value
 */
static
SCIP_RETCODE fixVariables(
   SCIP*                 scip,               /**< original SCIP data structure                                  */
   SCIP*                 subscip,            /**< SCIP data structure for the subproblem                        */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                               */
   int*                  selection,          /**< selected extreme points the heuristic will use, or NULL       */
   int*                  nactualpts,         /**< number of points per block that have actually been selected   */
   SCIP_HEURDATA*        heurdata,           /**< primal heuristic data                                         */
   SCIP_Real*            intfixingrate,      /**< percentage of integers that get actually fixed                */
   SCIP_Real*            zerofixingrate,     /**< percentage of variables fixed to zero                         */
   SCIP_Bool*            success             /**< pointer to store whether the problem was created successfully */
   )
{
   SCIP_VAR** vars;                          /* original scip variables                     */

   int nblocks;                              /* number of blocks                                   */
   int nbinvars;                             /* number of binary variables in the original problem */
   int nintvars;                             /* number of general integer variables                */

   int* neqpts;                              /* for each original variable, count the number of
						                              points where it has the same value as the relaxation solution */
   SCIP_Bool* zeroblocks;                    /* blocks that would be entirely fixed to zero                   */
   int fixingcounter;                        /* count how many original variables are fixed                   */
   int zerocounter;                          /* count how many variables are fixed to zero                    */

   int i;
   int j;

   /* check preconditions */
   assert(scip != NULL);
   assert(subscip != NULL);
   assert(subvars != NULL);
   assert(heurdata != NULL);
   assert(selection != NULL || heurdata->nusedpts < 0);
   assert(nactualpts != NULL);
   assert(intfixingrate != NULL);
   assert(zerofixingrate != NULL);
   assert(success != NULL);

   /* get required data of the original problem */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, NULL, &nbinvars, &nintvars, NULL, NULL) );

   nblocks = GCGgetNPricingprobs(scip);
   fixingcounter = 0;
   zerocounter = 0;

   *success = FALSE;

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &neqpts, nbinvars + nintvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &zeroblocks, nblocks) );

   for( i = 0; i < nblocks; ++i )
      zeroblocks[i] = TRUE;

   /* initialize counters for identical solution values
    *
    * @note: Zero values of extreme points are not stored explicitly in the master
    * variable data; therefore, we assume for each variable with solution value zero
    * that each extreme point also has value zero, and we will later on decrease the
    * counter for each point where this is not the case
    */
   for( i = 0; i < nbinvars + nintvars; ++i )
   {
      SCIP_VAR* var;
      int block;
      SCIP_Real solval;

      /* get variable, block and relaxation value */
      var = vars[i];
      assert(var != NULL);
      block = GCGvarGetBlock(var);
      solval = SCIPgetRelaxSolVal(scip, var);

      if( !SCIPisZero(scip, solval) || block == -1 )
         neqpts[i] = 0;
      else if( block == -2 )
      {
         SCIP_VAR** linkingpricingvars = GCGlinkingVarGetPricingVars(var);
         neqpts[i] = 0;
         for( j = 0; j < nblocks; ++j )
            if( linkingpricingvars[j] != NULL )
               neqpts[i] += nactualpts[j];
      }
      else
      {
         assert(block >= 0);
         neqpts[i] = nactualpts[block];
      }
   }

   SCIP_CALL( compareExtremePointsToRelaxSol(scip, selection, heurdata->nusedpts, neqpts, zeroblocks) );

   /* try to fix the binary and general integer variables */
   for( i = 0; i < nbinvars + nintvars; ++i )
   {
      SCIP_VAR* var;
      int block;                             /* current block we are working in                    */
      SCIP_Real solval;

      var = vars[i];
      assert(GCGvarIsOriginal(var));
      block = GCGvarGetBlock(var);
      solval = SCIPgetRelaxSolVal(scip, var);

      /* Variables which were directly copied from the original problem do not appear in any extreme point;
       * they are fixed like in the RENS heuristic
       */
      if( block == -1 )
      {
         if( SCIPisFeasIntegral(scip, solval) )
         {
            /* fix variable to current relaxation solution if it is integral;
             * use exact integral value, if the variable is only integral within numerical tolerances
             */
            solval = SCIPfloor(scip, solval + 0.5);
            SCIP_CALL( SCIPchgVarLbGlobal(subscip, subvars[i], solval) );
            SCIP_CALL( SCIPchgVarUbGlobal(subscip, subvars[i], solval) );

            fixingcounter++;
            if( SCIPisZero(scip, solval) )
               zerocounter++;
         }
      }
      /* For variables belonging to one or more blocks, we evaluate in how many percent of the
       * extreme points they have the same value as in the relaxation solution
       */
      else
      {
         int ntotalpts;
         SCIP_Real quoteqpts;

         assert(block == -2 || block >= 0);

         /* Calculate in how many extreme points the variable appears;
          * in case of linking variables, we need to consider points from all their blocks
          */
         if( block >= 0 )
            ntotalpts = nactualpts[block];
         else
         {
            SCIP_VAR** linkingpricingvars;

            assert(GCGoriginalVarIsLinking(var));
            linkingpricingvars = GCGlinkingVarGetPricingVars(var);

            ntotalpts = 0;
            for( j = 0; j < nblocks; ++j )
               if( linkingpricingvars[j] != NULL )
                  ntotalpts += nactualpts[j];
         }

         assert(neqpts[i] <= ntotalpts);
         quoteqpts = (SCIP_Real) neqpts[i] / (SCIP_Real) MAX(ntotalpts,1);

         SCIPdebugMessage("Variable %s: %d/%d (%.2f percent) extreme points identical to relaxation solution (value=%g).\n",
                     SCIPvarGetName(var), neqpts[i], ntotalpts, quoteqpts * 100, solval);

         /* The variable can be fixed if the relaxation value is shared by enough extreme points;
          * besides, we avoid fixing entire blocks to zero
          */
         if( quoteqpts >= heurdata->equalityrate && (block < 0 || !zeroblocks[block]) )
         {
            SCIP_CALL( SCIPchgVarLbGlobal(subscip, subvars[i], solval) );
            SCIP_CALL( SCIPchgVarUbGlobal(subscip, subvars[i], solval) );

            fixingcounter++;
            if( SCIPisZero(scip, solval) )
               zerocounter++;
         }
      }
   }

   *intfixingrate = (SCIP_Real) fixingcounter / (SCIP_Real) (MAX(nbinvars + nintvars, 1));
   *zerofixingrate = (SCIP_Real)zerocounter / MAX((SCIP_Real)fixingcounter, 1.0);

   /* If not enough variables were fixed, try to fix blocks which relaxation solution value zero,
    * until the minimum fixing rate is reached
    */
   while( *intfixingrate < heurdata->minfixingrate )
   {
      SCIPdebugMessage("  fixing rate only %5.2f --> trying to fix a zero block\n", *intfixingrate);

      /* get the next zero block */
      for( i = 0; i < nblocks; ++i )
         if( zeroblocks[i] )
         {
            /* fix variables */
            for( j = 0; j < nbinvars + nintvars; ++j )
               if( GCGvarGetBlock(vars[j]) == i )
               {
                  SCIP_Real quoteqpts;

                  /* evaluate percentage of extreme points having the same variable value as the relaxation solution */
                  assert(SCIPisZero(scip, SCIPgetRelaxSolVal(scip, vars[j])));
                  assert(neqpts[j] <= nactualpts[i]);
                  quoteqpts = (SCIP_Real) neqpts[j] / (SCIP_Real) MAX(nactualpts[i],1);

                  if( quoteqpts >= heurdata->equalityrate )
                  {
                     SCIP_CALL( SCIPchgVarLbGlobal(subscip, subvars[j], 0.0) );
                     SCIP_CALL( SCIPchgVarUbGlobal(subscip, subvars[j], 0.0) );

                     fixingcounter++;
                     zerocounter++;
                  }
               }

            zeroblocks[i] = FALSE;
            break;
         }

      *intfixingrate = (SCIP_Real)fixingcounter / (SCIP_Real)(MAX(nbinvars + nintvars, 1));
      *zerofixingrate = (SCIP_Real)zerocounter / MAX((SCIP_Real)fixingcounter, 1.0);

      if( i == nblocks )
         break;
   }

   /* if all variables were fixed or amount of fixed variables is insufficient, abort immediately */
   if( *intfixingrate < heurdata->minfixingrate )
   {
      SCIPstatisticPrintf("xprins statistic: fixed only %5.2f ( %5.2f zero) integer variables --> abort \n", *intfixingrate, *zerofixingrate);
   }
   if( fixingcounter == nbinvars + nintvars )
   {
      SCIPstatisticPrintf("xprins statistic: fixed all ( %5.2f zero) integer variables --> abort \n", *zerofixingrate);
   }

   *success = TRUE;

   /* free memory */
   SCIPfreeBufferArray(scip, &zeroblocks);
   SCIPfreeBufferArray(scip, &neqpts);

   return SCIP_OKAY;
}


/** creates a new solution for the original problem by copying the solution of the subproblem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 scip,               /**< original SCIP data structure                        */
   SCIP*                 subscip,            /**< SCIP structure of the subproblem                    */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                     */
   SCIP_HEUR*            heur,               /**< primal heuristic structure                          */
   SCIP_SOL*             subsol,             /**< solution of the subproblem                          */
   SCIP_Bool*            success             /**< used to store whether new solution was found or not */
   )
{
#ifdef SCIP_STATISTIC
   SCIP_HEURDATA* heurdata;
#endif
   SCIP_VAR** vars;                          /* the original problem's variables                */
   int        nvars;
   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */
   SCIP_Real* subsolvals;                    /* solution values of the subproblem               */

   assert(scip != NULL);
   assert(subscip != NULL);
   assert(subvars != NULL);
   assert(subsol != NULL);

#ifdef SCIP_STATISTIC
   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );
#endif

   /* get variables' data */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   assert(nvars <= SCIPgetNOrigVars(subscip));

   SCIP_CALL( SCIPallocBufferArray(scip, &subsolvals, nvars) );

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(subscip, subsol, nvars, subvars, subsolvals) );

   /* create new solution for the original problem */
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newsol, nvars, vars, subsolvals) );

   SCIPstatisticPrintf("xprins statistic: Solution %13.6e found at node %"SCIP_LONGINT_FORMAT"\n",
      SCIPgetSolTransObj(scip, newsol), SCIPsolGetNodenum(subsol));

   /* try to add new solution to scip */
#ifdef SCIP_STATISTIC
   if( !*success )
#endif
      SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, success) );

#ifdef SCIP_STATISTIC
   if( SCIPgetSolTransObj(scip, newsol) < heurdata->bestprimalbd )
      heurdata->bestprimalbd = SCIPgetSolTransObj(scip, newsol);

   SCIPstatisticPrintf("xprins statistic: Solution %13.6e found at node %"SCIP_LONGINT_FORMAT"\n",
      SCIPgetSolTransObj(scip, newsol), SCIPsolGetNodenum(subsol));
#endif

   SCIP_CALL( SCIPfreeSol(scip, &newsol) );

   SCIPfreeBufferArray(scip, &subsolvals);

   return SCIP_OKAY;
}

/** updates heurdata after a run of crossover */
static
void updateFailureStatistic(
   SCIP*                 scip,               /**< original SCIP data structure                        */
   SCIP_HEURDATA*        heurdata            /**< primal heuristic data                               */
   )
{
   /* increase number of failures, calculate next node at which crossover should be called and update actual solutions */
   heurdata->nfailures++;
   heurdata->nextnodenumber = (heurdata->nfailures <= 25
      ? SCIPgetNNodes(scip) + 100*(2LL << heurdata->nfailures) /*lint !e703*/
      : SCIP_LONGINT_MAX);
}



/*
 * Callback methods of primal heuristic
 */

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeXprins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free heuristic data */
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}

/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitXprins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* initialize data */
   heurdata->usednodes = 0;
   heurdata->nfailures = 0;
   heurdata->nextnodenumber = 0;

   /* create random number generator */
   SCIP_CALL( SCIPcreateRandom(scip, &heurdata->randnumgen,
         SCIPinitializeRandomSeed(scip, DEFAULT_RANDSEED), TRUE) );

   return SCIP_OKAY;
}

/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitXprins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free random number generator */
   SCIPfreeRandom(scip, &heurdata->randnumgen);

   return SCIP_OKAY;
}

#ifdef SCIP_STATISTIC
/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolXprins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* initialize statistical data */
   heurdata->avgfixrate = 0.0;
   heurdata->avgzerorate = 0.0;
   heurdata->totalsols = 0;
   heurdata->subsciptime = 0.0;
   heurdata->bestprimalbd = SCIPinfinity(scip);

   return SCIP_OKAY;
}

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
static
SCIP_DECL_HEUREXITSOL(heurExitsolXprins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;
   SCIP_Longint ncalls;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   ncalls = SCIPheurGetNCalls(heur);
   heurdata->avgfixrate /= MAX((SCIP_Real)ncalls, 1.0);
   heurdata->avgzerorate /= MAX((SCIP_Real)ncalls, 1.0);

   /* print detailed statistics */
   SCIPstatisticPrintf("LNS Statistics -- %s:\n", SCIPheurGetName(heur));
   SCIPstatisticPrintf("Calls            : %13"SCIP_LONGINT_FORMAT"\n", ncalls);
   SCIPstatisticPrintf("Failed Fixings   : %13"SCIP_LONGINT_FORMAT"\n", heurdata->nfixfails);
   SCIPstatisticPrintf("Sols             : %13"SCIP_LONGINT_FORMAT"\n", SCIPheurGetNSolsFound(heur));
   SCIPstatisticPrintf("Improving Sols   : %13"SCIP_LONGINT_FORMAT"\n", SCIPheurGetNBestSolsFound(heur));
   SCIPstatisticPrintf("Total Sols       : %13"SCIP_LONGINT_FORMAT"\n", heurdata->totalsols);
   SCIPstatisticPrintf("subSCIP time     : %13.2f\n", heurdata->subsciptime);
   SCIPstatisticPrintf("subSCIP nodes    : %13"SCIP_LONGINT_FORMAT"\n", heurdata->usednodes);
   SCIPstatisticPrintf("Avg. fixing rate : %13.2f\n", 100.0 * heurdata->avgfixrate);
   SCIPstatisticPrintf("Avg. zero rate   : %13.2f\n", 100.0 * heurdata->avgzerorate);
   SCIPstatisticPrintf("Best primal bd.  :");
   if( SCIPisInfinity(scip, heurdata->bestprimalbd) )
      SCIPstatisticPrintf("      infinity\n");
   else
      SCIPstatisticPrintf(" %13.6e\n", heurdata->bestprimalbd);
   SCIPstatisticPrintf("\n");

   return SCIP_OKAY;
}
#endif


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecXprins)
{  /*lint --e{715}*/

   SCIP* masterprob;
   SCIP_HEURDATA* heurdata;

   SCIP* subscip;
   SCIP_VAR** subvars;

   SCIP_Real memorylimit;                    /* memory limit for the subproblem                     */
   SCIP_Real timelimit;                      /* time limit for the subproblem                       */
   SCIP_Longint nstallnodes;                 /* node limit for the subproblem                       */
   SCIP_Real allfixingrate;                  /* percentage of all variables fixed                   */
   SCIP_Real intfixingrate;                  /* percentage of integer variables fixed               */
   SCIP_Real zerofixingrate;                 /* percentage of variables fixed to zero               */

   int* selection;                           /* selected extreme points the heuristic will use, or NULL */
   int* nactualpts;                          /* number of points per block that have actually been selected -- may be less than 'nusedpts' */
   int nblocks;

   SCIP_Bool success;
   SCIP_RETCODE retcode;

   int i;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   nblocks = GCGgetNPricingprobs(scip);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   *result = SCIP_DELAYED;

   /* do not execute the heuristic on invalid relaxation solutions
    * (which is the case if the node has been cut off)
    */
   if( !SCIPisRelaxSolValid(scip) )
   {
      SCIPdebugMessage("skipping Extreme Point RINS: invalid relaxation solution\n");
      return SCIP_OKAY;
   }

   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetStage(masterprob) > SCIP_STAGE_SOLVING || SCIPgetLPSolstat(masterprob) != SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIPdebugMessage("skipping Extreme Point RINS: master LP not solved to optimality.\n");
      return SCIP_OKAY;
   }

   assert(SCIPhasCurrentNodeLP(masterprob));

   /* if heuristic should be delayed, wait until certain number of nodes is reached */
   if( SCIPgetNNodes(scip) < heurdata->nextnodenumber )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* only continue with some fractional variables */
   if( SCIPgetNExternBranchCands(scip) == 0 )
      return SCIP_OKAY;

   /* check whether there is enough time and memory left */
   timelimit = 0.0;
   memorylimit = 0.0;
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit) )
      timelimit -= SCIPgetSolvingTime(scip);
   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );

   /* substract the memory already used by the main SCIP and the estimated memory usage of external software */
   if( !SCIPisInfinity(scip, memorylimit) )
   {
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
      memorylimit -= SCIPgetMemExternEstim(scip)/1048576.0;
   }

   /* abort if no time is left or not enough memory to create a copy of SCIP, including external memory usage */
   if( timelimit <= 0.0 || memorylimit <= 2.0*SCIPgetMemExternEstim(scip)/1048576.0 )
      return SCIP_OKAY;

   /* calculate the maximal number of branching nodes until heuristic is aborted */
   nstallnodes = (SCIP_Longint)(heurdata->nodesquot * SCIPgetNNodes(scip));

   /* reward Crossover if it succeeded often */
   nstallnodes = (SCIP_Longint)
                              (nstallnodes * (1.0 + 2.0*(SCIPheurGetNBestSolsFound(heur)+1.0)/(SCIPheurGetNCalls(heur)+1.0)));

   /* count the setup costs for the sub-MIP as 100 nodes */
   nstallnodes -= 100 * SCIPheurGetNCalls(heur);
   nstallnodes += heurdata->nodesofs;

   /* determine the node limit for the current process */
   nstallnodes -= heurdata->usednodes;
   nstallnodes = MIN(nstallnodes, heurdata->maxnodes);

   /* check whether we have enough nodes left to call subproblem solving */
   if( nstallnodes < heurdata->minnodes )
   {
      SCIPdebugMessage("skipping Extreme Point RINS: nstallnodes=%"SCIP_LONGINT_FORMAT", minnodes=%"SCIP_LONGINT_FORMAT"\n", nstallnodes, heurdata->minnodes);
      return SCIP_OKAY;
   }

   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   SCIPdebugMessage("Executing Extreme Point RINS heuristic ...\n");

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &subvars, SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nactualpts, nblocks) );
   selection = NULL;

   if( heurdata->nusedpts > 0 )
   {
      /* allocate memory */
      SCIP_CALL( SCIPallocBufferArray(scip, &selection, nblocks * heurdata->nusedpts) );

      /* initialize empty selection */
      for( i = 0; i < nblocks * heurdata->nusedpts; ++i )
         selection[i] = -1;

      /* for each block, select extreme points (represented by master variables) to perform RINS */
      success = FALSE;
      if( heurdata->randomization )
      {
         SCIPdebugMessage("selecting extreme points randomly...\n");
         SCIP_CALL( selectExtremePointsRandomized(scip, heurdata, selection, nactualpts, &success) );
         if( !success )
         {
            SCIPdebugMessage("    --> unsuccessful!\n");
         }
      }
      if( !heurdata->randomization || !success )
      {
         SCIPdebugMessage("selecting extreme points deterministically...\n");
         SCIP_CALL( selectExtremePoints(scip, heurdata, selection, nactualpts, &success) );
      }

      /* do not execute heuristic if no new selection of extreme points was found */
      if( !success )
      {
         SCIPdebugMessage("no proper selection could be created - aborting heuristic.\n");

         updateFailureStatistic(scip, heurdata);

         /* free memory */
         SCIPfreeBufferArray(scip, &selection);
         SCIPfreeBufferArray(scip, &nactualpts);
         SCIPfreeBufferArray(scip, &subvars);

         return SCIP_OKAY;
      }
   }
   else
   {
      if( heurdata->randomization )
      {
         SCIPwarningMessage(scip, "Randomization not supported when number of selected extreme points is not constant -- ignoring parameter\n");
      }

      SCIP_CALL( countExtremePoints(scip, selection, heurdata->nusedpts, nactualpts) );
   }

   /* initialize the subproblem */
   SCIP_CALL( SCIPcreate(&subscip) );
   SCIP_CALL( setupSubproblem(scip, subscip, subvars, heurdata, nstallnodes, timelimit, memorylimit) );
   SCIPdebugMessage("XP RINS subproblem: %d vars, %d conss\n", SCIPgetNVars(subscip), SCIPgetNConss(subscip));

   SCIPstatisticPrintf("xprins statistic: called at node %"SCIP_LONGINT_FORMAT"\n", SCIPgetNNodes(scip));

   /* fix variables the variables of the subproblem */
   SCIP_CALL( fixVariables(scip, subscip, subvars, selection, nactualpts, heurdata, &intfixingrate, &zerofixingrate, &success) );

#ifdef SCIP_STATISTIC
   /* for final statistics */
   heurdata->avgfixrate += intfixingrate;
   heurdata->avgzerorate += zerofixingrate;
#endif

   /* if creation of subscip was aborted (e.g. due to number of fixings), free subscip and abort */
   if( !success )
   {
      /* this run will be counted as a failure since the neighborhood of the
       * solution was not fruitful in the sense that it was too big
       */
      updateFailureStatistic(scip, heurdata);
#ifdef SCIP_STATISTIC
      ++heurdata->nfixfails;
#endif
      goto TERMINATE;
   }

   *result = SCIP_DIDNOTFIND;

   /* presolve the subproblem */
   retcode = SCIPpresolve(subscip);

   /* errors in solving the subproblem should not kill the overall solving process;
    * hence, the return code is caught and a warning is printed, only in debug mode, SCIP will stop.
    */
   if( retcode != SCIP_OKAY )
   {
#ifndef NDEBUG
      SCIP_CALL( retcode );
#endif
      SCIPwarningMessage(scip, "Error while presolving subproblem in XP RINS heuristic; sub-SCIP terminated with code <%d>\n",retcode);
      goto TERMINATE;
   }

   SCIPdebugMessage("XP RINS presolved subproblem: %d vars, %d conss, success=%u\n", SCIPgetNVars(subscip), SCIPgetNConss(subscip), success);

   allfixingrate = (SCIPgetNOrigVars(subscip) - SCIPgetNVars(subscip)) / (SCIP_Real)SCIPgetNOrigVars(subscip);

   /* additional variables added in presolving may lead to the subSCIP having more variables than the original */
   allfixingrate = MAX(allfixingrate, 0.0);

   /* after presolving, we should have at least reached a certain fixing rate over ALL variables (including continuous)
    * to ensure that not only the MIP but also the LP relaxation is easy enough
    */
   if( allfixingrate >= heurdata->minfixingrate / 2.0 )
   {
      SCIP_SOL** subsols;
      int nsubsols;

      /* solve the subproblem */
      SCIPdebugMessage("subSCIP: Solving... (node limit = %lld, time limit = %.2g)\n", nstallnodes, timelimit);

      /* Errors in the LP solver should not kill the overall solving process, if the LP is just needed for a heuristic.
       * Hence in optimized mode, the return code is catched and a warning is printed, only in debug mode, SCIP will stop.
       */
#ifdef NDEBUG
      retcode = SCIPsolve(subscip);
      if( retcode != SCIP_OKAY )
      {
         SCIPwarningMessage(scip, "Error while solving subproblem in XP RINS heuristic; sub-SCIP terminated with code <%d>\n",
            retcode);
      }
#else
      SCIP_CALL( SCIPsolve(subscip) );
#endif

#ifdef SCIP_STATISTIC
      heurdata->usednodes += SCIPgetNNodes(subscip);
      heurdata->subsciptime += SCIPgetTotalTime(subscip);
#endif

      /* check, whether a solution was found;
       * due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted
       */
      nsubsols = SCIPgetNSols(subscip);
      subsols = SCIPgetSols(subscip);
      success = FALSE;
#ifdef SCIP_STATISTIC
      heurdata->totalsols += nsubsols;
      for( i = 0; i < nsubsols; ++i )
#else
      for( i = 0; i < nsubsols && !success; ++i )
#endif
      {
         SCIP_CALL( createNewSol(scip, subscip, subvars, heur, subsols[i], &success) );
         if( success )
            *result = SCIP_FOUNDSOL;
      }

      SCIPstatisticPrintf("xprins statistic: fixed %6.3f integer variables ( %6.3f zero), %6.3f all variables, needed %6.1f sec (SCIP time: %6.1f sec), %"SCIP_LONGINT_FORMAT" nodes, found %d solutions, solution %10.4f found at node %"SCIP_LONGINT_FORMAT"\n",
         intfixingrate, zerofixingrate, allfixingrate, SCIPgetSolvingTime(subscip), SCIPgetSolvingTime(scip), SCIPgetNNodes(subscip), nsubsols,
         success ? SCIPgetPrimalbound(scip) : SCIPinfinity(scip), nsubsols > 0 ? SCIPsolGetNodenum(SCIPgetBestSol(subscip)) : -1 );

      if( !success )
      {
         /* if no new solution was found, run was a failure */
         updateFailureStatistic(scip, heurdata);
         SCIPdebugMessage(" -> no subMIP solution found - subSCIP status is %d\n", SCIPgetStatus(subscip));
      }
   }
   else
   {
      SCIPstatisticPrintf("xprins statistic: fixed only %6.3f integer variables ( %6.3f zero), %6.3f all variables --> abort \n", intfixingrate, zerofixingrate, allfixingrate);
   }

TERMINATE:
   /* free memory */
   SCIP_CALL( SCIPfree(&subscip) );
   SCIPfreeBufferArrayNull(scip, &selection);
   SCIPfreeBufferArray(scip, &nactualpts);
   SCIPfreeBufferArray(scip, &subvars);

   return SCIP_OKAY;
}



/*
 * primal heuristic specific interface methods
 */

/** creates the Extreme Point RINS primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurXprins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_HEUR* heur;

   /* create Extreme Point RINS primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecXprins, heurdata) );

   assert(heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeXprins) );
   SCIP_CALL( SCIPsetHeurInit(scip, heur, heurInitXprins) );
   SCIP_CALL( SCIPsetHeurExit(scip, heur, heurExitXprins) );
#ifdef SCIP_STATISTIC
   SCIP_CALL( SCIPsetHeurInitsol(scip, heur, heurInitsolXprins) );
   SCIP_CALL( SCIPsetHeurExitsol(scip, heur, heurExitsolXprins) );
#endif

   /* add Extreme Point RINS primal heuristic parameters */

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/equalityrate",
	 "minimum percentage of coincidence of relaxation and extreme pts",
         &heurdata->equalityrate, FALSE, DEFAULT_EQUALITYRATE, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/nodesofs",
         "number of nodes added to the contingent of the total nodes",
         &heurdata->nodesofs, FALSE, DEFAULT_NODESOFS, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/maxnodes",
         "maximum number of nodes to regard in the subproblem",
         &heurdata->maxnodes, TRUE, DEFAULT_MAXNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/minnodes",
         "minimum number of nodes required to start the subproblem",
         &heurdata->minnodes, TRUE, DEFAULT_MINNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/nusedpts",
         "number of extreme pts per block that will be taken into account (-1: all; 0: all which contribute to current relaxation solution)",
         &heurdata->nusedpts, FALSE, DEFAULT_NUSEDPTS, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/nodesquot",
         "contingent of sub problem nodes in relation to the number of nodes of the original problem",
         &heurdata->nodesquot, FALSE, DEFAULT_NODESQUOT, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/minfixingrate",
         "minimum percentage of integer variables that have to be fixed",
         &heurdata->minfixingrate, FALSE, DEFAULT_MINFIXINGRATE, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/minimprove",
         "factor by which crossover should at least improve the incumbent",
         &heurdata->minimprove, TRUE, DEFAULT_MINIMPROVE, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/randomization",
         "should the choice which sols to take be randomized?",
         &heurdata->randomization, TRUE, DEFAULT_RANDOMIZATION, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/copycuts",
         "if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?",
         &heurdata->copycuts, TRUE, DEFAULT_COPYCUTS, NULL, NULL) );

   return SCIP_OKAY;
}
