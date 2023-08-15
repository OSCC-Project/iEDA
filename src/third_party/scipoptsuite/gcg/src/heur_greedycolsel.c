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

/**@file   heur_greedycolsel.c
 * @brief  greedy column selection primal heuristic
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "heur_greedycolsel.h"
#include "gcg.h"
#include "pricer_gcg.h"


#define HEUR_NAME             "greedycolsel"
#define HEUR_DESC             "greedy column selection heuristic"
#define HEUR_DISPCHAR         'e'
#define HEUR_PRIORITY         0
#define HEUR_FREQ             1
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
/** @todo should heuristic be called during the pricing loop or only after solving a node relaxation? */
#define HEUR_TIMING           SCIP_HEURTIMING_DURINGLPLOOP | SCIP_HEURTIMING_DURINGPRICINGLOOP
#define HEUR_USESSUBSCIP      FALSE

#define DEFAULT_MINCOLUMNS    200             /**< minimum number of columns to regard in the master problem */
#define DEFAULT_USEOBJ        FALSE           /**< use objective coefficients as tie breakers */




/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   /* parameters */
   int                   mincolumns;         /**< minimum number of columns to regard in the master problem */
   SCIP_Bool             useobj;             /**< use objective coefficients as tie breakers                */

   /* data */
   SCIP_VAR**            zerovars;           /**< array of master variables corresponding to zero solutions */
   int                   maxzerovars;        /**< capacity of zerovars */
   int                   lastncols;          /**< number of columns in the last call of the heuristic       */
};




/*
 * Local methods
 */


/** how would the number of violated rows change if mastervar were increased?  */
static
int getViolationChange(
   SCIP*                 scip,
   SCIP_Real*            activities,
   SCIP_VAR*             mastervar
   )
{
   SCIP_COL* col;
   SCIP_ROW** colrows;
   SCIP_Real* colvals;
   int ncolrows;
   int violchange;

   int r;

   /* get the rows in which the master variable appears (only these must be regarded) */
   col = SCIPvarGetCol(mastervar);
   colrows = SCIPcolGetRows(col);
   colvals = SCIPcolGetVals(col);
   ncolrows = SCIPcolGetNLPNonz(col);
   assert(ncolrows == 0 || (colrows != NULL && colvals != NULL));

   violchange = 0;
   for( r = 0; r < ncolrows; r++ )
   {
      SCIP_ROW* row;
      int rowpos;

      row = colrows[r];
      rowpos = SCIProwGetLPPos(row);
      assert(-1 <= rowpos);

      if( rowpos >= 0 && !SCIProwIsLocal(row) )
      {
         SCIP_Real oldactivity;
         SCIP_Real newactivity;

         oldactivity = activities[rowpos];
         newactivity = oldactivity + colvals[r];

         if( SCIPisFeasLT(scip, oldactivity, SCIProwGetLhs(row)) || SCIPisFeasGT(scip, oldactivity, SCIProwGetRhs(row)) )
         {
            if( SCIPisFeasGE(scip, newactivity, SCIProwGetLhs(row)) && SCIPisFeasLE(scip, oldactivity, SCIProwGetRhs(row)) )
               violchange--;
         }
         else
         {
            if( SCIPisFeasLT(scip, newactivity, SCIProwGetLhs(row)) || SCIPisFeasGT(scip, newactivity, SCIProwGetRhs(row)) )
               violchange++;
         }
      }
   }

   return violchange;
}

/** get the index of the "best" master variable w.r.t. pseudo costs */
static
SCIP_RETCODE getBestMastervar(
   SCIP*                 scip,
   SCIP_SOL*             mastersol,
   SCIP_Real*            activities,
   int*                  blocknr,
   SCIP_Bool*            ignored,
   SCIP_Bool             useobj,
   int*                  index,
   int*                  violchange
   )
{
   SCIP* origprob;
   SCIP_VAR** mastervars;
   int nmastervars;

   int i;
   int tmpviolchange;
   SCIP_Real tmpobj;
   SCIP_Real curobj;

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(nmastervars >= 0);

   *index = -1;
   *violchange = INT_MAX;
   curobj = SCIPinfinity(scip);

   for( i = nmastervars - 1; i >= 0; i-- )
   {
      SCIP_VAR* mastervar;
      int block;

      mastervar = mastervars[i];
      assert(GCGvarIsMaster(mastervar));
      block = GCGvarGetBlock(mastervar);

      /** @todo handle copied original variables and linking variables */
      if( block < 0 )
         continue;

      /** @todo handle rays */
      if( GCGmasterVarIsRay(mastervar) )
         continue;

      /* ignore the master variable if the corresponding block is already full
       * or which are fixed
       */
      if( blocknr[block] < GCGgetNIdenticalBlocks(origprob, block )
            && !ignored[i]
            && !SCIPisEQ(scip, SCIPvarGetLbLocal(mastervar), SCIPvarGetUbLocal(mastervar))
            && SCIPisFeasGE(scip, SCIPgetSolVal(scip, mastersol, mastervar), SCIPvarGetUbLocal(mastervar)) )
      {
         tmpviolchange = getViolationChange(scip, activities, mastervar);
         tmpobj = SCIPvarGetObj(mastervar);
         if( tmpviolchange < *violchange ||
               (tmpviolchange == *violchange && SCIPisLE(scip, tmpobj, curobj) && useobj) )
         {
            *index = i;
            *violchange = tmpviolchange;
            curobj = tmpobj;
         }
      }
   }

   return SCIP_OKAY;
}

/** update activities */
static
SCIP_RETCODE updateActivities(
   SCIP*                 scip,
   SCIP_Real*            activities,
   SCIP_VAR*             mastervar
   )
{
   SCIP_COL* col;
   SCIP_ROW** colrows;
   SCIP_Real* colvals;
   int ncolrows;

   int r;

   assert(activities != NULL);

   col = SCIPvarGetCol(mastervar);
   colrows = SCIPcolGetRows(col);
   colvals = SCIPcolGetVals(col);
   ncolrows = SCIPcolGetNLPNonz(col);
   assert(ncolrows == 0 || (colrows != NULL && colvals != NULL));

   for( r = 0; r < ncolrows; ++r )
   {
      SCIP_ROW* row = colrows[r];
      int rowpos = SCIProwGetLPPos(row);

      assert(-1 <= rowpos);

      if( rowpos >= 0 && !SCIProwIsLocal(row) )
      {
         SCIP_Real oldactivity;
         SCIP_Real newactivity;

         assert(SCIProwIsInLP(row));

         /* update row activity */
         oldactivity = activities[rowpos];
         newactivity = oldactivity + colvals[r];
         if( SCIPisInfinity(scip, newactivity) )
            newactivity = SCIPinfinity(scip);
         else if( SCIPisInfinity(scip, -newactivity) )
            newactivity = -SCIPinfinity(scip);
         activities[rowpos] = newactivity;
      }
   }

   return SCIP_OKAY;
}

/** for a given block, search if there is a master variable corresponding to the zero solution;
 * @todo it would be more efficient to "mark" master variables as being trivial */
static
SCIP_RETCODE searchZeroMastervar(
   SCIP*                 scip,
   int                   block,
   SCIP_VAR**            zeromastervar
   )
{
   SCIP_VAR** mastervars;
   int nmastervars;

   int i;
   int j;

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   *zeromastervar = NULL;

   /* go through all master variables */
   for( i = 0; i < nmastervars && *zeromastervar == NULL; ++i )
   {
      SCIP_VAR* mastervar = mastervars[i];
      int b = GCGvarGetBlock(mastervar);

      /* only regard master variables belonging to the block we are searching for */
      if( b == block )
      {
         SCIP_Real* origvals = GCGmasterVarGetOrigvals(mastervar);
         int norigvars = GCGmasterVarGetNOrigvars(mastervar);

         /* check if all original variables contained in the master variable have value zero */
         for( j = 0; j < norigvars; ++j )
            if( !SCIPisZero(scip, origvals[j]) )
               break;

         /* if so, we have found the right master variable */
         if( j == norigvars )
            *zeromastervar = mastervar;
      }
   }

   return SCIP_OKAY;
}

/** for a given block, return the master variable corresponding to the zero solution,
 *  or NULL is there is no such variable available */
static
SCIP_RETCODE getZeroMastervar(
   SCIP*                 scip,
   SCIP_HEURDATA*        heurdata,
   int                   block,
   SCIP_VAR**            zeromastervar
   )
{
   /* if no zero solution is known for the block, look if a master variable has been added
    * and remember the variable for future use */
   if( heurdata->zerovars[block] == NULL )
      SCIP_CALL( searchZeroMastervar(scip, block, zeromastervar) );
   else
      *zeromastervar = heurdata->zerovars[block];

   return SCIP_OKAY;
}




/*
 * Callback methods of primal heuristic
 */


/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
#define heurCopyGreedycolsel NULL

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeGreedycolsel)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free heuristic data */
   SCIPfreeBlockMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitGreedycolsel)
{  /*lint --e{715}*/
   SCIP* origprob;
   SCIP_HEURDATA* heurdata;
   int nblocks;

   int i;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* get number of blocks */
   nblocks = GCGgetNPricingprobs(origprob);

   heurdata->lastncols = 0;

   /* allocate memory and initialize array with NULL pointers */
   if( nblocks > 0 )
   {
      heurdata->maxzerovars = SCIPcalcMemGrowSize(scip, nblocks);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &heurdata->zerovars, heurdata->maxzerovars) );
   }

   for( i = 0; i < nblocks; ++i )
      heurdata->zerovars[i] = NULL;

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitGreedycolsel)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free memory */
   SCIPfreeBlockMemoryArrayNull(scip, &heurdata->zerovars, heurdata->maxzerovars);

   return SCIP_OKAY;
}


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
#define heurInitsolGreedycolsel NULL


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolGreedycolsel NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecGreedycolsel)
{  /*lint --e{715}*/
   SCIP* origprob;                           /* SCIP structure of original problem  */
   SCIP_HEURDATA* heurdata;                  /* heuristic's data                    */
   SCIP_ROW** lprows;                        /* LP rows of master problem           */
   SCIP_SOL* mastersol;                      /* working master solution             */
   SCIP_SOL* origsol;                        /* working original solution           */
   SCIP_VAR** mastervars;
   SCIP_Real* activities;                    /* for each master LP row, activity of current master solution          */
   int* blocknr;                             /* for each pricing problem, block we are currently working in          */
   SCIP_Bool* ignored;                       /* for each master variable, store whether it has to be ignored         */
   SCIP_Bool allblocksfull;                  /* indicates if all blocks are full, i.e. all convexity constraints are satisfied */
   SCIP_Bool masterfeas;
   SCIP_Bool success;
   int minnewcols;                           /* minimum number of new columns necessary for calling the heuristic    */
   int nlprows;
   int nmastervars;
   int nblocks;
   int nviolrows;
   int violchange;

   int i;
   int j;
   int k;
   int index;

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   *result = SCIP_DELAYED;

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(nmastervars >= 0);

   /* calculate minimum number of new columns necessary for calling the heuristic;
    * this number is influenced by how successful the heuristic was in the past */
   minnewcols = heurdata->mincolumns * (int) (1.0 * ((1.0 + SCIPheurGetNCalls(heur)) / (1.0 + SCIPheurGetNBestSolsFound(heur))));

   /* if there are not enough new columns since last call, abort heuristic */
   if( nmastervars - heurdata->lastncols < minnewcols )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   SCIPdebugMessage("Executing Greedy Column Selection heuristic (nmastervars = %d) ...\n", nmastervars);

   /* get number of pricing problems */
   nblocks = GCGgetNPricingprobs(origprob);
   assert(nblocks >= 0);

   /* get master LP rows data */
   SCIP_CALL( SCIPgetLPRowsData(scip, &lprows, &nlprows) );
   assert(lprows != NULL);
   assert(nlprows >= 0);

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &blocknr, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &ignored, nmastervars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &activities, nlprows) );

   /* get memory for working solutions and row activities */
   SCIP_CALL( SCIPcreateSol(scip, &mastersol, heur) );
   SCIP_CALL( SCIPcreateSol(origprob, &origsol, heur) );

   /* initialize block and master variable information */
   BMSclearMemoryArray(blocknr, nblocks);
   BMSclearMemoryArray(ignored, nmastervars);
   allblocksfull = FALSE;

   /* initialize activities with zero and get number of violated rows of zero master solution */
   nviolrows = 0;
   for( i = 0; i < nlprows; i++ )
   {
      SCIP_ROW* row;

      row = lprows[i];
      assert(SCIProwGetLPPos(row) == i);

      if( !SCIProwIsLocal(row) )
      {
         activities[i] = 0;
         if( SCIPisFeasLT(scip, 0.0, SCIProwGetLhs(row)) || SCIPisFeasGT(scip, 0.0, SCIProwGetRhs(row)) )
            nviolrows++;
      }
   }

   SCIPdebugMessage("  -> %d master LP rows violated\n", nviolrows);

   masterfeas = FALSE;
   success = FALSE;

   /* try to increase master variables until all blocks are full */
   while( !allblocksfull && !success )
   {
      SCIP_VAR* mastervar;
      SCIP_VAR** origvars;
      SCIP_Real* origvals;
      int norigvars;
      int block;

      SCIP_CALL( getBestMastervar(scip, mastersol, activities, blocknr, ignored, heurdata->useobj, &index, &violchange) );
      assert(index >= -1 && index < nmastervars);

      /* if no master variable could be selected, abort */
      if( index == -1 )
      {
         assert(violchange == INT_MAX);
         SCIPdebugMessage("  -> no master variable could be selected\n");
         break;
      }

      /* get master variable */
      mastervar = mastervars[index];
      assert(GCGvarIsMaster(mastervar));
      assert(GCGvarGetBlock(mastervar) >= 0);
      assert(!GCGmasterVarIsRay(mastervar));

      /* get blocknr and original variables */
      block = GCGvarGetBlock(mastervar);
      origvars = GCGmasterVarGetOrigvars(mastervar);
      origvals = GCGmasterVarGetOrigvals(mastervar);
      norigvars = GCGmasterVarGetNOrigvars(mastervar);

      SCIPdebugMessage("  -> (block %d) selected master variable %s; violchange=%d\n",
            block, SCIPvarGetName(mastervar), violchange);

      /* increase master value by one and increase solution values in current original solution accordingly */
      SCIP_CALL( SCIPincSolVal(scip, mastersol, mastervar, 1.0) );

      /* update original solution accordingly */
      for( i = 0; i < norigvars; i++ )
      {
         assert(GCGvarIsOriginal(origvars[i]));

         /* linking variables are treated differently; if the variable already has been assigned a value,
          * one must check whether the value for the current block is the same (otherwise, the resulting
          * solution will be infeasible in any case) */
         if( GCGoriginalVarIsLinking(origvars[i]) )
         {
            SCIP_VAR** linkingpricingvars;
            SCIP_Bool hasvalue;

            /* check whether linking variable has already been assigned a value */
            linkingpricingvars = GCGlinkingVarGetPricingVars(origvars[i]);
            hasvalue = FALSE;
            for( j = 0; j < nblocks; j++ )
               if( linkingpricingvars[j] != NULL )
                  if( blocknr[j] > 0 )
                  {
                     hasvalue = TRUE;
                     break;
                  }

            /* if the linking variable has not been assigned a value yet, assign a value to
             * the variable and the corresponding copy in the master problem */
            if( !hasvalue )
            {
               SCIP_VAR* linkingmastervar;

               linkingmastervar = GCGoriginalVarGetMastervars(origvars[i])[0];
               assert(linkingmastervar != NULL);
               SCIP_CALL( SCIPincSolVal(origprob, origsol, origvars[i], origvals[i]) );
               SCIP_CALL( SCIPincSolVal(scip, mastersol, linkingmastervar, origvals[i]) );
            }
            /* otherwise, exclude the current master variable, if the point has a different value for it */
            else
            {
               SCIP_Real value;
               value = SCIPgetSolVal(origprob, origsol, origvars[i]);
               if( !SCIPisEQ(origprob, value, origvals[i]) )
               {
                  SCIPdebugMessage("    -> cannot use mastervar: origvar %s already has value %g in block %d, different to %g\n",
                        SCIPvarGetName(origvars[i]), value, j, origvals[i]);
                  ignored[index] = TRUE;
                  break;
               }
            }
         }
         else
         {
            SCIP_VAR* pricingvar;
            SCIP_VAR** origpricingvars;
#ifndef NDEBUG
            int norigpricingvars;
#endif

            /* if the variable is zero, nothing happens */
            if( SCIPisZero(scip, origvals[i]) )
               continue;

            pricingvar = GCGoriginalVarGetPricingVar(origvars[i]);
            assert(pricingvar != NULL);
            assert(GCGvarIsPricing(pricingvar));

            origpricingvars = GCGpricingVarGetOrigvars(pricingvar);

#ifndef NDEBUG
            norigpricingvars = GCGpricingVarGetNOrigvars(pricingvar);
            assert(blocknr[block] < norigpricingvars);
#endif

            /* increase the corresponding value */
            SCIP_CALL( SCIPincSolVal(origprob, origsol, origpricingvars[blocknr[block]], origvals[i]) );
         }
      }

      /* if the current master variable was set to be ignored, reset solution values and choose a new one */
      if( ignored[index] )
      {
         SCIP_CALL( SCIPincSolVal(scip, mastersol, mastervar, -1.0) );
         for( k = 0; k < i; k++ )
         {
            if( GCGoriginalVarIsLinking(origvars[k]) )
            {
               SCIP_VAR** linkingpricingvars;
               SCIP_Bool hasvalue;

               /* check whether linking variable has already been assigned a value */
               linkingpricingvars = GCGlinkingVarGetPricingVars(origvars[k]);
               hasvalue = FALSE;
               for( j = 0; j < nblocks; j++ )
               {
                  if( linkingpricingvars[j] != NULL )
                  {
                     if( blocknr[j] > 0 && j != block )
                     {
                        hasvalue = TRUE;
                        break;
                     }
                  }
               }

               /* if the linking variable has not had a value before, set it back to zero */
               if( !hasvalue )
               {
                  SCIP_VAR* linkingmastervar;

                  linkingmastervar = GCGoriginalVarGetMastervars(origvars[k])[0];
                  SCIP_CALL( SCIPincSolVal(origprob, origsol, origvars[k], -origvals[k]) );
                  SCIP_CALL( SCIPincSolVal(scip, mastersol, linkingmastervar, -origvals[k]) );
               }
            }
            else
            {
               SCIP_VAR* pricingvar;
               SCIP_VAR** origpricingvars;
#ifndef NDEBUG
               int norigpricingvars;
#endif

               pricingvar = GCGoriginalVarGetPricingVar(origvars[k]);
               assert(pricingvar != NULL);
               assert(GCGvarIsPricing(pricingvar));

               origpricingvars = GCGpricingVarGetOrigvars(pricingvar);

#ifndef NDEBUG
               norigpricingvars = GCGpricingVarGetNOrigvars(pricingvar);
               assert(blocknr[block] < norigpricingvars);
#endif

               /* decrease the corresponding value */
               SCIP_CALL( SCIPincSolVal(origprob, origsol, origpricingvars[blocknr[block]], -origvals[k]) );
            }
         }
         continue;
      }

      blocknr[block]++;

      /* try to add the solution to (original) solution pool */
      SCIP_CALL( SCIPtrySol(origprob, origsol, FALSE, FALSE, TRUE, TRUE, TRUE, &success) );

      /* check if all blocks are full */
      allblocksfull = TRUE;
      for( i = 0; i < nblocks && allblocksfull; i++ )
      {
         int nidentblocks;

         nidentblocks = GCGgetNIdenticalBlocks(origprob, i);

         /* in case the solution is feasible but the block is not full,
          * we need a zero solution for this block in order to generate
          * a corresponding master solution */
         if( success && blocknr[i] < nidentblocks )
         {
            SCIP_VAR* zeromastervar;

            /* fill the block with the zero solution */
            zeromastervar = NULL;
            SCIP_CALL( getZeroMastervar(scip, heurdata, i, &zeromastervar) );
            if( zeromastervar != NULL )
            {
               SCIPdebugMessage("  -> (block %d) selected zero master variable %s (%d times)\n",
                           i, SCIPvarGetName(zeromastervar), nidentblocks - blocknr[i]);

               SCIP_CALL( SCIPincSolVal(scip, mastersol, zeromastervar,
                     (SCIP_Real) nidentblocks - blocknr[i]) );
               blocknr[i] = nidentblocks;
            }
         }

         /** @todo >= should not happen, replace it by == ? */
         if( !(blocknr[i] >= nidentblocks) )
            allblocksfull = FALSE;
      }

      /* if we found a solution for the original instance,
       * also add the corresponding master solution */
      if( success && allblocksfull )
      {
#ifdef SCIP_DEBUG
         SCIP_CALL( SCIPtrySol(scip, mastersol, TRUE, TRUE, TRUE, TRUE, TRUE, &masterfeas) );
#else
         SCIP_CALL( SCIPtrySol(scip, mastersol, FALSE, FALSE, TRUE, TRUE, TRUE, &masterfeas) );
#endif
         if( !masterfeas )
         {
            SCIPdebugMessage("WARNING: original solution feasible, but no solution has been added to master problem.\n");
         }
      }

      /* update number of violated rows and activities array */
      nviolrows += violchange;
      SCIP_CALL( updateActivities(scip, activities, mastervars[index]) );
   }

   if( success )
   {
      *result = SCIP_FOUNDSOL;
      SCIPdebugMessage("heuristic successful - feasible solution found, obj=%g\n",
            SCIPgetSolOrigObj(origprob, origsol));
   }
   else
   {
      SCIPdebugMessage("no feasible solution found or solution already known; %d constraints violated.\n", nviolrows);
   }

   SCIP_CALL( SCIPfreeSol(origprob, &origsol) );
   SCIP_CALL( SCIPfreeSol(scip, &mastersol) );
   SCIPfreeBufferArray(scip, &activities);
   SCIPfreeBufferArray(scip, &ignored);
   SCIPfreeBufferArray(scip, &blocknr);

   heurdata->lastncols = nmastervars;

   return SCIP_OKAY;
}




/*
 * primal heuristic specific interface methods
 */

/** creates the greedy column selection primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurGreedycolsel(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create greedy column selection primal heuristic data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &heurdata) );

   heurdata->zerovars = NULL;
   heurdata->maxzerovars = 0;

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyGreedycolsel, heurFreeGreedycolsel, heurInitGreedycolsel, heurExitGreedycolsel,
         heurInitsolGreedycolsel, heurExitsolGreedycolsel, heurExecGreedycolsel,
         heurdata) );

   /* add greedy column selection primal heuristic parameters */
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/mincolumns",
         "minimum number of columns to regard in the master problem",
         &heurdata->mincolumns, FALSE, DEFAULT_MINCOLUMNS, 1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/useobj",
         "use objective coefficients as tie breakers",
         &heurdata->useobj, TRUE, DEFAULT_USEOBJ, NULL, NULL) );

   return SCIP_OKAY;
}
