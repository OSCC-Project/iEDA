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

/**@file   heur_relaxcolsel.c
 * @brief  relaxation based column selection primal heuristic
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "heur_relaxcolsel.h"
#include "gcg.h"
#include "pricer_gcg.h"


#define HEUR_NAME             "relaxcolsel"
#define HEUR_DESC             "column selection heuristic that tries to round a master LP solution in promising directions"
#define HEUR_DISPCHAR         'x'
#define HEUR_PRIORITY         -100
#define HEUR_FREQ             1
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERLPNODE
#define HEUR_USESSUBSCIP      FALSE

#define DEFAULT_MINCOLUMNS    200             /**< minimum number of columns to regard in the master problem */




/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   /* parameters */
   int                   mincolumns;         /**< minimum number of columns to regard in the master problem */

   /* data */
   SCIP_VAR**            zerovars;           /**< array of master variables corresponding to zero solutions */
   int                   maxzerovars;        /**< capacity of zerovars */
   int                   lastncols;          /**< number of columns in the last call of the heuristic       */
};




/*
 * Local methods
 */


/**
 * initialize working solution as the rounded down master LP solution (and its original counterpart)
 * and the master variable candidates for rounding up
 */
static
SCIP_RETCODE initializeStartsol(
   SCIP*                 scip,
   SCIP_SOL*             mastersol,
   SCIP_SOL*             origsol,
   int*                  blocknr,
   int*                  mastercands,
   SCIP_Real*            candfracs,
   int*                  nmastercands,
   SCIP_Bool*            success
   )
{
   SCIP* origprob;
   SCIP_VAR** mastervars;
   SCIP_Real* mastervals;
   int nmastervars;
   int nblocks;

   int i;
   int j;
   int k;

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(nmastervars >= 0);

   /* get number of blocks */
   nblocks = GCGgetNPricingprobs(origprob);
   assert( nblocks >= 0 );

   /* get master LP solution values */
   SCIP_CALL( SCIPallocBufferArray(scip, &mastervals, nmastervars) );
   SCIP_CALL( SCIPgetSolVals(scip, NULL, nmastervars, mastervars, mastervals) );

   *nmastercands = 0;
   *success = TRUE;

   /* loop over all given master variables */
   for( i = 0; i < nmastervars && *success; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_VARTYPE vartype;
      SCIP_Real solval;
      SCIP_Real roundval;
      SCIP_Real frac;
      int block;
      SCIP_VAR** origvars;
      SCIP_Real* origvals;
      int norigvars;

      /* get master variable */
      mastervar = mastervars[i];
      assert(GCGvarIsMaster(mastervar));
      vartype = SCIPvarGetType(mastervar);

      /* get solution value, its integer part and its fractionality */
      solval = mastervals[i];
      roundval = SCIPfeasFloor(scip, solval);
      frac = SCIPfeasFrac(scip, solval);

      /* get blocknr and corresponding original variables */
      block = GCGvarGetBlock(mastervar);
      origvars = GCGmasterVarGetOrigvars(mastervar);
      origvals = GCGmasterVarGetOrigvals(mastervar);
      norigvars = GCGmasterVarGetNOrigvars(mastervar);

      if( GCGmasterVarIsArtificial(mastervar) )
         continue;

      /* update master solution and original solution */

      /* treat variables representing rays separately */
      if( GCGmasterVarIsRay(mastervar) )
      {
         assert(block >= 0);

         if( SCIPisFeasPositive(scip, roundval) )
         {
            SCIPdebugMessage("  -> (block %d) select ray master variable %s (%d times)\n",
               block, SCIPvarGetName(mastervar), (int) roundval);

            /* set master solution value to rounded down solution */
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, roundval) );

            /* loop over all original variables contained in the current master variable */
            for( j = 0; j < norigvars; ++j )
            {
               SCIP_VAR* origvar;
               SCIP_Real origval;

               /* get original variable and its value in the master variable */
               origvar = origvars[j];
               origval = origvals[j];
               assert(GCGvarIsOriginal(origvar));

               if( SCIPisZero(scip, origval) )
                  continue;

               assert(!SCIPisZero(scip, origval));

               /* the original variable is a linking variable: just transfer the solution value of the direct copy (this is done later) */
               if( GCGoriginalVarIsLinking(origvar) )
                  continue;

               /* increase the corresponding value */
               SCIP_CALL( SCIPincSolVal(scip, origsol, origvar, origval * roundval) );
            }

            /* if the master variable is fractional, add it as a candidate */
            if( !SCIPisFeasFracIntegral(scip, frac) )
            {
               mastercands[*nmastercands] = i;
               candfracs[*nmastercands] = frac;
               ++(*nmastercands);
            }
         }

         continue;
      }

      assert(!GCGmasterVarIsRay(mastervar));

      /* treat variables directly transferred to the master problem (but not linking variables) */
      if( block == -1 )
      {
         SCIP_VAR* origvar;
         int origblock;

         /* get original variable and the block it belongs to */
         assert(norigvars == 1);
         origvar = origvars[0];
         assert(GCGvarIsOriginal(origvar));
         assert(origvals[0] == 1.0);
         origblock = GCGvarGetBlock(origvar);
         assert(origblock == -2 || origblock == -1);

         /* only increase solution value if the variable is not a linking variable
          * (these are treated later) */
         if( origblock == -2 )
            continue;

         /* if the variable should be integral but is not, add rounded down value;
          * otherwise, add (possibly fractional) value */
         if( (vartype == SCIP_VARTYPE_BINARY || vartype == SCIP_VARTYPE_INTEGER )
               && !SCIPisFeasFracIntegral(scip, frac) )
         {
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, roundval) );
            SCIP_CALL( SCIPsetSolVal(origprob, origsol, origvar, roundval) );
            mastercands[*nmastercands] = i;
            candfracs[*nmastercands] = frac;
            ++(*nmastercands);
         }
         else
         {
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, solval) );
            SCIP_CALL( SCIPsetSolVal(origprob, origsol, origvar, solval) );
         }
      }

      /* then, treat master variables representing extreme points and rays */
      else
      {
         if( !SCIPisFeasZero(scip, roundval) )
         {
            /* set master solution value to rounded down solution */
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, roundval) );

            SCIPdebugMessage("  -> (block %d) select master variable %s (%d times)\n",
                  block, SCIPvarGetName(mastervar), (int) roundval);

            for( j = 0; j < norigvars; ++j )
            {
               SCIP_VAR* origvar;
               SCIP_Real origval;

               /* get original variable and its value in the master variable */
               origvar = origvars[j];
               origval = origvals[j];
               assert(GCGvarIsOriginal(origvar));

               /* if the variable is zero, nothing happens */
               if( SCIPisZero(scip, origval) )
                  continue;

               /* linking variables are treated differently; if the variable already has been assigned a value,
                * one must check whether the value for the current block is the same (otherwise, the resulting
                * solution will be infeasible in any case) */
               if( GCGoriginalVarIsLinking(origvar) )
               {
                  SCIP_VAR** linkingpricingvars;
                  SCIP_Bool hasvalue;

                  /* check whether linking variable has already been assigned a value */
                  linkingpricingvars = GCGlinkingVarGetPricingVars(origvar);
                  hasvalue = FALSE;
                  for( k = 0; k < nblocks; ++k )
                     if( linkingpricingvars[k] != NULL )
                        if( blocknr[k] > 0 )
                        {
                           hasvalue = TRUE;
                           break;
                        }

                  /* if the linking variable has not been assigned a value yet, assign a value to
                   * the variable and the corresponding copy in the master problem */
                  if( !hasvalue )
                  {
                     SCIP_VAR* linkingmastervar;

                     linkingmastervar = GCGoriginalVarGetMastervars(origvar)[0];
                     assert(linkingmastervar != NULL);
                     SCIP_CALL( SCIPsetSolVal(origprob, origsol, origvar, origval) );
                     SCIP_CALL( SCIPsetSolVal(scip, mastersol, linkingmastervar, origval) );
                  }
                  /* otherwise, exclude the current master variable, if the point has a different value for it */
                  else
                  {
                     SCIP_Real value;
                     value = SCIPgetSolVal(origprob, origsol, origvar);
                     if( !SCIPisEQ(origprob, value, origval) )
                     {
                        SCIPdebugMessage("    -> cannot use mastervar: origvar %s already has value %g in block %d, different to %g\n",
                              SCIPvarGetName(origvar), value, j, origval);
                        *success = FALSE;
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

                  assert(GCGvarGetBlock(origvar) == block);

                  /* get the corresponding pricing variable */
                  pricingvar = GCGoriginalVarGetPricingVar(origvar);
                  assert(pricingvar != NULL);
                  assert(GCGvarIsPricing(pricingvar));

                  /* get original variables represented by origvar */
                  origpricingvars = GCGpricingVarGetOrigvars(pricingvar);
#ifndef NDEBUG
                  norigpricingvars = GCGpricingVarGetNOrigvars(pricingvar);
#endif

                  /* increase the corresponding value */
                  for( k = 0; k < (int) roundval; ++k )
                  {
                     int blockidx;
                     blockidx = blocknr[block] + k;
#ifndef NDEBUG
                     assert(blockidx < norigpricingvars);
#endif
                     SCIP_CALL( SCIPincSolVal(origprob, origsol, origpricingvars[blockidx], origval) );
                  }
               }
            }

            /* blocks have been filled, so increase blocknr */
            blocknr[block] += (int) roundval;
         }

         /* if the master variable is fractional, add it as a candidate */
         if( !SCIPisFeasFracIntegral(scip, frac) )
         {
            mastercands[*nmastercands] = i;
            candfracs[*nmastercands] = frac;
            ++(*nmastercands);
         }
      }
   }

   SCIPfreeBufferArray(scip, &mastervals);

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
      SCIP_VAR* mastervar;
      int b;

      mastervar = mastervars[i];
      b = GCGvarGetBlock(mastervar);

      /* only regard master variables belonging to the block we are searching for */
      if( b == block )
      {
         SCIP_Real* origvals;
         int norigvars;

         origvals = GCGmasterVarGetOrigvals(mastervar);
         norigvars = GCGmasterVarGetNOrigvars(mastervar);

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
#define heurCopyRelaxcolsel NULL

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeRelaxcolsel)
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
SCIP_DECL_HEURINIT(heurInitRelaxcolsel)
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
SCIP_DECL_HEUREXIT(heurExitRelaxcolsel)
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
#define heurInitsolRelaxcolsel NULL


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolRelaxcolsel NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecRelaxcolsel)
{  /*lint --e{715}*/
   SCIP* origprob;                           /* SCIP structure of original problem  */
   SCIP_HEURDATA* heurdata;                  /* heuristic's data                    */
   SCIP_SOL* mastersol;                      /* working master solution             */
   SCIP_SOL* origsol;                        /* working original solution           */
   SCIP_VAR** mastervars;
   SCIP_Real* candfracs;                     /* fractionalities of candidate variables                               */
   int* blocknr;                             /* for each pricing problem, block we are currently working in          */
   int* mastercands;                         /* master variables which are considered first for rounding up          */
   SCIP_Bool allblocksfull;                  /* indicates if all blocks are full, i.e. all convexity constraints are satisfied */
   SCIP_Bool masterfeas;
   SCIP_Bool success;
   int minnewcols;                           /* minimum number of new columns necessary for calling the heuristic    */
   int nmastervars;
   int nmastercands;
   int nblocks;

   int i;
   int j;
   int k;

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

   /* only call heuristic, if an optimal relaxation solution is at hand */
   if( SCIPgetStage(scip) > SCIP_STAGE_SOLVING || SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   assert(SCIPhasCurrentNodeLP(scip));

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* calculate minimum number of new columns necessary for calling the heuristic;
    * this number is influenced by how successful the heuristic was in the past */
   minnewcols = heurdata->mincolumns * (int) (1.0 * ((1.0 + SCIPheurGetNCalls(heur)) / (1.0 + SCIPheurGetNBestSolsFound(heur))));

   /* if there are not enough new columns since last call, abort heuristic */
   if( nmastervars - heurdata->lastncols < minnewcols )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   SCIPdebugMessage("Executing Relaxation Based Column Selection heuristic (nmastervars = %d) ...\n", nmastervars);

   /* get number of blocks */
   nblocks = GCGgetNPricingprobs(origprob);
   assert(nblocks >= 0);

   /* allocate memory and create working solutions */
   SCIP_CALL( SCIPcreateSol(scip, &mastersol, heur) );
   SCIP_CALL( SCIPcreateSol(origprob, &origsol, heur) );
   SCIP_CALL( SCIPallocBufferArray(scip, &blocknr, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &mastercands, nmastervars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &candfracs, nmastervars) );

   /* initialize the block information */
   BMSclearMemoryArray(blocknr, nblocks);
   BMSclearMemoryArray(candfracs, nmastervars);
   allblocksfull = FALSE;

   /* initialize empty candidate list */
   for( i = 0; i < nmastervars; ++i )
      mastercands[i] = -1;

   /* initialize working original solution as transformation of rounded down master LP solution
    * and get the candidate master variables for rounding up */
   SCIPdebugMessage("initializing starting solution...\n");
   SCIP_CALL( initializeStartsol(scip, mastersol, origsol, blocknr,
         mastercands, candfracs, &nmastercands, &success) );

   if( !success )
   {
      SCIPdebugMessage(" -> not successful.\n");
      goto TERMINATE;
   }

   masterfeas = FALSE;
   success = FALSE;

   SCIPdebugMessage("trying to round up...\n");

   /* first, loop over all candidates for rounding up */
   while( !allblocksfull && !success )
   {
      SCIP_VAR* mastervar;
      int candidx;
      SCIP_Real frac;
      SCIP_Bool compatible;

      SCIP_VAR** origvars;
      SCIP_Real* origvals;
      int norigvars;
      int block;

      /* search the candidate list for a master variable that can be rounded up;
       * take the variable with the highest fractionality */
      mastervar = NULL;
      candidx = -1;
      frac = 0.0;
      compatible = TRUE;
      for( i = 0; i < nmastercands; ++i )
      {
         int idx;
         SCIP_Real tmpfrac;

         idx = mastercands[i];
         tmpfrac = candfracs[i];
         if( idx != -1 && SCIPisFeasGT(scip, tmpfrac, frac) )
         {
            SCIP_VAR* tmpvar;

            tmpvar = mastervars[idx];
            block = GCGvarGetBlock(tmpvar);

            /* consider only variables whose block is not already full
             * or copied master variables */
            if( block == -1 || blocknr[block] < GCGgetNIdenticalBlocks(origprob, block) )
            {
               mastervar = tmpvar;
               candidx = i;
               frac = tmpfrac;
            }
         }
      }

      /* if no variable could be found, abort */
      if( candidx == -1 )
      {
         assert(mastervar == NULL);
         break;
      }

      /* remove the chosen variable from the candidate list */
      assert(candidx >= 0 && candidx < nmastercands);
      mastercands[candidx] = -1;
      candfracs[candidx] = 0.0;

      assert(mastervar != NULL);
      block = GCGvarGetBlock(mastervar);

      SCIPdebugMessage("  -> (block %d) selected master variable %s; frac=%g\n",
         block, SCIPvarGetName(mastervar), frac);

      /* get original variables */
      origvars = GCGmasterVarGetOrigvars(mastervar);
      origvals = GCGmasterVarGetOrigvals(mastervar);
      norigvars = GCGmasterVarGetNOrigvars(mastervar);

      /* increase master value by one and increase solution values in current original solution accordingly */
      SCIP_CALL( SCIPincSolVal(scip, mastersol, mastervar, 1.0) );

      /* update original solution accordingly
       * first, treat variables directly transferred to the master problem (but not linking variables) */
      if( block == -1 )
      {
         SCIP_VAR* origvar;
#ifndef NDEBUG
         int origblock;
#endif

         /* get original variable and the block it belongs to;
          * the variable should not be a linking variable
          * as those have not been added to the candidate list */
         assert(norigvars == 1);
         origvar = origvars[0];
         assert(GCGvarIsOriginal(origvar));
         assert(origvals[0] == 1.0);
#ifndef NDEBUG
         origblock = GCGvarGetBlock(origvar);
         assert(origblock == -1);
#endif

         /* set solution value to rounded up value */
         SCIP_CALL( SCIPincSolVal(origprob, origsol, origvar, 1.0) );
      }

      /* then, treat master variables representing extreme points and rays */
      else
      {
         for( i = 0; i < norigvars; ++i )
         {
            SCIP_VAR* origvar;
            SCIP_Real origval;

            origvar = origvars[i];
            origval = origvals[i];

            assert(GCGvarIsOriginal(origvar));

            /* linking variables are treated differently; if the variable already has been assigned a value,
             * one must check whether the value for the current block is the same (otherwise, the resulting
             * solution will be infeasible in any case) */
            if( GCGoriginalVarIsLinking(origvar) )
            {
               SCIP_VAR** linkingpricingvars;
               SCIP_Bool hasvalue;

               /* check whether linking variable has already been assigned a value */
               linkingpricingvars = GCGlinkingVarGetPricingVars(origvar);
               hasvalue = FALSE;
               for( j = 0; j < nblocks; ++j )
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

                  linkingmastervar = GCGoriginalVarGetMastervars(origvar)[0];
                  assert(linkingmastervar != NULL);
                  SCIP_CALL( SCIPincSolVal(origprob, origsol, origvar, origval) );
                  SCIP_CALL( SCIPincSolVal(scip, mastersol, linkingmastervar, origval) );
               }
               /* otherwise do not take this variable */
               else
               {
                  SCIP_Real value;
                  value = SCIPgetSolVal(origprob, origsol, origvar);
                  if( !SCIPisEQ(origprob, value, origval) )
                  {
                     SCIPdebugMessage("    -> cannot use mastervar: origvar %s already has value %g in block %d, different to %g\n",
                           SCIPvarGetName(origvar), value, j, origval);
                     compatible = FALSE;
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
               if( SCIPisZero(scip, origval) )
                  continue;

               pricingvar = GCGoriginalVarGetPricingVar(origvar);
               assert(pricingvar != NULL);
               assert(GCGvarIsPricing(pricingvar));

               origpricingvars = GCGpricingVarGetOrigvars(pricingvar);

#ifndef NDEBUG
               norigpricingvars = GCGpricingVarGetNOrigvars(pricingvar);
               assert(blocknr[block] < norigpricingvars);
#endif

               /* increase the corresponding value */
               SCIP_CALL( SCIPincSolVal(origprob, origsol, origpricingvars[blocknr[block]], origval) );
            }
         }

         /* if the current master variable cannot be selected (due to conflicting linking variables),
          * reset solution values and choose a new one */
         if( !compatible )
         {
            SCIP_CALL( SCIPincSolVal(scip, mastersol, mastervar, -1.0) );
            for( k = 0; k < i; ++k )
            {
               SCIP_VAR* origvar;
               SCIP_Real origval;

               origvar = origvars[k];
               origval = origvals[k];

               if( GCGoriginalVarIsLinking(origvar) )
               {
                  SCIP_VAR** linkingpricingvars;
                  SCIP_Bool hasvalue;

                  /* check whether linking variable has already been assigned a value */
                  linkingpricingvars = GCGlinkingVarGetPricingVars(origvar);
                  hasvalue = FALSE;
                  for( j = 0; j < nblocks; ++j )
                     if( linkingpricingvars[j] != NULL )
                        if( blocknr[j] > 0 && j != block )
                        {
                           hasvalue = TRUE;
                           break;
                        }

                  /* if the linking variable has not had a value before, set it back to zero */
                  if( !hasvalue )
                  {
                     SCIP_VAR* linkingmastervar;

                     linkingmastervar = GCGoriginalVarGetMastervars(origvar)[0];
                     SCIP_CALL( SCIPincSolVal(origprob, origsol, origvar, -origval) );
                     SCIP_CALL( SCIPincSolVal(scip, mastersol, linkingmastervar, -origval) );
                  }
               }
               else
               {
                  SCIP_VAR* pricingvar;
                  SCIP_VAR** origpricingvars;
#ifndef NDEBUG
                  int norigpricingvars;
#endif

                  pricingvar = GCGoriginalVarGetPricingVar(origvar);
                  assert(pricingvar != NULL);
                  assert(GCGvarIsPricing(pricingvar));

                  origpricingvars = GCGpricingVarGetOrigvars(pricingvar);

#ifndef NDEBUG
                  norigpricingvars = GCGpricingVarGetNOrigvars(pricingvar);
                  assert(blocknr[block] < norigpricingvars);
#endif

                  /* decrease the corresponding value */
                  SCIP_CALL( SCIPincSolVal(origprob, origsol, origpricingvars[blocknr[block]], -origval) );
               }
            }
            continue;
         }

         blocknr[block]++;
      }

      /* try to add the solution to (original) solution pool */
      SCIP_CALL( SCIPtrySol(origprob, origsol, FALSE, FALSE, TRUE, TRUE, TRUE, &success) );

      /* check if all blocks are full */
      allblocksfull = TRUE;
      for( i = 0; i < nblocks && allblocksfull; ++i )
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
   }

   if( success )
   {
      *result = SCIP_FOUNDSOL;
      SCIPdebugMessage("  -> heuristic successful - feasible solution found.\n");
   }
   else
   {
      SCIPdebugMessage("  -> no feasible solution found.\n");
   }

TERMINATE:
   SCIP_CALL( SCIPfreeSol(origprob, &origsol) );
   SCIP_CALL( SCIPfreeSol(scip, &mastersol) );
   SCIPfreeBufferArray(scip, &candfracs);
   SCIPfreeBufferArray(scip, &mastercands);
   SCIPfreeBufferArray(scip, &blocknr);

   heurdata->lastncols = nmastervars;

   return SCIP_OKAY;
}




/*
 * primal heuristic specific interface methods
 */

/** creates the relaxation based column selection primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurRelaxcolsel(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create relaxation based column selection primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );
   heurdata->zerovars = NULL;
   heurdata->maxzerovars = 0;

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyRelaxcolsel, heurFreeRelaxcolsel, heurInitRelaxcolsel, heurExitRelaxcolsel,
         heurInitsolRelaxcolsel, heurExitsolRelaxcolsel, heurExecRelaxcolsel,
         heurdata) );

   /* add relaxation based column selection primal heuristic parameters */
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/relaxcolsel/mincolumns",
         "minimum number of columns to regard in the master problem",
         &heurdata->mincolumns, FALSE, DEFAULT_MINCOLUMNS, 1, INT_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
