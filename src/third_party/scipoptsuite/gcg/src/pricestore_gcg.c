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

/**@file   pricestore_gcg.c
 * @brief  methods for storing priced cols (based on SCIP's separation storage)
 * @author Jonas Witt
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <assert.h>

#include "scip/def.h"
#include "scip/set.h"
#include "scip/stat.h"
#include "scip/clock.h"
#include "scip/lp.h"
#include "scip/var.h"
#include "scip/tree.h"
#include "scip/reopt.h"
#include "scip/event.h"
#include "scip/cons.h"
#include "scip/debug.h"

#include "gcg.h"
#include "pricestore_gcg.h"
#include "struct_pricestore_gcg.h"
#include "pricer_gcg.h"

/*
 * dynamic memory arrays
 */

/** resizes cols and score arrays to be able to store at least num entries */
static
SCIP_RETCODE pricestoreEnsureColsMem(
   GCG_PRICESTORE*       pricestore,          /**< price storage */
   int                   num                 /**< minimal number of slots in array */
   )
{
   assert(pricestore != NULL);
   assert(pricestore->scip != NULL);

   if( num > pricestore->colssize )
   {
      int newsize;

      newsize = SCIPcalcMemGrowSize(pricestore->scip, num);
      SCIP_CALL( SCIPreallocBlockMemoryArray(pricestore->scip, &pricestore->cols, pricestore->colssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(pricestore->scip, &pricestore->objparallelisms, pricestore->colssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(pricestore->scip, &pricestore->orthogonalities, pricestore->colssize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(pricestore->scip, &pricestore->scores, pricestore->colssize, newsize) );
      pricestore->colssize = newsize;
   }
   assert(num <= pricestore->colssize);

   return SCIP_OKAY;
}

/** creates price storage */
SCIP_RETCODE GCGpricestoreCreate(
   SCIP*                 scip,                /**< SCIP data structure */
   GCG_PRICESTORE**      pricestore,          /**< pointer to store price storage */
   SCIP_Real             efficiacyfac,          /**< factor of -redcost/norm in score function */
   SCIP_Real             objparalfac,         /**< factor of objective parallelism in score function */
   SCIP_Real             orthofac,            /**< factor of orthogonalities in score function */
   SCIP_Real             mincolorth,          /**< minimal orthogonality of columns to add
                                                  (with respect to columns added in the current round) */
   GCG_EFFICIACYCHOICE   efficiacychoice      /**< choice to base efficiacy on */
   )
{
   assert(pricestore != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, pricestore) );

   SCIP_CALL( SCIPcreateClock(scip, &(*pricestore)->priceclock) );

   (*pricestore)->scip = scip;
   (*pricestore)->cols = NULL;
   (*pricestore)->objparallelisms = NULL;
   (*pricestore)->orthogonalities = NULL;
   (*pricestore)->scores = NULL;
   (*pricestore)->colssize = 0;
   (*pricestore)->ncols = 0;
   (*pricestore)->nforcedcols = 0;
   (*pricestore)->nefficaciouscols = 0;
   (*pricestore)->ncolsfound = 0;
   (*pricestore)->ncolsfoundround = 0;
   (*pricestore)->ncolsapplied = 0;
   (*pricestore)->infarkas = FALSE;
   (*pricestore)->forcecols = FALSE;
   (*pricestore)->efficiacyfac = efficiacyfac;   /* factor of efficiacies in score function */
   (*pricestore)->objparalfac = objparalfac;     /* factor of objective parallelism in score function */
   (*pricestore)->orthofac = orthofac;           /* factor of orthogonalities in score function */
   (*pricestore)->mincolorth = mincolorth;       /* minimal orthogonality of columns to add
                                                      (with respect to columns added in the current round) */
   (*pricestore)->efficiacychoice = efficiacychoice;

   return SCIP_OKAY;
}

/** frees price storage */
SCIP_RETCODE GCGpricestoreFree(
   SCIP*                 scip,                /**< SCIP data structure */
   GCG_PRICESTORE**      pricestore           /**< pointer to store price storage */
   )
{
   assert(scip == (*pricestore)->scip);
   assert(pricestore != NULL);
   assert(*pricestore != NULL);
   assert((*pricestore)->ncols == 0);

   SCIPdebugMessage("Pricing time in pricestore = %f sec\n", GCGpricestoreGetTime(*pricestore));

   /* free clock */
   SCIP_CALL( SCIPfreeClock(scip, &(*pricestore)->priceclock) );

   SCIPfreeBlockMemoryArrayNull(scip, &(*pricestore)->cols, (*pricestore)->colssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(*pricestore)->objparallelisms, (*pricestore)->colssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(*pricestore)->orthogonalities, (*pricestore)->colssize);
   SCIPfreeBlockMemoryArrayNull(scip, &(*pricestore)->scores, (*pricestore)->colssize);
   SCIPfreeBlockMemory(scip, pricestore);

   return SCIP_OKAY;
}

/** informs price storage, that Farkas pricing starts now */
void GCGpricestoreStartFarkas(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);
   assert(pricestore->ncols == 0);

   pricestore->infarkas = TRUE;
}

/** informs price storage, that Farkas pricing is now finished */
void GCGpricestoreEndFarkas(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);
   assert(pricestore->ncols == 0);

   pricestore->infarkas = FALSE;
}

/** informs price storage, that the following cols should be used in any case */
void GCGpricestoreStartForceCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);
   assert(!pricestore->forcecols);

   pricestore->forcecols = TRUE;
}

/** informs price storage, that the following cols should no longer be used in any case */
void GCGpricestoreEndForceCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);
   assert(pricestore->forcecols);

   pricestore->forcecols = FALSE;
}

/** removes a non-forced col from the price storage */
static
void pricestoreDelCol(
   GCG_PRICESTORE*       pricestore,          /**< price storage */
   int                   pos,                 /**< position of col to delete */
   SCIP_Bool             freecol              /**< should col be freed */
   )
{
   assert(pricestore != NULL);
   assert(pricestore->cols != NULL);
   assert(pricestore->nforcedcols <= pos && pos < pricestore->ncols);

   if( SCIPisDualfeasNegative(pricestore->scip, GCGcolGetRedcost(pricestore->cols[pos])) )
      pricestore->nefficaciouscols--;

   /* free the column */
   if( freecol )
      GCGfreeGcgCol(&(pricestore->cols[pos]));

   /* move last col to the empty position */
   pricestore->cols[pos] = pricestore->cols[pricestore->ncols-1];
   pricestore->objparallelisms[pos] = pricestore->objparallelisms[pricestore->ncols-1];
   pricestore->orthogonalities[pos] = pricestore->orthogonalities[pricestore->ncols-1];
   pricestore->scores[pos] = pricestore->scores[pricestore->ncols-1];
   pricestore->ncols--;
}

/** for a given column, check if an identical column already exists in the price storage;
 *  if one exists, return its position, otherwise, return -1
 */
static
int pricestoreFindEqualCol(
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COL*              col                 /**< column to be checked */
   )
{
   int c;

   for( c = 0; c < pricestore->ncols; ++c )
      if( GCGcolIsEq(col, pricestore->cols[c]) )
         return c;

   return -1;
}

/** adds col to price storage;
 *  if the col should be forced to enter the LP, an infinite score will be used
 */
SCIP_RETCODE GCGpricestoreAddCol(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COL*              col,                /**< priced col */
   SCIP_Bool             forcecol            /**< should the col be forced to enter the LP? */
   )
{
   SCIP_Real colobjparallelism;
   SCIP_Real colscore;

   int oldpos;
   int pos;

   assert(pricestore != NULL);
   assert(pricestore->nforcedcols <= pricestore->ncols);
   assert(col != NULL);

   /* start timing */
   SCIP_CALL( SCIPstartClock(pricestore->scip, pricestore->priceclock) );

   /* a col is forced to enter the LP if
    *  - we construct the initial LP, or
    *  - it has infinite score factor, or
    * if it is a non-forced col and no cols should be added, abort
    */
   forcecol = forcecol || pricestore->forcecols;

   GCGcolComputeNorm(scip, col);

   if( forcecol )
   {
      colscore = SCIPinfinity(scip);
      colobjparallelism = 1.0;
   }
   else
   {
      /* initialize values to invalid (will be initialized during col filtering) */
      colscore = SCIP_INVALID;

      if( SCIPisPositive(scip, pricestore->objparalfac) )
         colobjparallelism = GCGcolComputeDualObjPara(scip, col);
      else
         colobjparallelism = 0.0; /* no need to calculate it */
   }

   oldpos = pricestoreFindEqualCol(pricestore, col);

   pos = -1;

   /* If the column is no duplicate of an existing one, add it */
   if( oldpos == -1 )
   {
      /* get enough memory to store the col */
      SCIP_CALL( pricestoreEnsureColsMem(pricestore, pricestore->ncols+1) );
      assert(pricestore->ncols < pricestore->colssize);

      if( forcecol )
      {
         /* make room at the beginning of the array for forced col */
         pos = pricestore->nforcedcols;
         pricestore->cols[pricestore->ncols] = pricestore->cols[pos];
         pricestore->objparallelisms[pricestore->ncols] = pricestore->objparallelisms[pos];
         pricestore->orthogonalities[pricestore->ncols] = pricestore->orthogonalities[pos];
         pricestore->scores[pricestore->ncols] = pricestore->scores[pos];
         pricestore->nforcedcols++;
      }
      else
         pos = pricestore->ncols;

      pricestore->ncols++;
      if( SCIPisDualfeasNegative(scip, GCGcolGetRedcost(col)) )
         pricestore->nefficaciouscols++;

      /* update statistics of total number of found cols */
      pricestore->ncolsfound++;
      pricestore->ncolsfoundround++;
   }
   /* Otherwise, if the new column is forced and the duplicate one is not,
    * remove the duplicate and replace it by the new column
    */
   else if( forcecol && oldpos >= pricestore->nforcedcols )
   {
      GCGfreeGcgCol(&pricestore->cols[oldpos]);
      pricestore->cols[oldpos] = pricestore->cols[pricestore->nforcedcols];
      pricestore->objparallelisms[oldpos] = pricestore->objparallelisms[pricestore->nforcedcols];
      pricestore->orthogonalities[oldpos] = pricestore->orthogonalities[pricestore->nforcedcols];
      pricestore->scores[oldpos] = pricestore->scores[pricestore->nforcedcols];

      pos = pricestore->nforcedcols;
      pricestore->nforcedcols++;
   }
   /* The column already exists and is not forced, free it */
   else
   {
      /* @todo: This is a little dangerous */
      GCGfreeGcgCol(&col);
   }

   if( pos > -1 )
   {
      SCIPdebugMessage("adding col %p to price storage of size %d (forcecol=%u)\n",
         (void*)col, pricestore->ncols, forcecol);

      /* add col to arrays */
      pricestore->cols[pos] = col;
      pricestore->objparallelisms[pos] = colobjparallelism;
      pricestore->orthogonalities[pos] = 1.0;
      pricestore->scores[pos] = colscore;
   }

   /* stop timing */
   SCIP_CALL( SCIPstopClock(pricestore->scip, pricestore->priceclock) );

   return SCIP_OKAY;
}

/** updates the orthogonalities and scores of the non-forced cols after the given col was added to the LP */
static
SCIP_RETCODE pricestoreUpdateOrthogonalities(
   GCG_PRICESTORE*       pricestore,          /**< price storage */
   GCG_COL*              col,                /**< col that was applied */
   SCIP_Real             mincolorthogonality /**< minimal orthogonality of cols to apply to LP */
   )
{
   int pos;

   assert(pricestore != NULL);

   pos = pricestore->nforcedcols;
   while( pos < pricestore->ncols )
   {
      SCIP_Real thisortho;

      /* update orthogonality */
      thisortho = GCGcolComputeOrth(pricestore->scip, col, pricestore->cols[pos]);

      if( thisortho < pricestore->orthogonalities[pos] )
      {
         if( thisortho < mincolorthogonality )
         {
            /* col is too parallel: delete the col */
            SCIPdebugMessage("    -> deleting parallel col %p after adding %p (pos=%d, orthogonality=%g, score=%g)\n",
               (void*) pricestore->cols[pos], (void*) col, pos, thisortho, pricestore->scores[pos]);
            pricestoreDelCol(pricestore, pos, TRUE);
            continue;
         }
         else
         {
            SCIP_Real colefficiacy;

            /* calculate col's efficacy */
            switch ( pricestore->efficiacychoice )
            {
            case GCG_EFFICIACYCHOICE_DANTZIG:
               colefficiacy = -1.0 * GCGcolGetRedcost(pricestore->cols[pos]);
               break;
            case GCG_EFFICIACYCHOICE_STEEPESTEDGE:
               colefficiacy = -1.0 * GCGcolGetRedcost(pricestore->cols[pos])/ GCGcolGetNorm(col);
               break;
            case GCG_EFFICIACYCHOICE_LAMBDA:
               SCIPerrorMessage("Lambda pricing not yet implemented.\n");
               return SCIP_INVALIDCALL;
            default:
               SCIPerrorMessage("Invalid efficiacy choice.\n");
               return SCIP_INVALIDCALL;
            }

            /* recalculate score */
            pricestore->orthogonalities[pos] = thisortho;
            assert( pricestore->objparallelisms[pos] != SCIP_INVALID ); /*lint !e777*/
            assert( pricestore->scores[pos] != SCIP_INVALID ); /*lint !e777*/


            pricestore->scores[pos] = pricestore->efficiacyfac * colefficiacy
               + pricestore->objparalfac * pricestore->objparallelisms[pos]
               + pricestore->orthofac * thisortho;
         }
      }
      pos++;
   }
   return SCIP_OKAY;
}

/** adds the given col to priced vars and updates the orthogonalities and scores of remaining cols */
static
SCIP_RETCODE pricestoreApplyCol(
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COL*              col,                /**< col to apply to the LP */
   SCIP_Bool             force,              /**< force column */
   SCIP_Real             mincolorthogonality,/**< minimal orthogonality of cols to apply to LP */
   SCIP_Real             score,              /**< score of column (or -1.0 if not specified) */
   SCIP_Bool*            added               /**< pointer to store whether the column was added */
   )
{
   assert(pricestore != NULL);
   assert(added != NULL);

   SCIP_CALL( GCGcreateNewMasterVarFromGcgCol(pricestore->scip, pricestore->infarkas, col, force, added, NULL, score) );
   assert(*added);

   /* update the orthogonalities if needed */
   if( SCIPisGT(pricestore->scip, mincolorthogonality, SCIPepsilon(pricestore->scip)) || SCIPisPositive(pricestore->scip, pricestore->orthofac))
      SCIP_CALL( pricestoreUpdateOrthogonalities(pricestore, col, mincolorthogonality) );

   return SCIP_OKAY;
}

/** returns the position of the best non-forced col in the cols array */
static
int pricestoreGetBestCol(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   SCIP_Real bestscore;
   int bestpos;
   int pos;

   assert(pricestore != NULL);

   bestscore = SCIP_REAL_MIN;
   bestpos = -1;
   for( pos = pricestore->nforcedcols; pos < pricestore->ncols; pos++ )
   {
      /* check if col is current best col */
      assert( pricestore->scores[pos] != SCIP_INVALID ); /*lint !e777*/
      if( pricestore->scores[pos] > bestscore )
      {
         bestscore = pricestore->scores[pos];
         bestpos = pos;
      }
   }

   return bestpos;
}

/** computes score for dual solution and initialized orthogonalities */
static
SCIP_RETCODE computeScore(
   GCG_PRICESTORE*       pricestore,          /**< price storage */
   int                   pos                  /**< position of col to handle */
   )
{
   GCG_COL* col;
   SCIP_Real colefficiacy;
   SCIP_Real colscore;

   col = pricestore->cols[pos];

   /* calculate cut's efficacy */
   switch ( pricestore->efficiacychoice )
   {
   case GCG_EFFICIACYCHOICE_DANTZIG:
      colefficiacy = -1.0 * GCGcolGetRedcost(pricestore->cols[pos]);
      break;
   case GCG_EFFICIACYCHOICE_STEEPESTEDGE:
      colefficiacy = -1.0 * GCGcolGetRedcost(pricestore->cols[pos])/ GCGcolGetNorm(col);
      break;
   case GCG_EFFICIACYCHOICE_LAMBDA:
      SCIPerrorMessage("Lambda pricing not yet implemented.\n");
      return SCIP_INVALIDCALL;
   default:
      SCIPerrorMessage("Invalid efficiacy choice.\n");
      return SCIP_INVALIDCALL;
   }

   assert( pricestore->objparallelisms[pos] != SCIP_INVALID ); /*lint !e777*/
   colscore = pricestore->efficiacyfac * colefficiacy
            + pricestore->objparalfac * pricestore->objparallelisms[pos]
            + pricestore->orthofac * 1.0;;
   assert( !SCIPisInfinity(pricestore->scip, colscore) );

   pricestore->scores[pos] = colscore;

   /* make sure that the orthogonalities are initialized to 1.0 */
   pricestore->orthogonalities[pos] = 1.0;

   return SCIP_OKAY;
}

/** adds cols to priced vars and clears price storage */
SCIP_RETCODE GCGpricestoreApplyCols(
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COLPOOL*          colpool,            /**< GCG column pool */
   SCIP_Bool             usecolpool,         /**< use column pool? */
   int*                  nfoundvars          /**< pointer to store number of variables that were added to the problem */
   )
{
   SCIP* scip;
   SCIP_Bool added;
   int* ncolsappliedprob;
   SCIP_Real mincolorthogonality;
   int maxpricecols;
   int maxpricecolsprob;
   int ncolsapplied;
   int pos;

   assert(pricestore != NULL);

   scip = pricestore->scip;

   SCIPdebugMessage("applying %d cols\n", pricestore->ncols);

   /* start timing */
   SCIP_CALL( SCIPstartClock(scip, pricestore->priceclock) );

   /* get maximal number of cols to add to the LP */
   maxpricecols = GCGpricerGetMaxColsRound(scip);
   maxpricecolsprob = GCGpricerGetMaxColsProb(scip);

   ncolsapplied = 0;
   SCIP_CALL( SCIPallocClearBufferArray(scip, &ncolsappliedprob, GCGgetNPricingprobs(GCGmasterGetOrigprob(scip))) );

   /* set minimal col orthogonality */
   mincolorthogonality = pricestore->mincolorth;
   mincolorthogonality = MAX(mincolorthogonality, SCIPepsilon(scip)); /*lint !e666 */

   /* Compute scores for all non-forced cols and initialize orthogonalities - make sure all cols are initialized again for the current dual solution */
   for( pos = pricestore->nforcedcols; pos < pricestore->ncols; pos++ )
   {
      SCIP_CALL( computeScore(pricestore, pos) );
   }

   /* apply all forced cols */
   for( pos = 0; pos < pricestore->nforcedcols; pos++ )
   {
      GCG_COL* col;
      int probnr;

      col = pricestore->cols[pos];
      assert(SCIPisInfinity(scip, pricestore->scores[pos]));

      probnr = GCGcolGetProbNr(col);

      /* add col to the priced vars and update orthogonalities */
      SCIPdebugMessage(" -> applying forced col %p (probnr = %d)\n", (void*) col, probnr);

      SCIP_CALL( pricestoreApplyCol(pricestore, col, TRUE, mincolorthogonality, pricestore->scores[pos], &added) );

      if( added )
      {
         ++ncolsapplied;
         ++ncolsappliedprob[probnr];
      }
   }

   /* apply non-forced cols */
   while( pricestore->ncols > pricestore->nforcedcols )
   {
      GCG_COL* col;
      int bestpos;
      SCIP_Real score;
      SCIP_Bool keep = FALSE;
      int probnr;

      /* get best non-forced col */
      bestpos = pricestoreGetBestCol(pricestore);
      assert(pricestore->nforcedcols <= bestpos && bestpos < pricestore->ncols);
      assert(pricestore->scores[bestpos] != SCIP_INVALID ); /*lint !e777*/
      col = pricestore->cols[bestpos];
      score = pricestore->scores[bestpos];
      assert(!SCIPisInfinity(scip, pricestore->scores[bestpos]));
      probnr = GCGcolGetProbNr(col);

      /* Do not add (non-forced) non-violated cols.
       * Note: do not take SCIPsetIsEfficacious(), because constraint handlers often add cols w.r.t. SCIPsetIsFeasPositive().
       * Note2: if pricerating/feastolfac != -1, constraint handlers may even add cols w.r.t. SCIPsetIsPositive(); those are currently rejected here
       */
      if( SCIPisDualfeasNegative(scip, GCGcolGetRedcost(col)) && ncolsapplied < maxpricecols && ncolsappliedprob[probnr] < maxpricecolsprob )
      {
         /* add col to the LP and update orthogonalities */
         SCIP_CALL( pricestoreApplyCol(pricestore, col, FALSE, mincolorthogonality, score, &added) );
         keep = FALSE;
         if( added )
         {
            SCIPdebugMessage(" -> applying col %p (pos=%d/%d, probnr=%d, efficacy=%g, objparallelism=%g, orthogonality=%g, score=%g)\n",
            (void*) col, bestpos+1, pricestore->ncols, probnr, GCGcolGetRedcost(pricestore->cols[bestpos]), pricestore->objparallelisms[bestpos],
            pricestore->orthogonalities[bestpos], pricestore->scores[bestpos]);

            ++ncolsapplied;
            ++ncolsappliedprob[probnr];
         }
      }
      else if( usecolpool )
      {
         SCIP_CALL( GCGcolpoolAddCol(colpool, col, &keep) );
      }

      /* delete column from the pricestore */
      pricestoreDelCol(pricestore, bestpos, !keep);
   }

   *nfoundvars = ncolsapplied;

   /* clear the price storage and reset statistics for price round */
   GCGpricestoreClearCols(pricestore);

   SCIPfreeBufferArray(scip, &ncolsappliedprob);

   /* stop timing */
   SCIP_CALL( SCIPstopClock(pricestore->scip, pricestore->priceclock) );

   return SCIP_OKAY;
}

/** clears the price storage without adding the cols to the LP */
void GCGpricestoreClearCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   int c;

   assert(pricestore != NULL);

   SCIPdebugMessage("clearing %d cols\n", pricestore->ncols);

   /* release cols */
   for( c = 0; c < pricestore->ncols; ++c )
   {
      GCGfreeGcgCol(&(pricestore->cols[c]));
   }

   /* reset counters */
   pricestore->ncols = 0;
   pricestore->nefficaciouscols = 0;
   pricestore->nforcedcols = 0;
   pricestore->ncolsfoundround = 0;

   /* if we have just finished the initial LP construction, free the (potentially large) cols array */
   if( pricestore->infarkas )
   {
      SCIPfreeBlockMemoryArrayNull(pricestore->scip, &pricestore->cols, pricestore->colssize);
      SCIPfreeBlockMemoryArrayNull(pricestore->scip, &pricestore->objparallelisms, pricestore->colssize);
      SCIPfreeBlockMemoryArrayNull(pricestore->scip, &pricestore->orthogonalities, pricestore->colssize);
      SCIPfreeBlockMemoryArrayNull(pricestore->scip, &pricestore->scores, pricestore->colssize);

      pricestore->colssize = 0;
   }
}

/** removes cols that are inefficacious w.r.t. the current dual solution from price storage without adding the cols to the LP */
void GCGpricestoreRemoveInefficaciousCols(
   GCG_PRICESTORE*       pricestore          /**< price storage */
   )
{
   int cnt;
   int c;

   assert( pricestore != NULL );

   /* check non-forced cols only */
   cnt = 0;
   c = pricestore->nforcedcols;
   while( c < pricestore->ncols )
   {
      if( !SCIPisDualfeasNegative(pricestore->scip, GCGcolGetRedcost(pricestore->cols[c])) )
      {
         pricestoreDelCol(pricestore, c, TRUE);
         ++cnt;
      }
      else
         ++c;
   }
   SCIPdebugMessage("removed %d non-efficacious cols\n", cnt);
}

/** get cols in the price storage */
GCG_COL** GCGpricestoreGetCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->cols;
}

/** get number of cols in the price storage */
int GCGpricestoreGetNCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->ncols;
}

/** get number of efficacious cols in the price storage */
int GCGpricestoreGetNEfficaciousCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->nefficaciouscols;
}

/** get total number of cols found so far */
int GCGpricestoreGetNColsFound(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->ncolsfound;
}

/** get number of cols found so far in current price round */
int GCGpricestoreGetNColsFoundRound(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->ncolsfoundround;
}

/** get total number of cols applied to the LPs */
int GCGpricestoreGetNColsApplied(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return pricestore->ncolsapplied;
}

/** gets time in seconds used for pricing cols from the pricestore */
SCIP_Real GCGpricestoreGetTime(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   )
{
   assert(pricestore != NULL);

   return SCIPgetClockTime(pricestore->scip, pricestore->priceclock);
}
