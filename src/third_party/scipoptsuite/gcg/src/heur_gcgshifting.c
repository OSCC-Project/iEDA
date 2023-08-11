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

/**@file   heur_gcgshifting.c
 * @brief  LP gcgrounding heuristic that tries to recover from intermediate infeasibilities and shifts continuous variables
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgshifting.h"
#include "gcg.h"
#include "relax_gcg.h"
#include "scip/misc.h"


#define HEUR_NAME             "gcgshifting"
#define HEUR_DESC             "LP rounding heuristic on original variables with infeasibility recovering also using continuous variables"
#define HEUR_DISPCHAR         's'
#define HEUR_PRIORITY         -5000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE
#define HEUR_USESSUBSCIP      FALSE

#define MAXSHIFTINGS          50        /**< maximal number of non improving shiftings */
#define WEIGHTFACTOR          1.1
#define DEFAULT_RANDSEED      31     /**< initial random seed */


/* locally defined heuristic data */
struct SCIP_HeurData
{
   SCIP_SOL*             sol;                /**< working solution */
   SCIP_RANDNUMGEN*      randnumgen;         /**< random number generator */
   SCIP_Longint          lastlp;             /**< last LP number where the heuristic was applied */
};




/*
 * local methods
 */

/** update row violation arrays after a row's activity value changed */
static
void updateViolations(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_ROW**            violrows,           /**< array with currently violated rows */
   int*                  violrowpos,         /**< position of LP rows in violrows array */
   int*                  nviolrows,          /**< pointer to the number of currently violated rows */
   SCIP_Real             oldactivity,        /**< old activity value of LP row */
   SCIP_Real             newactivity         /**< new activity value of LP row */
   )
{
   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_Bool oldviol;
   SCIP_Bool newviol;

   assert(violrows != NULL);
   assert(violrowpos != NULL);
   assert(nviolrows != NULL);

   lhs = SCIProwGetLhs(row);
   rhs = SCIProwGetRhs(row);
   oldviol = (SCIPisFeasLT(scip, oldactivity, lhs) || SCIPisFeasGT(scip, oldactivity, rhs));
   newviol = (SCIPisFeasLT(scip, newactivity, lhs) || SCIPisFeasGT(scip, newactivity, rhs));
   if( oldviol != newviol )
   {
      int rowpos;

      rowpos = SCIProwGetLPPos(row);
      assert(rowpos >= 0);

      if( oldviol )
      {
         int violpos;

         /* the row violation was repaired: remove row from violrows array, decrease violation count */
         violpos = violrowpos[rowpos];
         assert(0 <= violpos && violpos < *nviolrows);
         assert(violrows[violpos] == row);
         violrowpos[rowpos] = -1;
         if( violpos != *nviolrows-1 )
         {
            violrows[violpos] = violrows[*nviolrows-1];
            violrowpos[SCIProwGetLPPos(violrows[violpos])] = violpos;
         }
         (*nviolrows)--;
      }
      else
      {
         /* the row is now violated: add row to violrows array, increase violation count */
         assert(violrowpos[rowpos] == -1);
         violrows[*nviolrows] = row;
         violrowpos[rowpos] = *nviolrows;
         (*nviolrows)++;
      }
   }
}

/** update row activities after a variable's solution value changed */
static
SCIP_RETCODE updateActivities(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            activities,         /**< LP row activities */
   SCIP_ROW**            violrows,           /**< array with currently violated rows */
   int*                  violrowpos,         /**< position of LP rows in violrows array */
   int*                  nviolrows,          /**< pointer to the number of currently violated rows */
   int                   nlprows,            /**< number of rows in current LP */
   SCIP_VAR*             var,                /**< variable that has been changed */
   SCIP_Real             oldsolval,          /**< old solution value of variable */
   SCIP_Real             newsolval           /**< new solution value of variable */
   )
{
   SCIP_COL* col;
   SCIP_ROW** colrows;
   SCIP_Real* colvals;
   SCIP_Real delta;
   int ncolrows;
   int r;

   assert(activities != NULL);
   assert(nviolrows != NULL);
   assert(0 <= *nviolrows && *nviolrows <= nlprows);

   delta = newsolval - oldsolval;
   col = SCIPvarGetCol(var);
   colrows = SCIPcolGetRows(col);
   colvals = SCIPcolGetVals(col);
   ncolrows = SCIPcolGetNLPNonz(col);
   assert(ncolrows == 0 || (colrows != NULL && colvals != NULL));

   for( r = 0; r < ncolrows; ++r )
   {
      SCIP_ROW* row;
      int rowpos;

      row = colrows[r];
      rowpos = SCIProwGetLPPos(row);
      assert(-1 <= rowpos && rowpos < nlprows);

      if( rowpos >= 0 && !SCIProwIsLocal(row) )
      {
         SCIP_Real oldactivity;

         assert(SCIProwIsInLP(row));

         /* update row activity */
         oldactivity = activities[rowpos];
         if( !SCIPisInfinity(scip, -oldactivity) && !SCIPisInfinity(scip, oldactivity) )
         {
            SCIP_Real newactivity;

            newactivity = oldactivity + delta * colvals[r];
            if( SCIPisInfinity(scip, newactivity) )
               newactivity = SCIPinfinity(scip);
            else if( SCIPisInfinity(scip, -newactivity) )
               newactivity = -SCIPinfinity(scip);
            activities[rowpos] = newactivity;

            /* update row violation arrays */
            updateViolations(scip, row, violrows, violrowpos, nviolrows, oldactivity, newactivity);
         }
      }
   }

   return SCIP_OKAY;
}

/** returns a variable, that pushes activity of the row in the given direction with minimal negative impact on other rows;
 *  if variables have equal impact, chooses the one with best objective value improvement in corresponding direction;
 *  prefer fractional integers over other variables in order to become integral during the process;
 *  shifting in a direction is forbidden, if this forces the objective value over the upper bound, or if the variable
 *  was already shifted in the opposite direction
 */
static
SCIP_RETCODE selectShifting(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             rowactivity,        /**< activity of LP row */
   int                   direction,          /**< should the activity be increased (+1) or decreased (-1)? */
   SCIP_Real*            nincreases,         /**< array with weighted number of increasings per variables */
   SCIP_Real*            ndecreases,         /**< array with weighted number of decreasings per variables */
   SCIP_Real             increaseweight,     /**< current weight of increase/decrease updates */
   SCIP_VAR**            shiftvar,           /**< pointer to store the shifting variable, returns NULL if impossible */
   SCIP_Real*            oldsolval,          /**< pointer to store old solution value of shifting variable */
   SCIP_Real*            newsolval           /**< pointer to store new (shifted) solution value of shifting variable */
   )
{
   SCIP_COL** rowcols;
   SCIP_Real* rowvals;
   int nrowcols;
   SCIP_Real activitydelta;
   SCIP_Real bestshiftscore;
   SCIP_Real bestdeltaobj;
   int c;

   assert(direction == +1 || direction == -1);
   assert(nincreases != NULL);
   assert(ndecreases != NULL);
   assert(shiftvar != NULL);
   assert(oldsolval != NULL);
   assert(newsolval != NULL);

   /* get row entries */
   rowcols = SCIProwGetCols(row);
   rowvals = SCIProwGetVals(row);
   nrowcols = SCIProwGetNLPNonz(row);

   /* calculate how much the activity must be shifted in order to become feasible */
   activitydelta = (direction == +1 ? SCIProwGetLhs(row) - rowactivity : SCIProwGetRhs(row) - rowactivity);
   assert((direction == +1 && SCIPisPositive(scip, activitydelta))
      || (direction == -1 && SCIPisNegative(scip, activitydelta)));

   /* select shifting variable */
   bestshiftscore = SCIP_REAL_MAX;
   bestdeltaobj = SCIPinfinity(scip);
   *shiftvar = NULL;
   *newsolval = 0.0;
   *oldsolval = 0.0;
   for( c = 0; c < nrowcols; ++c )
   {
      SCIP_COL* col;
      SCIP_VAR* var;
      SCIP_Real val;
      SCIP_Real solval;
      SCIP_Real shiftscore;
      SCIP_Bool isinteger;
      SCIP_Bool isfrac;
      SCIP_Bool increase;

      col = rowcols[c];
      var = SCIPcolGetVar(col);
      val = rowvals[c];
      assert(!SCIPisZero(scip, val));
      solval = SCIPgetSolVal(scip, sol, var);

      isinteger = (SCIPvarGetType(var) == SCIP_VARTYPE_BINARY || SCIPvarGetType(var) == SCIP_VARTYPE_INTEGER);
      isfrac = isinteger && !SCIPisFeasIntegral(scip, solval);
      increase = (direction * val > 0.0);

      /* calculate the score of the shifting (prefer smaller values) */
      if( isfrac )
         shiftscore = increase ? -1.0 / (SCIPvarGetNLocksUp(var) + 1.0) :
            -1.0 / (SCIPvarGetNLocksDown(var) + 1.0);
      else
      {
         int probindex;
         probindex = SCIPvarGetProbindex(var);

         if( increase )
            shiftscore = ndecreases[probindex]/increaseweight;
         else
            shiftscore = nincreases[probindex]/increaseweight;
         if( isinteger )
            shiftscore += 1.0;
      }

      if( shiftscore <= bestshiftscore )
      {
         SCIP_Real deltaobj;
         SCIP_Real shiftval;

         if( !increase )
         {
            /* shifting down */
            assert(direction * val < 0.0);
            if( isfrac )
               shiftval = SCIPfeasFloor(scip, solval);
            else
            {
               SCIP_Real lb;

               assert(activitydelta/val < 0.0);
               shiftval = solval + activitydelta/val;
               assert(shiftval <= solval); /* may be equal due to numerical digit erasement in the subtraction */
               if( SCIPvarIsIntegral(var) )
                  shiftval = SCIPfeasFloor(scip, shiftval);
               lb = SCIPvarGetLbGlobal(var);
               shiftval = MAX(shiftval, lb);
            }
         }
         else
         {
            /* shifting up */
            assert(direction * val > 0.0);
            if( isfrac )
               shiftval = SCIPfeasCeil(scip, solval);
            else
            {
               SCIP_Real ub;

               assert(activitydelta/val > 0.0);
               shiftval = solval + activitydelta/val;
               assert(shiftval >= solval); /* may be equal due to numerical digit erasement in the subtraction */
               if( SCIPvarIsIntegral(var) )
                  shiftval = SCIPfeasCeil(scip, shiftval);
               ub = SCIPvarGetUbGlobal(var);
               shiftval = MIN(shiftval, ub);
            }
         }

         if( SCIPisEQ(scip, shiftval, solval) )
            continue;

         deltaobj = SCIPvarGetObj(var) * (shiftval - solval);
         if( shiftscore < bestshiftscore || deltaobj < bestdeltaobj )
         {
            bestshiftscore = shiftscore;
            bestdeltaobj = deltaobj;
            *shiftvar = var;
            *oldsolval = solval;
            *newsolval = shiftval;
         }
      }
   }

   return SCIP_OKAY;
}

/** returns a fractional variable, that has most impact on rows in opposite direction, i.e. that is most crucial to
 *  fix in the other direction;
 *  if variables have equal impact, chooses the one with best objective value improvement in corresponding direction;
 *  shifting in a direction is forbidden, if this forces the objective value over the upper bound
 */
static
SCIP_RETCODE selectEssentialRounding(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_Real             minobj,             /**< minimal objective value possible after shifting remaining fractional vars */
   SCIP_VAR**            lpcands,            /**< fractional variables in LP */
   int                   nlpcands,           /**< number of fractional variables in LP */
   SCIP_VAR**            shiftvar,           /**< pointer to store the shifting variable, returns NULL if impossible */
   SCIP_Real*            oldsolval,          /**< old (fractional) solution value of shifting variable */
   SCIP_Real*            newsolval           /**< new (shifted) solution value of shifting variable */
   )
{
   SCIP_Real bestdeltaobj;
   int maxnlocks;
   int v;

   assert(shiftvar != NULL);
   assert(oldsolval != NULL);
   assert(newsolval != NULL);

   /* select shifting variable */
   maxnlocks = -1;
   bestdeltaobj = SCIPinfinity(scip);
   *shiftvar = NULL;
   for( v = 0; v < nlpcands; ++v )
   {
      SCIP_VAR* var;
      SCIP_Real solval;

      var = lpcands[v];
      assert(SCIPvarGetType(var) == SCIP_VARTYPE_BINARY || SCIPvarGetType(var) == SCIP_VARTYPE_INTEGER);

      solval = SCIPgetSolVal(scip, sol, var);
      if( !SCIPisFeasIntegral(scip, solval) )
      {
         SCIP_Real shiftval;
         SCIP_Real obj;
         SCIP_Real deltaobj;
         int nlocks;

         obj = SCIPvarGetObj(var);

         /* shifting down */
         nlocks = SCIPvarGetNLocksUp(var);
         if( nlocks >= maxnlocks )
         {
            shiftval = SCIPfeasFloor(scip, solval);
            deltaobj = obj * (shiftval - solval);
            if( (nlocks > maxnlocks || deltaobj < bestdeltaobj) && minobj - obj < SCIPgetCutoffbound(scip) )
            {
               maxnlocks = nlocks;
               bestdeltaobj = deltaobj;
               *shiftvar = var;
               *oldsolval = solval;
               *newsolval = shiftval;
            }
         }

         /* shifting up */
         nlocks = SCIPvarGetNLocksDown(var);
         if( nlocks >= maxnlocks )
         {
            shiftval = SCIPfeasCeil(scip, solval);
            deltaobj = obj * (shiftval - solval);
            if( (nlocks > maxnlocks || deltaobj < bestdeltaobj) && minobj + obj < SCIPgetCutoffbound(scip) )
            {
               maxnlocks = nlocks;
               bestdeltaobj = deltaobj;
               *shiftvar = var;
               *oldsolval = solval;
               *newsolval = shiftval;
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** adds a given value to the fractionality counters of the rows in which the given variable appears */
static
void addFracCounter(
   int*                  nfracsinrow,        /**< array to store number of fractional variables per row */
   int                   nlprows,            /**< number of rows in LP */
   SCIP_VAR*             var,                /**< variable for which the counting should be updated */
   int                   incval              /**< value that should be added to the corresponding array entries */
   )
{
   SCIP_COL* col;
   SCIP_ROW** rows;
   int nrows;
   int r;

   col = SCIPvarGetCol(var);
   rows = SCIPcolGetRows(col);
   nrows = SCIPcolGetNLPNonz(col);
   for( r = 0; r < nrows; ++r )
   {
      int rowidx;

      rowidx = SCIProwGetLPPos(rows[r]);
      assert(0 <= rowidx && rowidx < nlprows);
      nfracsinrow[rowidx] += incval;
      assert(nfracsinrow[rowidx] >= 0);
   }
}



/*
 * Callback methods
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
#define heurCopyGcgshifting NULL

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
#define heurFreeGcgshifting NULL


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitGcgshifting) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(SCIPheurGetData(heur) == NULL);

   /* create heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );
   heurdata->lastlp = -1;

   /* create random number generator */
   SCIP_CALL( SCIPcreateRandom(scip, &heurdata->randnumgen,
        SCIPinitializeRandomSeed(scip, DEFAULT_RANDSEED), TRUE) );

   SCIPheurSetData(heur, heurdata);

   return SCIP_OKAY;
}

/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitGcgshifting) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* free heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );

   /* free random number generator */
   SCIPfreeRandom(scip, &heurdata->randnumgen);

   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}

/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolGcgshifting)
{
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   heurdata->lastlp = -1;

   return SCIP_OKAY;
}


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolGcgshifting NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecGcgshifting) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP* masterprob;
   SCIP_HEURDATA* heurdata;
   SCIP_SOL* sol;
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   SCIP_ROW** lprows;
   SCIP_Real* activities;
   SCIP_ROW** violrows;
   SCIP_Real* nincreases;
   SCIP_Real* ndecreases;
   int* violrowpos;
   int* nfracsinrow;
   SCIP_Real increaseweight;
   SCIP_Real obj;
   SCIP_Real minobj;
   int nlpcands;
   int nlprows;
   int nvars;
   int nfrac;
   int nviolrows;
   int minnviolrows;
   int nnonimprovingshifts;
   int c;
   int r;
   SCIP_Longint nlps;
   SCIP_Longint ncalls;
   SCIP_Longint nsolsfound;
   SCIP_Longint nnodes;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   *result = SCIP_DIDNOTRUN;

   /* do not execute the heuristic on invalid relaxation solutions
    * (which is the case if the node has been cut off)
    */
   if( !SCIPisRelaxSolValid(scip) )
   {
      SCIPdebugMessage("skipping GCG shifting: invalid relaxation solution\n");
      return SCIP_OKAY;
   }

   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetStage(masterprob) > SCIP_STAGE_SOLVING || SCIPgetLPSolstat(masterprob) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* don't call heuristic, if we have already processed the current LP solution */
   nlps = SCIPgetNLPs(masterprob);
   if( nlps == heurdata->lastlp )
      return SCIP_OKAY;
   heurdata->lastlp = nlps;

   /* don't call heuristic, if it was not successful enough in the past */
   ncalls = SCIPheurGetNCalls(heur);
   nsolsfound = 10*SCIPheurGetNBestSolsFound(heur) + SCIPheurGetNSolsFound(heur);
   nnodes = SCIPgetNNodes(scip);
   if( nnodes % ((ncalls/100)/(nsolsfound+1)+1) != 0 )
      return SCIP_OKAY;

   /* get fractional variables, that should be integral */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &lpcands, &lpcandssol, NULL, &nlpcands, NULL, NULL, NULL, NULL) );
   nfrac = nlpcands;

   /* only call heuristic, if LP solution is fractional */
   if( nfrac == 0 )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   /* get LP rows */
   SCIP_CALL( SCIPgetLPRowsData(scip, &lprows, &nlprows) );

   SCIPdebugMessage("executing GCG shifting heuristic: %d LP rows, %d fractionals\n", nlprows, nfrac);;

   /* get memory for activities, violated rows, and row violation positions */
   nvars = SCIPgetNVars(scip);
   SCIP_CALL( SCIPallocBufferArray(scip, &activities, nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &violrows, nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &violrowpos, nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nfracsinrow, nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nincreases, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &ndecreases, nvars) );
   BMSclearMemoryArray(nfracsinrow, nlprows);
   BMSclearMemoryArray(nincreases, nvars);
   BMSclearMemoryArray(ndecreases, nvars);

   /* get the activities for all globally valid rows;
    * the rows should be feasible, but due to numerical inaccuracies in the LP solver, they can be violated
    */
   nviolrows = 0;
   for( r = 0; r < nlprows; ++r )
   {
      SCIP_ROW* row;

      row = lprows[r];
      assert(SCIProwGetLPPos(row) == r);

      if( !SCIProwIsLocal(row) )
      {
         activities[r] = SCIPgetRowSolActivity(scip, row, GCGrelaxGetCurrentOrigSol(scip));
         if( SCIPisFeasLT(scip, activities[r], SCIProwGetLhs(row) )
            || SCIPisFeasGT(scip, activities[r], SCIProwGetRhs(row)) )
         {
            violrows[nviolrows] = row;
            violrowpos[r] = nviolrows;
            nviolrows++;
         }
         else
            violrowpos[r] = -1;
      }
   }

   /* calc the current number of fractional variables in rows */
   for( c = 0; c < nlpcands; ++c )
      addFracCounter(nfracsinrow, nlprows, lpcands[c], +1);

   /* get the working solution from heuristic's local data */
   sol = heurdata->sol;
   assert(sol != NULL);

   /* copy the current LP solution to the working solution */
   SCIP_CALL( SCIPlinkRelaxSol(scip, sol) );

   /* calculate the minimal objective value possible after rounding fractional variables */
   minobj = SCIPgetSolTransObj(scip, sol);
   /* since the heuristic timing was changed to AFTERNODE, it might happen that it is called on a
    * node with has been cut off; in that case, delay the heuristic
    */
   if( minobj >= SCIPgetCutoffbound(scip) )
   {
      *result = SCIP_DELAYED;
      SCIPfreeBufferArray(scip, &ndecreases);
      SCIPfreeBufferArray(scip, &nincreases);
      SCIPfreeBufferArray(scip, &nfracsinrow);
      SCIPfreeBufferArray(scip, &violrowpos);
      SCIPfreeBufferArray(scip, &violrows);
      SCIPfreeBufferArray(scip, &activities);
      return SCIP_OKAY;
   }
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_Real bestshiftval;

      obj = SCIPvarGetObj(lpcands[c]);
      bestshiftval = obj > 0.0 ? SCIPfeasFloor(scip, lpcandssol[c]) : SCIPfeasCeil(scip, lpcandssol[c]);
      minobj += obj * (bestshiftval - lpcandssol[c]);
   }

   /* try to shift remaining variables in order to become/stay feasible */
   nnonimprovingshifts = 0;
   minnviolrows = INT_MAX;
   increaseweight = 1.0;
   while( (nfrac > 0 || nviolrows > 0) && nnonimprovingshifts < MAXSHIFTINGS )
   {
      SCIP_VAR* shiftvar;
      SCIP_Real oldsolval;
      SCIP_Real newsolval;
      SCIP_Bool oldsolvalisfrac;
      int nprevviolrows;

      SCIPdebugMessage("GCG shifting heuristic: nfrac=%d, nviolrows=%d, obj=%g (best possible obj: %g), cutoff=%g\n",
         nfrac, nviolrows, SCIPgetSolOrigObj(scip, sol), SCIPretransformObj(scip, minobj),
         SCIPretransformObj(scip, SCIPgetCutoffbound(scip)));

      nprevviolrows = nviolrows;

      /* choose next variable to process:
       *  - if a violated row exists, shift a variable decreasing the violation, that has least impact on other rows
       *  - otherwise, shift a variable, that has strongest devastating impact on rows in opposite direction
       */
      shiftvar = NULL;
      oldsolval = 0.0;
      newsolval = 0.0;
      if( nviolrows > 0 && (nfrac == 0 || nnonimprovingshifts < MAXSHIFTINGS-1) )
      {
         SCIP_ROW* row;
         int rowidx;
         int rowpos;
         int direction;

         rowidx = -1;
         rowpos = -1;
         row = NULL;
         if( nfrac > 0 )
         {
            for( rowidx = nviolrows-1; rowidx >= 0; --rowidx )
            {
               row = violrows[rowidx];
               rowpos = SCIProwGetLPPos(row);
               assert(violrowpos[rowpos] == rowidx);
               if( nfracsinrow[rowpos] > 0 )
                  break;
            }
         }
         if( rowidx == -1 )
         {
            rowidx = SCIPrandomGetInt(heurdata->randnumgen, 0, nviolrows-1);
            row = violrows[rowidx];
            rowpos = SCIProwGetLPPos(row);
            assert(0 <= rowpos && rowpos < nlprows);
            assert(violrowpos[rowpos] == rowidx);
            assert(nfracsinrow[rowpos] == 0);
         }
         assert(violrowpos[rowpos] == rowidx);

         SCIPdebugMessage("GCG shifting heuristic: try to fix violated row <%s>: %g <= %g <= %g\n",
            SCIProwGetName(row), SCIProwGetLhs(row), activities[rowpos], SCIProwGetRhs(row));
         SCIPdebug( SCIP_CALL( SCIPprintRow(scip, row, NULL) ) );

         /* get direction in which activity must be shifted */
         assert(SCIPisFeasLT(scip, activities[rowpos], SCIProwGetLhs(row))
            || SCIPisFeasGT(scip, activities[rowpos], SCIProwGetRhs(row)));
         direction = SCIPisFeasLT(scip, activities[rowpos], SCIProwGetLhs(row)) ? +1 : -1;

         /* search a variable that can shift the activity in the necessary direction */
         SCIP_CALL( selectShifting(scip, sol, row, activities[rowpos], direction,
               nincreases, ndecreases, increaseweight, &shiftvar, &oldsolval, &newsolval) );
      }

      if( shiftvar == NULL && nfrac > 0 )
      {
         SCIPdebugMessage("GCG shifting heuristic: search rounding variable and try to stay feasible\n");
         SCIP_CALL( selectEssentialRounding(scip, sol, minobj, lpcands, nlpcands, &shiftvar, &oldsolval, &newsolval) );
      }

      /* check, whether shifting was possible */
      if( shiftvar == NULL || SCIPisEQ(scip, oldsolval, newsolval) )
      {
         SCIPdebugMessage("GCG shifting heuristic:  -> didn't find a shifting variable\n");
         break;
      }

      SCIPdebugMessage("GCG shifting heuristic:  -> shift var <%s>[%g,%g], type=%d, oldval=%g, newval=%g, obj=%g\n",
         SCIPvarGetName(shiftvar), SCIPvarGetLbGlobal(shiftvar), SCIPvarGetUbGlobal(shiftvar), SCIPvarGetType(shiftvar),
         oldsolval, newsolval, SCIPvarGetObj(shiftvar));

      /* update row activities of globally valid rows */
      SCIP_CALL( updateActivities(scip, activities, violrows, violrowpos, &nviolrows, nlprows,
            shiftvar, oldsolval, newsolval) );
      if( nviolrows >= nprevviolrows )
         nnonimprovingshifts++;
      else if( nviolrows < minnviolrows )
      {
         minnviolrows = nviolrows;
         nnonimprovingshifts = 0;
      }

      /* store new solution value and decrease fractionality counter */
      SCIP_CALL( SCIPsetSolVal(scip, sol, shiftvar, newsolval) );

      /* update fractionality counter and minimal objective value possible after shifting remaining variables */
      oldsolvalisfrac = !SCIPisFeasIntegral(scip, oldsolval)
         && (SCIPvarGetType(shiftvar) == SCIP_VARTYPE_BINARY || SCIPvarGetType(shiftvar) == SCIP_VARTYPE_INTEGER);
      obj = SCIPvarGetObj(shiftvar);
      if( (SCIPvarGetType(shiftvar) == SCIP_VARTYPE_BINARY || SCIPvarGetType(shiftvar) == SCIP_VARTYPE_INTEGER )
         && oldsolvalisfrac )
      {
         assert(SCIPisFeasIntegral(scip, newsolval));
         nfrac--;
         nnonimprovingshifts = 0;
         minnviolrows = INT_MAX;
         addFracCounter(nfracsinrow, nlprows, shiftvar, -1);

         /* the rounding was already calculated into the minobj -> update only if rounding in "wrong" direction */
         if( obj > 0.0 && newsolval > oldsolval )
            minobj += obj;
         else if( obj < 0.0 && newsolval < oldsolval )
            minobj -= obj;
      }
      else
      {
         /* update minimal possible objective value */
         minobj += obj * (newsolval - oldsolval);
      }

      /* update increase/decrease arrays */
      if( !oldsolvalisfrac )
      {
         int probindex;

         probindex = SCIPvarGetProbindex(shiftvar);
         assert(0 <= probindex && probindex < nvars);
         increaseweight *= WEIGHTFACTOR;
         if( newsolval < oldsolval )
            ndecreases[probindex] += increaseweight;
         else
            nincreases[probindex] += increaseweight;
         if( increaseweight >= 1e+09 )
         {
            int i;

            for( i = 0; i < nvars; ++i )
            {
               nincreases[i] /= increaseweight;
               ndecreases[i] /= increaseweight;
            }
            increaseweight = 1.0;
         }
      }

      SCIPdebugMessage("gcg shifting heuristic:  -> nfrac=%d, nviolrows=%d, obj=%g (best possible obj: %g)\n",
         nfrac, nviolrows, SCIPgetSolOrigObj(scip, sol), SCIPretransformObj(scip, minobj));
   }

   /* check, if the new solution is feasible */
   if( nfrac == 0 && nviolrows == 0 )
   {
      SCIP_Bool stored;

      /* check solution for feasibility, and add it to solution store if possible
       * neither integrality nor feasibility of LP rows has to be checked, because this is already
       * done in the shifting heuristic itself; however, we better check feasibility of LP rows,
       * because of numerical problems with activity updating
       */
      SCIP_CALL( SCIPtrySol(scip, sol, FALSE, FALSE, FALSE, FALSE, TRUE, &stored) );

      if( stored )
      {
         SCIPdebugMessage("found feasible shifted solution:\n");
         SCIPdebug(SCIPprintSol(scip, sol, NULL, FALSE));
         *result = SCIP_FOUNDSOL;
      }
   }

   /* free memory buffers */
   SCIPfreeBufferArray(scip, &ndecreases);
   SCIPfreeBufferArray(scip, &nincreases);
   SCIPfreeBufferArray(scip, &nfracsinrow);
   SCIPfreeBufferArray(scip, &violrowpos);
   SCIPfreeBufferArray(scip, &violrows);
   SCIPfreeBufferArray(scip, &activities);

   return SCIP_OKAY;
}




/*
 * heuristic specific interface methods
 */

/** creates the GCG shifting heuristic with infeasibility recovering and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurGcgshifting(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   /* include heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyGcgshifting, heurFreeGcgshifting, heurInitGcgshifting, heurExitGcgshifting,
         heurInitsolGcgshifting, heurExitsolGcgshifting, heurExecGcgshifting,
         NULL) );

   return SCIP_OKAY;
}

