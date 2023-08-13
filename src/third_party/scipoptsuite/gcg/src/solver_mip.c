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

/**@file   solver_mip.c
 * @brief  pricing solver solving the pricing problem as a sub-MIP, using SCIP
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Christian Puchert
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define DEBUG_PRICING_ALL_OUTPUT */

#include <assert.h>
#include <string.h>

#include "solver_mip.h"
#include "scip/cons_linear.h"
#include "scip/cons_knapsack.h"
#include "gcg.h"
#include "pricer_gcg.h"
#include "pub_solver.h"
#include "scip/scipdefplugins.h"
#include "pub_gcgcol.h"

#define SOLVER_NAME          "mip"
#define SOLVER_DESC          "pricing solver solving the pricing problem as a sub-MIP, using SCIP"
#define SOLVER_PRIORITY      0
#define SOLVER_HEURENABLED   TRUE            /**< indicates whether the heuristic solving method of the solver should be enabled */
#define SOLVER_EXACTENABLED  TRUE            /**< indicates whether the exact solving method of the solver should be enabled */

#define DEFAULT_CHECKSOLS            TRUE    /**< should solutions be checked extensively */
#define DEFAULT_STARTNODELIMIT       1000LL  /**< start node limit for heuristic pricing */
#define DEFAULT_STARTSTALLNODELIMIT  100LL   /**< start stalling node limit for heuristic pricing */
#define DEFAULT_STARTGAPLIMIT        0.2     /**< start gap limit for heuristic pricing */
#define DEFAULT_STARTSOLLIMIT        10      /**< start solution limit for heuristic pricing */
#define DEFAULT_NODELIMITFAC         1.0     /**< factor by which to increase node limit for heuristic pricing (1.0: add start limit) */
#define DEFAULT_STALLNODELIMITFAC    1.0     /**< factor by which to increase stalling node limit for heuristic pricing (1.0: add start limit) */
#define DEFAULT_GAPLIMITFAC          0.8     /**< factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit) */
#define DEFAULT_SOLLIMITFAC          1.0     /**< factor by which to increase solution limit for heuristic pricing (1.0: add start limit) */
#define DEFAULT_SETTINGSFILE         "-"     /**< settings file to be applied in pricing problems */



/** pricing solver data */
struct GCG_SolverData
{
   /* parameters */
   SCIP_Bool             checksols;           /**< should solutions be checked extensively */
   SCIP_Longint          startnodelimit;      /**< start node limit for heuristic pricing */
   SCIP_Longint          startstallnodelimit; /**< start stalling node limit for heuristic pricing */
   SCIP_Real             startgaplimit;       /**< start gap limit for heuristic pricing */
   int                   startsollimit;       /**< start solution limit for heuristic pricing */
   SCIP_Real             nodelimitfac;        /**< factor by which to increase node limit for heuristic pricing (1.0: add start limit) */
   SCIP_Real             stallnodelimitfac;   /**< factor by which to increase stalling node limit for heuristic pricing (1.0: add start limit) */
   SCIP_Real             gaplimitfac;         /**< factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit) */
   SCIP_Real             sollimitfac;         /**< factor by which to increase solution limit for heuristic pricing (1.0: add start limit) */
   char*                 settingsfile;        /**< settings file to be applied in pricing problems */

   /* solver data */
   SCIP_Longint*         curnodelimit;        /**< current node limit per pricing problem */
   SCIP_Longint*         curstallnodelimit;   /**< current stalling node limit per pricing problem */
   SCIP_Real*            curgaplimit;         /**< current gap limit per pricing problem */
   int*                  cursollimit;         /**< current solution limit per pricing problem */
};


/*
 * Local methods
 */

/** extracts ray from pricing problem */
static
SCIP_RETCODE createColumnFromRay(
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   int                   probnr,             /**< problem number */
   GCG_COL**             newcol              /**< column pointer to store new column */
)
{
   SCIP_VAR** probvars;
   SCIP_VAR** solvars;
   SCIP_Real* solvals;
   int nprobvars;
   int nsolvars;
   int i;

   assert(pricingprob != NULL);
   assert(newcol != NULL);
   assert(SCIPhasPrimalRay(pricingprob));

   probvars = SCIPgetOrigVars(pricingprob);
   nprobvars = SCIPgetNOrigVars(pricingprob);
   nsolvars = 0;
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &solvars, nprobvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &solvals, nprobvars) );

   /* store the primal ray values */
   for ( i = 0; i < nprobvars; i++ )
   {
      if ( SCIPisZero(pricingprob, SCIPgetPrimalRayVal(pricingprob, probvars[i])) )
         continue;

      assert(!SCIPisInfinity(pricingprob, SCIPgetPrimalRayVal(pricingprob, probvars[i])));
      assert(!SCIPisInfinity(pricingprob, -SCIPgetPrimalRayVal(pricingprob, probvars[i])));

      solvars[nsolvars] = probvars[i];
      solvals[nsolvars] = SCIPgetPrimalRayVal(pricingprob, probvars[i]);
      assert(!SCIPisInfinity(pricingprob, solvals[nsolvars]));
      /* todo: check if/ensure that this value is integral! */
      nsolvars++;

      SCIPdebugMessage("%s: %g (obj = %g)\n", SCIPvarGetName(probvars[i]), SCIPgetPrimalRayVal(pricingprob, probvars[i]), SCIPvarGetObj(probvars[i]));
   }

   SCIP_CALL( SCIPfreeSolve(pricingprob, TRUE) );
   SCIP_CALL( SCIPtransformProb(pricingprob) );

   /* todo: is it okay to write pricingprob into column structure? */
   SCIP_CALL( GCGcreateGcgCol(pricingprob, newcol, probnr, solvars, solvals, nsolvars, TRUE, SCIPinfinity(pricingprob)) );

   SCIPfreeBufferArray(pricingprob, &solvals);
   SCIPfreeBufferArray(pricingprob, &solvars);

   SCIPdebugMessage("pricingproblem has an unbounded ray!\n");

   return SCIP_OKAY;
}

/** solves the pricing problem again without presolving */
static
SCIP_RETCODE resolvePricingWithoutPresolving(
   SCIP*                 pricingprob         /**< pricing problem to solve again */
   )
{
   assert(pricingprob != NULL);
   SCIP_CALL( SCIPfreeTransform(pricingprob) );

   SCIP_CALL( SCIPsetIntParam(pricingprob, "presolving/maxrounds", 0) );
   SCIP_CALL( SCIPtransformProb(pricingprob) );
   SCIP_CALL( SCIPsolve(pricingprob) );
   SCIP_CALL( SCIPsetIntParam(pricingprob, "presolving/maxrounds", -1) );

   return SCIP_OKAY;
}

/** checks whether the given solution is equal to one of the former solutions in the sols array */
static
SCIP_RETCODE checkSolNew(
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   SCIP_SOL**            sols,               /**< array of solutions */
   int                   idx,                /**< index of the solution */
   SCIP_Bool*            isnew               /**< pointer to store whether the solution is new */
   )
{
   SCIP_VAR** probvars;
   int nprobvars;
   SCIP_Real* newvals;

   int s;
   int i;

   assert(pricingprob != NULL);
   assert(sols != NULL);
   assert(sols[idx] != NULL);
   assert(isnew != NULL);

   probvars = SCIPgetVars(pricingprob);
   nprobvars = SCIPgetNVars(pricingprob);

   *isnew = TRUE;

   SCIP_CALL( SCIPallocBufferArray(pricingprob, &newvals, nprobvars) );

   SCIP_CALL( SCIPgetSolVals(pricingprob, sols[idx], nprobvars, probvars, newvals) );

   for( s = 0; s < idx && *isnew == TRUE; s++ )
   {
      assert(sols[s] != NULL);
      /** @todo ensure that the solutions are sorted  */
      if( (!SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[s])) && !SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[idx])) )
        && !SCIPisEQ(pricingprob, SCIPgetSolOrigObj(pricingprob, sols[s]), SCIPgetSolOrigObj(pricingprob, sols[idx])) )
         continue;

      if( (SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[s])) && !SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[idx])) )
       ||(!SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[s])) &&  SCIPisInfinity(pricingprob, -SCIPgetSolOrigObj(pricingprob, sols[idx]))) )
         continue;


      if( SCIPsolGetOrigin(sols[s]) != SCIP_SOLORIGIN_ORIGINAL && SCIPsolGetOrigin(sols[idx]) != SCIP_SOLORIGIN_ORIGINAL )
         continue;

      for( i = 0; i < nprobvars; i++ )
         if( !SCIPisEQ(pricingprob, SCIPgetSolVal(pricingprob, sols[s], probvars[i]), newvals[i]) )
            break;

      if( i == nprobvars )
         *isnew = FALSE;
   }

   SCIPfreeBufferArray(pricingprob, &newvals);

   return SCIP_OKAY;
}

/** get the status of the pricing problem */
static
GCG_PRICINGSTATUS getPricingstatus(
  SCIP*                 pricingprob         /**< pricing problem SCIP data structure */
  )
{
   /* all SCIP statuses handled so far; these are currently:
    *    SCIP_STATUS_USERINTERRUPT
    *    SCIP_STATUS_NODELIMIT
    *    SCIP_STATUS_TOTALNODELIMIT
    *    SCIP_STATUS_STALLNODELIMIT
    *    SCIP_STATUS_TIMELIMIT
    *    SCIP_STATUS_MEMLIMIT
    *    SCIP_STATUS_GAPLIMIT
    *    SCIP_STATUS_SOLLIMIT
    *    SCIP_STATUS_BESTSOLLIMIT
    *    SCIP_STATUS_OPTIMAL
    *    SCIP_STATUS_INFEASIBLE
    *    SCIP_STATUS_UNBOUNDED
    *    SCIP_STATUS_INFORUNBD
    */
   /* @todo: can SCIP_STATUS_UNKNOWN happen, too? */
   assert((SCIPgetStatus(pricingprob) >= SCIP_STATUS_USERINTERRUPT && SCIPgetStatus(pricingprob) <= SCIP_STATUS_BESTSOLLIMIT)
          || SCIPgetStatus(pricingprob) == SCIP_STATUS_OPTIMAL
          || SCIPgetStatus(pricingprob) == SCIP_STATUS_INFEASIBLE
          || SCIPgetStatus(pricingprob) == SCIP_STATUS_UNBOUNDED
          || SCIPgetStatus(pricingprob) == SCIP_STATUS_INFORUNBD);

   /* translate SCIP solution status to GCG pricing status */
   switch( SCIPgetStatus(pricingprob) ) 
   {
   case SCIP_STATUS_USERINTERRUPT:
     SCIPdebugMessage("  -> interrupted, %d solutions found\n", SCIPgetNSols(pricingprob)); /*lint -fallthrough*/
   case SCIP_STATUS_UNKNOWN:
   case SCIP_STATUS_TOTALNODELIMIT:
   case SCIP_STATUS_TIMELIMIT:
   case SCIP_STATUS_MEMLIMIT:
   case SCIP_STATUS_BESTSOLLIMIT:
     return GCG_PRICINGSTATUS_UNKNOWN;

   case SCIP_STATUS_NODELIMIT:
   case SCIP_STATUS_STALLNODELIMIT:
   case SCIP_STATUS_GAPLIMIT:
   case SCIP_STATUS_SOLLIMIT:
     return GCG_PRICINGSTATUS_SOLVERLIMIT;

   case SCIP_STATUS_OPTIMAL:
     return GCG_PRICINGSTATUS_OPTIMAL;

   case SCIP_STATUS_INFEASIBLE:
     return GCG_PRICINGSTATUS_INFEASIBLE;

   case SCIP_STATUS_UNBOUNDED:
   case SCIP_STATUS_INFORUNBD:
     return GCG_PRICINGSTATUS_UNBOUNDED;

   default:
     SCIPerrorMessage("invalid SCIP status of pricing problem: %d\n", SCIPgetStatus(pricingprob));
     return GCG_PRICINGSTATUS_UNKNOWN;
   }
}

/** check whether a column contains an infinite solution value */
static
SCIP_Bool solutionHasInfiniteValue(
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   SCIP_SOL*             sol                 /**< solution to be checked */
   )
{
   SCIP_VAR** vars;
   int nvars;

   int i;

   vars = SCIPgetOrigVars(pricingprob);
   nvars = SCIPgetNOrigVars(pricingprob);

   for( i = 0; i < nvars; ++i )
      if( SCIPisInfinity(pricingprob, SCIPgetSolVal(pricingprob, sol, vars[i])) )
         return TRUE;

   return FALSE;
}

/** transforms feasible solutions of the pricing problem into columns */
static
SCIP_RETCODE getColumnsFromPricingprob(
   SCIP*                 scip,               /**< master problem SCIP data structure */
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   int                   probnr,             /**< problem number */
   SCIP_Bool             checksols           /**< should solutions be checked extensively */
)
{
   SCIP_SOL** probsols;
   int nprobsols;

   int s;

   probsols = SCIPgetSols(pricingprob);
   nprobsols = SCIPgetNSols(pricingprob);

   for( s = 0; s < nprobsols; s++ )
   {
      GCG_COL* col;
      SCIP_Bool feasible;
      assert(probsols[s] != NULL);
      SCIP_CALL( SCIPcheckSolOrig(pricingprob, probsols[s], &feasible, FALSE, FALSE) );

      if( !feasible )
      {
         SCIPwarningMessage(pricingprob, "solution of pricing problem %d not feasible:\n", probnr);
         SCIP_CALL( SCIPcheckSolOrig(pricingprob, probsols[s], &feasible, TRUE, TRUE) );
      }

      /* check whether the solution is equal to one of the previous solutions */
      if( checksols )
      {
         SCIP_Bool isnew;

         SCIP_CALL( checkSolNew(pricingprob, probsols, s, &isnew) );

         if( !isnew )
            continue;
      }

      /* Check whether the pricing problem solution has infinite values; if not, transform it to a column */
      if( !solutionHasInfiniteValue(pricingprob, probsols[s]) )
      {
         SCIP_CALL( GCGcreateGcgColFromSol(pricingprob, &col, probnr, probsols[s], FALSE, SCIPinfinity(pricingprob)) );
         SCIP_CALL( GCGpricerAddCol(scip, col) );
      }
      /* If the best solution has infinite values, try to repair it */
      else if( s == 0 )
      {
         SCIP_SOL* newsol;
         SCIP_Bool success;

         newsol = NULL;
         success = FALSE;

         SCIPdebugMessage("solution has infinite values, create a copy with finite values\n");

         SCIP_CALL( SCIPcreateFiniteSolCopy(pricingprob, &newsol, probsols[0], &success) );
         assert(success);
         assert(newsol != NULL);

         SCIP_CALL( GCGcreateGcgColFromSol(pricingprob, &col, probnr, newsol, FALSE, SCIPinfinity(pricingprob)) );
         SCIP_CALL( GCGpricerAddCol(scip, col) );
      }
   }

   return SCIP_OKAY;
}

/** solves the given pricing problem as a sub-SCIP */
static
SCIP_RETCODE solveProblem(
   SCIP*                 scip,               /**< master problem SCIP data structure */
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   int                   probnr,             /**< problem number */
   GCG_SOLVERDATA*       solverdata,         /**< solver data structure */
   SCIP_Real*            lowerbound,         /**< pointer to store lower bound */
   GCG_PRICINGSTATUS*    status              /**< pointer to store pricing problem status */
   )
{
   GCG_COL* col;
   SCIP_RETCODE retcode;
#ifdef SCIP_STATISTIC
   SCIP_Longint oldnnodes;
#endif

   assert(scip != NULL);
   assert(pricingprob != NULL);
   assert(probnr >= 0);
   assert(solverdata != NULL);
   assert(lowerbound != NULL);
   assert(status != NULL);

#ifdef SCIP_STATISTIC
   oldnnodes = SCIPgetNNodes(pricingprob);
#endif
   /* solve the pricing SCIP */
   retcode = SCIPsolve(pricingprob);
   if( retcode != SCIP_OKAY )
   {
      SCIPwarningMessage(pricingprob, "Pricing problem %d terminated with retcode = %d, ignoring\n", probnr, retcode);
      return SCIP_OKAY;
   }

   SCIPdebugMessage("  -> status = %d\n", SCIPgetStatus(pricingprob));
   SCIPdebugMessage("  -> nsols = %d\n", SCIPgetNSols(pricingprob));

#ifdef SCIP_STATISTIC
   SCIPstatisticMessage("P p %d: %"SCIP_LONGINT_FORMAT" no\n", probnr, SCIPgetNNodes(pricingprob) - oldnnodes);
#endif

   *status = getPricingstatus(pricingprob);

   assert(*status != GCG_PRICINGSTATUS_NOTAPPLICABLE);

   switch( *status )
   {
   case GCG_PRICINGSTATUS_INFEASIBLE:
      SCIPdebugMessage("  -> infeasible.\n");
      break;

   /* The pricing problem was declared to be unbounded and we should have a primal ray at hand,
    * so copy the primal ray into the solution structure and mark it to be a primal ray
    */
   case GCG_PRICINGSTATUS_UNBOUNDED:
      if( !SCIPhasPrimalRay(pricingprob) )
      {
         SCIP_CALL( resolvePricingWithoutPresolving(pricingprob) );
      }

      SCIPdebugMessage("  -> unbounded, creating column from ray\n");
      SCIP_CALL( createColumnFromRay(pricingprob, probnr, &col) );
      SCIP_CALL( GCGpricerAddCol(scip, col) );

      break;

   /* If the pricing problem is neither infeasible nor unbounded, try to extract feasible columns */
   case GCG_PRICINGSTATUS_UNKNOWN:
   case GCG_PRICINGSTATUS_SOLVERLIMIT:
   case GCG_PRICINGSTATUS_OPTIMAL:
      assert(SCIPgetNSols(pricingprob) > 0
        || (SCIPgetStatus(pricingprob) != SCIP_STATUS_OPTIMAL
          && SCIPgetStatus(pricingprob) != SCIP_STATUS_GAPLIMIT
          && SCIPgetStatus(pricingprob) != SCIP_STATUS_SOLLIMIT));

      /* Transform at most maxcols many solutions from the pricing problem into columns */
      SCIP_CALL( getColumnsFromPricingprob(scip, pricingprob, probnr, solverdata->checksols) );

      *lowerbound = SCIPgetDualbound(pricingprob);

      SCIPdebugMessage("  -> lowerbound = %.4g\n", *lowerbound);
      break;

   default:
      SCIPerrorMessage("Pricing problem %d has invalid status: %d\n", probnr, SCIPgetStatus(pricingprob));
      break; /*lint !e788 */
   }

   return SCIP_OKAY;
}


/*
 * Callback methods for pricing problem solver
 */

/** destructor of pricing solver to free user data (called when SCIP is exiting) */
static
GCG_DECL_SOLVERFREE(solverFreeMip)
{
   GCG_SOLVERDATA* solverdata;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   SCIPfreeMemory(scip, &solverdata);

   GCGsolverSetData(solver, NULL);

   return SCIP_OKAY;
}

/** initialization method of pricing solver (called after problem was transformed and solver is active) */
#define solverInitMip NULL

/** deinitialization method of pricing solver (called before transformed problem is freed and solver is active) */
#define solverExitMip NULL

/** solving process initialization method of pricing solver (called when branch and bound process is about to begin) */
static
GCG_DECL_SOLVERINITSOL(solverInitsolMip)
{
   GCG_SOLVERDATA* solverdata;
   int npricingprobs;
   int i;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   npricingprobs = GCGgetNPricingprobs(GCGmasterGetOrigprob(scip));

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->curnodelimit, npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->curstallnodelimit, npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->curgaplimit, npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->cursollimit, npricingprobs) );

   for( i = 0; i < npricingprobs; ++i )
   {
      solverdata->curnodelimit[i] = solverdata->startnodelimit;
      solverdata->curstallnodelimit[i] = solverdata->startstallnodelimit;
      solverdata->curgaplimit[i] = solverdata->startgaplimit;
      solverdata->cursollimit[i] = solverdata->startsollimit;
   }

   return SCIP_OKAY;
}

/** solving process deinitialization method of pricing solver (called before branch and bound process data is freed) */
static
GCG_DECL_SOLVEREXITSOL(solverExitsolMip)
{
   GCG_SOLVERDATA* solverdata;
   int npricingprobs;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   npricingprobs = GCGgetNPricingprobs(GCGmasterGetOrigprob(scip));

   SCIPfreeBlockMemoryArray(scip, &solverdata->cursollimit, npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &solverdata->curgaplimit, npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &solverdata->curstallnodelimit, npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &solverdata->curnodelimit, npricingprobs);

   return SCIP_OKAY;
}

#define solverUpdateMip NULL

/** solving method for pricing solver which solves the pricing problem to optimality */
static
GCG_DECL_SOLVERSOLVE(solverSolveMip)
{  /*lint --e{715}*/
   GCG_SOLVERDATA* solverdata;

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   if( strcmp(solverdata->settingsfile, "-") != 0 )
   {
      SCIP_CALL( SCIPreadParams(pricingprob, solverdata->settingsfile) );
   }

   SCIP_CALL( SCIPsetLongintParam(pricingprob, "limits/stallnodes", -1LL) );
   SCIP_CALL( SCIPsetLongintParam(pricingprob, "limits/nodes", -1LL) );
   SCIP_CALL( SCIPsetRealParam(pricingprob, "limits/gap", 0.0) );
   SCIP_CALL( SCIPsetIntParam(pricingprob, "limits/solutions", -1) );

#ifdef DEBUG_PRICING_ALL_OUTPUT
   SCIP_CALL( SCIPsetIntParam(pricingprob, "display/verblevel", SCIP_VERBLEVEL_HIGH) );
   SCIP_CALL( SCIPwriteParams(pricingprob, "pricing.set", TRUE, TRUE) );
#endif

   *lowerbound = -SCIPinfinity(pricingprob);

   SCIPdebugMessage("Solving pricing %d (pointer: %p)\n", probnr, (void*)pricingprob);
   SCIP_CALL( solveProblem(scip, pricingprob, probnr, solverdata, lowerbound, status) );

#ifdef DEBUG_PRICING_ALL_OUTPUT
   SCIP_CALL( SCIPsetIntParam(pricingprob, "display/verblevel", 0) );
   SCIP_CALL( SCIPprintStatistics(pricingprob, NULL) );
#endif

   return SCIP_OKAY;
}

/** heuristic solving method of mip solver */
static
GCG_DECL_SOLVERSOLVEHEUR(solverSolveHeurMip)
{  /*lint --e{715}*/
   GCG_SOLVERDATA* solverdata;

#ifdef DEBUG_PRICING_ALL_OUTPUT
   SCIP_CALL( SCIPsetIntParam(pricingprob, "display/verblevel", SCIP_VERBLEVEL_HIGH) );
#endif

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   *lowerbound = -SCIPinfinity(pricingprob);

   /* setup heuristic solver parameters */
   if( SCIPgetStage(pricingprob) == SCIP_STAGE_PROBLEM )
   {
      solverdata->curnodelimit[probnr] = solverdata->startnodelimit;
      solverdata->curstallnodelimit[probnr] = solverdata->startstallnodelimit;
      solverdata->curgaplimit[probnr] = solverdata->startgaplimit;
      solverdata->cursollimit[probnr] = solverdata->startsollimit;
   }
   else
   {
      if( SCIPgetStatus(pricingprob) == SCIP_STATUS_NODELIMIT )
      {
         if( solverdata->nodelimitfac > 1.0 )
            solverdata->curnodelimit[probnr] = (SCIP_Longint) (solverdata->curnodelimit[probnr] * solverdata->nodelimitfac);
         else
            solverdata->curnodelimit[probnr] += solverdata->startnodelimit;
      }
      else if( SCIPgetStatus(pricingprob) == SCIP_STATUS_STALLNODELIMIT )
      {
         if( solverdata->stallnodelimitfac > 1.0 )
            solverdata->curstallnodelimit[probnr] = (SCIP_Longint) (solverdata->curstallnodelimit[probnr] * solverdata->stallnodelimitfac);
         else
            solverdata->curstallnodelimit[probnr] += solverdata->startstallnodelimit;
      }
      else if( SCIPgetStatus(pricingprob) == SCIP_STATUS_GAPLIMIT )
      {
         if( solverdata->gaplimitfac < 1.0 )
            solverdata->curgaplimit[probnr] *= solverdata->gaplimitfac;
         else
            solverdata->curgaplimit[probnr] = MAX(solverdata->curgaplimit[probnr] - solverdata->startgaplimit, 0.0);
      }
      else if( SCIPgetStatus(pricingprob) == SCIP_STATUS_SOLLIMIT )
      {
         if( solverdata->sollimitfac > 1.0 )
            solverdata->cursollimit[probnr] = (int) (solverdata->cursollimit[probnr] * solverdata->sollimitfac);
         else
            solverdata->cursollimit[probnr] += solverdata->startsollimit;
      }
      else
      {
         *status = GCG_PRICINGSTATUS_UNKNOWN;
         return SCIP_OKAY;
      }
   }
   SCIP_CALL( SCIPsetLongintParam(pricingprob, "limits/nodes", solverdata->curnodelimit[probnr]) );
   SCIP_CALL( SCIPsetLongintParam(pricingprob, "limits/stallnodes", solverdata->curstallnodelimit[probnr]) );
   SCIP_CALL( SCIPsetRealParam(pricingprob, "limits/gap", solverdata->curgaplimit[probnr]) );
   SCIP_CALL( SCIPsetIntParam(pricingprob, "limits/solutions", solverdata->cursollimit[probnr]) );

   /* solve the pricing problem */
   SCIPdebugMessage("Solving pricing %d heuristically (pointer: %p)\n", probnr, (void*)pricingprob);
   SCIP_CALL( solveProblem(scip, pricingprob, probnr, solverdata, lowerbound, status) );

#ifdef DEBUG_PRICING_ALL_OUTPUT
   SCIP_CALL( SCIPsetIntParam(pricingprob, "display/verblevel", 0) );
   SCIP_CALL( SCIPprintStatistics(pricingprob, NULL) );
#endif

   return SCIP_OKAY;
}

/** creates the mip solver for pricing problems and includes it in GCG */
SCIP_RETCODE GCGincludeSolverMip(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP* origprob;
   GCG_SOLVERDATA* solverdata;

   origprob = GCGmasterGetOrigprob(scip);

   SCIP_CALL( SCIPallocMemory(scip, &solverdata) );
   solverdata->settingsfile = NULL;

   SCIP_CALL( GCGpricerIncludeSolver(scip, SOLVER_NAME, SOLVER_DESC, SOLVER_PRIORITY,
         SOLVER_HEURENABLED, SOLVER_EXACTENABLED,
         solverUpdateMip, solverSolveMip, solverSolveHeurMip, solverFreeMip, solverInitMip, solverExitMip,
         solverInitsolMip, solverExitsolMip, solverdata) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricingsolver/mip/checksols",
         "should solutions of the pricing MIPs be checked for duplicity?",
         &solverdata->checksols, TRUE, DEFAULT_CHECKSOLS, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(origprob, "pricingsolver/mip/startnodelimit",
         "start node limit for heuristic pricing",
         &solverdata->startnodelimit, TRUE, DEFAULT_STARTNODELIMIT, -1LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(origprob, "pricingsolver/mip/startstallnodelimit",
         "start stalling node limit for heuristic pricing",
         &solverdata->startstallnodelimit, TRUE, DEFAULT_STARTSTALLNODELIMIT, -1LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/mip/startgaplimit",
         "start gap limit for heuristic pricing",
         &solverdata->startgaplimit, TRUE, DEFAULT_STARTGAPLIMIT, 0.0, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origprob, "pricingsolver/mip/startsollimit",
         "start solution limit for heuristic pricing",
         &solverdata->startsollimit, TRUE, DEFAULT_STARTSOLLIMIT, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/mip/nodelimitfac",
         "factor by which to increase node limit for heuristic pricing (1.0: add start limit)",
         &solverdata->nodelimitfac, TRUE, DEFAULT_NODELIMITFAC, 1.0, SCIPinfinity(origprob), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/mip/stallnodelimitfac",
         "factor by which to increase stalling node limit for heuristic pricing (1.0: add start limit)",
         &solverdata->stallnodelimitfac, TRUE, DEFAULT_STALLNODELIMITFAC, 1.0, SCIPinfinity(origprob), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/mip/gaplimitfac",
         "factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit)",
         &solverdata->gaplimitfac, TRUE, DEFAULT_GAPLIMITFAC, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/mip/sollimitfac",
         "factor by which to increase solution limit for heuristic pricing (1.0: add start limit)",
         &solverdata->sollimitfac, TRUE, DEFAULT_SOLLIMITFAC, 1.0, SCIPinfinity(origprob), NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(origprob, "pricingsolver/mip/settingsfile",
         "settings file for pricing problems",
         &solverdata->settingsfile, TRUE, DEFAULT_SETTINGSFILE, NULL, NULL) );


   return SCIP_OKAY;
}
