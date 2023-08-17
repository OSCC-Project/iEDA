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

/**@file   solver_knapsack.c
 * @brief  knapsack solver for pricing problems
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "solver_knapsack.h"
#include "scip/cons_linear.h"
#include "scip/cons_knapsack.h"
#include "pricer_gcg.h"
#include "pub_solver.h"
#include "relax_gcg.h"
#include "pub_gcgcol.h"

#define SOLVER_NAME          "knapsack"
#define SOLVER_DESC          "knapsack solver for pricing problems"
#define SOLVER_PRIORITY      200

#define SOLVER_HEURENABLED   FALSE           /**< indicates whether the heuristic solving method of the solver should be enabled */
#define SOLVER_EXACTENABLED  TRUE            /**< indicates whether the exact solving method of the solver should be enabled */

/* knapsack pricing solver needs no solverdata */
/* struct GCG_SolverData {}; */


/*
 * Local methods
 */

/** solve the pricing problem as a knapsack problem, either exactly or approximately */
static
SCIP_RETCODE solveKnapsack(
   SCIP_Bool             exactly,            /**< should the pricing problem be solved to optimality or heuristically? */
   SCIP*                 scip,               /**< master problem SCIP data structure */
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   GCG_SOLVER*           solver,             /**< solver data structure */
   int                   probnr,             /**< problem number */
   SCIP_Real*            lowerbound,         /**< pointer to store lower bound */
   GCG_PRICINGSTATUS*    status              /**< pointer to store pricing problem status */
   )
{ /*lint -e715 */
   SCIP_CONS* cons;
   SCIP_CONSHDLR* conshdlr;
   SCIP_VAR** consvars;
   SCIP_Longint* consvals;
   int nconsvars;
   SCIP_VAR** solvars;
   SCIP_Real* solvals;
   int nsolvars;
   SCIP_VAR** pricingprobvars;
   int npricingprobvars;
   int nconss;

   int                   nitems;
   SCIP_Longint*         weights;
   SCIP_Real*            profits;
   SCIP_Real*            ubs;
   SCIP_Longint          capacity;
   SCIP_Longint          prelcapacity;
   int*                  items;
   int*                  solitems;
   int                   nsolitems;
   int*                  nonsolitems;
   int                   nnonsolitems;
   SCIP_Real             solval;
   SCIP_Bool             success;
   GCG_COL*              col;
   SCIP_Bool             inferbounds;
   int i;
   int j;
   int k;

   /* check preconditions */
   assert(scip != NULL);
   assert(pricingprob != NULL);
   assert(solver != NULL);
   assert(lowerbound != NULL);
   assert(status != NULL);

   assert(SCIPgetObjsense(pricingprob) == SCIP_OBJSENSE_MINIMIZE);

   pricingprobvars = SCIPgetVars(pricingprob);
   npricingprobvars = SCIPgetNVars(pricingprob);

   SCIPdebugMessage("Knapsack solver -- checking prerequisites\n");

   *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;

   /* check prerequisites: the pricing problem can be solved as a knapsack problem only if
    * - all variables are nonnegative integer variables
    * - there is only one constraint, which has infinite lhs and integer rhs
    */
   if( SCIPgetNBinVars(pricingprob) + SCIPgetNIntVars(pricingprob) < npricingprobvars )
   {
      SCIPdebugMessage("  -> pricing problem has continuous variables\n");
      return SCIP_OKAY;
   }
   for( i = SCIPgetNBinVars(pricingprob); i < SCIPgetNBinVars(pricingprob) + SCIPgetNIntVars(pricingprob); ++i )
   {
      if( SCIPisNegative(pricingprob, SCIPvarGetLbLocal(pricingprobvars[i])) )
      {
         SCIPdebugMessage("  -> pricing problem has variables with negative lower bounds\n");
         return SCIP_OKAY;
      }
   }

   nconss = SCIPgetNConss(pricingprob);
   if( nconss != 1 )
   {
      SCIPdebugMessage("  -> pricing problem has more than one constraint\n");
      return SCIP_OKAY;
   }

   cons = SCIPgetConss(pricingprob)[0];
   assert(cons != NULL);

   conshdlr = SCIPconsGetHdlr(cons);
   assert(conshdlr != NULL);

   consvals = NULL;

   /*
    * Check if the constraint is a knapsack constraint, and in that case,
    * get its variables, their coefficients as well as the capacity
    *
    * @note The constraint may be either of type 'linear' or 'knapsack';
    * the latter might be the case if the pricing problem has already been treated before in the loop
    * and if the constraint has therefore already been upgraded
    */
   if( strcmp(SCIPconshdlrGetName(conshdlr), "linear") == 0 )
   {
      SCIP_Real* realconsvals;

      if( !SCIPisIntegral(pricingprob, SCIPgetRhsLinear(pricingprob, cons)) ||
         !SCIPisInfinity(pricingprob, - SCIPgetLhsLinear(pricingprob, cons)) )
      {
         SCIPdebugMessage("  -> pricing constraint is bounded from below or has fractional rhs\n");
         return SCIP_OKAY;
      }

      consvars = SCIPgetVarsLinear(pricingprob, cons);
      realconsvals = SCIPgetValsLinear(pricingprob, cons);
      nconsvars = SCIPgetNVarsLinear(pricingprob, cons);

      SCIP_CALL( SCIPallocBufferArray(pricingprob, &consvals, nconsvars) );

      /* Check integrality of coefficients */
      for( i = 0; i < nconsvars; i++ )
      {
         if( !SCIPisIntegral(pricingprob, realconsvals[i]) )
         {
            SCIPdebugMessage("  -> pricing constraint has fractional coefficient\n");
            SCIPfreeBufferArray(pricingprob, &consvals);
            return SCIP_OKAY;
         }
         else
            consvals[i] = (SCIP_Longint) SCIPfloor(pricingprob, realconsvals[i]);
      }
      capacity = (SCIP_Longint) SCIPfloor(pricingprob, SCIPgetRhsLinear(pricingprob, cons));

      /* Check signs of variable coefficients in constraint and objective;
       * compute a preliminary capacity, used to deduce upper bounds for unbounded variables
       */
      prelcapacity = capacity;
      inferbounds = FALSE;
      for( i = 0; i < nconsvars; i++ )
      {
         if( SCIPisInfinity(pricingprob, SCIPvarGetUbLocal(consvars[i])) )
            inferbounds = TRUE;

         if( consvals[i] < 0 )
         {
            /* If a variable has an infinite upper bound, the capacity is not deducible */
            if( SCIPisInfinity(pricingprob, SCIPvarGetUbLocal(consvars[i])) )
            {
               SCIPdebugMessage("  -> variable with negative coefficient has no upper bound\n");
               SCIPfreeBufferArray(pricingprob, &consvals);
               return SCIP_OKAY;
            }

            /* increase capacity */
            prelcapacity -= (SCIP_Longint) SCIPfloor(pricingprob, consvals[i] * SCIPvarGetUbLocal(consvars[i]));
         }
      }

      SCIP_CALL( SCIPallocBufferArray(pricingprob, &ubs, nconsvars) );

      SCIPdebugMessage("Set variable upper bounds\n");

      /* infer upper bounds for unbounded variables */
      for( i = 0; i < nconsvars; i++ )
      {
         if( inferbounds && SCIPisInfinity(pricingprob, SCIPvarGetUbLocal(consvars[i])) )
         {
            ubs[i] = SCIPfloor(pricingprob, ABS((SCIP_Real)prelcapacity/consvals[i]));
            SCIPdebugMessage("  -> var <%s> %.2f/%"SCIP_LONGINT_FORMAT" = %.2f\n",
               SCIPvarGetName(consvars[i]), (SCIP_Real)prelcapacity, consvals[i], ubs[i]);
         }
         else
         {
            ubs[i] = SCIPvarGetUbLocal(consvars[i]);
            SCIPdebugMessage("  -> var <%s> %.2f\n", SCIPvarGetName(consvars[i]), ubs[i]);
         }

      }
   }
   else if( strcmp(SCIPconshdlrGetName(conshdlr), "knapsack") == 0 )
   {
      SCIP_Longint* consweights = SCIPgetWeightsKnapsack(pricingprob, cons);

      SCIPdebugMessage("Use knapsack solver – constraint is already of type 'knapsack'\n");

      consvars = SCIPgetVarsKnapsack(pricingprob, cons);
      nconsvars = SCIPgetNVarsKnapsack(pricingprob, cons);
      capacity = SCIPgetCapacityKnapsack(pricingprob, cons);

      SCIP_CALL( SCIPallocBufferArray(pricingprob, &consvals, nconsvars) );
      SCIP_CALL( SCIPallocBufferArray(pricingprob, &ubs, nconsvars) );
      BMScopyMemoryArray(consvals, consweights, nconsvars);

      for( i = 0; i < nconsvars; ++i )
      {
         assert(consvals[i] >= 0);
         assert(SCIPisGE(pricingprob, SCIPvarGetLbLocal(consvars[i]), 0.0));
         assert(SCIPisLE(pricingprob, SCIPvarGetUbLocal(consvars[i]), 1.0));

         ubs[i] = SCIPvarGetUbLocal(consvars[i]);
      }
   }
   else
   {
      SCIPdebugMessage("  -> constraint is of unknown type\n");
      return SCIP_OKAY;
   }

   *status = GCG_PRICINGSTATUS_UNKNOWN;

   /* Count number of knapsack items */
   SCIPdebugMessage("Count number of knapsack items:\n");
   nitems = 0;
   for( i = 0; i < nconsvars; i++ )
   {
      assert(!SCIPisInfinity(pricingprob, ubs[i]));
      SCIPdebugMessage("  -> <%s>: %d+%d\n", SCIPvarGetName(consvars[i]), nitems, (int)(ubs[i] - SCIPvarGetLbLocal(consvars[i]) + 0.5));
      nitems += (int)(ubs[i] - SCIPvarGetLbLocal(consvars[i]) + 0.5);
   }
   SCIPdebugMessage("-> %d items\n", nitems);

   SCIP_CALL( SCIPallocBufferArray(pricingprob, &solvars, npricingprobvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &solvals, npricingprobvars) );

   SCIP_CALL( SCIPallocBufferArray(pricingprob, &items, nitems) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &weights, nitems) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &profits, nitems) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &solitems, nitems) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &nonsolitems, nitems) );

   BMSclearMemoryArray(weights, nitems);

   /* Map variables to knapsack items, and set profits */
   SCIPdebugMessage("Set knapsack items\n");
   k = 0;
   for( i = 0; i < nconsvars; i++ )
   {
      assert(!SCIPisInfinity(pricingprob, ubs[i]));
      for( j = 0; j < (int)(ubs[i] - SCIPvarGetLbLocal(consvars[i]) + 0.5); ++j )
      {
         items[k] = i;
         profits[k] = - SCIPvarGetObj(consvars[i]);
         SCIPdebugMessage("  -> item %3d: <%s> (index %3d)\n", k, SCIPvarGetName(consvars[i]), i);

         k++;
      }
   }
   assert(k == nitems);

   /* Compute knapsack capacity, and set weights */
   SCIPdebugMessage("Compute knapsack capacity, current capacity = %"SCIP_LONGINT_FORMAT"\n", capacity);
   for( i = 0; i < nconsvars; i++ )
   {
      if( SCIPisEQ(pricingprob, SCIPvarGetUbLocal(consvars[i]), 0.0) )
         continue;
      if( SCIPisGE(pricingprob, SCIPvarGetLbLocal(consvars[i]), 1.0) )
      {
         SCIPdebugMessage("  -> variable <%s> has coeff %"SCIP_LONGINT_FORMAT" and lb %f --> increase capacity by %"SCIP_LONGINT_FORMAT"\n",
            SCIPvarGetName(consvars[i]), consvals[i], SCIPvarGetLbLocal(consvars[i]),
            (SCIP_Longint)SCIPfloor(pricingprob, SCIPvarGetLbLocal(consvars[i]) * consvals[i]));
         capacity -= (SCIP_Longint)SCIPfloor(pricingprob, SCIPvarGetLbLocal(consvars[i])) * consvals[i];
      }
   }

   SCIPdebugMessage("Compute weights\n");

   for( k = 0; k < nitems; k++ )
   {
      i = items[k];
      if( consvals[i] > 0 )
      {
         weights[k] = consvals[i];
         SCIPdebugMessage("  -> item %3d: weight = %"SCIP_LONGINT_FORMAT"\n", k, weights[k]);
      }
      else
      {
         capacity -= consvals[i];
         weights[k] = -consvals[i];
         profits[k] *= -1.0;

         SCIPdebugMessage("  -> item %3d: weight = %"SCIP_LONGINT_FORMAT" (negated from consval = %"SCIP_LONGINT_FORMAT")\n", k, weights[k], consvals[i]);
      }
   }

   SCIPdebugMessage("Knapsack capacity = %"SCIP_LONGINT_FORMAT"\n", capacity);

   success = TRUE;

   /* problem is infeasible */
   if( capacity < 0 )
   {
      SCIPdebugMessage("Pricing problem is infeasible\n");
      *status = GCG_PRICINGSTATUS_INFEASIBLE;
      goto TERMINATE;
   }
   else if( capacity == 0 )
   {
      SCIPdebugMessage("Knapsack has zero capacity\n");

      nsolitems = 0;
      nnonsolitems = nitems;
      for( i = 0; i < nitems; ++i )
         nonsolitems[i] = items[i];
   }
   else
   {
      SCIPdebugMessage("Solve pricing problem as knapsack\n");

      /* solve knapsack problem, all result pointers are needed! */
      if( exactly )
      {
         SCIP_CALL( SCIPsolveKnapsackExactly(pricingprob, nitems, weights, profits, capacity, items, solitems,
            nonsolitems, &nsolitems, &nnonsolitems, &solval, &success ));
      }
      else
      {
         SCIP_CALL( SCIPsolveKnapsackApproximately(pricingprob, nitems, weights, profits, capacity, items, solitems,
            nonsolitems, &nsolitems, &nnonsolitems, &solval ));
      }

      if( !success )
      {
         SCIPwarningMessage(pricingprob, "Knapsack solver could not solve pricing problem!");
         goto TERMINATE;
      }
      else if( exactly )
         *status = GCG_PRICINGSTATUS_OPTIMAL;

      SCIPdebugMessage("Knapsack solved, solval = %g\n", solval);
   }

   nsolvars = 0;

   for( i = 0; i < nsolitems; i++ )
   {
      assert(consvals[solitems[i]] >= 0 || !SCIPvarIsNegated(consvars[solitems[i]]));

      if( consvals[solitems[i]] >= 0 && !SCIPvarIsNegated(consvars[solitems[i]]) )
      {
         for( j = 0; j < nsolvars; ++j )
            if( solvars[j] == consvars[solitems[i]] )
               break;

         if( j == nsolvars )
         {
            solvars[j] = consvars[solitems[i]];
            solvals[j] = 1.0;
            ++nsolvars;
         }
         else
            solvals[j] += 1.0;
      }
   }

   for( i = 0; i < nnonsolitems; i++ )
   {
      assert(consvals[nonsolitems[i]] >= 0 || !SCIPvarIsNegated(consvars[nonsolitems[i]]));

      if( consvals[nonsolitems[i]] < 0 || SCIPvarIsNegated(consvars[nonsolitems[i]]) )
      {
         SCIP_VAR* solvar = SCIPvarIsNegated(consvars[nonsolitems[i]]) ? SCIPvarGetNegatedVar(consvars[nonsolitems[i]]) : consvars[nonsolitems[i]];

         for( j = 0; j < nsolvars; ++j )
            if( solvars[j] == solvar )
               break;

         if( j == nsolvars )
         {
            solvars[j] = solvar;
            solvals[j] = 1.0;
            ++nsolvars;
         }
         else
            solvals[j] += 1.0;
      }
   }

   for( i = 0; i < npricingprobvars; i++ )
   {
      if( SCIPisGE(pricingprob, SCIPvarGetLbLocal(pricingprobvars[i]), 1.0) )
      {
         for( j = 0; j < nsolvars; ++j )
            if( solvars[j] == pricingprobvars[i] )
               break;

         if( j == nsolvars )
         {
            solvars[j] = pricingprobvars[i];
            solvals[j] = SCIPfloor(pricingprob, SCIPvarGetLbLocal(pricingprobvars[i]));
            ++nsolvars;
         }
         else
            solvals[j] += SCIPfloor(pricingprob, SCIPvarGetLbLocal(pricingprobvars[i]));
      }
   }

   SCIP_CALL( GCGcreateGcgCol(pricingprob, &col, probnr, solvars, solvals, nsolvars, FALSE, SCIPinfinity(pricingprob)) );
   SCIP_CALL( GCGpricerAddCol(scip, col) );

   solval = 0.0;

   for( i = 0; i < nsolvars; ++i )
      solval += solvals[i] * SCIPvarGetObj(solvars[i]);

   *lowerbound = exactly ? solval : -SCIPinfinity(pricingprob);

 TERMINATE:
   SCIPfreeBufferArray(pricingprob, &nonsolitems);
   SCIPfreeBufferArray(pricingprob, &solitems);
   SCIPfreeBufferArray(pricingprob, &profits);
   SCIPfreeBufferArray(pricingprob, &weights);
   SCIPfreeBufferArray(pricingprob, &items);
   SCIPfreeBufferArray(pricingprob, &solvals);
   SCIPfreeBufferArray(pricingprob, &solvars);
   SCIPfreeBufferArray(pricingprob, &ubs);
   SCIPfreeBufferArray(pricingprob, &consvals);

   return SCIP_OKAY;
}

/*
 * Callback methods for pricing problem solver
 */

#define solverFreeKnapsack NULL
#define solverInitsolKnapsack NULL
#define solverExitsolKnapsack NULL
#define solverInitKnapsack NULL
#define solverExitKnapsack NULL
#define solverUpdateKnapsack NULL

/** exact solving method for knapsack solver */
static
GCG_DECL_SOLVERSOLVE(solverSolveKnapsack)
{  /*lint --e{715}*/

   /* solve the knapsack problem exactly */
   SCIP_CALL( solveKnapsack(TRUE, scip, pricingprob, solver, probnr, lowerbound, status) );

   return SCIP_OKAY;
}


/** heuristic solving method of knapsack solver */
static
GCG_DECL_SOLVERSOLVEHEUR(solverSolveHeurKnapsack)
{  /*lint --e{715}*/

   /* solve the knapsack problem approximately */
   SCIP_CALL( solveKnapsack(FALSE, scip, pricingprob, solver, probnr, lowerbound, status) );

   return SCIP_OKAY;
}


/** creates the knapsack solver for pricing problems and includes it in GCG */
SCIP_RETCODE GCGincludeSolverKnapsack(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( GCGpricerIncludeSolver(scip, SOLVER_NAME, SOLVER_DESC, SOLVER_PRIORITY,
         SOLVER_HEURENABLED, SOLVER_EXACTENABLED,
         solverUpdateKnapsack, solverSolveKnapsack, solverSolveHeurKnapsack,
         solverFreeKnapsack, solverInitKnapsack, solverExitKnapsack,
         solverInitsolKnapsack, solverExitsolKnapsack, NULL) );

   return SCIP_OKAY;
}
