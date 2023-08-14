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

/**@file    solver_xyz.c
 * @ingroup PRICINGSOLVERS
 * @brief   xyz solver for pricing problems (put your description here)
 * @author  Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "solver_xyz.h"
#include "pricer_gcg.h"
#include "pub_solver.h"

#define DEC_CLASSIFIERNAME   "xyz pricing problem solver"    /**< name of pricing solver */
#define DEC_DESC             "pricing solver template"       /**< short description of classification */
#define SOLVER_PRIORITY      0                               /**< priority of this pricing solver */

#define SOLVER_HEURENABLED   TRUE            /**< indicates whether the heuristic solving method of the solver should be enabled */
#define SOLVER_EXACTENABLED  TRUE            /**< indicates whether the exact solving method of the solver should be enabled */

/*
 * Data structures
 */

/* TODO: fill in the necessary propagator data */

/** pricing solver data */
struct GCG_SolverData
{
};


/*
 * Local methods
 */

/* put your local methods here, and declare them static */


/*
 * Callback methods of propagator
 */

/* TODO: Implement all necessary propagator methods. The methods with an #ifdef SCIP_DISABLED_CODE ... #else #define ... are optional */

/** destructor of pricing solver to free user data (called when SCIP is exiting) */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVERFREE(solverFreeXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverFreeXyz NULL
#endif

/** solving process initialization method of pricing solver (called when branch and bound process is about to begin) */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVERINITSOL(solverInitsolXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverInitsolXyz NULL
#endif

/** solving process deinitialization method of pricing solver (called before branch and bound process data is freed) */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVEREXITSOL(solverExitsolXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverExitsolXyz NULL
#endif

/** initialization method of pricing solver (called after problem was transformed and solver is active) */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVERINIT(solverInitXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverInitXyz NULL
#endif

/** deinitialization method of pricing solver (called before transformed problem is freed and solver is active) */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVEREXIT(solverExitXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverExitXyz NULL
#endif

/** update method for pricing solver, used to update solver specific pricing problem data */
#ifdef SCIP_DISABLED_CODE
static
GCG_DECL_SOLVERUPDATE(solverUpdateXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define solverUpdateXyz NULL
#endif

/** solving method for pricing solver which solves the pricing problem to optimality */
static
GCG_DECL_SOLVERSOLVE(solverSolveXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}

/* solving method for pricing solver using heuristic pricing only */
static
GCG_DECL_SOLVERSOLVEHEUR(solverSolveHeurXyz)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of xyz pricing problem solver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}

/** creates the Xyz pricing solver and includes it in GCG */
SCIP_RETCODE GCGincludeSolverXyz(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   GCG_SOLVERDATA* solverdata;

   /* create xyz solver data */
   solverdata = NULL;
   /* TODO: (optional) create pricing problem solver specific data here */

   /* include pricing problem solver */
   SCIP_CALL( GCGpricerIncludeSolver(scip, SOLVER_NAME, SOLVER_DESC, SOLVER_PRIORITY,
         SOLVER_HEURENABLED, SOLVER_EXACTENABLED,
         solverUpdateXyz, solverSolveXyz, solverSolveHeurXyz,
         solverFreeXyz, solverInitXyz, solverExitXyz,
         solverInitsolXyz, solverExitsolXyz, solverdata) );

   /* add xyz propagator parameters */
   /* TODO: (optional) add propagator specific parameters with SCIPaddTypeParam() here */

   return SCIP_OKAY;
}
