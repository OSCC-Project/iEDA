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

/**@file   solver.c
 * @brief  methods for GCG pricing solvers
 * @author Henri Lotze
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "pub_solver.h"
#include "solver.h"
#include "struct_solver.h"

#include "gcg.h"
#include "pricer_gcg.h"

#include <string.h>


/** compares two solvers w. r. t. their priorities */
SCIP_DECL_SORTPTRCOMP(GCGsolverComp)
{  /*lint --e{715}*/
   GCG_SOLVER* solver1 = (GCG_SOLVER*) elem1;
   GCG_SOLVER* solver2 = (GCG_SOLVER*) elem2;

   assert(solver1 != NULL);
   assert(solver2 != NULL);

   return solver2->priority - solver1->priority; /* prefer higher priorities */
}

/** creates a GCG pricing solver */
SCIP_RETCODE GCGsolverCreate(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER**          solver,             /**< pointer to pricing solver data structure */
   const char*           name,               /**< name of solver */
   const char*           desc,               /**< description of solver */
   int                   priority,           /**< priority of solver */
   SCIP_Bool             heurenabled,        /**< flag to indicate whether heuristic solving method of the solver is enabled */
   SCIP_Bool             exactenabled,        /**< flag to indicate whether exact solving method of the solver is enabled */
   GCG_DECL_SOLVERUPDATE((*solverupdate)),   /**< update method for solver */
   GCG_DECL_SOLVERSOLVE  ((*solversolve)),   /**< solving method for solver */
   GCG_DECL_SOLVERSOLVEHEUR((*solveheur)),   /**< heuristic solving method for solver */
   GCG_DECL_SOLVERFREE   ((*solverfree)),    /**< free method of solver */
   GCG_DECL_SOLVERINIT   ((*solverinit)),    /**< init method of solver */
   GCG_DECL_SOLVEREXIT   ((*solverexit)),    /**< exit method of solver */
   GCG_DECL_SOLVERINITSOL((*solverinitsol)), /**< initsol method of solver */
   GCG_DECL_SOLVEREXITSOL((*solverexitsol)), /**< exitsol method of solver */
   GCG_SOLVERDATA*       solverdata          /**< pricing solver data */
   )
{
   char paramname[SCIP_MAXSTRLEN];
   char paramdesc[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(solver != NULL);

   if( solveheur == NULL && solversolve == NULL )
   {
      SCIPwarningMessage(scip, "Solver <%s> has neither heuristic nor exact solving method and will not be included.\n", name);
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocMemory(scip, solver) ); /*lint !e866*/

   SCIP_ALLOC( BMSduplicateMemoryArray(&(*solver)->name, name, strlen(name)+1) );
   SCIP_ALLOC( BMSduplicateMemoryArray(&(*solver)->desc, desc, strlen(desc)+1) );

   (*solver)->solverupdate = solverupdate;
   (*solver)->solversolve = solversolve;
   (*solver)->solversolveheur = solveheur;
   (*solver)->solverfree = solverfree;
   (*solver)->solverinit = solverinit;
   (*solver)->solverexit = solverexit;
   (*solver)->solverinitsol = solverinitsol;
   (*solver)->solverexitsol = solverexitsol;
   (*solver)->solverdata = solverdata;

   SCIP_CALL( SCIPcreateCPUClock(scip, &((*solver)->optfarkasclock)) );
   SCIP_CALL( SCIPcreateCPUClock(scip, &((*solver)->optredcostclock)) );
   SCIP_CALL( SCIPcreateCPUClock(scip, &((*solver)->heurfarkasclock)) );
   SCIP_CALL( SCIPcreateCPUClock(scip, &((*solver)->heurredcostclock)) );

   (*solver)->optfarkascalls = 0;
   (*solver)->optredcostcalls = 0;
   (*solver)->heurfarkascalls = 0;
   (*solver)->heurredcostcalls = 0;

   if( solveheur != NULL )
   {
      (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "pricingsolver/%s/heurenabled", name);
      (void) SCIPsnprintf(paramdesc, SCIP_MAXSTRLEN, "flag to indicate whether heuristic solving method of solver <%s> is enabled", name);
      SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), paramname, paramdesc,
                     &((*solver)->heurenabled), FALSE, heurenabled, NULL, NULL));
   }
   else
      (*solver)->heurenabled = FALSE;

   if( solversolve != NULL )
   {
      (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "pricingsolver/%s/exactenabled", name);
      (void) SCIPsnprintf(paramdesc, SCIP_MAXSTRLEN, "flag to indicate whether exact solving method of solver <%s> is enabled", name);
      SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), paramname, paramdesc,
                     &((*solver)->exactenabled), FALSE, exactenabled, NULL, NULL));
   }
   else
      (*solver)->exactenabled = FALSE;

   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "pricingsolver/%s/priority", name);
   (void) SCIPsnprintf(paramdesc, SCIP_MAXSTRLEN, "priority of solver <%s>", name);
   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), paramname, paramdesc,
                  &((*solver)->priority), FALSE, priority, INT_MIN/4, INT_MAX/4, NULL, NULL));

   return SCIP_OKAY;
}

/** calls destructor and frees memory of GCG pricing solver */
SCIP_RETCODE GCGsolverFree(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER**          solver              /**< pointer to pricing solver data structure */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);
   assert(*solver != NULL);

   if( (*solver)->solverfree != NULL )
   {
      SCIP_CALL( (*solver)->solverfree(scip, *solver) );
   }

   BMSfreeMemoryArray(&(*solver)->name);
   BMSfreeMemoryArray(&(*solver)->desc);

   SCIP_CALL( SCIPfreeClock(scip, &((*solver)->optfarkasclock)) );
   SCIP_CALL( SCIPfreeClock(scip, &((*solver)->optredcostclock)) );
   SCIP_CALL( SCIPfreeClock(scip, &((*solver)->heurfarkasclock)) );
   SCIP_CALL( SCIPfreeClock(scip, &((*solver)->heurredcostclock)) );

   SCIPfreeMemory(scip, solver);

   return SCIP_OKAY;
}

/** initializes GCG pricing solver */
SCIP_RETCODE GCGsolverInit(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   SCIP_Bool resetstat;

   assert(scip != NULL);
   assert(solver != NULL);

   SCIP_CALL( SCIPgetBoolParam(scip, "misc/resetstat", &resetstat) );

   if( resetstat )
   {
      SCIP_CALL( SCIPresetClock(scip, solver->optfarkasclock) );
      SCIP_CALL( SCIPresetClock(scip, solver->optredcostclock) );
      SCIP_CALL( SCIPresetClock(scip, solver->heurfarkasclock) );
      SCIP_CALL( SCIPresetClock(scip, solver->heurredcostclock) );

      solver->optfarkascalls = 0;
      solver->optredcostcalls = 0;
      solver->heurfarkascalls = 0;
      solver->heurredcostcalls = 0;
   }

   if( solver->solverinit != NULL )
   {
      SCIP_CALL( solver->solverinit(scip, solver) );
   }

   return SCIP_OKAY;
}

/** calls exit method of GCG pricing solver */
SCIP_RETCODE GCGsolverExit(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   if( solver->solverexit != NULL )
   {
      SCIP_CALL( solver->solverexit(scip, solver) );
   }

   return SCIP_OKAY;
}

/** calls solving process initialization method of GCG pricing solver */
SCIP_RETCODE GCGsolverInitsol(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   if( solver->solverinitsol != NULL )
   {
      SCIP_CALL( solver->solverinitsol(scip, solver) );
   }

   return SCIP_OKAY;
}

/** calls solving process deinitialization method of GCG pricing solver */
SCIP_RETCODE GCGsolverExitsol(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   if( solver->solverexitsol != NULL )
   {
      SCIP_CALL( solver->solverexitsol(scip, solver) );
   }

   return SCIP_OKAY;
}

/** calls update method of GCG pricing solver */
SCIP_RETCODE GCGsolverUpdate(
   SCIP*                 pricingprob,        /**< the pricing problem that should be solved */
   GCG_SOLVER*           solver,             /**< pricing solver */
   int                   probnr,             /**< number of the pricing problem */
   SCIP_Bool             varobjschanged,     /**< have the objective coefficients changed? */
   SCIP_Bool             varbndschanged,     /**< have the lower and upper bounds changed? */
   SCIP_Bool             consschanged        /**< have the constraints changed? */
   )
{
   assert(pricingprob != NULL);
   assert(solver != NULL);

   if( solver->solverupdate != NULL )
   {
      SCIP_CALL( solver->solverupdate(pricingprob, solver, probnr, varobjschanged, varbndschanged, consschanged) );
   }

   return SCIP_OKAY;
}

/** calls heuristic or exact solving method of GCG pricing solver
 * @note This method has to be threadsafe!
 */
SCIP_RETCODE GCGsolverSolve(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   SCIP*                 pricingprob,        /**< the pricing problem that should be solved */
   GCG_SOLVER*           solver,             /**< pricing solver */
   SCIP_Bool             redcost,            /**< is reduced cost (TRUE) or Farkas (FALSE) pricing performed? */
   SCIP_Bool             heuristic,          /**< shall the pricing problem be solved heuristically? */
   int                   probnr,             /**< number of the pricing problem */
   SCIP_Real             dualsolconv,        /**< dual solution of the corresponding convexity constraint */
   SCIP_Real*            lowerbound,         /**< pointer to store lower bound of pricing problem */
   GCG_PRICINGSTATUS*    status,             /**< pointer to store the returned pricing status */
   SCIP_Bool*            solved              /**< pointer to store whether the solution method was called */
   )
{
   SCIP_CLOCK* clock;

   assert(scip != NULL);
   assert(pricingprob != NULL);
   assert(solver != NULL);
   assert(lowerbound != NULL);
   assert(status != NULL);

   *solved = FALSE;

   if( heuristic )
   {
      if( solver->heurenabled )
      {
         assert(solver->solversolveheur != NULL);

         if( redcost )
            clock = solver->heurredcostclock;
         else
            clock = solver->heurfarkasclock;

         #pragma omp critical (clock)
         {
            SCIP_CALL_ABORT( SCIPstartClock(scip, clock) );
         }

         SCIP_CALL( solver->solversolveheur(scip, pricingprob, solver, probnr, dualsolconv, lowerbound, status) );
         *solved = TRUE;

         #pragma omp critical (clock)
         {
            SCIP_CALL_ABORT( SCIPstopClock(scip, clock) );
         }
      }
   }
   else
   {
      if( solver->exactenabled )
      {
         assert(solver->solversolve != NULL);

         if( redcost )
            clock = solver->optredcostclock;
         else
            clock = solver->optfarkasclock;

         #pragma omp critical (clock)
         {
            SCIP_CALL_ABORT( SCIPstartClock(scip, clock) );
         }

         SCIP_CALL( solver->solversolve(scip, pricingprob, solver, probnr, dualsolconv, lowerbound, status) );
         *solved = TRUE;

         #pragma omp critical (clock)
         {
            SCIP_CALL_ABORT( SCIPstopClock(scip, clock) );
         }

      }
   }

   if( *status != GCG_PRICINGSTATUS_NOTAPPLICABLE && *solved )
   {
      if( redcost )
         if( heuristic )
            #pragma omp atomic
            ++solver->heurredcostcalls;
         else
            #pragma omp atomic
            ++solver->optredcostcalls;
      else
         if( heuristic )
            #pragma omp atomic
            ++solver->heurfarkascalls;
         else
            #pragma omp atomic
            ++solver->optfarkascalls;
   }

   return SCIP_OKAY;
}

/** gets user data of GCG pricing solver */
GCG_SOLVERDATA* GCGsolverGetData(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->solverdata;
}

/** sets user data of GCG pricing solver */
void GCGsolverSetData(
   GCG_SOLVER*           solver,             /**< pricing solver */
   GCG_SOLVERDATA*       solverdata          /**< pricing solver data */
   )
{
   assert(solver != NULL);

   solver->solverdata = solverdata;
}

/** gets name of GCG pricing solver */
const char* GCGsolverGetName(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->name;
}

/** gets description of GCG pricing solver */
const char* GCGsolverGetDesc(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);
   
   return solver->desc;
}

/** gets priority of GCG pricing solver */
int GCGsolverGetPriority(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);
   
   return solver->priority;
}

/** gets whether heuristic solving method of GCG pricing solver is enabled */
SCIP_Bool GCGsolverIsHeurEnabled(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->heurenabled;
}

/** gets whether exact solving method of GCG pricing solver is enabled */
SCIP_Bool GCGsolverIsExactEnabled(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);
   
   return solver->exactenabled;
}

/** gets number of exact Farkas pricing calls of pricing solver */
int GCGsolverGetOptFarkasCalls(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->optfarkascalls;
}

/** gets number of exact reduced cost pricing calls of pricing solver */
int GCGsolverGetOptRedcostCalls(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->optredcostcalls;
}

/** gets number of heuristic Farkas pricing calls of pricing solver */
int GCGsolverGetHeurFarkasCalls(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->heurfarkascalls;
}

/** gets number of heuristic reduced cost pricing calls of pricing solver */
int GCGsolverGetHeurRedcostCalls(
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(solver != NULL);

   return solver->heurredcostcalls;
}

/** gets exact Farkas pricing time of pricing solver */
SCIP_Real GCGsolverGetOptFarkasTime(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   return SCIPgetClockTime(scip, solver->optfarkasclock);
}

/** gets exact reduced cost pricing time of pricing solver */
SCIP_Real GCGsolverGetOptRedcostTime(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   return SCIPgetClockTime(scip, solver->optredcostclock);
}

/** gets heuristic Farkas pricing time of pricing solver */
SCIP_Real GCGsolverGetHeurFarkasTime(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   return SCIPgetClockTime(scip, solver->heurfarkasclock);
}

/** gets heuristic reduced cost pricing time of pricing solver */
SCIP_Real GCGsolverGetHeurRedcostTime(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   )
{
   assert(scip != NULL);
   assert(solver != NULL);

   return SCIPgetClockTime(scip, solver->heurredcostclock);
}
