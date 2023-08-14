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

/**@file    solver.h
 * @ingroup PRICING
 * @brief   private methods for GCG pricing solvers
 * @author  Henri Lotze
 * @author  Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_SOLVER_H_
#define GCG_SOLVER_H_

#include "type_solver.h"
#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup PRICING
 *
 * @{
 */

/** creates a GCG pricing solver */
SCIP_EXPORT
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
   );

/** calls destructor and frees memory of GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverFree(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER**          solver              /**< pointer to pricing solver data structure */
   );

/** initializes GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverInit(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   );

/** calls exit method of GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverExit(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   );

/** calls solving process initialization method of GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverInitsol(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   );

/** calls solving process deinitialization method of GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverExitsol(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVER*           solver              /**< pricing solver */
   );

/** calls update method of GCG pricing solver */
SCIP_EXPORT
SCIP_RETCODE GCGsolverUpdate(
   SCIP*                 pricingprob,        /**< the pricing problem that should be solved */
   GCG_SOLVER*           solver,             /**< pricing solver */
   int                   probnr,             /**< number of the pricing problem */
   SCIP_Bool             varobjschanged,     /**< have the objective coefficients changed? */
   SCIP_Bool             varbndschanged,     /**< have the lower and upper bounds changed? */
   SCIP_Bool             consschanged        /**< have the constraints changed? */
   );

/** calls heuristic or exact solving method of GCG pricing solver */
SCIP_EXPORT
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
   );
/**@} */
#ifdef __cplusplus
}
#endif

#endif
