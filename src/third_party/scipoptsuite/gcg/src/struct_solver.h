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

/**@file   struct_solver.h
 * @ingroup DATASTRUCTURES
 * @brief  data structures for solvers
 * @author Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_STRUCT_SOLVER_H__
#define GCG_STRUCT_SOLVER_H__

#include "type_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/** pricing problem solver data structure */
struct GCG_Solver
{
   char*                 name;               /**< solver name */
   char*                 desc;               /**< solver description */
   int                   priority;           /**< solver priority */
   SCIP_Bool             heurenabled;        /**< switch for heuristic solving method */
   SCIP_Bool             exactenabled;       /**< switch for exact solving method */
   GCG_SOLVERDATA*       solverdata;         /**< private solver data structure */

   GCG_DECL_SOLVERFREE((*solverfree));       /**< destruction method */
   GCG_DECL_SOLVERINIT((*solverinit));       /**< initialization method */
   GCG_DECL_SOLVEREXIT((*solverexit));       /**< deinitialization method */
   GCG_DECL_SOLVERINITSOL((*solverinitsol)); /**< solving process initialization method */
   GCG_DECL_SOLVEREXITSOL((*solverexitsol)); /**< solving process deinitialization method */
   GCG_DECL_SOLVERUPDATE((*solverupdate));   /**< update method */
   GCG_DECL_SOLVERSOLVE((*solversolve));     /**< solving callback method */
   GCG_DECL_SOLVERSOLVEHEUR((*solversolveheur)); /**< heuristic solving callback method */

   SCIP_CLOCK*           optfarkasclock;     /**< optimal farkas pricing time */
   SCIP_CLOCK*           optredcostclock;    /**< optimal reduced cost pricing time */
   SCIP_CLOCK*           heurfarkasclock;    /**< heuristic farkas pricing time */
   SCIP_CLOCK*           heurredcostclock;   /**< heuristic reduced cost pricing time */
   int                   optfarkascalls;     /**< optimal farkas pricing calls */
   int                   optredcostcalls;    /**< optimal reduced cost pricing calls */
   int                   heurfarkascalls;    /**< heuristic farkas pricing calls */
   int                   heurredcostcalls;   /**< heuristic reduced cost pricing calls */
};


#ifdef __cplusplus
}
#endif

#endif
