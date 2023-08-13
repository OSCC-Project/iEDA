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

/**@file   type_masterdiving.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for GCG diving heuristics on the master variables
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_MASTERDIVING_H__
#define __SCIP_TYPE_MASTERDIVING_H__

#include "scip/type_scip.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GCG_DivingData GCG_DIVINGDATA;   /**< locally defined diving data */


/** destructor of diving heuristic to free user data (called when GCG is exiting)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGFREE(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** initialization method of diving heuristic (called after problem was transformed)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGINIT(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** deinitialization method of diving heuristic (called before transformed problem is freed)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGEXIT(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** solving process initialization method of diving heuristic (called when branch and bound process is about to begin)
 *
 *  This method is called when the presolving was finished and the branch and bound process is about to begin.
 *  The diving heuristic may use this call to initialize its branch and bound specific data.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGINITSOL(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed)
 *
 *  This method is called before the branch and bound process is freed.
 *  The diving heuristic should use this call to clean up its branch and bound data.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGEXITSOL(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** execution initialization method of diving heuristic (called when execution of diving heuristic is about to begin)
 *
 *  This method is called when the execution of the diving heuristic starts, before the diving loop.
 *  The diving heuristic may use this call to collect data which is specific to this call of the heuristic.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGINITEXEC(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** execution deinitialization method of diving heuristic (called when execution data is freed)
 *
 *  This method is called before the execution of the heuristic stops.
 *  The diving heuristic should use this call to clean up its execution specific data.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - heur            : the diving heuristic itself
 */
#define GCG_DECL_DIVINGEXITEXEC(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur)

/** variable selection method of diving heuristic
 *
 *  Selects an original variable to dive on
 *
 *  input:
 *  - scip             : SCIP main data structure
 *  - heur             : the diving heuristic itself
 *  - tabulist         : an array containing variables that must not be chosen
 *  - tabulistsize     : the size of the array
 *  - bestcand         : pointer to store the SCIP_VAR* returned by the selection rule
 *  - bestcandmayround : pointer to store whether the variable may be rounded without losing LP feasibility
 */
#define GCG_DECL_DIVINGSELECTVAR(x) SCIP_RETCODE x (SCIP* scip, SCIP_HEUR* heur, SCIP_VAR** tabulist, int tabulistsize, SCIP_VAR** bestcand, SCIP_Bool* bestcandmayround)

#ifdef __cplusplus
}
#endif

#endif
