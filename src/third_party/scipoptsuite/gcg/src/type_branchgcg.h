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

/**@file   type_branchgcg.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for branching rules in GCG projects
 * @author Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_TYPE_BRANCHGCG_H__
#define GCG_TYPE_BRANCHGCG_H__

#include "scip/def.h"
#include "scip/type_result.h"
#include "scip/type_scip.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GCG_BranchData GCG_BRANCHDATA;   /**< branching data */
typedef struct GCG_Branchrule GCG_BRANCHRULE;   /**< branching rule */

/** type of variable bound: lower or upper bound */
enum GCG_BoundType
{
   GCG_BOUNDTYPE_LOWER = 0,            /**< lower bound */
   GCG_BOUNDTYPE_UPPER = 1,            /**< upper bound */
   GCG_BOUNDTYPE_FIXED = 2,            /**< variable fixed */
   GCG_BOUNDTYPE_NONE = 3              /**< no bound */
};
typedef enum GCG_BoundType GCG_BOUNDTYPE;

/** activation method for branchrule, called when a node in the master problem is activated,
 *  should perform changes to the current node's problem due to the branchdata
 *
 *  input:
 *  - scip            : SCIP main data structure of the master problem
 *  - branchdata      : the branching data
 */
#define GCG_DECL_BRANCHACTIVEMASTER(x) SCIP_RETCODE x (SCIP* scip, GCG_BRANCHDATA* branchdata)

/** deactivation method for branchrule, called when a node in the master problem is deactivated,
 *  should undo changes to the current node's problem due to the branchdata
 *
 *  input:
 *  - scip            : SCIP main data structure of the master problem
 *  - branchdata      : the branching data
 */
#define GCG_DECL_BRANCHDEACTIVEMASTER(x) SCIP_RETCODE x (SCIP* scip, GCG_BRANCHDATA* branchdata)

/** propagation method for branchrule, called when a node in the master problem is propagated,
 *  should perform propagation at the current node due to the branchdata
 *
 *  input:
 *  - scip            : SCIP main data structure of the master problem
 *  - branchdata      : the branching data
 *  - node            : the activated node
 *  - result          : pointer to store the result of the propagation call
 *
 *  possible return values for *result:
 *  - SCIP_CUTOFF     : the node is infeasible in the variable's bounds and can be cut off
 *  - SCIP_REDUCEDDOM : at least one domain reduction was found
 *  - SCIP_DIDNOTFIND : the propagator searched but did not find any domain reductions
 *  - SCIP_DIDNOTRUN  : the propagator was skipped
 *  - SCIP_DELAYED    : the propagator was skipped, but should be called again

 */
#define GCG_DECL_BRANCHPROPMASTER(x) SCIP_RETCODE x (SCIP* scip, GCG_BRANCHDATA* branchdata, SCIP_RESULT* result)

/** method for branchrule, called when the master LP is solved at one node,
 *  can store pseudocosts for the branching decisions
 *
 *  input:
 *  - scip            : SCIP main data structure of the original problem
 *  - branchdata      : the branching data
 *  - newlowerbound   : the new local lower bound
 *
 */
#define GCG_DECL_BRANCHMASTERSOLVED(x) SCIP_RETCODE x (SCIP* scip, GCG_BRANCHDATA* branchdata, SCIP_Real newlowerbound)

/** frees branching data of an origbranch constraint (called when the origbranch constraint is deleted)
 *
 *  input:
 *    scip            : SCIP main data structure of the original problem
 *    branchdata      : pointer to the branching data to free
 */
#define GCG_DECL_BRANCHDATADELETE(x) SCIP_RETCODE x (SCIP* scip, GCG_BRANCHDATA** branchdata)

#ifdef __cplusplus
}
#endif

#endif
