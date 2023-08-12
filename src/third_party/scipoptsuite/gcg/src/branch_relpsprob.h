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

/**@file   branch_relpsprob.h
 * @brief  reliable pseudo costs branching rule
 * @author Tobias Achterberg
 * @author Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_BRANCH_RELPSPROB_H__
#define GCG_BRANCH_RELPSPROB_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the reliable pseudo cost braching rule and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeBranchruleRelpsprob(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** execution reliability pseudo cost probing branching with the given branching candidates */
extern
SCIP_RETCODE SCIPgetRelpsprobBranchVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            branchcands,        /**< brancing candidates */
   SCIP_Real*            branchcandssol,     /**< solution value for the branching candidates */
   int                   nbranchcands,       /**< number of branching candidates */
   int                   nvars,              /**< number of variables to be watched by bdchgdata */
   SCIP_RESULT*          result,             /**< pointer to the result of the execution */
   SCIP_VAR**            branchvar           /**< pointer to the variable to branch on */
   );

#ifdef __cplusplus
}
#endif

#endif
