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

/**@file   branch_orig.h
 * @brief  branching rule for original problem in GCG
 * @author Gerald Gamrath
 * @ingroup BRANCHINGRULES
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_BRANCH_ORIG_H__
#define GCG_BRANCH_ORIG_H__

#include "type_branchgcg.h"
#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the branching on original variable branching rule and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeBranchruleOrig(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** get the original variable on which the branching was performed */
extern
SCIP_VAR* GCGbranchOrigGetOrigvar(
   GCG_BRANCHDATA*       branchdata          /**< branching data */
   );

/** get the type of the new bound which resulted of the performed branching */
extern
GCG_BOUNDTYPE GCGbranchOrigGetBoundtype(
   GCG_BRANCHDATA*       branchdata          /**< branching data */
   );

/** get the new bound which resulted of the performed branching */
extern
SCIP_Real GCGbranchOrigGetNewbound(
   GCG_BRANCHDATA*       branchdata          /**< branching data */
   );

/** updates extern branching candidates before branching */
extern
SCIP_RETCODE GCGbranchOrigUpdateExternBranchcands(
   SCIP*                 scip               /**< SCIP data structure */
);

#ifdef __cplusplus
}
#endif

#endif
