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

/**@file    branch_bpstrong.h
 * @brief   generic branch and price strong branching as described in
 *          Pecin, D., Pessoa, A., Poggi, M., Uchoa, E. Improved branch-cut-and-price for capacitated vehicle routing.
 *          In: Math. Prog. Comp. 9:61-100. Springer (2017).
 * @ingroup BRANCHINGRULES
 * @author  Oliver Gaul
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_BRANCH_BPSTRONG_H__
#define GCG_BRANCH_BPSTRONG_H__

#include "scip/scip.h"
#include "type_branchgcg.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** creates the xyz branching rule and includes it in SCIP */
extern SCIP_RETCODE SCIPincludeBranchruleBPStrong(
    SCIP *scip /**< SCIP data structure */
);

extern SCIP_RETCODE GCGbranchSelectCandidateStrongBranchingOrig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE       *origbranchrule,    /**< pointer storing original branching rule */
   SCIP_VAR              **branchvar,        /**< pointer to store output var pointer */
   SCIP_Bool             *upinf,             /**< pointer to store whether strong branching detected infeasibility in
                                                * the upbranch */
   SCIP_Bool             *downinf,           /**< pointer to store whether strong branching detected infeasibility in
                                                * the downbranch */
   SCIP_RESULT           *result,            /**< pointer to store result */
   SCIP_Bool             *stillusestrong     /**< pointer to store whether strong branching has reached a permanent
                                                * stopping condition for orig */
);

extern SCIP_RETCODE GCGbranchSelectCandidateStrongBranchingRyanfoster(
   SCIP*                 scip,               /**< original SCIP data structure */
   SCIP_BRANCHRULE*      rfbranchrule,       /**< Ryan-Foster branchrule */
   SCIP_VAR              **ovar1s,           /**< first elements of candidate pairs */
   SCIP_VAR              **ovar2s,           /**< second elements of candidate pairs */
   int                   *nspricingblock,    /**< pricing block numbers corresponding to input pairs */
   int                   npairs,             /**< number of input pairs */
   SCIP_VAR              **ovar1,            /**< pointer to store output var 1 pointer */
   SCIP_VAR              **ovar2,            /**< pointer to store output var 2 pointer */
   int                   *pricingblock,      /**< pointer to store output pricing block number */
   SCIP_Bool             *sameinf,           /**< pointer to store whether strong branching detected infeasibility in
                                                * the same branch */
   SCIP_Bool             *differinf,         /**< pointer to store whether strong branching detected infeasibility in
                                                * the differ branch */
   SCIP_RESULT           *result,            /**< pointer to store result */
   SCIP_Bool             *stillusestrong     /**< pointer to store whether strong branching has reached a permanent
                                                * stopping condition for Ryan-Foster */
);

#ifdef __cplusplus
}
#endif

#endif
