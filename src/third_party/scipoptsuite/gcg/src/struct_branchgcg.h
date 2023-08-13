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

/**@file   struct_branchgcg.h
 * @ingroup DATASTRUCTURES
 * @brief  data structures for branching rules
 * @author Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_STRUCT_BRANCHGCG_H__
#define GCG_STRUCT_BRANCHGCG_H__

#include "type_branchgcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/** branching rule */
struct GCG_Branchrule
{
   SCIP_BRANCHRULE*      branchrule;         /**< pointer to the SCIP branching rule */
   GCG_DECL_BRANCHACTIVEMASTER ((*branchactivemaster));     /**< node activation method of branching rule */
   GCG_DECL_BRANCHDEACTIVEMASTER ((*branchdeactivemaster)); /**< node deactivation method of branching rule */
   GCG_DECL_BRANCHPROPMASTER ((*branchpropmaster));         /**< propagation method of branching rule */
   GCG_DECL_BRANCHMASTERSOLVED((*branchmastersolved));      /**< lp solved method of branching rule */
   GCG_DECL_BRANCHDATADELETE ((*branchdatadelete));         /**< deinitialization method of branching rule */
};

#ifdef __cplusplus
}
#endif

#endif
