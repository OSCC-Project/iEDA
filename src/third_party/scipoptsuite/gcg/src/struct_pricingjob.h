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

/**@file   struct_pricingjob.h
 * @ingroup DATASTRUCTURES
 * @brief  data structure for pricing jobs
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_STRUCT_PRICINGJOB_H_
#define GCG_STRUCT_PRICINGJOB_H_

#include "scip/def.h"
#include "scip/type_misc.h"
#include "scip/scip.h"

#include "type_pricingjob.h"
#include "type_pricingprob.h"
#include "type_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/** pricing job data structure */
struct GCG_PricingJob
{
   /* problem data */
   GCG_PRICINGPROB*     pricingprob;        /**< data structure of the corresponding pricing problem */
   GCG_SOLVER*          solver;             /**< solver with which to solve the pricing problem */

   /* strategic parameters */
   int                  chunk;              /**< chunk the pricing job belongs to */
   SCIP_Real            score;              /**< current score of the pricing job */
   SCIP_Bool            heuristic;          /**< shall the pricing problem be solved heuristically? */
   int                  nheuriters;         /**< number of times the pricing job was performed heuristically */
};

#ifdef __cplusplus
}
#endif

#endif /* STRUCT_PRICINGJOB_H_ */
