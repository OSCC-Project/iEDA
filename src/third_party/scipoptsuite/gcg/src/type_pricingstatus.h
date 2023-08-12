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

/**@file   type_pricingstatus.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for pricing status
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GCG_TYPE_PRICINGSTATUS_H__
#define __GCG_TYPE_PRICINGSTATUS_H__

#ifdef __cplusplus
extern "C" {
#endif

/** GCG pricing status */
enum GCG_PricingStatus
{
   GCG_PRICINGSTATUS_UNKNOWN        = 0,     /**< the pricing solver terminated with an unknown status;
                                              *   feasible columns might have been found, but it is not known
                                              *   whether the pricing problem has been solved to optimality
                                              */
   GCG_PRICINGSTATUS_NOTAPPLICABLE  = 1,     /**< the pricing solver can not be applied on the pricing problem */
   GCG_PRICINGSTATUS_SOLVERLIMIT    = 2,     /**< a solver specific limit was reached, and the solver can be called again */
   GCG_PRICINGSTATUS_OPTIMAL        = 3,     /**< the pricing problem was solved to optimality, an optimal solution is available */

   GCG_PRICINGSTATUS_INFEASIBLE     = 4,     /**< the problem was proven to be infeasible */
   GCG_PRICINGSTATUS_UNBOUNDED      = 5      /**< the problem was proven to be unbounded */
};
typedef enum GCG_PricingStatus GCG_PRICINGSTATUS;

#ifdef __cplusplus
}
#endif

#endif
