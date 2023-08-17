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

/**@file   type_pricestore_gcg.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for storing priced cols
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GCG_TYPE_PRICESTORE_H__
#define __GCG_TYPE_PRICESTORE_H__

#ifdef __cplusplus
extern "C" {
#endif

/** possible settings for specifying the solution for which cuts are selected */
enum GCG_Efficiacychoice
{
   GCG_EFFICIACYCHOICE_DANTZIG = 0,          /**< use Dantzig's rule (reduced cost) to base efficacy on */
   GCG_EFFICIACYCHOICE_STEEPESTEDGE = 1,     /**< use steepest edge rule s( to base efficacy on */
   GCG_EFFICIACYCHOICE_LAMBDA = 2            /**< use lambda pricing to base efficacy on */
};
typedef enum GCG_Efficiacychoice GCG_EFFICIACYCHOICE;

typedef struct GCG_PriceStore GCG_PRICESTORE;     /**< storage for priced variables */

#ifdef __cplusplus
}
#endif

#endif
