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
/**@file   stat.h
 * @ingroup DECOMP
 * @brief  Prints information about the best decomposition
 * @author Alexander Gross
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef GCG_STAT_H__
#define GCG_STAT_H__

#include "scip/type_scip.h"
#include "scip/type_retcode.h"

#ifdef __cplusplus
extern "C" {
#endif

/** prints information about the best decomposition*/
extern
SCIP_RETCODE GCGwriteDecompositionData(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** prints additional information about the solving process */
extern
SCIP_RETCODE GCGwriteSolvingDetails(
   SCIP*                 scip                /**< SCIP data structure */
);

/** prints information about the creation of the Vars*/
extern
SCIP_RETCODE GCGwriteVarCreationDetails(
   SCIP*                 scip                /**< SCIP data structure */
);


#ifdef __cplusplus
}
#endif
#endif /* GCG_STAT_H_ */
