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

/**@file    bliss_automorph.h
 * @brief   automorphism recognition of SCIPs
 *
 * @author  Martin Bergner
 * @author  Daniel Peters
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include "scip/type_scip.h"
#include "scip/type_result.h"
#include "scip/type_misc.h"

#ifndef BLISS_AUTOMORPH_H_
#define BLISS_AUTOMORPH_H_



#ifdef __cplusplus
extern "C" {
#endif

/** compare two graphs w.r.t. automorphism */
SCIP_RETCODE cmpGraphPair(
   SCIP*                 origscip,           /**< SCIP data structure */
   SCIP*                 scip1,              /**< first SCIP data structure to compare */
   SCIP*                 scip2,              /**< second SCIP data structure to compare */
   int                   prob1,              /**< index of first pricing prob */
   int                   prob2,              /**< index of second pricing prob */
   SCIP_RESULT*          result,             /**< result pointer to indicate success or failure */
   SCIP_HASHMAP*         varmap,             /**< hashmap to save permutation of variables */
   SCIP_HASHMAP*         consmap,            /**< hashmap to save permutation of constraints */
   unsigned int          searchnodelimit,    /**< bliss search node limit (requires patched bliss version) */
   unsigned int          generatorlimit      /**< bliss generator limit (requires patched bliss version) */
   );

#ifdef __cplusplus
}
#endif
#endif /* BLISS_AUTOMORPH_H_ */
