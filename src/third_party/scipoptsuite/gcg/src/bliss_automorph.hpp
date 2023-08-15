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

/**@file    bliss_automorph.hpp
 * @brief   automorphism recognition (C++ interface)
 *
 * @author  Erik Muehmer
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_BLISS_AUTOMORPH_HPP
#define GCG_BLISS_AUTOMORPH_HPP

#include "class_partialdecomp.h"


/** compare two graphs w.r.t. automorphism */
SCIP_RETCODE cmpGraphPair(
   SCIP*                   scip,               /**< SCIP data structure */
   gcg::PARTIALDECOMP*     partialdec,         /**< partialdec the graphs should be compared for */
   int                     block1,             /**< index of first pricing prob */
   int                     block2,             /**< index of second pricing prob */
   SCIP_RESULT*            result,             /**< result pointer to indicate success or failure */
   SCIP_HASHMAP*           varmap,             /**< hashmap to save permutation of variables */
   SCIP_HASHMAP*           consmap,           /**< hashmap to save permutation of constraints */
   unsigned int            searchnodelimit,    /**< bliss search node limit (requires patched bliss version) */
   unsigned int            generatorlimit      /**< bliss generator limit (requires patched bliss version) */
);

#endif //GCG_BLISS_AUTOMORPH_HPP
