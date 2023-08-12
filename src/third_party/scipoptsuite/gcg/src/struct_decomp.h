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

/**@file   struct_decomp.h
 * @ingroup DECOMP
 * @ingroup DATASTRUCTURES
 * @brief  structure information for decomposition information in GCG projects
 * @author Martin Bergner
 * @author Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_STRUCT_DECOMP_H_
#define GCG_STRUCT_DECOMP_H_


#include "scip/scip.h"
#include "type_decomp.h"
#include "type_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

/** decomposition structure information */
struct DecDecomp
{
   SCIP_Bool             presolved;          /**< does the decomposition refer to the presolved problem? */
   int                   nblocks;            /**< number of blocks in this decomposition */
   SCIP_VAR***           subscipvars;        /**< two dimensional array of variables in each block
                                                  Usage: Usage: subscipvars[b][v] is variable v in block b */
   int*                  nsubscipvars;       /**< array of number of variables in each block
                                                  Usage: nsubscipvars[b] is the number of variables of block b */
   SCIP_CONS***          subscipconss;       /**< two dimensional array of constraints in each block
                                                  Usage: subscipcons[b][c] is constraint c in block b */
   int*                  nsubscipconss;      /**< array of number of constraints in each block
                                                  Usage: nsubscipcons[b] is the number of constraints of block b */
   SCIP_CONS**           linkingconss;       /**< array of constraints linking the blocks
                                                  Usage: linkingcons[c] is linking constraint number c */
   int                   nlinkingconss;      /**< number of linking constraints */
   SCIP_VAR**            linkingvars;        /**< array of variables linking the blocks
                                                  Usage: linkingvars[v] is linking variable number v */
   int                   nlinkingvars;       /**< number of linking variables */
   int                   nfixedlinkingvars;  /**< number of linking variables that are fixed */
   int                   nmastervars;        /**< number of linking variables that are purely master variables*/
   SCIP_VAR***           stairlinkingvars;   /**< array of variables staircaselinking the blocks */
   int*                  nstairlinkingvars;  /**< number of staircaselinking variables */
   SCIP_HASHMAP*         vartoblock;         /**< hashmap mapping variables to their blocks (from 1 to nblocks)
                                                  Usage: SCIPhashmapGetImage(vartoblock, var) returns b+1, where b is the block of
                                                    variable var. This map is somehow inverse of the subscipcvars array. */
   SCIP_HASHMAP*         constoblock;        /**< hashmap mapping constraints to their blocks (from 1 to nblocks)
                                                  Usage: Usage: SCIPhashmapGetImage(constoblock, cons) returns b+1, where b is the block of
                                                    constraint cons. This map is somehow inverse of the subscipconss array.*/
   SCIP_HASHMAP*         varindex;           /**< hashmap mapping variables to indices for a visual ordering */
   SCIP_HASHMAP*         consindex;          /**< hashmap mapping constraints to indices for visual ordering */
   DEC_DECTYPE           type;               /**< type of the decomposition */
   DEC_DETECTOR**        detectorchain;      /**< array of detectors that worked on this decomposition */
   DEC_DETECTOR*         detector;           /**< detector that found this decomposition */
   int                   sizedetectorchain;  /**< number of detectors that worked on this decomposition */
   int                   partialdecid;       /**< id of the partialdec this decomposition originates from */
   SCIP_Real*            detectorclocktimes; /**< times of the detectors that worked on this decomposition */
   SCIP_Real*            pctvarstoborder;    /**< percentages of variables assigned to the border of the corresponding detectors on this decomposition */
   SCIP_Real*            pctconsstoborder;   /**< percentages of constraints assigned to the border of the corresponding detectors on this decomposition */
   SCIP_Real*            pctvarstoblock;     /**< percentages of variables assigned to a block of the corresponding detectors on this decomposition */
   SCIP_Real*            pctconsstoblock;    /**< percentages of variables assigned to a block of the corresponding detectors on this decomposition */
   SCIP_Real*            pctvarsfromopen;    /**< percentages of variables assigned to a block or border of the corresponding detectors on this decomposition */
   SCIP_Real*            pctconssfromopen;   /**< percentages of constraints assigned to a block or the border of the corresponding detectors on this decomposition */
   int*                  nnewblocks;         /**< number of new blocks of the corresponding detectors on this decomposition */
   char*                 detectorchainstring;/**< string for the detector information working on that decomposition */
   SCIP_Real             maxwhitescore;      /**< maximum white score (i.e. [fraction of white area] (nonborder+nonblock)  ) */
};

#ifdef __cplusplus
}
#endif

#endif /* STRUCT_DECOMP_H_ */
