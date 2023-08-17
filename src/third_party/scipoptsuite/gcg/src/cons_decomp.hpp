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

/**@file   cons_decomp.hpp
 * @brief  C++ interface of cons_decomp
 * @author Erik Muehmer
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CONS_DECOMP_HPP
#define GCG_CONS_DECOMP_HPP

#include "class_partialdecomp.h"

/** @brief gets vector of all partialdecs
 * @returns finished partialdecs
 */
extern
std::vector<gcg::PARTIALDECOMP*>* GCGconshdlrDecompGetPartialdecs(
   SCIP*          scip  /**< SCIP data structure */
);

extern
gcg::PARTIALDECOMP* DECgetPartialdecToWrite(
   SCIP*                         scip,
   SCIP_Bool                     transformed
);

/** @brief local method to find a partialdec for a given id or NULL if no partialdec with such id is found
 * @returns partialdec pointer of partialdec with given id or NULL if it does not exist
 * @note returns NULL if no partialdec by this id is known */
extern
gcg::PARTIALDECOMP* GCGconshdlrDecompGetPartialdecFromID(
   SCIP* scip,          /**< SCIP data structure */
   int partialdecid     /**< partialdec id */
);

/** @brief adds a preexisting partial dec to be considered at the beginning of the detection
 *
 * @note refines the partialdec to be consistent, adds meta data/statistics
 * @returns SCIP return code
*/
extern
SCIP_RETCODE GCGconshdlrDecompAddPreexisitingPartialDec(
   SCIP* scip,                   /**< SCIP data structure */
   gcg::PARTIALDECOMP* partialdec/**< partial dec to add */
);

/** @brief deregisters a partialdec in the conshdlr
 *
 * Use this function at deletion of the partialdec.
 * The partialdec is not destroyed in this function, the conshdlr will not know that it exists.
 */
extern
void GCGconshdlrDecompDeregisterPartialdec(
   SCIP* scip,                       /**< SCIP data structure */
   gcg::PARTIALDECOMP* partialdec    /**< the partialdec */
);

/** @brief registers a partialdec in the conshdlr
 *
 * Use this function at initialization of the partialdec.
 * If the partialdec already exists in the conshdlr it is ignored.
 */
extern
void GCGconshdlrDecompRegisterPartialdec(
   SCIP* scip,                       /**< SCIP data structure */
   gcg::PARTIALDECOMP* partialdec    /**< the partialdec to register */
);

/**
 * @brief help method to access detprobdata for unpresolved problem
 *
 * @returns pointer to detprobdata in wrapper data structure
 */
extern
gcg::DETPROBDATA* GCGconshdlrDecompGetDetprobdataOrig(
   SCIP*                 scip                 /**< SCIP data structure */
);

/**
 * @brief help method to access detprobdata for transformed problem
 *
 * @returns pointer to detprobdata in wrapper data structure
 */
extern
gcg::DETPROBDATA* GCGconshdlrDecompGetDetprobdataPresolved(
   SCIP*                 scip                 /**< SCIP data structure */
);

/**
 * @brief initilizes the candidates data structures with selected partialdecs
 *
 * initializes it with all if there are no selected partialdecs,
 * sort them according to the current scoretype
 * @param scip SCIP data structure
 * @param candidates tuples of partialdecs and scores will be added to this vector (sorted w.r.t. the scores).
 * @param original choose candidates for the original problem?
 * @param printwarnings should warnings be printed?
 * @returns SCIP return code
 */
extern
SCIP_RETCODE GCGconshdlrDecompChooseCandidatesFromSelected(
   SCIP* scip,
   std::vector<std::pair<gcg::PARTIALDECOMP*, SCIP_Real> >& candidates,
   SCIP_Bool original,
   SCIP_Bool printwarnings
   );

/** @brief gets detector history of partialdec with given id
 * @returns detector history of partialdec as string
 */
extern
std::string GCGconshdlrDecompGetDetectorHistoryByPartialdecId(
   SCIP* scip,    /**< SCIP data structure */
   int id         /**< id of partialdec */
   );

#endif //GCG_CONS_DECOMP_HPP
