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

/**@file   heur_origdiving.h
 * @ingroup DIVINGHEURISTICS
 * @brief  primal heuristic interface for LP diving heuristics on the original variables
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_HEUR_ORIGDIVING_H__
#define GCG_HEUR_ORIGDIVING_H__


#include "scip/scip.h"
#include "type_origdiving.h"

#ifdef __cplusplus
extern "C" {
#endif

/** gets diving rule specific data of a diving heuristic */
SCIP_EXPORT
GCG_DIVINGDATA* GCGheurGetDivingDataOrig(
   SCIP_HEUR*               heur                    /**< primal heuristic */
   );

/** sets diving rule specific data of a diving heuristic */
SCIP_EXPORT
void GCGheurSetDivingDataOrig(
   SCIP_HEUR*               heur,                   /**< primal heuristic */
   GCG_DIVINGDATA*          divingdata              /**< diving rule specific data */
   );

/** creates an original diving heuristic and includes it in GCG */
SCIP_EXPORT
SCIP_RETCODE GCGincludeDivingHeurOrig(
   SCIP*                    scip,                   /**< SCIP data structure */
   SCIP_HEUR**              heur,                   /**< pointer to diving heuristic */
   const char*              name,                   /**< name of primal heuristic */
   const char*              desc,                   /**< description of primal heuristic */
   char                     dispchar,               /**< display character of primal heuristic */
   int                      priority,               /**< priority of the primal heuristic */
   int                      freq,                   /**< frequency for calling primal heuristic */
   int                      freqofs,                /**< frequency offset for calling primal heuristic */
   int                      maxdepth,               /**< maximal depth level to call heuristic at (-1: no limit) */
   GCG_DECL_DIVINGFREE      ((*divingfree)),        /**< destructor of diving heuristic */
   GCG_DECL_DIVINGINIT      ((*divinginit)),        /**< initialize diving heuristic */
   GCG_DECL_DIVINGEXIT      ((*divingexit)),        /**< deinitialize diving heuristic */
   GCG_DECL_DIVINGINITSOL   ((*divinginitsol)),     /**< solving process initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITSOL   ((*divingexitsol)),     /**< solving process deinitialization method of diving heuristic */
   GCG_DECL_DIVINGINITEXEC  ((*divinginitexec)),    /**< execution initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITEXEC  ((*divingexitexec)),    /**< execution deinitialization method of diving heuristic */
   GCG_DECL_DIVINGSELECTVAR ((*divingselectvar)),   /**< variable selection method of diving heuristic */
   GCG_DIVINGDATA*          divingdata              /**< diving rule specific data (or NULL) */
   );

/** creates event handler for origdiving event */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeEventHdlrOrigdiving(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
