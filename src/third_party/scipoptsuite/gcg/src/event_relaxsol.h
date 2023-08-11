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

/**@file   event_relaxsol.h
 * @ingroup EVENTS
 * @brief  eventhandler to update the relaxation solution in the original problem when the master LP has been solved
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_EVENT_RELAXSOL_H__
#define __SCIP_EVENT_RELAXSOL_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates event handler for relaxsol event */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeEventHdlrRelaxsol(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** return whether event has been triggered */
SCIP_EXPORT
SCIP_Bool GCGeventhdlrRelaxsolIsTriggered(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 masterprob          /**< the SCIP data structure for the master problem */
   );

#ifdef __cplusplus
}
#endif

#endif
