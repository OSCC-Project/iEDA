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

/**@file   sepa_master.h
 * @ingroup SEPARATORS
 * @brief  master separator
 * @author Gerald Gamrath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_SEPA_MASTER_H__
#define GCG_SEPA_MASTER_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the master separator and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeSepaMaster(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the array of original cuts saved in the separator data */
extern
SCIP_ROW** GCGsepaGetOrigcuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the number of cuts saved in the separator data */
extern
int GCGsepaGetNCuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** returns the array of master cuts saved in the separator data */
extern
SCIP_ROW** GCGsepaGetMastercuts(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** adds given original and master cut to master separator data */
extern
SCIP_RETCODE GCGsepaAddMastercuts(
   SCIP*                scip,               /**< SCIP data structure */
   SCIP_ROW*            origcut,            /**< pointer to orginal cut */
   SCIP_ROW*            mastercut           /**< pointer to master cut */
);

#ifdef __cplusplus
}
#endif

#endif
