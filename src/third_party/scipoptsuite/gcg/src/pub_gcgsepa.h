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

/**@file   pub_gcgsepa.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for GCG separators
 * @author Christian Puchert
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_PUB_GCGSEPA_H__
#define GCG_PUB_GCGSEPA_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup SEPARATORS_PUB
 * @{
 */

/** sets separator parameters values to
 *
 *  - SCIP_PARAMSETTING_DEFAULT which are the default values of all separator parameters
 *  - SCIP_PARAMSETTING_FAST such that the time spent for separator is decreased
 *  - SCIP_PARAMSETTING_AGGRESSIVE such that the separator are called more aggressively
 *  - SCIP_PARAMSETTING_OFF which turns off all separators
 */
SCIP_RETCODE GCGsetSeparators(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMSETTING     paramsetting        /**< parameter settings */
   );

#ifdef __cplusplus
}

#endif
/** @} */
#endif
