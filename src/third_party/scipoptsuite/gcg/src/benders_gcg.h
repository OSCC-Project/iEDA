/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2022 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   benders_gcg.h
 * @ingroup BENDERS
 * @brief  GCG Benders' decomposition
 * @author Stephen J. Maher
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_BENDERS_GCG_H__
#define __SCIP_BENDERS_GCG_H__


#include "scip/scip.h"
#include "scip/bendersdefcuts.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the GCG Benders' decomposition and includes it in SCIP
 *
 *  @ingroup BendersIncludes
 */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeBendersGcg(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 origprob            /**< the SCIP instance of the original problem */
   );

/**@addtogroup BENDERS
 *
 * @{
 */

/** returns the last relaxation solution */
SCIP_EXPORT
SCIP_SOL* SCIPbendersGetRelaxSol(
   SCIP_BENDERS*         benders             /**< the Benders' decomposition structure */
   );

/** returns the original problem for the given master problem */
SCIP_EXPORT
SCIP* GCGbendersGetOrigprob(
   SCIP*                 masterprob          /**< the master problem SCIP instance */
   );

/** @} */

#ifdef __cplusplus
}
#endif

#endif
