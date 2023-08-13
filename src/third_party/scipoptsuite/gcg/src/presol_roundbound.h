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

/**@file   presol_roundbound.h
 * @ingroup PRESOLVERS
 * @brief  roundbound presolver: round fractional bounds on integer variables
 * @author Tobias Achterberg
 * @author Michael Bastubbe
 *
 * This presolver ensures that, all integral variables, for which the
 * bounds are fractional, will get rounded new bounds.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PRESOL_ROUNDBOUND_H__
#define __SCIP_PRESOL_ROUNDBOUND_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the roundbound presolver and includes it in SCIP */
SCIP_EXPORT
SCIP_RETCODE SCIPincludePresolRoundbound(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
