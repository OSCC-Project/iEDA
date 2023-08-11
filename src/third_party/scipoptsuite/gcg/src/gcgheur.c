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

/**@file   gcgheur.c
 * @brief  public methods for GCG heuristics
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "gcg.h"
#include "pub_gcgheur.h"


/** resets the parameters to their default value */
static
SCIP_RETCODE setHeuristicsDefault(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);

   /* set specific parameters for LNS heuristics */
   SCIP_CALL( SCIPresetParam(scip, "heuristics/gcgrens/nodesofs") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/gcgrens/minfixingrate") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/gcgrins/nodesofs") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/gcgrins/minfixingrate") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/xpcrossover/nodesofs") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/xpcrossover/minfixingrate") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/xprins/nodesofs") );
   SCIP_CALL( SCIPresetParam(scip, "heuristics/xprins/minfixingrate") );

   return SCIP_OKAY;
}

/** sets the parameters to aggressive values */
static
SCIP_RETCODE setHeuristicsAggressive(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);

   /* set specific parameters for GCG RENS heuristic, if the heuristic is included */
#ifndef NDEBUG
   if( SCIPfindHeur(scip, "gcgrens") != NULL )
#endif
   {
      SCIP_CALL( SCIPsetLongintParam(scip, "heuristics/gcgrens/nodesofs", (SCIP_Longint)2000) );
      SCIP_CALL( SCIPsetRealParam(scip, "heuristics/gcgrens/minfixingrate", 0.3) );
   }

   /* set specific parameters for GCG RINS heuristic, if the heuristic is included */
#ifndef NDEBUG
   if( SCIPfindHeur(scip, "gcgrins") != NULL )
#endif
   {
      SCIP_CALL( SCIPsetIntParam(scip, "heuristics/gcgrins/nodesofs", 2000) );
      SCIP_CALL( SCIPsetRealParam(scip, "heuristics/gcgrins/minfixingrate", 0.3) );
   }

   /* set specific parameters for XP Crossover heuristic, if the heuristic is included */
#ifndef NDEBUG
   if( SCIPfindHeur(scip, "xpcrossover") != NULL )
#endif
   {
      SCIP_CALL( SCIPsetLongintParam(scip, "heuristics/xpcrossover/nodesofs", (SCIP_Longint)2000) );
      SCIP_CALL( SCIPsetRealParam(scip, "heuristics/xpcrossover/minfixingrate", 0.3) );
   }

   /* set specific parameters for XP RINS heuristic, if the heuristic is included */
#ifndef NDEBUG
   if( SCIPfindHeur(scip, "xprins") != NULL )
#endif
   {
      SCIP_CALL( SCIPsetLongintParam(scip, "heuristics/xprins/nodesofs", (SCIP_Longint)2000) );
      SCIP_CALL( SCIPsetRealParam(scip, "heuristics/xprins/minfixingrate", 0.3) );
   }

   return SCIP_OKAY;
}

/** sets the parameters to fast values */
static
SCIP_RETCODE setHeuristicsFast(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   char paramname[SCIP_MAXSTRLEN];
   int i;

#define NEXPENSIVEHEURS 11
   static const char* const expensiveheurs[NEXPENSIVEHEURS] = {
      "gcgcoefdiving",
      "gcgfeaspump",
      "gcgfracdiving",
      "gcgguideddiving",
      "gcglinesdiving",
      "gcgpscostdiving",
      "gcgrens",
      "gcgrins",
      "gcgveclendiving",
      "xpcrossover",
      "xprins"
   };

   assert(scip != NULL);

   SCIP_CALL( setHeuristicsDefault(scip) );

   /* explicitly turn off expensive heuristics, if included */
   for( i = 0; i < NEXPENSIVEHEURS; ++i )
   {
      (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/freq", expensiveheurs[i]);
      SCIP_CALL( SCIPsetIntParam(scip, paramname, -1) );
   }

   return SCIP_OKAY;
}

/** sets heuristic parameters values to
 *
 *  - SCIP_PARAMSETTING_DEFAULT which are the default values of all heuristic parameters
 *  - SCIP_PARAMSETTING_FAST such that the time spend for heuristic is decreased
 *  - SCIP_PARAMSETTING_AGGRESSIVE such that the heuristic are called more aggregative
 *  - SCIP_PARAMSETTING_OFF which turns off all heuristics
 */
SCIP_RETCODE GCGsetHeuristics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PARAMSETTING     paramsetting        /**< parameter settings */
   )
{
   assert(paramsetting == SCIP_PARAMSETTING_DEFAULT || paramsetting == SCIP_PARAMSETTING_FAST
      || paramsetting == SCIP_PARAMSETTING_AGGRESSIVE || paramsetting == SCIP_PARAMSETTING_OFF);

   switch( paramsetting )
   {
   case SCIP_PARAMSETTING_AGGRESSIVE:
      SCIP_CALL( setHeuristicsAggressive(scip) );
      break;
   case SCIP_PARAMSETTING_OFF:
      break;
   case SCIP_PARAMSETTING_FAST:
      SCIP_CALL( setHeuristicsFast(scip) );
      break;
   case SCIP_PARAMSETTING_DEFAULT:
      SCIP_CALL( setHeuristicsDefault(scip) );
      break;
   default:
      SCIPerrorMessage("The given paramsetting is invalid!\n");
      break;
   }

   return SCIP_OKAY;
}
