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

/**@file   heur_mastervecldiving.c
 * @brief  master LP diving heuristic that rounds variables with long column vectors
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_mastervecldiving.h"
#include "heur_masterdiving.h"


#define HEUR_NAME             "mastervecldiving"
#define HEUR_DESC             "master LP diving heuristic that rounds variables with long column vectors"
#define HEUR_DISPCHAR         'v'
#define HEUR_PRIORITY         -1003100
#define HEUR_FREQ             10
#define HEUR_FREQOFS          4
#define HEUR_MAXDEPTH         -1


/*
 * Callback methods
 */


/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. vector length:
 * - round variables in direction where objective value gets worse; for zero objective coefficient, round upwards
 * - round variable with least objective value deficit per row the variable appears in
 *   (we want to "fix" as many rows as possible with the least damage to the objective function)
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarMastervecldiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   SCIP_Real* lpcandsfrac;
   int nlpcands;
   SCIP_Real bestscore;
   int c;

   /* check preconditions */
   assert(scip != NULL);
   assert(heur != NULL);
   assert(bestcand != NULL);
   assert(bestcandmayround != NULL);

   /* get fractional variables that should be integral */
   SCIP_CALL( SCIPgetLPBranchCands(scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, NULL, NULL) );
   assert(lpcands != NULL);
   assert(lpcandsfrac != NULL);
   assert(lpcandssol != NULL);

   *bestcandmayround = TRUE;
   bestscore = SCIP_REAL_MAX;

   /* get best candidate */
   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;

      SCIP_Real obj;
      SCIP_Real objdelta;
      SCIP_Real frac;
      SCIP_Real score;
      int colveclen;

      int i;

      var = lpcands[c];

      /* if the variable is on the tabu list, do not choose it */
      for( i = 0; i < tabulistsize; ++i )
         if( tabulist[i] == var )
            break;
      if( i < tabulistsize )
         continue;

      frac = lpcandsfrac[c];
      obj = SCIPvarGetObj(var);
      objdelta = (1.0 - frac) * obj;

      colveclen = (SCIPvarGetStatus(var) == SCIP_VARSTATUS_COLUMN ? SCIPcolGetNNonz(SCIPvarGetCol(var)) : 0);

      /* check whether the variable is roundable */
      *bestcandmayround = *bestcandmayround && (SCIPvarMayRoundDown(var) || SCIPvarMayRoundUp(var));

      /* smaller score is better */
      score = (objdelta + SCIPsumepsilon(scip))/((SCIP_Real)colveclen+1.0);

      /* penalize negative scores (i.e. improvements in the objective) */
      if( score <= 0.0 )
         score *= 100.0;

      /* prefer decisions on binary variables */
      if( SCIPvarGetType(var) != SCIP_VARTYPE_BINARY )
         score *= 1000.0;

      /* check, if candidate is new best candidate */
      if( score < bestscore )
      {
         *bestcand = var;
         bestscore = score;
      }
   }

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the mastervecldiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurMastervecldiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEUR* heur;

   /* include diving heuristic */
   SCIP_CALL( GCGincludeDivingHeurMaster(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, NULL, NULL, NULL, NULL, NULL, NULL, NULL, heurSelectVarMastervecldiving, NULL) );

   assert(heur != NULL);

   return SCIP_OKAY;
}

