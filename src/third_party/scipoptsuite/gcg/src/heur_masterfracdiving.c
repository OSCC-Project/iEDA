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

/**@file   heur_masterfracdiving.c
 * @brief  master LP diving heuristic that chooses fixings w.r.t. the fractionalities
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_masterfracdiving.h"
#include "heur_masterdiving.h"


#define HEUR_NAME             "masterfracdiving"
#define HEUR_DESC             "master LP diving heuristic that chooses fixings w.r.t. the fractionalities"
#define HEUR_DISPCHAR         'f'
#define HEUR_PRIORITY         -1003000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          3
#define HEUR_MAXDEPTH         -1


/*
 * Callback methods
 */

/** variable selection method of diving heuristic;
 * finds best candidate variable w.r.t. fractionality:
 * - prefer variables that may not be rounded without destroying LP feasibility:
 *   - of these variables, round least fractional variable in corresponding direction
 * - if all remaining fractional variables may be rounded without destroying LP feasibility:
 *   - round variable with least increasing objective value
 * - binary variables are preferred
 */
static
GCG_DECL_DIVINGSELECTVAR(heurSelectVarMasterfracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_VAR** lpcands;
   SCIP_Real* lpcandssol;
   SCIP_Real* lpcandsfrac;
   int nlpcands;
   SCIP_Real bestobjgain;
   SCIP_Real bestfrac;
   SCIP_Bool bestcandmayrounddown;
   SCIP_Bool bestcandmayroundup;
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

   bestcandmayrounddown = TRUE;
   bestcandmayroundup = TRUE;
   bestobjgain = SCIPinfinity(scip);
   bestfrac = SCIP_INVALID;

   for( c = 0; c < nlpcands; ++c )
   {
      SCIP_VAR* var;
      SCIP_Bool mayrounddown;
      SCIP_Bool mayroundup;
      SCIP_Real frac;
      SCIP_Real obj;

      int i;

      var = lpcands[c];

      mayrounddown = SCIPvarMayRoundDown(var);
      mayroundup = SCIPvarMayRoundUp(var);
      frac = lpcandsfrac[c];
      obj = SCIPvarGetObj(var);

      /* if the variable is on the tabu list, do not choose it */
      for( i = 0; i < tabulistsize; ++i )
         if( tabulist[i] == var )
            break;
      if( i < tabulistsize )
         continue;

      if( mayrounddown || mayroundup )
      {
         /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
         if( bestcandmayrounddown || bestcandmayroundup )
         {
            SCIP_Real objgain;

            objgain = (1.0-frac)*obj;

            /* penalize too small fractions */
            if( ABS(1.0 - frac) < 0.01 )
               objgain *= 1000.0;

            /* prefer decisions on binary variables */
            if( !SCIPvarIsBinary(var) )
               objgain *= 1000.0;

            /* check, if candidate is new best candidate */
            if( SCIPisLT(scip, objgain, bestobjgain) || (SCIPisEQ(scip, objgain, bestobjgain) && frac > bestfrac) )
            {
               *bestcand = var;
               bestobjgain = objgain;
               bestfrac = frac;
               bestcandmayrounddown = mayrounddown;
               bestcandmayroundup = mayroundup;
            }
         }
      }
      else
      {
         /* penalize too small fractions */
         if( ABS(1.0-frac) < 0.01 )
            frac += 10.0;

         /* prefer decisions on binary variables */
         if( !SCIPvarIsBinary(var) )
            frac *= 1000.0;

         /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
         if( bestcandmayrounddown || bestcandmayroundup || frac > bestfrac )
         {
            *bestcand = var;
            bestfrac = frac;
            bestcandmayrounddown = FALSE;
            bestcandmayroundup = FALSE;
         }
         assert(bestfrac < SCIP_INVALID);
      }
   }

   *bestcandmayround = bestcandmayroundup || bestcandmayrounddown;

   return SCIP_OKAY;
}


/*
 * heuristic specific interface methods
 */

/** creates the masterfracdiving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeHeurMasterfracdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEUR* heur;

   /* include diving heuristic */
   SCIP_CALL( GCGincludeDivingHeurMaster(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, NULL, NULL, NULL, NULL, NULL, NULL, NULL, heurSelectVarMasterfracdiving, NULL) );

   assert(heur != NULL);

   return SCIP_OKAY;
}

