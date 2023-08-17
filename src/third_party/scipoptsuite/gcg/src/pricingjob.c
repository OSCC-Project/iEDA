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

/**@file   pricingjob.c
 * @brief  methods for working with pricing jobs
 * @author Christian Puchert
 *
 * Various methods to work with pricing jobs
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "pricingjob.h"
#include "pub_gcgcol.h"
#include "pub_pricingjob.h"

#include "gcg.h"
#include "pricer_gcg.h"
#include "pub_pricingprob.h"
#include "pub_solver.h"

#include "scip/scip.h"


/** create a pricing job */
SCIP_RETCODE GCGpricingjobCreate(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGJOB**      pricingjob,         /**< pricing job to be created */
   GCG_PRICINGPROB*      pricingprob,        /**< data structure of the corresponding pricing problem */
   GCG_SOLVER*           solver,             /**< pricing solver responsible for the pricing job */
   int                   chunk               /**< chunk that the pricing problem should belong to */
)
{
   SCIP_CALL( SCIPallocBlockMemory(scip, pricingjob) );

   (*pricingjob)->pricingprob = pricingprob;
   (*pricingjob)->solver = solver;
   (*pricingjob)->chunk = chunk;
   (*pricingjob)->score = 0.0;
   (*pricingjob)->heuristic = FALSE;
   (*pricingjob)->nheuriters = 0;

   return SCIP_OKAY;
}

/** free a pricing job */
void GCGpricingjobFree(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGJOB**      pricingjob          /**< pricing job to be freed */
)
{
   SCIPfreeBlockMemory(scip, pricingjob);
   *pricingjob = NULL;
}

/** setup a pricing job at the beginning of the pricing loop */
SCIP_RETCODE GCGpricingjobSetup(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGJOB*       pricingjob,         /**< pricing job */
   SCIP_Bool             heuristic,          /**< shall the pricing job be performed heuristically? */
   int                   scoring,            /**< scoring parameter */
   int                   nroundscol,         /**< number of previous pricing rounds for which the number of improving columns should be counted */
   SCIP_Real             dualsolconv,        /**< dual solution value of corresponding convexity constraint */
   int                   npointsprob,        /**< total number of extreme points generated so far by the pricing problem */
   int                   nraysprob           /**< total number of extreme rays generated so far by the pricing problem */
   )
{
   GCG_PRICINGPROB* pricingprob = GCGpricingjobGetPricingprob(pricingjob);
   assert(pricingprob != NULL);

   /* set the score; the larger, the better */
   switch( scoring )
   {
   case 'i':
      pricingjob->score = - (SCIP_Real) GCGpricingprobGetProbnr(pricingprob);
      break;
   case 'd':
      pricingjob->score = dualsolconv;
      break;
   case 'r':
      pricingjob->score = -(0.2 * npointsprob + nraysprob);
      break;
   case 'l':
      pricingjob->score = (SCIP_Real) GCGpricingprobGetNColsLastRounds(pricingprob, nroundscol);
      break;
   default:
      pricingjob->score = 0.0;
      break;
   }

   GCGpricingjobResetSolver(scip, pricingjob);
   if( heuristic )
      GCGpricingjobResetHeuristic(pricingjob);
   else
      GCGpricingjobSetExact(pricingjob);

   return SCIP_OKAY;
}

/** get the pricing problem structure associated with a pricing job */
GCG_PRICINGPROB* GCGpricingjobGetPricingprob(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->pricingprob;
}
/** get the pricing solver with which the pricing job is to be performed */
GCG_SOLVER* GCGpricingjobGetSolver(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->solver;
}

/** reset the pricing solver to be used to the one with the highest priority */
void GCGpricingjobResetSolver(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   GCG_SOLVER** solvers;
   int nsolvers;

   int i;

   solvers = GCGpricerGetSolvers(scip);
   nsolvers = GCGpricerGetNSolvers(scip);

   /* get first available solver;
    * assumption: solvers are sorted by priority
    */
   pricingjob->solver = NULL;
   for( i = 0; i < nsolvers; ++i )
   {
      if( GCGsolverIsHeurEnabled(solvers[i]) || GCGsolverIsExactEnabled(solvers[i]) )
      {
         pricingjob->solver = solvers[i];
         break;
      }
   }

   assert(pricingjob->solver != NULL);
}

/** get the next pricing solver to be used, or NULL of there is none */
void GCGpricingjobNextSolver(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   GCG_SOLVER** solvers;
   int nsolvers;

   int pos;
   int i;

   solvers = GCGpricerGetSolvers(scip);
   nsolvers = GCGpricerGetNSolvers(scip);

   /* get position of current solver */
   for( pos = 0; pos < nsolvers; ++pos )
      if( solvers[pos] == pricingjob->solver )
         break;
   assert(pos < nsolvers);

   /* get next available solver;
    * assumption: solvers are sorted by priority
    */
   pricingjob->solver = NULL;
   for( i = pos + 1; i < nsolvers; ++i )
   {
      if( GCGsolverIsHeurEnabled(solvers[i]) || GCGsolverIsExactEnabled(solvers[i]) )
      {
         pricingjob->solver = solvers[i];
         break;
      }
   }
}

/** get the chunk of a pricing job */
SCIP_Real GCGpricingjobGetChunk(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->chunk;
}

/** get the score of a pricing job */
SCIP_Real GCGpricingjobGetScore(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->score;
}

/** return whether the pricing job is to be performed heuristically */
SCIP_Bool GCGpricingjobIsHeuristic(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->heuristic;
}

/** set the pricing job to be performed exactly */
void GCGpricingjobSetExact(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   pricingjob->heuristic = FALSE;
}

/** reset number of heuristic pricing iterations of a pricing job */
void GCGpricingjobResetHeuristic(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   if( GCGsolverIsHeurEnabled(pricingjob->solver) )
      pricingjob->heuristic = TRUE;
   else
      pricingjob->heuristic = FALSE;
   pricingjob->nheuriters = 0;
}

/** update number of heuristic pricing iterations of a pricing job */
void GCGpricingjobIncreaseNHeurIters(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   if( pricingjob->heuristic )
      ++pricingjob->nheuriters;
}

/** get the number of heuristic pricing iterations of the pricing job */
int GCGpricingjobGetNHeurIters(
   GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   )
{
   return pricingjob->nheuriters;
}
