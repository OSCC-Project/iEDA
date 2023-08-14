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

/**@file   class_stabilization.cpp
 * @brief  class with functions for dual variable smoothing
 * @author Martin Bergner
 * @author Jonas Witt
 * @author Michael Bastubbe
 *
 * This is an implementation of dynamic alpha-schedule (based on subgradient information) stabilization including an optional
 * combination with a subgradient method based on the papers
 *
 * Pessoa, A., Sadykov, R., Uchoa, E., & Vanderbeck, F. (2013). In-Out Separation and Column Generation
 * Stabilization by Dual Price Smoothing. In Experimental Algorithms (pp. 354-365). Springer Berlin Heidelberg.
 *
 * Pessoa, A., Sadykov, R., Uchoa, E., & Vanderbeck, F. (2016). Automation and combination of linear-programming
 * based stabilization techniques in column generation.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */
#include "class_stabilization.h"
#include "pricer_gcg.h"
#include "gcg.h"
#include "pub_gcgcol.h"
#include "sepa_master.h"
#include "objscip/objscip.h"
#include "scip/cons_linear.h"
#include "scip_misc.h"

namespace gcg {

Stabilization::Stabilization(
   SCIP* scip,
   PricingType* pricingtype_,
   SCIP_Bool hybridascent_
   ) :scip_(scip), stabcenterconsvals((SCIP_Real*) NULL), stabcenterconsvalssize(0), nstabcenterconsvals(0),
      stabcentercutvals((SCIP_Real*) NULL), stabcentercutvalssize(0), nstabcentercutvals(0),
      stabcenterlinkingconsvals((SCIP_Real*) NULL), nstabcenterlinkingconsvals(0), stabcenterlinkingconsvalssize(0),
      stabcenterconv((SCIP_Real*) NULL), nstabcenterconv(0), dualdiffnorm(0.0),
      subgradientconsvals(NULL), subgradientconsvalssize(0), nsubgradientconsvals(0),
      subgradientcutvals(NULL), subgradientcutvalssize(0), nsubgradientcutvals(0),
      subgradientlinkingconsvals(NULL), subgradientlinkingconsvalssize(0),
      subgradientnorm(0.0), hybridfactor(0.0),
      pricingtype(pricingtype_), alpha(0.8), alphabar(0.8), hybridascent(hybridascent_), beta(0.0), nodenr(-1), k(0), t(0), hasstabilitycenter(FALSE),stabcenterbound(-SCIPinfinity(scip)),
      inmispricingschedule(FALSE), subgradientproduct(0.0)
{

}

Stabilization::~Stabilization()
{
   SCIPfreeBlockMemoryArrayNull(scip_, &stabcenterconsvals, stabcenterconsvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &stabcentercutvals, stabcentercutvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &stabcenterlinkingconsvals, stabcenterlinkingconsvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &subgradientconsvals, subgradientconsvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &subgradientcutvals, subgradientcutvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &subgradientlinkingconsvals, subgradientlinkingconsvalssize); /*lint !e64*/
   SCIPfreeBlockMemoryArrayNull(scip_, &stabcenterconv, nstabcenterconv); /*lint !e64*/
   scip_ = (SCIP*) NULL;
   stabcenterconsvals = (SCIP_Real*) NULL;
   stabcentercutvals = (SCIP_Real*) NULL;
   stabcenterlinkingconsvals = (SCIP_Real*) NULL;
   stabcenterconv = (SCIP_Real*) NULL;
   pricingtype = (PricingType*) NULL;
   nodenr = -1;


}

SCIP_RETCODE Stabilization::updateStabcenterconsvals()
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   int nconss = GCGgetNMasterConss(origprob);

   if( nconss == nstabcenterconsvals )
   {
      return SCIP_OKAY;
   }

   if( nconss > stabcenterconsvalssize )
   {
      int oldsize = stabcenterconsvalssize;
      stabcenterconsvalssize = SCIPcalcMemGrowSize(scip_, nconss);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &stabcenterconsvals, oldsize, stabcenterconsvalssize) );
   }
   assert(stabcenterconsvals != NULL);
   BMSclearMemoryArray(&stabcenterconsvals[nstabcenterconsvals], (size_t)nconss-nstabcenterconsvals); /*lint !e866*/

   nstabcenterconsvals = nconss;

   return SCIP_OKAY;
}

SCIP_RETCODE Stabilization::updateStabcentercutvals()
{
   int ncuts = GCGsepaGetNCuts(scip_);

   if( ncuts == nstabcentercutvals )
   {
      return SCIP_OKAY;
   }

   if( ncuts > stabcentercutvalssize )
   {
      int oldsize = stabcentercutvalssize;
      stabcentercutvalssize = SCIPcalcMemGrowSize(scip_, ncuts);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &stabcentercutvals, oldsize, stabcentercutvalssize) );
   }
   assert(stabcentercutvals != NULL);
   BMSclearMemoryArray(&stabcentercutvals[nstabcentercutvals], (size_t)ncuts-nstabcentercutvals); /*lint !e866*/

   nstabcentercutvals = ncuts;

   return SCIP_OKAY;
}

SCIP_RETCODE Stabilization::updateSubgradientconsvals()
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   int nconss = GCGgetNMasterConss(origprob);

   if( nconss == nsubgradientconsvals )
   {
      return SCIP_OKAY;
   }

   if( nconss > subgradientconsvalssize )
   {
      int oldsize = subgradientconsvalssize;
      subgradientconsvalssize = SCIPcalcMemGrowSize(scip_, nconss);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &subgradientconsvals, oldsize, subgradientconsvalssize) );
   }
   assert(subgradientconsvals != NULL);
   BMSclearMemoryArray(&subgradientconsvals[nsubgradientconsvals], (size_t)nconss-nsubgradientconsvals); /*lint !e866*/

   nsubgradientconsvals = nconss;

   return SCIP_OKAY;
}

SCIP_RETCODE Stabilization::updateSubgradientcutvals()
{
   int ncuts = GCGsepaGetNCuts(scip_);

   if( ncuts == nsubgradientcutvals )
   {
      return SCIP_OKAY;
   }

   if( ncuts > subgradientcutvalssize )
   {
      int oldsize = subgradientcutvalssize;
      subgradientcutvalssize = SCIPcalcMemGrowSize(scip_, ncuts);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &subgradientcutvals, oldsize, subgradientcutvalssize) );
   }
   assert(subgradientcutvals != NULL);
   BMSclearMemoryArray(&subgradientcutvals[nsubgradientcutvals], (size_t)ncuts-nsubgradientcutvals); /*lint !e866*/

   nsubgradientcutvals = ncuts;

   return SCIP_OKAY;
}


SCIP_RETCODE Stabilization::setNLinkingconsvals(
   int nlinkingconssnew
   )
{
   if( nlinkingconssnew > stabcenterlinkingconsvalssize)
   {
      int newsize = SCIPcalcMemGrowSize(scip_, nlinkingconssnew);
      SCIP_CALL(SCIPreallocBlockMemoryArray(scip_, &stabcenterlinkingconsvals, stabcenterlinkingconsvalssize, newsize));


      if( hybridascent )
      {
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &subgradientlinkingconsvals, stabcenterlinkingconsvalssize,
            newsize) );
         BMSclearMemoryArray(subgradientlinkingconsvals, nlinkingconssnew);
      }
      stabcenterlinkingconsvalssize = newsize;
   }

   nstabcenterlinkingconsvals = nlinkingconssnew;
   BMSclearMemoryArray(stabcenterlinkingconsvals, nstabcenterlinkingconsvals);


   return SCIP_OKAY;
}

SCIP_RETCODE Stabilization::setNConvconsvals(
   int nconvconssnew
   )
{
   SCIPfreeBlockMemoryArrayNull(scip_, &stabcenterconv, nstabcenterconv); /*lint !e64*/
   SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &stabcenterconv, nconvconssnew) );

   nstabcenterconv = nconvconssnew;
   BMSclearMemoryArray(stabcenterconv, nstabcenterconv);

   return SCIP_OKAY;
}

SCIP_Real Stabilization::linkingconsGetDual(
   int i
   )
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_Real subgradient = 0.0;

   assert(i < nstabcenterlinkingconsvals);
   assert(nstabcenterlinkingconsvals<= GCGgetNVarLinkingconss(origprob));
   assert(stabcenterlinkingconsvals != NULL);

   SCIP_CONS* cons = GCGgetVarLinkingconss(origprob)[i];

   if( hybridascent && hasstabilitycenter )
      subgradient = subgradientlinkingconsvals[i];

   return computeDual(stabcenterlinkingconsvals[i], pricingtype->consGetDual(scip_, cons), subgradient, 0.0, 0.0);
}

SCIP_RETCODE Stabilization::consGetDual(
   int                   i,                  /* index of the constraint */
   SCIP_Real*            dual                /* return pointer for dual value */
)
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_Real subgradient = 0.0;
#ifndef NDEBUG
   int nconss =  GCGgetNMasterConss(origprob);
#endif
   assert(i < nconss);
   assert(dual != NULL);

   SCIP_CONS* cons = GCGgetMasterConss(origprob)[i];

   if( i >= nstabcenterconsvals )
      SCIP_CALL( updateStabcenterconsvals() );

   assert(i < nstabcenterconsvals);
   assert(stabcenterconsvals != NULL);

   if( i >= nsubgradientconsvals && hybridascent )
      SCIP_CALL( updateSubgradientconsvals() );

   if( hybridascent && hasstabilitycenter )
      subgradient = subgradientconsvals[i];

   *dual = computeDual(stabcenterconsvals[i], pricingtype->consGetDual(scip_, cons), subgradient, SCIPgetLhsLinear(scip_, cons), SCIPgetRhsLinear(scip_, cons));
   return SCIP_OKAY;

}

SCIP_RETCODE Stabilization::rowGetDual(
   int                   i,                  /* index of the row */
   SCIP_Real*            dual                /* return pointer for dual value */
)
{
#ifndef NDEBUG
   int nrows = GCGsepaGetNCuts(scip_);
#endif
   assert(i < nrows);
   assert(dual != NULL);

   SCIP_ROW* row = GCGsepaGetMastercuts(scip_)[i];
   SCIP_Real subgradient = 0.0;

   if( i >= nstabcentercutvals )
      SCIP_CALL( updateStabcentercutvals() );

   assert(i < nstabcentercutvals);
   assert(stabcentercutvals != NULL);

   if( i >= nsubgradientcutvals && hybridascent )
      SCIP_CALL( updateSubgradientcutvals() );

   if( hybridascent && hasstabilitycenter )
   {
      assert(subgradientcutvals != NULL);
      subgradient = subgradientcutvals[i];
   }

   *dual = computeDual(stabcentercutvals[i], pricingtype->rowGetDual(row), subgradient, SCIProwGetLhs(row), SCIProwGetRhs(row));

   return SCIP_OKAY;
}

SCIP_Real Stabilization::convGetDual(
   int i
   )
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);

   assert(i < nstabcenterconv);
   assert(nstabcenterconv<= GCGgetNPricingprobs(origprob));
   assert(stabcenterconv != NULL);

   SCIP_CONS* cons = GCGgetConvCons(origprob, i);
   SCIP_Real subgradient = 0.0;

   return computeDual(stabcenterconv[i], pricingtype->consGetDual(scip_, cons), subgradient, (SCIP_Real) GCGgetNIdenticalBlocks(origprob, i), (SCIP_Real) GCGgetNIdenticalBlocks(origprob, i));
}

SCIP_RETCODE Stabilization::updateStabilityCenter(
   SCIP_Real             lowerbound,         /**< lower bound due to lagrange function corresponding to current (stabilized) dual vars */
   SCIP_Real*            dualsolconv,        /**< corresponding feasible dual solution for convexity constraints */
   GCG_COL**             pricingcols         /**< columns of the pricing problems */
   )
{
   assert(dualsolconv != NULL);
   SCIPdebugMessage("Updating stability center: ");

   /* in case the bound is not improving and we have a stability center, do nothing */
   if( SCIPisLE(scip_, lowerbound, stabcenterbound) && hasstabilitycenter )
   {
      SCIPdebugPrintf("no bound increase: %g <= %g\n", lowerbound, SCIPnodeGetLowerbound(SCIPgetCurrentNode(scip_)));
      return SCIP_OKAY;
   }

   SCIPdebugPrintf("bound increase: %g > %g\n", lowerbound, SCIPnodeGetLowerbound(SCIPgetCurrentNode(scip_)));

   /* first update the arrays */
   SCIP_CALL( updateStabcenterconsvals() );
   SCIP_CALL( updateStabcentercutvals() );

   if( hybridascent )
   {
      SCIP_CALL( updateSubgradientconsvals() );
      SCIP_CALL( updateSubgradientcutvals() );
   }

   /* get new dual values */
   SCIP* origprob = GCGmasterGetOrigprob(scip_);

   int nconss = GCGgetNMasterConss(origprob);
   int ncuts = GCGsepaGetNCuts(scip_);
   int nprobs = GCGgetNPricingprobs(origprob);

   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   assert(nconss <= nstabcenterconsvals);
   assert(ncuts <= nstabcentercutvals);

   for( int i = 0; i < nconss; ++i )
   {
      SCIP_CALL( consGetDual(i, &stabcenterconsvals[i]) );
   }

   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_CALL( rowGetDual(i, &stabcentercutvals[i]) );
   }

   for( int i = 0; i < nstabcenterlinkingconsvals; ++i)
   {
      stabcenterlinkingconsvals[i] = linkingconsGetDual(i);
   }

   for( int i = 0; i < nprobs; ++i )
   {
      if(!GCGisPricingprobRelevant(origprob, i))
         continue;
      stabcenterconv[i] = dualsolconv[i];
   }

   if( hybridascent )
      calculateSubgradient(pricingcols);

   hasstabilitycenter = TRUE;
   stabcenterbound = lowerbound;

   return SCIP_OKAY;
}

SCIP_Real Stabilization::computeDual(
      SCIP_Real         center,
      SCIP_Real         current,
      SCIP_Real         subgradient,         /**< subgradient (or 0.0 if not needed) */
      SCIP_Real         lhs,                 /**< lhs (or 0.0 if not needed) */
      SCIP_Real         rhs                  /**< rhs (or 0.0 if not needed) */
      ) const
{
   SCIP_Real usedalpha = alpha;
   SCIP_Real usedbeta = beta;

   if ( inmispricingschedule )
   {
      usedalpha = alphabar;
      usedbeta = 0.0;
   }

   if( hasstabilitycenter && (SCIPisZero(scip_, usedbeta) || SCIPisZero(scip_, usedalpha)) )
      return usedalpha*center+(1.0-usedalpha)*current;
   else if( hasstabilitycenter && SCIPisPositive(scip_, usedbeta) )
   {
      SCIP_Real dual = center + hybridfactor * (beta * (center + subgradient * dualdiffnorm / subgradientnorm) + (1.0 - beta) * current - center);

      /* make sure dual solution has the correct sign */
      if( SCIPisInfinity(scip_, rhs) )
         dual = MAX(dual, 0.0);
      else if( SCIPisInfinity(scip_, -lhs) )
         dual = MIN(dual, 0.0);

      return dual;
   }
   else
      return current;
}

void Stabilization::updateIterationCount()
{
   ++t;
}

void Stabilization::updateIterationCountMispricing()
{
   ++k;
}

void Stabilization::updateNode()
{
   if( nodenr != SCIPnodeGetNumber(SCIPgetCurrentNode(scip_)) )
   {
      nodenr = SCIPnodeGetNumber(SCIPgetCurrentNode(scip_));
      k = 0;
      t = 1;
      alpha = 0.8;
      hasstabilitycenter = FALSE;
      stabcenterbound = -SCIPinfinity(scip_);
      inmispricingschedule = FALSE;
   }
}

/**< update information for hybrid stabilization with dual ascent */
SCIP_RETCODE Stabilization::updateHybrid()
{
   if( hasstabilitycenter && hybridascent && !inmispricingschedule )
   {
      /* first update the arrays */
      SCIP_CALL( updateStabcenterconsvals() );
      SCIP_CALL( updateStabcentercutvals() );

      SCIP_CALL( updateSubgradientconsvals() );
      SCIP_CALL( updateSubgradientcutvals() );

      if( SCIPisPositive(scip_, alpha) )
      {
         calculateDualdiffnorm();
         calculateBeta();
         calculateHybridFactor();
      }
   }

   return SCIP_OKAY;
}

void Stabilization::updateAlphaMisprice()
{
   SCIPdebugMessage("Alphabar update after mispricing\n");
   updateIterationCountMispricing();
   alphabar = MAX(0.0, 1-k*(1-alpha));
   SCIPdebugMessage("alphabar updated to %g in mispricing iteration k=%d and node pricing iteration t=%d \n", alphabar, k, t);
}

void Stabilization::updateAlpha()
{
   SCIPdebugMessage("Alpha update after successful pricing\n");
   updateIterationCount();

   /* There is a sign error in the stabilization paper:
    * if the scalar product (subgradientproduct) is positive, the angle is less than 90° and we want to decrease alpha
    */
   if( SCIPisNegative(scip_, subgradientproduct) )
   {
      increaseAlpha();
   }
   else
   {
      decreaseAlpha();
   }

}

void Stabilization::increaseAlpha()
{
   /* to avoid numerical problems, we assure alpha <= 0.9 */
   alpha = MIN(0.9, alpha+(1-alpha)*0.1);

   SCIPdebugMessage("alpha increased to %g\n", alpha);
}

void Stabilization::decreaseAlpha()
{
   alpha = MAX(0.0, alpha-0.1);

   SCIPdebugMessage("alpha decreased to %g\n", alpha);
}

SCIP_Real Stabilization::calculateSubgradientProduct(
   GCG_COL**            pricingcols         /**< solutions of the pricing problems */
   )
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_CONS** origmasterconss = GCGgetOrigMasterConss(origprob);
   SCIP_CONS** masterconss = GCGgetMasterConss(origprob);
   SCIP_CONS** linkingconss = GCGgetVarLinkingconss(origprob);
   int nlinkingconss = GCGgetNVarLinkingconss(origprob);
   int* linkingconsblocks = GCGgetVarLinkingconssBlock(origprob);
   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   int nconss = GCGgetNMasterConss(origprob);
   assert(nconss <= nstabcenterconsvals);
   SCIP_ROW** mastercuts = GCGsepaGetMastercuts(scip_);
   SCIP_ROW** origmastercuts = GCGsepaGetOrigcuts(scip_);
   int ncuts = GCGsepaGetNCuts(scip_);
   assert(ncuts <= nstabcentercutvals);

   SCIP_Real gradientproduct = 0.0;

   /* masterconss */
   for( int i = 0; i < nconss; ++i )
   {
      SCIP_VAR** vars = NULL;
      SCIP_Real* vals = NULL;
      int nvars;
      SCIP_Real lhs; /* can also be rhs, but we need only one */

      SCIP_CONS* origcons = origmasterconss[i];
      nvars = GCGconsGetNVars(origprob, origcons);
      SCIP_CALL( SCIPallocBufferArray(origprob, &vars, nvars) );
      SCIP_CALL( SCIPallocBufferArray(origprob, &vals, nvars) );
      GCGconsGetVars(origprob, origcons, vars, nvars);
      GCGconsGetVals(origprob, origcons, vals, nvars);

      SCIP_Real dual =  pricingtype->consGetDual(scip_, masterconss[i]);
      SCIP_Real stabdual;

      SCIP_CALL( consGetDual(i, &stabdual) );

      assert(!SCIPisInfinity(scip_, ABS(dual)));

      if( SCIPisFeasPositive(scip_, stabdual) )
      {
         lhs = GCGconsGetLhs(origprob, origcons);
      }
      else if( SCIPisFeasNegative(scip_, stabdual) )
      {
         lhs = GCGconsGetRhs(origprob, origcons);
      }
      else
      {
         SCIPfreeBufferArray(origprob, &vals);
         SCIPfreeBufferArray(origprob, &vars);
         continue;
      }

      for( int j = 0; j < nvars; ++j )
      {
         SCIP_Real val = 0.0;
         assert(GCGvarIsOriginal(vars[j]));
         if( GCGvarGetBlock(vars[j]) < 0 )
         {
            SCIP_VAR* mastervar = GCGoriginalVarGetMastervars(vars[j])[0];
            assert(GCGvarIsMaster(mastervar));
            val = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
            assert( !SCIPisInfinity(scip_, val) );
         }
         else
         {
            int block = GCGvarGetBlock(vars[j]);
            if( !GCGisPricingprobRelevant(origprob, block) )
               continue;

            assert(pricingcols[block] != NULL);

            SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(vars[j]);
            assert(GCGvarIsPricing(pricingvar));
            SCIP* pricingprob = GCGgetPricingprob(origprob, block);
            assert(pricingprob != NULL);
            val = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         assert(stabcenterconsvals != NULL);
         assert(vals != NULL);
         gradientproduct -= (dual - stabcenterconsvals[i]) * vals[j] * val;
      }

      assert(stabcenterconsvals != NULL);
      assert(!SCIPisInfinity(scip_, ABS(lhs)));

      gradientproduct += (dual - stabcenterconsvals[i]) * lhs;
      SCIPfreeBufferArray(origprob, &vals);
      SCIPfreeBufferArray(origprob, &vars);
   }

   /* mastercuts */
   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_COL** cols;
      SCIP_Real* vals;
      int nvars;
      SCIP_Real lhs; /* can also be rhs, but we need only one */

      SCIP_ROW* origcut = origmastercuts[i];
      nvars = SCIProwGetNNonz(origcut);
      cols = SCIProwGetCols(origcut);
      vals = SCIProwGetVals(origcut);

      SCIP_Real dual = pricingtype->rowGetDual(mastercuts[i]);
      assert(!SCIPisInfinity(scip_, ABS(dual)));

      SCIP_Real stabdual;

      SCIP_CALL( rowGetDual(i, &stabdual) );

      if( SCIPisFeasGT(scip_, stabdual, 0.0) )
      {
         lhs = SCIProwGetLhs(origcut);
      }
      else if( SCIPisFeasLT(scip_, stabdual, 0.0) )
      {
         lhs = SCIProwGetRhs(origcut);
      }
      else
      {
         continue;
      }
      for( int j = 0; j < nvars; ++j )
      {
         SCIP_Real val = 0.0;
         SCIP_VAR* var = SCIPcolGetVar(cols[j]);
         assert(GCGvarIsOriginal(var));

         /* Linking or master variable */
         if( GCGvarGetBlock(var) < 0 )
         {
            SCIP_VAR* mastervar = GCGoriginalVarGetMastervars(var)[0];
            assert(GCGvarIsMaster(mastervar));
            val = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         /* Variable in a pricing problem */
         else
         {
            int block = GCGvarGetBlock(var);
            if( !GCGisPricingprobRelevant(origprob, block) )
               continue;

            assert(pricingcols[block] != NULL);

            SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(var);
            assert(GCGvarIsPricing(pricingvar));
            SCIP* pricingprob = GCGgetPricingprob(origprob, block);
            assert(pricingprob != NULL);
            val = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         assert(stabcentercutvals != NULL);
         assert(vals != NULL);
         gradientproduct -= (dual - stabcentercutvals[i]) * vals[j] * val;
      }

      assert(!SCIPisInfinity(scip_, ABS(lhs)));
      assert(stabcentercutvals != NULL);

      gradientproduct +=  (dual - stabcentercutvals[i]) * lhs;
   }

   /* linkingconss */
   for( int i = 0; i < nlinkingconss; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_VAR* pricingvar;
      SCIP_CONS* linkingcons = linkingconss[i];
      int block = linkingconsblocks[i];
      mastervar = SCIPgetVarsLinear(scip_, linkingcons)[0];
      assert(GCGvarIsMaster(mastervar));

      pricingvar = GCGlinkingVarGetPricingVars(GCGmasterVarGetOrigvars(mastervar)[0])[block];
      assert(GCGvarIsPricing(pricingvar));
      SCIP* pricingprob = GCGgetPricingprob(origprob, block);
      assert(pricingprob != NULL);

      assert(stabcenterlinkingconsvals != NULL);
      SCIP_Real dual = pricingtype->consGetDual(scip_, linkingcons) - stabcenterlinkingconsvals[i];

      SCIP_Real stabdual = linkingconsGetDual(i);

      if( SCIPisFeasZero(origprob, stabdual) )
         continue;

      assert(pricingcols[block] != NULL);

      SCIP_Real masterval = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
      SCIP_Real pricingval = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
      assert(!SCIPisInfinity(scip_, ABS(masterval)));
      assert(!SCIPisInfinity(scip_, ABS(pricingval)));
      assert(!SCIPisInfinity(scip_, ABS(dual)));
      gradientproduct -= dual * (masterval - pricingval);
   }

   SCIPdebugMessage("Update gradient product with value %g.\n", gradientproduct);

   return gradientproduct;
}

/** calculates the subgradient (with linking variables) */
void Stabilization::calculateSubgradient(
   GCG_COL**            pricingcols         /**< columns of the pricing problems */
)
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_CONS** origmasterconss = GCGgetOrigMasterConss(origprob);

   SCIP_CONS** linkingconss = GCGgetVarLinkingconss(origprob);
   int nlinkingconss = GCGgetNVarLinkingconss(origprob);
   int* linkingconsblocks = GCGgetVarLinkingconssBlock(origprob);
   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   int nconss = GCGgetNMasterConss(origprob);
   assert(nconss <= nstabcenterconsvals);
   SCIP_ROW** origmastercuts = GCGsepaGetOrigcuts(scip_);
   int ncuts = GCGsepaGetNCuts(scip_);
   assert(ncuts <= nstabcentercutvals);

   subgradientnorm = 0.0;

   /* masterconss */
   for( int i = 0; i < nconss; ++i )
   {
      SCIP_VAR** vars = NULL;
      SCIP_Real* vals = NULL;
      int nvars;
      SCIP_Real activity;
      SCIP_Real infeasibility;

      SCIP_CONS* origcons = origmasterconss[i];
      nvars = GCGconsGetNVars(origprob, origcons);
      SCIPallocBufferArray(origprob, &vars, nvars);
      SCIPallocBufferArray(origprob, &vals, nvars);
      GCGconsGetVars(origprob, origcons, vars, nvars);
      GCGconsGetVals(origprob, origcons, vals, nvars);

      SCIP_Real dual = stabcenterconsvals[i];
      assert(!SCIPisInfinity(scip_, ABS(dual)));

      activity = 0.0;

      for( int j = 0; j < nvars; ++j )
      {
         SCIP_Real val = 0.0;
         assert(GCGvarIsOriginal(vars[j]));
         if( GCGvarGetBlock(vars[j]) < 0 )
         {
            SCIP_VAR* mastervar = GCGoriginalVarGetMastervars(vars[j])[0];
            assert(GCGvarIsMaster(mastervar));
            val = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
            assert( !SCIPisInfinity(scip_, val) );
         }
         else
         {
            int block = GCGvarGetBlock(vars[j]);
            if( !GCGisPricingprobRelevant(origprob, block) )
               continue;

            assert(pricingcols[block] != NULL);

            SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(vars[j]);
            assert(GCGvarIsPricing(pricingvar));
            SCIP* pricingprob = GCGgetPricingprob(origprob, block);
            assert(pricingprob != NULL);
            val = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         assert(vals != NULL);
         activity += vals[j] * val;
      }

      infeasibility = 0.0;

      if( SCIPisFeasPositive(scip_, dual) /* || SCIPisInfinity(origprob, SCIPgetRhsLinear(origprob, origcons)) */)
      {
         infeasibility = GCGconsGetLhs(origprob, origcons) - activity;
      }
      else if( SCIPisFeasNegative(scip_, dual) /* || SCIPisInfinity(origprob, SCIPgetLhsLinear(origprob, origcons)) */)
      {
         infeasibility = GCGconsGetRhs(origprob, origcons) - activity;
      }

      assert(subgradientconsvals != NULL);
      assert(!SCIPisInfinity(scip_, SQR(infeasibility)));

      subgradientconsvals[i] = infeasibility;

      if( SCIPisPositive(scip_, SQR(infeasibility)) )
         subgradientnorm += SQR(infeasibility);

      SCIPfreeBufferArray(origprob, &vals);
      SCIPfreeBufferArray(origprob, &vars);
   }

   /* mastercuts */
   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_COL** cols;
      SCIP_Real* vals;
      int nvars;
      SCIP_Real activity;
      SCIP_Real infeasibility;

      SCIP_ROW* origcut = origmastercuts[i];
      nvars = SCIProwGetNNonz(origcut);
      cols = SCIProwGetCols(origcut);
      vals = SCIProwGetVals(origcut);

      activity = 0.0;

      SCIP_Real dual = stabcentercutvals[i];
      assert(!SCIPisInfinity(scip_, ABS(dual)));
      for( int j = 0; j < nvars; ++j )
      {
         SCIP_Real val = 0.0;
         SCIP_VAR* var = SCIPcolGetVar(cols[j]);
         assert(GCGvarIsOriginal(var));

         /* Linking or master variable */
         if( GCGvarGetBlock(var) < 0 )
         {
            SCIP_VAR* mastervar = GCGoriginalVarGetMastervars(var)[0];
            assert(GCGvarIsMaster(mastervar));
            val = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         /* Variable in a pricing problem */
         else
         {
            int block = GCGvarGetBlock(var);
            if( !GCGisPricingprobRelevant(origprob, block) )
               continue;

            assert(pricingcols[block] != NULL);

            SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(var);
            assert(GCGvarIsPricing(pricingvar));
            SCIP* pricingprob = GCGgetPricingprob(origprob, block);
            assert(pricingprob != NULL);
            val = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
            assert(!SCIPisInfinity(scip_, ABS(val)));
         }
         assert(stabcentercutvals != NULL);
         assert(vals != NULL);
         activity += vals[j] * val;
      }

      infeasibility = 0.0;

      if( SCIPisFeasPositive(scip_, dual) )
      {
         infeasibility = SCIProwGetLhs(origcut) - activity;
      }
      else if( SCIPisFeasNegative(scip_, dual) )
      {
         infeasibility = SCIProwGetRhs(origcut) - activity;
      }

      assert(subgradientcutvals != NULL);
      assert(!SCIPisInfinity(scip_, SQR(infeasibility)));

      subgradientcutvals[i] = infeasibility;

      if( SCIPisPositive(scip_, SQR(infeasibility)) )
         subgradientnorm += SQR(infeasibility);
   }

   /* linkingconss */
   for( int i = 0; i < nlinkingconss; ++i )
   {
      SCIP_VAR* mastervar;
      SCIP_VAR* pricingvar;
      SCIP_CONS* linkingcons = linkingconss[i];
      int block = linkingconsblocks[i];
      SCIP_Real activity;
      SCIP_Real infeasibility;
      mastervar = SCIPgetVarsLinear(scip_, linkingcons)[0];
      assert(GCGvarIsMaster(mastervar));

      pricingvar = GCGlinkingVarGetPricingVars(GCGmasterVarGetOrigvars(mastervar)[0])[block];
      assert(GCGvarIsPricing(pricingvar));
      SCIP* pricingprob = GCGgetPricingprob(origprob, block);
      assert(pricingprob != NULL);

      assert(pricingcols[block] != NULL);

      assert(stabcenterlinkingconsvals != NULL);
      SCIP_Real masterval = SCIPgetSolVal(scip_, (SCIP_SOL*) NULL, mastervar);
      SCIP_Real pricingval = GCGcolGetSolVal(pricingprob, pricingcols[block], pricingvar);
      assert(!SCIPisInfinity(scip_, ABS(masterval)));
      assert(!SCIPisInfinity(scip_, ABS(pricingval)));
      activity = (masterval - pricingval);

      infeasibility = activity;

      assert(subgradientlinkingconsvals != NULL);
      assert(!SCIPisInfinity(scip_, SQR(infeasibility)));

      subgradientlinkingconsvals[i] = infeasibility;

      if( SCIPisPositive(scip_, SQR(infeasibility)) )
         subgradientnorm += SQR(infeasibility);
   }


   assert(!SCIPisNegative(scip_, subgradientnorm));

   subgradientnorm = SQRT(subgradientnorm);

   SCIPdebugMessage("Update subgradient and subgradientnorm with value %g.\n", subgradientnorm);
}

/**< calculate norm of difference between stabcenter and current duals */
void Stabilization::calculateDualdiffnorm()
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_CONS** masterconss = GCGgetMasterConss(origprob);
   SCIP_CONS** linkingconss = GCGgetVarLinkingconss(origprob);
   int nlinkingconss = GCGgetNVarLinkingconss(origprob);
   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   int nconss = GCGgetNMasterConss(origprob);
   assert(nconss <= nstabcenterconsvals);
   SCIP_ROW** mastercuts = GCGsepaGetMastercuts(scip_);
   int ncuts = GCGsepaGetNCuts(scip_);
   assert(ncuts <= nstabcentercutvals);

   dualdiffnorm = 0.0;

   /* masterconss */
   assert(stabcenterconsvals != NULL);

   for( int i = 0; i < nconss; ++i )
   {
      SCIP_Real dualdiff = SQR(stabcenterconsvals[i] - pricingtype->consGetDual(scip_, masterconss[i]));

      if( SCIPisPositive(scip_, dualdiff) )
         dualdiffnorm += dualdiff;
   }

   /* mastercuts */
   assert(stabcenterconsvals != NULL);

   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_Real dualdiff = SQR(stabcentercutvals[i] - pricingtype->rowGetDual(mastercuts[i]));

      if( SCIPisPositive(scip_, dualdiff) )
         dualdiffnorm += dualdiff;
   }

   /* linkingconss */
   assert(stabcenterlinkingconsvals != NULL);

   for( int i = 0; i < nlinkingconss; ++i )
   {
      SCIP_Real dualdiff = SQR(stabcenterlinkingconsvals[i] - pricingtype->consGetDual(scip_, linkingconss[i]));

      if( SCIPisPositive(scip_, dualdiff) )
         dualdiffnorm += dualdiff;
   }

   dualdiffnorm = SQRT(dualdiffnorm);
   SCIPdebugMessage("Update dualdiffnorm with value %g.\n", dualdiffnorm);
}

/**< calculate beta */
void Stabilization::calculateBeta()
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_CONS** masterconss = GCGgetMasterConss(origprob);
   SCIP_CONS** linkingconss = GCGgetVarLinkingconss(origprob);
   int nlinkingconss = GCGgetNVarLinkingconss(origprob);
   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   int nconss = GCGgetNMasterConss(origprob);
   assert(nconss <= nstabcenterconsvals);
   SCIP_ROW** mastercuts = GCGsepaGetMastercuts(scip_);
   int ncuts = GCGsepaGetNCuts(scip_);
   assert(ncuts <= nstabcentercutvals);

   beta = 0.0;

   /* masterconss */
   assert(stabcenterconsvals != NULL);

   for( int i = 0; i < nconss; ++i )
   {
      SCIP_Real dualdiff = ABS(pricingtype->consGetDual(scip_, masterconss[i]) - stabcenterconsvals[i]);
      SCIP_Real product = dualdiff * ABS(subgradientconsvals[i]);

      if( SCIPisPositive(scip_, product) )
         beta += product;
   }

   /* mastercuts */
   assert(stabcentercutvals != NULL || ncuts == 0);

   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_Real dualdiff = ABS(pricingtype->rowGetDual(mastercuts[i]) - stabcentercutvals[i]);
      SCIP_Real product = dualdiff * ABS(subgradientcutvals[i]);

      if( SCIPisPositive(scip_, product) )
         beta += product;
   }

   /* linkingconss */
   assert(stabcenterlinkingconsvals != NULL);

   for( int i = 0; i < nlinkingconss; ++i )
   {
      SCIP_Real dualdiff = ABS(pricingtype->consGetDual(scip_, linkingconss[i]) - stabcenterlinkingconsvals[i]);
      SCIP_Real product = dualdiff * ABS(subgradientlinkingconsvals[i]);

      if( SCIPisPositive(scip_, product) )
         beta += product;
   }

   if( SCIPisPositive(scip_, subgradientnorm) )
      beta = beta / (subgradientnorm * dualdiffnorm);

   SCIPdebugMessage("Update beta with value %g.\n", beta);

   assert( ( SCIPisPositive(scip_, beta) || SCIPisZero(scip_, subgradientnorm)) && SCIPisLE(scip_, beta, 1.0) );
}

/**< calculate factor that is needed in hybrid stabilization */
void Stabilization::calculateHybridFactor()
{
   SCIP* origprob = GCGmasterGetOrigprob(scip_);
   SCIP_CONS** masterconss = GCGgetMasterConss(origprob);

   SCIP_CONS** linkingconss = GCGgetVarLinkingconss(origprob);
   int nlinkingconss = GCGgetNVarLinkingconss(origprob);
   assert(nstabcenterlinkingconsvals <= GCGgetNVarLinkingconss(origprob) );
   int nconss = GCGgetNMasterConss(origprob);
   assert(nconss <= nstabcenterconsvals);
   SCIP_ROW** mastercuts = GCGsepaGetMastercuts(scip_);
   int ncuts = GCGsepaGetNCuts(scip_);
   assert(ncuts <= nstabcentercutvals);

   SCIP_Real divisornorm = 0.0;

   /* masterconss */
   assert(stabcenterconsvals != NULL);

   for( int i = 0; i < nconss; ++i )
   {
      SCIP_Real divisor = SQR((beta - 1.0) * stabcenterconsvals[i]
                        + beta * (subgradientconsvals[i] * dualdiffnorm / subgradientnorm)
                        + (1 - beta) * pricingtype->consGetDual(scip_, masterconss[i]));

      if( SCIPisPositive(scip_, divisor) )
         divisornorm += divisor;
   }

   /* mastercuts */
   assert(stabcenterconsvals != NULL);

   for( int i = 0; i < ncuts; ++i )
   {
      SCIP_Real divisor = SQR((beta - 1.0) * stabcentercutvals[i]
                        + beta * (subgradientcutvals[i] * dualdiffnorm / subgradientnorm)
                        + (1 - beta) * pricingtype->rowGetDual(mastercuts[i]));

      if( SCIPisPositive(scip_, divisor) )
         divisornorm += divisor;
   }

   /* linkingconss */
   assert(stabcenterlinkingconsvals != NULL);

   for( int i = 0; i < nlinkingconss; ++i )
   {
      SCIP_Real divisor = SQR((beta - 1.0) * stabcenterlinkingconsvals[i]
                        + beta * (subgradientlinkingconsvals[i] * dualdiffnorm / subgradientnorm)
                        + (1 - beta) * pricingtype->consGetDual(scip_, linkingconss[i]));

      if( SCIPisPositive(scip_, divisor) )
         divisornorm += divisor;
   }

   divisornorm = SQRT(divisornorm);

   hybridfactor = ((1 - alpha) * dualdiffnorm) / divisornorm;

   SCIPdebugMessage("Update hybridfactor with value %g.\n", hybridfactor);

   assert( SCIPisPositive(scip_, hybridfactor) );
}


SCIP_Bool Stabilization::isStabilized()
{
   if( inmispricingschedule )
      return SCIPisGT(scip_, alphabar, 0.0);
   return SCIPisGT(scip_, alpha, 0.0);
}

/** enabling mispricing schedule */
void Stabilization::activateMispricingSchedule(
)
{
   inmispricingschedule = TRUE;
}

/** disabling mispricing schedule */
void Stabilization::disablingMispricingSchedule(
)
{
   inmispricingschedule = FALSE;
   k=0;
}

/** is mispricing schedule enabled */
SCIP_Bool Stabilization::isInMispricingSchedule(
) const
{
   return inmispricingschedule;
}

/** update subgradient product */
SCIP_RETCODE Stabilization::updateSubgradientProduct(
   GCG_COL**            pricingcols         /**< solutions of the pricing problems */
)
{
   /* first update the arrays */
   SCIP_CALL( updateStabcenterconsvals() );
   SCIP_CALL( updateStabcentercutvals() );

   subgradientproduct = calculateSubgradientProduct(pricingcols);

   return SCIP_OKAY;
}


} /* namespace gcg */
