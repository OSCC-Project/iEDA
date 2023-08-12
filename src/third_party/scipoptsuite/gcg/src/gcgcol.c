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

/**@file   gcgcol.c
 * @brief  methods for working with gcg column structure
 * @author Jonas Witt
 *
 * Various methods to work with gcg column structure
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "pub_gcgcol.h"

#include "gcg.h"
#include "scip/def.h"
#include "scip/scip.h"
#include "scip/cons_linear.h"
#include "scip_misc.h"
#include "blockmemshell/memory.h"
#include "pricer_gcg.h"
#include "sepa_master.h"

#include <assert.h>

/** create a gcg column */
SCIP_RETCODE GCGcreateGcgCol(
   SCIP*                pricingprob,        /**< SCIP data structure (pricing problem) */
   GCG_COL**            gcgcol,             /**< pointer to store gcg column */
   int                  probnr,             /**< number of corresponding pricing problem */
   SCIP_VAR**           vars,               /**< (sorted) array of variables of corresponding pricing problem */
   SCIP_Real*           vals,               /**< array of solution values (belonging to vars) */
   int                  nvars,              /**< number of variables */
   SCIP_Bool            isray,              /**< is the column a ray? */
   SCIP_Real            redcost             /**< last known reduced cost */
   )
{
   int i;
   int nnonz;

   SCIP_CALL( SCIPallocBlockMemory(pricingprob, gcgcol) );

   (*gcgcol)->maxvars = SCIPcalcMemGrowSize(pricingprob, nvars);
   SCIP_CALL( SCIPallocBlockMemoryArray(pricingprob, &((*gcgcol)->vars), (*gcgcol)->maxvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(pricingprob, &((*gcgcol)->vals), (*gcgcol)->maxvars) );

   (*gcgcol)->pricingprob = pricingprob;
   (*gcgcol)->probnr = probnr;
   (*gcgcol)->isray = isray;
   (*gcgcol)->redcost = redcost;
   (*gcgcol)->age = 0;
   (*gcgcol)->mastercoefs = NULL;
   (*gcgcol)->mastercuts = NULL;
   (*gcgcol)->linkvars = NULL;
   (*gcgcol)->nmastercoefs = 0;
   (*gcgcol)->nmastercuts = 0;
   (*gcgcol)->maxmastercoefs = 0;
   (*gcgcol)->maxmastercuts = 0;
   (*gcgcol)->nlinkvars = 0;
   (*gcgcol)->initcoefs = FALSE;


   nnonz = 0;
   for( i = 0; i < nvars; ++i )
   {
      SCIP_VAR* origvar;
      SCIP_Real scalar;
      SCIP_Real constant;
      SCIP_Real origval;

      scalar = 1.0;
      constant = 0.0;

      origvar = vars[i];

      /* todo: capture vars? */
      SCIP_CALL( SCIPvarGetOrigvarSum(&origvar, &scalar, &constant) );

      assert( !SCIPisZero(pricingprob, scalar) );

      origval = (vals[i] - constant) / scalar;

      /* round origval if possible to avoid numerical troubles */
      if( SCIPvarIsIntegral(origvar) && SCIPisFeasIntegral(pricingprob, origval) )
         origval = SCIPround(pricingprob, origval);

      if( !SCIPisZero(pricingprob, origval) )
      {
         (*gcgcol)->vars[nnonz] = origvar;
         (*gcgcol)->vals[nnonz] = origval;
         ++nnonz;
      }
   }

   (*gcgcol)->nvars = nnonz;

   /* sort vars and vals array w.r.t. variable index */
   SCIPsortPtrReal((void**)(*gcgcol)->vars, (double*)(*gcgcol)->vals, SCIPvarComp, nnonz);

#ifndef NDEBUG
   for( i = 1 ; i < (*gcgcol)->nvars; ++i )
   {
      assert( SCIPvarCompare((*gcgcol)->vars[i-1], (*gcgcol)->vars[i]) != 0 );
   }
#endif
   return SCIP_OKAY;
}

/** free a gcg column */
void GCGfreeGcgCol(
   GCG_COL**            gcgcol              /**< pointer to store gcg column */
   )
{
   assert(gcgcol != NULL);
   assert(*gcgcol != NULL);

   /* todo: release vars? */
   assert((*gcgcol)->nvars == 0 || (*gcgcol)->vars != NULL);
   SCIPfreeBlockMemoryArrayNull((*gcgcol)->pricingprob, &(*gcgcol)->vars, (*gcgcol)->maxvars);
   assert((*gcgcol)->nvars == 0 || (*gcgcol)->vals != NULL);
   SCIPfreeBlockMemoryArrayNull((*gcgcol)->pricingprob, &(*gcgcol)->vals, (*gcgcol)->maxvars);
   SCIPfreeBlockMemoryArrayNull((*gcgcol)->pricingprob, &(*gcgcol)->mastercoefs, (*gcgcol)->maxmastercoefs);
   SCIPfreeBlockMemoryArrayNull((*gcgcol)->pricingprob, &(*gcgcol)->linkvars, (*gcgcol)->maxlinkvars);
   SCIPfreeBlockMemoryArrayNull((*gcgcol)->pricingprob, &(*gcgcol)->mastercuts, (*gcgcol)->maxmastercuts);
   SCIPfreeBlockMemory((*gcgcol)->pricingprob, gcgcol);
}

/** create a gcg column from a solution to a pricing problem */
SCIP_RETCODE GCGcreateGcgColFromSol(
   SCIP*                pricingprob,        /**< SCIP data structure (pricing problem) */
   GCG_COL**            gcgcol,             /**< pointer to store gcg column */
   int                  prob,               /**< number of corresponding pricing problem */
   SCIP_SOL*            sol,                /**< solution of pricing problem with index prob */
   SCIP_Bool            isray,              /**< is column a ray? */
   SCIP_Real            redcost             /**< last known reduced cost */
)
{
   SCIP_VAR** solvars;
   SCIP_VAR** colvars;

   SCIP_Real* colvals;

   int nsolvars;
   int ncolvars;

   int i;

   solvars = SCIPgetOrigVars(pricingprob);
   nsolvars = SCIPgetNOrigVars(pricingprob);

   SCIP_CALL( SCIPallocBufferArray(pricingprob, &colvars, nsolvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob, &colvals, nsolvars) );

   ncolvars = 0;

   for( i = 0; i < nsolvars; ++i )
   {
      SCIP_VAR* solvar;
      SCIP_Real solval;

      solvar = solvars[i];
      solval = SCIPgetSolVal(pricingprob, sol, solvar);

      /* round solval if possible to avoid numerical troubles */
      if( SCIPvarIsIntegral(solvar) && SCIPisFeasIntegral(pricingprob, solval) )
         solval = SCIPround(pricingprob, solval);

      if( SCIPisZero(pricingprob, solval) )
      {
         continue;
      }

      colvars[ncolvars] = solvar;
      colvals[ncolvars] = solval;
      ++ncolvars;
   }

   SCIP_CALL( GCGcreateGcgCol(pricingprob, gcgcol, prob, colvars, colvals, ncolvars, isray, redcost) );

   SCIPfreeBufferArray(pricingprob, &colvals);
   SCIPfreeBufferArray(pricingprob, &colvars);

   return SCIP_OKAY;
}

/** comparison method for sorting gcg columns by non-decreasing reduced cost */
SCIP_DECL_SORTPTRCOMP(GCGcolCompRedcost)
{
   SCIP_Real redcost1;
   SCIP_Real redcost2;

   redcost1 = GCGcolGetRedcost((GCG_COL*) elem1);
   redcost2 = GCGcolGetRedcost((GCG_COL*) elem2);

   if( redcost1 < redcost2 )
      return -1;
   else if( redcost1 > redcost2 )
      return +1;
   else
      return 0;
}

/** comparison method for sorting gcg columns by non-increasing age */
SCIP_DECL_SORTPTRCOMP(GCGcolCompAge)
{
   int age1;
   int age2;

   age1 = GCGcolGetAge((GCG_COL*) elem1);
   age2 = GCGcolGetAge((GCG_COL*) elem2);

   if( age1 < age2 )
      return +1;
   else if( age1 > age2 )
      return -1;
   else
      return 0;
}

/** comparison method for gcg columns. Returns TRUE iff columns are equal */
SCIP_Bool GCGcolIsEq(
   GCG_COL*             gcgcol1,
   GCG_COL*             gcgcol2
)
{
   SCIP* pricingprob;

   SCIP_VAR** vars1;
   SCIP_VAR** vars2;

   SCIP_Real* vals1;
   SCIP_Real* vals2;

   int nvars1;
   int nvars2;
   int probnr1;
   int probnr2;

   int i;


   probnr1 = GCGcolGetProbNr(gcgcol1);
   probnr2 = GCGcolGetProbNr(gcgcol2);

   if( probnr1 != probnr2 )
      return FALSE;

   nvars1 = GCGcolGetNVars(gcgcol1);
   nvars2 = GCGcolGetNVars(gcgcol2);

   if( nvars1 != nvars2 )
      return FALSE;

   pricingprob = GCGcolGetPricingProb(gcgcol1);
   vars1 = GCGcolGetVars(gcgcol1);
   vars2 = GCGcolGetVars(gcgcol2);

   vals1 = GCGcolGetVals(gcgcol1);
   vals2 = GCGcolGetVals(gcgcol2);

   for( i = 0; i < nvars1; ++i )
   {
      SCIP_VAR* var1;
      SCIP_VAR* var2;

      SCIP_Real val1;
      SCIP_Real val2;

      var1 = vars1[i];
      var2 = vars2[i];

      val1 = vals1[i];
      val2 = vals2[i];

      if( SCIPvarCompare(var1, var2) != 0 || !SCIPisEQ(pricingprob, val1, val2) )
      {
         return FALSE;
      }
   }

   return TRUE;

}

/** get pricing problem index of gcg column */
int GCGcolGetProbNr(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->probnr;
}

/** get pricing problem of gcg column */
SCIP* GCGcolGetPricingProb(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->pricingprob;
}

/** get variables of gcg column */
SCIP_VAR** GCGcolGetVars(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->vars;
}

/** get values of gcg column */
SCIP_Real* GCGcolGetVals(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->vals;
}

/** get number of variables of gcg column */
int GCGcolGetNVars(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->nvars;
}

/** is gcg column a ray? */
SCIP_Bool GCGcolIsRay(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->isray;
}

/** get reduced cost of gcg column */
SCIP_Real GCGcolGetRedcost(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->redcost;
}

/** get age of gcg column */
int GCGcolGetAge(
   GCG_COL*             gcgcol
   )
{
   return gcgcol->age;
}

/** update reduced cost of variable and increase age */
void GCGcolUpdateRedcost(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real            redcost,            /**< new reduced cost */
   SCIP_Bool            growold             /**< change age depending on reduced cost? */
   )
{
   gcgcol->redcost = redcost;

   if( !growold )
      return;

   if( !SCIPisNegative(gcgcol->pricingprob, redcost) )
      ++(gcgcol->age);
   else
      gcgcol->age = 0;
}

/** get master coefficients of column */
SCIP_Real* GCGcolGetMastercoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->mastercoefs;
}

/** get number of master coefficients of column */
int GCGcolGetNMastercoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->nmastercoefs;
}

/** set master coefficients information of column */
SCIP_RETCODE GCGcolSetMastercoefs(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real*           mastercoefs,        /**< array of master coefficients */
   int                  nmastercoefs        /**< new number of master coefficients */
   )
{
   int i;

   SCIPdebugMessage("Col set master coefs\n");
   assert(gcgcol->nmastercoefs == 0);
   if( nmastercoefs == 0 )
      return SCIP_OKAY;

   gcgcol->maxmastercoefs = SCIPcalcMemGrowSize(gcgcol->pricingprob, nmastercoefs);
   SCIP_CALL( SCIPallocBlockMemoryArray(gcgcol->pricingprob, &(gcgcol->mastercoefs), gcgcol->maxmastercoefs) );

   for( i = 0; i < nmastercoefs; ++i )
   {
      SCIP_Real coef = mastercoefs[i];
      gcgcol->mastercoefs[i] = coef;
   }

   gcgcol->nmastercoefs = nmastercoefs;

   return SCIP_OKAY;
}

/** set norm of column */
void GCGcolSetNorm(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real            norm                /**< norm of column */
   )
{
   gcgcol->norm = norm;
}

/** get norm of column */
void GCGcolComputeNorm(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   int i;
   SCIP_Real norm;

   SCIP_Real* solvals;
   SCIP_Real* mastercoefs;
   int nmastercoefs;
   SCIP_Real* mastercuts;
   int nmastercuts;
   int* linkvars;
   int nlinkvars;

   assert(scip != NULL);
   assert(gcgcol != NULL);

   solvals = GCGcolGetVals(gcgcol);
   nmastercoefs = GCGcolGetNMastercoefs(gcgcol);
   mastercoefs = GCGcolGetMastercoefs(gcgcol);
   nmastercuts = GCGcolGetNMastercuts(gcgcol);
   mastercuts = GCGcolGetMastercuts(gcgcol);
   nmastercuts = GCGcolGetNMastercuts(gcgcol);
   nlinkvars = GCGcolGetNLinkvars(gcgcol);
   linkvars = GCGcolGetLinkvars(gcgcol);

   norm = 0.0;
   /* compute scalar of master values of gcg columns */
   for( i = 0; i < nmastercoefs; ++i )
   {
      if( !SCIPisZero(scip, mastercoefs[i]))
         norm += SQR(mastercoefs[i]);
   }

   for( i = 0; i < nmastercuts; ++i )
   {
      if( !SCIPisZero(scip, mastercuts[i]))
         norm += SQR(mastercuts[i]);
   }


   for( i = 0; i < nlinkvars; ++i )
   {
      if( !SCIPisZero(scip, solvals[linkvars[i]]) )
         norm += solvals[linkvars[i]];
   }

   /* consider convexity constraint */
   norm += 1.0;

   gcgcol->norm = norm;
}

/** set master coefficients of column as initialized */
SCIP_RETCODE GCGcolSetInitializedCoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   assert(!gcgcol->initcoefs);
   gcgcol->initcoefs = TRUE;
   return SCIP_OKAY;
}

/** return if master coefficients of column have been initialized */
SCIP_Bool GCGcolGetInitializedCoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->initcoefs;
}

/** get master coefficients of column */
int* GCGcolGetLinkvars(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->linkvars;
}

/** get number of master coefficients of column */
int GCGcolGetNLinkvars(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->nlinkvars;
}

/** set master coefficients information of column */
SCIP_RETCODE GCGcolSetLinkvars(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   int*                 linkvars,           /**< array of linking variable indices for gcgcol->var */
   int                  nlinkvars           /**< number of linking variables in gcgcol->var */
   )
{
   int i;

   assert(gcgcol->nlinkvars == 0);

   gcgcol->maxlinkvars = SCIPcalcMemGrowSize(gcgcol->pricingprob, nlinkvars);
   SCIP_CALL( SCIPallocBlockMemoryArray(gcgcol->pricingprob, &(gcgcol->linkvars), gcgcol->maxlinkvars) );

   for( i = 0; i < nlinkvars; ++i )
   {
      gcgcol->linkvars[i] = linkvars[i];
   }

   gcgcol->nlinkvars = nlinkvars;

   return SCIP_OKAY;
}

/** get master cut coefficients of column */
SCIP_Real* GCGcolGetMastercuts(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->mastercuts;
}

/** get number of master cut coefficients of column */
int GCGcolGetNMastercuts(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->nmastercuts;
}

/** get norm of column */
SCIP_Real GCGcolGetNorm(
   GCG_COL*             gcgcol              /**< gcg column structure */
   )
{
   return gcgcol->norm;
}

/** update master cut coefficients information of column */
SCIP_RETCODE GCGcolUpdateMastercuts(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real*           newmastercuts,      /**< pointer to new array of master cut coefficients */
   int                  nnewmastercuts      /**< new number of master cut coefficients */
   )
{
   int i;
   int newsize;

   i = gcgcol->nmastercuts + nnewmastercuts;
   newsize = SCIPcalcMemGrowSize(gcgcol->pricingprob, i);
   if( i > gcgcol->maxmastercuts )
   {
      SCIP_CALL( SCIPreallocBlockMemoryArray(GCGcolGetPricingProb(gcgcol), &(gcgcol->mastercuts),
            gcgcol->maxmastercuts, newsize) );
   }

   gcgcol->maxmastercuts = newsize;

   for( i = 0; i < nnewmastercuts; ++i )
   {
      gcgcol->mastercuts[gcgcol->nmastercuts] = newmastercuts[i];
      ++(gcgcol->nmastercuts);
   }

   return SCIP_OKAY;
}

/** return solution value of variable in gcg column */
SCIP_Real GCGcolGetSolVal(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol,             /**< gcg column */
   SCIP_VAR*            var                 /**< variable */
   )
{
   SCIP_VAR** vars;
   SCIP_Real* vals;
   int nvars;
   int pos;
   SCIP_Bool found;

   vars = gcgcol->vars;
   vals = gcgcol->vals;
   nvars = gcgcol->nvars;

   found = SCIPsortedvecFindPtr((void**) vars, SCIPvarComp, (void*) var, nvars, &pos);

   if( !found )
   {
      return 0.0;
   }

   return vals[pos];
}

/** returns whether the col's age exceeds the age limit */
SCIP_Bool GCGcolIsAged(
   GCG_COL*             col,                /**< col to check */
   int                   agelimit            /**< maximum age a col can reach before it is deleted from the pool, or -1 */
   )
{
   assert(col != NULL);

   return (agelimit >= 0 && col->age > agelimit);
}

/** compute parallelism of column to dual objective */
SCIP_Real GCGcolComputeDualObjPara(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol              /**< gcg column */
)
{
   SCIP_Real para;

   int i;

   SCIP_CONS** masterconss;
   SCIP_ROW** cuts;

   int prob;

   SCIP_Real* mastercoefs;
   int nmastercoefs;
   SCIP_Real* mastercuts;
   int nmastercuts;

   SCIP_Real dualobjnorm;


   assert(scip != NULL);
   assert(gcgcol != NULL);

   prob = GCGcolGetProbNr(gcgcol);
   nmastercoefs = GCGcolGetNMastercoefs(gcgcol);
   mastercoefs = GCGcolGetMastercoefs(gcgcol);
   nmastercuts = GCGcolGetNMastercuts(gcgcol);
   mastercuts = GCGcolGetMastercuts(gcgcol);
   masterconss = GCGgetMasterConss(GCGmasterGetOrigprob(scip));
   cuts = GCGsepaGetMastercuts(scip);

   para = 0.0;

   dualobjnorm = 0.0;

   /* compute scalar of master values of gcg columns */
   for( i = 0; i < nmastercoefs; ++i )
   {
      SCIP_Real lhs;
      SCIP_Real rhs;

      lhs = SCIPgetLhsLinear(scip, masterconss[i]);
      rhs = SCIPgetRhsLinear(scip, masterconss[i]);

      if( !SCIPisInfinity(scip, -lhs))
      {
         dualobjnorm += SQR(lhs);

         if( SCIPisPositive(scip, mastercoefs[i]) )
            para += mastercoefs[i] * lhs;
      }
      else if( !SCIPisInfinity(scip, rhs) )
      {
         dualobjnorm += SQR(rhs);

         if(SCIPisNegative(scip, mastercoefs[i] ) )
            para += mastercoefs[i] * rhs;
      }
   }

   for( i = 0; i < nmastercuts; ++i )
   {
      SCIP_Real lhs;
      SCIP_Real rhs;

      if( !SCIProwIsInLP(cuts[i]) )
         continue;

      lhs = SCIProwGetLhs(cuts[i]);
      rhs = SCIProwGetRhs(cuts[i]);

      if( !SCIPisInfinity(scip, -lhs))
      {
         dualobjnorm += SQR(lhs);

         if( SCIPisPositive(scip, mastercuts[i]) )
            para += mastercuts[i] * lhs;
      }
      else if( !SCIPisInfinity(scip, rhs) )
      {
         dualobjnorm += SQR(rhs);

         if(SCIPisNegative(scip, mastercuts[i] ) )
            para += mastercuts[i] * rhs;
      }
   }

   for( i = 0; i < GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)); ++i )
      dualobjnorm += SQR(GCGgetNIdenticalBlocks(GCGmasterGetOrigprob(scip), i));

   para += SQR(GCGgetNIdenticalBlocks(GCGmasterGetOrigprob(scip), prob));

   assert(!SCIPisInfinity(scip, ABS(para)));

   dualobjnorm = SQRT(dualobjnorm);
   assert(!SCIPisInfinity(scip, dualobjnorm));
   assert(SCIPisPositive(scip, dualobjnorm));
   assert(SCIPisPositive(scip, gcgcol->norm));

   para = para / (dualobjnorm * gcgcol->norm);

   return para;
}

/** compute orthogonality of two gcg columns */
SCIP_Real GCGcolComputeOrth(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol1,            /**< first gcg column */
   GCG_COL*             gcgcol2             /**< second gcg column */
)
{
   int i;
   int j;
   SCIP_Real para = 0.0;
   SCIP_Real norm1 = 0.0;
   SCIP_Real norm2 = 0.0;

   int prob1;

   SCIP_VAR** solvars1;
   SCIP_Real* solvals1;
   SCIP_Real* mastercoefs1;
   int nmastercoefs1;
   SCIP_Real* mastercuts1;
   int nmastercuts1;
   int* linkvars1;
   int nlinkvars1;

   int prob2;

   SCIP_VAR** solvars2;
   SCIP_Real* solvals2;
   SCIP_Real* mastercoefs2;
   SCIP_Real* mastercuts2;
   int* linkvars2;
   int nlinkvars2;

   assert(scip != NULL);
   assert(gcgcol1 != NULL);
   assert(gcgcol2 != NULL);

   prob1 = GCGcolGetProbNr(gcgcol1);
   solvars1 = GCGcolGetVars(gcgcol1);
   solvals1 = GCGcolGetVals(gcgcol1);
   nmastercoefs1 = GCGcolGetNMastercoefs(gcgcol1);
   mastercoefs1 = GCGcolGetMastercoefs(gcgcol1);
   nmastercuts1 = GCGcolGetNMastercuts(gcgcol1);
   mastercuts1 = GCGcolGetMastercuts(gcgcol1);
   nlinkvars1 = GCGcolGetNLinkvars(gcgcol1);
   linkvars1 = GCGcolGetLinkvars(gcgcol1);

   prob2 = GCGcolGetProbNr(gcgcol2);
   solvars2 = GCGcolGetVars(gcgcol2);
   solvals2 = GCGcolGetVals(gcgcol2);
   mastercoefs2 = GCGcolGetMastercoefs(gcgcol2);
   mastercuts2 = GCGcolGetMastercuts(gcgcol2);
   nlinkvars2 = GCGcolGetNLinkvars(gcgcol2);
   linkvars2 = GCGcolGetLinkvars(gcgcol2);

   /* compute scalar of master values of gcg columns */
   for( i = 0; i < nmastercoefs1; ++i )
   {
      if( SCIPisPositive(scip, mastercoefs1[i] * mastercoefs2[i]) )
         para += mastercoefs1[i] * mastercoefs2[i];

      if( SCIPisPositive(scip, mastercoefs1[i]) )
         norm1 += SQR(mastercoefs1[i]);
      if( SCIPisPositive(scip, mastercoefs2[i]) )
         norm2 += SQR(mastercoefs2[i]);
   }

   for( i = 0; i < nmastercuts1; ++i )
   {
      if( SCIPisPositive(scip, mastercuts1[i] * mastercuts2[i]) )
         para += mastercuts1[i] * mastercuts2[i];

      if( SCIPisPositive(scip, mastercuts1[i]) )
         norm1 += SQR(mastercuts1[i]);
      if( SCIPisPositive(scip, mastercuts2[i]) )
         norm2 += SQR(mastercuts2[i]);
   }

   for( i = 0; i < nlinkvars1; ++i )
   {
      SCIP_VAR* linkvar1;
      SCIP_Real linkval1;
      linkvar1 = solvars1[linkvars1[i]];
      linkval1 = solvals1[linkvars1[i]];

      norm1 += SQR(linkval1);

      for( j = 0; j < nlinkvars2; ++j )
      {
         SCIP_VAR* linkvar2;
         SCIP_Real linkval2;
         linkvar2 = solvars2[linkvars2[j]];
         linkval2 = solvals2[linkvars2[j]];

         if( linkvar1 == linkvar2 )
         {
            para += linkval1 * linkval2;
            break;
         }
      }
   }

   for( i = 0; i < nlinkvars2; ++i )
   {
      SCIP_Real linkval2;

      linkval2 = solvals2[linkvars2[i]];

      norm2 += SQR(linkval2);
   }


   /* scalar for convexitiy constraints */
   if( prob1 == prob2 )
      para *= 1.0;

   norm1 *= 1.0;
   norm2 *= 1.0;

   norm1 = SQRT(norm1);
   norm2 = SQRT(norm2);

   assert(SCIPisPositive(scip, norm1) && SCIPisPositive(scip, norm2));

   para = para/(norm1*norm2);

   return 1.0 - para;
}
