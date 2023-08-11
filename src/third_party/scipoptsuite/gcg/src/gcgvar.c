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

/**@file   gcgvar.c
 * @brief  GCG variable access functions
 * @author Martin Bergner
 * @author Christian Puchert
 *
 * @todo capture and release variables stored in other variable's data?
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "gcg.h"
#include "pub_gcgvar.h"
#include "struct_vardata.h"
#include "relax_gcg.h"
#include "scip_misc.h"
#include "scip/cons_linear.h"

#define STARTMAXMASTERVARS 8
#define STARTMAXORIGVARS 1

/*
 * Vardata methods
 */

/** callback method called when an original GCG variable is deleted */
static
SCIP_DECL_VARDELORIG(GCGvarDelOrig)
{
   if( *vardata == NULL )
      return SCIP_OKAY;

   /*lint -e715 */
   if( (*vardata)->vartype == GCG_VARTYPE_ORIGINAL )
   {
      if( (*vardata)->blocknr == -2 )
      {
         int nblocks;

         nblocks = GCGgetNPricingprobs(scip);
         assert(nblocks > 0);

         assert((*vardata)->data.origvardata.linkingvardata != NULL);
         if( (*vardata)->data.origvardata.linkingvardata->linkconss != NULL )
         {
            int i;
            assert((*vardata)->data.origvardata.linkingvardata->pricingvars != NULL);

            for( i = 0; i < nblocks; i++ )
            {
               assert(((*vardata)->data.origvardata.linkingvardata->linkconss[i] == NULL)
                  == ((*vardata)->data.origvardata.linkingvardata->pricingvars[i] == NULL));
            }

            SCIPfreeBlockMemoryArray(scip, &((*vardata)->data.origvardata.linkingvardata->linkconss), nblocks);
            SCIPfreeBlockMemoryArray(scip, &((*vardata)->data.origvardata.linkingvardata->pricingvars), nblocks);
         }
         else if( (*vardata)->data.origvardata.linkingvardata->pricingvars != NULL )
            SCIPfreeBlockMemoryArray(scip, &((*vardata)->data.origvardata.linkingvardata->pricingvars), nblocks);

         SCIPfreeBlockMemory(scip, &((*vardata)->data.origvardata.linkingvardata));
         (*vardata)->data.origvardata.linkingvardata = NULL;
      }
      assert((*vardata)->data.origvardata.linkingvardata == NULL);
      assert((*vardata)->data.origvardata.mastervars != NULL);
      assert((*vardata)->data.origvardata.mastervals != NULL);
      SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.origvardata.mastervars), (*vardata)->data.origvardata.maxmastervars);
      SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.origvardata.mastervals), (*vardata)->data.origvardata.maxmastervars);

      if( (*vardata)->data.origvardata.ncoefs > 0 )
      {
         assert((*vardata)->data.origvardata.coefs != NULL);
         assert((*vardata)->data.origvardata.masterconss != NULL);
         SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.origvardata.coefs), (*vardata)->data.origvardata.ncoefs);
         SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.origvardata.masterconss), (*vardata)->data.origvardata.ncoefs);
      }
   }
   if( (*vardata)->vartype == GCG_VARTYPE_PRICING )
   {
      assert((*vardata)->data.pricingvardata.norigvars >= 1);
      SCIPfreeBlockMemoryArray(scip, &((*vardata)->data.pricingvardata.origvars), (*vardata)->data.pricingvardata.maxorigvars);
   }
   assert((*vardata)->vartype != GCG_VARTYPE_MASTER);
   SCIPfreeBlockMemory(scip, vardata);

   return SCIP_OKAY;
}


/** callback method called when a transformed GCG variable is deleted */
static
SCIP_DECL_VARDELTRANS(gcgvardeltrans)
{
   /*lint -e715 */
   assert((*vardata)->vartype == GCG_VARTYPE_MASTER);
   SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.mastervardata.origvals), (*vardata)->data.mastervardata.maxorigvars);
   SCIPfreeBlockMemoryArrayNull(scip, &((*vardata)->data.mastervardata.origvars), (*vardata)->data.mastervardata.maxorigvars);
   SCIPhashmapFree(&((*vardata)->data.mastervardata.origvar2val));

   SCIPfreeBlockMemory(scip, vardata);

   return SCIP_OKAY;
}



/** returns TRUE or FALSE whether variable is a pricing variable or not */
SCIP_Bool GCGvarIsPricing(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   if( vardata == NULL )
      return FALSE;

   return vardata->vartype == GCG_VARTYPE_PRICING;
}

#ifndef NDEBUG
/** returns TRUE or FALSE whether variable is a master variable or not */
SCIP_Bool GCGvarIsMaster(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->vartype == GCG_VARTYPE_MASTER;
}
#endif

#ifndef NDEBUG
/** returns TRUE or FALSE whether variable is a original variable or not */
SCIP_Bool GCGvarIsOriginal(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->vartype == GCG_VARTYPE_ORIGINAL;
}
#endif

#ifndef NDEBUG
/** returns TRUE or FALSE whether variable is a linking variable or not */
SCIP_Bool GCGoriginalVarIsLinking(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->blocknr == -2;
}
#endif

#ifndef NDEBUG
/** returns TRUE or FALSE whether variable is a directly transferred variable or not */
SCIP_Bool GCGoriginalVarIsTransVar(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->blocknr == -1;
}
#endif

#ifndef NDEBUG
/** returns the pricing var of an original variable */
SCIP_VAR* GCGoriginalVarGetPricingVar(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.linkingvardata == NULL);
   assert(!GCGoriginalVarIsLinking(var));
   if( vardata->data.origvardata.pricingvar != NULL )
      assert(GCGvarIsPricing(vardata->data.origvardata.pricingvar));
   return vardata->data.origvardata.pricingvar;
}
#endif

/** returns the pricing var of an original variable */
void GCGoriginalVarSetPricingVar(
   SCIP_VAR*             var,                /**< SCIP variable structure */
   SCIP_VAR*             pricingvar          /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(pricingvar != NULL);
   assert(GCGvarIsOriginal(var));
   assert(GCGvarIsPricing(pricingvar));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.linkingvardata == NULL);
   assert(!GCGoriginalVarIsLinking(var));
   vardata->data.origvardata.pricingvar = pricingvar;
}

/** creates the data for all variables of the original program */
SCIP_RETCODE GCGcreateOrigVarsData(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_VAR** vars;
   int nvars;
   int i;

   assert(scip != NULL);

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   /* loop over the variables in the original problem */
   for( i = 0; i < nvars; i++ )
   {
      assert(vars[i] != NULL);
      SCIP_CALL( GCGorigVarCreateData(scip, vars[i]) );
   }

   return SCIP_OKAY;
}

/** frees the data for all variables of the original program */
SCIP_RETCODE GCGfreeOrigVarsData(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_VAR** vars;
   int nvars;
   int i;

   assert(scip != NULL);

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   /* loop over the variables in the original problem */
   for( i = 0; i < nvars; i++ )
   {
      SCIP_VAR* var;
      assert(vars[i] != NULL);
      var = vars[i];

      if( SCIPvarGetData(var) != NULL )
      {
         SCIP_VARDATA* oldvardata;
         oldvardata = SCIPvarGetData(var);

         SCIP_CALL( GCGvarDelOrig(scip, var, &oldvardata) );
         SCIPvarSetData(var, NULL);
      }
   }

   return SCIP_OKAY;
}

/** creates the data for a variable of the original program */
SCIP_RETCODE GCGorigVarCreateData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< pointer to variable object */
   )
{
   SCIP_VARDATA* vardata;

   assert(scip != NULL);
   assert(var != NULL);
   assert(SCIPvarIsOriginal(var) || SCIPvarGetStatus(var) == SCIP_VARSTATUS_LOOSE);
   /* create the vardata and initialize its values */
   SCIP_CALL( SCIPallocBlockMemory(scip, &vardata) );
   vardata->vartype = GCG_VARTYPE_ORIGINAL;
   vardata->blocknr = -1;
   vardata->data.origvardata.pricingvar = NULL;
   vardata->data.origvardata.coefs = NULL;
   vardata->data.origvardata.masterconss = NULL;
   vardata->data.origvardata.ncoefs = 0;
   vardata->data.origvardata.nmastervars = 0;
   vardata->data.origvardata.maxmastervars = SCIPcalcMemGrowSize(scip, STARTMAXMASTERVARS);
   vardata->data.origvardata.linkingvardata = NULL;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(vardata->data.origvardata.mastervars),
         vardata->data.origvardata.maxmastervars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(vardata->data.origvardata.mastervals),
         vardata->data.origvardata.maxmastervars) );

   if( SCIPvarGetData(var) != NULL )
   {
      SCIP_VARDATA* oldvardata;
      oldvardata = SCIPvarGetData(var);

      SCIP_CALL( GCGvarDelOrig(scip, var, &oldvardata) );
   }
   SCIPvarSetData(var, vardata);
   if( SCIPvarIsOriginal(var) )
   {
      SCIPvarSetDelorigData(var, GCGvarDelOrig);
      if( SCIPvarGetTransVar(var) != NULL )
      {
         SCIPvarSetData(SCIPvarGetProbvar(SCIPvarGetTransVar(var)), vardata);
      }
   }
   else
   {
      //assert(SCIPvarIsTransformedOrigvar(var));
      SCIPvarSetDeltransData(var, GCGvarDelOrig);
   }

   return SCIP_OKAY;
}

/** copies the pricing variable data to a master problem variable. This is used in the Benders' decomposition mode when
 * subproblems are merged into the master problem.
 */
SCIP_RETCODE GCGcopyPricingvarDataToMastervar(
   SCIP*                 scip,               /**< master SCIP data structure */
   SCIP_VAR*             pricingvar,         /**< the pricing problem variable is copied from */
   SCIP_VAR*             mastervar           /**< the master variable that the vardata is copied to */
   )
{
   SCIP_VARDATA* targetvardata;
   SCIP_VAR* origvar;
   assert(pricingvar != NULL);
   assert(mastervar != NULL);
   assert(GCGvarIsPricing(pricingvar));
   /* we can't assert that mastervar is a master variable because it may not have the appropriate vardata yet */

   assert(GCGpricingVarGetNOrigvars(pricingvar) == 1);

   /* create vardata */
   SCIP_CALL( SCIPallocBlockMemory(scip, &targetvardata) );
   targetvardata->vartype = GCG_VARTYPE_MASTER;
   targetvardata->blocknr = -1;
   targetvardata->data.mastervardata.isray = FALSE;
   targetvardata->data.mastervardata.isartificial = FALSE;
   targetvardata->data.mastervardata.norigvars = 1;
   targetvardata->data.mastervardata.maxorigvars = 1;
   targetvardata->data.mastervardata.index = -1;

   /* save corresoponding origvar */
   origvar = GCGpricingVarGetOrigvars(pricingvar)[0];
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(targetvardata->data.mastervardata.origvars), 1) ); /*lint !e506*/
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(targetvardata->data.mastervardata.origvals), 1) ); /*lint !e506*/
   SCIP_CALL( SCIPhashmapCreate(&(targetvardata->data.mastervardata.origvar2val), SCIPblkmem(scip), 1));
   targetvardata->data.mastervardata.origvars[0] = origvar;
   targetvardata->data.mastervardata.origvals[0] = 1.0;
   SCIPhashmapInsertReal(targetvardata->data.mastervardata.origvar2val, origvar, 1.0);

   SCIPvarSetData(mastervar, targetvardata);
   SCIPvarSetDeltransData(mastervar, gcgvardeltrans);

   return SCIP_OKAY;
}

#ifndef NDEBUG
/** returns the pricing variables of an linking variable */
SCIP_VAR** GCGlinkingVarGetPricingVars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGoriginalVarIsLinking(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.linkingvardata != NULL);
   assert(vardata->data.origvardata.linkingvardata->pricingvars != NULL);

   return vardata->data.origvardata.linkingvardata->pricingvars;
}
#endif

/** sets the pricing var of the corresponding linking variable at the specified position */
void GCGlinkingVarSetPricingVar(
   SCIP_VAR*             origvar,            /**< original variable */
   int                   pricingprobnr,      /**< number of pricing problem */
   SCIP_VAR*             var                 /**< pricing variable */
   )
{
   SCIP_VARDATA* vardata;
   assert(origvar != NULL);
   assert(var != NULL);
   assert(pricingprobnr >= 0);

   assert(GCGoriginalVarIsLinking(origvar));
   assert(GCGvarIsPricing(var));

   vardata = SCIPvarGetData(origvar);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.linkingvardata != NULL);
   assert(vardata->data.origvardata.linkingvardata->pricingvars != NULL);

   vardata->data.origvardata.linkingvardata->pricingvars[pricingprobnr] = var;
}

/** returns the blocks the linking variable is in */
SCIP_RETCODE GCGlinkingVarGetBlocks(
   SCIP_VAR*             var,                /**< SCIP variable structure */
   int                   nblocks,            /**< size of array blocks */
   int*                  blocks              /**< array to store the blocks of the linking variable */
   )
{
   SCIP_VARDATA* vardata;
   int i;
   int j;

   assert(var != NULL);
   assert(nblocks == 0 || blocks != NULL);

   assert(GCGoriginalVarIsLinking(var));
   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.linkingvardata != NULL);
   assert(vardata->data.origvardata.linkingvardata->nblocks > 0);

   /* the blocks array must be large enough to hold all block numbers */
   if( nblocks < vardata->data.origvardata.linkingvardata->nblocks )
   {
      SCIPerrorMessage("array too small to store all block numbers!\n");
      return SCIP_INVALIDDATA;
   }
   assert(nblocks >= vardata->data.origvardata.linkingvardata->nblocks);

   /* fill the blocks array */
   j = -1;
   for( i = 0; i < vardata->data.origvardata.linkingvardata->nblocks; ++i )
   {
      /* search the next block the linking variable is contained in */
      do
         ++j;
      while ( vardata->data.origvardata.linkingvardata->pricingvars[j] == NULL );
      blocks[i] = j;
   }

   return SCIP_OKAY;
}

#ifndef NDEBUG
/** returns the number of blocks the linking variable is in */
int GCGlinkingVarGetNBlocks(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   assert(GCGoriginalVarIsLinking(var));
   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.linkingvardata != NULL);
   assert(vardata->data.origvardata.linkingvardata->nblocks > 0);
   return vardata->data.origvardata.linkingvardata->nblocks;
}
#endif

#ifndef NDEBUG
/** returns the original var of a pricing variable */
SCIP_VAR* GCGpricingVarGetOriginalVar(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsPricing(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.pricingvardata.norigvars >= 0);
   assert(vardata->data.pricingvardata.origvars != NULL);
   assert(vardata->data.pricingvardata.origvars[0] != NULL);
   assert(vardata->blocknr >= 0); /* variable belongs to exactly one block */

   return vardata->data.pricingvardata.origvars[0];
}
#endif

/** adds the original var to the pricing variable */
SCIP_RETCODE GCGpricingVarAddOrigVar(
   SCIP*                 scip,               /**< SCIP variable structure */
   SCIP_VAR*             pricingvar,         /**< pricing variable */
   SCIP_VAR*             origvar             /**< original pricing variable */
   )
{
   SCIP_VARDATA* vardata;
   assert(pricingvar != NULL);
   assert(origvar != NULL);
   assert(GCGvarIsPricing(pricingvar));
   assert(GCGvarIsOriginal(origvar));

   vardata = SCIPvarGetData(pricingvar);
   assert(vardata != NULL);
   assert(vardata->data.pricingvardata.norigvars >= 0);
   assert(vardata->data.pricingvardata.origvars != NULL);
   assert(vardata->data.pricingvardata.origvars[0] != NULL);
   assert(vardata->blocknr >= 0); /* variable belongs to exactly one block */

   /* realloc origvars array of the pricing variable, if needed */
   if( vardata->data.pricingvardata.maxorigvars == vardata->data.pricingvardata.norigvars )
   {
      int newsize = SCIPcalcMemGrowSize(scip, vardata->data.pricingvardata.norigvars+1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(vardata->data.pricingvardata.origvars), vardata->data.pricingvardata.maxorigvars,
            newsize) );
      SCIPdebugMessage("origvars array of var %s resized from %d to %d\n", SCIPvarGetName(origvar),
         vardata->data.pricingvardata.maxorigvars, newsize);
      vardata->data.pricingvardata.maxorigvars = newsize;
   }

   vardata->data.pricingvardata.origvars[vardata->data.pricingvardata.norigvars] = origvar;
   vardata->data.pricingvardata.norigvars++;

   return SCIP_OKAY;
}

#ifndef NDEBUG
/** returns the number of master variables the original variable is contained in */
int GCGoriginalVarGetNMastervars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   assert(vardata->data.origvardata.nmastervars >= 0);

   return vardata->data.origvardata.nmastervars;
}
#endif

#ifndef NDEBUG
/** returns the master variables the original variable is contained in */
SCIP_VAR** GCGoriginalVarGetMastervars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.mastervars != NULL);
   return vardata->data.origvardata.mastervars;
}
#endif

#ifndef NDEBUG
/** returns the fraction of master variables the original variable is contained in */
SCIP_Real* GCGoriginalVarGetMastervals(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.mastervals != NULL);
   return vardata->data.origvardata.mastervals;
}
#endif

#ifndef NDEBUG
/** returns the coefficients of master constraints the original variable is contained in */
SCIP_Real* GCGoriginalVarGetCoefs(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.coefs != NULL || vardata->data.origvardata.ncoefs == 0 );
   return vardata->data.origvardata.coefs;
}
#endif

#ifndef NDEBUG
/** returns the number of coefficients of master constraints the original variable is contained in */
int GCGoriginalVarGetNCoefs(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.coefs != NULL || vardata->data.origvardata.ncoefs == 0 );
   return vardata->data.origvardata.ncoefs;
}
#endif

/** sets the number of master variables the original variable is contained in */
void GCGoriginalVarSetNCoefs(
   SCIP_VAR*             var,                /**< SCIP variable structure */
   int                   ncoefs              /**< number of coefficient to set */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(ncoefs >= 0);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.coefs != NULL || vardata->data.origvardata.ncoefs == 0 );
   if( ncoefs == 0 )
      assert(vardata->data.origvardata.coefs == NULL);

   vardata->data.origvardata.ncoefs = ncoefs;
}

/** adds a coefficient of the master variable to the coefs array for the resp. constraint */
SCIP_RETCODE GCGoriginalVarAddCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to add coef */
   SCIP_Real             val,                /**< coefficent to set */
   SCIP_CONS*            cons                /**< constraint the variable is in */
   )
{
   SCIP_VARDATA* vardata;
   assert(scip != NULL);
   assert(var != NULL);
   assert(!SCIPisZero(scip, val));
   assert(cons != NULL);
   assert(GCGvarIsOriginal(var));
   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(vardata->data.origvardata.coefs), (size_t)vardata->data.origvardata.ncoefs, (size_t)vardata->data.origvardata.ncoefs+1) );
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(vardata->data.origvardata.masterconss), (size_t)vardata->data.origvardata.ncoefs, (size_t)vardata->data.origvardata.ncoefs+1) );

   assert(vardata->data.origvardata.coefs != NULL);
   assert(vardata->data.origvardata.masterconss != NULL);

   vardata->data.origvardata.coefs[vardata->data.origvardata.ncoefs] = val;
   vardata->data.origvardata.masterconss[vardata->data.origvardata.ncoefs] = cons;
   vardata->data.origvardata.ncoefs++;

   return SCIP_OKAY;
}

#ifndef NDEBUG
/** returns the fraction of master variables the original variable is contained in */
SCIP_CONS** GCGoriginalVarGetMasterconss(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->data.origvardata.masterconss;
}
#endif

/** adds variable to a new block, making a linkingvariable out of it, if necessary */
SCIP_RETCODE GCGoriginalVarAddBlock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< var that is added to a block */
   int                   newblock,           /**< the new block the variable will be in */
   int                   nblocks,            /**< total number of pricing problems */
   DEC_DECMODE           mode                /**< the decomposition mode */
   )
{
   SCIP_VARDATA* vardata;
   int blocknr;
   assert(scip != NULL);
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(nblocks >= 0);
   assert((newblock >= 0 && newblock < nblocks)
      || (GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS && newblock == -2));
   blocknr = GCGvarGetBlock(var);
   assert(newblock >= 0 || (newblock == -2 && blocknr > -1));
   /* the variable was only in one block so far, so set up the linking variable data */
   if( blocknr > -1 )
   {
      SCIP_CALL( SCIPallocBlockMemory(scip, &vardata->data.origvardata.linkingvardata) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vardata->data.origvardata.linkingvardata->pricingvars, nblocks) );
      BMSclearMemoryArray(vardata->data.origvardata.linkingvardata->pricingvars, nblocks);

      if( mode != DEC_DECMODE_BENDERS )
      {
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vardata->data.origvardata.linkingvardata->linkconss, nblocks) );
         BMSclearMemoryArray(vardata->data.origvardata.linkingvardata->linkconss, nblocks);
      }
      else
         vardata->data.origvardata.linkingvardata->linkconss = NULL;

      /* store old block; store the original variable, it will be exchanged for the correct pricing variable later */
      vardata->data.origvardata.linkingvardata->pricingvars[blocknr] = var;
      vardata->data.origvardata.linkingvardata->nblocks = 1;

      vardata->blocknr = -2;
   }
   assert(GCGoriginalVarIsLinking(var));

   /* store new block. In the Benders' decomposition mode, it is possible to have linking variables that only correspond
    * to a single block. In this case, we still want the variables in the master problem, but we need to ensure that the
    * linking variables are added correctly
    */
   if( newblock >= 0 && vardata->data.origvardata.linkingvardata->pricingvars[newblock] == NULL )
   {
      assert(mode == DEC_DECMODE_BENDERS || vardata->data.origvardata.linkingvardata->linkconss[newblock] == NULL);
      vardata->data.origvardata.linkingvardata->pricingvars[newblock] = var;
      vardata->data.origvardata.linkingvardata->nblocks++;
   }
   assert(vardata->data.origvardata.linkingvardata->nblocks <= nblocks);
   return SCIP_OKAY;
}

#ifndef NDEBUG
/** returns the linking constraints */
SCIP_CONS** GCGlinkingVarGetLinkingConss(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsOriginal(var));
   assert(GCGoriginalVarIsLinking(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.linkingvardata != NULL);
   //assert(vardata->data.origvardata.linkingvardata->linkconss != NULL);
   return vardata->data.origvardata.linkingvardata->linkconss;
}
#endif

/** sets the linking constraints */
void GCGlinkingVarSetLinkingCons(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_CONS*            cons,               /**< linking constraint */
   int                   index               /**< index of pricing problem */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(cons != NULL);
   assert(index >= 0);
   assert(GCGvarIsOriginal(var));
   assert(GCGoriginalVarIsLinking(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.origvardata.linkingvardata != NULL);
   assert(vardata->data.origvardata.linkingvardata->linkconss != NULL);
   vardata->data.origvardata.linkingvardata->linkconss[index] = cons;
}

/** returns TRUE or FALSE whether a master variable is a direct copy of a linking variable or not */
SCIP_Bool GCGmasterVarIsLinking(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   /* the master variable is a direct copy from an original variable */
   if( GCGvarGetBlock(var) == -1 )
   {
      SCIP_VAR** origvars;
      origvars = GCGmasterVarGetOrigvars(var);

      if( GCGmasterVarIsArtificial(var) )
         return FALSE;

      return GCGoriginalVarIsLinking(origvars[0]);
   }
   else
      return FALSE;
}

#ifndef NDEBUG
/** returns whether the master variable is a ray */
SCIP_Bool GCGmasterVarIsRay(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->data.mastervardata.isray;
}
#endif

#ifndef NDEBUG
/** returns TRUE or FALSE whether a master variable is an artificial variable */
SCIP_Bool GCGmasterVarIsArtificial(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   return vardata->data.mastervardata.isartificial;
}
#endif

#ifndef NDEBUG
/** returns the number of original variables the master variable is contained in */
int GCGmasterVarGetNOrigvars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.mastervardata.norigvars >= 0);
   assert(vardata->data.mastervardata.origvars != NULL || vardata->data.mastervardata.norigvars == 0);
   assert(vardata->data.mastervardata.origvals != NULL || vardata->data.mastervardata.norigvars == 0);
//   assert(vardata->blocknr != -1 || vardata->data.mastervardata.norigvars == 1 );

   return vardata->data.mastervardata.norigvars;
}
#endif

#ifndef NDEBUG
/** returns the original variables the master variable is contained in */
SCIP_VAR** GCGmasterVarGetOrigvars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.mastervardata.origvars != NULL || vardata->data.mastervardata.norigvars == 0);
//   assert(vardata->blocknr != -1 || vardata->data.mastervardata.origvars != NULL);
//   assert(vardata->blocknr != -1 || vardata->data.mastervardata.origvars[0] != NULL);
//   assert(vardata->blocknr != -1 || GCGvarGetBlock(vardata->data.mastervardata.origvars[0]) == -1
//      || GCGoriginalVarIsLinking(vardata->data.mastervardata.origvars[0]));


   return vardata->data.mastervardata.origvars;
}
#endif

#ifndef NDEBUG
/** returns the fraction of original variables the master variable is contained in */
SCIP_Real* GCGmasterVarGetOrigvals(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.mastervardata.origvals != NULL || vardata->data.mastervardata.norigvars == 0);
   return vardata->data.mastervardata.origvals;
}
#endif

/* returns the fraction of the original variable the master variable is contained in */
SCIP_Real GCGmasterVarGetOrigval(
   SCIP_VAR*             mastervar,         /**< master variable data structure */
   SCIP_VAR*             origvar            /**< original variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   SCIP_Real origval;
   assert(mastervar != NULL);
   assert(GCGvarIsMaster(mastervar));

   vardata = SCIPvarGetData(mastervar);
   assert(vardata != NULL);

   assert(vardata->data.mastervardata.origvar2val != NULL || vardata->data.mastervardata.norigvars == 0);

   if( vardata->data.mastervardata.origvar2val == NULL )
      return SCIP_INVALID;
   if( GCGoriginalVarIsLinking(origvar) )
   {
      if( !GCGisLinkingVarInBlock(origvar, vardata->blocknr))
         return SCIP_INVALID;
   }
   else if( vardata->blocknr != GCGvarGetBlock(origvar) )
      return SCIP_INVALID;

   origval = SCIPhashmapGetImageReal(vardata->data.mastervardata.origvar2val, origvar);
   return origval == SCIP_INVALID ? 0.0 : origval;
}

#ifndef NDEBUG
SCIP_HASHMAP* GCGmasterVarGetOrigvalmap(
   SCIP_VAR*             var         /**< master variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsMaster(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.mastervardata.origvar2val != NULL || vardata->data.mastervardata.norigvars == 0);
   return vardata->data.mastervardata.origvar2val;
}
#endif

#ifndef NDEBUG
/** returns the number of original variables the pricing variable is contained in */
int GCGpricingVarGetNOrigvars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsPricing(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.pricingvardata.norigvars >= 0);
   return vardata->data.pricingvardata.norigvars;
}
#endif

#ifndef NDEBUG
/** returns the original variables the pricing variable is contained in */
SCIP_VAR** GCGpricingVarGetOrigvars(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(GCGvarIsPricing(var));

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->data.pricingvardata.origvars != NULL);
   return vardata->data.pricingvardata.origvars;
}
#endif

#ifndef NDEBUG
/** returns the block of the variable */
int GCGvarGetBlock(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);

   assert(vardata->blocknr >= -2);
   return vardata->blocknr;
}
#endif

/** sets the block of the variable */
void GCGvarSetBlock(
   SCIP_VAR*             var,                /**< variable to set block for */
   int                   block               /**< block to set */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(block >= -1);

   vardata = SCIPvarGetData(var);
   assert(vardata != NULL);
   vardata->blocknr = block;
}

#ifndef NDEBUG
/** returns TRUE if the linking variable is in the block, FALSE otherwise */
SCIP_Bool GCGisLinkingVarInBlock(
   SCIP_VAR*             var,                /**< variabel data structure */
   int                   block               /**< pricing problem number */
   )
{
   SCIP_VAR** pricingvars;

   assert(var != NULL);
   assert(block >= 0);

   assert(GCGoriginalVarIsLinking(var));
   assert(GCGvarIsOriginal(var));

   pricingvars = GCGlinkingVarGetPricingVars(var);

   return pricingvars[block] != NULL;
}
#endif

/** determines if the master variable is in the given block */
SCIP_Bool GCGisMasterVarInBlock(
   SCIP_VAR*             mastervar,          /**< master variable */
   int                   block               /**< block number to check */
   )
{
   int varblock;

   assert(mastervar != NULL);
   assert(block >= 0);

   varblock = GCGvarGetBlock(mastervar);

   /* the master variable is a direct copy from an original variable */
   if( varblock == -1 )
   {
      SCIP_VAR** origvars;

      origvars = GCGmasterVarGetOrigvars(mastervar);

      if( GCGmasterVarIsArtificial(mastervar) )
         return FALSE;

      /* the corresponding original variable is a linking variable */
      if( GCGoriginalVarIsLinking(origvars[0]) )
         return GCGisLinkingVarInBlock(origvars[0], block);
      else
         return FALSE;
   }
   else
      return varblock == block;
}

/** informs an original variable, that a variable in the master problem was created,
 * that contains a part of the original variable.
 * Saves this information in the original variable's data
 * @todo this method needs a little love
 */
SCIP_RETCODE GCGoriginalVarAddMasterVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_VAR*             var,                /**< master variable */
   SCIP_Real             val                 /**< fraction of the original variable */
   )
{
   SCIP_VARDATA* vardata;

   assert(scip != NULL);
   assert(origvar != NULL);
   assert(var != NULL);
   assert(GCGisOriginal(scip));
   vardata = SCIPvarGetData(origvar);

   assert(vardata != NULL);
   assert(GCGvarIsOriginal(origvar));
   assert(vardata->data.origvardata.mastervars != NULL);
   assert(vardata->data.origvardata.mastervals != NULL);
   assert(vardata->data.origvardata.nmastervars >= 0);
   assert(vardata->data.origvardata.maxmastervars >= vardata->data.origvardata.nmastervars);

   /* realloc mastervars array of the original variable, if needed */
   if( vardata->data.origvardata.maxmastervars == vardata->data.origvardata.nmastervars )
   {
      int newsize = SCIPcalcMemGrowSize(scip, vardata->data.origvardata.nmastervars+1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(vardata->data.origvardata.mastervars), vardata->data.origvardata.maxmastervars,
            newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(vardata->data.origvardata.mastervals), vardata->data.origvardata.maxmastervars,
            newsize) );
      SCIPdebugMessage("mastervars array of var %s resized from %d to %d\n", SCIPvarGetName(origvar),
         vardata->data.origvardata.maxmastervars, newsize);
      vardata->data.origvardata.maxmastervars = newsize;
   }
   /* add information to the original variable's vardata */
   vardata->data.origvardata.mastervars[vardata->data.origvardata.nmastervars] = var;
   vardata->data.origvardata.mastervals[vardata->data.origvardata.nmastervars] = val;
   vardata->data.origvardata.nmastervars++;

   return SCIP_OKAY;
}

/** informs an original variable, that a variable in the master problem was deleted,
 * that contains a part of the original variable.
 * Update the information in the original variable's data
 * @todo this method needs a little love
 */
SCIP_RETCODE GCGoriginalVarRemoveMasterVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_VAR*             var                 /**< master variable */
   )
{
   SCIP_VARDATA* vardata;
   int i;

   assert(scip != NULL);
   assert(origvar != NULL);
   assert(var != NULL);

   vardata = SCIPvarGetData(origvar);

   assert(vardata != NULL);
   assert(GCGvarIsOriginal(origvar));
   assert(vardata->data.origvardata.mastervars != NULL);
   assert(vardata->data.origvardata.mastervals != NULL);
   assert(vardata->data.origvardata.nmastervars > 0);
   assert(vardata->data.origvardata.maxmastervars >= vardata->data.origvardata.nmastervars);

   for( i = 0; i < vardata->data.origvardata.nmastervars; ++i )
   {
      if( vardata->data.origvardata.mastervars[i] == var )
      {
         vardata->data.origvardata.mastervars[i] = vardata->data.origvardata.mastervars[vardata->data.origvardata.nmastervars - 1];
         vardata->data.origvardata.mastervals[i] = vardata->data.origvardata.mastervals[vardata->data.origvardata.nmastervars - 1];
         (vardata->data.origvardata.nmastervars)--;

         break;
      }
   }
   assert(i <= vardata->data.origvardata.nmastervars);
#ifndef NDEBUG
   for( ; i < vardata->data.origvardata.nmastervars; ++i )
   {
      assert(vardata->data.origvardata.mastervars[i] != var);
   }
#endif

   return SCIP_OKAY;
}

/** creates the corresponding pricing variable for the given original variable */
SCIP_RETCODE GCGoriginalVarCreatePricingVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_VAR**            var                 /**< pricing variable */
   )
{
   SCIP_VARDATA* vardata;
   char name[SCIP_MAXSTRLEN];
   int pricingprobnr;
   assert(scip != NULL);
   assert(origvar != NULL);
   assert(var != NULL);
   assert(GCGvarIsOriginal(origvar));
   assert(!GCGoriginalVarIsLinking(origvar));
   assert(GCGoriginalVarGetPricingVar(origvar) == NULL);

   /* get the number of the pricing block to which the variable belongs */
   pricingprobnr = GCGvarGetBlock(origvar);

   /* create variable data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &vardata) );
   vardata->vartype = GCG_VARTYPE_PRICING;
   vardata->blocknr = pricingprobnr;
   vardata->data.pricingvardata.maxorigvars = STARTMAXORIGVARS;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(vardata->data.pricingvardata.origvars), vardata->data.pricingvardata.maxorigvars) ); /*lint !e506*/
   vardata->data.pricingvardata.origvars[0] = origvar;
   vardata->data.pricingvardata.norigvars = 1;

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "pr%d_%s", pricingprobnr, SCIPvarGetName(origvar));
   SCIP_CALL( SCIPcreateVar(scip, var, name, SCIPvarGetLbGlobal(origvar),
         SCIPvarGetUbGlobal(origvar), 0.0, SCIPvarGetType(origvar),
         TRUE, FALSE, GCGvarDelOrig, NULL, NULL, NULL, vardata) );

   return SCIP_OKAY;
}

/** creates the corresponding pricing variable for the given original variable */
SCIP_RETCODE GCGlinkingVarCreatePricingVar(
   SCIP*                 pricingscip,        /**< pricing problem SCIP data structure */
   int                   pricingprobnr,      /**< number of the pricing problem */
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_VAR**            var                 /**< pointer to store new pricing variable */
   )
{
   SCIP_VARDATA* vardata;
   char name[SCIP_MAXSTRLEN];

   assert(pricingscip != NULL);
   assert(pricingprobnr >= 0);
   assert(origvar != NULL);
   assert(GCGoriginalVarIsLinking(origvar));
   assert(var != NULL);

   /* create variable data */
   SCIP_CALL( SCIPallocBlockMemory(pricingscip, &vardata) );
   vardata->vartype = GCG_VARTYPE_PRICING;
   vardata->blocknr = pricingprobnr;
   vardata->data.pricingvardata.maxorigvars = STARTMAXORIGVARS;
   SCIP_CALL( SCIPallocBlockMemoryArray(pricingscip, &(vardata->data.pricingvardata.origvars), vardata->data.pricingvardata.maxorigvars) ); /*lint !e506*/
   vardata->data.pricingvardata.origvars[0] = origvar;
   vardata->data.pricingvardata.norigvars = 1;

   /* create and add variable */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "pr%d_%s", pricingprobnr, SCIPvarGetName(origvar));
   SCIP_CALL( SCIPcreateVar(pricingscip, var, name, SCIPvarGetLbGlobal(origvar),
         SCIPvarGetUbGlobal(origvar), 0.0, SCIPvarGetType(origvar),
         TRUE, FALSE, GCGvarDelOrig, NULL, NULL, NULL, vardata) );

   return SCIP_OKAY;
}

/** creates the corresponding constraint in the master problem for the linking variable */
SCIP_RETCODE GCGlinkingVarCreateMasterCons(
   SCIP*                 masterscip,         /**< msater problem SCIP data structure */
   int                   pricingprobnr,      /**< number of the pricing problem */
   SCIP_VAR*             origvar,            /**< original variable */
   SCIP_CONS**           linkcons            /**< constraint linking pricing variables */
   )
{
   char name[SCIP_MAXSTRLEN];

   assert(masterscip != NULL);
   assert(pricingprobnr >= 0);
   assert(origvar != NULL);
   assert(GCGoriginalVarIsLinking(origvar));
   assert(linkcons != NULL);

   /* add corresponding linking constraint to the master problem */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "l_%s_%d", SCIPvarGetName(origvar), pricingprobnr);
   SCIP_CALL( SCIPcreateConsLinear(masterscip, linkcons, name, 0, NULL, NULL, 0.0, 0.0,
         TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIP_OKAY;
}

/** creates the master var and initializes the vardata */
SCIP_RETCODE GCGcreateMasterVar(
   SCIP*                 scip,               /**< master SCIP data structure */
   SCIP*                 origscip,           /**< original SCIP data structure */
   SCIP*                 pricingscip,        /**< pricing problem SCIP data structure */
   SCIP_VAR**            newvar,             /**< pointer to store new master variable */
   const char*           varname,            /**< new variable name */
   SCIP_Real             objcoeff,           /**< new objective coefficient */
   SCIP_VARTYPE          vartype,            /**< new variable type */
   SCIP_Bool             solisray,           /**< indicates whether new variable is a ray */
   int                   prob,               /**< number of pricing problem that created this variable */
   int                   nsolvars,           /**< number of variables in the solution */
   SCIP_Real*            solvals,            /**< values of variables in the solution */
   SCIP_VAR**            solvars,            /**< variables with non zero coefficient in the solution */
   SCIP_Bool             auxiliaryvar        /**< is new variable an Benders' auxiliary variables? */
   )
{
   SCIP_VARDATA* newvardata;
   SCIP_Real lb;
   int i;
   int j;
   SCIP_Bool trivialsol;

   assert(scip != NULL);
   assert(pricingscip != NULL);
   assert(newvar != NULL);
   assert(varname != NULL);
   assert(!SCIPisInfinity(pricingscip, ABS(objcoeff)));
   assert(vartype == SCIP_VARTYPE_INTEGER || vartype == SCIP_VARTYPE_CONTINUOUS);
   assert(prob >= 0);
   assert(nsolvars >= 0);
   assert(solvals != NULL || nsolvars == 0);
   assert(solvars != NULL || nsolvars == 0);

   trivialsol = FALSE;

   lb = 0.0;
   if( auxiliaryvar )
      lb = -SCIPinfinity(scip);

   /* create data for the new variable in the master problem */
   SCIP_CALL( SCIPallocBlockMemory(scip, &newvardata) );
   newvardata->vartype = GCG_VARTYPE_MASTER;
   newvardata->blocknr = prob;
   newvardata->data.mastervardata.index = -1;

   /* store whether the variable represents a ray */
   newvardata->data.mastervardata.isray = solisray;
   newvardata->data.mastervardata.isartificial = FALSE;

   /* create variable in the master problem */
   SCIP_CALL( SCIPcreateVar(scip, newvar, varname, lb, SCIPinfinity(scip), /* GCGrelaxGetNIdenticalBlocks(origprob, prob) */
         objcoeff, vartype, TRUE, TRUE, NULL, NULL, gcgvardeltrans, NULL, newvardata) );

   /* count number of non-zeros */
   newvardata->data.mastervardata.norigvars = 0;
   newvardata->data.mastervardata.maxorigvars = 0;

   for( i = 0; i < nsolvars; i++ )
   {
      assert(solvars != NULL);
      assert(solvals != NULL);

      assert(!SCIPisInfinity(scip, solvals[i]));
      if( !SCIPisZero(scip, solvals[i]) )
      {
         newvardata->data.mastervardata.norigvars++;
      }
   }

   /*
    * if we have not added any original variable to the mastervariable, all coefficients were 0.
    * In that case, we will add all variables in the pricing problem
    */
   if( newvardata->data.mastervardata.norigvars == 0 && !auxiliaryvar )
   {
      newvardata->data.mastervardata.norigvars = SCIPgetNOrigVars(pricingscip);
      trivialsol = TRUE;
   }

   if( newvardata->data.mastervardata.norigvars > 0 )
   {
      newvardata->data.mastervardata.maxorigvars = SCIPcalcMemGrowSize(scip, newvardata->data.mastervardata.norigvars);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(newvardata->data.mastervardata.origvars), newvardata->data.mastervardata.maxorigvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(newvardata->data.mastervardata.origvals), newvardata->data.mastervardata.maxorigvars) );
      SCIP_CALL( SCIPhashmapCreate(&(newvardata->data.mastervardata.origvar2val), SCIPblkmem(scip), newvardata->data.mastervardata.norigvars) );
   }
   else
   {
      newvardata->data.mastervardata.origvars = NULL;
      newvardata->data.mastervardata.origvals = NULL;
      newvardata->data.mastervardata.origvar2val = NULL;
   }

   /* number of original variables already saved in mastervardata */
   j = 0;

   /* update variable datas */
   for( i = 0; i < nsolvars && !trivialsol; i++ )
   {
      SCIP_Real solval;

      assert(solvars != NULL);
      assert(solvals != NULL);

      solval = solvals[i];

      if( !SCIPisZero(scip, solval) )
      {
         SCIP_VAR* origvar;
         assert(GCGvarIsPricing(solvars[i]));

         origvar = GCGpricingVarGetOrigvars(solvars[i])[0];
         assert(origvar != NULL);

         assert(newvardata->data.mastervardata.origvars != NULL);
         assert(newvardata->data.mastervardata.origvals != NULL);
         assert(GCGvarIsOriginal(origvar));
         assert(!solisray || vartype == SCIP_VARTYPE_CONTINUOUS || SCIPisIntegral(scip, solval) || SCIPvarGetType(solvars[i]) == SCIP_VARTYPE_CONTINUOUS);

         /* round solval if possible to avoid numerical troubles */
         if( SCIPvarIsIntegral(solvars[i]) && SCIPisIntegral(scip, solval) )
            solval = SCIPround(scip, solval);

         /* save in the master problem variable's data the quota of the corresponding original variable */
         newvardata->data.mastervardata.origvars[j] = origvar;
         newvardata->data.mastervardata.origvals[j] = solval;
         SCIPhashmapInsertReal(newvardata->data.mastervardata.origvar2val, origvar, solval);
         /* save the quota in the original variable's data */
         SCIP_CALL( GCGoriginalVarAddMasterVar(origscip, origvar, *newvar, solval) );
         j++;
      }
   }
   if( trivialsol )
   {
      SCIP_VAR** pricingvars;
      int npricingvars;

      pricingvars = SCIPgetOrigVars(pricingscip);
      npricingvars = SCIPgetNOrigVars(pricingscip);
      for( j = 0; j < npricingvars; ++j )
      {
         SCIP_VAR* origvar;
         assert(GCGvarIsPricing(pricingvars[j]));

         origvar = GCGpricingVarGetOrigvars(pricingvars[j])[0];
         assert(origvar != NULL);

         assert(newvardata->data.mastervardata.origvars != NULL);
         assert(newvardata->data.mastervardata.origvals != NULL);
         assert(GCGvarIsOriginal(origvar));
         /* save in the master problem variable's data the quota of the corresponding original variable */
         newvardata->data.mastervardata.origvars[j] = origvar;
         newvardata->data.mastervardata.origvals[j] = 0.0;
         SCIPhashmapInsertReal(newvardata->data.mastervardata.origvar2val, origvar, 0.0);
         /* save the quota in the original variable's data */
         SCIP_CALL( GCGoriginalVarAddMasterVar(origscip, origvar, *newvar, 0.0) );
      }
   }
   assert(j == newvardata->data.mastervardata.norigvars);
return SCIP_OKAY;
}

/** creates initial master variables and the vardata */
SCIP_RETCODE GCGcreateInitialMasterVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< original variable */
   SCIP_VAR**            newvar              /**< pointer to store new variable */

   )
{
   SCIP_VARDATA* newvardata;
   int blocknr;

   blocknr = GCGvarGetBlock(var);
   assert( blocknr == -1 || blocknr == -2
      || GCGgetMasterDecompMode(scip) == DEC_DECMODE_BENDERS || GCGgetMasterDecompMode(scip) == DEC_DECMODE_ORIGINAL);

   if( blocknr == -1 )
   {
      SCIPdebugMessage("var %s is in no block - copy it directly to the master\n", SCIPvarGetName(var));
   }
   else
   {
      SCIPdebugMessage("var %s is a linking variable - copy it directly to the master\n", SCIPvarGetName(var));
   }

   /* create vardata */
   SCIP_CALL( SCIPallocBlockMemory(scip, &newvardata) );
   newvardata->vartype = GCG_VARTYPE_MASTER;
   newvardata->blocknr = -1;
   newvardata->data.mastervardata.isray = FALSE;
   newvardata->data.mastervardata.isartificial = FALSE;
   newvardata->data.mastervardata.norigvars = 1;
   newvardata->data.mastervardata.maxorigvars = 1;
   newvardata->data.mastervardata.index = -1;

   /* save corresoponding origvar */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(newvardata->data.mastervardata.origvars), 1) ); /*lint !e506*/
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(newvardata->data.mastervardata.origvals), 1) ); /*lint !e506*/
   SCIP_CALL( SCIPhashmapCreate(&(newvardata->data.mastervardata.origvar2val), SCIPblkmem(scip), 1) );
   newvardata->data.mastervardata.origvars[0] = var;
   newvardata->data.mastervardata.origvals[0] = 1.0;
   SCIPhashmapInsertReal(newvardata->data.mastervardata.origvar2val, var, 1.0);

   /* create variable in the master problem */
   SCIP_CALL( SCIPcreateVar(scip, newvar, SCIPvarGetName(var),
         SCIPvarGetLbGlobal(var), SCIPvarGetUbGlobal(var), SCIPvarGetObj(var), SCIPvarGetType(var),
         TRUE, TRUE, NULL, NULL, gcgvardeltrans, NULL, newvardata) );

   return SCIP_OKAY;
}

/** creates artificial variable and the vardata */
SCIP_RETCODE GCGcreateArtificialVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            newvar,             /**< pointer to store new variable */
   const char*           name,               /**< name of variable, or NULL for automatic name creation */
   SCIP_Real             objcoef             /**< objective coefficient of artificial variable */
   )
{
   SCIP_VARDATA* newvardata;

   /* create vardata */
   SCIP_CALL( SCIPallocBlockMemory(scip, &newvardata) );
   newvardata->vartype = GCG_VARTYPE_MASTER;
   newvardata->blocknr = -1;
   newvardata->data.mastervardata.isray = FALSE;
   newvardata->data.mastervardata.norigvars = 0;
   newvardata->data.mastervardata.maxorigvars = 0;
   newvardata->data.mastervardata.isartificial = TRUE;

   newvardata->data.mastervardata.origvars = NULL;
   newvardata->data.mastervardata.origvals = NULL;
   newvardata->data.mastervardata.origvar2val = NULL;
   newvardata->data.mastervardata.index = -1;

   /* create variable in the master problem */
   SCIP_CALL( SCIPcreateVar(scip, newvar, name,
         0.0, SCIPinfinity(scip), objcoef, SCIP_VARTYPE_IMPLINT,
         TRUE, TRUE, NULL, NULL, gcgvardeltrans, NULL, newvardata) );

   return SCIP_OKAY;
}

/* adds the vardata to the auxiliary variable */
SCIP_RETCODE GCGaddDataAuxiliaryVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             auxiliaryvar,       /**< the auxiliary variable */
   int                   probnumber          /**< the subproblem number */
   )
{
   SCIP_VARDATA* newvardata;

   assert(scip != NULL);
   assert(auxiliaryvar != NULL);

   /* create data for the new variable in the master problem */
   SCIP_CALL( SCIPallocBlockMemory(scip, &newvardata) );
   newvardata->vartype = GCG_VARTYPE_MASTER;
   newvardata->blocknr = probnumber;

   /* store whether the variable represents a ray */
   newvardata->data.mastervardata.isray = FALSE;

   /* count number of non-zeros */
   newvardata->data.mastervardata.norigvars = 0;
   newvardata->data.mastervardata.maxorigvars = 0;

   newvardata->data.mastervardata.origvars = NULL;
   newvardata->data.mastervardata.origvals = NULL;
   newvardata->data.mastervardata.origvar2val = NULL;

   /* setting the variable data */
   SCIPvarSetData(auxiliaryvar, newvardata);

   /* setting the deltrans callback */
   SCIPvarSetDeltransData(auxiliaryvar, gcgvardeltrans);

   return SCIP_OKAY;
}

/** set creation node of variable */
void GCGsetCreationNode(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Longint          creationNode        /**< node in which the variable is created */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(creationNode >= 0);

   vardata = SCIPvarGetData(var);
   vardata->creationnode = creationNode;
}

#ifndef NDEBUG
/** return creation node of variable */
long long int GCGgetCreationNode(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->creationnode;
}
#endif

/** store creation time */
void GCGsetCreationTime(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Real             time                /**< time at which the variable is created */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(time >= 0.0);

   vardata = SCIPvarGetData(var);
   vardata->creationtime = time;
}

#ifndef NDEBUG
/** return stored creation time */
SCIP_Real GCGgetCreationTime(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->creationtime;
}
#endif

/** store pricing reduced cost call */
void GCGsetRootRedcostCall(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Longint          rootredcostcall     /**< iteration at which the variable is created */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(rootredcostcall >= -1);

   vardata = SCIPvarGetData(var);
   vardata->rootredcostcall = rootredcostcall;
}

#ifndef NDEBUG
/** return stored pricing reduced cost call */
SCIP_Longint GCGgetRootRedcostCall(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->rootredcostcall;
}
#endif

/** store iteration */
void GCGsetIteration(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Longint          iteration           /**< iteration at which the variable is created */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(iteration >= 0);

   vardata = SCIPvarGetData(var);
   vardata->iteration = iteration;
}

#ifndef NDEBUG
/** return stored iteration */
SCIP_Longint GCGgetIteration(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->iteration;
}
#endif

/** store gap */
void GCGsetVarGap(
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Real             gap                 /**< present gap when variable is created */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);
   assert(gap >= 0.0);

   vardata = SCIPvarGetData(var);
   vardata->gap = gap;
}

#ifndef NDEBUG
/** return stored gap */
SCIP_Real GCGgetVarGap(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->gap;
}
#endif

/** store reduced cost */
void GCGsetRedcost(
   SCIP*                 scip,               /**< master SCIP data structure */
   SCIP_VAR*             var,                /**< variable data structure */
   SCIP_Real             redcost             /**< reduced cost of the variable at creation */
   )
{
   SCIP_VARDATA* vardata;
   assert(scip != NULL);
   assert(var != NULL);
   assert(SCIPisLE(scip, redcost, 0.0));

   vardata = SCIPvarGetData(var);
   vardata->redcost = redcost;
}

#ifndef NDEBUG
/** return stored reduced cost */
SCIP_Real GCGgetRedcost(
   SCIP_VAR*             var                 /**< variable data structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->redcost;
}
#endif

/** updates the statistics part of the variable */
void GCGupdateVarStatistics(
    SCIP*                scip,               /**< master SCIP data structure */
    SCIP*                origprob,           /**< original SCIP data structure */
    SCIP_VAR*            newvar,             /**< new variable for statistic update */
    SCIP_Real            redcost             /**< reduced cost of the variable */
    )
{
   SCIP_Longint redcostcall;
   assert(scip != NULL);
   assert(GCGisMaster(scip));
   assert(origprob != NULL);
   assert(GCGisOriginal(origprob));
   assert(newvar != NULL);

   redcostcall = -1;
   GCGsetCreationNode(newvar, SCIPnodeGetNumber(SCIPgetCurrentNode(origprob)));
   GCGsetCreationTime(newvar, SCIPgetSolvingTime(scip));

   GCGsetRootRedcostCall(newvar, redcostcall);
   GCGsetIteration(newvar, SCIPgetNLPIterations(scip));
   GCGsetVarGap(newvar, MIN(SCIPgetGap(origprob), SCIPgetGap(scip))); /*lint !e666*/
   GCGsetRedcost(origprob, newvar, redcost);

}

/**  prints the given variable: name, type (original, master or pricing) block number,
 * and the list of all variables related to the given variable
 */
void GCGprintVar(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< File to write information to, or NULL for stdout */
   SCIP_VAR*             var                 /**< variable that should be printed */
   )
{
   int i;
   int blocknr;
   assert(GCGvarIsOriginal(var) || GCGvarIsMaster(var) || GCGvarIsPricing(var));

   blocknr = GCGvarGetBlock(var);

   if( GCGvarIsOriginal(var) )
   {
      SCIP_VAR** mastervars;
      SCIP_Real* mastervals;
      int  nmastervars;

      if( GCGoriginalVarIsLinking(var) )
      {
         SCIP_VAR** pricingvars;
         int nblocks;
         int j;
         pricingvars = GCGlinkingVarGetPricingVars(var);
         nblocks = GCGlinkingVarGetNBlocks(var);
         SCIPinfoMessage(scip, file, "Variable %s (linking): %d block%s (", SCIPvarGetName(var), nblocks, nblocks == 1 ? "":"s" );
         /*lint --e{440}*/
         for( i = 0, j = 0; j < nblocks; ++i )
         {
            if( pricingvars[i] != NULL )
            {
               SCIPinfoMessage(scip, file, "%d ", i);
               ++j;
            }
         }
         SCIPinfoMessage(scip, file, ")\n");
      }
      else
      {
         SCIPinfoMessage(scip, file, "Variable %s (original): block %d\n", SCIPvarGetName(var), blocknr);
      }

      mastervars = GCGoriginalVarGetMastervars(var);
      mastervals = GCGoriginalVarGetMastervals(var);
      nmastervars = GCGoriginalVarGetNMastervars(var);
      SCIPinfoMessage(scip, file, "mastervars:");
      for( i = 0; i < nmastervars-1; i++ )
      {
         SCIPinfoMessage(scip, file, "%s (%g), ", SCIPvarGetName(mastervars[i]), mastervals[i]);
      }
      SCIPinfoMessage(scip, file, "%s (%g)\n", SCIPvarGetName(mastervars[nmastervars-1]), mastervals[nmastervars-1]);
   }
   else if( GCGvarIsPricing(var) )
   {
      SCIP_VAR** origvars;
      int  norigvars;

      origvars = GCGpricingVarGetOrigvars(var);
      norigvars = GCGpricingVarGetNOrigvars(var);

      SCIPinfoMessage(scip, file, "Variable %s (pricing): block %d\n", SCIPvarGetName(var), blocknr);
      SCIPinfoMessage(scip, file, "origvars:");
      for( i = 0; i < norigvars-1; i++ )
      {
         SCIPinfoMessage(scip, file, "%s, ", SCIPvarGetName(origvars[i]));
      }
      SCIPinfoMessage(scip, file, "%s\n", SCIPvarGetName(origvars[norigvars-1]));
   }
   else if( GCGvarIsMaster(var) )
   {
      SCIP_VAR** origvars;
      int  norigvars;
      SCIP_Real* origvals;

      origvars = GCGmasterVarGetOrigvars(var);
      norigvars = GCGmasterVarGetNOrigvars(var);
      origvals = GCGmasterVarGetOrigvals(var);
      SCIPinfoMessage(scip, file, "Variable %s (master): block %d\n", SCIPvarGetName(var), blocknr);
      SCIPinfoMessage(scip, file, "origvars:");
      for( i = 0; i < norigvars-1; i++ )
      {
         SCIPinfoMessage(scip, file, "%s (%g), ", SCIPvarGetName(origvars[i]), origvals[i]);
      }
      SCIPinfoMessage(scip, file, "%s (%g)\n", SCIPvarGetName(origvars[norigvars-1]), origvals[norigvars-1]);
   }
}

#ifndef NDEBUG
/* returns the index of the mater variable */
int GCGmasterVarGetIndex(
   SCIP_VAR*             var                 /**< SCIP variable structure */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   return vardata->data.mastervardata.index;
}
#endif

/* sets the index of the master variable */
void GCGmasterVarSetIndex(
   SCIP_VAR*             var,                /**< SCIP variable structure */
   int                   index               /**< index */
   )
{
   SCIP_VARDATA* vardata;
   assert(var != NULL);

   vardata = SCIPvarGetData(var);
   vardata->data.mastervardata.index = index;
}
