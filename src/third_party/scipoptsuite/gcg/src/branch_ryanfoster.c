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

/**@file   branch_ryanfoster.c
 * @brief  branching rule for original problem in GCG implementing the Ryan and Foster branching scheme
 * @author Gerald Gamrath
 * @author Oliver Gaul
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/*#define SCIP_DEBUG*/
#include <assert.h>
#include <string.h>

#include "branch_ryanfoster.h"
#include "branch_bpstrong.h"
#include "gcg.h"
#include "relax_gcg.h"
#include "cons_masterbranch.h"
#include "cons_origbranch.h"
#include "scip/nodesel_bfs.h"
#include "scip/nodesel_dfs.h"
#include "scip/nodesel_estimate.h"
#include "scip/nodesel_hybridestim.h"
#include "scip/nodesel_restartdfs.h"
#include "scip/branch_allfullstrong.h"
#include "scip/branch_fullstrong.h"
#include "scip/branch_inference.h"
#include "scip/branch_mostinf.h"
#include "scip/branch_leastinf.h"
#include "scip/branch_pscost.h"
#include "scip/branch_random.h"
#include "scip/branch_relpscost.h"
#include "pricer_gcg.h"
#include "scip/cons_varbound.h"
#include "type_branchgcg.h"

#define BRANCHRULE_NAME          "ryanfoster"
#define BRANCHRULE_DESC          "ryan and foster branching in generic column generation"
#define BRANCHRULE_PRIORITY      10
#define BRANCHRULE_MAXDEPTH      -1
#define BRANCHRULE_MAXBOUNDDIST  1.0

/* strong branching */
#define DEFAULT_USESTRONG        FALSE

#define DEFAULT_MINPHASE0OUTCANDS      10
#define DEFAULT_MAXPHASE0OUTCANDS      50
#define DEFAULT_MAXPHASE0OUTCANDSFRAC  0.7
#define DEFAULT_PHASE1GAPWEIGHT        0.25

#define DEFAULT_MINPHASE1OUTCANDS      3
#define DEFAULT_MAXPHASE1OUTCANDS      20
#define DEFAULT_MAXPHASE1OUTCANDSFRAC  0.7
#define DEFAULT_PHASE2GAPWEIGHT        1
/**/


/** branching data for branching decisions */
struct GCG_BranchData
{
   SCIP_VAR*             var1;               /**< first original variable on which the branching is done */
   SCIP_VAR*             var2;               /**< second original variable on which the branching is done */
   SCIP_Bool             same;               /**< should each master var contain either both or none of the vars? */
   int                   blocknr;            /**< number of the block in which branching was performed */
   SCIP_CONS*            pricecons;          /**< constraint enforcing the branching restriction in the pricing problem */
};



/*
 * Callback methods for enforcing branching constraints
 */


/** copy method for master branching rule */
static
SCIP_DECL_BRANCHCOPY(branchCopyRyanfoster)
{
   assert(branchrule != NULL);
   assert(scip != NULL);
   SCIPdebugMessage("pricer copy called.\n");

   return SCIP_OKAY;
}


/** callback activation method */
static
GCG_DECL_BRANCHACTIVEMASTER(branchActiveMasterRyanfoster)
{
   SCIP* origscip;
   SCIP* pricingscip;

   assert(scip != NULL);
   assert(branchdata != NULL);
   assert(branchdata->var1 != NULL);
   assert(branchdata->var2 != NULL);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   pricingscip = GCGgetPricingprob(origscip, branchdata->blocknr);
   assert(pricingscip != NULL);

   SCIPdebugMessage("branchActiveMasterRyanfoster: %s(%s, %s)\n", ( branchdata->same ? "same" : "differ" ),
      SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2));

   assert(GCGvarIsOriginal(branchdata->var1));
   /** @todo it is not clear if linking variables interfere with ryan foster branching */
   assert(GCGvarGetBlock(branchdata->var1) == branchdata->blocknr);

   assert(GCGvarIsOriginal(branchdata->var2));
   assert(GCGvarGetBlock(branchdata->var2) == branchdata->blocknr);

   /* create corresponding constraint in the pricing problem, if not yet created */
   if( branchdata->pricecons == NULL )
   {
      char name[SCIP_MAXSTRLEN];

      if( branchdata->same )
      {
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "same(%s,%s)", SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2));

         SCIP_CALL( SCIPcreateConsVarbound(pricingscip,
               &(branchdata->pricecons), name, GCGoriginalVarGetPricingVar(branchdata->var1),
               GCGoriginalVarGetPricingVar(branchdata->var2), -1.0, 0.0, 0.0,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
      }
      else
      {
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "differ(%s,%s)", SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2));

         SCIP_CALL( SCIPcreateConsVarbound(pricingscip,
               &(branchdata->pricecons), name, GCGoriginalVarGetPricingVar(branchdata->var1),
               GCGoriginalVarGetPricingVar(branchdata->var2), 1.0, -SCIPinfinity(scip), 1.0,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
      }
   }
   /* add constraint to the pricing problem that enforces the branching decision */
   SCIP_CALL( SCIPaddCons(pricingscip, branchdata->pricecons) );

   return SCIP_OKAY;
}

/** callback deactivation method */
static
GCG_DECL_BRANCHDEACTIVEMASTER(branchDeactiveMasterRyanfoster)
{
   SCIP* origscip;
   SCIP* pricingscip;

   assert(scip != NULL);
    assert(branchdata != NULL);
   assert(branchdata->var1 != NULL);
   assert(branchdata->var2 != NULL);
   assert(branchdata->pricecons != NULL);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   pricingscip = GCGgetPricingprob(origscip, branchdata->blocknr);
   assert(pricingscip != NULL);

   SCIPdebugMessage("branchDeactiveMasterRyanfoster: %s(%s, %s)\n", ( branchdata->same ? "same" : "differ" ),
      SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2));

   /* remove constraint from the pricing problem that enforces the branching decision */
   assert(branchdata->pricecons != NULL);
   SCIP_CALL( SCIPdelCons(pricingscip, branchdata->pricecons) );

   return SCIP_OKAY;
}

/** callback propagation method */
static
GCG_DECL_BRANCHPROPMASTER(branchPropMasterRyanfoster)
{
   SCIP_VAR** vars;
   SCIP_Real val1;
   SCIP_Real val2;
   int nvars;
   int propcount;
   int i;
   int j;

   assert(scip != NULL);
   assert(branchdata != NULL);
   assert(branchdata->var1 != NULL);
   assert(branchdata->var2 != NULL);
   assert(branchdata->pricecons != NULL);

   assert(GCGmasterGetOrigprob(scip) != NULL);

   SCIPdebugMessage("branchPropMasterRyanfoster: %s(%s, %s)\n", ( branchdata->same ? "same" : "differ" ),
      SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2));

   *result = SCIP_DIDNOTFIND;

   propcount = 0;

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   /* iterate over all master variables */
   for( i = 0; i < nvars; i++ )
   {
      int norigvars;
      SCIP_Real* origvals;
      SCIP_VAR** origvars;

      origvars = GCGmasterVarGetOrigvars(vars[i]);
      origvals = GCGmasterVarGetOrigvals(vars[i]);
      norigvars = GCGmasterVarGetNOrigvars(vars[i]);

      /* only look at variables not fixed to 0 */
      if( !SCIPisFeasZero(scip, SCIPvarGetUbLocal(vars[i])) )
      {
         assert(GCGvarIsMaster(vars[i]));

         /* if variable belongs to a different block than the branching restriction, we do not have to look at it */
         if( branchdata->blocknr != GCGvarGetBlock(vars[i]) )
            continue;

         /* save the values of the original variables for the current master variable */
         val1 = 0.0;
         val2 = 0.0;
         for( j = 0; j < norigvars; j++ )
         {
            if( origvars[j] == branchdata->var1 )
            {
               val1 = origvals[j];
               continue;
            }
            if( origvars[j] == branchdata->var2 )
            {
               val2 = origvals[j];
            }
         }

         /* if branching enforces that both original vars are either both contained or none of them is contained
          * and the current master variable has different values for both of them, fix the variable to 0 */
         if( branchdata->same && !SCIPisEQ(scip, val1, val2) )
         {
            SCIP_CALL( SCIPchgVarUb(scip, vars[i], 0.0) );
            propcount++;
         }
         /* if branching enforces that both original vars must be in different mastervars, fix all
          * master variables to 0 that contain both */
         if( !branchdata->same && SCIPisEQ(scip, val1, 1.0) && SCIPisEQ(scip, val2, 1.0) )
         {
            SCIP_CALL( SCIPchgVarUb(scip, vars[i], 0.0) );
            propcount++;
         }
      }
   }

   SCIPdebugMessage("Finished propagation of branching decision constraint: %s(%s, %s), %d vars fixed.\n",
      ( branchdata->same ? "same" : "differ" ), SCIPvarGetName(branchdata->var1), SCIPvarGetName(branchdata->var2), propcount);

   if( propcount > 0 )
   {
      *result = SCIP_REDUCEDDOM;
   }

   return SCIP_OKAY;
}

/** callback deletion method for branching data*/
static
GCG_DECL_BRANCHDATADELETE(branchDataDeleteRyanfoster)
{
   assert(scip != NULL);
   assert(branchdata != NULL);

   SCIPdebugMessage("branchDataDeleteRyanfoster: %s(%s, %s)\n", ( (*branchdata)->same ? "same" : "differ" ),
      SCIPvarGetName((*branchdata)->var1), SCIPvarGetName((*branchdata)->var2));

   /* release constraint that enforces the branching decision */
   if( (*branchdata)->pricecons != NULL )
   {
      SCIP_CALL( SCIPreleaseCons(GCGgetPricingprob(scip, (*branchdata)->blocknr),
            &(*branchdata)->pricecons) );
   }

   SCIPfreeBlockMemory(scip, branchdata);
   *branchdata = NULL;

   return SCIP_OKAY;
}

/*
 * Callback methods
 */

/** for the two given original variables, create two Ryan&Foster branching nodes, one for same, one for differ */
static
SCIP_RETCODE createChildNodesRyanfoster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR*             ovar1,              /**< first original variable */
   SCIP_VAR*             ovar2,              /**< second original variable */
   int                   blocknr,            /**< number of the pricing block */
   SCIP_Bool             sameinf,            /**< is the samebranch known to be infeasible? */
   SCIP_Bool             differinf           /**< is the differbranch known to be infeasible? */
   )
{
   SCIP* masterscip;
   SCIP_VAR* pricingvar1;
   SCIP_VAR* pricingvar2;
   GCG_BRANCHDATA* branchsamedata;
   GCG_BRANCHDATA* branchdifferdata;
   char samename[SCIP_MAXSTRLEN];
   char differname[SCIP_MAXSTRLEN];

   SCIP_VAR** origvars1;
   SCIP_VAR** origvars2;
   int norigvars1;
   int maxorigvars1;
   int v;

   SCIP_NODE* child1;
   SCIP_NODE* child2;
   SCIP_CONS* cons1;
   SCIP_CONS* cons2;
   SCIP_CONS** origbranchconss1;
   SCIP_CONS** origbranchconss2;


   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(ovar1 != NULL);
   assert(ovar2 != NULL);
   assert(GCGvarIsOriginal(ovar1));
   assert(GCGvarIsOriginal(ovar2));

   origbranchconss1 = NULL;
   origbranchconss2 = NULL;

   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   SCIPdebugMessage("Ryanfoster branching rule: branch on original variables %s and %s!\n",
      SCIPvarGetName(ovar1), SCIPvarGetName(ovar2));

   /* for cons_masterbranch */

   if(!sameinf)
   {
      /* create child-node of the current node in the b&b-tree */
      SCIP_CALL( SCIPcreateChild(masterscip, &child1, 0.0, SCIPgetLocalTransEstimate(masterscip)) );

      /* allocate branchdata and store information */
      SCIP_CALL( SCIPallocBlockMemory(scip, &branchsamedata) );
      branchsamedata->var1 = ovar1;
      branchsamedata->var2 = ovar2;
      branchsamedata->same = TRUE;
      branchsamedata->blocknr = blocknr;
      branchsamedata->pricecons = NULL;

      /* define name for origbranch constraints */
      (void) SCIPsnprintf(samename, SCIP_MAXSTRLEN, "same(%s,%s)", SCIPvarGetName(branchsamedata->var1),
         SCIPvarGetName(branchsamedata->var2));
   }

   if(!differinf)
   {
      /* create child-node of the current node in the b&b-tree */
      SCIP_CALL( SCIPcreateChild(masterscip, &child2, 0.0, SCIPgetLocalTransEstimate(masterscip)) );

      /* allocate branchdata and store information */
      SCIP_CALL( SCIPallocBlockMemory(scip, &branchdifferdata) );
      branchdifferdata->var1 = ovar1;
      branchdifferdata->var2 = ovar2;
      branchdifferdata->same = FALSE;
      branchdifferdata->blocknr = blocknr;
      branchdifferdata->pricecons = NULL;

      /* define name for origbranch constraints */
      (void) SCIPsnprintf(differname, SCIP_MAXSTRLEN, "differ(%s,%s)", SCIPvarGetName(branchsamedata->var1),
         SCIPvarGetName(branchsamedata->var2));
   }

   pricingvar1 = GCGoriginalVarGetPricingVar( !differinf? branchdifferdata->var1 : branchsamedata->var1 );
   pricingvar2 = GCGoriginalVarGetPricingVar( !differinf? branchdifferdata->var2 : branchsamedata->var2 );
   assert(GCGvarIsPricing(pricingvar1));
   assert(GCGvarIsPricing(pricingvar2));
   assert(GCGvarGetBlock(pricingvar1) == GCGvarGetBlock(pricingvar2));
   assert(GCGpricingVarGetNOrigvars(pricingvar1) == GCGpricingVarGetNOrigvars(pricingvar2));

   norigvars1 = GCGpricingVarGetNOrigvars(pricingvar1);
   assert(norigvars1 == GCGpricingVarGetNOrigvars(pricingvar2));

   origvars1 = GCGpricingVarGetOrigvars(pricingvar1);
   origvars2 = GCGpricingVarGetOrigvars(pricingvar2);

   if( norigvars1 > 0 )
   {
      maxorigvars1 = SCIPcalcMemGrowSize(masterscip, norigvars1);
      if(!sameinf)
         SCIP_CALL( SCIPallocBlockMemoryArray(masterscip, &origbranchconss1, maxorigvars1) );
      if(!differinf)
         SCIP_CALL( SCIPallocBlockMemoryArray(masterscip, &origbranchconss2, maxorigvars1) );
   }
   else
   {
      maxorigvars1 = 0;
   }

   /* add branching decision as varbound constraints to original problem */
   for( v = 0; v < norigvars1; v++ )
   {
      SCIP_CONS* origcons;
      SCIP_CONS* origcons2;

      assert(GCGvarGetBlock(origvars1[v]) == GCGvarGetBlock(origvars2[v]));
      assert(origbranchconss1 != NULL || sameinf );
      assert(origbranchconss2 != NULL || differinf );

      /* create constraint for same-child */
      if( !sameinf )
      {
         SCIP_CALL( SCIPcreateConsVarbound(scip, &origcons, samename, origvars1[v], origvars2[v],
            -1.0, 0.0, 0.0, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );

         origbranchconss1[v] = origcons;
      }

      /* create constraint for differ-child */
      if( !differinf )
      {
         SCIP_CALL( SCIPcreateConsVarbound(scip, &origcons2, differname, origvars1[v], origvars2[v],
               1.0, -SCIPinfinity(scip), 1.0, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );

         origbranchconss2[v] = origcons2;
      }
   }

   if( !sameinf )
   {
      /* create and add the masterbranch constraints */
      SCIP_CALL( GCGcreateConsMasterbranch(masterscip, &cons1, samename, child1,
         GCGconsMasterbranchGetActiveCons(masterscip), branchrule, branchsamedata, origbranchconss1, norigvars1,
         maxorigvars1) );

      SCIP_CALL( SCIPaddConsNode(masterscip, child1, cons1, NULL) );

      /* release constraints */
      SCIP_CALL( SCIPreleaseCons(masterscip, &cons1) );
   }

   if( !differinf )
   {
      /* create and add the masterbranch constraints */
      SCIP_CALL( GCGcreateConsMasterbranch(masterscip, &cons2, differname, child2,
         GCGconsMasterbranchGetActiveCons(masterscip), branchrule, branchdifferdata, origbranchconss2, norigvars1,
         maxorigvars1) );

      SCIP_CALL( SCIPaddConsNode(masterscip, child2, cons2, NULL) );

      /* release constraints */
      SCIP_CALL( SCIPreleaseCons(masterscip, &cons2) );
   } 
   return SCIP_OKAY;
}

/** branching execution method for fractional LP solutions */
static
SCIP_DECL_BRANCHEXECLP(branchExeclpRyanfoster)
{  /*lint --e{715}*/
   SCIP* origscip;
   SCIP_Bool feasible;

   SCIP_VAR** branchcands;
   int nbranchcands;

   int v1;

   SCIP_VAR* mvar1;
   SCIP_VAR* ovar1;
   SCIP_VAR* ovar2;

   SCIP_Bool usestrong;

   SCIP_VAR** ovar1s;
   SCIP_VAR** ovar2s;
   int *nspricingblock;
   int npairs;
   SCIP_Bool duplicate;
   SCIP_Bool stillusestrong;

   int pricingblock;
   SCIP_Bool sameinf;
   SCIP_Bool differinf;

   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIPdebugMessage("Execrel method of ryanfoster branching\n");

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   *result = SCIP_DIDNOTRUN;

   sameinf = FALSE;
   differinf = FALSE;

   /* do not perform Ryan & Foster branching if we have neither a set partitioning nor a set covering structure */
   if( !GCGisMasterSetCovering(origscip) && !GCGisMasterSetPartitioning(origscip) )
   {
      SCIPdebugMessage("Not executing Ryan&Foster branching, master is neither set covering nor set partitioning\n");
      return SCIP_OKAY;
   }

   if( GCGcurrentNodeIsGeneric(scip) )
   {
      SCIPdebugMessage("Not executing Ryan&Foster branching, node was branched by generic branchrule\n");
      return SCIP_OKAY;
   }

   if( GCGrelaxIsOrigSolFeasible(origscip) )
   {
      SCIPdebugMessage("node cut off, since origsol was feasible, solval = %f\n",
         SCIPgetSolOrigObj(origscip, GCGrelaxGetCurrentOrigSol(origscip)));

      *result = SCIP_DIDNOTFIND;

      return SCIP_OKAY;
   }

   /* the current original solution is not integral, now we have to branch;
    * first, get the master problem and all variables of the master problem
    */
   SCIP_CALL( SCIPgetLPBranchCands(scip, &branchcands, NULL, NULL, &nbranchcands, NULL, NULL) );

   SCIP_CALL( SCIPgetBoolParam(origscip, "branching/ryanfoster/usestrong", &usestrong) );

   if( usestrong )
   {
      npairs = 0;
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ovar1s, 0) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ovar2s, 0) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nspricingblock, 0) );
   }

   /* now search for two (fractional) columns mvar1, mvar2 in the master and 2 original variables ovar1, ovar2
    * s.t. mvar1 contains both ovar1 and ovar2 and mvar2 contains ovar1, but not ovar2
    */
   ovar1 = NULL;
   ovar2 = NULL;
   mvar1 = NULL;
   feasible = FALSE;

   /* select first fractional column (mvar1) */
   for( v1 = 0; v1 < nbranchcands && !feasible; v1++ )
   {
      int o1;
      int j;

      SCIP_VAR** origvars1;
      int norigvars1;
      SCIP_VAR* mvar2 = NULL;

      mvar1 = branchcands[v1];
      assert(GCGvarIsMaster(mvar1));

      origvars1 = GCGmasterVarGetOrigvars(mvar1);
      norigvars1 = GCGmasterVarGetNOrigvars(mvar1);

      /* select first original variable ovar1, that should be contained in both master variables */
      for( o1 = 0; o1 < norigvars1 && !feasible; o1++ )
      {
         int v2;

         ovar1 = origvars1[o1];
         /* if we deal with a trivial variable, skip it */
         if( SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar1)[o1]) || GCGoriginalVarGetNCoefs(ovar1) == 0 )
            continue;

         /* mvar1 contains ovar1, look for mvar2 which constains ovar1, too */
         for( v2 = v1+1; v2 < nbranchcands && !feasible; v2++ )
         {
            SCIP_VAR** origvars2;
            int norigvars2;
            int o2;
            SCIP_Bool contained = FALSE;

            mvar2 = branchcands[v2];
            assert(GCGvarIsMaster(mvar2));

            origvars2 = GCGmasterVarGetOrigvars(mvar2);
            norigvars2 = GCGmasterVarGetNOrigvars(mvar2);

            /* check whether ovar1 is contained in mvar2, too */
            for( j = 0; j < norigvars2; j++ )
            {
               /* if we deal with a trivial variable, skip it */
               if( SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar2)[j]) ||
                                        GCGoriginalVarGetNCoefs(origvars2[j]) == 0 )
                  continue;

               if( origvars2[j] == ovar1 )
               {
                  contained = TRUE;
                  break;
               }
            }

            /* mvar2 does not contain ovar1, so look for another mvar2 */
            if( !contained )
               continue;

            /* mvar2 also contains ovar1, now look for ovar2 contained in mvar1, but not in mvar2 */
            for( o2 = 0; o2 < norigvars1; o2++ )
            {
               /* if we deal with a trivial variable, skip it */
               if( !SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar1)[o2]) ||
                                         GCGoriginalVarGetNCoefs(origvars1[o2]) == 0 )
                  continue;

               ovar2 = origvars1[o2];
               if( ovar2 == ovar1 )
                  continue;

               /* check whether ovar2 is contained in mvar2, too */
               contained = FALSE;
               for( j = 0; j < norigvars2; j++ )
               {
                  /* if we deal with a trivial variable, skip it */
                  if( !SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar2)[j]) )
                     continue;

                  if( origvars2[j] == ovar2 )
                  {
                     contained = TRUE;
                     break;
                  }
               }

               /* ovar2 should be contained in mvar1 but not in mvar2, so look for another ovar2,
                * if the current one is contained in mvar2
                */
               if( contained )
                  continue;

               /* if we arrive here, ovar2 is contained in mvar1 but not in mvar2, so everything is fine */
               if( !usestrong )
               {
                  feasible = TRUE;
                  break;
               }
               else
               {
                  /* we need to check first whether we already found this pair */
                  duplicate = FALSE;
                  for( int z=0; z<npairs; z++ )
                  {
                     if( ovar1s[z] == ovar1 && ovar2s[z] == ovar2 )
                     {
                        duplicate = TRUE;
                        break;
                     }
                  }

                  if( !duplicate )
                  {
                     SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &ovar1s, npairs, npairs+1) );
                     SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &ovar2s, npairs, npairs+1) );
                     SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nspricingblock, npairs, npairs+1) );
                     
                     ovar1s[npairs] = ovar1;
                     ovar2s[npairs] = ovar2;
                     nspricingblock[npairs] = GCGvarGetBlock(mvar1);

                     npairs++;
                  }
               }
            }

            /* we did not find an ovar2 contained in mvar1, but not in mvar2,
             * now look for one contained in mvar2, but not in mvar1
             */
            if( !feasible )
            {
               for( o2 = 0; o2 < norigvars2; o2++ )
               {
                  /* if we deal with a trivial variable, skip it */
                  if( SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar2)[o2]) ||
                                           GCGoriginalVarGetNCoefs(origvars2[o2]) == 0 )
                     continue;

                  ovar2 = origvars2[o2];

                  if( ovar2 == ovar1 )
                     continue;

                  contained = FALSE;
                  for( j = 0; j < norigvars1; j++ )
                  {
                     /* if we deal with a trivial variable, skip it */
                     if( SCIPisZero(origscip, GCGmasterVarGetOrigvals(mvar1)[j]) ||
                                              GCGoriginalVarGetNCoefs(origvars1[j]) == 0 )
                        continue;
                     if( origvars1[j] == ovar2 )
                     {
                        contained = TRUE;
                        break;
                     }
                  }

                  /* ovar2 should be contained in mvar2 but not in mvar1, so look for another ovar2,
                   * if the current one is contained in mvar1
                   */
                  if( contained )
                     continue;

                  /* if we arrive here, ovar2 is contained in mvar2 but not in mvar1, so everything is fine */
                  if( !usestrong )
                  {
                     feasible = TRUE;
                     break;
                  }
                  else
                  {
                     /* we need to check first whether we already found this pair */
                     duplicate = FALSE;
                     for( int z=0; z<npairs; z++ )
                     {
                        if( ovar1s[z] == ovar1 && ovar2s[z] == ovar2 )
                        {
                           duplicate = TRUE;
                           break;
                        }
                     }

                     if( !duplicate )
                     {
                        SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &ovar1s, npairs, npairs+1) );
                        SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &ovar2s, npairs, npairs+1) );
                        SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &nspricingblock, npairs, npairs+1) );
                        
                        ovar1s[npairs] = ovar1;
                        ovar2s[npairs] = ovar2;
                        nspricingblock[npairs] = GCGvarGetBlock(mvar1);

                        npairs++;
                     }
                  }
               }
            }
         }
      }
   }

   if( usestrong )
   {
      if( npairs>0 )
      {
         GCGbranchSelectCandidateStrongBranchingRyanfoster(origscip, branchrule, ovar1s, ovar2s, nspricingblock,
                                                           npairs, &ovar1, &ovar2, &pricingblock, &sameinf, &differinf,
                                                           result, &stillusestrong);

         SCIPfreeBlockMemoryArray(scip, &ovar1s, npairs);
         SCIPfreeBlockMemoryArray(scip, &ovar2s, npairs);
         SCIPfreeBlockMemoryArray(scip, &nspricingblock, npairs);
         if( !stillusestrong )
         {
            SCIPsetBoolParam(origscip, "branching/ryanfoster/usestrong", FALSE);
         }
      }
   }

   if( !feasible && ( !usestrong || npairs==0 ) )
   {
      SCIPdebugMessage("Ryanfoster branching rule could not find variables to branch on!\n");
      return SCIP_OKAY;
   }

   /* create the two child nodes in the branch-and-bound tree */
   SCIP_CALL( createChildNodesRyanfoster(origscip, branchrule, ovar1, ovar2, GCGvarGetBlock(mvar1), sameinf,
                                         differinf) );

   *result = SCIP_BRANCHED;

   return SCIP_OKAY;
}

/** branching execution method for relaxation solutions */
static
SCIP_DECL_BRANCHEXECEXT(branchExecextRyanfoster)
{  /*lint --e{715}*/

   *result = SCIP_DIDNOTRUN;

   return SCIP_OKAY;

}

/** branching execution method for not completely fixed pseudo solutions */
static
SCIP_DECL_BRANCHEXECPS(branchExecpsRyanfoster)
{  /*lint --e{715}*/
   SCIP_CONS** origbranchconss;
   GCG_BRANCHDATA* branchdata;
   SCIP_VAR** branchcands;
   SCIP_VAR* ovar1;
   SCIP_VAR* ovar2;
   SCIP_Bool feasible;
   int norigbranchconss;
   int nbranchcands;
   int o1;
   int o2;
   int c;

   SCIP* origscip;

   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIPdebugMessage("Execps method of ryanfoster branching\n");

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);


   *result = SCIP_DIDNOTRUN;

   /* do not perform Ryan & Foster branching if we have neither a set partitioning nor a set covering structure */
   if( !GCGisMasterSetCovering(origscip) || !GCGisMasterSetPartitioning(origscip) )
   {
      SCIPdebugMessage("Not executing Ryanfoster branching, master is neither set covering nor set partitioning\n");
      return SCIP_OKAY;
   }

   if( GCGcurrentNodeIsGeneric(scip) )
   {
      SCIPdebugMessage("Not executing Ryanfoster branching, node was branched by generic branchrule\n");
      return SCIP_OKAY;
   }

   /* get unfixed variables and stack of active origbranchconss */
   SCIP_CALL( SCIPgetPseudoBranchCands(origscip, &branchcands, NULL, &nbranchcands) );
   GCGconsOrigbranchGetStack(origscip, &origbranchconss, &norigbranchconss);

   ovar1 = NULL;
   ovar2 = NULL;
   feasible = FALSE;

   /* select first original variable ovar1 */
   for( o1 = 0; o1 < nbranchcands && !feasible; ++o1 )
   {
      ovar1 = branchcands[o1];

      /* select second original variable o2 */
      for( o2 = o1 + 1; o2 < nbranchcands; ++o2 )
      {
         ovar2 = branchcands[o2];

         assert(ovar2 != ovar1);

         /* check whether we already branched on this combination of variables */
         for( c = 0; c < norigbranchconss; ++c )
         {
            if( GCGconsOrigbranchGetBranchrule(origbranchconss[c]) == branchrule )
               continue;

            branchdata = GCGconsOrigbranchGetBranchdata(origbranchconss[c]);

            if( (branchdata->var1 == ovar1 && branchdata->var2 == ovar2 )
               || (branchdata->var1 == ovar2 && branchdata->var2 == ovar1) )
            {
               break;
            }
         }

         /* we did not break, so there is no active origbranch constraint with both variables */
         if( c == norigbranchconss )
         {
            feasible = TRUE;
            break;
         }
      }
   }

   if( !feasible )
   {
      SCIPdebugMessage("Ryanfoster branching rule could not find variables to branch on!\n");
      return SCIP_OKAY;
   }

   /* create the two child nodes in the branch-and-bound tree */
   SCIP_CALL( createChildNodesRyanfoster(origscip, branchrule, ovar1, ovar2, GCGvarGetBlock(ovar1), FALSE, FALSE) );

   *result = SCIP_BRANCHED;

   return SCIP_OKAY;
}

/** initialization method of branching rule (called after problem was transformed) */
static
SCIP_DECL_BRANCHINIT(branchInitRyanfoster)
{
   SCIP* origprob;

   origprob = GCGmasterGetOrigprob(scip);
   assert(branchrule != NULL);
   assert(origprob != NULL);

   SCIP_CALL( GCGrelaxIncludeBranchrule(origprob, branchrule, branchActiveMasterRyanfoster,
         branchDeactiveMasterRyanfoster, branchPropMasterRyanfoster, NULL, branchDataDeleteRyanfoster) );

   return SCIP_OKAY;
}



/* define not used callback as NULL*/
#define branchFreeRyanfoster NULL
#define branchExitRyanfoster NULL
#define branchInitsolRyanfoster NULL
#define branchExitsolRyanfoster NULL


/*
 * branching specific interface methods
 */

/** creates the Ryan-Foster LP braching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleRyanfoster(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;

   /* create branching rule data */
   branchruledata = NULL;

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchrule(scip, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY,
         BRANCHRULE_MAXDEPTH, BRANCHRULE_MAXBOUNDDIST, branchCopyRyanfoster,
         branchFreeRyanfoster, branchInitRyanfoster, branchExitRyanfoster, branchInitsolRyanfoster,
         branchExitsolRyanfoster, branchExeclpRyanfoster, branchExecextRyanfoster, branchExecpsRyanfoster,
         branchruledata) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/usestrong",
         "should strong branching be used to determine the variables on which the branching is performed?",
         NULL, FALSE, DEFAULT_USESTRONG, NULL, NULL) );

   /* strong branching */
   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/minphase0outcands",
         "minimum number of output candidates from phase 0 during strong branching",
         NULL, FALSE, DEFAULT_MINPHASE0OUTCANDS, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/maxphase0outcands",
         "maximum number of output candidates from phase 0 during strong branching",
         NULL, FALSE, DEFAULT_MAXPHASE0OUTCANDS, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/maxphase0outcandsfrac",
         "maximum number of output candidates from phase 0 as fraction of total cands during strong branching",
         NULL, FALSE, DEFAULT_MAXPHASE0OUTCANDSFRAC, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/phase1gapweight",
         "how much impact should the node gap have on the number of precisely evaluated candidates in phase 1 during strong branching?",
         NULL, FALSE, DEFAULT_PHASE1GAPWEIGHT, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/minphase1outcands",
         "minimum number of output candidates from phase 1 during strong branching",
         NULL, FALSE, DEFAULT_MINPHASE1OUTCANDS, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/maxphase1outcands",
         "maximum number of output candidates from phase 1 during strong branching",
         NULL, FALSE, DEFAULT_MAXPHASE1OUTCANDS, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/maxphase1outcandsfrac",
         "maximum number of output candidates from phase 1 as fraction of phase 1 cands during strong branching",
         NULL, FALSE, DEFAULT_MAXPHASE1OUTCANDSFRAC, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "branching/ryanfoster/phase2gapweight",
         "how much impact should the node gap have on the number of precisely evaluated candidates in phase 2 during strong branching?",
         NULL, FALSE, DEFAULT_PHASE2GAPWEIGHT, 0, 1, NULL, NULL) );

   return SCIP_OKAY;
}
