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

/**@file   cons_masterbranch.c
 * @ingroup CONSHDLRS
 * @brief  constraint handler for storing the branching decisions at each node of the tree
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Christian Puchert
 * @author Marcel Schmickerath
 * @author Erik Muehmer
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/*#define SCIP_DEBUG*/
#include <assert.h>
#include <string.h>

#include "gcg.h"
#include "branch_generic.h"
#include "cons_masterbranch.h"
#include "cons_origbranch.h"
#include "relax_gcg.h"
#include "pricer_gcg.h"

#include "scip/cons_linear.h"

/*#define CHECKPROPAGATEDVARS*/

/* constraint handler properties */
#define CONSHDLR_NAME          "masterbranch"
#define CONSHDLR_DESC          "store branching decision at nodes of the tree constraint handler"
#define CONSHDLR_ENFOPRIORITY         0 /**< priority of the constraint handler for constraint enforcing */
#define CONSHDLR_CHECKPRIORITY  2000000 /**< priority of the constraint handler for checking feasibility */
#define CONSHDLR_PROPFREQ             1 /**< frequency for propagating domains; zero means only preprocessing propagation */
#define CONSHDLR_EAGERFREQ          100 /**< frequency for using all instead of only the useful constraints in separation,
                                         * propagation and enforcement, -1 for no eager evaluations, 0 for first only */
#define CONSHDLR_DELAYPROP        FALSE /**< should propagation method be delayed, if other propagators found reductions? */
#define CONSHDLR_NEEDSCONS         TRUE /**< should the constraint handler be skipped, if no constraints are available? */
#define CONSHDLR_PROPTIMING SCIP_PROPTIMING_ALWAYS

#define EVENTHDLR_NAME         "origvarbound"
#define EVENTHDLR_DESC         "event handler for bound changes on original variables"


/** constraint data for masterbranch constraints */
struct SCIP_ConsData
{
   char*                 name;               /**< name of the constraint */
   int                   npropvars;          /**< number of variables that existed the last time the related node was propagated,
                                                  used to determine whether the constraint should be repropagated */
   SCIP_Bool             needprop;           /**< should the constraint be propagated? */
   SCIP_NODE*            node;               /**< the node at which the constraint is sticking */
   int                   nactivated;         /**< number of times the constraint has been activated so far */

   SCIP_CONS*            parentcons;         /**< the masterbranch constraint of the parent node */
   SCIP_CONS**           childconss;         /**< array of the masterbranch constraints of child nodes */
   int                   nchildconss;        /**< number of the masterbranch constraints of child nodes */
   int                   maxchildconss;     /**< capacity of childconss array */
   SCIP_CONS*            probingtmpcons;     /**< pointer to save the last child in the childconss array if it is overwritten in probing mode */
   SCIP_CONS*            origcons;           /**< the corresponding origbranch cons in the original program */

   GCG_BRANCHDATA*       branchdata;         /**< branching data stored by the branching rule at the corresponding origcons constraint
                                              *   containing information about the branching restrictions */
   SCIP_BRANCHRULE*      branchrule;         /**< branching rule that created the corresponding node in the original problem and imposed
                                              *   branching restrictions */

   /* local bound changes on original variables that belong to a unique block */
   SCIP_VAR**            localbndvars;       /**< original variables of bound changes stored at the current node */
   SCIP_BOUNDTYPE*       localbndtypes;      /**< types of the bound changes stored at the current node */
   SCIP_Real*            localnewbnds;       /**< new bounds for the bound changes stored at the current node */
   SCIP_Real*            localoldbnds;       /**< old bounds for the bound changes stored at the current node */

   int*                  nlocalbndchgstreated; /**< number of bound changes of the nodes on the way from the current node to
                                                *   the root node that have been treated so far */
   int                   maxlocalbndchgstreated; /**< capacity of nlocalbndchgstreated */
   int                   nlocalbndchgs;      /**< number of bound changes */
   int                   maxlocalbndchgs;    /**< capacity of corresponding arrays */
   int                   nbranchingchgs;     /**< number of bound changes due to branching (<= nlocalbndchgs) */

   /* local bound changes on original variables that have been directly copied to the master problem */
   SCIP_VAR**            copiedvars;         /**< original variables on which local bounds were changed */
   GCG_BOUNDTYPE*        copiedvarbndtypes;  /**< types of the new local bounds of the coped original variables */
   SCIP_Real*            copiedvarbnds;      /**< new lower/upper bounds of the coped original variables */
   int                   ncopiedvarbnds;     /**< number of new local bounds stored */
   int                   maxcopiedvarbnds;   /**< size of copiedvars, copiedvarbndtypes, and copiedvarbnds arrays */

   /* constraints that enforce the branching restrictions on the original problem */
   SCIP_CONS**           origbranchconss;    /**< constraints in the original problem that enforce the branching decision */
   int                   norigbranchconss;   /**< number of constraints in the original problem that enforce the branching decision */
   int                   maxorigbranchconss;/**< capacity of origbranchconss array */
};

/** constraint handler data */
struct SCIP_ConshdlrData
{
   /* active masterbranch constraints on the path from the root node to the current node */
   SCIP_CONS**           stack;              /**< stack for storing active constraints */
   int                   nstack;             /**< number of elements on the stack */
   int                   maxstacksize;       /**< maximum size of the stack */

   /* global bound changes on the original problem */
   SCIP_VAR**            pendingvars;        /**< pricing variables or master variable copies corresponding to pending bound changes (global bound changes) */
   SCIP_BOUNDTYPE*       pendingbndtypes;    /**< types of the pending bound changes (global bound changes) */
   SCIP_Real*            pendingnewbnds;     /**< new bounds corresponding to pending bound changes (global bound changes) */
   SCIP_Real*            pendingoldbnds;     /**< old bounds corresponding to pending bound changes (global bound changes) */
   int                   npendingbnds;       /**< number of pending bound changes (global bound changes) */
   SCIP_Bool             pendingbndsactivated; /**< were pending bound changes already activated? */
   int                   maxpendingbnds;     /**< size of the array corresponding to pending bound changes */
   SCIP_Bool             enforceproper;      /**< should proper variables be enforced? */

   /* information needed by applyLocalBndchgsToPricedMastervars */
   SCIP_VAR***           collectedbndvars;   /**< collected original variables of bound changes stored for each block */
   SCIP_Real**           collectedlbnds;     /**< collected upper bound changes for each block */
   SCIP_Real**           collectedubnds;     /**< collected upper bound changes for each block */
   int                   maxblocknum;        /**< maximal number of blocks that can be handled by the collected* arrays */
   int*                  ncollectedbndvars;  /**< number of collected bndvars per block */
   int*                  maxcollectedbndvars;/**< capacity of the collected* arrays per block */
   int**                 linkingvaridxs;     /**< position of linking vars in collectedbndvars */
   int                   maxlinkingvaridxs;  /**< capacity of linkingvaridxs */
};

/*
 * Local methods
 */

/** initialize the consdata data structure */
static
SCIP_RETCODE initializeConsdata(
   SCIP*                 scip,               /**< SCIP data structure*/
   SCIP_CONS*            cons                /**< constraint for which the consdata is created */
   )
{
#ifdef SCIP_DEBUG
   SCIP_CONS* origcons_parent;
   SCIP_CONS* parent_origcons;
#endif

   SCIP* origscip;
   SCIP_CONSDATA* consdata;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS* origcons;

   SCIP_DOMCHG* domchg;
   SCIP_BOUNDCHG* boundchg;
   SCIP_VAR* boundchgvar;

   int i;

   assert(scip != NULL);
   assert(GCGisMaster(scip));
   assert(cons != NULL);

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);
   assert(SCIPconsGetHdlr(cons) == conshdlr);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* get corresponding origbranch constraint in the original problem */
   origcons = GCGconsOrigbranchGetActiveCons(origscip);
   assert(origcons != NULL);

   /* @fixme: There should be an assertion instead; I guess consdata->origcons should be NULL */
   if( consdata->origcons != origcons ) /*rootnode?*/
   {
      SCIPdebugMessage("set root origcons\n");
      consdata->origcons = origcons;
      GCGconsOrigbranchSetMastercons(origcons, cons);
   }

   /* @fixme: Why should anything else happen? */
   if( GCGconsOrigbranchGetNChildconss(origcons) == 0 )
   {
      consdata->nchildconss = 0;
      consdata->childconss = NULL;
   }

   /*GCGconsOrigbranchSetMastercons(origcons, cons);*/


#ifdef SCIP_DEBUG
   if( consdata->parentcons != NULL )
      parent_origcons = SCIPconsGetData(consdata->parentcons)->origcons;
   else
      parent_origcons = NULL;

   if( consdata->origcons != NULL )
      origcons_parent = GCGconsOrigbranchGetParentcons(consdata->origcons);
   else
      origcons_parent = NULL;

   SCIPdebugMessage("cons: %s, origcons: %s, parent: %s => %s\n", SCIPconsGetName(cons), consdata->origcons == NULL? "NULL" : SCIPconsGetName( consdata->origcons ),
      parent_origcons == NULL? "NULL" : SCIPconsGetName(parent_origcons), origcons_parent == NULL? "NULL" : SCIPconsGetName(origcons_parent) );
#endif

   assert(SCIPgetCurrentNode(scip) == consdata->node || consdata->node == SCIPgetRootNode(scip));
/*    assert((SCIPgetNNodesLeft(scip)+SCIPgetNNodes(scip) == 1) == (consdata->node == SCIPgetRootNode(scip))); */
   assert(SCIPnodeGetDepth(GCGconsOrigbranchGetNode(consdata->origcons)) == SCIPnodeGetDepth(consdata->node));
   assert(consdata->parentcons != NULL || SCIPnodeGetDepth(consdata->node) == 0);

   assert(consdata->parentcons == NULL ||
      SCIPconsGetData(consdata->parentcons)->origcons == GCGconsOrigbranchGetParentcons(consdata->origcons));

   consdata->maxlocalbndchgstreated = SCIPcalcMemGrowSize(scip, conshdlrdata->nstack+1);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->nlocalbndchgstreated, (size_t)consdata->maxlocalbndchgstreated) );

   /* get all bound changes at the corresponding node in the original problem */

   domchg = SCIPnodeGetDomchg(GCGconsOrigbranchGetNode(origcons));
   consdata->nlocalbndchgs = SCIPdomchgGetNBoundchgs(domchg);
   consdata->nlocalbndchgstreated[conshdlrdata->nstack] = consdata->nlocalbndchgs;

   if( consdata->nlocalbndchgs > 0 )
   {
      consdata->maxlocalbndchgs = SCIPcalcMemGrowSize(scip, consdata->nlocalbndchgs);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->localbndvars, consdata->maxlocalbndchgs) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->localbndtypes, consdata->maxlocalbndchgs) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->localnewbnds, consdata->maxlocalbndchgs) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consdata->localoldbnds, consdata->maxlocalbndchgs) );
   }
   else
   {
      consdata->maxlocalbndchgs = 0;
   }

   consdata->nbranchingchgs = 0;

   for( i = 0; i < consdata->nlocalbndchgs; ++i )
   {
      boundchg = SCIPdomchgGetBoundchg(domchg, i);

      consdata->localbndvars[i] = SCIPboundchgGetVar(boundchg);
      consdata->localnewbnds[i] = SCIPboundchgGetNewbound(boundchg);
      consdata->localbndtypes[i] = SCIPboundchgGetBoundtype(boundchg);

      if( SCIPboundchgGetBoundchgtype(boundchg) == SCIP_BOUNDCHGTYPE_BRANCHING )
      {
         consdata->nbranchingchgs++;
         assert(consdata->nbranchingchgs == i+1);
      }
   }

   consdata->needprop = TRUE;

   assert((consdata->parentcons == NULL) == (conshdlrdata->nstack == 0));
   if( consdata->parentcons != NULL )
   {
      SCIP_CONSDATA* parentdata = SCIPconsGetData(consdata->parentcons);

      assert(consdata->parentcons == conshdlrdata->stack[conshdlrdata->nstack-1]);
      assert(SCIPconsGetData(conshdlrdata->stack[0])->parentcons == NULL);

      /* check whether bound changes were added in nodes on the path
       * to the current node after activation of the parent node
       */
      for( i = 1; i < conshdlrdata->nstack; ++i )
      {
         int ndomboundchgs;
         SCIP_CONSDATA* stackconsdata = SCIPconsGetData(conshdlrdata->stack[i]);
         domchg = SCIPnodeGetDomchg(GCGconsOrigbranchGetNode(stackconsdata->origcons));
         ndomboundchgs = SCIPdomchgGetNBoundchgs(domchg);

         assert(ndomboundchgs >= parentdata->nlocalbndchgstreated[i]);

         if( ndomboundchgs != parentdata->nlocalbndchgstreated[i] )
         {
            int diff;
            int j;

            diff = ndomboundchgs - parentdata->nlocalbndchgstreated[i];

            if( consdata->maxlocalbndchgs < consdata->nlocalbndchgs + diff )
            {
               int newmaxsize = SCIPcalcMemGrowSize(scip, consdata->nlocalbndchgs + diff);
               SCIP_CALL(SCIPreallocBlockMemoryArray(scip, &consdata->localbndvars, consdata->maxlocalbndchgs,
                                                     (size_t) newmaxsize));
               SCIP_CALL(SCIPreallocBlockMemoryArray(scip, &consdata->localbndtypes, consdata->maxlocalbndchgs,
                                                     (size_t) newmaxsize));
               SCIP_CALL(SCIPreallocBlockMemoryArray(scip, &consdata->localnewbnds, consdata->maxlocalbndchgs,
                                                     (size_t) newmaxsize));
               SCIP_CALL(SCIPreallocBlockMemoryArray(scip, &consdata->localoldbnds, consdata->maxlocalbndchgs,
                                                     (size_t) newmaxsize));
               consdata->maxlocalbndchgs = newmaxsize;
            }

            /* add bound changes to the boundchanges array */
            for( j = 0; j < ndomboundchgs; ++j )
            {
               int bndchgindex;
               SCIP_BOUNDTYPE boundchgtype;
               SCIP_Real boundchgnewbound;

               boundchg = SCIPdomchgGetBoundchg(domchg, j);
               boundchgvar = SCIPboundchgGetVar(boundchg);
               boundchgtype = SCIPboundchgGetBoundtype(boundchg);
               boundchgnewbound = SCIPboundchgGetNewbound(boundchg);

               if( j < stackconsdata->nlocalbndchgstreated[i] )
               {
                  assert(stackconsdata->localbndvars[j] == boundchgvar
                     && SCIPisEQ(scip, stackconsdata->localnewbnds[j], boundchgnewbound)
                     && stackconsdata->localbndtypes[j] == boundchgtype);
                  continue;
               }
               if( j < parentdata->nlocalbndchgstreated[i] )
                  continue;

               bndchgindex = consdata->nlocalbndchgs + j - parentdata->nlocalbndchgstreated[i];

               consdata->localbndvars[bndchgindex] = boundchgvar;
               consdata->localnewbnds[bndchgindex] = boundchgnewbound;
               consdata->localbndtypes[bndchgindex] = boundchgtype;
            }

            consdata->nlocalbndchgs += diff;
         }

         consdata->nlocalbndchgstreated[i] = ndomboundchgs;
      }
   }

   return SCIP_OKAY;
}

/** add a global bound change on the original problem to the pending bound changes array */
static
SCIP_RETCODE addPendingBndChg(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable on which the bound change is applied (corresponding master variable copy or pricing variable) */
   SCIP_BOUNDTYPE        boundtype,          /**< type of the bound (lower or upper) */
   SCIP_Real             oldbound,           /**< previous bound value */
   SCIP_Real             newbound            /**< new bound value */
   )
{
   SCIP_CONSHDLR*     conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   if( conshdlr == NULL )
   {
      SCIPerrorMessage("masterbranch constraint handler not found\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert((conshdlrdata->npendingbnds > 0) || conshdlrdata->pendingbndsactivated);

   /* reallocate memory if needed */
   if( conshdlrdata->npendingbnds >= conshdlrdata->maxpendingbnds )
   {
      int newsize = SCIPcalcMemGrowSize(scip, conshdlrdata->npendingbnds+5);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->pendingvars), conshdlrdata->maxpendingbnds, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->pendingbndtypes), conshdlrdata->maxpendingbnds, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->pendingoldbnds), conshdlrdata->maxpendingbnds, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->pendingnewbnds), conshdlrdata->maxpendingbnds, newsize) );
      conshdlrdata->maxpendingbnds = newsize;
   }

   /* store pending bound change */
   conshdlrdata->pendingvars[conshdlrdata->npendingbnds] = var;
   conshdlrdata->pendingbndtypes[conshdlrdata->npendingbnds] = boundtype;
   conshdlrdata->pendingoldbnds[conshdlrdata->npendingbnds] = oldbound;
   conshdlrdata->pendingnewbnds[conshdlrdata->npendingbnds] = newbound;
   conshdlrdata->npendingbnds++;
   conshdlrdata->pendingbndsactivated = FALSE;

   return SCIP_OKAY;
}

/** For a given global bound change on a pricing variable, check if the global bounds on all corresponding original variables are still the same
 *
 *  @return TRUE if the variable is in a relevant block AND all variables identical to it have the same bounds
 */
static
SCIP_Bool checkAggregatedGlobalBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             bndvar,             /**< pricing variable whose new global bound is to be checked */
   SCIP_BOUNDTYPE        bndtype,            /**< type of the new global bound */
   SCIP_Real             bound               /**< new global bound */
   )
{
   SCIP_VAR** identvars;
   int nidentvars;
   SCIP_Bool identical;

   assert(GCGvarIsPricing(bndvar));

   /* get all identical variables */
   identvars = GCGpricingVarGetOrigvars(bndvar);
   nidentvars = GCGpricingVarGetNOrigvars(bndvar);

   identical = TRUE;

   /* If the variable was not aggregated, there is nothing to check */
   if( nidentvars > 1 )
   {
      int i;

      /* Check if the bounds of all identical variables are equal to the one of the representative */
      for( i = 0; i < nidentvars; ++i )
      {
         SCIP_Real identbound = bndtype == SCIP_BOUNDTYPE_UPPER ? SCIPvarGetUbGlobal(identvars[i]) : SCIPvarGetLbGlobal(identvars[i]);
         if( !SCIPisEQ(scip, identbound, bound) )
         {
            SCIPerrorMessage("Var <%s> has new local %s bound %g, but identical var <%s> has %g -- don't know how to handle!\n",
               SCIPvarGetName(bndvar), bndtype == SCIP_BOUNDTYPE_UPPER ? "upper" : "lower",
                  bound, SCIPvarGetName(identvars[i]), identbound);
            identical = FALSE;
         }
      }
   }

   return identical;
}

/** apply global bound changes on original problem variables either
 *  to their copies in the master problem and/or to the corresponding pricing problem variables
 */
static
SCIP_RETCODE applyGlobalBndchgsToPricingprobs(
   SCIP*                 scip                /* SCIP data structure */
   )
{
   SCIP* origscip;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   if( !conshdlrdata->pendingbndsactivated )
   {
      assert(conshdlrdata->npendingbnds > 0);
      for( i = 0; i < conshdlrdata->npendingbnds; i++ )
      {
         /* This should not have an effect on linking variables */
         assert(GCGvarIsMaster(conshdlrdata->pendingvars[i]) || GCGvarIsPricing(conshdlrdata->pendingvars[i]));

         if( GCGvarIsMaster(conshdlrdata->pendingvars[i]) )
         {
            if( conshdlrdata->pendingbndtypes[i] == SCIP_BOUNDTYPE_LOWER )
            {
               if( SCIPisLT(scip, SCIPvarGetLbGlobal(conshdlrdata->pendingvars[i]), conshdlrdata->pendingnewbnds[i]) )
               {
                  SCIPdebugMessage("Global lower bound of mastervar <%s> set to %g\n", SCIPvarGetName(conshdlrdata->pendingvars[i]),
                     conshdlrdata->pendingnewbnds[i]);
                  SCIP_CALL( SCIPchgVarLbGlobal(scip, conshdlrdata->pendingvars[i], conshdlrdata->pendingnewbnds[i]) );
               }
            }
            else
            {
               if( SCIPisGT(scip, SCIPvarGetUbGlobal(conshdlrdata->pendingvars[i]), conshdlrdata->pendingnewbnds[i]) )
               {
                  SCIPdebugMessage("Global upper bound of mastervar <%s> set to %g\n", SCIPvarGetName(conshdlrdata->pendingvars[i]),
                     conshdlrdata->pendingnewbnds[i]);
                  SCIP_CALL( SCIPchgVarUbGlobal(scip, conshdlrdata->pendingvars[i], conshdlrdata->pendingnewbnds[i]) );
               }
            }
         }
         else
         {
            /* this is a global boundchange on a variable that belongs to a block,
             * we have to adjust the bound of the corresponding variable in the pricing problem
             */

            /* check if all identical variables have the same global bound */
            if( !checkAggregatedGlobalBounds(origscip, conshdlrdata->pendingvars[i], conshdlrdata->pendingbndtypes[i], conshdlrdata->pendingnewbnds[i]) )
               continue;

            if( conshdlrdata->pendingbndtypes[i] == SCIP_BOUNDTYPE_LOWER )
            {
               SCIP_CALL( SCIPchgVarLbGlobal(GCGgetPricingprob(origscip, GCGvarGetBlock(conshdlrdata->pendingvars[i]) ),
                     conshdlrdata->pendingvars[i], conshdlrdata->pendingnewbnds[i]) );
            }
            else
            {
               SCIP_CALL( SCIPchgVarUbGlobal(GCGgetPricingprob(origscip, GCGvarGetBlock(conshdlrdata->pendingvars[i]) ),
                     conshdlrdata->pendingvars[i], conshdlrdata->pendingnewbnds[i]) );
            }
         }
      }
      conshdlrdata->pendingbndsactivated = TRUE;
   }

   return SCIP_OKAY;
}

/** apply global bound changes on original problem variables to the master problem */
static
SCIP_RETCODE applyGlobalBndchgsToPricedMastervars(
   SCIP*                 scip,               /**< SCIP data structure */
   int*                  propcount           /**< number of applied bound changes */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_VAR** vars;
   int nvars;
   int i;
   int k;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* get master problem variables generated during pricing */
   vars = GCGmasterGetPricedvars(scip);
   nvars = GCGmasterGetNPricedvars(scip);

   assert((conshdlrdata->npendingbnds > 0) || conshdlrdata->pendingbndsactivated);

   /* iterate over all master variables and apply global bound changes */
   if( conshdlrdata->npendingbnds > 0 && conshdlrdata->pendingbndsactivated )
   {
      for( i = 0; i < nvars; i++ )
      {
         SCIP_Bool ismastervarrelevant;
         SCIP_VAR** origvars;
         SCIP_HASHMAP* origvals;
         int norigvars;
         int blocknr;
         blocknr = GCGvarGetBlock(vars[i]);

         /* get the original variables that are contained in the master variable */
         assert(GCGvarIsMaster(vars[i]));
         norigvars = GCGmasterVarGetNOrigvars(vars[i]);
         origvars = GCGmasterVarGetOrigvars(vars[i]);
         origvals = GCGmasterVarGetOrigvalmap(vars[i]);

         assert(blocknr < GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)));
         assert(norigvars >= 0);
         assert(origvars != NULL || norigvars == 0);

         /* only look at master variables not globally fixed to zero that belong to a block */
         ismastervarrelevant = !SCIPisFeasZero(scip, SCIPvarGetUbGlobal(vars[i]));
         ismastervarrelevant = ismastervarrelevant && (norigvars > 0);
         ismastervarrelevant = ismastervarrelevant && (blocknr >= 0 || GCGmasterVarIsLinking(vars[i])); /*lint !e613*/
         if( !ismastervarrelevant )
            continue;

         /* iterate over global bound changes on the original variables
          * that have not yet been checked for the master variables
          */
         for( k = 0; k < conshdlrdata->npendingbnds; k++ )
         {
            int bndchgblocknr;
            SCIP_VAR** bndchgorigvars;
            SCIP_Real val;

            assert(!GCGvarIsOriginal(conshdlrdata->pendingvars[k]));

            bndchgblocknr = GCGvarGetBlock(conshdlrdata->pendingvars[k]);
            if( GCGvarIsMaster(conshdlrdata->pendingvars[k]) )
            {
               assert(bndchgblocknr == -1);
               bndchgorigvars = GCGmasterVarGetOrigvars(conshdlrdata->pendingvars[k]);
            }
            else if( GCGvarIsPricing(conshdlrdata->pendingvars[k]) )
               bndchgorigvars = GCGpricingVarGetOrigvars(conshdlrdata->pendingvars[k]);
            else
            {
               SCIPerrorMessage("Variable %s is not pricing nor master.\n", SCIPvarGetName(conshdlrdata->pendingvars[k]));
               assert(GCGvarIsMaster(conshdlrdata->pendingvars[k]) || GCGvarIsPricing(conshdlrdata->pendingvars[k]));
               bndchgorigvars = NULL;
            }
            assert(bndchgblocknr < GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)));
            assert(bndchgorigvars != NULL);
            assert(origvars != NULL);

            /* The bound change is only relevant for the master variable if either
             *  - the bound change was performed in the same block as the master variable, or
             *  - the master variable is a copied linking variable and the bound change was performed
             *    in one of the blocks that the variable is linking
             */
            if( (bndchgblocknr != blocknr )
               && !(GCGmasterVarIsLinking(vars[i]) && GCGisLinkingVarInBlock(origvars[0], bndchgblocknr)) )
               continue;

            assert(bndchgorigvars[0] != NULL);

            val = SCIPhashmapGetImageReal(origvals, bndchgorigvars[0]);
            /* variables belong to the same block -> set origval to 0.0 if not in map */
            if( val == SCIP_INVALID )
               val = 0.0;

            /* if the variable contains a part of the branching variable that violates the bound,
             * fix the master variable to 0
             */
            /* @todo: This is the wrong way to treat bound changes on original variable copies in the master problem;
             * I think they have already been treated during constraint activation
             */

            assert(GCGvarGetBlock(bndchgorigvars[0]) == blocknr || (GCGisLinkingVarInBlock(bndchgorigvars[0], blocknr))); /*lint !e613*/

            /* new lower bound */
            if( conshdlrdata->pendingbndtypes[k] == SCIP_BOUNDTYPE_LOWER &&
               SCIPisFeasLT(scip, val, conshdlrdata->pendingnewbnds[k]) )
            {
               SCIP_CALL( SCIPchgVarUbGlobal(scip, vars[i], 0.0) );
               ++(*propcount);
               break;
            }
            /* new upper bound */
            if( conshdlrdata->pendingbndtypes[k] == SCIP_BOUNDTYPE_UPPER &&
               SCIPisFeasGT(scip, val, conshdlrdata->pendingnewbnds[k]) )
            {
               SCIP_CALL( SCIPchgVarUbGlobal(scip, vars[i], 0.0) );
               ++(*propcount);
               break;
            }
         }
      }
      conshdlrdata->pendingbndsactivated = TRUE;
      conshdlrdata->npendingbnds = 0;

      SCIPdebugMessage("Finished handling of pending global bound changes: %d changed bounds\n", *propcount);
   }

   return SCIP_OKAY;
}

/** reset bound changes on pricing variables (called when a node is deactivated) */
static
SCIP_RETCODE resetPricingVarBound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             pricingvar,         /**< variable on which the bound change should be performed */
   SCIP_CONSDATA*        consdata,           /**< constraint data structure */
   int                   i,                  /**< index of the information in the constraint data structure */
   int                   blocknr             /**< number of the pricing problem */
   )
{
   SCIP* origscip;

   assert(scip != NULL);
   assert(pricingvar != NULL);
   assert(consdata != NULL);
   assert(consdata->nactivated >= 1);
   assert(consdata->localbndvars != NULL);
   assert(consdata->localbndtypes != NULL);
   assert(consdata->localnewbnds != NULL);
   assert(consdata->localoldbnds != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   assert(blocknr >= 0 && blocknr < GCGgetNPricingprobs(origscip));

   /* lower bound was changed */
   if( consdata->localbndtypes[i] == SCIP_BOUNDTYPE_LOWER )
   {

      if( GCGgetNIdenticalBlocks(origscip, blocknr) > 1 || GCGgetNIdenticalBlocks(origscip, blocknr) == 0 )
         return SCIP_OKAY;

      assert(SCIPisGE(scip, SCIPvarGetLbLocal(pricingvar), consdata->localnewbnds[i])
         || SCIPisLE(scip, SCIPvarGetLbLocal(pricingvar), SCIPvarGetLbGlobal(consdata->localbndvars[i])));

      if( SCIPisEQ(scip, SCIPvarGetLbGlobal(consdata->localbndvars[i]), consdata->localnewbnds[i]) )
         return SCIP_OKAY;

      if( SCIPisGT(scip, consdata->localoldbnds[i], consdata->localnewbnds[i]) )
         return SCIP_OKAY;

      if( SCIPisGT(scip, SCIPvarGetLbGlobal(consdata->localbndvars[i]), consdata->localoldbnds[i]) )
      {
         SCIP_CALL( SCIPchgVarLb(GCGgetPricingprob(origscip, blocknr), pricingvar, SCIPvarGetLbGlobal(consdata->localbndvars[i])) );
         SCIPdebugMessage("relaxed lower bound of pricing var %s from %g to global bound %g (%s)\n",
            SCIPvarGetName(pricingvar), consdata->localnewbnds[i], SCIPvarGetLbGlobal(consdata->localbndvars[i]), consdata->name);
      }
      else
      {
         SCIP_CALL( SCIPchgVarLb(GCGgetPricingprob(origscip, blocknr), pricingvar, consdata->localoldbnds[i]) );
         SCIPdebugMessage("relaxed lower bound of pricing var %s from %g to %g (%s)\n",
            SCIPvarGetName(pricingvar), consdata->localnewbnds[i], consdata->localoldbnds[i], consdata->name);
      }
   }
   /* upper bound was changed */
   else
   {
      if( GCGgetNIdenticalBlocks(origscip, blocknr) > 1 || GCGgetNIdenticalBlocks(origscip, blocknr) == 0 )
         return SCIP_OKAY;

      assert(SCIPisLE(scip, SCIPvarGetUbLocal(pricingvar), consdata->localnewbnds[i])
         || SCIPisGE(scip, SCIPvarGetUbLocal(pricingvar), SCIPvarGetUbGlobal(consdata->localbndvars[i])));

      if( SCIPisEQ(scip, SCIPvarGetUbGlobal(consdata->localbndvars[i]), consdata->localnewbnds[i]) )
         return SCIP_OKAY;

      if( SCIPisLT(scip, consdata->localoldbnds[i], consdata->localnewbnds[i]) )
         return SCIP_OKAY;

      if( SCIPisLT(scip, SCIPvarGetUbGlobal(consdata->localbndvars[i]), consdata->localoldbnds[i]) )
      {
         SCIP_CALL( SCIPchgVarUb(GCGgetPricingprob(origscip, blocknr), pricingvar, SCIPvarGetUbGlobal(consdata->localbndvars[i])) );
         SCIPdebugMessage("relaxed upper bound of pricing var %s from %g to global bound %g (%s)\n",
            SCIPvarGetName(pricingvar), consdata->localnewbnds[i], SCIPvarGetUbGlobal(consdata->localbndvars[i]), consdata->name);
      }
      else
      {
         SCIP_CALL( SCIPchgVarUb(GCGgetPricingprob(origscip, blocknr), pricingvar, consdata->localoldbnds[i]) );
         SCIPdebugMessage("relaxed upper bound of pricing var %s from %g to %g (%s)\n",
            SCIPvarGetName(pricingvar), consdata->localnewbnds[i], consdata->localoldbnds[i], consdata->name);
      }
   }

   return SCIP_OKAY;
}

/** perform bound changes on pricing variables (called when a node is activated) */
static
SCIP_RETCODE tightenPricingVarBound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             pricingvar,         /**< variable on which the bound change should be performed */
   SCIP_CONSDATA*        consdata,           /**< constraint data structure */
   int                   i,                  /**< index of the information in the constraint data structure */
   int                   blocknr             /**< number of the pricing problem */
   )
{
   SCIP* origscip;

   assert(scip != NULL);
   assert(pricingvar != NULL);
   assert(consdata != NULL);
   assert(consdata->nactivated >= 1);
   assert(consdata->localbndvars != NULL);
   assert(consdata->localbndtypes != NULL);
   assert(consdata->localnewbnds != NULL);
   assert(consdata->localoldbnds != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   assert(blocknr >= 0 && blocknr < GCGgetNPricingprobs(origscip));

   /* lower bound was changed */
   if( consdata->localbndtypes[i] == SCIP_BOUNDTYPE_LOWER )
   {
      consdata->localoldbnds[i] = SCIPvarGetLbLocal(pricingvar);

      if( SCIPisGT(scip, consdata->localnewbnds[i], consdata->localoldbnds[i]) )
      {
         SCIP_CALL( SCIPchgVarLb(GCGgetPricingprob(origscip, blocknr), pricingvar, consdata->localnewbnds[i]) );
         SCIPdebugMessage("tightened lower bound of var %s from %g to %g\n",
            SCIPvarGetName(pricingvar), consdata->localoldbnds[i], consdata->localnewbnds[i]);
      }
   }
   /* upper bound was changed */
   else
   {
      assert(consdata->localbndtypes[i] == SCIP_BOUNDTYPE_UPPER);

      consdata->localoldbnds[i] = SCIPvarGetUbLocal(pricingvar);

      if( SCIPisLT(scip, consdata->localnewbnds[i], consdata->localoldbnds[i]) )
      {
         SCIP_CALL( SCIPchgVarUb(GCGgetPricingprob(origscip, blocknr), pricingvar, consdata->localnewbnds[i]) );
         SCIPdebugMessage("tightened upper bound of var %s from %g to %g\n",
            SCIPvarGetName(pricingvar), consdata->localoldbnds[i], consdata->localnewbnds[i]);
      }
   }

   return SCIP_OKAY;
}

/** For a given local bound change on an original variable, check if the bounds on the variables identical to it are the same
 *
 *  @note If the variable is represented by another one, we check only the representative;
 *        otherwise, we check all variables identical to it
 *
 *  @return TRUE if the variable is in a relevant block AND all variables identical to it have the same bounds
 */
static
SCIP_Bool checkAggregatedLocalBounds(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            bndvars,            /**< all variables whose local bounds were changed */
   SCIP_Real*            bounds,             /**< corresponding new local bounds */
   int                   nbndvars,           /**< number of all variables whose local bounds were changed */
   SCIP_VAR*             bndvar,             /**< original variable whose local bound was changed and which is to be checked */
   SCIP_BOUNDTYPE        bndtype,            /**< type of the new local bound */
   SCIP_VAR*             pricingvar          /**< pricing variable corresponding to the original variable */
   )
{
   SCIP_VAR** identvars;
   int nidentvars;

   assert(GCGvarIsOriginal(bndvar));
   assert(GCGvarIsPricing(pricingvar));

   /* get variables with which the original variable was aggregated */
   identvars = GCGpricingVarGetOrigvars(pricingvar);
   nidentvars = GCGpricingVarGetNOrigvars(pricingvar);

   /* First case: The variable is not represented by another one - check the bounds of all variables it represents */
   if( identvars[0] == bndvar )
   {
      SCIP_Bool identical = TRUE;

      /* If the variable was not aggregated, there is nothing to check */
      if( nidentvars > 1 )
      {
         int i;
         int j;
         SCIP_Real* identbounds;  /* most recent bounds of all identical variables */

         SCIP_CALL( SCIPallocBufferArray(scip, &identbounds, nidentvars) );
         for( j = 0; j < nidentvars; ++j )
            identbounds[j] = SCIP_INVALID;

         /* For all variables on which a bound was changed *and* which are identical to the current variable,
          * get the most recent bound
          */
         for( i = 0; i < nbndvars; ++i )
         {
            assert(GCGvarIsOriginal(bndvars[i]));

            if( GCGvarGetBlock(bndvars[i]) < 0 )
               continue;

            if( GCGpricingVarGetOrigvars(GCGoriginalVarGetPricingVar(bndvars[i]))[0] == identvars[0] )
               for( j = 0; j < nidentvars; ++j )
                  if( identvars[j] == bndvars[i] )
                     identbounds[j] = bounds[i];
         }

         /* Check if the bounds of all identical variables are equal to the one of the representative */
         for( j = 1; j < nidentvars; ++j )
         {
            if( !SCIPisEQ(scip, identbounds[j], identbounds[0]) )
            {
               SCIPerrorMessage("Var <%s> has new local %s bound %g, but identical var <%s> has %g -- don't know how to handle!\n",
                  SCIPvarGetName(bndvar), bndtype == SCIP_BOUNDTYPE_UPPER ? "upper" : "lower",
                     identbounds[0], SCIPvarGetName(identvars[j]), identbounds[j]);
               identical = FALSE;
            }
         }

         SCIPfreeBufferArray(scip, &identbounds);
      }

      return identical;
   }

   /* Second case: The variable is represented by another one due to aggregation; check if its representative has the same bound */
   else
   {
      int i;
      SCIP_Real reprbound;
      SCIP_Real bound;

      /* Get the most recent bound for the bound change variable as well as for its representative */
      reprbound = SCIP_INVALID;
      bound = SCIP_INVALID;
      for( i = 0; i < nbndvars; ++i )
      {
         assert(GCGvarIsOriginal(bndvars[i]));

         if( bndvars[i] == identvars[0] )
            reprbound = bounds[i];
         else if( bndvars[i] == bndvar )
            bound = bounds[i];
      }

      /* Check if the bounds are equal */
      if( !SCIPisEQ(scip, bound, reprbound) )
      {
         SCIPerrorMessage("Var <%s> has new local %s bound %g, but representative <%s> has %g -- don't know how to handle!\n",
            SCIPvarGetName(bndvar), bndtype == SCIP_BOUNDTYPE_UPPER ? "upper" : "lower",
            bound, SCIPvarGetName(identvars[0]), reprbound);
      }

      /* Since the block is not relevant, there is no corresponding pricing variable */
      return FALSE;
   }
}

/** apply local bound changes in the original problem to the pricing problems */
static
SCIP_RETCODE applyLocalBndchgsToPricingprobs(
   SCIP*                 scip,               /* SCIP data structure */
   SCIP_CONS*            cons                /* current masterbranch constraint */
   )
{
   SCIP* origscip;
   SCIP_CONSDATA* consdata;
   int i;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* iterate over all local bound changes in the original problem */
   for( i = 0; i < consdata->nlocalbndchgs; i++ )
   {
      int blocknr;
      assert(GCGvarIsOriginal(consdata->localbndvars[i]));
      blocknr = GCGvarGetBlock(consdata->localbndvars[i]);
      assert(blocknr < GCGgetNPricingprobs(origscip));

      /* if variable belongs to no block, skip it here because the bound changes are treated in the propagation */
      if( blocknr == -1 )
         continue;

      else if( blocknr >= 0 )
      {
         if( checkAggregatedLocalBounds(scip, consdata->localbndvars, consdata->localnewbnds, consdata->nlocalbndchgs, consdata->localbndvars[i],
         consdata->localbndtypes[i], GCGoriginalVarGetPricingVar(consdata->localbndvars[i])) )
         {
            SCIPdebugMessage("adjusting bound of pricing var <%s>\n", SCIPvarGetName(consdata->localbndvars[i]));
            /* set corresponding bound in the pricing problem */
            SCIP_CALL( tightenPricingVarBound(scip, GCGoriginalVarGetPricingVar(consdata->localbndvars[i]), consdata, i, blocknr) );
         }
      }

      else if( blocknr == -2 )
      {
         int j;
         int npricingprobs;
         SCIP_VAR** pricingvars;
         SCIP_Bool aggregated;

         npricingprobs = GCGgetNPricingprobs(origscip);
         pricingvars = GCGlinkingVarGetPricingVars(consdata->localbndvars[i]);
         aggregated = FALSE;
         /* check the blocks in which the linking variable appears */
         for( j = 0; j < npricingprobs; ++j )
         {
            if( pricingvars[j] == NULL )
               continue;

            if( !checkAggregatedLocalBounds(scip, consdata->localbndvars, consdata->localnewbnds, consdata->nlocalbndchgs,
               consdata->localbndvars[i], consdata->localbndtypes[i], pricingvars[j]) )
               aggregated = TRUE;
         }
         if( aggregated )
            continue;

         SCIPdebugMessage("adjusting bound of linking pricing var <%s>\n", SCIPvarGetName(consdata->localbndvars[i]));

         /* set corresponding bound in the pricing problem */
         for( j = 0; j < npricingprobs; ++j )
         {
            if( pricingvars[j] == NULL )
               continue;

            SCIP_CALL( tightenPricingVarBound(scip, pricingvars[j], consdata, i, j) );
         }
      }

      else
      {
         SCIPerrorMessage("blocknr = %d is not valid! This is a serious error!", GCGvarGetBlock(consdata->localbndvars[i]));
         SCIPABORT();
      }
   }

   return SCIP_OKAY;
}

/** undo local bound changes in the original problem to the pricing problems */
static
SCIP_RETCODE undoLocalBndchgsToPricingprobs(
   SCIP*                 scip,               /* SCIP data structure */
   SCIP_CONS*            cons                /* current masterbranch constraint */
   )
{
   SCIP* origscip;
   SCIP_CONSDATA* consdata;
   int i;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* iterate over all local bound changes in the original problem */
   for( i = consdata->nlocalbndchgs - 1; i >= 0; i-- )
   {
      int blocknr;
      blocknr = GCGvarGetBlock(consdata->localbndvars[i]);
      assert(GCGvarIsOriginal(consdata->localbndvars[i]));
      assert(blocknr < GCGgetNPricingprobs(origscip));

      /* if variable belongs to no block, local bound in master was set, is reset automatically */
      if( blocknr == -1 )
         continue;

      else if( blocknr >= 0 )
      {
         assert(GCGgetPricingprob(origscip, GCGvarGetBlock(consdata->localbndvars[i])) != NULL);

         /* reset corresponding bound in the pricing problem */

         SCIP_CALL( resetPricingVarBound(scip,
               GCGoriginalVarGetPricingVar(consdata->localbndvars[i]), consdata, i, blocknr));
      }
      else if( blocknr == -2 )
      {
         int j;
         SCIP_VAR** pricingvars;
         int npricingprobs;

         /* if the variable is linking, we have to perform the same step as above for every existing block*/
         assert(GCGoriginalVarIsLinking(consdata->localbndvars[i]));
         pricingvars = GCGlinkingVarGetPricingVars(consdata->localbndvars[i]);
         npricingprobs = GCGgetNPricingprobs(origscip);

         /* reset corresponding bound in the pricing problem */
         /* lower bound was changed */
         for( j = 0; j < npricingprobs; ++j )
         {
            assert(GCGgetPricingprob(origscip, j) != NULL);
            if( pricingvars[j] == NULL )
               continue;

            assert(GCGgetPricingprob(origscip, j) != NULL);

            /* reset corresponding bound in the pricing problem */
            SCIP_CALL( resetPricingVarBound(scip, pricingvars[j], consdata, i, j) );
         }
      }
      else
      {
         SCIPerrorMessage("blocknr = %d is not valid! This is a serious error!", blocknr);
         SCIPABORT();
      }
   }

   return SCIP_OKAY;
}

static
SCIP_RETCODE ensureCollectedBndvarsSize(
   SCIP* scip,
   SCIP_CONSHDLRDATA *conshdlrdata,
   int blocknr,
   int minsize
   )
{
   int oldsize = conshdlrdata->maxcollectedbndvars[blocknr];
   if( oldsize < minsize )
   {
      int newsize = SCIPcalcMemGrowSize(scip, minsize);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->collectedbndvars[blocknr]), oldsize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->collectedlbnds[blocknr]), oldsize, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->collectedubnds[blocknr]), oldsize, newsize) );
      conshdlrdata->maxcollectedbndvars[blocknr] = newsize;
   }
   return SCIP_OKAY;
}

static
SCIP_RETCODE ensureLinkingvarIndxsSize(
   SCIP* scip,
   SCIP_CONSHDLRDATA *conshdlrdata,
   int minsize
)
{
   int oldsize = conshdlrdata->maxlinkingvaridxs;
   if( oldsize < minsize )
   {
      int newsize = SCIPcalcMemGrowSize(scip, minsize);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->linkingvaridxs), oldsize, newsize) );
      BMSclearMemoryArray(&conshdlrdata->linkingvaridxs[oldsize], (newsize - oldsize));
      conshdlrdata->maxlinkingvaridxs = newsize;
   }
   return SCIP_OKAY;
}

/** apply local bound changes on the original variables on newly generated master variables */
static
SCIP_RETCODE applyLocalBndchgsToPricedMastervars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< current masterbranch constraint */
   int*                  propcount           /**< number of applied bound changes */
   )
{
   SCIP_CONSHDLR *conshdlr;
   SCIP_CONSHDLRDATA *conshdlrdata;
   SCIP_CONSDATA *consdata;
   SCIP_CONSDATA *curconsdata;
   SCIP_VAR **vars;
   int nvars;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* get master problem variables generated during pricing */
   vars = GCGmasterGetPricedvars(scip);
   nvars = GCGmasterGetNPricedvars(scip);

   if( consdata->npropvars < nvars )
   {
      SCIP_CONS *curcons;
      SCIP_HASHMAP *origvar2idx;
      int nbndvars;
      int nlocalbndchgs;
      int i;
      int j;
      int k;
      SCIP_VAR*** collectedbndvars;
      SCIP_Real** collectedlbnds;
      SCIP_Real** collectedubnds;
      int* ncollectedbndvars;
      int hashmapsize;
      int nblocks;
      int blocknr;
      SCIP_VAR *bndvar;
      int idx;
      SCIP_Real bnd;
      SCIP_Bool islinking;
      int* linkingvaridxs;
      int nlinkingvars;

      nblocks = GCGgetNPricingprobs(GCGmasterGetOrigprob(scip));
      nlinkingvars = 0;

      collectedbndvars = conshdlrdata->collectedbndvars;
      collectedlbnds = conshdlrdata->collectedlbnds;
      collectedubnds = conshdlrdata->collectedubnds;
      ncollectedbndvars = conshdlrdata->ncollectedbndvars;

      BMSclearMemoryArray(ncollectedbndvars, conshdlrdata->maxblocknum);

      /* propagate local bound changes on the path from the current node to the root */
      curcons = cons;
      hashmapsize = 0;
      while ( curcons != NULL )
      {
         curconsdata = SCIPconsGetData(curcons);
         assert(curconsdata != NULL);
         //if( curconsdata->npropvars == nvars )
         //   break;
         hashmapsize += (conshdlrdata->enforceproper ? curconsdata->nlocalbndchgs : curconsdata->nbranchingchgs);
         curcons = curconsdata->parentcons;
      }
      if( hashmapsize > SCIPgetNVars(GCGmasterGetOrigprob(scip)))
         hashmapsize = SCIPgetNVars(GCGmasterGetOrigprob(scip));

      SCIP_CALL(SCIPhashmapCreate(&origvar2idx, SCIPblkmem(scip), hashmapsize));

      curcons = cons;
      nbndvars = 0;
      while ( curcons != NULL)
      {
         curconsdata = SCIPconsGetData(curcons);
         assert(curconsdata != NULL);
         //if( curconsdata->npropvars == nvars )
         //   break;

         /* propagate all bound changes or only the branching bound changes, depending on the setting for the enforcement of proper variables */
         nlocalbndchgs = (conshdlrdata->enforceproper ? curconsdata->nlocalbndchgs : curconsdata->nbranchingchgs);

         /* iterate over bound changes performed at the current node's equivalent in the original tree */
         for ( k = 0; k < nlocalbndchgs; ++k )
         {
            linkingvaridxs = NULL;
            bndvar = curconsdata->localbndvars[k];
            islinking = GCGoriginalVarIsLinking(bndvar);

            /* get the block the original variable is in */
            blocknr = GCGvarGetBlock(bndvar);
            assert(GCGvarIsOriginal(bndvar));
            assert(blocknr < nblocks);

            if( blocknr >= 0 || islinking)
            {
               idx = SCIPhashmapGetImageInt(origvar2idx, bndvar);
               bnd = curconsdata->localnewbnds[k];

               if( islinking )
               {
                  blocknr = 0;
                  if( idx < INT_MAX )
                  {
                     linkingvaridxs = conshdlrdata->linkingvaridxs[idx];
                  }
                  else
                  {
                     ensureLinkingvarIndxsSize(scip, conshdlrdata, nlinkingvars + 1);
                     linkingvaridxs = conshdlrdata->linkingvaridxs[nlinkingvars];
                     if( linkingvaridxs == NULL )
                     {
                        SCIP_CALL( SCIPallocBlockMemoryArray(scip, &linkingvaridxs, conshdlrdata->maxblocknum) );
                        conshdlrdata->linkingvaridxs[nlinkingvars] = linkingvaridxs;
                     }
                     SCIPhashmapInsertInt(origvar2idx, bndvar, nlinkingvars);
                     ++nlinkingvars;

                     for( i = 0; i < nblocks; ++i )
                        linkingvaridxs[i] = INT_MAX;
                  }
               }

               do
               {
                  assert(blocknr >= 0 && blocknr < GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)));

                  if( islinking )
                  {
                     if( !GCGisLinkingVarInBlock(bndvar, blocknr) )
                        continue;

                     idx = linkingvaridxs[blocknr];
                  }

                  if( curconsdata->localbndtypes[k] == SCIP_BOUNDTYPE_LOWER )
                  {
                     if( idx < INT_MAX )
                     {
                        if( collectedlbnds[blocknr][idx] == SCIP_INVALID || collectedlbnds[blocknr][idx] < bnd )
                           collectedlbnds[blocknr][idx] = bnd;
                     }
                     else
                     {
                        SCIP_CALL(
                           ensureCollectedBndvarsSize(scip, conshdlrdata, blocknr, ncollectedbndvars[blocknr] + 1));
                        if ( islinking )
                           linkingvaridxs[blocknr] = ncollectedbndvars[blocknr];
                        else
                           SCIPhashmapInsertInt(origvar2idx, bndvar, ncollectedbndvars[blocknr]);
                        collectedbndvars[blocknr][ncollectedbndvars[blocknr]] = bndvar;
                        collectedlbnds[blocknr][ncollectedbndvars[blocknr]] = bnd;
                        collectedubnds[blocknr][ncollectedbndvars[blocknr]] = SCIP_INVALID;
                        ++ncollectedbndvars[blocknr];
                        ++nbndvars;
                     }
                  }
                  else
                  {
                     if( idx < INT_MAX )
                     {
                        if( collectedubnds[blocknr][idx] == SCIP_INVALID || collectedubnds[blocknr][idx] > bnd )
                           collectedubnds[blocknr][idx] = bnd;
                     }
                     else
                     {
                        SCIP_CALL(
                           ensureCollectedBndvarsSize(scip, conshdlrdata, blocknr, ncollectedbndvars[blocknr] + 1));
                        if ( islinking )
                           linkingvaridxs[blocknr] = ncollectedbndvars[blocknr];
                        else
                           SCIPhashmapInsertInt(origvar2idx, bndvar, ncollectedbndvars[blocknr]);
                        collectedbndvars[blocknr][ncollectedbndvars[blocknr]] = bndvar;
                        collectedubnds[blocknr][ncollectedbndvars[blocknr]] = bnd;
                        collectedlbnds[blocknr][ncollectedbndvars[blocknr]] = SCIP_INVALID;
                        ++ncollectedbndvars[blocknr];
                        ++nbndvars;
                     }
                  }
               }
               while( islinking && (++blocknr) < nblocks );
            }
         }
         /* proceed with the parent node */
         curcons = curconsdata->parentcons;
      }

      if( nbndvars > 0 )
      {
         SCIP_Real lbnd;
         SCIP_Real ubnd;
         SCIP_HASHMAP* origvals;
         SCIP_Real origval;

         /* iterate over all master variables created after the current node was left the last time */
         for ( i = consdata->npropvars; i < nvars; i++ )
         {
            assert(GCGvarIsMaster(vars[i]));
            blocknr = GCGvarGetBlock(vars[i]);
            assert(blocknr >= 0 && blocknr < GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)));

            /** @todo check if this really works with linking variables */

            /* only look at variables not already fixed to 0 or that belong to no block */
            if((SCIPisFeasZero(scip, SCIPvarGetUbLocal(vars[i]))))
               continue;

            origvals = GCGmasterVarGetOrigvalmap(vars[i]);

            /* iterate over all original variables whose bound was changed */
            for ( j = 0; j < ncollectedbndvars[blocknr]; j++ )
            {
               bndvar = collectedbndvars[blocknr][j];

               assert(GCGvarGetBlock(bndvar) == blocknr || (GCGoriginalVarIsLinking(bndvar) && GCGisLinkingVarInBlock(bndvar, blocknr)));

               origval = SCIPhashmapGetImageReal(origvals, bndvar);
               /* variables belong to the same block -> set origval to 0.0 if not in map */
               if( origval == SCIP_INVALID )
                  origval = 0.0;
               lbnd = collectedlbnds[blocknr][j];
               ubnd = collectedubnds[blocknr][j];

               /* branching imposes new bound */
               if( (lbnd != SCIP_INVALID && SCIPisFeasLT(scip, origval, lbnd))
                   || (ubnd != SCIP_INVALID && SCIPisFeasGT(scip, origval, ubnd)) )
               {
                  SCIPdebugMessage("Changing upper bound of var %s\n", SCIPvarGetName(vars[i]));
                  SCIP_CALL(SCIPchgVarUb(scip, vars[i], 0.0));
                  ++(*propcount);
                  break;
               }
            }
         }
      }
      SCIPhashmapFree(&origvar2idx);
   }

   SCIPdebugMessage("Finished propagation of newly created variables: %d changed bounds\n", *propcount);

   return SCIP_OKAY;
}

/** apply local bound changes on original variables that have been directly copied to the master problem */
static
SCIP_RETCODE applyLocalBndchgsToCopiedMastervars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< current masterbranch constraint */
   int*                  propcount           /**< number of applied bound changes */
   )
{
   SCIP_CONSDATA* consdata;
   int i;

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* apply local bound changes */
   for( i = 0; i < consdata->nlocalbndchgs; i++ )
   {
      SCIP_VAR* mastervar;

      assert(GCGvarIsOriginal(consdata->localbndvars[i]));

      /** @todo this might lead to an error with linking variables ? */
      if( GCGvarGetBlock(consdata->localbndvars[i]) >= 0 )
         continue;

      assert(GCGoriginalVarGetNMastervars(consdata->localbndvars[i]) >= 1);

      mastervar = GCGoriginalVarGetMastervars(consdata->localbndvars[i])[0];
      assert(GCGvarGetBlock(mastervar) == -1);

      if( consdata->localbndtypes[i] == SCIP_BOUNDTYPE_LOWER )
      {
         if( SCIPisLT(scip, SCIPvarGetLbLocal(mastervar), consdata->localnewbnds[i]) )
         {
            SCIP_CALL( SCIPchgVarLb(scip, mastervar, consdata->localnewbnds[i]) );
            ++(*propcount);
            SCIPdebugMessage("changed lb of copied original var %s locally to %g\n", SCIPvarGetName(consdata->localbndvars[i]), consdata->localnewbnds[i]);
         }
      }
      else
      {
         if( SCIPisGT(scip, SCIPvarGetUbLocal(mastervar), consdata->localnewbnds[i]) )
         {
            SCIP_CALL( SCIPchgVarUb(scip, mastervar, consdata->localnewbnds[i]) );
            ++(*propcount);
            SCIPdebugMessage("changed ub of copied original var %s locally to %g\n", SCIPvarGetName(consdata->localbndvars[i]), consdata->localnewbnds[i]);
         }
      }
   }

   SCIPdebugMessage("Finished propagation of bounds of copied original variables: %d bounds changed.\n", *propcount);

   return SCIP_OKAY;
}

/*
 * Callback methods
 */


/** destructor of constraint handler to free constraint handler data (called when SCIP is exiting) */
static
SCIP_DECL_CONSFREE(consFreeMasterbranch)
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* free constraint handler storage */
   assert(conshdlrdata->stack == NULL);
   SCIPfreeMemory(scip, &conshdlrdata);

   return SCIP_OKAY;
}


/** initialization method of constraint handler (called after problem was transformed) */
static
SCIP_DECL_CONSINIT(consInitMasterbranch)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* prepare stack */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &conshdlrdata->stack, conshdlrdata->maxstacksize) );
   conshdlrdata->nstack = 0;

   /* prepare pending bound changes */
   conshdlrdata->npendingbnds = 0;
   conshdlrdata->maxpendingbnds = SCIPcalcMemGrowSize(scip, 1);
   conshdlrdata->pendingbndsactivated = TRUE;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->pendingvars), conshdlrdata->maxpendingbnds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->pendingbndtypes), conshdlrdata->maxpendingbnds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->pendingoldbnds), conshdlrdata->maxpendingbnds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->pendingnewbnds), conshdlrdata->maxpendingbnds) );

   conshdlrdata->maxblocknum = SCIPcalcMemGrowSize(scip, GCGgetNPricingprobs(GCGmasterGetOrigprob(scip)));
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->collectedbndvars),  conshdlrdata->maxblocknum) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->collectedlbnds),  conshdlrdata->maxblocknum) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->collectedubnds),  conshdlrdata->maxblocknum) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->ncollectedbndvars),  conshdlrdata->maxblocknum) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(conshdlrdata->maxcollectedbndvars),  conshdlrdata->maxblocknum) );
   BMSclearMemoryArray(conshdlrdata->collectedbndvars, conshdlrdata->maxblocknum);
   BMSclearMemoryArray(conshdlrdata->collectedlbnds, conshdlrdata->maxblocknum);
   BMSclearMemoryArray(conshdlrdata->collectedubnds, conshdlrdata->maxblocknum);
   BMSclearMemoryArray(conshdlrdata->ncollectedbndvars, conshdlrdata->maxblocknum);
   BMSclearMemoryArray(conshdlrdata->maxcollectedbndvars, conshdlrdata->maxblocknum);

   conshdlrdata->maxlinkingvaridxs = 0;
   conshdlrdata->linkingvaridxs = NULL;

   return SCIP_OKAY;
}


/** solving process initialization method of constraint handler (called when branch and bound process is about to begin) */
static
SCIP_DECL_CONSINITSOL(consInitsolMasterbranch)
{  /*lint --e{715}*/
   SCIP_CONS* cons;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   /* create masterbranch constraint for the root node */
   SCIP_CALL( GCGcreateConsMasterbranch(scip, &cons, "root-masterbranch", NULL,  NULL, NULL, NULL, NULL, 0, 0) );
   GCGconsOrigbranchSetMastercons(GCGconsOrigbranchGetActiveCons(GCGmasterGetOrigprob(scip)), cons);

   conshdlrdata->nstack = 1;
   conshdlrdata->stack[0] = cons;

   return SCIP_OKAY;
}


/** deinitialization method of constraint handler (called before transformed problem is freed) */
static
SCIP_DECL_CONSEXIT(consExitMasterbranch)
{  /*lint --e{715}*/
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->nstack == 1 || SCIPgetNNodes(scip) == 0);

   /* free stack */
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->stack), conshdlrdata->maxstacksize);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->pendingvars), conshdlrdata->maxpendingbnds);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->pendingbndtypes), conshdlrdata->maxpendingbnds);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->pendingoldbnds), conshdlrdata->maxpendingbnds);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->pendingnewbnds), conshdlrdata->maxpendingbnds);

   for( i = 0; i < conshdlrdata->maxblocknum; ++i )
   {
      if( conshdlrdata->maxcollectedbndvars[i] > 0)
      {
         SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedbndvars[i]), conshdlrdata->maxcollectedbndvars[i]);
         SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedlbnds[i]), conshdlrdata->maxcollectedbndvars[i]);
         SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedubnds[i]), conshdlrdata->maxcollectedbndvars[i]);
      }
   }
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedbndvars), conshdlrdata->maxblocknum);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedlbnds), conshdlrdata->maxblocknum);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->collectedubnds), conshdlrdata->maxblocknum);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->ncollectedbndvars), conshdlrdata->maxblocknum);
   SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->maxcollectedbndvars), conshdlrdata->maxblocknum);

   for( i = 0; i < conshdlrdata->maxlinkingvaridxs; ++i )
   {
      if( conshdlrdata->linkingvaridxs[i] != NULL)
         SCIPfreeBlockMemoryArray(scip, &(conshdlrdata->linkingvaridxs[i]), conshdlrdata->maxblocknum);
      else
         break;
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(conshdlrdata->linkingvaridxs), conshdlrdata->maxlinkingvaridxs);

   return SCIP_OKAY;
}


/** constraint activation notification method of constraint handler */
static
SCIP_DECL_CONSACTIVE(consActiveMasterbranch)
{
   SCIP* origscip;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(cons != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL);

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert((consdata->node == NULL) == (consdata->parentcons == NULL));

   /* @fixme: This is a hack */
   if( consdata->node == NULL )
   {
      SCIPdebugMessage("root node not present in masterconsdata!\n");
      consdata->node = SCIPgetRootNode(scip);
   }

   assert(consdata->node != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);


   SCIPdebugMessage("Activating ");
   /* If the node is activated the first time, we have to initialize the constraint data first */
   if( consdata->nactivated == 0 )
   {
      SCIPdebugPrintf("for the first time\n");
      SCIP_CALL( initializeConsdata(scip, cons) );
   }
   else
      SCIPdebugPrintf("\n");

   consdata->nactivated++;

   /* The node has to be repropagated if new variables were created after the node was left the last time
    * or if new bound changes on directly transferred variables were found
    */
   assert(GCGmasterGetNPricedvars(scip) >= consdata->npropvars);
   if( GCGmasterGetNPricedvars(scip) > consdata->npropvars || consdata->ncopiedvarbnds > 0 )
   {
      consdata->needprop = TRUE;
      SCIP_CALL( SCIPrepropagateNode(scip, consdata->node) );
   }

   if( consdata->nlocalbndchgs - consdata->nlocalbndchgstreated[conshdlrdata->nstack] > 0 )
      SCIPdebugMessage("added %d boundchanges from previous nodes!\n", consdata->nlocalbndchgs - consdata->nlocalbndchgstreated[conshdlrdata->nstack]);

   /* put constraint on the stack */
   if( conshdlrdata->nstack >= conshdlrdata->maxstacksize )
   {
      int newsize = SCIPcalcMemGrowSize(scip,  conshdlrdata->nstack+1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(conshdlrdata->stack), conshdlrdata->maxstacksize, newsize) );

      SCIPdebugMessage("reallocating Memory for stack! %d --> %d\n", conshdlrdata->maxstacksize, newsize);
      conshdlrdata->maxstacksize = newsize;
   }

   conshdlrdata->stack[conshdlrdata->nstack] = cons;
   (conshdlrdata->nstack)++;

   SCIPdebugMessage("Activating masterbranch constraint: <%s> [stack size: %d], needprop = %u.\n",
      consdata->name, conshdlrdata->nstack, consdata->needprop);

   /* apply global bound changes in the original problem to the pricing problems */
   SCIP_CALL( applyGlobalBndchgsToPricingprobs(scip) );

   /* apply local bound changes in the original problem to the pricing problems */
   SCIP_CALL( applyLocalBndchgsToPricingprobs(scip, cons) );

   /* call branching specific activation method */
   if( consdata->branchrule != NULL )
   {
      SCIP_CALL( GCGrelaxBranchActiveMaster(origscip, consdata->branchrule, consdata->branchdata) );
   }

   return SCIP_OKAY;
}

/** constraint deactivation notification method of constraint handler */
static
SCIP_DECL_CONSDEACTIVE(consDeactiveMasterbranch)
{
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONSDATA* consdata;
   SCIP* origscip;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(cons != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL || conshdlrdata->nstack == 1);
   assert(conshdlrdata->nstack > 0);
   assert(conshdlrdata->nstack == 1 || cons == conshdlrdata->stack[conshdlrdata->nstack-1]);

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->nactivated >= 1);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   if( !conshdlrdata->pendingbndsactivated )
   {
      SCIPdebugMessage("We need repropagation\n");
      consdata->needprop = TRUE;
   }

   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
      consdata->npropvars = GCGmasterGetNPricedvars(scip);

   /* remove constraint from the stack */
   (conshdlrdata->nstack)--;

   SCIPdebugMessage("Deactivating masterbranch constraint: <%s> [stack size: %d].\n",
      consdata->name, conshdlrdata->nstack);

   /* undo local bound changes in the original problem to the pricing problems */
   SCIP_CALL( undoLocalBndchgsToPricingprobs(scip, cons) );

   /* call branching specific deactivation method */
   if( consdata->branchrule != NULL )
   {
      SCIP_CALL( GCGrelaxBranchDeactiveMaster(origscip, consdata->branchrule, consdata->branchdata) );
   }

   return SCIP_OKAY;
}


/** frees specific constraint data */
static
SCIP_DECL_CONSDELETE(consDeleteMasterbranch)
{
   SCIP* origscip;
   SCIP_CONSDATA* parentconsdata;
   SCIP_CONSDATA** childconsdatas;
   SCIP_CONS** childconss;
   int nchildconss;
   int i;

   assert(scip != NULL);
   assert(conshdlr != NULL);

   assert(cons != NULL);
   assert(consdata != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(*consdata != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   SCIPdebugMessage("Deleting masterbranch constraint: <%s>.\n", (*consdata)->name);

   /* remove original branching constraints if not yet done
    * (might happen if node is cut off before branching decisions are transferred to the original problem)
    */
   SCIP_CALL( GCGconsMasterbranchReleaseOrigbranchConss(scip, origscip, cons) );

   /* free arrays with local bound changes on copied original variables */
   if( (*consdata)->maxcopiedvarbnds > 0 )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &((*consdata)->copiedvars), (*consdata)->maxcopiedvarbnds);
      SCIPfreeBlockMemoryArrayNull(scip, &((*consdata)->copiedvarbndtypes), (*consdata)->maxcopiedvarbnds);
      SCIPfreeBlockMemoryArrayNull(scip, &((*consdata)->copiedvarbnds), (*consdata)->maxcopiedvarbnds);
   }

   /* free arrays with local bound changes on original variables belonging to a unique block */
   SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->nlocalbndchgstreated, (*consdata)->maxlocalbndchgstreated);
   if( (*consdata)->maxlocalbndchgs > 0 )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->localoldbnds, (*consdata)->maxlocalbndchgs);
      SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->localnewbnds, (*consdata)->maxlocalbndchgs);
      SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->localbndtypes, (*consdata)->maxlocalbndchgs);
      SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->localbndvars, (*consdata)->maxlocalbndchgs);
   }

   /* delete branchdata if the corresponding origcons has already been deleted;
    * otherwise, it will be deleted by the corresponding origbranch constraint
    */
   if( (*consdata)->origcons == NULL && (*consdata)->branchdata != NULL )
   {
      SCIP_CALL( GCGrelaxBranchDataDelete(origscip, (*consdata)->branchrule, &(*consdata)->branchdata) );
      (*consdata)->branchdata = NULL;
      (*consdata)->branchdata = NULL;
   }
   else
   {
      if( (*consdata)->branchdata != NULL )
      {
         SCIP_CALL( GCGrelaxBranchDataDelete(origscip, (*consdata)->branchrule, &(*consdata)->branchdata) );
         (*consdata)->branchdata = NULL;
         if( (*consdata)->origcons != NULL )
         {
            GCGconsOrigbranchSetBranchdata((*consdata)->origcons, NULL);
         }
      }
   }

   /* set the mastercons pointer of the corresponding origcons to NULL */
   if( (*consdata)->origcons != NULL )
   {
      assert(GCGconsOrigbranchGetMastercons((*consdata)->origcons) == cons);
      GCGconsOrigbranchSetMastercons((*consdata)->origcons, NULL);
   }

   /* remove branching constraints at child nodes */
   nchildconss = (*consdata)->nchildconss;
   if( nchildconss > 0 )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &childconsdatas, nchildconss) );
      SCIP_CALL( SCIPallocBufferArray(scip, &childconss, nchildconss) );

      for( i = 0; i < nchildconss; ++i )
      {
         if( (*consdata)->childconss != NULL && (*consdata)->childconss[i] != NULL )
         {
            childconsdatas[i] = SCIPconsGetData((*consdata)->childconss[i]);
            childconss[i] = (*consdata)->childconss[i];
         }
         else
         {
            childconsdatas[i] = NULL;
            childconss[i] = NULL;
         }
      }

      /* delete childnodes */
      for( i = 0; i < nchildconss; ++i )
      {
         SCIPdebugMessage("Deleting %d childnodes\n", nchildconss);

         if( childconss[i] != NULL )
         {
            /*SCIP_CALL( consDeleteMasterbranch(scip, conshdlr, childcons[i], &childconsdatas[i]) );*/
            SCIP_CALL( SCIPreleaseCons(scip, &childconss[i]) );
            childconss[i] = NULL;
         }
      }

      SCIPfreeBufferArray(scip, &childconss);
      SCIPfreeBufferArray(scip, &childconsdatas);
   }
   assert((*consdata)->nchildconss == 0);

   /* set the pointer in the parent node to NULL */
   if( (*consdata)->parentcons != NULL )
   {
      SCIP_Bool isinprobing;
#ifndef NDEBUG
      SCIP_Bool childdeleted = FALSE;
#endif

      parentconsdata = SCIPconsGetData((*consdata)->parentcons);

      isinprobing = (SCIPgetStage(scip) <= SCIP_STAGE_SOLVING && SCIPinProbing(scip)) || (SCIPgetStage(origscip) <= SCIP_STAGE_SOLVING && SCIPinProbing(origscip));
      if( isinprobing )
         parentconsdata->probingtmpcons = NULL;

      for( i = 0; i < parentconsdata->nchildconss; ++i )
      {
         if( parentconsdata->childconss[i] == cons )
         {
            parentconsdata->childconss[i] = parentconsdata->childconss[parentconsdata->nchildconss-1];

            parentconsdata->childconss[parentconsdata->nchildconss-1] = NULL;
            parentconsdata->nchildconss--;
#ifndef NDEBUG
            childdeleted = TRUE;
#endif
            break;
         }
      }
      assert(childdeleted || isinprobing);
   }

   SCIPfreeBlockMemoryArrayNull(scip, &(*consdata)->childconss, (*consdata)->maxchildconss);

   if( (*consdata)->name != NULL )
   {
      SCIPfreeBlockMemoryArray(scip, &(*consdata)->name, strlen((*consdata)->name)+1);
   }

   SCIPfreeBlockMemoryNull(scip, consdata);
   *consdata = NULL;

   return SCIP_OKAY;
}


/** domain propagation method of constraint handler */
static
SCIP_DECL_CONSPROP(consPropMasterbranch)
{  /*lint --e{715}*/
   SCIP* origscip;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS* cons;
   SCIP_CONSDATA* consdata;

   int propcount;

   assert(conshdlr != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL);

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   *result = SCIP_DIDNOTRUN;

   /* the constraint data of the cons related to the current node */
   cons = conshdlrdata->stack[conshdlrdata->nstack-1];
   assert(cons != NULL);

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( !consdata->needprop && consdata->ncopiedvarbnds == 0 )
   {
      SCIPdebugMessage("No propagation of masterbranch constraint needed: <%s>, stack size = %d.\n",
         consdata->name, conshdlrdata->nstack);

      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   SCIPdebugMessage("Starting propagation of masterbranch constraint: <%s>, stack size = %d, newvars = %d, npendingbnds = %d, npropbounds = %d.\n",
      consdata->name, conshdlrdata->nstack, GCGmasterGetNPricedvars(scip) - consdata->npropvars, conshdlrdata->npendingbnds, consdata->ncopiedvarbnds);

   *result = SCIP_DIDNOTFIND;

   propcount = 0;

   /* apply global bound changes on original problem variables to the master problem */
   SCIP_CALL( applyGlobalBndchgsToPricedMastervars(scip, &propcount) );

   /* apply local bound changes on the original variables on newly generated master variables */
   SCIP_CALL( applyLocalBndchgsToPricedMastervars(scip, cons, &propcount) );

   /* apply local bound changes on original variables that have been directly copied to the master problem */
   SCIP_CALL( applyLocalBndchgsToCopiedMastervars(scip, cons, &propcount) );

   /* call branching rule specific propagation method */
   if( consdata->branchrule != NULL )
   {
      /** @todo count number of propagations */
      SCIP_CALL( GCGrelaxBranchPropMaster(origscip, consdata->branchrule, consdata->branchdata, result) );
   }

   if( *result != SCIP_CUTOFF && propcount > 0 )
      *result = SCIP_REDUCEDDOM;

   consdata->needprop = FALSE;
   consdata->npropvars = GCGmasterGetNPricedvars(scip);

   return SCIP_OKAY;
}


/** enforcement method for LP solutions */
static
SCIP_DECL_CONSENFOLP(consEnfolpMasterbranch)
{  /*lint --e{715}*/
   *result = SCIP_FEASIBLE;

   return SCIP_OKAY;
}

/** enforcement method for pseudo solutions */
static
SCIP_DECL_CONSENFOPS(consEnfopsMasterbranch)
{  /*lint --e{715}*/
   *result = SCIP_FEASIBLE;

   return SCIP_OKAY;
}

/** check method for solutions */
static
SCIP_DECL_CONSCHECK(consCheckMasterbranch)
{  /*lint --e{715}*/
   *result = SCIP_FEASIBLE;

   return SCIP_OKAY;
}

/** variable lock method */
static
SCIP_DECL_CONSLOCK(consLockMasterbranch)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/*
 * Callback methods of event handler
 */

/** solving process initialization method of event handler (called when branch and bound process is about to begin) */
static
SCIP_DECL_EVENTINITSOL(eventInitsolOrigvarbound)
{  /*lint --e{715}*/
   SCIP_VAR** vars;
   int nvars;
   int i;

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   for( i = 0; i < nvars; i++ )
   {
      SCIP_CALL( SCIPcatchVarEvent(scip, vars[i], SCIP_EVENTTYPE_GBDCHANGED | SCIP_EVENTTYPE_BOUNDCHANGED,
            eventhdlr, NULL, NULL) );
   }

   return SCIP_OKAY;
}

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecOrigvarbound)
{  /*lint --e{715}*/
   SCIP* masterscip;
   SCIP_EVENTTYPE eventtype;
   SCIP_VAR* var;
   SCIP_Real oldbound;
   SCIP_Real newbound;
   int blocknr;
#ifdef SCIP_DEBUG
   SCIP_Bool handled = FALSE;
#endif
   SCIP_VAR** mastervars;
#ifndef NDEBUG
   int nmastervars;
   SCIP_Real* mastervals;
#endif

   /* get master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   /* get event data */
   eventtype = SCIPeventGetType(event);
   var = SCIPeventGetVar(event);
   oldbound = SCIPeventGetOldbound(event);
   newbound = SCIPeventGetNewbound(event);

   SCIPdebugMessage("eventexec: eventtype = 0x%x, var = %s, oldbound = %f, newbound = %f\n", (unsigned int) eventtype, SCIPvarGetName(var), oldbound, newbound);

   if( !GCGrelaxIsInitialized(scip) )
   {
//      assert(SCIPvarGetData(var) == NULL);
      SCIPdebugMessage("Ignoring since in presolving / propagating.\n");
      return SCIP_OKAY;
   }

   assert(GCGvarIsOriginal(var));
   blocknr = GCGvarGetBlock(var);

   mastervars = GCGoriginalVarGetMastervars(var);
#ifndef NDEBUG
   nmastervars = GCGoriginalVarGetNMastervars(var);
   mastervals = GCGoriginalVarGetMastervals(var);
#endif

   /* A global bound change might turn the current relaxation solution invalid */
   if( SCIPisRelaxSolValid(scip)
      && (((eventtype & SCIP_EVENTTYPE_GLBCHANGED) != 0 && SCIPisFeasLT(scip, SCIPgetRelaxSolVal(scip, var), newbound))
      || ((eventtype & SCIP_EVENTTYPE_GUBCHANGED) != 0 && SCIPisFeasGT(scip, SCIPgetRelaxSolVal(scip, var), newbound))) )
   {
      SCIP_CALL( SCIPmarkRelaxSolInvalid(scip) );
   }

   /* deal with variables present in the pricing */
   if( blocknr >= 0 && GCGisPricingprobRelevant(scip, blocknr) )
   {
      SCIPdebugMessage("Pricing var!\n");
      if( (eventtype & SCIP_EVENTTYPE_GLBCHANGED) != 0 )
      {
#ifdef SCIP_DEBUG
         handled = TRUE;
#endif
         SCIP_CALL( addPendingBndChg(masterscip, GCGoriginalVarGetPricingVar(var), SCIP_BOUNDTYPE_LOWER, oldbound, newbound) );
      }
      if( (eventtype & SCIP_EVENTTYPE_GUBCHANGED) != 0 )
      {
#ifdef SCIP_DEBUG
         handled = TRUE;
#endif
         SCIP_CALL( addPendingBndChg(masterscip, GCGoriginalVarGetPricingVar(var), SCIP_BOUNDTYPE_UPPER, oldbound, newbound) );
      }
   }
   /* deal with variables appearing in the master only */
   if( blocknr == -1 && SCIPgetStage(masterscip) >= SCIP_STAGE_SOLVING )
   {
      assert(nmastervars == 1);
      assert(mastervals[0] == 1);
      assert(mastervars[0] != NULL);

      SCIPdebugMessage("Master var!\n");
      if( (eventtype & SCIP_EVENTTYPE_GLBCHANGED) != 0 )
      {
#ifdef SCIP_DEBUG
         handled = TRUE;
#endif
         SCIP_CALL( addPendingBndChg(masterscip, mastervars[0], SCIP_BOUNDTYPE_LOWER, oldbound, newbound) );
      }
      if( (eventtype & SCIP_EVENTTYPE_GUBCHANGED) != 0 )
      {
#ifdef SCIP_DEBUG
         handled = TRUE;
#endif
         SCIP_CALL( addPendingBndChg(masterscip, mastervars[0], SCIP_BOUNDTYPE_UPPER, oldbound, newbound) );
      }
   }
   /* deal with linking variables */
   if( blocknr == -2 )
   {
      int npricingprobs;
      SCIP_VAR** pricingvars;
      int i;

      SCIPdebugMessage("Linking var!\n");
      pricingvars = GCGlinkingVarGetPricingVars(var);
      npricingprobs = GCGgetNPricingprobs(scip);

      assert(nmastervars >= 1);
      assert(mastervals[0] == 1);
      assert(mastervars[0] != NULL);
      assert(GCGvarGetBlock(mastervars[0]) == -1);

      if( (eventtype & SCIP_EVENTTYPE_GLBCHANGED) != 0 )
      {
         if( SCIPgetStage(masterscip) >= SCIP_STAGE_SOLVING )
         {
#ifdef SCIP_DEBUG
            handled = TRUE;
#endif
            /* add the bound change in the master */
            SCIP_CALL( addPendingBndChg(masterscip, mastervars[0], SCIP_BOUNDTYPE_LOWER, oldbound, newbound) );
         }

         /* add the bound change to the pricing problems */
         for( i = 0; i < npricingprobs; ++i )
         {
            if( pricingvars[i] == NULL )
               continue;
#ifdef SCIP_DEBUG
            handled = TRUE;
#endif
            SCIP_CALL( addPendingBndChg(masterscip, pricingvars[i], SCIP_BOUNDTYPE_LOWER, oldbound, newbound) );
         }
      }
      if( (eventtype & SCIP_EVENTTYPE_GUBCHANGED) != 0 )
      {
         if( SCIPgetStage(masterscip) >= SCIP_STAGE_SOLVING )
         {
#ifdef SCIP_DEBUG
            handled = TRUE;
#endif
            /* add the bound change in the master */
            SCIP_CALL( addPendingBndChg(masterscip, mastervars[0], SCIP_BOUNDTYPE_UPPER, oldbound, newbound) );
         }

         /* add the bound change to the pricing problems */
         for( i = 0; i < npricingprobs; ++i )
         {
            if( pricingvars[i] == NULL )
               continue;
#ifdef SCIP_DEBUG
            handled = TRUE;
#endif
            SCIP_CALL( addPendingBndChg(masterscip, pricingvars[i], SCIP_BOUNDTYPE_UPPER, oldbound, newbound) );
         }

      }

   }
#ifdef SCIP_DEBUG
   if( !handled )
   {
      SCIPdebugMessage("Effectively ignoring this change\n");
   }
#endif

   return SCIP_OKAY;
}


/*
 * interface methods
 */


/** creates the handler for masterbranch constraints and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlrMasterbranch(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP* origscip;
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_EVENTHDLR* eventhdlr;
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   SCIP_CALL( SCIPallocMemory(scip, &conshdlrdata) );
   conshdlrdata->stack = NULL;
   conshdlrdata->nstack = 0;
   conshdlrdata->maxstacksize = 25;

   /* include constraint handler */
   SCIP_CALL( SCIPincludeConshdlrBasic(scip, &conshdlr, CONSHDLR_NAME, CONSHDLR_DESC,
         CONSHDLR_ENFOPRIORITY, CONSHDLR_CHECKPRIORITY, CONSHDLR_EAGERFREQ, CONSHDLR_NEEDSCONS,
         consEnfolpMasterbranch, consEnfopsMasterbranch, consCheckMasterbranch, consLockMasterbranch,
         conshdlrdata) );
   assert(conshdlr != NULL);

   SCIP_CALL( SCIPsetConshdlrFree(scip, conshdlr, consFreeMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrInit(scip, conshdlr, consInitMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrExit(scip, conshdlr, consExitMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrInitsol(scip, conshdlr, consInitsolMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrDelete(scip, conshdlr, consDeleteMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrActive(scip, conshdlr, consActiveMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrDeactive(scip, conshdlr, consDeactiveMasterbranch) );
   SCIP_CALL( SCIPsetConshdlrProp(scip, conshdlr, consPropMasterbranch, CONSHDLR_PROPFREQ,
         CONSHDLR_DELAYPROP, CONSHDLR_PROPTIMING) );

   /* create event handler data */
   eventhdlrdata = NULL;

   /* include event handler into original SCIP */
   SCIP_CALL( SCIPincludeEventhdlrBasic(origscip, &eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC,
         eventExecOrigvarbound, eventhdlrdata) );
   assert(eventhdlr != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetEventhdlrInitsol(origscip, eventhdlr, eventInitsolOrigvarbound) );

   SCIP_CALL( SCIPaddBoolParam(origscip, "relaxing/gcg/enforceproper",
         "should propagated bound changes in the original be enforced in the master (only proper vars)?",
         &conshdlrdata->enforceproper, FALSE, TRUE, NULL, NULL) );

   return SCIP_OKAY;
}

/** creates and captures a masterbranch constraint */
SCIP_RETCODE GCGcreateConsMasterbranch(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of the constraint */
   SCIP_NODE*            node,               /**< node at which the constraint should be created */
   SCIP_CONS*            parentcons,         /**< parent constraint */
   SCIP_BRANCHRULE*      branchrule,         /**< pointer to the branching rule */
   GCG_BRANCHDATA*       branchdata,         /**< branching data */
   SCIP_CONS**           origbranchconss,    /**< original constraints enforcing the branching decision */
   int                   norigbranchconss,   /**< number of original constraints */
   int                   maxorigbranchconss  /**< capacity of origbranchconss */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(node != NULL || parentcons == NULL);
   if( node != NULL )
      assert((parentcons == NULL) == (SCIPnodeGetDepth(node) == 0));
   else
      assert(parentcons == NULL);

   /* find the masterbranch constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* create constraint data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &consdata) );

   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(consdata->name), name, strlen(name)+1) );

   consdata->npropvars = 0;
   consdata->needprop = TRUE;
   consdata->node = node;
   consdata->nactivated = 0;

   consdata->parentcons = parentcons;
   consdata->childconss = NULL;
   consdata->nchildconss = 0;
   consdata->maxchildconss = 0;
   consdata->probingtmpcons = NULL;
   consdata->origcons = NULL;

   consdata->branchdata = branchdata;
   consdata->branchrule = branchrule;

   consdata->localbndvars = NULL;
   consdata->localbndtypes = NULL;
   consdata->localnewbnds = NULL;
   consdata->localoldbnds = NULL;
   consdata->nlocalbndchgstreated = NULL;
   consdata->maxlocalbndchgstreated = 0;
   consdata->nlocalbndchgs = 0;
   consdata->maxlocalbndchgs = 0;
   consdata->nbranchingchgs = 0;

   consdata->copiedvars = NULL;
   consdata->copiedvarbndtypes = NULL;
   consdata->copiedvarbnds = NULL;
   consdata->ncopiedvarbnds = 0;
   consdata->maxcopiedvarbnds = 0;

   consdata->maxorigbranchconss = maxorigbranchconss;
   consdata->origbranchconss = origbranchconss;
   consdata->norigbranchconss = norigbranchconss;

   SCIPdebugMessage("Creating masterbranch constraint with parent %p.\n", (void*) parentcons);

   /* create constraint */
   SCIP_CALL( SCIPcreateCons(scip, cons, name, conshdlr, consdata, FALSE, FALSE, FALSE, FALSE, TRUE,
         TRUE, FALSE, FALSE, FALSE, TRUE) );

   /* add the new masterbranch constraint to the parent node's masterbranch constraint data
    * (unless the current node is the root node)
    */
   if( parentcons != NULL )
   {
      SCIP_CONSDATA* parentdata;

      parentdata = SCIPconsGetData(parentcons);
      assert(parentdata != NULL);

      if( SCIPinProbing(scip) || SCIPinProbing(GCGmasterGetOrigprob(scip)) )
      {
         parentdata->probingtmpcons = *cons;
      }
      else
      {
         int newmaxsize;
         ++parentdata->nchildconss;
         newmaxsize = SCIPcalcMemGrowSize(scip, parentdata->nchildconss);
         if( parentdata->nchildconss == 1 )
         {
            SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(parentdata->childconss), newmaxsize) );
            parentdata->childconss[0] = NULL;
         }
         else
         {
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(parentdata->childconss), parentdata->maxchildconss, newmaxsize) );
            parentdata->childconss[parentdata->nchildconss - 1] = NULL;
         }
         parentdata->maxchildconss = newmaxsize;

         assert(parentdata->childconss[parentdata->nchildconss - 1] == NULL);
         parentdata->childconss[parentdata->nchildconss - 1] = *cons;
      }
   }

   return SCIP_OKAY;
}

/** returns the name of the constraint */
char* GCGconsMasterbranchGetName(
   SCIP_CONS*            cons                /**< masterbranch constraint for which the data is requested */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->name;
}

/** returns the node in the B&B tree at which the given masterbranch constraint is sticking */
SCIP_NODE* GCGconsMasterbranchGetNode(
   SCIP_CONS*            cons                /**< constraint pointer */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->node;
}

/** returns the masterbranch constraint of the B&B father of the node at which the
  * given masterbranch constraint is sticking
  */
SCIP_CONS* GCGconsMasterbranchGetParentcons(
   SCIP_CONS*            cons                /**< constraint pointer */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->parentcons;
}

/** returns the number of masterbranch constraints of the children of the node at which the
  * given masterbranch constraint is sticking
  */
int GCGconsMasterbranchGetNChildconss(
   SCIP_CONS*            cons                /**< constraint pointer */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->nchildconss;
}

/** returns a masterbranch constraint of a child of the node at which the
  * given masterbranch constraint is sticking
  */
SCIP_CONS* GCGconsMasterbranchGetChildcons(
   SCIP_CONS*            cons,                /**< constraint pointer */
   int                   childnr              /**< index of the child node */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->childconss != NULL);

   assert(consdata->nchildconss > childnr);

   return consdata->childconss[childnr];
}

/** returns the origbranch constraint of the node in the original program corresponding to the node
  * at which the given masterbranch constraint is sticking
  */
SCIP_CONS* GCGconsMasterbranchGetOrigcons(
   SCIP_CONS*            cons                /**< constraint pointer */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->origcons;
}

/** sets the origbranch constraint of the node in the master program corresponding to the node
  * at which the given masterbranchbranch constraint is sticking
  */
void GCGconsMasterbranchSetOrigcons(
   SCIP_CONS*            cons,               /**< constraint pointer */
   SCIP_CONS*            origcons            /**< original branching constraint */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);
   assert(consdata->origcons == NULL || origcons == NULL);

   consdata->origcons = origcons;
}

/** returns the branching data for a given masterbranch constraint */
GCG_BRANCHDATA* GCGconsMasterbranchGetBranchdata(
   SCIP_CONS*            cons                /**< constraint pointer */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->branchdata;
}

/** returns the branching rule of the constraint */
SCIP_BRANCHRULE* GCGconsMasterbranchGetBranchrule(
   SCIP_CONS*            cons                /**< masterbranch constraint for which the data is requested */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->branchrule;
}

/** adds a bound change on an original variable that was directly copied to the master problem */
SCIP_RETCODE GCGconsMasterbranchAddCopiedVarBndchg(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< masterbranch constraint to which the bound change is added */
   SCIP_VAR*             var,                /**< variable on which the bound change was performed */
   GCG_BOUNDTYPE         boundtype,          /**< bound type of the bound change */
   SCIP_Real             newbound            /**< new bound of the variable after the bound change */
   )
{
   SCIP_CONSDATA* consdata;

   assert(scip != NULL);
   assert(GCGisMaster(scip));
   assert(cons != NULL);
   assert(var != NULL);

   /* get constraint data */
   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   /* realloc the arrays, if needed */
   if( consdata->ncopiedvarbnds >= consdata->maxcopiedvarbnds )
   {
      int newsize = SCIPcalcMemGrowSize(scip, consdata->ncopiedvarbnds+1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(consdata->copiedvars), consdata->maxcopiedvarbnds, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(consdata->copiedvarbndtypes), consdata->maxcopiedvarbnds, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(consdata->copiedvarbnds), consdata->maxcopiedvarbnds, newsize) );
      consdata->maxcopiedvarbnds = newsize;
   }

   SCIPdebugMessage("Bound change on copied original variable stored at masterbranch constraint: <%s>.\n", SCIPconsGetName(cons));

   /* store the new bound change */
   consdata->copiedvars[consdata->ncopiedvarbnds] = var;
   consdata->copiedvarbndtypes[consdata->ncopiedvarbnds] = boundtype;
   consdata->copiedvarbnds[consdata->ncopiedvarbnds] = newbound;
   consdata->ncopiedvarbnds++;

   /* mark the corresponding master node to be repropagated */
   SCIP_CALL( SCIPrepropagateNode(scip, consdata->node) );

   return SCIP_OKAY;
}

/** returns the constraints in the original problem that enforce the branching decision */
SCIP_CONS** GCGconsMasterbranchGetOrigbranchConss(
   SCIP_CONS*            cons                /**< masterbranch constraint for which the data is requested */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->origbranchconss;
}

/** returns the number of constraints in the original problem that enforce the branching decision */
int GCGconsMasterbranchGetNOrigbranchConss(
   SCIP_CONS*            cons                /**< masterbranch constraint for which the data is requested */
   )
{
   SCIP_CONSDATA* consdata;

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   return consdata->norigbranchconss;
}

/** releases the constraints in the original problem that enforce the branching decision
 *  and frees the array holding the constraints
 */
SCIP_RETCODE GCGconsMasterbranchReleaseOrigbranchConss(
   SCIP*                 masterscip,         /**< master problem SCIP instance */
   SCIP*                 origscip,           /**< original SCIP instance */
   SCIP_CONS*            cons                /**< masterbranch constraint for which the data is freed */
   )
{
   SCIP_CONSDATA* consdata;

   assert(GCGisMaster(masterscip));
   assert(GCGisOriginal(origscip));

   consdata = SCIPconsGetData(cons);
   assert(consdata != NULL);

   if( consdata->norigbranchconss > 0 )
   {
      int c;

      for( c = consdata->norigbranchconss - 1; c >= 0; --c )
      {
         SCIP_CALL( SCIPreleaseCons(origscip, &consdata->origbranchconss[c]) );
      }
      SCIPfreeBlockMemoryArray(masterscip, &consdata->origbranchconss, consdata->maxorigbranchconss);
      consdata->origbranchconss = NULL;
      consdata->norigbranchconss = 0;
      consdata->maxorigbranchconss = 0;
   }

   return SCIP_OKAY;
}

/** returns the masterbranch constraint of the current node */
SCIP_CONS* GCGconsMasterbranchGetActiveCons(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLR*     conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   /* get constraint handler */
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   /* get constraint handler data */
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL);

   if( conshdlrdata->nstack == 0 )
      return NULL;

   assert(conshdlrdata->stack[conshdlrdata->nstack-1] != NULL);
   return conshdlrdata->stack[conshdlrdata->nstack-1];
}

/** returns the stack and the number of elements on it */
void GCGconsMasterbranchGetStack(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS***          stack,              /**< return value: pointer to the stack */
   int*                  nstackelements      /**< return value: pointer to int, for number of elements on the stack */
   )
{
   SCIP_CONSHDLR*     conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL);

   *stack = conshdlrdata->stack;
   *nstackelements = conshdlrdata->nstack;
}

/** returns the number of elements on the stack */
int GCGconsMasterbranchGetNStackelements(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLR*     conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   assert(conshdlrdata->stack != NULL);

   return conshdlrdata->nstack;
}

/** adds initial constraint to root node */
SCIP_RETCODE GCGconsMasterbranchAddRootCons(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;
   SCIP_CONS* cons;

   assert(scip != NULL);
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);
   cons = conshdlrdata->stack[0];
   conshdlrdata->stack[0] = NULL;
   assert(conshdlrdata->nstack == 1);
   conshdlrdata->nstack = 0;

   SCIP_CALL( SCIPaddConsNode(scip, SCIPgetRootNode(scip), cons, SCIPgetRootNode(scip)) );
   SCIP_CALL( SCIPreleaseCons(scip, &cons) );

   return SCIP_OKAY;
}

/** check whether the node was generated by generic branching */
SCIP_Bool GCGcurrentNodeIsGeneric(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONS* mastercons;
   SCIP_BRANCHRULE* branchrule;

   assert(scip != NULL);
   assert(GCGisMaster(scip));

   mastercons = GCGconsMasterbranchGetActiveCons(scip);

   /* @todo: Why might mastercons be NULL? */
   if( mastercons == NULL || SCIPnodeGetDepth(GCGconsMasterbranchGetNode(mastercons)) == 0 )
      return FALSE;

   branchrule = GCGconsMasterbranchGetBranchrule(mastercons);

   if( branchrule == NULL || strcmp(SCIPbranchruleGetName(branchrule), "generic") != 0 )
      return FALSE;

   return TRUE;
}

/** checks the consistency of the masterbranch constraints in the problem */
void GCGconsMasterbranchCheckConsistency(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLR*     conshdlr;
   int nconss;
#ifndef NDEBUG
   int i;
   SCIP_CONS** conss;
#endif

   assert(scip != NULL);
   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   nconss = SCIPconshdlrGetNConss(conshdlr);
#ifndef NDEBUG
   conss = SCIPconshdlrGetConss(conshdlr);

   for( i = 0; i < nconss; i++ )
   {
      SCIP_CONSDATA* consdata;
#ifdef SCIP_DEBUG
      SCIP_CONS* parent_origcons;
      SCIP_CONS* origcons_parent;
#endif

      consdata = SCIPconsGetData(conss[i]);
      assert(consdata != NULL);

      assert(consdata->probingtmpcons == NULL || SCIPinProbing(scip));
      assert(consdata->probingtmpcons == NULL || SCIPconsGetData(consdata->probingtmpcons)->parentcons == conss[i]);
      assert(consdata->origcons == NULL || GCGconsOrigbranchGetMastercons(consdata->origcons) == conss[i]);

#ifdef SCIP_DEBUG
      if( consdata->parentcons != NULL )
         parent_origcons = SCIPconsGetData(consdata->parentcons)->origcons;
      else
         parent_origcons = NULL;

      if( consdata->origcons != NULL )
         origcons_parent = GCGconsOrigbranchGetParentcons(consdata->origcons);
      else
         origcons_parent = NULL;

      SCIPdebugMessage("cons: %s (node %p), origcons: %s, parent %s: %s => %s\n",
         SCIPconsGetName(conss[i]),
         (void*) consdata->node,
         consdata->origcons == NULL? "NULL" : SCIPconsGetName(consdata->origcons),
         consdata->parentcons == NULL? "NULL" : SCIPconsGetName(consdata->parentcons),
         parent_origcons == NULL? "NULL" :  SCIPconsGetName(parent_origcons),
         origcons_parent == NULL? "NULL" : SCIPconsGetName(origcons_parent) );
#endif
   }
#endif

   SCIPdebugMessage("checked consistency of %d masterbranch constraints, all ok!\n", nconss);
}
