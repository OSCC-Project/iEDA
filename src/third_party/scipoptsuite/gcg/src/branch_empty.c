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

/**@file   branch_empty.c
 * @brief  branching rule for the original problem while real branching is applied in the master
 * @author Marcel Schmickerath
 * @author Martin Bergner
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/*#define SCIP_DEBUG*/
#include <assert.h>
#include <string.h>

#include "branch_empty.h"
#include "relax_gcg.h"
#include "gcg.h"
#include "branch_orig.h"
#include "cons_masterbranch.h"
#include "cons_origbranch.h"
#include "scip/branch_allfullstrong.h"
#include "scip/branch_fullstrong.h"
#include "scip/branch_inference.h"
#include "scip/branch_mostinf.h"
#include "scip/branch_leastinf.h"
#include "scip/branch_pscost.h"
#include "scip/branch_random.h"
#include "scip/branch_relpscost.h"
#include "scip/cons_varbound.h"
#include "type_branchgcg.h"

#define BRANCHRULE_NAME          "empty"
#define BRANCHRULE_DESC          "branching rule for the original problem while real branching is applied in the master"
#define BRANCHRULE_PRIORITY      1000000
#define BRANCHRULE_MAXDEPTH      -1
#define BRANCHRULE_MAXBOUNDDIST  1.0


/*
 * Callback methods for enforcing branching constraints
 */

/** copy default SCIP branching rules in order to solve restrictions of the original problem as a subSCIP without
 *  Dantzig-Wolfe decomposition
 */
static
SCIP_RETCODE includeSCIPBranchingRules(
   SCIP*                 scip
)
{
   assert(scip != NULL);

   SCIP_CALL( SCIPincludeBranchruleAllfullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleFullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleInference(scip) );
   SCIP_CALL( SCIPincludeBranchruleMostinf(scip) );
   SCIP_CALL( SCIPincludeBranchruleLeastinf(scip) );
   SCIP_CALL( SCIPincludeBranchrulePscost(scip) );
   SCIP_CALL( SCIPincludeBranchruleRandom(scip) );
   SCIP_CALL( SCIPincludeBranchruleRelpscost(scip) );

   return SCIP_OKAY;
}

/** for a new branch-and-bound node on the master problem
 *  add an original branching constraint that holds the branching decision to the corresponding node in the original problem
 */
static
SCIP_RETCODE createOrigbranchConstraint(
   SCIP*                 scip,
   SCIP_NODE*            childnode,
   SCIP_CONS*            masterbranchchildcons
)
{
   char* consname;
   SCIP_BRANCHRULE* branchrule;
   GCG_BRANCHDATA* branchdata;
   SCIP_CONS* origcons;
   SCIP_CONS** origbranchconss;
   int norigbranchconss;

   int i;

   assert(scip != NULL);
   assert(masterbranchchildcons != NULL);

   /* get name and branching information from the corresponding masterbranch constraint */
   consname = GCGconsMasterbranchGetName(masterbranchchildcons);
   branchrule = GCGconsMasterbranchGetBranchrule(masterbranchchildcons);
   branchdata = GCGconsMasterbranchGetBranchdata(masterbranchchildcons);

   /* create an origbranch constraint and add it to the node */
   SCIPdebugMessage("Create original branching constraint %s\n", consname);
   SCIP_CALL( GCGcreateConsOrigbranch(scip, &origcons, consname, childnode,
            GCGconsOrigbranchGetActiveCons(scip), branchrule, branchdata) );
   if( branchdata == NULL )
   {
      SCIPdebugMessage("origbranch with no branchdata created\n");
   }
   SCIP_CALL( SCIPaddConsNode(scip, childnode, origcons, NULL) );

   /* add those constraints to the node that enforce the branching decision in the original problem */
   origbranchconss = GCGconsMasterbranchGetOrigbranchConss(masterbranchchildcons);
   norigbranchconss = GCGconsMasterbranchGetNOrigbranchConss(masterbranchchildcons);
   for( i = 0; i < norigbranchconss; ++i )
   {
      SCIP_CALL( SCIPaddConsNode(scip, childnode, origbranchconss[i], NULL) );
   }

   /* notify the original and master branching constraint about each other */
   GCGconsOrigbranchSetMastercons(origcons, masterbranchchildcons);
   GCGconsMasterbranchSetOrigcons(masterbranchchildcons, origcons);

   SCIP_CALL( SCIPreleaseCons(scip, &origcons) );

   /* release array of original branching constraints */
   SCIP_CALL( GCGconsMasterbranchReleaseOrigbranchConss(GCGgetMasterprob(scip), scip, masterbranchchildcons) );

   return SCIP_OKAY;
}

/* apply a branching decision on the original variables to the corresponding node */
static
SCIP_RETCODE applyOriginalBranching(
   SCIP*                 scip,
   SCIP_NODE*            childnode,
   SCIP_CONS*            masterbranchchildcons
   )
{
   GCG_BRANCHDATA* branchdata;
   SCIP_VAR* boundvar;
   GCG_BOUNDTYPE boundtype;
   SCIP_Real newbound;

   /* get branching decision */
   branchdata = GCGconsMasterbranchGetBranchdata(masterbranchchildcons);
   assert(branchdata != NULL);
   boundvar = GCGbranchOrigGetOrigvar(branchdata);
   boundtype = GCGbranchOrigGetBoundtype(branchdata);
   newbound = GCGbranchOrigGetNewbound(branchdata);

   assert(boundvar != NULL);
   assert(boundtype == GCG_BOUNDTYPE_LOWER || boundtype == GCG_BOUNDTYPE_UPPER || boundtype == GCG_BOUNDTYPE_FIXED);

   if( boundtype == GCG_BOUNDTYPE_LOWER || boundtype == GCG_BOUNDTYPE_FIXED )
   {
      SCIP_CALL( SCIPchgVarLbNode(scip, childnode, boundvar, newbound) );
   }

   if( boundtype == GCG_BOUNDTYPE_UPPER || boundtype == GCG_BOUNDTYPE_FIXED )
   {
      SCIP_CALL( SCIPchgVarUbNode(scip, childnode, boundvar, newbound) );
   }

   if( GCGvarGetBlock(boundvar) == -1 )
   {
      SCIP_CALL( GCGconsMasterbranchAddCopiedVarBndchg(GCGgetMasterprob(scip), masterbranchchildcons,
         boundvar, boundtype, newbound) );
   }

   return SCIP_OKAY;
}

/** creates branch-and-bound nodes in the original problem corresponding to those in the master problem */
static
SCIP_RETCODE createBranchNodesInOrigprob(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< result pointer */
)
{
   SCIP* masterscip;
   SCIP_BRANCHRULE* branchrule;
   SCIP_CONS* masterbranchcons;
   int nchildnodes;
   SCIP_Bool enforcebycons;

   int i;

   assert(scip != NULL);
   assert(result != NULL);

   *result = SCIP_DIDNOTRUN;

   /* get master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   /* get masterbranch constraint at the current node */
   masterbranchcons = GCGconsMasterbranchGetActiveCons(masterscip);

   /* @todo: Why should this happen? */
   if( masterbranchcons == NULL )
      return SCIP_OKAY;

   /* get the children of the current node */
   nchildnodes = GCGconsMasterbranchGetNChildconss(masterbranchcons);

   /* check if the focus node of the master problem has children */
   if( nchildnodes <= 0 && SCIPgetStage(masterscip) != SCIP_STAGE_SOLVED && SCIPgetNChildren(masterscip) >= 1 )
   {
      SCIP_NODE* child;

      SCIPdebugMessage("create dummy child in origprob, because there is also a child in the master\n");

      /* create dummy child */
      SCIP_CALL( SCIPcreateChild(scip, &child, 0.0, SCIPgetLocalTransEstimate(scip)) );

      *result = SCIP_BRANCHED;
      return SCIP_OKAY;
   }

   if( nchildnodes <= 0 )
   {
      SCIPdebugMessage("node cut off, since there is no successor node\n");

      *result = SCIP_CUTOFF;
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPgetBoolParam(scip, "branching/orig/enforcebycons", &enforcebycons) );

   /* for each child, create a corresponding node in the original problem as well as an origbranch constraint */
   for( i = 0; i < nchildnodes; ++i )
   {
      SCIP_NODE* childnode;
      SCIP_CONS* masterbranchchildcons = GCGconsMasterbranchGetChildcons(masterbranchcons, i);
      assert(masterbranchchildcons != NULL);

      /* create a child node and an origbranch constraint holding the branching decision */
      SCIP_CALL( SCIPcreateChild(scip, &childnode, 0.0, SCIPgetLocalTransEstimate(scip)) );
      SCIP_CALL( createOrigbranchConstraint(scip, childnode, masterbranchchildcons) );

      /* get branching rule */
      branchrule = GCGconsMasterbranchGetBranchrule(masterbranchchildcons);

      /* If a branching decision on an original variable was made, apply it */
      if( !enforcebycons && branchrule != NULL && strcmp(SCIPbranchruleGetName(branchrule), "orig") == 0 )
      {
         SCIP_CALL( applyOriginalBranching(scip, childnode, masterbranchchildcons) );
      }

      /* @fixme: this should actually be an assertion */
      if( SCIPnodeGetNumber(GCGconsOrigbranchGetNode(GCGconsOrigbranchGetActiveCons(scip))) != SCIPnodeGetNumber(GCGconsMasterbranchGetNode(GCGconsMasterbranchGetActiveCons(GCGgetMasterprob(scip)))) )
      {
   #ifdef SCIP_DEBUG
         SCIPwarningMessage(scip, "norignodes = %lld; nmasternodes = %lld\n",
            SCIPnodeGetNumber(GCGconsOrigbranchGetNode(GCGconsOrigbranchGetActiveCons(scip))),
            SCIPnodeGetNumber(GCGconsMasterbranchGetNode(GCGconsMasterbranchGetActiveCons(GCGgetMasterprob(scip)))));
   #endif
      }
   }

   *result = SCIP_BRANCHED;

   assert(nchildnodes > 0);

   return SCIP_OKAY;
}


/** copy method for empty branching rule */
static
SCIP_DECL_BRANCHCOPY(branchCopyEmpty)
{
   assert(branchrule != NULL);
   assert(scip != NULL);

   /* SubSCIPs are solved with SCIP rather than GCG;
    * therefore, only the default SCIP branching rules are included into the subSCIP.
    */
   SCIP_CALL( includeSCIPBranchingRules(scip) );

   return SCIP_OKAY;
}

/** destructor of branching rule to free user data (called when SCIP is exiting) */
#define branchFreeEmpty NULL

/** initialization method of branching rule (called after problem was transformed) */
#define branchInitEmpty NULL

/** deinitialization method of branching rule (called before transformed problem is freed) */
#define branchExitEmpty NULL

/** solving process initialization method of branching rule (called when branch and bound process is about to begin) */
#define branchInitsolEmpty NULL

/** solving process deinitialization method of branching rule (called before branch and bound process data is freed) */
#define branchExitsolEmpty NULL

/** branching execution method for fractional LP solutions */
static
SCIP_DECL_BRANCHEXECLP(branchExeclpEmpty)
{  /*lint --e{715}*/
   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIP_CALL( createBranchNodesInOrigprob(scip, result) );

   return SCIP_OKAY;
}

/** branching execution method relaxation solutions */
static
SCIP_DECL_BRANCHEXECEXT(branchExecextEmpty)
{  /*lint --e{715}*/
   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIP_CALL( createBranchNodesInOrigprob(scip, result) );

   return SCIP_OKAY;
}

/** branching execution method for not completely fixed pseudo solutions */
static
SCIP_DECL_BRANCHEXECPS(branchExecpsEmpty)
{  /*lint --e{715}*/
   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIP_CALL( createBranchNodesInOrigprob(scip, result) );

   return SCIP_OKAY;
}


/*
 * branching specific interface methods
 */

/** creates the empty branching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleEmpty(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   SCIP_BRANCHRULEDATA* branchruledata;

   /* create inference branching rule data */
   branchruledata = NULL;

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchrule(scip, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY,
      BRANCHRULE_MAXDEPTH, BRANCHRULE_MAXBOUNDDIST,
      branchCopyEmpty, branchFreeEmpty, branchInitEmpty, branchExitEmpty, branchInitsolEmpty,
      branchExitsolEmpty, branchExeclpEmpty, branchExecextEmpty, branchExecpsEmpty,
      branchruledata) );

   return SCIP_OKAY;
}
