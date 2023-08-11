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

/**@file   cons_integralorig.c
 * @ingroup CONSHDLRS
 * @brief  constraint handler for enforcing integrality of the transferred master solution in the original problem
 * @author Gerald Gamrath
 *         Marcel Schmickerath
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */
#include <assert.h>
#include <string.h>

#include "cons_integralorig.h"
#include "pricer_gcg.h"
#include "cons_masterbranch.h"
#include "pub_gcgvar.h"
#include "scip/struct_branch.h"
#include "relax_gcg.h"
#include "gcg.h"

#include "branch_orig.h"

#define CONSHDLR_NAME          "integralorig"
#define CONSHDLR_DESC          "integrality constraint"
#define CONSHDLR_ENFOPRIORITY      1000 /**< priority of the constraint handler for constraint enforcing */
#define CONSHDLR_CHECKPRIORITY     1000 /**< priority of the constraint handler for checking feasibility */
#define CONSHDLR_EAGERFREQ           -1 /**< frequency for using all instead of only the useful constraints in separation,
                                              *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
#define CONSHDLR_NEEDSCONS        FALSE /**< should the constraint handler be skipped, if no constraints are available? */


/** constraint handler data */
struct SCIP_ConshdlrData
{
   SCIP_BRANCHRULE**           branchrules;              /**< stack for storing active branchrules */
   int                         nbranchrules;             /**< number of active branchrules */
};

/** insert branchrule in constraint handler data */
SCIP_RETCODE GCGconsIntegralorigAddBranchrule(
   SCIP*                 scip,
   SCIP_BRANCHRULE*      branchrule
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
   assert(conshdlr != NULL);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   assert(conshdlrdata->nbranchrules >= 0);
   if( conshdlrdata->nbranchrules == 0 )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &(conshdlrdata->branchrules), 1) ); /*lint !e506*/
      conshdlrdata->nbranchrules = 1;
   }
   else
   {
      SCIP_CALL( SCIPreallocMemoryArray(scip, &(conshdlrdata->branchrules), (size_t)conshdlrdata->nbranchrules+1) );
      ++conshdlrdata->nbranchrules;
   }

   assert(conshdlrdata->nbranchrules > 0);

   conshdlrdata->branchrules[conshdlrdata->nbranchrules-1] = branchrule;

   return SCIP_OKAY;
}

/** sort branchrules with respect to priority */
static
void sortBranchrules(
   SCIP_BRANCHRULE**      branchrules,
   int                    nbranchrules
   )
{
   SCIP_BRANCHRULE* tmp;
   int pos;
   int i;

   tmp = NULL;

   assert(nbranchrules >= 0);
   assert(branchrules != NULL);

   for( pos=0; pos<nbranchrules; ++pos )
   {
      int maxi = pos;
      for( i=pos+1; i<nbranchrules; ++i )
      {
         if( branchrules[i]->priority > branchrules[maxi]->priority )
         {
            maxi = i;
         }
      }
      if( maxi != pos )
      {
         tmp = branchrules[pos];
         branchrules[pos] = branchrules[maxi];
         branchrules[maxi] = tmp;
      }
   }
}

/*
 * Callback methods
 */

/** constraint enforcing method of constraint handler for LP solutions */
static
SCIP_DECL_CONSENFOLP(consEnfolpIntegralOrig)
{  /*lint --e{715}*/
   SCIP* origprob;
   SCIP_Bool discretization;
   int i;
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(scip != NULL);
   assert(conss == NULL);
   assert(nconss == 0);
   assert(result != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(scip != NULL);

   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPdebugMessage("LP solution enforcing method of integralorig constraint\n");

   *result = SCIP_FEASIBLE;

   /* if we use the discretization without continuous variables, we do not have to check for integrality of the solution in the
    * original variable space, we obtain it by enforcing integrality of the master solution*/
   SCIP_CALL( SCIPgetBoolParam(origprob, "relaxing/gcg/discretization", &discretization) );
   if( discretization && SCIPgetNContVars(origprob) == 0 )
   {
      return SCIP_OKAY;
   }

   /* if the transferred master solution is feasible, the current node is solved to optimality and can be pruned */
   if( GCGrelaxIsOrigSolFeasible(origprob) )
   {
      SCIPdebugMessage("Orig sol is feasible\n");
      *result = SCIP_FEASIBLE;
      return SCIP_OKAY;
   }

   sortBranchrules(conshdlrdata->branchrules, conshdlrdata->nbranchrules);

   i = 0;

   while( *result != SCIP_BRANCHED && 
          *result != SCIP_REDUCEDDOM && 
          i < conshdlrdata->nbranchrules )
   {
      assert(conshdlrdata->branchrules[i] != NULL);

      if( conshdlrdata->branchrules[i]->branchexeclp == NULL )
      {
         ++i;
         continue;
      }

      SCIPdebugMessage("Call exec lp method of %s\n", SCIPbranchruleGetName(conshdlrdata->branchrules[i]));
      /** todo handle bool allowaddcons; here default TRUE */
      SCIP_CALL( conshdlrdata->branchrules[i]->branchexeclp(scip, conshdlrdata->branchrules[i], TRUE, result) );
      ++i;
   }

   return SCIP_OKAY;

}


/** constraint enforcing method of constraint handler for pseudo solutions */
static
SCIP_DECL_CONSENFOPS(consEnfopsIntegralOrig)
{  /*lint --e{715}*/
   SCIP* origprob;
   SCIP_Bool discretization;
   SCIP_CONSHDLRDATA* conshdlrdata;
   int i;

   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(scip != NULL);
   assert(conss == NULL);
   assert(nconss == 0);
   assert(result != NULL);

   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);
   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   *result = SCIP_FEASIBLE;
   i = 0;

   /* if we use the discretization without continuous variables, we do not have to check for integrality of the solution in the
    * original variable space, we obtain it by enforcing integrality of the master solution*/
   SCIP_CALL( SCIPgetBoolParam(origprob, "relaxing/gcg/discretization", &discretization) );
   if( discretization && SCIPgetNContVars(origprob) == 0 )
   {
      return SCIP_OKAY;
   }

   assert(SCIPgetNPseudoBranchCands(origprob) > 0);

   sortBranchrules(conshdlrdata->branchrules, conshdlrdata->nbranchrules);

   while( *result != SCIP_BRANCHED && i < conshdlrdata->nbranchrules )
   {
      assert(conshdlrdata->branchrules[i] != NULL);

      if( conshdlrdata->branchrules[i]->branchexecps == NULL )
      {
         ++i;
         continue;
      }
      /** todo handle bool allowaddcons; here default TRUE */
      SCIP_CALL( conshdlrdata->branchrules[i]->branchexecps(scip, conshdlrdata->branchrules[i], TRUE, result) );
      ++i;
   }

   return SCIP_OKAY;
}


/** feasibility check method of constraint handler for integral solutions */
static
SCIP_DECL_CONSCHECK(consCheckIntegralOrig)
{  /*lint --e{715}*/
   SCIP* origprob;
   SCIP_SOL* origsol;
   SCIP_VAR** origvars;
   int norigvars;
   SCIP_Real solval;
   SCIP_Bool discretization;
   int v;

   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);
   assert(scip != NULL);

   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   SCIPdebugMessage("Check method of integralorig constraint\n");

   *result = SCIP_FEASIBLE;

   /* if we use the discretization without continuous variables, we do not have to check for integrality of the solution in the
    * original variable space, we obtain it by enforcing integrality of the master solution*/
   SCIP_CALL( SCIPgetBoolParam(origprob, "relaxing/gcg/discretization", &discretization) );
   if( discretization && SCIPgetNContVars(origprob) == 0 )
   {
      return SCIP_OKAY;
   }

   /* get corresponding origsol in order to check integrality */
   SCIP_CALL( GCGtransformMastersolToOrigsol(origprob, sol, &origsol) );

   origvars = SCIPgetVars(origprob);
   norigvars = SCIPgetNVars(origprob);

   /* check for each integral original variable whether it has a fractional value */
   for( v = 0; v < norigvars && *result == SCIP_FEASIBLE; v++ )
   {
      if( SCIPvarGetType(origvars[v]) == SCIP_VARTYPE_CONTINUOUS )
         continue;

      solval = 0.0;
      assert(GCGvarIsOriginal(origvars[v]));

      solval = SCIPgetSolVal(origprob, origsol, origvars[v]);

      if( !SCIPisFeasIntegral(origprob, solval) )
      {
         *result = SCIP_INFEASIBLE;

         if( printreason )
         {
            SCIPinfoMessage(scip, NULL, "violation: integrality condition of variable <%s> = %.15g\n",
               SCIPvarGetName(origvars[v]), solval);
         }
      }
   }

   SCIPfreeSol(origprob, &origsol);

   return SCIP_OKAY;
}


/** variable rounding lock method of constraint handler */
static
SCIP_DECL_CONSLOCK(consLockIntegralOrig)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}

/** destructor of constraint handler to free constraint handler data (called when SCIP is exiting) */
static
SCIP_DECL_CONSFREE(consFreeIntegralOrig)
{
   SCIP_CONSHDLRDATA* conshdlrdata;

   assert(scip != NULL);
   assert(conshdlr != NULL);
   assert(strcmp(SCIPconshdlrGetName(conshdlr), CONSHDLR_NAME) == 0);

   conshdlrdata = SCIPconshdlrGetData(conshdlr);
   assert(conshdlrdata != NULL);

   SCIPdebugMessage("freeing integralorig constraint handler\n");

   if( conshdlrdata->nbranchrules > 0 )
   {
      assert(conshdlrdata->branchrules != NULL);
      SCIPfreeMemoryArray(scip, &(conshdlrdata->branchrules) );
      conshdlrdata->branchrules = NULL;
      conshdlrdata->nbranchrules = 0;
   }

   /* free constraint handler storage */
   assert(conshdlrdata->branchrules == NULL);
   SCIPfreeMemory(scip, &conshdlrdata);

   return SCIP_OKAY;
}

/*
 * constraint specific interface methods
 */

/** creates the handler for integrality constraint and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlrIntegralOrig(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CONSHDLR* conshdlr;
   SCIP_CONSHDLRDATA* conshdlrdata;

   /* create integral constraint handler data */
   conshdlrdata = NULL;
   SCIP_CALL( SCIPallocMemory(scip, &conshdlrdata) );
   conshdlrdata->branchrules = NULL;
   conshdlrdata->nbranchrules = 0;

   /* include constraint handler */
   SCIP_CALL( SCIPincludeConshdlrBasic(scip, &conshdlr, CONSHDLR_NAME, CONSHDLR_DESC,
         CONSHDLR_ENFOPRIORITY, CONSHDLR_CHECKPRIORITY, CONSHDLR_EAGERFREQ, CONSHDLR_NEEDSCONS,
         consEnfolpIntegralOrig, consEnfopsIntegralOrig, consCheckIntegralOrig, consLockIntegralOrig,
         conshdlrdata) );
   assert(conshdlr != NULL);

   SCIP_CALL( SCIPsetConshdlrFree(scip, conshdlr, consFreeIntegralOrig) );

   return SCIP_OKAY;
}
