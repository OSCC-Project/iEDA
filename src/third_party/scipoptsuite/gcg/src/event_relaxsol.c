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

/**@file   event_relaxsol.c
 * @brief  eventhandler to update the relaxation solution in the original problem when the master LP has been solved
 * @author Christian Puchert
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string.h>
#include "event_relaxsol.h"
#include "relax_gcg.h"
#include "pricer_gcg.h"
#include "gcg.h"
#include "event_mastersol.h"

#define EVENTHDLR_NAME         "relaxsol"
#define EVENTHDLR_DESC         "eventhandler to update the relaxation solution in the original problem when the master LP has been solved"


/*
 * Data structures
 */

/** event handler data */
struct SCIP_EventhdlrData
{
   SCIP_Bool             triggered;          /**< flag to indicate whether event has been triggered */
};


/*
 * Callback methods of event handler
 */

/** destructor of event handler to free user data (called when SCIP is exiting) */
static
SCIP_DECL_EVENTFREE(eventFreeRelaxsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   SCIPfreeMemory(scip, &eventhdlrdata);
   SCIPeventhdlrSetData(eventhdlr, NULL);

   return SCIP_OKAY;
}

/** initialization method of event handler (called after problem was transformed) */
static
SCIP_DECL_EVENTINIT(eventInitRelaxsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* notify SCIP that your event handler wants to react on the event type lp solved and solution found */
   SCIP_CALL( SCIPcatchEvent(scip, SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, NULL) );
   eventhdlrdata->triggered = FALSE;

   return SCIP_OKAY;
}

/** deinitialization method of event handler (called before transformed problem is freed) */
static
SCIP_DECL_EVENTEXIT(eventExitRelaxsol)
{  /*lint --e{715}*/

   /* notify SCIP that your event handler wants to drop the event type lp solved and solution found */
   SCIP_CALL( SCIPdropEvent(scip, SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, -1) );

   return SCIP_OKAY;
}

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecRelaxsol)
{  /*lint --e{715}*/
   SCIP* origprob;
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* Only transfer the master solution if it is an LP solution or if it is a feasible solution that
    * comes from a master heuristic; otherwise it is assumed to already come from the original problem
    */
   if( (SCIPeventGetType(event) & SCIP_EVENTTYPE_SOLFOUND) && SCIPsolGetHeur(SCIPeventGetSol(event)) == NULL
      && GCGeventhdlrMastersolIsTriggered(origprob) )
      return SCIP_OKAY;

   eventhdlrdata->triggered = TRUE;

   if( SCIPeventGetType(event) & SCIP_EVENTTYPE_LPSOLVED )
   {
      SCIPdebugMessage("Transferring master LP solution to the original problem\n");
      SCIP_CALL( GCGrelaxUpdateCurrentSol(origprob) );
   }
   else if( SCIPeventGetType(event) & SCIP_EVENTTYPE_SOLFOUND )
   {
      SCIP_SOL* sol = SCIPeventGetSol(event);
      SCIP_SOL* origsol;
      SCIP_Bool stored;

      SCIPdebugMessage("Master feasible solution found by <%s> -- transferring to original problem\n",
         SCIPsolGetHeur(sol) == NULL ? "relaxation" : SCIPheurGetName(SCIPsolGetHeur(sol)));

      /* transform the master solution to the original variable space */
      SCIP_CALL( GCGtransformMastersolToOrigsol(origprob, sol, &origsol) );

      SCIP_CALL( SCIPtrySolFree(origprob, &origsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
      SCIPdebugMessage("  ->%s stored\n", stored ? "" : " not");
   }

   eventhdlrdata->triggered = FALSE;

   return SCIP_OKAY;
}

/** creates event handler for relaxsol event */
SCIP_RETCODE SCIPincludeEventHdlrRelaxsol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_EVENTHDLR* eventhdlr;
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   eventhdlr = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &eventhdlrdata) );
   assert(eventhdlrdata != NULL);

   /* include event handler into GCG */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC,
         eventExecRelaxsol, eventhdlrdata) );
   assert(eventhdlr != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetEventhdlrFree(scip, eventhdlr, eventFreeRelaxsol) );
   SCIP_CALL( SCIPsetEventhdlrInit(scip, eventhdlr, eventInitRelaxsol) );
   SCIP_CALL( SCIPsetEventhdlrExit(scip, eventhdlr, eventExitRelaxsol) );

   return SCIP_OKAY;
}

/** return whether event has been triggered */
SCIP_Bool GCGeventhdlrRelaxsolIsTriggered(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 masterprob          /**< the SCIP data structure for the master problem */
   )
{
   SCIP_EVENTHDLR* eventhdlr;
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(masterprob != NULL);

   /* the relaxation solution event handler is not included if BENDERS or ORIGINAL mode is used. As such, it will
    * never be triggered. In this case, it will always return FALSE.
    */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
      return FALSE;

   eventhdlr = SCIPfindEventhdlr(masterprob, EVENTHDLR_NAME);
   assert(eventhdlr != NULL);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   return eventhdlrdata->triggered;
}
