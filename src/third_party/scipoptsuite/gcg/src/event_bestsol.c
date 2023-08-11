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

/**@file   event_bestsol.c
 * @brief  eventhdlr to record the best primal bound for each heuristic
 * @author Christian Puchert
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string.h>
#include "event_bestsol.h"

#ifdef SCIP_STATISTIC
#define EVENTHDLR_NAME         "bestsol"
#define EVENTHDLR_DESC         "event handler to record the best primal bound for each heuristic"

/*
 * Data structures
 */

/** event handler data */
struct SCIP_EventhdlrData
{
   SCIP_HEUR**           heurs;              /**< heuristics known to this event handler               */
   SCIP_Real*            bestprimalbd;       /**< array to store best primal bounds for each heuristic */
};

/*
 * Callback methods of event handler
 */

/** destructor of event handler to free user data (called when SCIP is exiting) */
static
SCIP_DECL_EVENTFREE(eventFreeBestsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   SCIPfreeMemory(scip, &eventhdlrdata);

   return SCIP_OKAY;
}

/** initialization method of event handler (called after problem was transformed) */
static
SCIP_DECL_EVENTINIT(eventInitBestsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   int nheurs;
   int i;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   nheurs = SCIPgetNHeurs(scip);

   /* allocate memory */
   SCIP_CALL( SCIPduplicateMemoryArray(scip, &eventhdlrdata->heurs, SCIPgetHeurs(scip), nheurs) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &eventhdlrdata->bestprimalbd, nheurs) );

   for( i = 0; i < nheurs; ++i )
      eventhdlrdata->bestprimalbd[i] = SCIPinfinity(scip);

   /* notify SCIP that your event handler wants to react on the event types best solution found and node solved */
   SCIP_CALL( SCIPcatchEvent(scip, SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, NULL) );

   return SCIP_OKAY;
}

/** deinitialization method of event handler (called before transformed problem is freed) */
static
SCIP_DECL_EVENTEXIT(eventExitBestsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* free memory */
   SCIPfreeMemoryArray(scip, &eventhdlrdata->heurs);
   SCIPfreeMemoryArray(scip, &eventhdlrdata->bestprimalbd);

   /* notify SCIP that your event handler wants to drop the event types best solution found and node solved */
   SCIP_CALL( SCIPdropEvent(scip, SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, -1) );

   return SCIP_OKAY;
}

/** solving process deinitialization method of event handler (called before branch and bound process data is freed) */
static
SCIP_DECL_EVENTEXITSOL(eventExitsolBestsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   const char* probname;
   int nheurs;
   int i;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   nheurs = SCIPgetNHeurs(scip);
   probname = SCIPgetProbName(scip);

   /* output statistics */
   for( i = 0; i < nheurs; ++i )
   {
      SCIPstatisticPrintf("Heuristic statistics (%s) -- %s : bestprimalbound = %13.6e\n",
         strncmp(probname, "master", 6) == 0 ? "master" : "original",
         SCIPheurGetName(eventhdlrdata->heurs[i]), eventhdlrdata->bestprimalbd[i]);
   }

   return SCIP_OKAY;
}

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecBestsol)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   int nheurs;
   SCIP_SOL* sol;
   SCIP_HEUR* solheur;
   SCIP_Real obj;
   int i;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   nheurs = SCIPgetNHeurs(scip);

   /* get new primal solution */
   sol = SCIPeventGetSol(event);
   assert(sol != NULL);

   /* get the heuristic that found the solution */
   solheur = SCIPgetSolHeur(scip, sol);

   /* get the objective value */
   obj = SCIPgetSolTransObj(scip, sol);

   /* if the solution was found by a relaxation, there is nothing to do */
   if( solheur == NULL )
      return SCIP_OKAY;

   /* search the heuristic that found the solution */
   for( i = 0; i < nheurs && eventhdlrdata->heurs[i] != solheur; ++i ) ;

   /* if the heuristic was not found in the problem, then the solution comes
    * from another problem; in that case, no statistics are collected here
    */
   if( i == nheurs )
      return SCIP_OKAY;

   /* update the best objective value for that heuristic */
   if( obj < eventhdlrdata->bestprimalbd[i] )
      eventhdlrdata->bestprimalbd[i] = obj;

   return SCIP_OKAY;
}
#endif

/** creates event handler for bestsol event */
SCIP_RETCODE SCIPincludeEventHdlrBestsol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
#ifdef SCIP_STATISTIC
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   SCIP_EVENTHDLR* eventhdlr;

   /* create bestsol event handler data */
   eventhdlrdata = NULL;
   SCIP_CALL( SCIPallocMemory(scip, &eventhdlrdata) );
   assert(eventhdlrdata != NULL);

   eventhdlr = NULL;

   /* include event handler into SCIP */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC,
         eventExecBestsol, eventhdlrdata) );
   assert(eventhdlr != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetEventhdlrFree(scip, eventhdlr, eventFreeBestsol) );
   SCIP_CALL( SCIPsetEventhdlrInit(scip, eventhdlr, eventInitBestsol) );
   SCIP_CALL( SCIPsetEventhdlrExit(scip, eventhdlr, eventExitBestsol) );
   SCIP_CALL( SCIPsetEventhdlrExitsol(scip, eventhdlr, eventExitsolBestsol) );
#endif

   return SCIP_OKAY;
}
