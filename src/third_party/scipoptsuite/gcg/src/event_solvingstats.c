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

/**@file   event_solvingstats.c
 * @brief  eventhdlr for writing various types of information during the solving process
 * @author Gerald Gamrath
 *
 * If the filename is specified, a file is created and the eventhandler is installed to catch all events announcing that
 * a node was solved or that a new best solution was found.
 * Whenever one of these things happens, a line is printed to the file with the following information:
 * 1) solving time
 * 2) number of processed nodes (including the current node)
 * 3) number of open nodes
 * 4) number of LP iterations
 * 5) number of variables in the master problem
 * 6) current global dual bound
 * 7) current primal bound
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "event_solvingstats.h"
#include "pricer_gcg.h"
#include <string.h>

#define EVENTHDLR_NAME         "solvingstats"
#define EVENTHDLR_DESC         "event handler for best solutions found"

#define DEFAULT_FILENAME       ""   /**< filename to write to */

/*
 * Data structures
 */

/** event handler data */
struct SCIP_EventhdlrData
{
   SCIP*                 origprob;           /**< pointer to the original SCIP instance */
   FILE*                 file;               /**< file to which statistics should be written */
   char*                 filename;           /**< name of file to which statistics should be written */
};

/** destructor of event handler to free user data (called when SCIP is exiting) */
static
SCIP_DECL_EVENTFREE(eventFreeSolvingstats)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);
   assert(eventhdlrdata->file == NULL);

   SCIPfreeMemory(scip, &eventhdlrdata);

   return SCIP_OKAY;
}

/** initialization method of event handler (called after problem was transformed) */
static
SCIP_DECL_EVENTINIT(eventInitSolvingstats)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* check if we need to open the file */
   if( strlen(eventhdlrdata->filename) > 0 )
   {
      assert(eventhdlrdata->origprob != NULL);
      assert(eventhdlrdata->file == NULL);

      eventhdlrdata->file = fopen(eventhdlrdata->filename, "w");

      if( eventhdlrdata->file == NULL )
      {
         SCIPerrorMessage("cannot create file <%s> for writing\n", eventhdlrdata->filename);
         SCIPprintSysError(eventhdlrdata->filename);
         return SCIP_FILECREATEERROR;
      }

      /* notify SCIP that your event handler wants to react on the event types best solution found and node solved */
      SCIP_CALL( SCIPcatchEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND | SCIP_EVENTTYPE_NODESOLVED, eventhdlr, NULL, NULL) );
   }

   return SCIP_OKAY;
}

/** deinitialization method of event handler (called before transformed problem is freed) */
static
SCIP_DECL_EVENTEXIT(eventExitSolvingstats)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(scip != NULL);
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   if( eventhdlrdata->file != NULL )
   {
      (void) fclose(eventhdlrdata->file);

      /* notify SCIP that your event handler wants to drop the event types best solution found and node solved */
      SCIP_CALL( SCIPdropEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND | SCIP_EVENTTYPE_NODESOLVED, eventhdlr, NULL, -1) );

      eventhdlrdata->file = NULL;
   }

   return SCIP_OKAY;
}

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecSolvingstats)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);
   assert(event != NULL);
   assert(scip != NULL);
   /*assert(SCIPeventGetType(event) & (SCIP_EVENTTYPE_BESTSOLFOUND | SCIP_EVENTTYPE_NODESOLVED) != 0);*/

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);
   assert(eventhdlrdata->file != NULL);

   SCIPdebugMessage("exec method of event handler for writing information during the solving process\n");

   SCIPinfoMessage(scip, eventhdlrdata->file, "%8.2f %7"SCIP_LONGINT_FORMAT" %7d %10"SCIP_LONGINT_FORMAT" %d %16.9g %16.9g\n",
      SCIPgetSolvingTime(scip), SCIPgetNNodes(scip), SCIPgetNNodesLeft(scip), SCIPgetNLPIterations(scip),
      SCIPgetNVars(scip), SCIPretransformObj(eventhdlrdata->origprob, SCIPgetDualbound(scip)),
      SCIPretransformObj(eventhdlrdata->origprob, SCIPgetPrimalbound(scip)));

   return SCIP_OKAY;
}

/** includes event handler for best solution found */
SCIP_RETCODE SCIPincludeEventHdlrSolvingstats(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   SCIP_EVENTHDLR* eventhdlr;

   /* create bounds reader data */
   SCIP_CALL( SCIPallocMemory(scip, &eventhdlrdata) );
   eventhdlrdata->origprob = GCGmasterGetOrigprob(scip);
   eventhdlrdata->file = NULL;
   eventhdlrdata->filename = NULL;
   eventhdlr = NULL;

   /* create event handler for events on watched variables */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC, eventExecSolvingstats, eventhdlrdata) );
   assert(eventhdlr != NULL);

   SCIP_CALL( SCIPsetEventhdlrFree(scip, eventhdlr, eventFreeSolvingstats) );
   SCIP_CALL( SCIPsetEventhdlrInit(scip, eventhdlr, eventInitSolvingstats) );
   SCIP_CALL( SCIPsetEventhdlrExit(scip, eventhdlr, eventExitSolvingstats) );

   /* add boundwriting parameters */
   SCIP_CALL( SCIPaddStringParam(eventhdlrdata->origprob,
         "eventhdlr/"EVENTHDLR_NAME"/filename",
         "filename to write all bounds to",
         &eventhdlrdata->filename, FALSE, DEFAULT_FILENAME, NULL, NULL) );

   return SCIP_OKAY;
}
