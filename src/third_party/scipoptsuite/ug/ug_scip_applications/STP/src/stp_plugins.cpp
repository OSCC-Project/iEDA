/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
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
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   stp_plugins.cpp
 * @brief  Steiner Tree Solver user plugins
 * @author Yuji Shinano
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "ug_scip/scipUserPlugins.h"
#include "ug_scip/scipParaSolver.h"
#include "ug_scip/scipParaInitiator.h"

#include "scip/scipdefplugins.h"
#include "cons_stp.h"
#include "relax_stp.h"
#include "relax_stpenum.h"
#include "relax_stpdp.h"
#include "reader_stp.h"
#include "reader_gr.h"
#include "heur_tm.h"
#include "heur_local.h"
#include "heur_prune.h"
#include "heur_ascendprune.h"
#include "heur_slackprune.h"
#include "heur_rec.h"
#include "pricer_stp.h"
#include "event_bestsol.h"
#include "probdata_stp.h"
#include "dialog_stp.h"
#include "prop_stp.h"
#include "branch_stp.h"

using namespace UG;
using namespace ParaSCIP;

/* to provide rank ans size to user  */
static ParaComm *paraComm = 0;

class StpUserPlugins : public ScipUserPlugins {
   void operator()(SCIP *scip)
   {
      /* include steiner tree readers */
      SCIP_CALL_ABORT( SCIPincludeReaderStp(scip) );
      SCIP_CALL_ABORT( SCIPincludeReaderGr(scip) );

      /* include default SCIP plugins */
      // SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );

      /* include STP dialog */
      SCIP_CALL_ABORT( SCIPincludeDialogStp(scip) );

      /* include steiner tree constraint handler */
      SCIP_CALL_ABORT( SCIPincludeConshdlrStp(scip) );

      /* include relaxators */
      SCIP_CALL_ABORT( SCIPincludeRelaxStp(scip) );
      SCIP_CALL_ABORT( SCIPincludeRelaxStpenum(scip) );
      SCIP_CALL_ABORT( SCIPincludeRelaxStpdp(scip) );

      /* include heuristics */
      SCIP_CALL_ABORT( SCIPStpIncludeHeurTM(scip) );
      SCIP_CALL_ABORT( SCIPStpIncludeHeurLocal(scip) );
      SCIP_CALL_ABORT( SCIPStpIncludeHeurRec(scip) );
      SCIP_CALL_ABORT( SCIPStpIncludeHeurPrune(scip) );
      SCIP_CALL_ABORT( SCIPStpIncludeHeurSlackPrune(scip) );


#if 0
      /* include ascend-and-prune heuristic */
      SCIP_CALL_ABORT( SCIPincludeHeurAscendPrune(scip) );
#endif

      /* include event handler for printing primal solution development */
      SCIP_CALL_ABORT( SCIPincludeEventHdlrBestsol(scip) );

      /* include branching rule */
      SCIP_CALL_ABORT( SCIPincludeBranchruleStp(scip) );

      /* include propagator */
      SCIP_CALL_ABORT( SCIPincludePropStp(scip) );

      SCIP_CALL_ABORT( SCIPsetSubscipsOff(scip, TRUE) );

      SCIP_CALL_ABORT( SCIPprobdataSetDefaultParams(scip) );
   }

   void writeUserSolution(SCIP *scip, int nSolvers, double dual)
   {
      SCIPprobdataSetDualBound(scip, dual);
      SCIPprobdataSetNSolvers(scip, nSolvers);
      SCIPprobdataWriteLogfileEnd(scip);
   }

   void newSubproblem(
         SCIP *scip,
         const ScipParaDiffSubproblemBranchLinearCons *linearConss,
         const ScipParaDiffSubproblemBranchSetppcCons *setppcConss)
   {
      if( linearConss && setppcConss )
      {
         initReceivedSubproblem(scip, linearConss->nLinearConss, linearConss->consNames, setppcConss->nSetppcConss, setppcConss->consNames);
      } else if( linearConss && !setppcConss )
      {
         initReceivedSubproblem(scip, linearConss->nLinearConss, linearConss->consNames, 0, 0);
      } else if( !linearConss && setppcConss )
      {
         initReceivedSubproblem(scip, 0, 0, setppcConss->nSetppcConss, setppcConss->consNames);
      } else
      {
         initReceivedSubproblem(scip, 0, 0, 0, 0);
      }

   }

};


void
setUserPlugins(ParaInitiator *inInitiator)
{
   ScipParaInitiator *initiator = dynamic_cast<ScipParaInitiator *>(inInitiator);
   initiator->setUserPlugins(new StpUserPlugins());
   paraComm = initiator->getParaComm();
}

void
setUserPlugins(ParaInstance *inInstance)
{
   ScipParaInstance *instance = dynamic_cast<ScipParaInstance *>(inInstance);
   instance->setUserPlugins(new StpUserPlugins());
}

void
setUserPlugins(ParaSolver *inSolver)
{
   ScipParaSolver *solver = dynamic_cast<ScipParaSolver *>(inSolver);
   solver->setUserPlugins(new StpUserPlugins());
   if( !paraComm )
   {
      paraComm = solver->getParaComm();
   }
}

extern "C"
int getUgRank()
{
   return paraComm->getRank();
}

extern "C"
int getUgSize()
{
   return paraComm->getSize();
}

extern "C"
const char*
getBranchLinearConsName(const char* names, int i)
{
   const char *name = names;
   for( int j = 0; j < i; j++)
   {
      assert(*name);
      name += (std::strlen(name) + 1);
   }
   assert(*name);
   return name;
}

extern "C"
const char*
getBranchSetppcConsName(const char* names, int i)
{
   const char *name = names;
   for( int j = 0; j < i; j++)
   {
      assert(*name);
      name += (std::strlen(name) + 1);
   }
   assert(*name);
   return name;
}




