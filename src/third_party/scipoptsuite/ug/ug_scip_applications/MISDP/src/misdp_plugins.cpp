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

/**@file   misdpPlugins.cpp
 * @brief  MISDP user plugins
 * @author Chuen-Teck See
 * @author Tristan Gally
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "ug_scip/scipUserPlugins.h"
#include "ug_scip/scipParaSolver.h"
#include "ug_scip/scipParaInitiator.h"

#include "objscip/objscipdefplugins.h"
#include "scipsdp/cons_sdp.h"
#include "scipsdp/cons_savedsdpsettings.h"
#include "scipsdp/cons_savesdpsol.h"
#include "scipsdp/relax_sdp.h"
#include "scipsdp/objreader_sdpa.h"
#include "scipsdp/reader_cbf.h"
#include "scipsdp/prop_sdpredcost.h"
#include "scipsdp/disp_sdpiterations.h"
#include "scipsdp/disp_sdpavgiterations.h"
#include "scipsdp/disp_sdpfastsettings.h"
#include "scipsdp/disp_sdppenalty.h"
#include "scipsdp/disp_sdpunsolved.h"
#include "scipsdp/branch_sdpmostfrac.h"
#include "scipsdp/branch_sdpmostinf.h"
#include "scipsdp/branch_sdpobjective.h"
#include "scipsdp/branch_sdpinfobjective.h"
#include "scipsdp/heur_sdpfracdiving.h"
#include "scipsdp/heur_sdprand.h"
#include "scipsdp/prop_companalcent.h"
#include "scipsdp/prop_sdpobbt.h"

using namespace UG;
using namespace ParaSCIP;

class MisdpUserPlugins : public ScipUserPlugins {
   void operator()(SCIP *scip)
   {
      SCIP_CALL_ABORT( SCIPincludeObjReader(scip, new scip::ObjReaderSDPA(scip), FALSE) ); //last arg is TRUE originally
      SCIP_CALL_ABORT( SCIPincludeReaderCbf(scip) );

      SCIP_CALL_ABORT( SCIPincludeConshdlrSdp(scip) );
      SCIP_CALL_ABORT( SCIPincludeConshdlrSavedsdpsettings(scip) );
      SCIP_CALL_ABORT( SCIPincludeConshdlrSavesdpsol(scip) );
      SCIP_CALL_ABORT( SCIPincludeRelaxSdp(scip) );
      SCIP_CALL_ABORT( SCIPincludePropSdpredcost(scip) );
      SCIP_CALL_ABORT( SCIPincludeBranchruleSdpmostfrac(scip) );
      SCIP_CALL_ABORT( SCIPincludeBranchruleSdpmostinf(scip) );
      SCIP_CALL_ABORT( SCIPincludeBranchruleSdpobjective(scip) );
      SCIP_CALL_ABORT( SCIPincludeBranchruleSdpinfobjective(scip) );
      SCIP_CALL_ABORT( SCIPincludeHeurSdpFracdiving(scip) );
      SCIP_CALL_ABORT( SCIPincludeHeurSdpRand(scip) );
      SCIP_CALL_ABORT( SCIPincludePropSdpObbt(scip) );
      SCIP_CALL_ABORT( SCIPincludePropCompAnalCent(scip) );

      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "timing/clocktype", 2) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "lp/solvefreq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "relaxing/SDP/freq", 1) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/epsilon", 1e-9) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/sumepsilon", 1e-6) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/feastol", 1e-6) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "lp/cleanuprows", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "lp/cleanuprowsroot", FALSE) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "nodeselection/hybridestim/stdpriority", 1000000) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "nodeselection/hybridestim/maxplungedepth", 0) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "nodeselection/hybridestim/estimweight", 0.0) );

#ifdef SCIP_DISABLED_CODE
      /* The display columns are deactivated by default anyways within UG, and currently there is a
       * problem with MPI where the display columns are not copied correctly before the parameters
       * are updated.
       */
      SCIP_CALL_ABORT( SCIPincludeDispSdpiterations(scip) );
      SCIP_CALL_ABORT( SCIPincludeDispSdpavgiterations(scip) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/lpiterations/active", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/lpavgiterations/active", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/nfrac/active", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/curcols/active", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/strongbranchs/active", 0) );
      SCIP_CALL_ABORT( SCIPincludeDispSdpfastsettings(scip) );
      SCIP_CALL_ABORT( SCIPincludeDispSdppenalty(scip) );
      SCIP_CALL_ABORT( SCIPincludeDispSdpunsolved(scip) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/sdpfastsettings/active", 0) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/sdppenalty/active", 0) );
#endif
   }

   void newSubproblem(
         SCIP *scip,
         const ScipParaDiffSubproblemBranchLinearCons *linearConss,
         const ScipParaDiffSubproblemBranchSetppcCons *setppcConss)
   {
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "timing/clocktype", 2) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "lp/solvefreq", -1) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "relaxing/SDP/freq", 1) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/epsilon", 1e-9) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/sumepsilon", 1e-6) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "numerics/feastol", 1e-6) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "lp/cleanuprows", FALSE) );
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "lp/cleanuprowsroot", FALSE) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "nodeselection/hybridestim/stdpriority", 1000000) );
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "nodeselection/hybridestim/maxplungedepth", 0) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "nodeselection/hybridestim/estimweight", 0.0) );
   }

};

void
setUserPlugins(ParaInitiator *inInitiator)
{
   ScipParaInitiator *initiator = dynamic_cast<ScipParaInitiator *>(inInitiator);
   initiator->setUserPlugins(new MisdpUserPlugins());
}

void
setUserPlugins(ParaInstance *inInstance)
{
   ScipParaInstance *instance = dynamic_cast<ScipParaInstance *>(inInstance);
   instance->setUserPlugins(new MisdpUserPlugins());
}

void
setUserPlugins(ParaSolver *inSolver)
{
   ScipParaSolver *solver = dynamic_cast<ScipParaSolver *>(inSolver);
   solver->setUserPlugins(new MisdpUserPlugins());
}
