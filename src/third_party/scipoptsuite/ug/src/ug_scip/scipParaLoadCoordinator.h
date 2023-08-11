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

/**@file    paraLoadCoordinator.h
 * @brief   Load Coordinator.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_LOADCOORDINATOR_H__
#define __SCIP_PARA_LOADCOORDINATOR_H__

#include <fstream>
#include <list>
#include <queue>
#include "ug/paraDef.h"
#include "ug/paraTimer.h"
#include "ug/paraDeterministicTimer.h"
#include "ug_bb/bbParaLoadCoordinator.h"
#include "ug_bb/bbParaNodePool.h"
#include "ug_bb/bbParaInitiator.h"
#include "ug_bb/bbParaSolverState.h"
#include "ug_bb/bbParaCalculationState.h"
#include "ug_bb/bbParaNode.h"
#include "ug_bb/bbParaParamSet.h"
#include "ug_bb/bbParaTagDef.h"
#include "ug_bb/bbParaLoadCoordinatorTerminationState.h"
#include "ug_bb/bbParaSolverPool.h"
#include "ug_bb/bbParaSolution.h"
#include "ug_bb/bbParaInstance.h"
#include "ug_bb/bbParaDiffSubproblem.h"
#include "scipParaComm.h"

#ifdef UG_WITH_UGS
#include "ugs/ugsDef.h"
#include "ugs/ugsParaCommMpi.h"
#endif

namespace ParaSCIP
{

///
/// Class for LoadCoordinator
///
class ScipParaLoadCoordinator : public UG::BbParaLoadCoordinator
{

   typedef int(ScipParaLoadCoordinator::*ScipMessageHandlerFunctionPointer)(int, int);

   ///////////////////////
   ///
   /// Message handlers
   ///
   ///////////////////////

   ///
   /// function to process TagInitialStat message
   /// @return always 0 (for extension)
   ///
   int processTagInitialStat(
         int source,                                      ///< source solver rank
         int tag                                          ///< TagInitialStat
         );

public:

   ///
   /// constructor
   ///
   ScipParaLoadCoordinator(
#ifdef UG_WITH_UGS
         UGS::UgsParaCommMpi *inComUgs,          ///< communicator used for UGS
#endif
         UG::ParaComm *inComm,                     ///< communicator used
         UG::ParaParamSet *inParaParamSet,       ///< UG parameter set used
         UG::ParaInitiator *paraInitiator,       ///< ParaInitiator for initialization of solving algorithm
         bool *racingSolversExist,               ///< indicate racing solver exits or not
         UG::ParaTimer *paraTimer,               ///< ParaTimer used
         UG::ParaDeterministicTimer *detTimer    ///< DeterministicTimer used
         );

   ///
   /// destructor
   ///
   virtual ~ScipParaLoadCoordinator(
         )
   {
   }

};

}

#endif // __SCIP_PARA_LOADCOORDINATOR_H__

