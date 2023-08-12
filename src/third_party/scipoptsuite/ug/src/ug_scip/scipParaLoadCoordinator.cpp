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

/**@file    paraLoadCoordinator.cpp
 * @brief   Load coordinator.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifdef _MSC_VER
#include <functional>
#else
#include <unistd.h>
#endif
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <climits>
#include <algorithm>
#include <iomanip>

#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "scipParaInitialStat.h"
#include "scipParaLoadCoordinator.h"

using namespace UG;
using namespace ParaSCIP;

ScipParaLoadCoordinator::ScipParaLoadCoordinator(
#ifdef UG_WITH_UGS
      UGS::UgsParaCommMpi *inCommUgs,
#endif
      ParaComm *inComm,
      ParaParamSet *inParaParamSet,
      ParaInitiator *inParaInitiator,
      bool *inRacingSolversExist,
      ParaTimer *inParaTimer,
      ParaDeterministicTimer *inParaDetTimer
      )
      : UG::BbParaLoadCoordinator(
#ifdef UG_WITH_UGS
            inCommUgs,
#endif
            N_SCIP_TAGS,
            inComm,
            inParaParamSet,
            inParaInitiator,
            inRacingSolversExist,
            inParaTimer,
            inParaDetTimer
            )
{

   // std::cout << "ScipParaLoadCoordinator constructor" << std::endl;

   ScipMessageHandlerFunctionPointer *scipMessageHandler = reinterpret_cast<ScipMessageHandlerFunctionPointer *>(messageHandler);

   /** register message handlers */
   scipMessageHandler[TagInitialStat] = &ParaSCIP::ScipParaLoadCoordinator::processTagInitialStat;

#if SCIP_APIVERSION < 101
   if( paraParams->getIntParamValue(UG::RampUpPhaseProcess) ==  3 )
   {
      std::cout << "*** Self-Split ramp-up (RampUpPhaseProcess = 3) cannot work with this version of SCIP ***" << std::endl;
      exit(1);
   }
#endif

}

int
ScipParaLoadCoordinator::processTagInitialStat(
      int source,
      int tag
      )
{
   DEF_SCIP_PARA_COMM( scipParaComm, paraComm );

   ScipParaInitialStat *initialStat = dynamic_cast<ScipParaInitialStat *>(scipParaComm->createParaInitialStat());
   initialStat->receive(paraComm, source);
   if( maxDepthInWinnerSolverNodes < initialStat->getMaxDepth() )
   {
      maxDepthInWinnerSolverNodes = initialStat->getMaxDepth();
   }
   dynamic_cast<BbParaInitiator *>(paraInitiator)->accumulateInitialStat(initialStat);
   delete initialStat;
   return 0;
}

