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

/**@file    paraParamSet.cpp
 * @brief   Parameter set for UG framework.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string>
#include <map>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cfloat>
#include <climits>
#include <cassert>
#include <scip/scip.h>
#include "ug_bb/bbParaComm.h"
#include "scipParaParamSet.h"

using namespace ParaSCIP;

// ParaParam *BbParaParamSet::paraParams[ParaParamsSize];

ScipParaParamSet::ScipParaParamSet(
      )
      : UG::BbParaParamSet(ScipParaParamsSize)
{

  /** bool params */
  paraParams[RootNodeSolvabilityCheck] = new UG::ParaParamBool(
          "RootNodeSolvabilityCheck",
          "# Indicate if root node solvability is checked before transfer or not. TRUE: root node solvability is checked, FALSE: no check [Default value: FALSE]",
          false);
  paraParams[CustomizedToSharedMemory] = new UG::ParaParamBool(
          "CustomizedToSharedMemory",
          "# Customized to shared memory environment, if it runs on it. [Default value: TRUE]",
          true);
  paraParams[LocalBranching] = new UG::ParaParamBool(
          "LocalBranching",
          "# Apply distributed local branching. [Default value: FALSE]",
          false);

   /** int params */
  paraParams[AddDualBoundCons] = new UG::ParaParamInt(
         "AddDualBoundCons",
         "# Adding constraint: objective func >= dualBoundValue (This is not a good idea, because it creates many degenerate solutions) : 0 - no adding, 1 - adding to discarded ParaNodes only, 2 - adding always, 3 - adding at warm start [Default value: 0]",
         0,
         0,
         3);


   /** longint params */

   /** real params */
   std::ostringstream s;
   s << "# Memory limit for a process [Default value: " << SCIP_MEM_NOLIMIT << "][0," << SCIP_MEM_NOLIMIT << "]";
   static char memLimitStr[256];
   strcpy(memLimitStr, s.str().c_str());
   paraParams[MemoryLimit] = new UG::ParaParamReal(
         "MemoryLimit",
          memLimitStr,
         (SCIP_Real)SCIP_MEM_NOLIMIT,
         0.0,
         (SCIP_Real)SCIP_MEM_NOLIMIT);

   /** char params */

   /** string params */

}


