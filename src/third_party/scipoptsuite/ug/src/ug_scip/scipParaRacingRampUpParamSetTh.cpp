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

/**@file    scipParaRacingRampUpParamSetTh.cpp
 * @brief   ScipParaRacingRampUpParamSet extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaCommTh.h"
#include "scipDiffParamSetTh.h"
#include "scipParaRacingRampUpParamSetTh.h"
#include <cstring>

using namespace ParaSCIP;

/** create Datatype */
ScipParaRacingRampUpParamSetTh *
ScipParaRacingRampUpParamSetTh::createDatatype(
      )
{
   ScipDiffParamSetTh *scipDiffParamSetTh = dynamic_cast<ScipDiffParamSetTh *>(scipDiffParamSet);

   if( scipDiffParamSetInfo )
   {
      return new ScipParaRacingRampUpParamSetTh(
            terminationCriteria,
            nNodesLeft,
            timeLimit,
            scipRacingParamSeed,
            permuteProbSeed,
            generateBranchOrderSeed,
            scipDiffParamSetTh->clone()
            );
   }
   else
   {
      return new ScipParaRacingRampUpParamSetTh(
            terminationCriteria,
            nNodesLeft,
            timeLimit,
            scipRacingParamSeed,
            permuteProbSeed,
            generateBranchOrderSeed,
            0
            );
   }
}

int
ScipParaRacingRampUpParamSetTh::send(
      UG::ParaComm *comm,
      int dest)
{

   DEF_SCIP_PARA_COMM( commTh, comm);

   PARA_COMM_CALL(
      commTh->uTypeSend((void *)createDatatype(), UG::ParaRacingRampUpParamType, dest, UG::TagRacingRampUpParamSet)
   );

   return 0;

}

int
ScipParaRacingRampUpParamSetTh::receive(
      UG::ParaComm *comm,
      int source)
{

   DEF_SCIP_PARA_COMM( commTh, comm);

   ScipParaRacingRampUpParamSetTh *received;
   PARA_COMM_CALL(
      commTh->uTypeReceive((void **)&received, UG::ParaRacingRampUpParamType, source, UG::TagRacingRampUpParamSet)
   );

   terminationCriteria = received->terminationCriteria;
   nNodesLeft = received->nNodesLeft;
   timeLimit = received->timeLimit;
   scipRacingParamSeed = received->scipRacingParamSeed;
   permuteProbSeed = received->permuteProbSeed;
   generateBranchOrderSeed = received->generateBranchOrderSeed;
   scipDiffParamSetInfo = received->scipDiffParamSetInfo;
   if( scipDiffParamSetInfo )
   {
      ScipDiffParamSetTh *scipDiffParamSetTh = dynamic_cast<ScipDiffParamSetTh *>(received->scipDiffParamSet);
      scipDiffParamSet = scipDiffParamSetTh->clone();
   }

   delete received;

   return 0;

}
