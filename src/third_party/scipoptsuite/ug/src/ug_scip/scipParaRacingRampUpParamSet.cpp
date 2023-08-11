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

/**@file    scipParaRacingRampUpParamSet.cpp
 * @brief   ParaRacingRampUpParamSet extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaComm.h"
#include "scipParaRacingRampUpParamSet.h"

using namespace UG;
using namespace ParaSCIP;

#ifdef UG_WITH_ZLIB
/** write scipParaRacingRampUpParamSet */
void 
ScipParaRacingRampUpParamSet::write(
    gzstream::ogzstream &out
    )
{
   out.write((char *)&scipRacingParamSeed, sizeof(int));
   out.write((char *)&permuteProbSeed, sizeof(int));
   out.write((char *)&generateBranchOrderSeed, sizeof(int));
   out.write((char *)&scipDiffParamSetInfo, sizeof(int));
   if( scipDiffParamSetInfo )
   {
      scipDiffParamSet->write(out);
   }
}

/** read scipParaRacingRampUpParamSet */
bool 
ScipParaRacingRampUpParamSet::read(
     ParaComm *comm,
     gzstream::igzstream &in
     )
{
   in.read((char *)&scipRacingParamSeed, sizeof(int));
   if( in.eof() ) return false;
   in.read((char *)&permuteProbSeed, sizeof(int));
   in.read((char *)&generateBranchOrderSeed, sizeof(int));
   in.read((char *)&scipDiffParamSetInfo, sizeof(int));
   if( scipDiffParamSetInfo )
   {
      DEF_SCIP_PARA_COMM(scipParaComm, comm);
      scipDiffParamSet = scipParaComm->createScipDiffParamSet();
      scipDiffParamSet->read(comm, in);
   }
   return true;
}
#endif
