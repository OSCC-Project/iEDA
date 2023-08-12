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

/**@file    scipParaSolution.cpp
 * @brief   ParaSolution extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "scipParaSolution.h"

using namespace ParaSCIP;

#ifdef UG_WITH_ZLIB

void
ScipParaSolution::write(
      gzstream::ogzstream &out
      )
{
   out.write((char *)&objectiveFunctionValue, sizeof(double));
   out.write((char *)&nVars, sizeof(int));
   for(int i = 0; i < nVars; i++ )
   {
      out.write((char *)&indicesAmongSolvers[i], sizeof(int));
      out.write((char *)&values[i], sizeof(SCIP_Real));
   }
}

bool
ScipParaSolution::read(
      UG::ParaComm *comm,
      gzstream::igzstream &in
      )
{
   in.read((char *)&objectiveFunctionValue, sizeof(double));
   if( in.eof() ) return false;
   in.read((char *)&nVars, sizeof(int));
   indicesAmongSolvers = new int[nVars];
   values = new SCIP_Real[nVars];
   for(int i = 0; i < nVars; i++ )
   {
      in.read((char *)&indicesAmongSolvers[i], sizeof(int));
      in.read((char *)&values[i], sizeof(SCIP_Real));
   }
   return true;
}

#endif
