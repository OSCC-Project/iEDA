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

/**@file   weights.cpp
 * @brief  weight class for graphs
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "weights.h"

namespace gcg {

Weights::Weights(
      int varweight_,
      int vbinary_,
      int vcontinous_,
      int vinteger_,
      int vimplint_,
      int consweight_
   ): varweight(varweight_),
      vbinary(vbinary_),
      vcontinous(vcontinous_),
      vinteger(vinteger_),
      vimplint(vimplint_),
      consweight(consweight_)
{
   // TODO Auto-generated constructor stub

}

Weights::Weights()
: varweight(1),
  vbinary(1),
  vcontinous(1),
  vinteger(1),
  vimplint(1),
  consweight(1)
{

}

Weights::~Weights()
{
   // TODO Auto-generated destructor stub
}

int Weights::calculate(SCIP_CONS* cons) const
{ /*lint -e715*/
   return consweight;
}
int Weights::calculate(SCIP_VAR* var) const

{
   int weight;

   assert(var != NULL);

   switch ( SCIPvarGetType(var) ) {
   case SCIP_VARTYPE_CONTINUOUS:
      weight = vcontinous;
      break;
   case SCIP_VARTYPE_INTEGER:
      weight = vinteger;
      break;
   case SCIP_VARTYPE_IMPLINT:
      weight = vimplint;
      break;
   case SCIP_VARTYPE_BINARY:
      weight = vbinary;
      break;
   default:
      weight = varweight;
      break;
   }

   return weight;
}
} /* namespace gcg */
