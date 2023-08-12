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

/**@file   weights.h
 * @brief  weight class for graphs
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef WEIGHTS_H_
#define WEIGHTS_H_
#include "objscip/objscip.h"

namespace gcg {

class Weights
{
protected:
   int varweight;                      /**< weight of a variable vertex */
   int vbinary;                        /**< weight of a binary variable vertex */
   int vcontinous;                     /**< weight of a continuous variable vertex */
   int vinteger;                       /**< weight of an integer variable vertex */
   int vimplint;                       /**< weight of an implicit integer variable vertex */
   int consweight;                     /**< weight of a constraint vertex */

public:
   Weights(
      int varweight_,
      int vbinary_,
      int vcontinous_,
      int vinteger_,
      int vimplint_,
      int consweight_
   );
   Weights();

   virtual ~Weights();
   int calculate(SCIP_CONS* cons) const;
   int calculate(SCIP_VAR* var) const;
};

} /* namespace gcg */
#endif /* WEIGHTS_H_ */
