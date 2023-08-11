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

/**@file    paraSolution.h
 * @brief   Base class for solution.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_SOLUTION_H__
#define __BB_PARA_SOLUTION_H__

#include "ug/paraSolution.h"

namespace UG
{

///
/// class for solution
///
/// This class should NOT have any data member.
///
class BbParaSolution : public ParaSolution
{

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   /// default constructor
   ///
   BbParaSolution(
      )
   {
   }

   ///
   /// destructor
   ///
   virtual ~BbParaSolution(
       )
   {
   }

   ///
   /// get objective function value
   /// @return objective function value
   ///
   virtual double getObjectiveFunctionValue(
         ) = 0;

   ///
   /// get cutoff value
   /// @return cutoff value
   ///
   virtual double getCutOffValue(
         )
   {
      throw "This solver cannot obtain a special cut off value";
   }

};

typedef BbParaSolution *BbParaSolutionPtr;

}

#endif // __BB_PARA_SOLUTION_H__
