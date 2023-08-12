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


#ifndef __PARA_SOLUTION_H__
#define __PARA_SOLUTION_H__

#include "paraComm.h"
#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif

namespace UG
{

///
/// class for solution
///
/// This class should NOT have any data member.
///
class ParaSolution
{

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   /// default constructor
   ///
   ParaSolution(
      )
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaSolution(
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
   /// create clone of this object
   /// @return pointer to ParaSolution object
   ///
   virtual ParaSolution *clone(
         ParaComm *comm
         ) = 0;

   ///
   /// broadcast solution data
   ///
   virtual void bcast(
         ParaComm *comm,           ///< communicator used
         int root                  ///< root rank for broadcast
         ) = 0;

   ///
   /// send solution data
   ///
   virtual void send(
         ParaComm *comm,           ///< communicator used
         int destination           ///< destination rank
         ) = 0;

   ///
   /// receive solution data
   ///
   virtual void receive(
         ParaComm *comm,           ///< communicator used
         int source                ///< source rank
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// function to write ParaSolution object to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out   ///< gzstream for output
         ) = 0;

   ///
   /// function to read ParaSolution object from checkpoint file
   ///
   virtual bool read(
         ParaComm *comm,           ///< communicator used
         gzstream::igzstream &in   ///< gzstream for input
         ) = 0;

#endif

   ///
   /// stringfy ParaSolution object
   /// @return string to show inside of this object
   ///
   virtual const std::string toString(
         )
   {
      return std::string("");
   }

};

typedef ParaSolution *ParaSolutionPtr;

}

#endif // __PARA_SOLUTION_H__
