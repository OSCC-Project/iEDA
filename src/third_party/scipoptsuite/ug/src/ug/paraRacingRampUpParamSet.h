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

/**@file    paraRacingRampUpParamSet.h
 * @brief   Base class for racing ramp-up parameter set.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_RACING_RAMP_UP_PARAM_SET_H__
#define __PARA_RACING_RAMP_UP_PARAM_SET_H__

#include "paraComm.h"
#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif

namespace UG
{

static const int RacingTerminationNotDefined = -1;

///
/// class ParaRacingRampUpParamSet
/// (parameter set for racing ramp-up)
///
///
class ParaRacingRampUpParamSet
{

protected:

   int terminationCriteria;    ///< termination criteria of racing ramp-up : 0: number of nodes left, 1: time limit
                               ///< meaning must be defined in a derived class

public:

   ///
   /// default constructor
   ///
   ParaRacingRampUpParamSet(
         )
         : terminationCriteria(RacingTerminationNotDefined)
   {
   }

   ///
   /// constructor
   ///
   ParaRacingRampUpParamSet(
         int inTerminationCriteria   ///< termination criteria of racing ramp-up
         )
         : terminationCriteria(inTerminationCriteria)
   {
   }

   ///
   ///  destructor
   ///
   virtual ~ParaRacingRampUpParamSet(
         )
   {
   }

   ///
   /// get termination criteria
   /// @return an int value to show termination criteria
   ///
   int getTerminationCriteria(
         )
   {
      return terminationCriteria;
   }

   ///
   /// set winner rank
   /// TODO: this function and also getWinnerRank should be removed
   ///
   virtual void setWinnerRank(
         int rank
         )
   {
   }


   ///
   /// send ParaRacingRampUpParamSet
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         ParaComm *comm,           ///< communicator used
         int destination           ///< destination rank
         ) = 0;

   ///
   /// receive ParaRacingRampUpParamSet
   /// @return always 0 (for future extensions)
   ///
   virtual int receive(
         ParaComm *comm,           ///< communicator used
         int source                ///< source rank
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// write to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out  ///< gzstream for output
         ) = 0;

   ///
   /// read from checkpoint file
   ///
   virtual bool read(
         ParaComm *comm,           ///< communicator used
         gzstream::igzstream &in   ///< gzstream for input
         ) = 0;

#endif

   ///
   /// stringfy ParaRacingRampUpParamSet
   /// @return string to show inside of this object
   ///
   virtual const std::string toString(
         ) = 0;

   ///
   /// get strategy
   /// @return an int value which shows strategy
   ///
   virtual int getStrategy() = 0;

};

typedef ParaRacingRampUpParamSet *ParaRacingRampUpParamSetPtr;

}

#endif // __PARA_RACING_RAMP_UP_PARAM_SET_H__

