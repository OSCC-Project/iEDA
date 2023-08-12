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


#ifndef __BB_PARA_RACING_RAMP_UP_PARAM_SET_H__
#define __BB_PARA_RACING_RAMP_UP_PARAM_SET_H__

#include "ug/paraRacingRampUpParamSet.h"
#include "bbParaComm.h"
#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif

namespace UG
{

static const int RacingTerminateWithNNodesLeft = 0;
static const int RacingTerminateWithTimeLimit = 1;

///
/// class BbParaRacingRampUpParamSet
/// (parameter set for racing ramp-up)
///
///
class BbParaRacingRampUpParamSet : public ParaRacingRampUpParamSet
{

protected:

   int nNodesLeft;             ///< stop racing number of nodes left
   double timeLimit;           ///< stop racing time limit

public:

   ///
   /// default constructor
   ///
   BbParaRacingRampUpParamSet(
         )
         : ParaRacingRampUpParamSet(RacingTerminationNotDefined),
           nNodesLeft(-1),
           timeLimit(-1.0)
   {
   }

   ///
   /// constructor
   ///
   BbParaRacingRampUpParamSet(
         int inTerminationCriteria,  ///< termination criteria of racing ramp-up
         int inNNodesLeft,           ///< stop racing number of nodes left
         double inTimeLimit          ///< stop racing time limit
         )
         : ParaRacingRampUpParamSet(inTerminationCriteria),
           nNodesLeft(inNNodesLeft),
           timeLimit(inTimeLimit)
   {
   }

   ///
   ///  destructor
   ///
   virtual ~BbParaRacingRampUpParamSet(
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
   /// get stop racing number of nodes left
   /// @return the number of nodes left
   ///
   int getStopRacingNNodesLeft(
         )
   {
      return nNodesLeft;
   }

   ///
   /// get stop racing time limimt
   /// @return time to stop racing
   ///
   double getStopRacingTimeLimit(
         )
   {
      return timeLimit;
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
   /// send BbParaRacingRampUpParamSet
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         ParaComm *comm,           ///< communicator used
         int destination           ///< destination rank
         ) = 0;

   ///
   /// receive BbParaRacingRampUpParamSet
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
   /// stringfy BbParaRacingRampUpParamSet
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

typedef BbParaRacingRampUpParamSet *BbParaRacingRampUpParamSetPtr;

}

#endif // __BB_PARA_RACING_RAMP_UP_PARAM_SET_H__

