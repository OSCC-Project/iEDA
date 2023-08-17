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

/**@file    scipParaRacingRampUpParamSet.h
 * @brief   ParaRacingRampUpParamSet extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_H__
#define __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_H__

#include "scip/scip.h"
#include "ug_bb/bbParaRacingRampUpParamSet.h"
#include "ug_bb/bbParaComm.h"
#include "scipDiffParamSet.h"

namespace ParaSCIP
{

/** The racing ramp-up parameter set for SCIP solver */
class ScipParaRacingRampUpParamSet : public UG::BbParaRacingRampUpParamSet
{
protected:
   int scipRacingParamSeed;             /**< seed to generate SCIP racing parameter */
   int permuteProbSeed;                 /**< seed to permute problem */
   int generateBranchOrderSeed;         /**< seed to generate branching order */
   int scipDiffParamSetInfo;            /**< 1: with scipDiffParamSet, 0: no scipDiffParamSet */
   ScipDiffParamSet *scipDiffParamSet;  /**< scip parameter set different from default values for racing ramp-up */
public:
   /** default constructor */
   ScipParaRacingRampUpParamSet(
         )
         : BbParaRacingRampUpParamSet(), scipRacingParamSeed(-1),
           permuteProbSeed(0), generateBranchOrderSeed(0), scipDiffParamSetInfo(0), scipDiffParamSet(0)
   {
   }

   ScipParaRacingRampUpParamSet(
         int inTerminationCriteria,
         int inNNodesLeft,
         double inTimeLimit,
         int inScipRacingParamSeed,
         int inPermuteProbSeed,
         int inGenerateBranchOrderSeed,
         ScipDiffParamSet *inScipDiffParamSet
         )
         : BbParaRacingRampUpParamSet(inTerminationCriteria, inNNodesLeft, inTimeLimit),
           scipRacingParamSeed(inScipRacingParamSeed),permuteProbSeed(inPermuteProbSeed),
           generateBranchOrderSeed(inGenerateBranchOrderSeed), scipDiffParamSetInfo(0), scipDiffParamSet(inScipDiffParamSet)
   {
      if( inScipDiffParamSet ) scipDiffParamSetInfo = 1;
   }

   /** destructor */
   virtual ~ScipParaRacingRampUpParamSet()
   {
      if( scipDiffParamSet ) delete scipDiffParamSet;
   }


   /** getter of permuteProbSeed */
   int getPermuteProbSeed(
         )
   {
      return permuteProbSeed;
   }

   /** getter of generateBranchOrderSeed */
   int getGenerateBranchOrderSeed(
         )
   {
      return generateBranchOrderSeed;
   }

   /** getter of ScipDiffParamSet */
   ScipDiffParamSet *getScipDiffParamSet(
         )
   {
      return scipDiffParamSet;
   }

   int getScipRacingParamSeed(
         )
   {
      return scipRacingParamSeed;
   }

#ifdef UG_WITH_ZLIB
   /** write scipParaRacingRampUpParamSet */
   void write(
         gzstream::ogzstream &out
         );

   /** read scipParaRacingRampUpParamSet */
   bool read(
         UG::ParaComm *comm,
         gzstream::igzstream &in
         );
#endif

   /** stringfy ScipParaRacingRampUpParamSet */
   const std::string toString(
         )
   {
      std::ostringstream s;
      s << "[ SCIP racing parameter seed; " << scipRacingParamSeed;
      s << ", Permutate problem seed: " << permuteProbSeed << ", Generate branch order seed: " << generateBranchOrderSeed << " ]" << std::endl;
      if( scipDiffParamSetInfo )
      {
         s << scipDiffParamSet->toString();
      }
      return s.str();
   }

   int getStrategy(
         )
   {
      return scipRacingParamSeed;
   }
};

}



#endif    // __SCIP_PARA_RACING_RAMP_UP_PARAM_SET_H__

