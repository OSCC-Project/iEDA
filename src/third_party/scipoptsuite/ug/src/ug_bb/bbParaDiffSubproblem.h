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

/**@file    paraDiffSubproblem.h
 * @brief   Base class for a container which has difference between instance and subproblem.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_DIFF_SUBPROBLEM_H__
#define __BB_PARA_DIFF_SUBPROBLEM_H__

#include <iostream>
#include <fstream>

#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "ug/paraDiffSubproblem.h"
#include "ug/paraComm.h"
#include "ug/paraInstance.h"
#include "bbParaNodesMerger.h"


/** uncomment this define to activate debugging on given solution */
/** PARASCIP_DEBUG only valid for PARASCIP */
// #define UG_DEBUG_SOLUTION "timtab2-trans.sol"

namespace UG
{

class ParaInitiator;
class BbParaRacingRampUpParamSet;

///
/// Class for the difference between instance and subproblem
///
/// This class should NOT have any data member.
///
class BbParaDiffSubproblem : public ParaDiffSubproblem
{

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   ///  default constructor
   ///
   BbParaDiffSubproblem(
         )
   {
   }

   ///
   ///  destractor¥
   ///
   virtual ~BbParaDiffSubproblem(
         )
   {
   }

#ifdef UG_WITH_ZLIB

   ///
   /// function to read BbParaDiffSubproblem object from checkpoint file
   ///
   virtual void read(
         ParaComm *comm,           ///< communicator used
         gzstream::igzstream &in,  ///< gzstream for input
         bool onlyBoundChanges     ///< indicate if only bound changes are output or not
         ) = 0;

#endif

   ///
   /// get the number of bound changes
   /// @return the number of bound changes
   ///
   virtual int getNBoundChanges(
         ) = 0;

   ///
   /// get the number of fixed variables
   /// @return the number of fixed variables
   ///
   virtual int getFixedVariables(
         ParaInstance *instance,            ///< pointer to instance object
         BbParaFixedVariable **fixedVars    ///< array of fixed variables
         ) = 0;

   ///
   /// create new BbParaDiffSubproblem object using fixed variables information
   /// @return pointer to BbParaDiffSubproblem object
   ///
   virtual BbParaDiffSubproblem* createDiffSubproblem(
         ParaComm *comm,                  ///< communicator used
         ParaInitiator *initiator,        ///< point to ParaInitiator object
         int n,                           ///< the number of fixed variables
         BbParaFixedVariable *fixedVars     ///< array of the fixed variables
         ) = 0;

   ///
   /// stringfy statistics of BbParaDiffSubproblem object
   /// @return string to show some statistics of this object
   ///
   virtual const std::string toStringStat(
         )
   {
      return std::string("");
   }


   ///
   /// set winner racing parameters at warm start racing
   ///
   virtual void setWinnerParams(
         BbParaRacingRampUpParamSet *winerParams  ///< pointer to winner racing ramp-up parameters
         )
   {
      std::cout << "**** virtual function UG::BbParaDiffSubproblem::setWinnerParams() is called. *****" << std::endl;
   }

   ///
   /// get winner racing parameters at warm start racing
   /// @return winner racing ramp-up parameters
   ///
   virtual BbParaRacingRampUpParamSet *getWinnerParams(
         )
   {
      std::cout << "**** virtual function UG::BbParaDiffSubproblem::getWinnerParams() is called. *****" << std::endl;
      return 0;
   }

#ifdef UG_DEBUG_SOLUTION

   ///
   /// check if an optimal solution is included in this subproblem or not (for debugging)
   ///
   virtual bool isOptimalSolIncluded(
         ) = 0;

   ///
   /// set indicator to show that an optimal solution is included (for debugging)
   ///
   virtual void setOptimalSolIndicator(
         int i       ///< 1: when an optimal solution is included, 0 others
         ) = 0;

#endif

};

}

#endif    // __BB_PARA_DIFF_SUBPROBLEM_H__
