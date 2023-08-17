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

/**@file    paraInitiator.h
 * @brief   Base class of initiator that maintains original problem and incumbent solution.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_INITIATOR_H__
#define __PARA_INITIATOR_H__

#include <string>
#include "paraComm.h"
#include "paraParamSet.h"
#include "paraInstance.h"
#include "paraDiffSubproblem.h"
#include "paraSolution.h"
#include "paraTask.h"
#include "paraRacingRampUpParamSet.h"
#include "paraInitialStat.h"
#include "uggithash.h"

#ifdef UG_WITH_UGS
#include "ugs/ugsDef.h"
#include "ugs/ugsParaCommMpi.h"
#endif

namespace UG
{

///
/// Class for initiator
///
class ParaInitiator
{

protected:
   ParaComm   *paraComm;                        ///< communicator used
   ParaTimer  *timer;                           ///< timer used
   char       *prefixWarm;                      ///< prefix of warm start files

public:

   ///
   /// constructor
   ///
   ParaInitiator(
         ParaComm *inComm,                     ///< communicator used
         ParaTimer *inTimer                    ///< timer used
         ) : 
        paraComm(inComm), 
        timer(inTimer), 
		  prefixWarm(0)
   {
      std::cout << "The following solver is parallelized by UG version "
            << UG_VERSION / 100 << "." << (UG_VERSION / 10) % 10 << "." << UG_VERSION % 10
            << " [GitHash: " << getGitHash() << "]" <<  std::endl;
   }

   ///
   /// destructor
   ///
   virtual ~ParaInitiator(
         )
   {
   }

   ///
   /// check if the execution is warm started (restarted) or not
   /// @return true if warm stated (restarted), false if it is not
   ///
   bool isWarmStarted(
         )
   {
      return prefixWarm != 0;
   }

   ///
   /// get prefix of warm start (restart) files
   /// @return prefix string
   ///
   const char *getPrefixWarm(
         )
   {
      return prefixWarm;
   }

   ///
   /// get communicator being used
   /// @return pointer to communicator object
   ///
   ParaComm  *getParaComm(
         )
   {
      return paraComm;
   }

   ///
   /// initialize initiator
   /// @return 0 if initialized normally, 1 if the problem is solved in init
   ///
   virtual int init(
         ParaParamSet *params,     ///< UG parameter used
         int argc,                 ///< the number of command line arguments
         char **argv               ///< array of the arguments
         ) = 0;

   ///
   /// reinitizalie initiator
   /// TODO: this function should be in inherited class
   /// @return 0 if reinitialized normally, 1 if the problem is solved in reinit
   ///
   virtual int reInit(
         int nRestartedRacing      ///< the number of restarted racing
         ) = 0;


   ///
   /// get instance object
   /// @return pointer to ParaInstance object
   ///
   virtual ParaInstance *getParaInstance(
         ) = 0;

   ///
   /// send solver initialization message
   ///
   virtual void sendSolverInitializationMessage(
         ) = 0;

   ///
   /// generate racing ramp-up parameter sets
   /// TODO: this function may be in inherited class
   ///
   virtual void generateRacingRampUpParameterSets(
         int nParamSets,                                    ///< number of parameter sets to be generated
         ParaRacingRampUpParamSet **racingRampUpParamSets   ///< array of the racing parameter sets
         ) = 0;

   ///
   /// get epsilon specified
   /// @return epsilon
   ///
   virtual double getEpsilon() = 0;

   ///
   /// write solution
   ///
   virtual void writeSolution(
         const std::string& message                        ///< message head string
         ) = 0;

   ///
   /// write ParaInstance
   ///
   virtual void writeParaInstance(
         const std::string& filename                      ///< output file name
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// write checkpoint solution
   ///
   virtual void writeCheckpointSolution(
         const std::string& filename                      ///< output file name
         ) = 0;

   ///
   /// read solution from checkpoint file
   /// @return objective function value of the solution
   ///
   virtual double readSolutionFromCheckpointFile(
         char *afterCheckpointingSolutionFileName        ///< name of after checkpointing solution file
         ) = 0;

#endif

   ///
   /// write solver runtime parameters
   ///
   virtual void writeSolverParameters(
         std::ostream *os                               ///< output stream to write solver parameters
         ) = 0;

   ///
   /// output final solver statistics
   ///
   virtual void outputFinalSolverStatistics(
         std::ostream *os,                              ///< output stream to write final solver statistics
         double time                                    ///< computing time
         ) = 0;

   ///
   /// get solving status string
   /// @return string to show solving status
   ///
   virtual std::string getStatus(
         ) = 0;

   ///
   /// print solver version
   ///
   virtual void printSolverVersion(
         std::ostream *os                                ///< output file (or NULL for standard output)
         ) = 0;

#ifdef UG_WITH_UGS

   ///
   /// read ugs incumbent solution **/
   ///
   virtual bool readUgsIncumbentSolution(
         UGS::UgsParaCommMpi *ugsComm,        ///< communicator used to communicate with ugs solvers
         int source                           ///< source ugs solver rank
         ) = 0;

   ///
   /// write ugs incumbent solution
   ///
   virtual void writeUgsIncumbentSolution(
         UGS::UgsParaCommMpi *ugsComm         ///< communicator used to communicate with ugs solvers
         ) = 0;

#endif

};

typedef ParaInitiator *ParaInitiatorPtr;

}

#endif // __PARA_INITIATOR_HPP__
