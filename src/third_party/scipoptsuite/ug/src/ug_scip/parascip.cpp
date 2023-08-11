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

/**@file    parascip.cpp
 * @brief   ParaSCIP MAIN.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cfloat>
#include "ug/paraInstance.h"
#include "ug/paraLoadCoordinator.h"
#include "ug/paraParamSet.h"
#include "ug/paraRacingRampUpParamSet.h"
#include "ug/paraInitiator.h"
#include "ug_bb/bbParaNodeMpi.h"
#include "ug/paraSysTimer.h"
#include "scip/scip.h"
#include "scipParaCommMpi.h"
#include "scipParaInstance.h"
#include "scipParaDeterministicTimer.h"
#include "scipParaSolver.h"
#include "scipParaInitiator.h"
#include "scipParaLoadCoordinator.h"

using namespace UG;
using namespace ParaSCIP;

extern void
setUserPlugins(ParaInitiator *initiator);
extern void
setUserPlugins(ParaInstance *instance);
extern void
setUserPlugins(ParaSolver *solver);

void
outputCommandLineMessages(
      char **argv
      )
{
   std::cout << std::endl;
   std::cout << "syntax: " << argv[0] << " #MPI_processes(#solvers + 1) parascip_param_file problem_file_name "
             << "[-l <logfile>] [-q] [-sl <settings>] [-s <settings>] [-sr <root_settings>] [-w <prefix_warm>] [-sth <number>] [-fsol <solution_file>] [-isol <initial solution file]" << std::endl;
   std::cout << "  -l <logfile>        : copy output into log file" << std::endl;
   std::cout << "  -q                  : suppress screen messages" << std::endl;
   std::cout << "  -sl <settings>      : load parameter settings (.set) file for LC presolving" << std::endl;
   std::cout << "  -s <settings>       : load parameter settings (.set) file for solvers" << std::endl;
   std::cout << "  -sr <root_settings> : load parameter settings (.set) file for root" << std::endl;
   std::cout << "  -w <prefix_warm>    : warm start file prefix ( prefix_warm_nodes.gz and prefix_warm_solution.txt are read )" << std::endl;
   std::cout << "  -fsol <solution file> : specify output solution file" << std::endl;
   std::cout << "  -isol <intial solution file> : specify initial solution file" << std::endl;
   std::cout << "File names need to be specified by full path form." << std::endl;
}

void
outputParaParamSet(
      ParaParamSet *paraParamSet,
      ParaInitiator *paraInitiator
      )
{
   if( !paraParamSet->getBoolParamValue(Quiet) )
   {
      std::ofstream ofsParamsOutputFile;
      std::ostringstream s;
      if( paraInitiator->getPrefixWarm() )
      {
         s << paraInitiator->getPrefixWarm();
      }
      else
      {
         s << paraParamSet->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName();
      }
      s << ".prm";
      ofsParamsOutputFile.open(s.str().c_str());
      if( !ofsParamsOutputFile ){
         std::cout << "Cannot open ParaParams output file: file name = " << s.str() << std::endl;
         exit(1);
      }
      paraParamSet->write(&ofsParamsOutputFile);
      ofsParamsOutputFile.close();
   }
}

void
outputSolverParams(
      ParaParamSet *paraParamSet,
      ParaInitiator *paraInitiator
      )
{
   if( !paraParamSet->getBoolParamValue(Quiet) )
   {
      std::ofstream ofsSolverParamsOutputFile;
      std::ostringstream s;
      if( paraInitiator->getPrefixWarm() )
      {
         s << paraInitiator->getPrefixWarm();
      }
      else
      {
         s << paraParamSet->getStringParamValue(LogSolvingStatusFilePath)
         << paraInitiator->getParaInstance()->getProbName();
      }
      s << "_solver.prm";
      ofsSolverParamsOutputFile.open(s.str().c_str());
      if( !ofsSolverParamsOutputFile ){
         std::cout << "Cannot open Solver parameters output file: file name = " << s.str() << std::endl;
         exit(1);
      }
      paraInitiator->writeSolverParameters(&ofsSolverParamsOutputFile);
      ofsSolverParamsOutputFile.close();
   }
}

/**************************************************************************************
 *                                                                                    *
 * Command line see outputCommandLineMessages()                                       *
 *                                                                                    *
 **************************************************************************************/
int
main (
      int  argc,
      char **argv
     )
{
   // mtrace();
   static const int solverOrigin = 1;

   bool racingSolversExist = false;
   ParaDeterministicTimer *detTimer = 0;

   ParaSysTimer sysTimer;
   sysTimer.start();

#ifdef UG_WITH_UGS
   char *configFileName = 0;
   for( int i = 1; i < argc; ++i )
   {
      if ( strcmp(argv[i], "-ugsc") == 0 )
      {
         i++;
         if( i < argc )
         {
            configFileName = argv[i];
            break;
         }
         else
         {
            std::cerr << "missing file name after parameter '-ugsc" << std::endl;
            exit(1);
         }
      }
   }

   ScipParaCommMpi *comm = 0;
   UGS::UgsParaCommMpi *commUgs = 0;   // commUgs != 0 means ParaScip runs under UGS
   if( configFileName )
   {
      commUgs = new UGS::UgsParaCommMpi();
      commUgs->init(argc,argv);
      comm = new ScipParaCommMpi(commUgs->getSolverMPIComm());
      comm->setUgsComm(commUgs);
   }
   else
   {
      comm = new ScipParaCommMpi();
   }
#else
   // ParaComm *comm = new PARA_COMM_TYPE();
   ScipParaCommMpi *comm = new ScipParaCommMpi();
#endif

   comm->init(argc,argv);

   ParaTimer *paraTimer = new ParaTimerMpi();
   paraTimer->init(comm);

   ParaParamSet *paraParamSet = comm->createParaParamSet();

#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
   SCIP_CALL_ABORT( SCIPcreateMesshdlrPThreads(1) );
   SCIPmessageSetDefaultHandler();
#endif

   if( comm->getRank() == 0 )
   {
      if( argc < 3 )
      {
         outputCommandLineMessages(argv);
         return 1;
      }
      paraParamSet->read(comm, argv[1]);

      paraParamSet->bcast(comm, 0);
      comm->lcInit(paraParamSet);
      ParaInitiator *paraInitiator = new ScipParaInitiator(comm, paraTimer);
      setUserPlugins(paraInitiator);
      if( paraInitiator->init(paraParamSet, argc, argv) )
      {
         if( dynamic_cast<ScipParaInitiator *>(paraInitiator)->isSolvedAtInit() )
         {
            paraInitiator->outputFinalSolverStatistics(0, paraTimer->getElapsedTime());
            delete paraInitiator;
            delete paraParamSet;
            delete paraTimer;
            if( detTimer ) delete detTimer;
//            sysTimer.stop();
//            std::cout << "[ Rank: " << comm->getRank() << " ], UTime = " << sysTimer.getUTime()
//                  << ", STime = " << sysTimer.getSTime() << ", RTime = " << sysTimer.getRTime() << std::endl;
            comm->abort();
            delete comm;
#ifdef UG_WITH_UGS
            if( commUgs ) delete commUgs;
#endif
            return 0;
         }
      }
      std::cout << "** Initiatior was initilized after " << paraTimer->getElapsedTime() << " sec." << std::endl;
      ParaInstance *paraInstance = paraInitiator->getParaInstance();
      if( paraParamSet->getIntParamValue(OutputParaParams) > 0 )
      {
         outputParaParamSet(paraParamSet, paraInitiator);
         outputSolverParams(paraParamSet, paraInitiator);
      }
      paraInstance->bcast(comm, 0, paraParamSet->getIntParamValue(InstanceTransferMethod) );
      paraInitiator->sendSolverInitializationMessage();  // This messages should be received in constructor of the target Solver
      std::cout << "** Instance data were sent to all solvers after " << paraTimer->getElapsedTime() << " sec." << std::endl;
      ParaLoadCoordinator *paraLc;
      if( paraParamSet->getBoolParamValue(Deterministic) )
      {
          detTimer = new ScipParaDeterministicTimer();
      }
      if( dynamic_cast<ScipParaInitiator *>(paraInitiator)->isSolvedAtInit() )
      {
#ifdef UG_WITH_UGS
         paraLc = new ScipParaLoadCoordinator(commUgs, comm, paraParamSet, paraInitiator, &racingSolversExist, paraTimer, detTimer);
#else
         paraLc = new ScipParaLoadCoordinator(comm, paraParamSet, paraInitiator, &racingSolversExist, paraTimer, detTimer);
#endif
         delete paraLc;
         delete paraInitiator;
         delete paraParamSet;
         delete paraTimer;
         if( detTimer ) delete detTimer;

//         sysTimer.stop();
//         std::cout << "[ Rank: " << comm->getRank() << " ], UTime = " << sysTimer.getUTime()
//               << ", STime = " << sysTimer.getSTime() << ", RTime = " << sysTimer.getRTime() << std::endl;

         comm->abort();
         delete comm;
#ifdef UG_WITH_UGS
         if( commUgs ) delete commUgs;
#endif
         return 0;
      }
      else
      {
#ifdef UG_WITH_UGS
         paraLc = new ScipParaLoadCoordinator(commUgs, comm, paraParamSet, paraInitiator, &racingSolversExist, paraTimer, detTimer);
#else
         paraLc = new ScipParaLoadCoordinator(comm, paraParamSet, paraInitiator, &racingSolversExist, paraTimer, detTimer);
#endif
      }
      if( paraInitiator->isWarmStarted() )
      {
#ifdef UG_WITH_ZLIB
         paraLc->warmStart();
#endif
      }
      else
      {
         if( paraParamSet->getIntParamValue(RampUpPhaseProcess) == 0 ||
               paraParamSet->getIntParamValue(RampUpPhaseProcess) == 3 )
         {
            ParaTask *rootNode = new BbParaNodeMpi(
                  TaskId(), TaskId(), 0, -DBL_MAX, -DBL_MAX, -DBL_MAX,
                  dynamic_cast<ScipParaInitiator *>(paraInitiator)->makeRootNodeDiffSubproblem());
            paraLc->run(rootNode);
         }
         else if( paraParamSet->getIntParamValue(RampUpPhaseProcess) >= 1 &&
                  paraParamSet->getIntParamValue(RampUpPhaseProcess) <= 2 )  // racing ramp-up
         {
            ParaRacingRampUpParamSet **racingRampUpParams = new ParaRacingRampUpParamSetPtr[comm->getSize()];
            paraInitiator->generateRacingRampUpParameterSets( (comm->getSize()-1), racingRampUpParams );
            for( int i = 1; i < comm->getSize(); i++ )
            {
               int noKeep = 0;
               PARA_COMM_CALL(
                     comm->send( &noKeep, 1, ParaINT, i, UG::TagKeepRacing)
                     );
               PARA_COMM_CALL(
                     racingRampUpParams[i-solverOrigin]->send(comm, i)
                     );
            }
            ParaTask *rootNode = new BbParaNodeMpi(
                  TaskId(), TaskId(), 0, -DBL_MAX, -DBL_MAX, -DBL_MAX,
                  dynamic_cast<ScipParaInitiator *>(paraInitiator)->makeRootNodeDiffSubproblem());
            paraLc->run(rootNode, (comm->getSize()-1), racingRampUpParams );
            for( int i = 1; i < comm->getSize(); i++ )
            {
               if( racingRampUpParams[i-solverOrigin] ) delete racingRampUpParams[i-solverOrigin];
            }
            delete [] racingRampUpParams;
         }
         else
         {
            THROW_LOGICAL_ERROR2("Invalid RampUpPhaseProcess: ", paraParamSet->getIntParamValue(RampUpPhaseProcess) )
         }
      }
      delete paraLc;
      if( paraInitiator ) delete paraInitiator;
   }
   else
   {
      if( argc < 3 )
      {
         return 1;
      }
      paraParamSet->bcast(comm, 0);
      comm->solverInit(paraParamSet);
      int nNonLinearConsHdlrs = 0;
      PARA_COMM_CALL(
            comm->bcast( &nNonLinearConsHdlrs, 1, ParaINT, 0 )
      );
      ParaInstance *paraInstance = comm->createParaInstance();
      setUserPlugins(paraInstance);
      if( nNonLinearConsHdlrs > 0 )
      {
         paraParamSet->setIntParamValue(InstanceTransferMethod,2);
      }
      if( paraParamSet->getIntParamValue(InstanceTransferMethod) == 2 )
      {
         ScipParaInstanceMpi *scipParaInstanceMpi = dynamic_cast<ScipParaInstanceMpi *>(paraInstance);
         scipParaInstanceMpi->setFileName(argv[2]);
      }
      paraInstance->bcast(comm, 0, paraParamSet->getIntParamValue(InstanceTransferMethod) );
      if( paraParamSet->getBoolParamValue(Deterministic) )
      {
          detTimer = new ScipParaDeterministicTimer();
      }
      ParaSolver *paraSolver = new ScipParaSolver(argc, argv, comm, paraParamSet, paraInstance, detTimer);
      // setUserPlugins(paraSolver);   too late!! user plugin is necessary when instance data are read
      dynamic_cast<ScipParaSolver *>(paraSolver)->setProblemFileName(argv[2]);
      // if( paraParamSet->getIntParamValue(RampUpPhaseProcess) == 0 || paraSolver->isWarmStarted() )
      if( paraParamSet->getIntParamValue(RampUpPhaseProcess) == 0 ||
            paraParamSet->getIntParamValue(RampUpPhaseProcess) == 3 )
      {
         paraSolver->run();
      }
      else if( paraParamSet->getIntParamValue(RampUpPhaseProcess) >= 1 &&
               paraParamSet->getIntParamValue(RampUpPhaseProcess) <= 2 ) // racing ramp-up
      {
         int source;
         int tag;
         (void)comm->probe(&source, &tag);
         if( tag == TagKeepRacing )
         {
            int keep = 0;
            PARA_COMM_CALL(
                  comm->receive( &keep, 1, ParaINT, 0, UG::TagKeepRacing)
                  );
            if( keep == 0 )
            {
               dynamic_cast<BbParaSolver *>(paraSolver)->setKeepRacing(false);
            }
            else
            {
               dynamic_cast<BbParaSolver *>(paraSolver)->setKeepRacing(true);
            }
            (void)comm->probe(&source, &tag);
            ParaRacingRampUpParamSet *racingRampUpParamSet = new ScipParaRacingRampUpParamSetMpi();
            PARA_COMM_CALL(
                  racingRampUpParamSet->receive(comm, 0)
                  );
            paraSolver->run( racingRampUpParamSet );
            // delete racingRampUpParamSet;   // racingRampUpParamSet is set in Solver object and deleted in the object
         }
         else
         {
            if( tag == TagTerminateRequest )
            {
               PARA_COMM_CALL(
                     comm->receive( NULL, 0, ParaBYTE, source, TagTerminateRequest )
                     );
               // when solver is deleted, solver's destructor sends termination status
            }
            else
            {
               THROW_LOGICAL_ERROR2("Invalid Tag is recicv3ed in ParaSCIP solver main: ", tag )
            }
         }
      }
      else
      {
         THROW_LOGICAL_ERROR2("Invalid RampUpPhaseProcess: ", paraParamSet->getIntParamValue(RampUpPhaseProcess) )
      }
      delete paraSolver;
      //if( paraInstance ) delete paraInstance;  /** deleted in paraSolver destructor */
   }

   delete paraParamSet;

//   sysTimer.stop();
//   std::cout << "[ Rank: " << comm->getRank() << " ], UTime = " << sysTimer.getUTime()
//         << ", STime = " << sysTimer.getSTime() << ", RTime = " << sysTimer.getRTime() << std::endl;


#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
   SCIPfreeMesshdlrPThreads();
#endif

   delete paraTimer;
   if( detTimer ) delete detTimer;

   if( racingSolversExist ) comm->abort();

   delete comm;

#ifdef UG_WITH_UGS
   if( commUgs ) delete  commUgs;
#endif

   return 0;

} /* END main */
