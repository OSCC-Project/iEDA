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

/**@file    scipParaInitiator.cpp
 * @brief   ParaInitiator extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

// #define UG_SCIP_SOL_FEASIBILITY_CHECK_IN_LC

#include <cctype>
#include <sstream>
#include "scipParaInstance.h"
#include "scipParaInitiator.h"
#include "scipParaObjMessageHdlr.h"
#include "scipParaInitialStat.h"
#include "scipParaParamSet.h"
#ifdef UG_DEBUG_SOLUTION
#ifndef WITH_DEBUG_SOLUTION
#define WITH_DEBUG_SOLUTION
#endif
#include "scip/debug.h"
#endif
#ifdef  __linux__
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#endif

// #define UG_SCIP_SOL_FEASIBILITY_CHECK_IN_LC

using namespace UG;
using namespace ParaSCIP;

#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
extern long long virtualMemUsedAtLc;
extern double memoryLimitOfSolverSCIP;
#ifdef  __linux__
static int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

static long long getVmSize(){
    FILE* file = fopen("/proc/self/status", "r");
    long long result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result*1024;  // result value is in KB
}
#endif
#endif

bool
ScipParaInitiator::addRootNodeCuts(
      )
{
   SCIP_Longint originalLimitsNodes;
   SCIP_CALL_ABORT( SCIPgetLongintParam(scip, "limits/nodes", &originalLimitsNodes) );
   SCIP_CALL_ABORT( SCIPsetLongintParam(scip, "limits/nodes", 1) );
   if( scipDiffParamSetRoot ) scipDiffParamSetRoot->setParametersInScip(scip);
   if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
   {
      double timeRemains =  std::max( 0.0, (paraParams->getRealParamValue(UG::TimeLimit) - timer->getElapsedTime()) );
      // SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
   }
   SCIP_RETCODE ret = SCIPsolve(scip);
   if( ret != SCIP_OKAY )
   {
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIPprintError(ret, NULL);
#else
      SCIPprintError(ret);
#endif
      abort();
   }

   /* reset LC parameter settings */
   SCIP_CALL_ABORT( SCIPresetParams(scip) );
   if( paraParams->getBoolParamValue(NoPreprocessingInLC) )
   {
      SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrounds", 0));
   }
   else
   {
      if( settingsNameLC )
      {
         SCIP_CALL_ABORT( SCIPreadParams(scip, settingsNameLC) );
      }
   }

   /* don't catch control+c */
   SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );
   // Then, solver status should be checked
   SCIP_STATUS status = SCIPgetStatus(scip);
   if( status == SCIP_STATUS_OPTIMAL )   // when sub-MIP is solved at root node, the solution may not be saved
   {
      return false;
   }
   else
   {
      if( status == SCIP_STATUS_MEMLIMIT  )
      {
         std::cout << "Warning: SCIP was interrupted because the memory limit was reached" << std::endl;
         return false;
      }
      if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 &&
            timer->getElapsedTime() > paraParams->getRealParamValue(UG::TimeLimit) )
      {
         return true; // pretended to add cuts, anyway, timelimit.
      }
   }

   SCIP_CUT** cuts;
   int ncuts;
   int ncutsadded;

   ncutsadded = 0;
   cuts = SCIPgetPoolCuts(scip);
   ncuts = SCIPgetNPoolCuts(scip);
   for( int c = 0; c < ncuts; ++c )
   {
      SCIP_ROW* row;

      row = SCIPcutGetRow(cuts[c]);
      assert(!SCIProwIsLocal(row));
      assert(!SCIProwIsModifiable(row));
      if( SCIPcutGetAge(cuts[c]) == 0 && SCIProwIsInLP(row) )
      {
         char name[SCIP_MAXSTRLEN];
         SCIP_CONS* cons;
         SCIP_COL** cols;
         SCIP_VAR** vars;
         int ncols;
         int i;

         /* create a linear constraint out of the cut */
         cols = SCIProwGetCols(row);
         ncols = SCIProwGetNNonz(row);

         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, ncols) );
         for( i = 0; i < ncols; ++i )
            vars[i] = SCIPcolGetVar(cols[i]);

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_%d", SCIProwGetName(row), SCIPgetNRuns(scip));
         SCIP_CALL_ABORT( SCIPcreateConsLinear(scip, &cons, name, ncols, vars, SCIProwGetVals(row),
               SCIProwGetLhs(row) - SCIProwGetConstant(row), SCIProwGetRhs(row) - SCIProwGetConstant(row),
               TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE) );
         SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
         SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );

         SCIPfreeBufferArray(scip, &vars);

         ncutsadded++;
      }
   }

   // SCIP_CALL_ABORT( SCIPsetLongintParam(scip, "limits/nodes", originalLimitsNodes) );

   return true;
}

/** init function */
int
ScipParaInitiator::init(
      ParaParamSet *inParaParams,
      int     argc,
      char**  argv
      )
{
   int i;
   bool quiet = false;
#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
   bool noUpgrade = false;
#endif
   paraParams = inParaParams;

   probname = argv[2];

   /********************
    * Parse parameters *
    ********************/
   if( std::string(paraParams->getStringParamValue(SolverSettingsForInitialPresolving)) != "" )
   {
      settingsNameLC = const_cast<char*> (paraParams->getStringParamValue(SolverSettingsForInitialPresolving));
   }
   if( std::string(paraParams->getStringParamValue(SolverSettingsAtRootNode)) != "" )
   {
      settingsNameRoot = const_cast<char*> (paraParams->getStringParamValue(SolverSettingsAtRootNode));
   }
   if( std::string(paraParams->getStringParamValue(SolverSettingsExceptRootNode)) != "" )
   {
      settingsName = const_cast<char*> (paraParams->getStringParamValue(SolverSettingsExceptRootNode));
   }
   if( std::string(paraParams->getStringParamValue(SolverSettingsAtRacing)) != "" )
   {
      racingSettingsName = const_cast<char*> (paraParams->getStringParamValue(SolverSettingsAtRacing));
   }
   for( i = 3; i < argc; ++i )   /** the first argument is runtime parameter file for ParaSCIP */
                                 /** the second argument is problem file name */
   {
      if( strcmp(argv[i], "-l") == 0 )
      {
         i++;
         if( i < argc )
            logname = argv[i];
         else
         {
            std::cerr << "missing log filename after parameter '-l'" << std::endl;
            exit(1);
         }
      }
      else if( strcmp(argv[i], "-q") == 0 )
         quiet = true;
      else if( strcmp(argv[i], "-s") == 0 )
      {
         i++;
         if( i < argc )
            settingsName = argv[i];
         else
         {
            std::cerr << "missing settings filename after parameter '-s'" << std::endl;
            exit(1);
         }
      }
      else if( strcmp(argv[i], "-sr") == 0 )
      {
         i++;
         if( i < argc )
            settingsNameRoot = argv[i];
         else
         {
            std::cerr << "missing settings filename after parameter '-sr'" << std::endl;
            exit(1);
         }
      }
      else if( strcmp(argv[i], "-sl") == 0 )
      {
         i++;
         if( i < argc )
            settingsNameLC = argv[i];
         else
         {
            std::cerr << "missing settings filename after parameter '-sl'" << std::endl;
            exit(1);
         }
      }
      else if( strcmp(argv[i], "-w") == 0)
      {
#ifdef UG_WITH_ZLIB
         i++;
         if( i < argc )
         {
            prefixWarm = argv[i];
            char nodesFileName[256];
            sprintf(nodesFileName,"%s_nodes_LC0.gz",prefixWarm);
            checkpointTasksStream.open(nodesFileName, std::ios::in | std::ios::binary);
            if( !checkpointTasksStream.good() ){
                std::cerr << "ERROR: Opening file `" << nodesFileName << "' failed.\n";
                exit(1);
            }
         }
         else
         {
            std::cerr << "missing settings filename after parameter '-w'" << std::endl;
            exit(1);
         }
#else
         std::cerr << "Cannot work with parameter '-w' compiling without zlib" << std::endl;
         exit(1);
#endif
      }
      else if( strcmp(argv[i], "-racing") == 0 )
      {
         i++;
         if( i < argc )
         {
            racingSettingsName = argv[i];
         }
         else
         {
            std::cerr << "missing settings filename after parameter '-racing'" << std::endl;
            exit(1);
         }
      }
      else if ( strcmp(argv[i], "-isol") == 0 )
      {
         i++;
         if( i < argc )
         {
            isolname = argv[i];
          }
          else
          {
             std::cerr << "missing settings filename after parameter '-isol'" << std::endl;
             exit(1);
          }
      }
      else if( strcmp(argv[i], "-objlimit") == 0 )
      {
         i++;
         if( i < argc )
         {
            objlimit = atof(argv[i]);
          }
          else
          {
             std::cerr << "missing objective limit after parameter '-objlimit'" << std::endl;
             exit(1);
          }
      }
      else if ( strcmp(argv[i], "-sth") == 0 )
      {
         i++;  // just omit this parameter and the following number.
      }
      else if ( strcmp(argv[i], "-fsol" ) == 0 )
      {
         i++;
         if( i < argc )
         {
            solutionFileName = argv[i];
          }
          else
          {
             std::cerr << "missing solution filename after parameter '-fsol'" << std::endl;
             exit(1);
          }
      }
#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
      else if ( strcmp(argv[i], "-nou" ) == 0 )
      {
         noUpgrade = true;
      }
#endif
#ifdef UG_WITH_UGS
      else if( strcmp(argv[i], "-ugsc") == 0 )
      {
         i++;
      }
#endif
      else
      {
         THROW_LOGICAL_ERROR3("invalid parameter <", argv[i], ">");
      }
   }

   /*********
    * Setup *
    *********/
   /* initialize SCIP */
   SCIP_CALL( SCIPcreate(&scip) );

   /********************
    * Setup clock type *
    ********************/
   SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) ); // always use wall clock time
   if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
   {
      double timeRemains = std::max( 0.0, (paraParams->getRealParamValue(UG::TimeLimit) - timer->getElapsedTime()) );
      SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
   }

   /*******************
    * Install plugins *
    *******************/
   /* include default SCIP plugins */
   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
   /** user include plugins */
   includeUserPlugins(scip);  // user plugin must set later, since it also sets user parameters
                              // We should include here

   /* initialize finalDual bound */
   finalDualBound = -SCIPinfinity(scip);
   /* output solver version */
   printSolverVersion(NULL);

   /* Make sure that copying of symmetry handling constraints works. This is a workaround: Symmetry constraints are
    * usually not copied, but then we cannot proceed here. Thus, copying is forced. This is correct, but slows down the
    * sequential SCIP version a little. This solution is here until a better solution has been found. */
   SCIP_RETCODE paramretcode = SCIPsetBoolParam(scip, "constraints/orbisack/forceconscopy", TRUE);
   if( paramretcode != SCIP_OKAY && paramretcode != SCIP_PARAMETERUNKNOWN )
   {
      SCIP_CALL_ABORT( paramretcode );
   }
   paramretcode = SCIPsetBoolParam(scip, "constraints/orbitope/forceconscopy", TRUE);
   if( paramretcode != SCIP_OKAY && paramretcode != SCIP_PARAMETERUNKNOWN )
   {
      SCIP_CALL_ABORT( paramretcode );
   }
   paramretcode = SCIPsetBoolParam(scip, "constraints/symresack/forceconscopy", TRUE);
   if( paramretcode != SCIP_OKAY && paramretcode != SCIP_PARAMETERUNKNOWN )
   {
      SCIP_CALL_ABORT( paramretcode );
   }

   /*************************************************
    * set quiet message handler, if it is necessary *
    *************************************************/
   messagehdlr = NULL;
   if( paraParams->getBoolParamValue(Quiet) )
   {
      ScipParaObjMessageHdlr* objmessagehdlr = new ScipParaObjMessageHdlr(paraComm, NULL, TRUE, FALSE);
      SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, objmessagehdlr, TRUE) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
      SCIP_CALL_ABORT( SCIPsetMessagehdlr(scip, messagehdlr) );
      SCIP_CALL_ABORT( SCIPmessagehdlrRelease(&messagehdlr));
#endif
      SCIPmessageSetErrorPrinting(ParaSCIP::scip_errorfunction, (void*) objmessagehdlr);
   }
   else
   {
      if( logname != NULL || quiet  )
      {
         if( logname != NULL )
         {
            std::ostringstream os;
            os << logname << paraComm->getRank();
            logfile = fopen(os.str().c_str(), "a"); // append to log file */
            if( logfile == NULL )
            {
               THROW_LOGICAL_ERROR3("cannot open log file <", logname, "> for writing");
            }
         }

         ScipParaObjMessageHdlr* objmessagehdlr = new ScipParaObjMessageHdlr(paraComm, logfile, quiet, FALSE);
         SCIP_CALL_ABORT( SCIPcreateObjMessagehdlr(&messagehdlr, objmessagehdlr, TRUE) );
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(messagehdlr) );
#else
         SCIP_CALL_ABORT( SCIPsetMessagehdlr(scip, messagehdlr) );
         SCIP_CALL_ABORT( SCIPmessagehdlrRelease(&messagehdlr));
#endif
         SCIPmessageSetErrorPrinting(ParaSCIP::scip_errorfunction, (void*) objmessagehdlr);
      }
   }

   if( probname != NULL )
   {
      /***********************
       * Version information *
       ***********************/
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      SCIPprintVersion(NULL);
#else
      SCIPprintVersion(scip, NULL);
#endif
      SCIPinfoMessage(scip, NULL, "\n");
      /*****************
       * Load settings *
       *****************/
      DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
      SCIP *paramScip = 0;
      if( !(settingsName == NULL && settingsNameRoot == NULL && settingsNameLC == NULL ) )
      {
         /* initialize SCIP to get diff params */
         SCIP_CALL( SCIPcreate(&paramScip) );
         /* include default SCIP plugins */
         SCIP_CALL( SCIPincludeDefaultPlugins(paramScip) );
         /** user include plugins */
         includeUserPlugins(paramScip);   // need to install user parameters
      }

      if( settingsName != NULL )
      {
         SCIP_CALL_ABORT( SCIPreadParams(paramScip, settingsName) );
         scipDiffParamSet =  scipParaComm->createScipDiffParamSet(paramScip);
      }
      else
      {
         scipDiffParamSet = scipParaComm->createScipDiffParamSet(scip);
      }

      if( settingsNameRoot != NULL )
      {
         SCIP_CALL_ABORT( SCIPreadParams(scip, settingsNameRoot) );
         // SCIP_CALL( SCIPresetParams(paramScip) );
         SCIP_CALL_ABORT( SCIPreadParams(paramScip, settingsNameRoot) );
         scipDiffParamSetRoot = scipParaComm->createScipDiffParamSet(paramScip);
         // Root settings are used for LC. They should be a part of root process.
         // SCIP_CALL( SCIPresetParams(scip) );
      }
      else
      {
         scipDiffParamSetRoot = scipParaComm->createScipDiffParamSet(scip);
      }

      if( settingsNameLC != NULL )
      {
         // SCIP_CALL( SCIPresetParams(scip) );
         SCIP_CALL_ABORT( SCIPreadParams(scip, settingsNameLC) );
      }


      if( paramScip )
      {
         SCIP_CALL_ABORT( SCIPfree(&paramScip) );
         paramScip = 0;
      }


      /**************
       * Start SCIP *
       **************/
      /** user include plugins */
      // includeUserPlugins(scip);  // need to set user plugins: should be set

      // Problem Creation

      SCIP_RETCODE retcode = SCIPreadProb(scip, probname, NULL);
      if( retcode != SCIP_OKAY )
      {
         std::cout << "error reading file <" << probname << ">" << std::endl;
         SCIP_CALL( SCIPfreeProb(scip) );
         exit(1);
      }

      /* transform the problem */
      SCIP_CALL_ABORT( SCIPtransformProb(scip));

      /* read initail solution, if it is specified */
      if( isolname )
      {
         // NOTE:
         // When CPLEX license file cannot find, SCIPtransformProb(scip) may fail
         // SCIP_CALL_ABORT( SCIPreadSol(scip, isolname) );
         SCIP_CALL_ABORT( SCIPreadProb(scip, isolname, 0) );
      }

      /* change problem name */
      char *probNameFromFileName;
      char *temp = new char[strlen(probname)+1];
      (void) strcpy(temp, probname);
      SCIPsplitFilename(temp, NULL, &probNameFromFileName, NULL, NULL);
      SCIP_CALL_ABORT( SCIPsetProbName(scip, probNameFromFileName));
      delete [] temp;

      /* presolve problem */
      if( paraParams->getBoolParamValue(NoPreprocessingInLC) )
      {
         if( SCIPfindConshdlr(scip, "pseudoboolean") == NULL || SCIPconshdlrGetNConss(SCIPfindConshdlr(scip, "pseudoboolean")) == 0 ) // this is workaround for a bug.
         {
            SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrounds", 0) );
            std::cout << "No LC presolving is specified." << std::endl;
         }
         else
         {
            std::cout << "Default LC presolving (default)." << std::endl;
         }
      }
      else
      {
         if( settingsNameLC )
         {
            SCIP_CALL_ABORT( SCIPreadParams(scip, settingsNameLC) );
            std::cout << "LC presolving settings file is specified." << std::endl;
         }
         else
         {
            // SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_FAST, TRUE) );
            std::cout << "Default LC presolving (default)." << std::endl;
         }
      }
      // SCIP_CALL_ABORT( SCIPsetIntParam(scip, "constraints/quadratic/replacebinaryprod", 0));
      // SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/nonlinear/reformulate", FALSE));

      /* don't catch control+c */
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );

      /* objlimit is specified */
      if( EPSLT( objlimit, DBL_MAX, MINEPSILON ) )
      {
         SCIP_CALL_ABORT( SCIPsetObjlimit(scip, objlimit) );
      }


// #ifdef _COMM_PTH
//      SCIP_CALL( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );
// #endif

      SCIP_Bool originalUpgradeKnapsack;
      SCIP_Bool originalUpgradeLogicor;
      SCIP_Bool originalUpgradeSetppc;
      SCIP_Bool originalUpgradeVarbound;
      bool onlyLinearConss = onlyLinearConsHandler();
#ifdef _COMM_MPI_WORLD
      if( onlyLinearConss )
      {
         std::cout << "** Original problem has only linear constraints" << std::endl;
      }
      else
      {
         std::cout << "** Original problem has non-linear constraints" << std::endl;
      }
      if( paraParams->getIntParamValue(UG::InstanceTransferMethod) != 2 )
      {
         if( onlyLinearConss )
         {
            SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/knapsack", &originalUpgradeKnapsack));
            SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/logicor", &originalUpgradeLogicor));
            SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/setppc", &originalUpgradeSetppc));
            SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/varbound", &originalUpgradeVarbound));
            SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", FALSE));
            SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", FALSE));
            SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", FALSE));
            SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", FALSE));
         }
      }
#else
      if( noUpgrade )
      {
         SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/knapsack", &originalUpgradeKnapsack));
         SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/logicor", &originalUpgradeLogicor));
         SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/setppc", &originalUpgradeSetppc));
         SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/varbound", &originalUpgradeVarbound));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", FALSE));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", FALSE));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", FALSE));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", FALSE));
      }
#endif

#ifdef UG_DEBUG_SOLUTION
      SCIPdebugSolDisable(scip);
#endif

      // instance = new PARA_INSTANCE_TYPE(scip, paraParams->getIntParamValue(InstanceTransferMethod));
      /** instance needs to be generated befor presolving **/
      instance = scipParaComm->createScipParaInstance(scip, paraParams->getIntParamValue(InstanceTransferMethod));

      std::ostringstream os;
      if( solutionFileName )
      {
         os << solutionFileName;
      }
      else
      {
         os << paraParams->getStringParamValue(SolutionFilePath);
         os << instance->getProbName() << ".sol";
      }
      solutionFile = fopen(os.str().c_str(), "a");  // if solution file exists, append
      if( solutionFile == NULL )
      {
         THROW_LOGICAL_ERROR3("cannot open solution file <", os.str(), "> for writing");
      }

#if defined(_COMM_PTH) || defined (_COMM_CPP11)
#ifdef  __linux__
      long long vmSizeBeforePresolving = getVmSize(); 
      virtualMemUsedAtLc = std::max(vmSizeBeforePresolving, (SCIPgetMemTotal(scip) + SCIPgetMemExternEstim(scip)));
      std::cout << "** Before presolving: virtualMemUsedAtLc = " << virtualMemUsedAtLc << ", getVmSize() = " << getVmSize() << ", SCIPgetMemUsed() = " << SCIPgetMemUsed(scip) << ", SCIPgetMemTotal() = " << SCIPgetMemTotal(scip) << ", SCIPgetMemExternEstim() = " << SCIPgetMemExternEstim(scip) << std::endl;
#else
      virtualMemUsedAtLc = SCIPgetMemTotal(scip) + SCIPgetMemExternEstim(scip);
      std::cout << "** Before presolving: virtualMemUsedAtLc = " << virtualMemUsedAtLc << ", SCIPgetMemUsed() = " << SCIPgetMemUsed(scip) << ", SCIPgetMemTotal() = " << SCIPgetMemTotal(scip) << ", SCIPgetMemExternEstim() = " << SCIPgetMemExternEstim(scip) << std::endl;
#endif
      // if( (dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - (virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR))/(paraComm->getSize()-1) > virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR ) 
      if( dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) > virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR*paraComm->getSize() )
      {
         // SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", (dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - (virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR))/(paraComm->getSize()-1))); // LC has SCIP env.
         SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR*paraComm->getSize()) ); // LC has SCIP env.
      }
      else
      {
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrounds", 0) );
         std::cout << "** No LC presolving is applied, since memory limit becomes " 
           // << (dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - (virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR))/(paraComm->getSize()-1) 
           << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)
           << " < "
           // << virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR 
           << virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR*paraComm->getSize()
           << std::endl;
         SCIP_CALL( SCIPpresolve(scip) );
         solvedAtInit = true;
         setFinalSolverStatus(MemoryLimitIsReached);
         // finalDualBound = SCIPgetDualbound(scip);
         std::cout << "=== solved at Init ===" << std::endl;
         finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
         writeSolution("Final Solution");
         return 1;
      }
      // std::cout << "** set memory limit for presolving in LC to " << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)/(paraComm->getSize()*SCIP_MEMORY_COPY_FACTOR) << " for SCIP **" << std::endl;
      std::cout << "** set memory limit for presolving in LC to " << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - virtualMemUsedAtLc*SCIP_MEMORY_COPY_FACTOR*paraComm->getSize() << " for SCIP **" << std::endl;
#else
      SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/memory", dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)) ); 
      std::cout << "** set memory limit for presolving in LC and solvers to " << dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) << " for each SCIP **" << std::endl;
#endif

#if SCIP_APIVERSION >= 101
      if( SCIPfindPresol(scip, "milp") != NULL )
      {
#ifdef _COMM_MPI_WORLD
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/milp/threads", 0) );
#else
         SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/milp/threads", (paraComm->getSize() - 1)) );
#endif
      }
#endif

      if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 )
      {
         double timeRemains = std::max( 0.0, (paraParams->getRealParamValue(UG::TimeLimit) - timer->getElapsedTime()) );
         SCIP_CALL_ABORT( SCIPsetIntParam(scip,"timing/clocktype", 2) );         // to confirm this for the scip environment actually to work
         SCIP_CALL_ABORT( SCIPsetRealParam(scip,"limits/time", timeRemains) );
      }

      SCIP_CALL( SCIPpresolve(scip) );
      SCIP_STATUS scipStatus = SCIPgetStatus(scip);

      if( scipStatus == SCIP_STATUS_OPTIMAL || 
          scipStatus == SCIP_STATUS_INFEASIBLE )   // when sub-MIP is solved at root node, the solution may not be saved
      {
         solvedAtInit = true;
         setFinalSolverStatus(ProblemWasSolved);
         // finalDualBound = SCIPgetDualbound(scip);
         std::cout << "=== solved at Init ===" << std::endl;
         finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
         writeSolution("Final Solution");
         return 1;
      }
      else if( scipStatus == SCIP_STATUS_MEMLIMIT )
      {
         solvedAtInit = true;
         setFinalSolverStatus(MemoryLimitIsReached);
         // finalDualBound = SCIPgetDualbound(scip);
         std::cout << "=== solved at Init ===" << std::endl;
         finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
         writeSolution("Final Solution");
         return 1;
      }
      else
      {
         if( scipStatus == SCIP_STATUS_TIMELIMIT )
         {
            solvedAtInit = true;
            setFinalSolverStatus(HardTimeLimitIsReached);
            // finalDualBound = SCIPgetDualbound(scip);
            std::cout << "=== solved at Init ===" << std::endl;
            finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
            writeSolution("Final Solution");
            return 1;
         }
         else
         {
            /* adding root node cuts, if necessary */
            if( paraParams->getBoolParamValue(UseRootNodeCuts) )
            {
               if( !addRootNodeCuts() )
               {
                  solvedAtInit = true;
                  setFinalSolverStatus(ProblemWasSolved);
                  // finalDualBound = SCIPgetDualbound(scip);
                  std::cout << "=== solved at Init ===" << std::endl;
                  finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
                  writeSolution("Final Solution");
                  return 1;
               }
               else
               {
                  if( paraParams->getRealParamValue(UG::TimeLimit) > 0.0 &&
                        timer->getElapsedTime() > paraParams->getRealParamValue(UG::TimeLimit) )
                  {
                     solvedAtInit = true;
                     setFinalSolverStatus(HardTimeLimitIsReached);
                     // finalDualBound = SCIPgetDualbound(scip);
                     std::cout << "=== solved at Init ===" << std::endl;
                     finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
                     writeSolution("Final Solution");
                     return 1;
                  }
               }
            }
         }
      }

#if defined(_COMM_PTH) || defined (_COMM_CPP11)
#ifdef  __linux__
      // int nCores = sysconf(_SC_NPROCESSORS_CONF);
      long long vmSize = getVmSize();
      long long vmSizeForSolver = (vmSize - vmSizeBeforePresolving)/paraComm->getSize() + vmSizeBeforePresolving;
      virtualMemUsedAtLc = std::max(vmSizeForSolver, (SCIPgetMemTotal(scip) + SCIPgetMemExternEstim(scip)))/SCIP_PRESOLVIG_MEMORY_FACTOR; /// in genral, it looks over estimated
      std::cout << "** Estimated virtualMemUsedAtSolver = " << virtualMemUsedAtLc << ", getVmSize() = " << vmSize << ", SCIPgetMemUsed() = " << SCIPgetMemUsed(scip) << ", SCIPgetMemTotal() = " << SCIPgetMemTotal(scip) << ", SCIPgetMemExternEstim() = " << SCIPgetMemExternEstim(scip) << std::endl;
#else
      virtualMemUsedAtLc = SCIPgetMemTotal(scip) + SCIPgetMemExternEstim(scip);
      std::cout << "** Estimated virtualMemUsedAtSolver = " << virtualMemUsedAtLc << ", SCIPgetMemUsed() = " << SCIPgetMemUsed(scip) << ", SCIPgetMemTotal() = " << SCIPgetMemTotal(scip) << ", SCIPgetMemExternEstim() = " << SCIPgetMemExternEstim(scip) << std::endl;
#endif
      // memoryLimitOfSolverSCIP = (dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit) - (virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR))/((paraComm->getSize()-1)*SCIP_MEMORY_COPY_FACTOR);
      memoryLimitOfSolverSCIP = dynamic_cast<ScipParaParamSet *>(paraParams)->getRealParamValue(MemoryLimit)/paraComm->getSize() - virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR*paraComm->getSize()*SCIP_MEMORY_COPY_FACTOR;
      std::cout << "** set memory limit for solvers to " << memoryLimitOfSolverSCIP << " for each SCIP **" << std::endl;
      if( memoryLimitOfSolverSCIP < virtualMemUsedAtLc*SCIP_FIXED_MEMORY_FACTOR )
      {     
         solvedAtInit = true;
         setFinalSolverStatus(MemoryLimitIsReached);
         // finalDualBound = SCIPgetDualbound(scip);
         std::cout << "=== solved at Init ===" << std::endl;
         finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
         writeSolution("Final Solution");
         return 1;
      }
#endif

      // output presolved instance information
      int nNonLinearConsHdlrs = 0;
      outputProblemInfo(&nNonLinearConsHdlrs);
#ifdef _COMM_MPI_WORLD
      if( SCIPgetNActiveBenders(scip) > 0 ) nNonLinearConsHdlrs++;
      PARA_COMM_CALL(
            paraComm->bcast( &nNonLinearConsHdlrs, 1, ParaINT, 0 )
      );
      if( nNonLinearConsHdlrs > 0 )
      {
         paraParams->setIntParamValue(InstanceTransferMethod,2);
      }
#endif
      std::cout << "** Instance transfer method used: " <<  paraParams->getIntParamValue(InstanceTransferMethod) << std::endl;

      if( SCIPgetNVars(scip) == 0 )  // all variables were fixed in presolve
      {
         SCIP_CALL( SCIPsolve(scip) );
         solvedAtInit = true;
         setFinalSolverStatus(ProblemWasSolved);
         // finalDualBound = SCIPgetDualbound(scip);
         std::cout << "=== solved at Init ===" << std::endl;
         finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
         writeSolution("Final Solution");
         return 1;
      }

      /** check if feasible solution is found or not. If it was found, then generate paraSolution */
      SCIP_SOL *sol = SCIPgetBestSol(scip);
      // std::cout << "solobj = " << SCIPgetSolOrigObj(scip, sol) << ", objlimit = " << SCIPgetObjlimit(scip) << std::endl;
      if( sol )
      {
         if( EPSLT( objlimit, DBL_MAX, MINEPSILON ) )
         {
            if( ( SCIPgetObjsense(scip) == SCIP_OBJSENSE_MINIMIZE && SCIPgetSolOrigObj(scip, sol) < objlimit ) ||
                ( SCIPgetObjsense(scip) == SCIP_OBJSENSE_MAXIMIZE && SCIPgetSolOrigObj(scip, sol) > objlimit ) ) 
            {
               int nVars = SCIPgetNVars(scip);
               SCIP_VAR **vars = SCIPgetVars(scip);
               SCIP_Real *vals = new SCIP_Real[nVars];
               SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
               solution = scipParaComm->createScipParaSolution(
                              0,
                              SCIPgetSolTransObj(scip, sol),  // Only this value may be used
                              nVars,
                              vars,
                              vals
                              );
               delete [] vals;
            }
            else
            {
               solution = scipParaComm->createScipParaSolution(
                              0,
                              ( ( objlimit / ( SCIPgetTransObjscale(scip) * SCIPgetObjsense(scip) ) ) - SCIPgetTransObjoffset(scip) ), // Only this value may be used
                              0,
                              (SCIP_VAR **)0,
                              0
                              );
            }
         }
         else
         {
            int nVars = SCIPgetNVars(scip);
            SCIP_VAR **vars = SCIPgetVars(scip);
            SCIP_Real *vals = new SCIP_Real[nVars];
            SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
            solution = scipParaComm->createScipParaSolution(
                           0,
                           SCIPgetSolTransObj(scip, sol),  // Only this value may be used
                           0,
                           (SCIP_VAR **)0,
                           0
                           );
            delete [] vals;
         }
      }
      else
      {
         if( EPSLT( objlimit, DBL_MAX, MINEPSILON ) )
         {
            solution = scipParaComm->createScipParaSolution(
                           0,
                           ( ( objlimit / ( SCIPgetTransObjscale(scip) * SCIPgetObjsense(scip) ) ) - SCIPgetTransObjoffset(scip) ), // Only this value may be used
                           0,
                           (SCIP_VAR **)0,
                           0
                           );
         }
      }
      
      

      // instance = new PARA_INSTANCE_TYPE(scip, paraParams->getIntParamValue(InstanceTransferMethod));
      #ifdef _COMM_MPI_WORLD
      /** In ParaSCIP case, instance have to provided for the presolved instance **/
      delete instance;
      instance = scipParaComm->createScipParaInstance(scip, paraParams->getIntParamValue(InstanceTransferMethod));
      #endif
      // instance = scipParaComm->createScipParaInstance(scip, paraParams->getIntParamValue(InstanceTransferMethod));

      /* for debugging
      std::string subcipprefix("presolved_");
      std::string subcipfilename;
      std::ostringstream oss;
      oss << subcipprefix;
      oss << instance->getProbName();
      subcipfilename = oss.str();
      subcipfilename += ".lp";
      if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
      {
         SCIP_CALL_ABORT( SCIPwriteTransProblem(scip, subcipfilename.c_str(), "lp", FALSE) );
      }
      ************************/

#ifdef _COMM_MPI_WORLD
      if( onlyLinearConss && paraParams->getIntParamValue(UG::InstanceTransferMethod) != 2 )
      {
         // restore original parameters
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", originalUpgradeKnapsack));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", originalUpgradeLogicor));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", originalUpgradeSetppc));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", originalUpgradeVarbound));
      }
#else
      if( noUpgrade )
      {
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", originalUpgradeKnapsack));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", originalUpgradeLogicor));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", originalUpgradeSetppc));
         SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", originalUpgradeVarbound));
      }
#endif

      int maxrounds = 0;
      SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/maxrounds", &maxrounds));
      if( !paraParams->getBoolParamValue(Quiet) && maxrounds != 0 )
      {
         os << ".trans";
         transSolutionFile = fopen(os.str().c_str(), "a");  // if trans. solution file exists, append
         if( transSolutionFile == NULL )
         {
            THROW_LOGICAL_ERROR3("cannot open solution file <", os.str(), "> for writing");
         }
      }
      if( paraParams->getBoolParamValue(UG::OutputPresolvedInstance) )
      {
         std::ostringstream os2;
         os2 << paraParams->getStringParamValue(UG::LogSolvingStatusFilePath);
         if( onlyLinearConss )
         {
            os2 << instance->getProbName() << "_presolved.lp";
         }
         else
         {
            os2 << instance->getProbName() << "_presolved.cip";
         }
         SCIP_CALL_ABORT( SCIPwriteTransProblem(scip, os2.str().c_str(), NULL, FALSE));
      }
   }
   else
   {
      std::cout << std::endl;
      std::cout << "syntax: " << argv[0] << "#solvers ppscip_param_file problem_file_name "
                << "[-l <logfile>] [-q] [-sl <settings>] [-s <settings>] [-sr <root_settings>] [-w <prefix_warm>] [-sth <number>]" << std::endl;
      std::cout << "  -l <logfile>        : copy output into log file" << std::endl;
      std::cout << "  -q                  : suppress screen messages" << std::endl;
      std::cout << "  -sl <settings>      : load parameter settings (.set) file for LC presolving" << std::endl;
      std::cout << "  -s <settings>       : load parameter settings (.set) file for solvers" << std::endl;
      std::cout << "  -sr <root_settings> : load parameter settings (.set) file for root" << std::endl;
      std::cout << "  -w <prefix_warm>    : warm start file prefix ( prefix_warm_nodes.gz and prefix_warm_solution.txt are read )" << std::endl;
      std::cout << "  -sth <number>       : the number of solver threads used(FiberSCIP)" << std::endl;
      THROW_LOGICAL_ERROR1("invalid parameter");
   }

   if( solution )
   {
      if( !paraParams->getBoolParamValue(Quiet) )
      {
         writeSolution("");
      }
      else
      {
         writeSolution("Updated");
      }
   }

   return 0;
}

/** reInit function */
int
ScipParaInitiator::reInit(
      int nRestartedRacing
      )
{
   /** save incumbent solution */
   char initSolFileName[256];

   if( isolname )
   {
      (void) sprintf(initSolFileName,"%s.%d",isolname, nRestartedRacing);
      if( !rename( isolname, initSolFileName ) )
      {
         std::cout << "Warning: initial solution file name cannot rename: " << isolname << ", " << nRestartedRacing << "'th restarted file." << std::endl;
         // perror("cannot rename");
         // THROW_LOGICAL_ERROR1("cannot rename solution file");
      }
      if( !generatedIsolname )
      {
         (void) sprintf(initSolFileName,"%s", isolname );
         generatedIsolname = new char[strlen(initSolFileName)+1];
         (void) strcpy(generatedIsolname, initSolFileName);
      }
   }
   else
   {
      if( !generatedIsolname )
      {
         (void) sprintf(initSolFileName,"i_%s.sol", instance->getProbName() );
         generatedIsolname = new char[strlen(initSolFileName)+1];
         (void) strcpy(generatedIsolname, initSolFileName);
      }
   }

   FILE *fp = fopen(generatedIsolname, "w");
   if( !fp )
   {
      std::cout << "Could not open " << generatedIsolname << " file to reinitialize for restart." << std::endl;
      abort();
   }
   assert( SCIPgetBestSol(scip) );
   SCIP_CALL_ABORT( SCIPprintBestSol( scip, fp, FALSE) );
   (void) fclose(fp);

   // Problem Creation
   SCIP_RETCODE retcode = SCIPreadProb(scip, probname, NULL);
   if( retcode != SCIP_OKAY )
   {
      std::cout << "error reading file <" << probname << ">" << std::endl;
      SCIP_CALL( SCIPfreeProb(scip) );
      exit(1);
   }

   // std::cout << "problem name in initiator ' " << SCIPgetProbName(scip) << std::endl;

   SCIP_CALL_ABORT( SCIPtransformProb(scip));
   // NOTE:
   // When CPLEX license file cannot find, SCIPtransformProb(scip) may fail
   SCIP_CALL_ABORT( SCIPreadSol(scip, generatedIsolname) );

   /* change problem name */
   char *probNameFromFileName;
   char *temp = new char[strlen(probname)+1];
   (void) strcpy(temp, probname);
   SCIPsplitFilename(temp, NULL, &probNameFromFileName, NULL, NULL);
   SCIP_CALL_ABORT( SCIPsetProbName(scip, probNameFromFileName));
   delete [] temp;

#if ( defined(_COMM_PTH) || defined(_COMM_CPP11) )
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );
#endif

#ifdef _COMM_MPI_WORLD
   SCIP_Bool originalUpgradeKnapsack;
   SCIP_Bool originalUpgradeLogicor;
   SCIP_Bool originalUpgradeSetppc;
   SCIP_Bool originalUpgradeVarbound;
   bool onlyLinearConss = onlyLinearConsHandler();
   if( onlyLinearConss )
   {
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/knapsack", &originalUpgradeKnapsack));
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/logicor", &originalUpgradeLogicor));
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/setppc", &originalUpgradeSetppc));
      SCIP_CALL_ABORT( SCIPgetBoolParam(scip, "constraints/linear/upgrade/varbound", &originalUpgradeVarbound));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", FALSE));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", FALSE));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", FALSE));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", FALSE));
   }
#endif

   SCIP_CALL( SCIPpresolve(scip) );
   SCIP_STATUS scipStatus = SCIPgetStatus(scip);

   // output presolved instance information
   int nNonLinearConsHdlrs = 0;
   outputProblemInfo(&nNonLinearConsHdlrs);

   if( scipStatus == SCIP_STATUS_OPTIMAL ||
       scipStatus == SCIP_STATUS_INFEASIBLE )   // when sub-MIP is solved at root node, the solution may not be saved
   {
      solvedAtReInit = true;
      setFinalSolverStatus(ProblemWasSolved);
      // finalDualBound = SCIPgetDualbound(scip);
      std::cout << "=== solved at reInit ===" << std::endl;
      finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
      writeSolution("Final Solution");
      return 1;
   }
   else
   {
      /* adding root node cuts, if necessary */
      if( paraParams->getBoolParamValue(UseRootNodeCuts) )
      {
         if( !addRootNodeCuts() )
         {
            solvedAtReInit = true;
            setFinalSolverStatus(ProblemWasSolved);
            // finalDualBound = SCIPgetDualbound(scip);
            std::cout << "=== solved at reInit ===" << std::endl;
            finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
            writeSolution("Final Solution");
            return 1;
         }
      }
   }

   if( SCIPgetNVars(scip) == 0 )  // all variables were fixed in presolve
   {
      SCIP_CALL( SCIPsolve(scip) );
      solvedAtReInit = true;
      setFinalSolverStatus(ProblemWasSolved);
      // finalDualBound = SCIPgetDualbound(scip);
      std::cout << "=== solved at reInit ===" << std::endl;
      finalDualBound = instance->convertToInternalValue(SCIPgetDualbound(scip));
      writeSolution("Final Solution");
      return 1;
   }

   DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
   /** check if feasible solution is found or not. If it was found, then generate paraSolution */
   SCIP_SOL *sol = SCIPgetBestSol(scip);
   assert(sol);
   int nVars = SCIPgetNVars(scip);
   SCIP_VAR **vars = SCIPgetVars(scip);
   SCIP_Real *vals = new SCIP_Real[nVars];
   SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
   assert(solution);
   delete solution;
   solution = scipParaComm->createScipParaSolution(
                  0,
                  SCIPgetSolTransObj(scip, sol),  // Only this value may be used
                  nVars,
                  vars,
                  vals
                  );
   if( !paraParams->getBoolParamValue(Quiet) )
   {
      writeSolution("[Reinitialize]");
   }
   delete [] vals;
   // instance = new PARA_INSTANCE_TYPE(scip, paraParams->getIntParamValue(InstanceTransferMethod));
   assert(instance);
   delete instance;
   instance = scipParaComm->createScipParaInstance(scip, paraParams->getIntParamValue(InstanceTransferMethod));

   /* for debugging
   std::string subcipprefix("presolved_");
   std::string subcipfilename;
   std::ostringstream oss;
   oss << instance->getProbName();
   subcipfilename = oss.str();
   subcipfilename += ".lp";
   if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL_ABORT( SCIPwriteTransProblem(scip, subcipfilename.c_str(), "lp", FALSE) );
   }
   ************************/

#ifdef _COMM_MPI_WORLD
   if( onlyLinearConss )
   {
      // restore original parameters
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/knapsack", originalUpgradeKnapsack));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/logicor", originalUpgradeLogicor));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/setppc", originalUpgradeSetppc));
      SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "constraints/linear/upgrade/varbound", originalUpgradeVarbound));
   }
#endif

   return 0;

}

bool
ScipParaInitiator::tryToSetIncumbentSolution(
      BbParaSolution *sol,
      bool checksol
      )
{
   ScipParaSolution *tempSol = dynamic_cast< ScipParaSolution * >(sol);

   if( tempSol->getNVars() == 0 )
   {
      delete tempSol;
      return false;
   }

   /* If there is an active Benders' decomposition plugin, the stored solutions are not valid for the original problem.
    * This is due to the auxiliary variable not being present in the original problem. Thus, it is not possible to set
    * the incumbent solution
    */
   if( checksol && SCIPgetNActiveBenders(scip) > 0 )
   {
      delete tempSol;
      return false;
   }

   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */

   paraComm->lockApp();  /* lock is necessary, if Solver runs as thread */

   /* the solution from the working node should have at least as many variables as we have in the load coordinator scip
    * it may be more if inactive variable had to be copied, i.e.,
    * SCIPgetNVars(scip) is the number of active variables in the load coordinator scip
    * tempSol->getNVars() is the number of original variables in the working node scip (scipParaSolver)
    */
   SCIP_VAR** vars = 0;
   ScipParaInstance *scipInstance = dynamic_cast<ScipParaInstance *>(instance);
   /*
   int maxrounds = 0;
   SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/maxrounds", &maxrounds));
   // if( paraParams->getIntParamValue(InstanceTransferMethod) == 2    // original file read
   if( maxrounds == 0 )                                                 // nopreprocessing
   {
      //assert(SCIPgetNVars(scip) <= tempSol->getNVars());
      // create new solution for the original problem
      SCIP_CALL_ABORT( SCIPcreateOrigSol(scip, &newsol, 0) );
      vars = SCIPgetOrigVars(scip);
   }
   else
   {
   */
      if( checksol && SCIPgetNVars(scip) > tempSol->getNVars() )
      {
         std::cout << "*** You should check the solution! ***" << std::endl;
         std::cout << "checksol  = " << checksol << std::endl;
         std::cout << "SCIPgetNVars(scip) = " << SCIPgetNVars(scip) << ", " << tempSol->getNVars() << std::endl;
         delete tempSol;
         return false;
      }
      assert(SCIPgetNVars(scip) == tempSol->getNVars());
      SCIP_CALL_ABORT( SCIPcreateSol(scip, &newsol, 0) );
      vars = SCIPgetVars(scip);
   // }

   int i;
   if( scipInstance->isOriginalIndeciesMap() )
   {
      int n = SCIPgetNVars(scip);
      SCIP_Real *orgSolValues = new SCIP_Real[tempSol->getNVars()];
      scipInstance->getSolValuesForOriginalProblem(tempSol, orgSolValues);
      // for( i = 0; i < n; i++ )
      // assert(SCIPgetNVars(scip) == tempSol->getNVars());
      for( i = 0; i < n; i++ )
      // for( i = 0; i < scipInstance->getVarIndexRange(); i++ )
      {
         SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[i], orgSolValues[i]) );
         // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[tempSol->indexAmongSolvers(i)], orgSolValues[i]) );
         // int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));
         // int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));
         // int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));
         /* skip inactive variable */
         // if( probindex < 0 ) continue;
         // assert(i == probindex); /* this is just a temporory assert, maybe this holds... */
         // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[probindex], orgSolValues[i]) );
         // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[i], orgSolValues[i]) );
      }
      delete [] orgSolValues;
   }
   else
   {
      // assert( SCIPgetNVars(scip) ==  tempSol->getNVars() );
      assert( SCIPgetNVars(scip) <=  tempSol->getNVars() );

      // for( i = 0; i < tempSol->getNVars(); i++ )
      // {
      //   if( EPSGT(tempSol->getValues()[i],0.0, MINEPSILON) )
      //   {
      //      std::cout << "inex[" << i << "] = " << tempSol->indexAmongSolvers(i) << " = " << tempSol->getValues()[i] << std::endl;
      //   }
      // }

      for( i = 0; i < tempSol->getNVars(); i++ )
      {
         /* if index is larger-equal than number of active vars, then it's probably an inactive variable which had to be copied via SCIPcopy
          * so we just ignore its value
          */
         // if( tempSol->indexAmongSolvers(i) >= SCIPgetNVars(scip) ) continue;
         //  if( tempSol->indexAmongSolvers(i) > ( tempSol->getNVars() - 1 ) ) break;
         //  if( scipInstance->isOriginalIndeciesMap() )
         // {
         //    int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));
         //    SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[probindex], tempSol->getValues()[i]) );
         // }
         // else
         // {
         // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[tempSol->indexAmongSolvers(i)], tempSol->getValues()[i]) );
         if ( tempSol->indexAmongSolvers(i) >= 0 )
         {
            SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[tempSol->indexAmongSolvers(i)], tempSol->getValues()[i]) );
            // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[i], tempSol->getValues()[i]) );
            // }
         }
         /*
         if( scipInstance->isOriginalIndeciesMap() )
         {
            if( scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i)) >= 0 && scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i)) < SCIPgetNVars(scip) )
            {
               SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i))], tempSol->getValues()[i]) );
            }
         }
         else
         {
            SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[tempSol->indexAmongSolvers(i)], tempSol->getValues()[i]) );
         }
         */
      }
      // if( i != tempSol->getNVars() )
      // {
      //   /** the given solution should be generated in original space,
      //    * therefore the solution values cannot use for ParaSCIP
      //    */
      //   SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
      //   delete tempSol;
      //   std::cout << "solution size mismatch! Call Yuji!" << std::endl;
      //   return false;
      // }
   }

   SCIP_Bool success;
   // checksol = true;

   bool primalValueUpdated = false;

   if( paraParams->getBoolParamValue(CheckFeasibilityInLC) == false )
   {
      if( checksol )  // checksol == true only when this routine is called to add solution from checkpoint file.
                      // therefore,  no need to lock.
      {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
         SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
         SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
      }
      else
      {
         if( SCIPisTransformed(scip) && SCIPgetNSols(scip) > 0 )
         {
            double prevValue = SCIPgetPrimalbound(scip);
            SCIP_CALL_ABORT( SCIPaddSolFree(scip, &newsol, &success) );
            // if( EPSLT( SCIPgetPrimalbound(scip), prevValue, getEpsilon() ) )
            if( !EPSEQ( SCIPgetPrimalbound(scip), prevValue, getEpsilon() ) )  // I found increase case of this
            {
               primalValueUpdated = true;
            }
         }
         else
         {
            SCIP_CALL_ABORT( SCIPaddSolFree(scip, &newsol, &success) );
            primalValueUpdated = true; 
         }
         // SCIP_CALL_ABORT( SCIPaddSolFree(scip, &newsol, &success) );
         // primalValueUpdated = true; 
      }

      // std::cout << "** 2 ** success = " << success << std::endl;

      paraComm->unlockApp();

      if( success && primalValueUpdated )
      {
         if( solution )
         {
            delete solution;
        }
         solution = tempSol;
         if( !paraParams->getBoolParamValue(Quiet) )
         {
            writeSolution("");
         }
         else
         {
            writeSolution("Updated");
         }
         /// relax tolerance, since very small difference raised the following assertion
         // if( !EPSEQ( SCIPgetUpperbound(scip), solution->getObjectiveFunctionValue(), (SCIPfeastol(scip) )*100.0) )
         if( !EPSEQ( SCIPgetUpperbound(scip), solution->getObjectiveFunctionValue(), SCIPfeastol(scip) ) )
         {
            std::cout << "*** A new solution generated in a Solver does not match with the upperbound in LC ***" << std::endl;
            std::cout << "SCIPgetUpperbound(scip) = " << SCIPgetUpperbound(scip) << std::endl;
            std::cout << "solution->getObjectiveFunctionValue() = " << solution->getObjectiveFunctionValue() << std::endl;
            std::cout << "Difference = " << (SCIPgetUpperbound(scip) - solution->getObjectiveFunctionValue()) 
                      << ", SCIPfeastol(scip) = "  << SCIPfeastol(scip) << std::endl;
         }
         return true;
      }
      else
      {
         if( checksol )
         {
            SCIP_CALL_ABORT( SCIPcreateOrigSol(scip, &newsol, 0) );
            vars = SCIPgetOrigVars(scip);
            if( scipInstance->isOriginalIndeciesMap() )
            {
               for( i = 0; i < tempSol->getNVars(); i++ )
               {
                  int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));

                  /* skip inactive variable */
                  if( probindex < 0 ) continue;

                  // assert(i == probindex); /* this is just a temporory assert, maybe this holds... */

                  SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[probindex], tempSol->getValues()[i]) );
               }
            }
            else
            {
               for( i = 0; i < tempSol->getNVars(); i++ )
               {
                  // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[tempSol->indexAmongSolvers(i)], tempSol->getValues()[i]) );
                  SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[i], tempSol->getValues()[i]) );
               }
            }

            // if( i != tempSol->getNVars() ) /* this should not happen */
            // {
            //   /** the given solution should be generated in original space,
            //    * therefore the solution values cannot use for ParaSCIP
            //    */
            //   SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
            //   delete tempSol;
            //   std::cout << "solution size mismatch! Call Yuji!" << std::endl;
            //   return false;
            //}
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
            SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
            SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
            // std::cout << "** 3 ** success = " << success << std::endl;
            if( success )
            {
               if( solution )
               {
                  delete solution;
              }
               solution = tempSol;
               if( !paraParams->getBoolParamValue(Quiet) )
               {
                  writeSolution("");
               }
               else
               {
                  writeSolution("Updated");
               }
               assert( EPSEQ( SCIPgetUpperbound(scip), solution->getObjectiveFunctionValue(), SCIPfeastol(scip) ) );
               return true;
            }
         }
         delete tempSol;
         return false;
      }
   }
   else
   {
      //
      // The following routine is not tested yet.
      //
      // std::cout << "Print sol. orig"  << std::endl;
      // SCIP_CALL_ABORT( SCIPprintSol(scip, newsol, NULL, FALSE) );
      // std::cout << "Print sol. trans"  << std::endl;
      // SCIP_CALL_ABORT( SCIPprintTransSol(scip, newsol, NULL, FALSE) );
      // int nVars = SCIPgetNVars(scip);
      // for( int i = 0; i < nVars; i ++)
      // {
      //    std::cout << i << ": " << SCIPvarGetName(vars[tempSol->indexAmongSolvers(i)]) << std::endl;
      //    std::cout << i << ": " << SCIPvarGetName(vars[i]) << std::endl;
      // }

#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
      assert(success);
      if( success )
      {
         // SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
         if( solution )
         {
            delete solution;
         }
         solution = tempSol;
         if( !paraParams->getBoolParamValue(Quiet) )
         {
            writeSolution("");
         }
         else
         {
            writeSolution("Updated");
         }
         paraComm->unlockApp();
         assert( EPSEQ( SCIPgetUpperbound(scip), solution->getObjectiveFunctionValue(), SCIPfeastol(scip) ) );
         return true;
      }
      else
      {
         SCIP_VAR* var = 0;
         for( i = 0; i < tempSol->getNVars(); i++ )
         {
            int probindex = scipInstance->getOrigProbIndex(tempSol->indexAmongSolvers(i));

            /* skip inactive variable */
            if( probindex < 0 ) continue;

            assert(i == probindex); /* this is just a temporory assert, maybe this holds... */

            var = vars[probindex];

            if( SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS ) continue;
            if( tempSol->getValues()[i] > 0.0 )
            {
               tempSol->setValue(i, SCIPfeasFloor(scip,tempSol->getValues()[i]));
            }
            else
            {
               tempSol->setValue(i, SCIPfeasCeil(scip, tempSol->getValues()[i]) );
            }
            SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, var, tempSol->getValues()[i]) );
         }

#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
         SCIP_CALL_ABORT( SCIPtrySol(scip, newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
         SCIP_CALL_ABORT( SCIPtrySol(scip, newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
         if( success )
         {
            tempSol->setObjectiveFuntionValue(convertToInternalValue(SCIPsolGetOrigObj(newsol)));
            SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
            if( solution )
            {
               delete solution;
            }
            solution = tempSol;
            if( !paraParams->getBoolParamValue(Quiet) )
            {
               writeSolution("");
            }
            else
            {
               writeSolution("Updated");
            }
            paraComm->unlockApp();
            assert( EPSEQ( SCIPgetUpperbound(scip), solution->getObjectiveFunctionValue(), SCIPfeastol(scip) ) );
            return true;
         }
         else
         {
            fprintf(solutionFile, "*** Rejected Solution ***\n");
            SCIP_CALL_ABORT(SCIPprintSol(scip, newsol, solutionFile, FALSE));
            if( transSolutionFile )
            {
               fprintf(transSolutionFile, "*** Rejected Solution ***\n");
               SCIP_CALL_ABORT(SCIPprintTransSol(scip, newsol, transSolutionFile, FALSE));
            }

            if( SCIPisFeasLT( scip, convertToExternalValue(tempSol->getObjectiveFunctionValue()), SCIPgetPrimalbound(scip) ) )
            {
               // paraComm->lockApp();
               std::cout << "Current scip primal value = " << SCIPgetPrimalbound(scip) << std::endl;
               std::cout << "Objective value = " << convertToExternalValue(tempSol->getObjectiveFunctionValue()) << std::endl;
               std::cout << "Initiator did not accept solution!" << std::endl;
               // paraComm->unlockApp();
            }
            SCIP_CALL_ABORT( SCIPfreeSol(scip, &newsol) );
            delete tempSol;
            paraComm->unlockApp();
            return false;
         }
      }
   }
}

void
ScipParaInitiator::sendSolverInitializationMessage(
      )
{
   assert(scipDiffParamSetRoot && scipDiffParamSet);
   scipDiffParamSetRoot->bcast(paraComm, 0);
   scipDiffParamSet->bcast(paraComm, 0);
   int warmStarted = 0;
   if( isWarmStarted() )
   {
      warmStarted = 1;
   }
   paraComm->bcast(&warmStarted,1, ParaINT, 0);
   double incumbentValue;
   if( solution )
   {
      incumbentValue = solution->getObjectiveFunctionValue();
   }
   else
   {
      SCIP_SOL *sol = SCIPgetBestSol(scip);
	   if ( sol )
	   {
	      int nVars = SCIPgetNVars(scip);
	      SCIP_VAR **vars = SCIPgetVars(scip);
	      SCIP_Real *vals = new SCIP_Real[nVars];
	      SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
	      DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
	      solution = scipParaComm->createScipParaSolution(
	            0,
	            SCIPgetSolTransObj(scip,sol),
	            nVars,
	             vars,
	             vals
	             );
	      delete [] vals;
	      incumbentValue = solution->getObjectiveFunctionValue();
	   }
	   else
	   {
	      incumbentValue = DBL_MAX;
	   }
   }
   paraComm->bcast(&incumbentValue, 1, ParaDOUBLE, 0);

   if( paraParams->getBoolParamValue(NoUpperBoundTransferInRacing) )
   {
      int solutionExists = 0;
      paraComm->bcast(&solutionExists, 1, ParaINT, 0);
   }
   else
   {
      /** if a feasible solution exists, broadcast the solution */
       if( paraParams->getBoolParamValue(DistributeBestPrimalSolution) )
       {
          /* bcast solution if it is necessary */
          int solutionExists = 0;
          if( solution )
          {
             solutionExists = 1;
             paraComm->bcast(&solutionExists, 1, ParaINT, 0);
             solution->bcast(paraComm, 0);
          }
          else
          {
             paraComm->bcast(&solutionExists, 1, ParaINT, 0);
          }
       }
   }
   
   // allocate here, since the number of variables is fixed when the instance data are sent
   if( ( paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 1 ||
         paraParams->getIntParamValue(UG::RampUpPhaseProcess) == 2 ) &&
         paraParams->getBoolParamValue(UG::CommunicateTighterBoundsInRacing) )
   {
      assert( instance->getNVars() > 0 );
      assert( instance->getVarIndexRange() > 0 );
      // IndexRange must be bigger than NVars
      tightenedVarLbs = new double[instance->getVarIndexRange()];
      tightenedVarUbs = new double[instance->getVarIndexRange()];
      for( int i = 0; i < instance->getVarIndexRange(); i++ )
      {
         tightenedVarLbs[i] = -DBL_MAX;
         tightenedVarUbs[i] = DBL_MAX;
      }
   }
   
}

/** get gap */
double
ScipParaInitiator::getAbsgap(
      double dualBoundValue
      )
{
   if( !solution ) return SCIPinfinity(scip);
   SCIP_Real primalbound = instance->convertToExternalValue(solution->getObjectiveFunctionValue());
   SCIP_Real dualbound = instance->convertToExternalValue(dualBoundValue);
   return REALABS((primalbound - dualbound));
}

/** get gap */
double
ScipParaInitiator::getGap(
      double dualBoundValue
      )
{
   if( !solution ) return SCIPinfinity(scip);
   SCIP_Real primalbound = instance->convertToExternalValue(solution->getObjectiveFunctionValue());
   SCIP_Real dualbound = instance->convertToExternalValue(dualBoundValue);

   if( SCIPisEQ(scip, primalbound, dualbound) )
      return 0.0;
   else if( SCIPisZero(scip, dualbound)
      || SCIPisZero(scip, primalbound)
      || SCIPisInfinity(scip, REALABS(primalbound))
      || SCIPisInfinity(scip, REALABS(dualbound))
      || primalbound * dualbound < 0.0 )
      return SCIPinfinity(scip);
   else
      return REALABS((primalbound - dualbound)/MIN(REALABS(dualbound),REALABS(primalbound)));
}

/** get epsilon */
double
ScipParaInitiator::getEpsilon(
      )
{
   SCIP_Real epsilon;
   SCIP_CALL_ABORT( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon));
   return epsilon;
}

void
ScipParaInitiator::writeSolution(
      const std::string& message
      )
{
   std::ostringstream osold;
   if( message == "Final Solution" )
   {
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      if( paraParams->getBoolParamValue(Quiet) )
      {
         SCIPmessageSetDefaultHandler();    // If no message handler is set, it cannot write solution,too.
      }
#endif
      if( solvedAtInit || (!paraParams->getBoolParamValue(Quiet)) )  // when solutionFileName is specified, it is always updated
      {
          fprintf(solutionFile, "[ Final Solution ]\n");
      }
      if( transSolutionFile )
      {
         fprintf(transSolutionFile, "[ Final Solution ]\n");
      }
   }
   else if ( message == "Updated" )
   {
      assert( solutionFileName || (paraParams->getBoolParamValue(Quiet) && !solutionFileName ) );
      fclose(solutionFile);
      std::ostringstream os;
      if( solutionFileName )
      {
         osold << solutionFileName << ".old";
         os << solutionFileName;
      }
      else
      {
         osold << paraParams->getStringParamValue(SolutionFilePath);
         osold << instance->getProbName() << ".sol.old";
         os << paraParams->getStringParamValue(SolutionFilePath);
         os << instance->getProbName() << ".sol";
      }
      // std::cout << "File Name: " << os.str().c_str() << " to " << osold.str().c_str() << std::endl;
      if( rename(os.str().c_str(), osold.str().c_str()) )
      {
          std::cerr << "Rename falied from " <<  "File Name: " << os.str().c_str() << " to " << osold.str().c_str() << std::endl;
          exit(1);
      }
      solutionFile = fopen(os.str().c_str(), "a");  // if solution file exists, append
      fprintf(solutionFile, "[ Final Solution ]\n");
   }
   else
   {
      fprintf(solutionFile,"%s\n",message.c_str());
      if( transSolutionFile )
      {
         fprintf(transSolutionFile,"%s\n", message.c_str());
      }
   }
   SCIP_SOL* sol = SCIPgetBestSol(scip);
   if( sol )
   {
      if( solvedAtInit || (!(message == "Final Solution")) || (!paraParams->getBoolParamValue(Quiet)) )  // when solutionFileName is specified, it is always updated
      {
         SCIP_CALL_ABORT( SCIPprintBestSol( scip, solutionFile, FALSE) );
      }
      if( transSolutionFile )
      {
         if( SCIPsolGetOrigin(sol) != SCIP_SOLORIGIN_ORIGINAL )
         {
            ScipParaInstance *scipInstance = dynamic_cast<ScipParaInstance *>(instance);
            if( scipInstance->isOriginalIndeciesMap() )
            {
               SCIP_CALL_ABORT( SCIPprintBestSol( scipInstance->getParaInstanceScip(), transSolutionFile, FALSE) );
            }
            else
            {
               SCIP_CALL_ABORT( SCIPprintBestTransSol( scip, transSolutionFile, FALSE) );
            }
         }
         else
         {
            fprintf(transSolutionFile, "best solution is defined in original space - cannot print it as transformed solution\n");
         }
      }
      /*
      if( userPlugins )
      {
         userPlugins->writeUserSolution(scip);
      }
      */
   }
   else
   {
      fprintf(solutionFile, "No Solution\n");
      if( transSolutionFile )
      {
         fprintf(transSolutionFile, "No Solution\n");
      }
   }
   if ( message == "Updated" )
   {
      remove(osold.str().c_str());
   }
}

void
ScipParaInitiator::writeParaInstance(
      const std::string& filename
      )
{
   FILE *file = fopen(filename.c_str(),"a");
   if( !file )
   {
      std::cout << "file : " << filename << "cannot open." << std::endl;
      exit(1);
   }
   ScipParaInstance *scipInstance = dynamic_cast<ScipParaInstance *>(instance);
   if( scipInstance->isOriginalIndeciesMap() )
   {
      SCIP_CALL_ABORT( SCIPprintOrigProblem(scipInstance->getParaInstanceScip(), file, "lp", FALSE) );
   }
   else
   {
      SCIP_CALL_ABORT( SCIPprintTransProblem(scip, file, "lp", FALSE) );
   }
}

/** write solver runtime parameters */
void
ScipParaInitiator::writeSolverParameters(
      std::ostream *os
      )
{
   if( scipDiffParamSetRoot->nDiffParams() == 0 )
   {
      *os << "[ SCIP parameters for root Solver are all default values ]" << std::endl;
   }
   else
   {
      *os << "[ Not default SCIP parameters for root Solver are as follows ]" << std::endl;
      *os << scipDiffParamSetRoot->toString();
   }

   if( scipDiffParamSet->nDiffParams() == 0 )
   {
       *os << "[ SCIP parameters for NOT root Solvers are all default values ]" << std::endl;
   }
   else
   {
      *os << "[ Not default SCIP parameters for NOT root Solvers are as follows ]" << std::endl;
      *os << scipDiffParamSet->toString();
   }

}

#ifdef UG_WITH_ZLIB
/** write checkpoint solution */
void
ScipParaInitiator::writeCheckpointSolution(
      const std::string& filename
      )
{
   gzstream::ogzstream checkpointSolutionStream;
   checkpointSolutionStream.open(filename.c_str(), std::ios::out | std::ios::binary);
   if( !checkpointSolutionStream )
   {
      std::cout << "Checkpoint file for solution cannot open. file name = " << filename << std::endl;
      exit(1);
   }
   if( solution )
      solution->write(checkpointSolutionStream);
   checkpointSolutionStream.close();     /** empty solution file is necessary,
                                          * because it is removed next at the next checkpoint */
}

/** read solution from checkpoint file */
double
ScipParaInitiator::readSolutionFromCheckpointFile(
      char *afterCheckpointingSolutionFileName
      )
{
   char tempSolutionFileName[256];
   sprintf(tempSolutionFileName,"%s_solution.gz", prefixWarm );
   gzstream::igzstream  checkpointSolutionStream;
   checkpointSolutionStream.open(tempSolutionFileName, std::ios::in | std::ios::binary);
   if( !checkpointSolutionStream )
   {
      std::cout << "checkpoint solution file cannot open: file name = " <<  tempSolutionFileName << std::endl;
      exit(1);
   }
   if( solution )
   {
       ScipParaSolution *sol = dynamic_cast<ScipParaSolution*>(paraComm->createParaSolution());
       if( sol->read(paraComm, checkpointSolutionStream) )
       {
          if( solution->getObjectiveFunctionValue() > sol->getObjectiveFunctionValue() )
          {
             delete solution;
             solution = sol;
          }
       }
       else
       {
          delete sol;
       }
   }
   else
   {
      solution = dynamic_cast<ScipParaSolution*>(paraComm->createParaSolution());
      if( !solution->read(paraComm, checkpointSolutionStream) )
      {
         delete solution;
         solution = 0;
         checkpointSolutionStream.close();
      }
   }
   checkpointSolutionStream.close();
   if( solution )
   {
      if( !tryToSetIncumbentSolution(dynamic_cast<BbParaSolution *>(solution->clone(paraComm)), true) )
      {
         std::cout << "***** Given solution is wrong! ***************************" << std::endl;
         std::cout << "***** If the solution was given from checkpoint file,  ***" << std::endl;
         std::cout << "***** it might be generated in original problem space   **" << std::endl;
         std::cout << "***** Only primal value is used. *************************" << std::endl;
         std::cout << "***** You should better to use -isol option.  ************" << std::endl;
         std::cout << "***** Or, better to use no distribute solution option. ***" << std::endl;
      }
   }

   /** check if after checkpoing solution file exists or not */
   checkpointSolutionStream.open(afterCheckpointingSolutionFileName, std::ios::in | std::ios::binary);
   if( checkpointSolutionStream )
   {
      /** set up from after checkpointing solution file */
      ScipParaSolution *sol = dynamic_cast<ScipParaSolution*>(paraComm->createParaSolution());
      if( sol->read(paraComm, checkpointSolutionStream) )
      {
         if( !solution )
         {
            solution = sol;
            if( tryToSetIncumbentSolution(dynamic_cast<BbParaSolution *>(solution->clone(paraComm)), true) )
            {
               std::cout << "***** After checkpoint solution is RIGHT! ****************" << std::endl;
            }
         }
         else
         {
            if( solution->getObjectiveFunctionValue() > sol->getObjectiveFunctionValue() )
            {
               delete solution;
               solution = sol;
               if( tryToSetIncumbentSolution(dynamic_cast<BbParaSolution *>(solution->clone(paraComm)), true) )
               {
                  std::cout << "***** After checkpoint solution is RIGHT! ****************" << std::endl;
               }
            }
         }
      }
      else
      {
         delete sol;
      }
      checkpointSolutionStream.close();
   }

   if( solution )
   {
      return solution->getObjectiveFunctionValue();
   }
   else
   {
      return DBL_MAX;
   }
}
#endif

/** generate racing ramp-up parameter sets */
void
ScipParaInitiator::generateRacingRampUpParameterSets(
      int nParamSets,
      ParaRacingRampUpParamSet **racingRampUpParamSets
      )
{
   ScipDiffParamSet *racingScipDiffParamSet = 0;  // assume all default

   if ( std::string(paraParams->getStringParamValue(UG::RacingParamsDirPath)) != std::string("") )
   {
      DEF_SCIP_PARA_COMM( scipParaComm, paraComm);

      for( int n = 1; n < scipParaComm->getSize() ; n++ )
      {
         std::ostringstream oss;
         oss << paraParams->getStringParamValue(UG::RacingParamsDirPath)
               << "/"
               << std::setfill('0') << std::setw(5)
               << ( ( (n - 1) % paraParams->getIntParamValue(UG::NMaxRacingBaseParameters) ) + 1 )
               << ".set";

         SCIP_CALL_ABORT( SCIPresetParams(scip) );
         if( SCIPreadParams(scip, oss.str().c_str()) != SCIP_OKAY )
         {
            std::cout << "Cannot read racing parameter file = " << oss.str().c_str() << std::endl;
            exit(1);
         }

         racingScipDiffParamSet = scipParaComm->createScipDiffParamSet(scip);
         racingRampUpParamSets[n-1] = scipParaComm->createScipParaRacingRampUpParamSet(
               paraParams->getIntParamValue(UG::RacingRampUpTerminationCriteria),
               paraParams->getIntParamValue(UG::StopRacingNumberOfNodesLeft),
               paraParams->getRealParamValue(UG::StopRacingTimeLimit),
               n,
               ( (n - 1) % paraParams->getIntParamValue(UG::NMaxRacingBaseParameters)),    // the number of variable permutation seed; start from default: 0
               0,
               racingScipDiffParamSet
               );
         SCIP_CALL_ABORT( SCIPresetParams(scip) );
      }
   }
   else
   {
      if( racingSettingsName )
      {
         SCIP_CALL_ABORT( SCIPresetParams(scip) );
         SCIP_CALL_ABORT( SCIPreadParams(scip, racingSettingsName) );
         DEF_SCIP_PARA_COMM( scipParaComm, paraComm );
         racingScipDiffParamSet = scipParaComm->createScipDiffParamSet(scip);
         SCIP_CALL_ABORT( SCIPresetParams(scip) );
      }
      /*
      else
      {
         racingScipDiffParamSet = 0;  //  all default
      }*/


      int n = 0;     /**< keep the number of generated params */
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      int npm = -1;  /**< keep the number of variable permutation seed; start from default: -1 */
#else
      int npm = 0;   /**< keep the number of variable permutation seed; start from default: 0 */
#endif
      int nbo = 0;   /**< keep the number of branching order seed */

      DEF_SCIP_PARA_COMM( scipParaComm, paraComm);

      for(;;)
      {
         for( int i = 0; i < paraParams->getIntParamValue(MaxNRacingParamSetSeed); i++ )
         {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
            if( npm > ( paraParams->getIntParamValue(TryNVariablegOrderInRacing) - 1 ) ) npm = -1;
#else
            if( npm > ( paraParams->getIntParamValue(TryNVariablegOrderInRacing) - 1 ) ) npm = 0;
#endif
            if( nbo > paraParams->getIntParamValue(TryNBranchingOrderInRacing) ) nbo = 0;
            racingRampUpParamSets[n] = scipParaComm->createScipParaRacingRampUpParamSet(
                  paraParams->getIntParamValue(RacingRampUpTerminationCriteria),
                  paraParams->getIntParamValue(StopRacingNumberOfNodesLeft),
                  paraParams->getRealParamValue(StopRacingTimeLimit),
                  i,
                  npm,
                  nbo,
                  racingScipDiffParamSet
                  );
            npm++;
            nbo++;
            n++;
            if( n >= nParamSets ) return;
         }
      }
   }
}

/** get solving status string */
std::string
ScipParaInitiator::getStatus(
      )
{
   SCIP_SOL* sol = SCIPgetBestSol(scip);
   if( sol )
   {
      return std::string("solution found exist");
   }
   else
   {
      return std::string("no solution");
   }
}

/** print solver version **/
void
ScipParaInitiator::printSolverVersion(
      std::ostream *os           /**< output file (or NULL for standard output) */
      )
{
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
   SCIPprintVersion( NULL );
#else
   SCIPprintVersion( scip, NULL );
#endif

   SCIPprintExternalCodes(scip, NULL);
}

/** set initial stat on initiator */
void
ScipParaInitiator::accumulateInitialStat(
      ParaInitialStat *initialStat
      )
{
   ScipParaInitialStat *scipInitialStat = dynamic_cast<ScipParaInitialStat *>(initialStat);
   ScipParaInstance *scipInstance = dynamic_cast<ScipParaInstance *>(instance);
   if( scipInstance->isOriginalIndeciesMap() )
   {
      scipInitialStat->accumulateOn(scipInstance->getParaInstanceScip());;
   }
   else
   {
      scipInitialStat->accumulateOn(scip);
   }
}

/** set initial stat on DiffSubproblem */
void
ScipParaInitiator::setInitialStatOnDiffSubproblem(
      int                minDepth,
      int                maxDepth,
      BbParaDiffSubproblem *diffSubproblem
      )
{
   ScipParaDiffSubproblem *scipDiffSubproblem = dynamic_cast<ScipParaDiffSubproblem *>(diffSubproblem);
   ScipParaInstance *scipInstance = dynamic_cast<ScipParaInstance *>(instance);
   if( scipInstance->isOriginalIndeciesMap() )
   {
      scipDiffSubproblem->addInitialBranchVarStats(minDepth, maxDepth, scipInstance->getParaInstanceScip());
   }
   else
   {
      scipDiffSubproblem->addInitialBranchVarStats(minDepth, maxDepth, scip);
   }
}

/** set final solver status */
void
ScipParaInitiator::setFinalSolverStatus(
      FinalSolverState state
      )
{
   finalState = state;
}

/** set number of nodes solved */
void
ScipParaInitiator::setNumberOfNodesSolved(
      long long n
      )
{
   nSolved = n;
}

/** set final dual bound  */
void
ScipParaInitiator::setDualBound(
      double bound
      )
{
   if( bound > finalDualBound )
   {
      finalDualBound = bound;
   }
}

/** output solution status */
void
ScipParaInitiator::outputFinalSolverStatistics(
      std::ostream *os,
      double time
      )
{
   if( os == 0 )
   {
      os = &std::cout;
   }

   if( finalState !=  Aborted )
   {
      *os << "SCIP Status        : ";
   }

   switch ( finalState )
   {
   case InitialNodesGenerated:
      *os << "initial nodes were generated" << std::endl;
      break;
   case Aborted:
      *os << std::endl;
      break;
   case HardTimeLimitIsReached:
      *os << "solving was interrupted [hard time limit reached]" << std::endl;
      break;
   case MemoryLimitIsReached:
      *os << "solving was interrupted [memory limit reached]" << std::endl;
      break;
   case GivenGapIsReached:
      *os << "solving was interrupted [given gap reached]" << std::endl;
      break;
   case ComputingWasInterrupted:
      *os << "solving was interrupted" << std::endl;
      break;
   case ProblemWasSolved:
      if( SCIPgetNSols(scip) > 0 )
      {
         *os << "problem is solved" << std::endl;
      }
      else
      {
         *os << "problem is solved [infeasible]" << std::endl;
      }
      break;
   case RequestedSubProblemsWereSolved:
      *os << "requested subproblems are solved" << std::endl;
      break;
   default:
      THROW_LOGICAL_ERROR1("invalid final state");
   }

   *os << "Total Time         : " << time << std::endl;
   *os << "  solving          : " << time << std::endl;
   *os << "  presolving       : " << SCIPgetPresolvingTime(scip) << " (included in solving)" << std::endl;
   *os << "B&B Tree           :" << std::endl;
   *os << "  nodes (total)    : " << nSolved << std::endl;
   *os << "Solution           :" << std::endl;
   *os << "  Solutions found  : " << SCIPgetNSols(scip) << std::endl;
   SCIP_Real primalbound = SCIPinfinity(scip);
   // std::cout << "** Nsols = " << SCIPgetNSols(scip) << std::endl;
   // if( solution )
   // {
   //     std::cout << "*** obj = " << convertToExternalValue(solution->getObjectiveFuntionValue()) << std::endl;
   // }
   if( SCIPgetNSols(scip) != 0 )
   {
      primalbound = SCIPgetPrimalbound(scip);
   }
   else
   {
      if( solution && solution->getNVars() > 0 )
      {
         primalbound = solution->getObjectiveFunctionValue();   // This solution should be in original problem space.
                                                               // So, do not have to convert to original space.
      }
      else
      {
    	  if( EPSLT( objlimit, DBL_MAX, MINEPSILON ) )
    	  {
    		  primalbound = objlimit;
    	  }
      }
   }
   *os << "  Primal Bound     : ";
   if( SCIPisInfinity(scip, REALABS(primalbound) ) )
   {
      *os << "         -" << std::endl;
   }
   else
   {
      (*os).setf(std::ios::showpos);
      *os << std::scientific << std::showpoint << std::setprecision(14) << primalbound << std::endl;
      (*os).unsetf(std::ios::showpos);
   }

   finalDualBound = instance->convertToExternalValue(finalDualBound); // converted to original one
   SCIP_Real finalGap = 0.0;
   if( SCIPisEQ(scip, primalbound, finalDualBound) )
      finalGap = 0.0;
   else if( SCIPisZero(scip, finalDualBound)
         || SCIPisZero(scip, primalbound)
         || SCIPisInfinity(scip, REALABS(primalbound))
         || SCIPisInfinity(scip, REALABS(finalDualBound))
         || primalbound * finalDualBound < 0.0 )
      finalGap = SCIPinfinity(scip);
   else
      finalGap = REALABS((primalbound - finalDualBound)/MIN(REALABS(finalDualBound),REALABS(primalbound)));

#if SCIP_VERSION > 302 || ( SCIP_VERSION == 302 && SCIP_SUBVERSION == 1 )
   *os << "  Dual Bound       : ";
#else
   *os << "Dual Bound         : ";
#endif
   if( SCIPisInfinity(scip, REALABS(finalDualBound) ) )
      *os << "         -" << std::endl;
   else
   {
      (*os).setf(std::ios::showpos);
      *os <<  std::scientific << std::showpoint << std::setprecision(14)
      << finalDualBound << std::endl;
      (*os).unsetf(std::ios::showpos);
   }
   *os << "Gap                : ";
   if( SCIPgetNSols(scip) == 0 )
   {
      *os << "         -" << std::endl;
   }
   else if( SCIPisInfinity(scip, finalGap ) )
      *os << "  infinite" << std::endl;
   else
   {
      *os << std::fixed << std::setprecision(5) << 100.0 * finalGap << " %" << std::endl;
   }
   if( finalGap > SCIPepsilon(scip) && !SCIPisInfinity(scip, REALABS(primalbound) ) )
   {
      *os << std::scientific << "* Warning: final gap: " << finalGap << " is greater than SCIPepsilon: " <<  SCIPepsilon(scip) << std::endl;
   }

   assert( finalState != ProblemWasSolved ||
         ( finalState == ProblemWasSolved && ( SCIPgetNSols(scip) == 0 || ( SCIPgetNSols(scip) != 0 ) ) ) );

   if( userPlugins )
   {
      userPlugins->writeUserSolution(scip, paraComm->getSize()-1, finalDualBound);
   }
}

void
ScipParaInitiator::outputProblemInfo(
      int *nNonLinearConsHdlrs
      )
{
   std::cout << "Original Problem   :" << std::endl;
   std::cout << "  Problem name     : " << SCIPgetProbName(scip) << std::endl;
   std::cout << "  Variables        : " << SCIPgetNOrigVars(scip)
         << " (" << SCIPgetNOrigBinVars(scip) << " binary, "
         << SCIPgetNOrigIntVars(scip) << " integer, "
         << SCIPgetNOrigImplVars(scip) << " implicit integer, "
         << SCIPgetNOrigContVars(scip) << " continuous)" << std::endl;
   std::cout << "  Constraints      : " << SCIPgetNOrigConss(scip) << std::endl;
   std::cout << "  Objective sense  : " <<  (SCIPgetObjsense(scip) == SCIP_OBJSENSE_MINIMIZE ? "minimize" : "maximize") << std::endl;
   std::cout << "Presolved Problem  :" << std::endl;
   std::cout << "  Variables        : " << SCIPgetNVars(scip)
         << " (" << SCIPgetNBinVars(scip) << " binary, "
         << SCIPgetNIntVars(scip) << " integer, "
         << SCIPgetNImplVars(scip) << " implicit integer, "
         << SCIPgetNContVars(scip) << " continuous)" << std::endl;
   std::cout << "  Constraints      : " << SCIPgetNConss(scip) << std::endl;

   std::cout << "Constraints        : Number" << std::endl;
   for( int i = 0; i < SCIPgetNConshdlrs(scip); ++i )
   {
      SCIP_CONSHDLR* conshdlr;
      int startnactiveconss;
      int maxnactiveconss;
      conshdlr = SCIPgetConshdlrs(scip)[i];
      startnactiveconss = SCIPconshdlrGetStartNActiveConss(conshdlr);
      maxnactiveconss = SCIPconshdlrGetMaxNActiveConss(conshdlr);
      if( startnactiveconss > 0 )
      {
         std::cout << "  " << std::setw(17) << std::left << SCIPconshdlrGetName(conshdlr) << ": "
                   <<  startnactiveconss <<  ( maxnactiveconss > startnactiveconss ? '+' : ' ') << std::endl;
         if ( std::string(SCIPconshdlrGetName(conshdlr)) != std::string("linear") )
         {
            *nNonLinearConsHdlrs += startnactiveconss;
         }
      }
   }
}

bool 
ScipParaInitiator::onlyLinearConsHandler(
      )
{
   for( int i = 0; i < SCIPgetNConss(scip); ++i )
   {
      SCIP_CONS** conss = SCIPgetConss(scip);
      SCIP_CONSHDLR* conshdlr = SCIPconsGetHdlr(conss[i]);
      if( std::string(SCIPconshdlrGetName(conshdlr)) != std::string("linear") )
      {
         return false;
      }
   }
   return true;
}

void
ScipParaInitiator::setUserPlugins(
      ScipUserPlugins *inUi
      )
{
   userPlugins = inUi;
}

#ifdef UG_WITH_UGS
/** read ugs incumbent solution **/
bool
ScipParaInitiator::readUgsIncumbentSolution(
      UGS::UgsParaCommMpi *ugsComm,
      int source
      )
{
   UGS::UgsUpdateIncumbent *updateIncumbent = new UGS::UgsUpdateIncumbent();
   updateIncumbent->receive(ugsComm->getMyComm(), source);

   if( ( instance->getOrigObjSense() == SCIP_OBJSENSE_MINIMIZE && (convertToExternalValue(solution->getObjectiveFunctionValue()) - updateIncumbent->getObjective() ) > MINEPSILON ) ||
         ( instance->getOrigObjSense() == SCIP_OBJSENSE_MAXIMIZE && ( updateIncumbent->getObjective() - convertToExternalValue(solution->getObjectiveFunctionValue()) ) > MINEPSILON ) )
   {
      // std::cout << "################### SCIP: going to set the incumbnet !!! ##############" << std::endl;
      // std::cout << "################### in SCIP = " <<  convertToExternalValue(solution->getObjectiveFuntionValue()) << ", from outside = " << updateIncumbent->getObjective()  << std::endl;
      std::ostringstream s;
      s <<  ugsComm->getSolverName(updateIncumbent->getUpdatedSolverId())  << "-" << updateIncumbent->getSeqNumber() << ".sol";
      SCIP_SOL *origSol;
      SCIP_CALL_ABORT( SCIPcreateOrigSol(scip, &origSol, NULL) );
      SCIP_Bool partial = FALSE;
      SCIP_Bool error = FALSE;
      SCIP_CALL_ABORT( SCIPreadSolFile(scip, s.str().c_str(), origSol, FALSE, &partial, &error ) );

      // std::cout << "################## partial = " << partial << ", error = " << error << " ##############" << std::endl;

      DEF_SCIP_PARA_COMM( scipParaComm, paraComm);

      if( solution ) delete solution;
      if( (!partial) && (!error) )
      {
         SCIP_Bool stored = FALSE;
         SCIP_CALL_ABORT( SCIPtrySol(scip, origSol, FALSE, TRUE, TRUE, TRUE, FALSE, &stored) );

         // std::cout << "################## stored = " << stored << " ##############" << std::endl;

         if( stored )
         {
            SCIP_SOL *sol = SCIPgetBestSol(scip);
            int nVars = SCIPgetNVars(scip);
            SCIP_VAR **vars = SCIPgetVars(scip);
            SCIP_Real *vals = new SCIP_Real[nVars];
            SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );
            solution = scipParaComm->createScipParaSolution(
                           0,
                           SCIPgetSolTransObj(scip, sol),  // Only this value may be used
                           nVars,
                           vars,
                           vals
                           );
            delete [] vals;
         }
         else
         {
            solution = scipParaComm->createScipParaSolution(
                           0,
                           convertToInternalValue(updateIncumbent->getObjective()), // Only this value may be used
                           0,
                           (SCIP_VAR **)0,
                           0
                           );
         }
      }
      else
      {
         solution = scipParaComm->createScipParaSolution(
                        0,
                        convertToInternalValue(updateIncumbent->getObjective()), // Only this value may be used
                        0,
                        (SCIP_VAR **)0,
                        0
                        );
      }

      SCIP_CALL_ABORT( SCIPfreeSol(scip, &origSol) );

      delete updateIncumbent;

      return true;

   }
   else
   {
      delete updateIncumbent;
      return false;
   }
}

/** write ugs incumbent solution **/
void
ScipParaInitiator::writeUgsIncumbentSolution(
      UGS::UgsParaCommMpi *ugsComm
      )
{
   /* Write out the solution */
   seqNumber++;
   std::ostringstream s;
   s << ugsComm->getMySolverName() << "-" << seqNumber << ".sol";
   FILE *fp = fopen(s.str().c_str(), "w");
   if( fp == NULL )
   {
      fprintf (stderr, "Cannot open solution file to write. File name = %s\n", s.str().c_str());
      exit(1);
   }

   SCIP_SOL* sol = SCIPgetBestSol(scip);

   assert(sol);
   fprintf( fp, "# ");

   SCIP_CALL_ABORT( SCIPprintSol(scip, sol, fp, TRUE) );

   fclose(fp);

   UGS::UgsUpdateIncumbent *uui = new UGS::UgsUpdateIncumbent(ugsComm->getMySolverId(), seqNumber, convertToExternalValue(solution->getObjectiveFunctionValue()) );
   uui->send(ugsComm->getMyComm(),0);
   // std::cout << "Rank" << ugsComm->getRank() << " Sent to 0: " << uui->toString() << std::endl;
   delete uui;

   return;
}
#endif
