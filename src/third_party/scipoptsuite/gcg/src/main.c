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

/**@file   main.c
 * @brief  Main file for C compilation
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Christian Puchert
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <string.h>

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/scipshell.h"
#include "gcgplugins.h"
#include "cons_decomp.h"

#include "relax_gcg.h"
#include "cons_decomp.h"
#include "gcg.h"

#if SCIP_VERSION < 801
#error GCG can only be compiled with SCIP version 8.0.1 or higher
#endif


/** read in parameter file
 * @returns SCIP return code */
static
SCIP_RETCODE readParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< parameter file name */
   )
{
   if( SCIPfileExists(filename) )
   {
      SCIPinfoMessage(scip, NULL, "reading user parameter file <%s>\n", filename);
      SCIP_CALL( SCIPreadParams(scip, filename) );
   }
   else
      SCIPinfoMessage(scip, NULL, "user parameter file <%s> not found - using default parameters\n", filename);

   return SCIP_OKAY;
}

/** runs GCG from the command line
 * @returns SCIPreturn code */
static
SCIP_RETCODE fromCommandLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< input file name */
   const char*           decname             /**< decomposition file name (or NULL) */
   )
{
   /********************
    * Problem Creation *
    ********************/

   SCIPinfoMessage(scip, NULL, "\nread problem <%s>\n", filename);
   SCIPinfoMessage(scip, NULL, "============\n\n");
   SCIP_CALL( SCIPreadProb(scip, filename, NULL) );

   SCIP_CALL( GCGtransformProb(scip) );

   if( decname != NULL )
   {
      SCIPinfoMessage(scip, NULL, "\nread decomposition <%s>\n", decname);
      SCIPinfoMessage(scip, NULL, "==================\n\n");
      SCIP_CALL( SCIPreadProb(scip, decname, NULL) );
   }

   /*******************
    * Problem Solving *
    *******************/
   SCIPinfoMessage(scip, NULL, "\nsolve problem\n");
   SCIPinfoMessage(scip, NULL, "=============\n\n");

   SCIP_CALL( GCGsolve(scip) );

   SCIPinfoMessage(scip, NULL, "\nprimal solution:\n");
   SCIPinfoMessage(scip, NULL, "================\n\n");
   SCIP_CALL( SCIPprintBestSol(scip, NULL, FALSE) );

   /**************
    * Statistics *
    **************/

   SCIPinfoMessage(scip, NULL, "\nStatistics\n");
   SCIPinfoMessage(scip, NULL, "==========\n");

   SCIP_CALL( GCGprintStatistics(scip, NULL) );

   return SCIP_OKAY;
}

/** evaluates command line parameters and runs GCG appropriately in the given SCIP instance
 * @returns SCIP return code */
static
SCIP_RETCODE SCIPprocessGCGShellArguments(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   argc,               /**< number of shell parameters */
   char**                argv,               /**< array with shell parameters */
   const char*           defaultsetname      /**< name of default settings file */
   )
{  /*lint --e{850}*/
   char* probname = NULL;
   char* decname = NULL;
   char* settingsname = NULL;
   char* mastersetname = NULL;
   char* logname = NULL;
   SCIP_Bool quiet;
   SCIP_Bool paramerror;
   SCIP_Bool interactive;
   SCIP_Real primalreference = SCIP_UNKNOWN;
   SCIP_Real dualreference = SCIP_UNKNOWN;
   const char* dualrefstring;
   const char* primalrefstring;
   int i;


   /********************
    * Parse parameters *
    ********************/

   quiet = FALSE;
   paramerror = FALSE;
   interactive = FALSE;
   primalrefstring = NULL;
   dualrefstring = NULL;
   for( i = 1; i < argc; ++i )
   {
      if( strcmp(argv[i], "-l") == 0 )
      {
         i++;
         if( i < argc )
            logname = argv[i];
         else
         {
            SCIPinfoMessage(scip, NULL, "missing log filename after parameter '-l'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-q") == 0 )
         quiet = TRUE;
      else if( strcmp(argv[i], "-s") == 0 )
      {
         i++;
         if( i < argc )
            settingsname = argv[i];
         else
         {
            SCIPinfoMessage(scip, NULL, "missing settings filename after parameter '-s'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-m") == 0 )
      {
         i++;
         if( i < argc )
            mastersetname = argv[i];
         else
         {
            SCIPinfoMessage(scip, NULL, "missing settings filename for master program after parameter '-m'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-f") == 0 )
      {
         i++;
         if( i < argc )
            probname = argv[i];
         else
         {
            SCIPinfoMessage(scip, NULL, "missing problem filename after parameter '-f'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-d") == 0 )
      {
         i++;
         if( i < argc )
            decname = argv[i];
         else
         {
            SCIPinfoMessage(scip, NULL, "missing decomposition filename after parameter '-d'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-c") == 0 )
      {
         i++;
         if( i < argc )
         {
            SCIP_CALL( SCIPaddDialogInputLine(scip, argv[i]) );
            interactive = TRUE;
         }
         else
         {
            SCIPinfoMessage(scip, NULL, "missing command line after parameter '-c'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-b") == 0 )
      {
         i++;
         if( i < argc )
         {
            SCIP_FILE* file;

            file = SCIPfopen(argv[i], "r");
            if( file == NULL )
            {
               SCIPinfoMessage(scip, NULL, "cannot read command batch file <%s>\n", argv[i]);
               SCIPprintSysError(argv[i]);
               paramerror = TRUE;
            }
            else
            {
               while( !SCIPfeof(file) )
               {
                  char buffer[SCIP_MAXSTRLEN];

                  (void)SCIPfgets(buffer, SCIP_MAXSTRLEN, file);
                  if( buffer[0] != '\0' )
                  {
                     SCIP_CALL( SCIPaddDialogInputLine(scip, buffer) );
                  }
               }
               SCIPfclose(file);
               interactive = TRUE;
            }
         }
         else
         {
            SCIPinfoMessage(scip, NULL, "missing command batch filename after parameter '-b'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-o") == 0 )
      {
         if( i >= argc - 2 )
         {
            printf("wrong usage of reference objective parameter '-o': -o <primref> <dualref>\n");
            paramerror = TRUE;
         }
         else
         {
            /* do not parse the strings directly, the settings could still influence the value of +-infinity */
            primalrefstring = argv[i + 1];
            dualrefstring = argv[i+2];
         }
         i += 2;
      }
      else
      {
         SCIPinfoMessage(scip, NULL, "invalid parameter <%s>\n", argv[i]);
         paramerror = TRUE;
      }
   }
   if( interactive && probname != NULL )
   {
      SCIPinfoMessage(scip, NULL, "cannot mix batch mode '-c' and '-b' with file mode '-f'\n");
      paramerror = TRUE;
   }
   if( probname == NULL && decname != NULL )
   {
      SCIPinfoMessage(scip, NULL, "cannot read decomposition file without given problem\n");
      paramerror = TRUE;
   }

   if( !paramerror )
   {

      /***********************************
       * create log file message handler *
       ***********************************/

      if( quiet )
      {
         SCIPsetMessagehdlrQuiet(scip, quiet);
      }

      if( logname != NULL )
      {
         SCIPsetMessagehdlrLogfile(scip, logname);
      }


      /***********************************
       * Version and library information *
       ***********************************/

      SCIPprintVersion(scip, NULL);
      SCIPinfoMessage(scip, NULL, "\n");

      SCIPprintExternalCodes(scip, NULL);
      SCIPinfoMessage(scip, NULL, "\n");

      /*****************
       * Load settings *
       *****************/

      if( settingsname != NULL )
      {
         SCIP_CALL( readParams(scip, settingsname) );
      }
      else if( defaultsetname != NULL )
      {
         SCIP_CALL( readParams(scip, defaultsetname) );
      }

      if( mastersetname != NULL )
      {
         SCIP_CALL( readParams(GCGgetMasterprob(scip), mastersetname) );
      }

      /**************
       * Start SCIP *
       **************/

      if( probname != NULL )
      {
         SCIP_Bool validatesolve = FALSE;

         if( primalrefstring != NULL && dualrefstring != NULL )
         {
            char *endptr;
            if( ! SCIPparseReal(scip, primalrefstring, &primalreference, &endptr) ||
                     ! SCIPparseReal(scip, dualrefstring, &dualreference, &endptr) )
            {
               printf("error parsing primal and dual reference values for validation: %s %s\n", primalrefstring, dualrefstring);
               return SCIP_ERROR;
            }
            else
               validatesolve = TRUE;
         }
         SCIP_CALL( fromCommandLine(scip, probname, decname) );

         /* validate the solve */
         if( validatesolve )
         {
            SCIP_CALL( SCIPvalidateSolve(scip, primalreference, dualreference, SCIPfeastol(scip), FALSE, NULL, NULL, NULL) );
         }
      }
      else
      {
         SCIPinfoMessage(scip, NULL, "\n");
         SCIP_CALL( SCIPstartInteraction(scip) );
      }
   }
   else
   {
      SCIPinfoMessage(scip, NULL, "\nsyntax: %s [-l <logfile>] [-q] [-s <settings>] [-f <problem>] [-m <mastersettings>] [-d <decomposition>] [-b <batchfile>] [-c \"command\"]\n"
            "  -l <logfile>           : copy output into log file\n"
            "  -q                     : suppress screen messages\n"
            "  -s <settings>          : load parameter settings (.set) file\n"
            "  -m <mastersettings>    : load parameter settings for master program (.set) file\n",
            argv[0]);
      SCIPinfoMessage(scip, NULL, "  -f <problem>           : load and solve problem file\n"
         "  -d <decomposition>     : load decomposition file\n"
         "  -o <primref> <dualref> : pass primal and dual objective reference values for validation at the end of the solve\n"
         "  -b <batchfile>         : load and execute dialog command batch file (can be used multiple times)\n"
         "  -c \"command\"           : execute single line of dialog commands (can be used multiple times)\n\n");
   }

   return SCIP_OKAY;
}

/** runs the interactive shell
 * @returns SCIP return code */
static
SCIP_RETCODE SCIPrunGCGShell(
   int                   argc,               /**< number of shell parameters */
   char**                argv,               /**< array with shell parameters */
   const char*           defaultsetname      /**< name of default settings file */
   )
{
   SCIP* scip = NULL;

   /*********
    * Setup *
    *********/

   /* initialize SCIP */
   SCIP_CALL( SCIPcreate(&scip) );
   GCGprintVersion(scip, NULL);


   /* include coloring plugins */
   SCIP_CALL( SCIPincludeGcgPlugins(scip) );

   /**********************************
    * Process command line arguments *
    **********************************/
   SCIP_CALL( SCIPprocessGCGShellArguments(scip, argc, argv, defaultsetname) );


   /********************
    * Deinitialization *
    ********************/

   SCIP_CALL( SCIPfree(&scip) );

   BMScheckEmptyMemory();

   return SCIP_OKAY;
}

/** main function called first
 * @returns 0 if success, -1 otherwise */
int
main(
   int                   argc,               /**< number of arguments */
   char**                argv                /**< array of arguments */
   )
{
  SCIP_RETCODE retcode;

  retcode = SCIPrunGCGShell(argc, argv, "gcg.set");

  if( retcode != SCIP_OKAY )
  {
     SCIPprintError(retcode);
     return -1;
  }

  return 0;
}
