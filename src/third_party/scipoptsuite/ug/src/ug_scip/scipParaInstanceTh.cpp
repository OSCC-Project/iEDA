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

/**@file    scipParaInstanceTh.cpp
 * @brief   ScipParaInstance extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "ug_bb/bbParaComm.h"
#include "scipParaInstanceTh.h"
#include "scip/scipdefplugins.h"
// #ifdef UG_DEBUG_SOLUTION
// #include "scip/debug.h"
// #include "scip/struct_scip.h"
// #include "scip/struct_set.h"
// #endif

using namespace UG;
using namespace ParaSCIP;

const static char *PRESOLVED_INSTANCE = "presolved.cip";

void
ScipParaInstance::copyScipEnvironment(
      SCIP **targetscip
      )
{
   char probname[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(*targetscip != NULL);
   SCIP_Bool success = TRUE;

   /* copy all plugins and settings */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
   SCIP_CALL_ABORT( SCIPcopyPlugins(scip, *targetscip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
         TRUE, TRUE, TRUE, TRUE, &success) );
#else
   #if SCIP_APIVERSION >= 100
      #if SCIP_APIVERSION >= 101
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, *targetscip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #else
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, *targetscip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #endif
   #elif SCIP_APIVERSION >= 17
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, *targetscip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #else
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, *targetscip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #endif
#endif
   SCIP_CALL_ABORT( SCIPcopyParamSettings(scip, *targetscip) );

   /* create the variable mapping hash map */
   SCIP_HASHMAP* varmap = 0;
   if( SCIPgetNVars(scip) > 0 )
   {
      SCIP_CALL_ABORT( SCIPhashmapCreate(&varmap, SCIPblkmem(*targetscip), SCIPgetNVars(scip)) );
   }
   SCIP_HASHMAP* conssmap = 0;
   if( SCIPgetNConss(scip) > 0 )
   {
      SCIP_CALL_ABORT( SCIPhashmapCreate(&conssmap, SCIPblkmem(*targetscip), SCIPgetNConss(scip)) );   
   }

   /* create problem in the target SCIP */
   /* get name of the original problem and add the suffix string */
   (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_%s", SCIPgetProbName(scip), "solver");
   // SCIP_CALL_ABORT( SCIPcreateProb(*targetscip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );
   /* create problem in the target SCIP and copying the source original problem data */
   SCIP_CALL_ABORT( SCIPcopyProb(scip, *targetscip, varmap, conssmap, TRUE, probname) );

   /* copy all variables and constraints */
   if(  SCIPgetNVars(scip) > 0 )
   {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      SCIP_CALL_ABORT( SCIPcopyVars(scip, *targetscip, varmap, conssmap, TRUE) );
#else
      SCIP_CALL_ABORT( SCIPcopyVars(scip, *targetscip, varmap, conssmap, NULL, NULL, 0, TRUE) );
#endif
   } 
   if( SCIPgetNConss(scip) > 0 )
   {
      SCIP_CALL_ABORT( SCIPcopyConss(scip, *targetscip, varmap, conssmap, TRUE, FALSE, &success) );
   }

#if SCIP_APIVERSION > 39
   if( success )
   {
      SCIP_Bool valid;

      /* copy the Benders' decomposition plugins explicitly, because it requires the variable mapping hash map */
      SCIP_CALL_ABORT( SCIPcopyBenders(scip, *targetscip, NULL, TRUE, &valid) );
   }
#endif

   if( !success )
   {
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIPhashmapFree(&conssmap);
      }
      if(  SCIPgetNVars(scip) > 0 )
      {
         SCIPhashmapFree(&varmap);
      }
      std::cerr << "Some constraint handler did not perform a valid copy. Cannot solve this instance." << std::endl;
      exit(1);
   }

   /* free hash map */
   if( SCIPgetNConss(scip) > 0 )
   {
      SCIPhashmapFree(&conssmap);
   }
   if(  SCIPgetNVars(scip) > 0 )
   {
      SCIPhashmapFree(&varmap);
   }

}

int
ScipParaInstanceTh::bcast(
      ParaComm *comm,
      int root,
      int method
      )
{
   DEF_PARA_COMM( commTh, comm);

#if 0  // sender side creation

   if( commTh->getRank() == root )
   {
      for( int i = 0; i < commTh->getSize(); i++ )
      {
         if( i != root )
         {
            SCIP *newScip;
            SCIP_CALL_ABORT( SCIPcreate(&newScip) );
            copyScipEnvironment(&newScip);   // this copy call should be serialized
            PARA_COMM_CALL(
               commTh->uTypeSend((void *)newScip, ParaInstanceType, i, TagParaInstance)
            );
         }
      }
   }
   else
   {
      // SCIP *received;
      PARA_COMM_CALL(
         commTh->uTypeReceive((void **)&scip, ParaInstanceType, root, TagParaInstance)
      );
      // scip = received;
   }
#else  // receiver side creation
   if( commTh->getRank() == root )
   {

      SCIP *tempScip;
      SCIP_Bool success = TRUE;

      commTh->lockApp();
      SCIP_CALL_ABORT( SCIPcreate(&tempScip) );

      /* copy all plugins and settings */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, tempScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, &success) );
#else
   #if SCIP_APIVERSION >= 100
      #if SCIP_APIVERSION >= 101
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, tempScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #else
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, tempScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #endif
   #elif SCIP_APIVERSION >= 17
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, tempScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #else
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, tempScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #endif
#endif
      SCIP_CALL_ABORT( SCIPcopyParamSettings(scip, tempScip) );

      /* create the variable mapping hash map */
      SCIP_HASHMAP* varmap = 0;
      if( SCIPgetNVars(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPhashmapCreate(&varmap, SCIPblkmem(tempScip), SCIPgetNVars(scip)) );
      }
      SCIP_HASHMAP* conssmap = 0;
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPhashmapCreate(&conssmap, SCIPblkmem(tempScip), SCIPgetNConss(scip)) );
      }
      /* create problem in the target SCIP */
      SCIP_CALL_ABORT( SCIPcopyProb(scip, tempScip, varmap, conssmap, TRUE, "") );

      // commTh->lockApp();
      /* copy all variables and constraints */
      if( SCIPgetNVars(scip) > 0 )
      {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
         SCIP_CALL_ABORT( SCIPcopyVars(scip, tempScip, varmap, conssmap, TRUE) );
#else
         SCIP_CALL_ABORT( SCIPcopyVars(scip, tempScip, varmap, conssmap, NULL, NULL, 0, TRUE) );
#endif
      }
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPcopyConss(scip, tempScip, varmap, conssmap, TRUE, FALSE, &success) );
      }

#if SCIP_APIVERSION > 39
      if( success )
      {
         SCIP_Bool valid;

         /* copy the Benders' decomposition plugins explicitly, because it requires the variable mapping hash map */
         SCIP_CALL_ABORT( SCIPcopyBenders(scip, tempScip, NULL, TRUE, &valid) );
      }
#endif

      commTh->unlockApp();
      if( !success )
      {
         if( SCIPgetNConss(scip) > 0 )
         {
            SCIPhashmapFree(&conssmap);
         }
         if( SCIPgetNVars(scip) > 0 )
         {
            SCIPhashmapFree(&varmap);
         }
         SCIPfree(&tempScip);
         std::cerr << "Some constraint handler did not perform a valid copy. Cannot solve this instance." << std::endl;
         exit(1);
      }

      nVars = SCIPgetNVars(scip);                // original number
      varIndexRange = nVars;
      int n = SCIPgetNVars(tempScip);   // copy may increase the number

      if( nVars == n )
      {
         if( SCIPgetNConss(scip) > 0 )
         {
            SCIPhashmapFree(&conssmap);
         }
         if( SCIPgetNVars(scip) > 0 )
         {
            SCIPhashmapFree(&varmap);
         }
         SCIPfree(&tempScip);
         std::cout << "** ParaScipInstance copy does not increase the number of variables. **" << std::endl;
      }
      else
      {
         assert(n > nVars);
         std::cout << "** ParaScipInstance copy increased the number of variables. **" << std::endl;
         varIndexRange = SCIPgetNTotalVars(tempScip);
         copyIncreasedVariables = true;
         /**
         // mapToOriginalIndecies = new int[n];
         mapToOriginalIndecies = new int[SCIPgetNTotalVars(tempScip)];   // need to allocate enough for SCIPvarGetIndex(copyvar)
         for( int i = 0; i < SCIPgetNTotalVars(tempScip); i++ )
         {
            mapToOriginalIndecies[i] = -1;
         }
         SCIP_VAR **tempVars = SCIPgetVars(tempScip);
         for( int i = 0; i < n; i++ )
         {
            // mapToOriginalIndecies[i] = SCIPvarGetIndex(tempVars[i]);
            mapToOriginalIndecies[SCIPvarGetIndex(tempVars[i])] = i;
         }

         SCIP_CALL_ABORT( SCIPtransformProb(scip));
         orgScip = scip;
         nVars = n;
         varIndexRange = SCIPgetNTotalVars(tempScip);
         scip = tempScip;

         if( SCIPgetNConss(scip) > 0 )
         {
            SCIPhashmapFree(&conssmap);
         }
         if( SCIPgetNVars(scip) > 0 )
         {
            SCIPhashmapFree(&varmap);
         }
         SCIP_CALL_ABORT( SCIPtransformProb(scip));
         std::cout << "** ParaScipInstance is copied twice. **" << std::endl;
         ***/
      }

      if( method == 0 )
      {
         for( int i = 0; i < commTh->getSize(); i++ )
         {
            if( i != root )
            {
               PARA_COMM_CALL(
                  commTh->uTypeSend((void *)scip, ParaInstanceType, i, TagParaInstance)
               );
            }
         }
      }
      else
      {
         for( int i = 0; i < commTh->getSize(); i++ )
         {
            if( i != root )
            {
               PARA_COMM_CALL(
                  commTh->send(NULL, 0, ParaBYTE, i, TagParaInstance)
               );
            }
         }
      }
   }
   else
   {
      if( method == 0 )
      {
         SCIP *received;

         PARA_COMM_CALL(
            commTh->uTypeReceive((void **)&received, ParaInstanceType, root, TagParaInstance)
         );

         commTh->lockApp();
         SCIP_CALL_ABORT( SCIPcreate(&scip) );
         char probname[SCIP_MAXSTRLEN];

         assert(received != NULL);
         assert(scip != NULL);
         SCIP_Bool success = TRUE;

         /* copy all plugins and settings */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
         SCIP_CALL_ABORT( SCIPcopyPlugins(received, scip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
               TRUE, TRUE, TRUE, TRUE, &success) );
#else
   #if SCIP_APIVERSION >= 100
      #if SCIP_APIVERSION >= 101
          SCIP_CALL_ABORT( SCIPcopyPlugins(received, scip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #else
          SCIP_CALL_ABORT( SCIPcopyPlugins(received, scip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #endif
   #elif SCIP_APIVERSION >= 17
      SCIP_CALL_ABORT( SCIPcopyPlugins(received, scip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #else
      SCIP_CALL_ABORT( SCIPcopyPlugins(received, scip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #endif
#endif
         // std::cout <<  "copy plugins : success = " << success << std::endl;
         if( !success )
         {
             std::cout << "Error in SCIPcopyPlugins" << std::endl;
             abort();
         }
         SCIP_CALL_ABORT( SCIPcopyParamSettings(received, scip) );


         /* create the variable mapping hash map */
         SCIP_HASHMAP* varmap = 0;
         if( SCIPgetNVars(received) > 0 )
         {
            SCIP_CALL_ABORT( SCIPhashmapCreate(&varmap, SCIPblkmem(scip), SCIPgetNVars(received)) );
         }
         SCIP_HASHMAP* conssmap = 0;
         if( SCIPgetNConss(received) > 0 )
         {
            SCIP_CALL_ABORT( SCIPhashmapCreate(&conssmap, SCIPblkmem(scip), SCIPgetNConss(received)) );
         }

         /* create problem in the target SCIP */
         /* get name of the original problem and add the suffix string */
         (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_%s", SCIPgetProbName(received), "solver");
         // SCIP_CALL_ABORT( SCIPcreateProb(scip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );
         /* create problem in the target SCIP and copying the source original problem data */
         SCIP_CALL_ABORT( SCIPcopyProb(received, scip, varmap, conssmap, TRUE, probname) );

         // commTh->lockApp();
         /* copy all variables and constraints */
         if( SCIPgetNVars(received) > 0 )
         {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
            SCIP_CALL_ABORT( SCIPcopyVars(received, scip, varmap, conssmap, TRUE) );
#else
            SCIP_CALL_ABORT( SCIPcopyVars(received, scip, varmap, conssmap, NULL, NULL, 0, TRUE) );
#endif
         }
         if( SCIPgetNConss(received) > 0 )
         {
            SCIP_CALL_ABORT( SCIPcopyConss(received, scip, varmap, conssmap, TRUE, FALSE, &success) );
         }

#if SCIP_APIVERSION > 39
         if( success )
         {
            SCIP_Bool valid;

            /* copy the Benders' decomposition plugins explicitly, because it requires the variable mapping hash map */
            SCIP_CALL_ABORT( SCIPcopyBenders(received, scip, NULL, TRUE, &valid) );
         }
#endif

         commTh->unlockApp();
         if( !success )
         {
            if( SCIPgetNConss(received) > 0 )
            {
               SCIPhashmapFree(&conssmap);
            }
            if( SCIPgetNVars(received) > 0 )
            {
               SCIPhashmapFree(&varmap);
            }
            std::cerr << "Some constraint handler did not perform a valid copy. Cannot solve this instance." << std::endl;
            exit(1);
         }

         nVars = SCIPgetNVars(received);
         varIndexRange = nVars;
         int n = SCIPgetNVars(scip);

	 // std::cout << "nVars = " << nVars << ", varIndexRange = " << varIndexRange << ", n = " << n << std::endl;

         // assert( nVars == n );
         assert( nVars <= n );
         if( nVars < n )
         {
            copyIncreasedVariables = true;
            varIndexRange = n;
         }
         // mapToProbIndecies = new int[nVars];
         // mapToProbIndecies = new int[SCIPgetNTotalVars(scip)];
         mapToOriginalIndecies = new int[SCIPgetNTotalVars(scip)];    // need to allocate enough for SCIPvarGetIndex(copyvar)
         mapToSolverLocalIndecies = new int[SCIPgetNTotalVars(scip)];
         for( int i = 0; i < SCIPgetNTotalVars(scip); i++ )
         {
            // mapToProbIndecies[i] = -1;
            mapToOriginalIndecies[i] = -1;
            mapToSolverLocalIndecies[i] = -1;
         }
         // SCIP_VAR **srcVars = SCIPgetVars(received);
         SCIP_VAR **srcVars = SCIPgetVars(received);

         assert(SCIPgetNTotalVars(scip) >= SCIPgetNVars(received));

         // SCIP_VAR **targetVars = SCIPgetVars(scip);
         // int countOrigVars = 0;
         // for( int i = 0; i < varIndexRange; i++ )
         //for( int i = 0; i < SCIPgetNTotalVars(scip); i++ )
         for( int i = 0; i < SCIPgetNVars(received); i++ )
         {
            SCIP_VAR* copyvar = (SCIP_VAR*)SCIPhashmapGetImage(varmap, (void*)srcVars[i]);

            // std::cout << i << ": index = " << SCIPvarGetIndex(copyvar) << std::endl;
            // assert(SCIPvarGetIndex(copyvar) >= 0);
            // if( copyvar && SCIPvarGetProbindex(copyvar) < nVars )
            // {
            // assert( copyvar );
            if( copyvar )
            {
               // assert(SCIPvarGetProbindex(copyvar) >= 0);
               // mapToOriginalIndecies[SCIPvarGetIndex(copyvar)] = SCIPvarGetProbindex(copyvar);
               mapToOriginalIndecies[SCIPvarGetIndex(copyvar)] = i;
               mapToSolverLocalIndecies[i] = SCIPvarGetIndex(copyvar);
               // mapToProbIndecies[SCIPvarGetIndex(copyvar)] = SCIPvarGetProbindex(copyvar);
               // assert( copyvar == targetVars[mapToProbIndecies[SCIPvarGetIndex(copyvar)]] );
               // std::cout << "mapToOriginalIndecies[" << SCIPvarGetIndex(copyvar) << "] = " << SCIPvarGetProbindex(copyvar) << std::endl;
               // std::cout << i << ": " << SCIPvarGetName(copyvar) << std::endl;
               // std::cout << i << ": " << SCIPvarGetName(srcVars[SCIPvarGetProbindex(copyvar)]) << std::endl;
               // std::cout << i << ": "  << SCIPvarGetName(srcVars[mapToOriginalIndecies[SCIPvarGetIndex(copyvar)]]) << std::endl;
               // countOrigVars++;
            }
            /*
            else
            {
               // throw "Logical error occurred";
               // assert( i >= nVars );
               mapToOriginalIndecies[i] = -1;  // should not be used
            }
            */
         }
         // std::cout << "%%% debg: nVars = " << nVars << std::endl;
         // assert(nVars == countOrigVars);

         /* free hash map */
         if( SCIPgetNConss(received) > 0 )
         {
            SCIPhashmapFree(&conssmap);
         }
         if( SCIPgetNVars(received) > 0 )
         {
            SCIPhashmapFree(&varmap);
         }
      }
      else
      {
         PARA_COMM_CALL(
             commTh->receive( NULL, 0, ParaBYTE, root, TagParaInstance )
         );
      }

   }
#endif

   return 0;

}

ScipParaInstanceTh::ScipParaInstanceTh(
      SCIP *inScip,
      int  method
      ) : ScipParaInstance(inScip)
{
   if( method == 1)
   {
      SCIP_CALL_ABORT( SCIPwriteTransProblem(scip, PRESOLVED_INSTANCE, "cip", TRUE ) );
   }
}

/** create presolved problem instance that is solved by ParaSCIP */
void
ScipParaInstance::createProblem(
     SCIP *inScip,
     int  method,
     bool noPreprocessingInLC,  // LC preprocesing settings
     bool usetRootNodeCuts,
     ScipDiffParamSet  *scipDiffParamSetRoot,
     ScipDiffParamSet  *scipDiffParamSet,
     char *settingsNameLC,      // LC preprocesing settings
     char *isolname
     )
{
   if( method == 1 )
   {
      scip = inScip;
      // scipDiffParamSet->setParametersInScip(scip);
      SCIP_CALL_ABORT( SCIPreadProb(scip, PRESOLVED_INSTANCE, "cip" ) );
   }
// #ifdef UG_DEBUG_SOLUTION
//    if( method == 0 )
//    {
//       assert(scip->set->debugsoldata == NULL);
//       SCIP_CALL_ABORT( SCIPdebugSolDataCreate(&((scip->set)->debugsoldata)));
//       SCIPdebugSetMainscipset(scip->set);
//    }
// #endif
   if( method == 0 && SCIPgetStage(inScip) == SCIP_STAGE_INIT )
   {
      if( SCIPgetStage(scip) == SCIP_STAGE_PROBLEM)
      {
         SCIP_CALL_ABORT( SCIPtransformProb(scip));
      }
      if( scip == inScip ) return;
 
      /****************************************/
      /* the following code need to be tested */
      /****************************************/
      std::cout << "* If you use check mechanism, you should check the following codes. *" << std::endl;
      SCIP_Bool success = TRUE;
      char probname[SCIP_MAXSTRLEN];
      /* copy all plugins and settings */
#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, inScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
          TRUE, TRUE, TRUE, TRUE, &success) );
#else
   #if SCIP_APIVERSION >= 100
      #if SCIP_APIVERSION >= 101
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, inScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #else
          SCIP_CALL_ABORT( SCIPcopyPlugins(scip, inScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
      #endif
   #elif SCIP_APIVERSION >= 17
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, inScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #else
      SCIP_CALL_ABORT( SCIPcopyPlugins(scip, inScip, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
           TRUE, TRUE, TRUE, TRUE, FALSE, &success) );
   #endif
#endif
      SCIP_CALL_ABORT( SCIPcopyParamSettings(scip, inScip) );

      /* create the variable mapping hash map */
      SCIP_HASHMAP* varmap = 0;
      if( SCIPgetNVars(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPhashmapCreate(&varmap, SCIPblkmem(inScip), SCIPgetNVars(scip)) );
      }
      SCIP_HASHMAP* conssmap = 0;
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPhashmapCreate(&conssmap, SCIPblkmem(inScip), SCIPgetNConss(scip)) );
      }

      /* create problem in the target SCIP */
      /* get name of the original problem and add the suffix string */
      (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_%s", SCIPgetProbName(scip), "solver_created");
      SCIP_CALL_ABORT( SCIPcreateProb(inScip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );
      /* create problem in the target SCIP and copying the source original problem data */
      SCIP_CALL_ABORT( SCIPcopyProb(scip, inScip, varmap, conssmap, TRUE, probname) );

      /* copy all variables and constraints */
      if( SCIPgetNVars(scip) > 0 )
      {
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
         SCIP_CALL_ABORT( SCIPcopyVars(scip, inScip, varmap, conssmap, TRUE) );
#else
         SCIP_CALL_ABORT( SCIPcopyVars(scip, inScip, varmap, conssmap, NULL, NULL, 0, TRUE) );
#endif
      }
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIP_CALL_ABORT( SCIPcopyConss(scip, inScip, varmap, conssmap, TRUE, FALSE, &success) );
      }

#if SCIP_APIVERSION > 39
      if( success )
      {
         SCIP_Bool valid;

         /* copy the Benders' decomposition plugins explicitly, because it requires the variable mapping hash map */
         SCIP_CALL_ABORT( SCIPcopyBenders(scip, inScip, NULL, TRUE, &valid) );
      }
#endif

      if( !success )
      {
         if( SCIPgetNConss(scip) > 0 )
         {
            SCIPhashmapFree(&conssmap);
         }
         if( SCIPgetNVars(scip) > 0 )
         {
            SCIPhashmapFree(&varmap);
         }
         std::cerr << "Some constraint handler did not perform a valid copy. Cannot solve this instance." << std::endl;
         exit(1);
      }

      /* free hash map */
      if( SCIPgetNConss(scip) > 0 )
      {
         SCIPhashmapFree(&conssmap);
      }
      if( SCIPgetNVars(scip) > 0 )
      {
         SCIPhashmapFree(&varmap);
      }
   }
   if( method == 2 )
   {
      std::cout << "You should use instance transfer method 0 or 1!" << std::endl;
      exit(0);
   }
}

