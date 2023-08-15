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

/**@file   clscons_miplibconstypes.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies constraints according to their miplib constypes
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clscons_miplibconstypes.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "miplibconstype"       /**< name of classifier */
#define DEC_DESC                  "miplib constypes"     /**< short description of classification*/
#define DEC_PRIORITY              0

#define DEC_ENABLED               TRUE


/*
 * Data structures
 */

/** classifier handler data */
struct DEC_ClassifierData
{
};


/*
 * Local methods
 */

/* put your local methods here, and declare them static */


/*
 * classifier callback methods
 */

/** destructor of classifier to free user data (called when GCG is exiting) */
#define classifierFree NULL

static
DEC_DECL_CONSCLASSIFY(classifierClassify)
{
   gcg::DETPROBDATA* detprobdata;
   if( transformed )
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataPresolved(scip);
   }
   else
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataOrig(scip);
   }

   std::vector<int> nfoundconstypesrangedsinglecount( (int) SCIP_CONSTYPE_GENERAL + 1, 0 );
   std::vector<int> nfoundconstypesrangeddoublecount( (int) SCIP_CONSTYPE_GENERAL + 1, 0 );

   std::vector<int> classforcons = std::vector<int>( detprobdata->getNConss(), -1 );
   gcg::ConsPartition* classifier;

   /* firstly, assign all constraints to classindices */
   for( int c = 0; c < detprobdata->getNConss(); ++ c )
   {
      SCIP_CONS* cons;
      SCIP_Real lhs;
      SCIP_Real rhs;
      SCIP_Real* vals = NULL;
      SCIP_VAR** vars = NULL;
      int nvars;
      int i;

      cons = detprobdata->getCons(c);

      nvars =  GCGconsGetNVars(scip, cons );

      lhs = GCGconsGetLhs(scip, cons);
      rhs = GCGconsGetRhs(scip, cons);
      if( nvars != 0 )
      {
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals, nvars));
         SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vars, nvars));
         SCIP_CALL_ABORT( GCGconsGetVals(scip, cons, vals, nvars ) );
         SCIP_CALL_ABORT( GCGconsGetVars(scip, cons, vars, nvars ) );
      }

      for( i = 0; i < nvars; i++ )
      {
         assert(!SCIPisZero(scip, vals[i]) );
      }


      /* is constraint of type SCIP_CONSTYPE_EMPTY? */
      if( nvars == 0 )
      {
         SCIPdebugMsg(scip, "classified as EMPTY: ");
         SCIPdebugPrintCons(scip, cons, NULL);
         nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_EMPTY]++;
         nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_EMPTY]++;
         classforcons[c] = SCIP_CONSTYPE_EMPTY;
         continue;
      }

      /* is constraint of type SCIP_CONSTYPE_FREE? */
      if( SCIPisInfinity(scip, rhs) && SCIPisInfinity(scip, -lhs) )
      {
         SCIPdebugMsg(scip, "classified as FREE: ");
         SCIPdebugPrintCons(scip, cons, NULL);
         nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_FREE]++;
         nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_FREE]++;
         classforcons[c] = SCIP_CONSTYPE_FREE;
         SCIPfreeBufferArray(scip, &vars);
         SCIPfreeBufferArray(scip, &vals);
         continue;
      }

      /* is constraint of type SCIP_CONSTYPE_SINGLETON? */
      if( nvars == 1 )
      {
         SCIPdebugMsg(scip, "classified as SINGLETON: ");
         SCIPdebugPrintCons(scip, cons, NULL);
         nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_SINGLETON] += 2 ;
         nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_SINGLETON]++;
         classforcons[c] = SCIP_CONSTYPE_SINGLETON;
         SCIPfreeBufferArray(scip, &vars) ;
         SCIPfreeBufferArray(scip, &vals) ;
         continue;
      }

      /* is constraint of type SCIP_CONSTYPE_AGGREGATION? */
      if( nvars == 2 && SCIPisEQ(scip, lhs, rhs) )
      {
         SCIPdebugMsg(scip, "classified as AGGREGATION: ");
         SCIPdebugPrintCons(scip, cons, NULL);
         nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_AGGREGATION]++;
         nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_AGGREGATION]++;
         classforcons[c] = SCIP_CONSTYPE_AGGREGATION;
         SCIPfreeBufferArray(scip, &vars) ;
         SCIPfreeBufferArray(scip, &vals) ;
         continue;
      }

      /* is constraint of type SCIP_CONSTYPE_{VARBOUND}? */
      if( nvars == 2 )
      {
         SCIPdebugMsg(scip, "classified as VARBOUND: ");
         SCIPdebugPrintCons(scip, cons, NULL);
         nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_VARBOUND] += 2 ;
         nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_VARBOUND]++;
         classforcons[c] = SCIP_CONSTYPE_VARBOUND;
         SCIPfreeBufferArray(scip, &vars) ;
         SCIPfreeBufferArray(scip, &vals) ;
         continue;
      }

      /* is constraint of type SCIP_CONSTYPE_{SETPARTITION, SETPACKING, SETCOVERING, CARDINALITY, INVKNAPSACK}? */
      {
         SCIP_Real scale;
         SCIP_Real b;
         SCIP_Bool unmatched;
         int nnegbinvars;

         unmatched = FALSE;
         nnegbinvars = 0;

         scale = REALABS(vals[0]);
         for( i = 0; i < nvars && !unmatched; i++ )
         {
            unmatched = unmatched || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_CONTINUOUS;
            unmatched = unmatched || SCIPisLE(scip, SCIPvarGetLbGlobal(vars[i]), -1.0);
            unmatched = unmatched || SCIPisGE(scip, SCIPvarGetUbGlobal(vars[i]), 2.0);
            unmatched = unmatched || !SCIPisEQ(scip, REALABS(vals[i]), scale);

            if( vals[i] < 0.0 )
               nnegbinvars++;
         }

         if( !unmatched )
         {
            if( SCIPisEQ(scip, lhs, rhs) )
            {
               b = rhs/scale + nnegbinvars;
               if( SCIPisEQ(scip, 1.0, b) )
               {
                  SCIPdebugMsg(scip, "classified as SETPARTITION: ");
                  SCIPdebugPrintCons(scip, cons, NULL);
                  nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_SETPARTITION] += 1 ;
                  nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_SETPARTITION]++;
                  classforcons[c] = SCIP_CONSTYPE_SETPARTITION;
                  SCIPfreeBufferArray(scip, &vars) ;
                  SCIPfreeBufferArray(scip, &vals) ;
                  continue;
               }
               else if( SCIPisIntegral(scip, b) && !SCIPisNegative(scip, b) )
               {
                  SCIPdebugMsg(scip, "classified as CARDINALITY: ");
                  SCIPdebugPrintCons(scip, cons, NULL);
                  nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_CARDINALITY] += 1 ;
                  nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_CARDINALITY]++;
                  classforcons[c] = SCIP_CONSTYPE_CARDINALITY;
                  SCIPfreeBufferArray(scip, &vars);
                  SCIPfreeBufferArray(scip, &vals);
                  continue;
               }
            }

            b = rhs/scale + nnegbinvars;
            if( SCIPisEQ(scip, 1.0, b) )
            {
               SCIPdebugMsg(scip, "classified as SETPACKING: ");
               SCIPdebugPrintCons(scip, cons, NULL);
               nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_SETPACKING] += 1 ;
               nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_SETPACKING]++;
               classforcons[c] = SCIP_CONSTYPE_SETPACKING;
               rhs = SCIPinfinity(scip);
            }
            else if( SCIPisIntegral(scip, b) && !SCIPisNegative(scip, b) )
            {
               SCIPdebugMsg(scip, "classified as INVKNAPSACK: ");
               SCIPdebugPrintCons(scip, cons, NULL);
               nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_INVKNAPSACK] += 1 ;
                nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_INVKNAPSACK]++;
                classforcons[c] = SCIP_CONSTYPE_INVKNAPSACK;
               rhs = SCIPinfinity(scip);
            }

            b = lhs/scale + nnegbinvars;
            if( SCIPisEQ(scip, 1.0, b) )
            {
               SCIPdebugMsg(scip, "classified as SETCOVERING: ");
               SCIPdebugPrintCons(scip, cons, NULL);
               nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_SETCOVERING] += 1 ;
               nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_SETCOVERING]++;
               classforcons[c] = SCIP_CONSTYPE_SETCOVERING;
               lhs = -SCIPinfinity(scip);
            }

            if( SCIPisInfinity(scip, -lhs) && SCIPisInfinity(scip, rhs) )
            {
               SCIPfreeBufferArray(scip, &vars);
               SCIPfreeBufferArray(scip, &vals);
               continue;
            }
         }
      }

      /* is constraint of type SCIP_CONSTYPE_{EQKNAPSACK, BINPACKING, KNAPSACK}? */
      /* @todo If coefficients or rhs are not integral, we currently do not check
       * if the constraint could be scaled (finitely), such that they are.
       */
      {
         SCIP_Real b;
         SCIP_Bool unmatched;

         b = rhs;
         unmatched = FALSE;
         for( i = 0; i < nvars && !unmatched; i++ )
         {
            unmatched = unmatched || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_CONTINUOUS;
            unmatched = unmatched || SCIPisLE(scip, SCIPvarGetLbGlobal(vars[i]), -1.0);
            unmatched = unmatched || SCIPisGE(scip, SCIPvarGetUbGlobal(vars[i]), 2.0);
            unmatched = unmatched || !SCIPisIntegral(scip, vals[i]);

            if( SCIPisNegative(scip, vals[i]) )
               b -= vals[i];
         }
         unmatched = unmatched || !detprobdata->isFiniteNonnegativeIntegral(scip, b);

         if( !unmatched )
         {
            if( SCIPisEQ(scip, lhs, rhs) )
            {
               SCIPdebugMsg(scip, "classified as EQKNAPSACK: ");
               SCIPdebugPrintCons(scip, cons, NULL);
               nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_EQKNAPSACK] += 1 ;
               nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_EQKNAPSACK]++;
               classforcons[c] = SCIP_CONSTYPE_EQKNAPSACK;
               SCIPfreeBufferArray(scip, &vars);
               SCIPfreeBufferArray(scip, &vals);
               continue;
            }
            else
            {
               SCIP_Bool matched;

               matched = FALSE;
               for( i = 0; i < nvars && !matched; i++ )
               {
                  matched = matched || SCIPisEQ(scip, b, REALABS(vals[i]));
               }

               SCIPdebugMsg(scip, "classified as %s: ", matched ? "BINPACKING" : "KNAPSACK");
               SCIPdebugPrintCons(scip, cons, NULL);
               nfoundconstypesrangeddoublecount[matched ? SCIP_CONSTYPE_BINPACKING : SCIP_CONSTYPE_KNAPSACK] += 1 ;
               nfoundconstypesrangedsinglecount[matched ? SCIP_CONSTYPE_BINPACKING : SCIP_CONSTYPE_KNAPSACK]++;
               classforcons[c] = matched ? SCIP_CONSTYPE_BINPACKING : SCIP_CONSTYPE_KNAPSACK;

            }

            if( SCIPisInfinity(scip, -lhs) )
            {
               SCIPfreeBufferArray(scip, &vars);
               SCIPfreeBufferArray(scip, &vals);
               continue;
            }
            else
               rhs = SCIPinfinity(scip);
         }
      }

      /* is constraint of type SCIP_CONSTYPE_{INTKNAPSACK}? */
      {
         SCIP_Real b;
         SCIP_Bool unmatched;

         unmatched = FALSE;

         b = rhs;
         unmatched = unmatched || !detprobdata->isFiniteNonnegativeIntegral(scip, b);

         for( i = 0; i < nvars && !unmatched; i++ )
         {
            unmatched = unmatched || SCIPvarGetType(vars[i]) == SCIP_VARTYPE_CONTINUOUS;
            unmatched = unmatched || SCIPisNegative(scip, SCIPvarGetLbGlobal(vars[i]));
            unmatched = unmatched || !SCIPisIntegral(scip, vals[i]);
            unmatched = unmatched || SCIPisNegative(scip, vals[i]);
         }

         if( !unmatched )
         {
            SCIPdebugMsg(scip, "classified as INTKNAPSACK: ");
            SCIPdebugPrintCons(scip, cons, NULL);
            nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_INTKNAPSACK] += 1 ;
            nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_INTKNAPSACK]++;
            classforcons[c] = SCIP_CONSTYPE_INTKNAPSACK;

            if( SCIPisInfinity(scip, -lhs) )
            {
               SCIPfreeBufferArray(scip, &vars);
               SCIPfreeBufferArray(scip, &vals);
               continue;
            }
            else
               rhs = SCIPinfinity(scip);
         }
      }

      /* is constraint of type SCIP_CONSTYPE_{MIXEDBINARY}? */
      {
         SCIP_Bool unmatched;

         unmatched = FALSE;
         for( i = 0; i < nvars && !unmatched; i++ )
         {
            if( SCIPvarGetType(vars[i]) != SCIP_VARTYPE_CONTINUOUS
               && (SCIPisLE(scip, SCIPvarGetLbGlobal(vars[i]), -1.0)
                  || SCIPisGE(scip, SCIPvarGetUbGlobal(vars[i]), 2.0)) )
               unmatched = TRUE;
         }

         if( !unmatched )
         {
            SCIPdebugMsg(scip, "classified as MIXEDBINARY (%d): ", detprobdata->isRangedRow(scip, lhs, rhs) ? 2 : 1);
            SCIPdebugPrintCons(scip, cons, NULL);
            nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_MIXEDBINARY] += 1 ;
            nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_MIXEDBINARY]++;
            classforcons[c] = SCIP_CONSTYPE_MIXEDBINARY;
            SCIPfreeBufferArray(scip, &vars) ;
            SCIPfreeBufferArray(scip, &vals) ;
            continue;

         }
      }

      /* no special structure detected */
      SCIPdebugMsg(scip, "classified as GENERAL: ");
      SCIPdebugPrintCons(scip, cons, NULL);
      nfoundconstypesrangeddoublecount[SCIP_CONSTYPE_GENERAL] += 1 ;
      nfoundconstypesrangedsinglecount[SCIP_CONSTYPE_GENERAL]++;
      classforcons[c] = SCIP_CONSTYPE_GENERAL;
      SCIPfreeBufferArray(scip, &vars);
      SCIPfreeBufferArray(scip, &vals);
   }

   classifier = new gcg::ConsPartition(scip, "constypes according to miplib", (int) SCIP_CONSTYPE_GENERAL + 1, detprobdata->getNConss() );

   /* set class names and descriptions of every class */
   for( int c = 0; c < classifier->getNClasses(); ++ c )
   {
      std::string name;
      std::stringstream text;
      switch( c )
      {
         case (int) SCIP_CONSTYPE_EMPTY:
            name = "empty";
            break;
         case SCIP_CONSTYPE_FREE:
            name = "free";
            break;
         case SCIP_CONSTYPE_SINGLETON:
            name = "singleton";
            break;
         case SCIP_CONSTYPE_AGGREGATION:
            name = "aggregation";
            break;
         case SCIP_CONSTYPE_VARBOUND:
            name = "varbound";
            break;
         case SCIP_CONSTYPE_SETPARTITION:
            name = "setpartition";
            break;
         case SCIP_CONSTYPE_SETPACKING:
            name = "setpacking";
            break;
         case SCIP_CONSTYPE_SETCOVERING:
            name = "setcovering";
            break;
         case SCIP_CONSTYPE_CARDINALITY:
            name = "cardinality";
            break;
         case SCIP_CONSTYPE_INVKNAPSACK:
            name = "invknapsack";
            break;
         case SCIP_CONSTYPE_EQKNAPSACK:
            name = "eqknapsack";
            break;
         case SCIP_CONSTYPE_BINPACKING:
            name = "binpacking";
            break;
         case SCIP_CONSTYPE_KNAPSACK:
            name = "knapsack";
            break;
         case SCIP_CONSTYPE_INTKNAPSACK:
            name = "intknapsack";
            break;
         case SCIP_CONSTYPE_MIXEDBINARY:
            name = "mixed binary";
            break;
         case SCIP_CONSTYPE_GENERAL:
            name = "general";
            break;
         default:
            name = "unknown";
            break;
      }


#ifdef WRITE_ORIG_CONSTYPES
         myfile << " " <<  nfoundconstypesrangeddoublecount[c] << ",";
#endif

      classifier->setClassName( c, name.c_str() );
      text << "This class contains all constraints that are of (miplib) constype \"" << name << "\".";
      classifier->setClassDescription( c, text.str().c_str() );
   }

#ifdef WRITE_ORIG_CONSTYPES
      myfile << std::endl;
      myfile.close();
#endif



   for( int i = 0; i < classifier->getNConss(); ++ i )
   {
      classifier->assignConsToClass( i, classforcons[i] );
   }



   classifier->removeEmptyClasses();
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Consclassifier \"%s\" yields a classification with %d different constraint classes \n", classifier->getName(), classifier->getNClasses() );

   detprobdata->addConsPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeConsClassifierMiplibConstypes(
   SCIP*                 scip                /**< SCIP data structure */
) {
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL(
      DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata,
         classifierFree, classifierClassify));

   return SCIP_OKAY;
}
