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
/* You sh
ould have received a copy of the GNU Lesser General Public License  */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   clscons_nnonzeros.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies constraints according to their nonzero entries
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clscons_nnonzeros.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "nnonzeros"       /**< name of classifier */
#define DEC_DESC                  "nnonezero entries"     /**< short description of classification*/
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

   std::vector<int> nconssforclass( 0 );
   std::vector<int> differentNNonzeros( 0 );
   std::vector<int> classForCons( detprobdata->getNConss(), - 1 );
   int counterClasses = 0;

   /* firstly, assign all constraints to classindices */
   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      int consnnonzeros = detprobdata->getNVarsForCons( i );
      bool nzalreadyfound = false;

      /* check if number of nonzeros belongs to an existing class index */
      for( size_t nzid = 0; nzid < differentNNonzeros.size(); ++ nzid )
      {
         if( consnnonzeros == differentNNonzeros[nzid] )
         {
            nzalreadyfound = true;
            classForCons[i] = nzid;
            ++ nconssforclass[nzid];
            break;
         }
      }

      /* if not, create a new class index */
      if( ! nzalreadyfound )
      {
         classForCons[i] = counterClasses;
         ++ counterClasses;
         differentNNonzeros.push_back( consnnonzeros );
         nconssforclass.push_back( 1 );
      }
   }

   /* secondly, use these information to create a ConsPartition */
   gcg::ConsPartition* classifier = new gcg::ConsPartition(scip, "nonzeros", (int) differentNNonzeros.size(), detprobdata->getNConss() );

   /* set class names and descriptions of every class */
   for( int c = 0; c < classifier->getNClasses(); ++ c )
   {
      std::stringstream text;
      text << differentNNonzeros[c];
      classifier->setClassName( c, text.str().c_str() );
      text.str( "" );
      text.clear();
      text << "This class contains all constraints with " << differentNNonzeros[c] << " nonzero coefficients.";
      classifier->setClassDescription( c, text.str().c_str() );
   }

   /* copy the constraint assignment information found in first step */
   for( int i = 0; i < classifier->getNConss(); ++ i )
   {
      classifier->assignConsToClass( i, classForCons[i] );
   }
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Consclassifier \"%s\" yields a classification with %d  different constraint classes \n", classifier->getName(), classifier->getNClasses() );

   detprobdata->addConsPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

/** creates the handler for classifier for Nnonzeros and includes it in SCIP */
SCIP_RETCODE SCIPincludeConsClassifierNNonzeros(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL(
      DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata,
         classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
