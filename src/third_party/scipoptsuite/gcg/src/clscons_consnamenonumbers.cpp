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

/**@file   clscons_consnamenonumbers.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies constraints according to names (without digits)
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clscons_consnamenonumbers.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "consnamenonumbers"       /**< name of classifier */
#define DEC_DESC                  "constraint names (remove digits; check for identity)"     /**< short description of classification*/
#define DEC_PRIORITY              0

#define DEC_ENABLED               FALSE


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

/** removes all digits from string str */
void removeDigits(
   char *str,
   int *nremoved
)
{
   char digits[11] = "0123456789";
   * nremoved = 0;

   for( int i = 0; i < 10; ++ i )
   {
      char digit = digits[i];
      size_t j = 0;
      while( j < strlen( str ) )
      {
         if( str[j] == digit )
         {
            * nremoved = * nremoved + 1;
            for( size_t k = j; k < strlen( str ); ++ k )
            {
               str[k] = str[k + 1];
            }
         }
         else
            ++ j;
      }
   }
}


static
DEC_DECL_CONSCLASSIFY(classifierClassify) {
   gcg::DETPROBDATA* detprobdata;
   if( transformed )
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataPresolved(scip);
   }
   else
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataOrig(scip);
   }

   std::vector < std::string > consnamesToCompare( detprobdata->getNConss(), "" );
   std::vector<int> nConssConstype( 0 );
   std::vector<int> classForCons = std::vector<int>( detprobdata->getNConss(), - 1 );
   std::vector < std::string > nameClasses( 0 );
   gcg::ConsPartition* classifier;

   /* firstly, remove all digits from the consnames */
   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      int nremoved;
      char consname[SCIP_MAXSTRLEN];
      strcpy(consname, SCIPconsGetName(detprobdata->getCons(i)));

      removeDigits(consname, &nremoved);
      consnamesToCompare[i] = std::string(consname);
   }

   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      /* check if string belongs to an existing name class */
      bool belongstoexistingclass = false;

      for( size_t j = 0; j < nameClasses.size(); ++ j )
      {
         if( nameClasses[j] == consnamesToCompare[i] )
         {
            belongstoexistingclass = true;
            classForCons[i] = j;
            nConssConstype[j] ++;
            break;
         }
      }
      /* if not, create a new class */
      if( !belongstoexistingclass )
      {
         nameClasses.push_back(consnamesToCompare[i]);
         nConssConstype.push_back(1);
         classForCons[i] = nameClasses.size() - 1;
      }
   }

   /* secondly, use these information to create a ConsPartition */
   classifier = new gcg::ConsPartition(scip, "consnames", (int) nameClasses.size(), detprobdata->getNConss());

   /* set all class names and descriptions */
   for( int c = 0; c < classifier->getNClasses(); ++ c )
   {
      std::stringstream text;
      classifier->setClassName(c, nameClasses[c].c_str());
      text << "This class contains all constraints with name \"" << nameClasses[c] << "\".";
      classifier->setClassDescription(c, text.str().c_str());
   }

   /* copy the constraint assignment information found in first step */
   for( int i = 0; i < classifier->getNConss(); ++ i )
   {
      classifier->assignConsToClass(i, classForCons[i]);
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Consclassifier \"%s\" yields a classification with %d different constraint classes \n", classifier->getName(), classifier->getNClasses());

   detprobdata->addConsPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeConsClassifierForConsnamesDigitFreeIdentical(
   SCIP *scip                /**< SCIP data structure */
) {
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL(
      DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata,
         classifierFree, classifierClassify));

   return SCIP_OKAY;
}
