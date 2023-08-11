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

/**@file   clscons_consnamelevenshtein.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies constraints according to levenshtein distance graph of their names
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clscons_consnamelevenshtein.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>
#include <queue>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "consnamelevenshtein"       /**< name of classifier */
#define DEC_DESC                  "constraint names (according to levenshtein distance graph)"     /**< short description of classification*/
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

/** returns levenshtein distance between two strings */
int calcLevenshteinDistance(
   std::string s,
   std::string t
)
{
   /* easy cases */
   if( s.compare( t ) == 0 )
      return 0;
   if( s.length() == 0 )
      return t.length();
   if( t.length() == 0 )
      return s.length();

   /* vectors to store integer distances */
   std::vector<int> prev( t.length() + 1 );
   std::vector<int> curr( t.length() + 1 );

   /* initialize prev (previous row of distances) */
   for( size_t i = 0; i < prev.size(); ++i )
   {
      prev[i] = i;
   }
   for( size_t i = 0; i < s.length(); ++i )
   {
      /* calculate curr (row distances) from the previous one */

      curr[0] = i + 1;

      /* fill remaining of row using 'Bellman' equality */
      for( size_t j = 0; j < t.length(); ++j )
      {
         int cost = ( s[i] == t[j] ) ? 0 : 1;
         curr[j + 1] = std::min( curr[j] + 1, std::min( prev[j + 1] + 1, prev[j] + cost ) );
      }

      /* copy curr to prev for next iteration */
      for( size_t j = 0; j < prev.size(); ++j )
         prev[j] = curr[j];
   }

   return curr[t.length()];
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

   std::vector < std::string > consnamesToCompare(detprobdata->getNConss(), "");
   std::vector<int> nConssConstype;
   std::vector<int> classForCons(detprobdata->getNConss(), - 1);
   std::vector<bool> alreadyReached(detprobdata->getNConss(), false);
   std::queue<int> helpqueue;
   int nUnreachedConss = detprobdata->getNConss();
   int currentClass = - 1;
   int nmaxconss = 5000;
   int connectivity = 1;

   std::stringstream classifierName;
   classifierName << "lev-dist-" << connectivity;
   gcg::ConsPartition* classifier = new gcg::ConsPartition(scip, classifierName.str().c_str(), 0, detprobdata->getNConss());

   /* if number of conss exceeds this number, skip calculating such a classifier */
   if( detprobdata->getNConss() > nmaxconss )
   {

      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " skipped levenshtein distance based constraint classes calculating since number of constraints  %d  exceeds limit %d \n", detprobdata->getNConss(), nmaxconss );
      delete classifier;
      return SCIP_ERROR;
   }

   std::vector<std::vector<int>> levenshteindistances(detprobdata->getNConss(), std::vector<int>( detprobdata->getNConss(), - 1 ));

   /* read consnames */
   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      consnamesToCompare[i] = std::string(SCIPconsGetName(detprobdata->getCons(i)));
   }

   /* calculate levenshtein distances pairwise */
   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      for( int j = i + 1; j < detprobdata->getNConss(); ++ j )
      {
         levenshteindistances[i][j] = calcLevenshteinDistance(consnamesToCompare[i], consnamesToCompare[j]);
         levenshteindistances[j][i] = levenshteindistances[i][j];
      }
   }

   /* repeat doing breadth first search until every constraint is assigned to a class */
   while( nUnreachedConss > 0 )
   {
      int firstUnreached = - 1;
      currentClass ++;
      assert( helpqueue.empty() );
      for( int i = 0; i < detprobdata->getNConss(); ++ i )
      {
         if( classForCons[i] == - 1 )
         {
            firstUnreached = i;
            break;
         }
      }

      helpqueue.push(firstUnreached);
      alreadyReached[firstUnreached] = true;
      classForCons[firstUnreached] = currentClass;
      -- nUnreachedConss;

      /* consider all constraints which are connected to the current constraint by means of levenshtein distance */
      while( ! helpqueue.empty() )
      {
         int nodecons = helpqueue.front();
         helpqueue.pop();
         for( int j = 0; j < detprobdata->getNConss(); ++ j )
         {

            if( alreadyReached[j] )
               continue;

            if( j == nodecons )
               continue;

            if( levenshteindistances[j][nodecons] > connectivity )
               continue;

            alreadyReached[j] = true;
            classForCons[j] = currentClass;
            -- nUnreachedConss;
            helpqueue.push(j);
         }
      }

      /* create a new class with found constraints in ConsPartition*/
      std::stringstream text;
      text << "This class contains all constraints with a name similar to \"" << consnamesToCompare[firstUnreached] << "\".";
      classifier->addClass(consnamesToCompare[firstUnreached].c_str(), text.str().c_str(), gcg::BOTH);
   }

   /* assign constraint indices to classes */
   for( int i = 0; i < detprobdata->getNConss(); ++ i )
   {
      classifier->assignConsToClass(i, classForCons[i]);
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Consclassifier levenshtein: connectivity of %d yields a classification with %d different constraint classes. \n", connectivity, currentClass + 1);

   detprobdata->addConsPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeConsClassifierConsnameLevenshtein(

   SCIP *scip                /**< SCIP data structure */
) {
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL(
      DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata,
         classifierFree, classifierClassify));

   return SCIP_OKAY;
}
