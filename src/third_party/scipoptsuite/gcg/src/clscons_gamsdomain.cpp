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

/**@file    clscons_gamsdomain.cpp
 * @ingroup CLASSIFIERS
 * @brief   Classifies by domains from which constraints are created TODO: what is together in one class?
 * @author  Stefanie Koß
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#define SCIP_DEBUG

#include "clscons_gamsdomain.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <string>
#include <set>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "gamsdomain"                 /**< name of classifier */
#define DEC_DESC                  "domain in GAMS file"        /**< short description of classification*/
#define DEC_PRIORITY              0

#define DEC_ENABLED               TRUE


/*
 * Data structures
 */
struct DEC_ClassifierData
{
   std::map<std::string, std::set<int>>*    constodomain;       /**< maps constraint name to the corresponding set of domain indices */
};

/*
 * Local methods
 */

/* put your local methods here, and declare them static */


/*
 * classifier callback methods
 */

/** destructor of classifier to free user data (called when GCG is exiting) */
static
DEC_DECL_FREECONSCLASSIFIER(classifierFree)
{
   DEC_CLASSIFIERDATA* classifierdata;

   assert(scip != NULL);

   classifierdata = DECconsClassifierGetData(classifier);
   assert(classifierdata != NULL);
   assert(strcmp(DECconsClassifierGetName(classifier), DEC_CLASSIFIERNAME) == 0);

   delete classifierdata->constodomain;

   SCIPfreeMemory(scip, &classifierdata);

   return SCIP_OKAY;
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

   int ncons = detprobdata->getNConss();
   std::vector<int> nconssForClass( 0 );            // [i] holds number of constraints for class i
   std::vector<std::set<int>> domainForClass( 0 );  // [i] holds domain for class i
   std::vector<int> classForCons( ncons, - 1 );     // [i] holds class index for constraint i -> indexing over detection internal constraint array!
   int counterClasses = 0;

   DEC_CONSCLASSIFIER* classifier = DECfindConsClassifier(scip, DEC_CLASSIFIERNAME);
   assert(classifier != NULL);

   DEC_CLASSIFIERDATA* classdata = DECconsClassifierGetData(classifier);
   assert(classdata != NULL);

   /* firstly, assign all constraints to classindices */
   // iterate over constraints in detection and lookup in classdata->constodomain
   for( int consid = 0; consid < detprobdata->getNConss(); ++ consid )
   {
      SCIP_CONS* cons = detprobdata->getCons(consid);
      std::string consname = std::string( SCIPconsGetName( cons ) );

      auto domainiter = classdata->constodomain->find(consname);
      std::set<int> domain;
      if( domainiter != classdata->constodomain->end() )
      {
         domain = domainiter->second;
      }
      else
      {
         domain = std::set<int>();
         domain.insert(-1);
      }
      
      bool classfound = false;

      /* check if class for domain exists */
      for( size_t classid = 0; classid < domainForClass.size(); ++classid )
      {
         if( domain == domainForClass[classid] )
         {
            classfound = true;
            classForCons[consid] = classid;
            ++nconssForClass[classid];
            break;
         }
      }

      /* if not, create a new class index */
      if( !classfound )
      {
         classForCons[consid] = counterClasses;
         ++counterClasses;
         domainForClass.push_back( domain );
         nconssForClass.push_back( 1 );
      }
   }
   assert( counterClasses == (int) domainForClass.size() );

   /* secondly, use these information to create a ConsPartition */
   gcg::ConsPartition* partition = new gcg::ConsPartition(scip, "gamsdomain", counterClasses, detprobdata->getNConss() );

   /* set class names and descriptions of every class */
   for( int c = 0; c < partition->getNClasses(); ++ c )
   {
      std::stringstream text;
      text << "{";
      for( auto iter : domainForClass[c] )
      {
         text << iter;

      }
      text << "}";
      partition->setClassName( c, text.str().c_str() );
      text.str( "" );
      text.clear();
      //text << "This class contains all constraints with gams domain" << domainForClass[c] << ".";
      partition->setClassDescription( c, text.str().c_str() );
   }

   /* copy the constraint assignment information found in first step */
   for( int i = 0; i < partition->getNConss(); ++ i )
   {
      partition->assignConsToClass( i, classForCons[i] );
   }
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Consclassifier \"%s\" yields a classification with %d  different constraint classes \n", partition->getName(), partition->getNClasses() );

   detprobdata->addConsPartition(partition);

   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

/** adds an entry to clsdata->constodomain */
SCIP_RETCODE DECconsClassifierGamsdomainAddEntry(
   DEC_CONSCLASSIFIER*   classifier,
   SCIP_CONS*            cons,
   int                   symDomIdx[],
   int*                  symDim
)
{
   assert(classifier != NULL);
   DEC_CLASSIFIERDATA* classdata = DECconsClassifierGetData(classifier);
   assert(classdata != NULL);

   std::string consname = SCIPconsGetName( cons );
   std::set<int> domainset;
   for( int i = 0; i < *symDim; ++i)
   {
      domainset.insert(symDomIdx[i]);
   }
   classdata->constodomain->insert({consname, domainset});

   return SCIP_OKAY;
}

/** creates the handler for gamsdomain classifier and includes it in SCIP */
SCIP_RETCODE SCIPincludeConsClassifierGamsdomain(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &classifierdata) );
   assert(classifierdata != NULL);
   classifierdata->constodomain = new std::map<std::string, std::set<int>>();

   SCIP_CALL( DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
