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

/**@file    clscons_gamssymbol.cpp
 * @ingroup CLASSIFIERS
 * @brief   gamssymbol constraint classifier (classifies by corresponding GAMS symbol)
 * @author  Stefanie Koß
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#define SCIP_DEBUG

#include "clscons_gamssymbol.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <string>
#include <map>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "gamssymbol"                 /**< name of classifier */
#define DEC_DESC                  "symbol in GAMS file"        /**< short description of classification*/
#define DEC_PRIORITY              0

#define DEC_ENABLED               TRUE


/*
 * Data structures
 */
struct DEC_ClassifierData
{
   std::map<std::string, int>*      constosymbol;             /**< maps constraint name to the corresponding symbol index */
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

   delete classifierdata->constosymbol;

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
   std::vector<int> nconssForClass( 0 );        // [i] holds number of constraints for class i
   std::vector<int> symbolidxForClass( 0 );     // [i] holds symbol index for class i
   std::vector<int> classForCons( ncons, - 1 ); // [i] holds class index for constraint i -> indexing over detection internal constraint array!
   int counterClasses = 0;

   DEC_CONSCLASSIFIER* classifier = DECfindConsClassifier(scip, DEC_CLASSIFIERNAME);
   assert(classifier != NULL);

   DEC_CLASSIFIERDATA* classdata = DECconsClassifierGetData(classifier);
   assert(classdata != NULL);

   /* firstly, assign all constraints to classindices */
   // iterate over constraints in detection and lookup in classdata->constosymbol
   // iterating over classdata->constosymbol and lookup constraints with getIndexForCons fails with assertion if constraint is not found -> should return error value?
   for( int consid = 0; consid < detprobdata->getNConss(); ++ consid )
   {
      // int consid = detprobdata->getIndexForCons(iter.second);
      SCIP_CONS* cons = detprobdata->getCons(consid);
      std::string consname = std::string( SCIPconsGetName( cons ) );

      auto symbolidxiter = classdata->constosymbol->find(consname);
      int symbolidx;
      if( symbolidxiter != classdata->constosymbol->end() )
      {
         symbolidx = symbolidxiter->second;
      }
      else
      {
         symbolidx = -1;
      }
      
      bool classfound = false;

      /* check if class for symbol index exists */
      for( size_t classid = 0; classid < symbolidxForClass.size(); ++classid )
      {
         if( symbolidx == symbolidxForClass[classid] )
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
         symbolidxForClass.push_back( symbolidx );
         nconssForClass.push_back( 1 );
      }
   }
   assert( counterClasses == (int) symbolidxForClass.size() );

   /* secondly, use these information to create a ConsPartition */
   gcg::ConsPartition* partition = new gcg::ConsPartition(scip, "gamssymbols", counterClasses, detprobdata->getNConss() );

   /* set class names and descriptions of every class */
   for( int c = 0; c < partition->getNClasses(); ++ c )
   {
      std::stringstream text;
      text << symbolidxForClass[c];
      partition->setClassName( c, text.str().c_str() );
      text.str( "" );
      text.clear();
      text << "This class contains all constraints with gams symbol index" << symbolidxForClass[c] << ".";
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

/** adds an entry to clsdata->constosymbol */
SCIP_RETCODE DECconsClassifierGamssymbolAddEntry(
   DEC_CONSCLASSIFIER*   classifier,
   SCIP_CONS*            cons,
   int                   symbolIdx
)
{
   assert(classifier != NULL);
   DEC_CLASSIFIERDATA* classdata = DECconsClassifierGetData(classifier);
   assert(classdata != NULL);

   std::string consname = SCIPconsGetName( cons );
   classdata->constosymbol->insert({consname, symbolIdx});

   return SCIP_OKAY;
}

/** creates the handler for gamssymbol classifier and includes it in SCIP */
SCIP_RETCODE SCIPincludeConsClassifierGamssymbol(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &classifierdata) );
   assert(classifierdata != NULL);
   classifierdata->constosymbol = new std::map<std::string, int>();

   SCIP_CALL( DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
