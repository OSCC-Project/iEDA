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

/**@file    clsvar_gamsdomain.cpp
 * @ingroup CLASSIFIERS
 * @brief   Classifies by domains from which variables are created TODO: what is together in one class?
 * @author  Stefanie Koß
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clsvar_gamsdomain.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <map>
#include <set>
#include <stdio.h>
#include <sstream>
#include <vector>

#include "class_detprobdata.h"

#include "class_varpartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "gamsdomain"              /**< name of classifier */
#define DEC_DESC                  "domain in gams file"     /**< short description of classification */
#define DEC_PRIORITY              0                         /**< priority of this classifier */

#define DEC_ENABLED               TRUE


/*
 * Data structures
 */

/** classifier handler data */
struct DEC_ClassifierData
{
   std::map<std::string, std::set<int>>*      vartodomain;            /**< maps variable name to the corresponding set of domain indices */
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
DEC_DECL_FREEVARCLASSIFIER(classifierFree)
{
   assert(scip != NULL);

   DEC_CLASSIFIERDATA* classifierdata = DECvarClassifierGetData(classifier);
   assert(classifierdata != NULL);
   assert(strcmp(DECvarClassifierGetName(classifier), DEC_CLASSIFIERNAME) == 0);

   delete classifierdata->vartodomain;

   SCIPfreeMemory(scip, &classifierdata);

   return SCIP_OKAY;
}

static
DEC_DECL_VARCLASSIFY(classifierClassify)
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

   int nvar = detprobdata->getNVars();
   std::vector<int> nvarsForClass( 0 );                // [i] holds number of variables for class i
   std::vector<std::set<int>> domainForClass( 0 );     // [i] holds domain for class i
   std::vector<int> classForVar( nvar, - 1 );          // [i] holds class index for variable i -> indexing over detection internal variable array!
   int counterClasses = 0;

   DEC_VARCLASSIFIER* classifier = DECfindVarClassifier(scip, DEC_CLASSIFIERNAME);
   assert(classifier != NULL);

   DEC_CLASSIFIERDATA* classdata = DECvarClassifierGetData(classifier);
   assert(classdata != NULL);

   /* firstly, assign all variables to classindices */
   // iterate over variables in detection and lookup in classdata->vartodomain
   // iterating over classdata->vartosymbol and lookup variables with getIndexForVar fails with assertion if variable is not found -> should return error value?
   for( int varid = 0; varid < detprobdata->getNVars(); ++ varid )
   {
      SCIP_VAR* var = detprobdata->getVar(varid);
      std::string varname = std::string( SCIPvarGetName( var ) );
      auto domainiter = classdata->vartodomain->find(varname);
      std::set<int> domain;
      if( domainiter != classdata->vartodomain->end() )
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
            classForVar[varid] = classid;
            ++nvarsForClass[classid];
            break;
         }
      }

      /* if not, create a new class index */
      if( !classfound )
      {
         classForVar[varid] = counterClasses;
         ++counterClasses;
         domainForClass.push_back( domain );
         nvarsForClass.push_back( 1 );
      }
   }
   assert( counterClasses == (int) domainForClass.size() );

   /* secondly, use these information to create a ConsPartition */
   gcg::VarPartition* partition = new gcg::VarPartition(scip, "gamsdomain", counterClasses, detprobdata->getNVars() );

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
      //text << "This class contains all variables with gams domain" << domainForClass[c] << ".";
      partition->setClassDescription( c, text.str().c_str() );
   }

   /* copy the constraint assignment information found in first step */
   for( int i = 0; i < partition->getNVars(); ++ i )
   {
      partition->assignVarToClass( i, classForVar[i] );
   }
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Varclassifier \"%s\" yields a classification with %d  different variable classes \n", partition->getName(), partition->getNClasses() );

   detprobdata->addVarPartition(partition);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

/** adds an entry to clsdata->vartodomain */
SCIP_RETCODE DECvarClassifierGamsdomainAddEntry(
   DEC_VARCLASSIFIER*   classifier,
   SCIP_VAR*            var,
   int                  symDomIdx[],
   int*                 symDim
)
{
   assert(classifier != NULL);
   DEC_CLASSIFIERDATA* classdata = DECvarClassifierGetData(classifier);
   assert(classdata != NULL);

   std::string varname = SCIPvarGetName( var );
   char varnametrans[SCIP_MAXSTRLEN];
   (void) SCIPsnprintf(varnametrans, SCIP_MAXSTRLEN, "t_%s", varname.c_str());
   std::string nametrans(varnametrans);
   std::set<int> domainset;
   for( int i = 0; i < *symDim; ++i)
   {
      domainset.insert(symDomIdx[i]);
   }
   classdata->vartodomain->insert({varname, domainset});
   classdata->vartodomain->insert({varnametrans, domainset});

   return SCIP_OKAY;
}

/** creates the handler for gamsdomain classifier and includes it in SCIP */
SCIP_RETCODE SCIPincludeVarClassifierGamsdomain(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &classifierdata) );
   assert(classifierdata != NULL);
   classifierdata->vartodomain = new std::map<std::string, std::set<int>>();

   SCIP_CALL( DECincludeVarClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
