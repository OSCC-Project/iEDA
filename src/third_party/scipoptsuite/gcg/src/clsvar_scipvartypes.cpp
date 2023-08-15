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

/**@file   clsvar_scipvartypes.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies variables according to their scip vartypes
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clsvar_scipvartypes.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_varpartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "scipvartype"       /**< name of classifier */
#define DEC_DESC                  "scipvartypes"     /**< short description of classification*/
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

   // CLASSIFICATION
   std::vector < SCIP_VARTYPE > foundVartypes( 0 );
   std::vector<int> classForVars = std::vector<int>( detprobdata->getNVars(), - 1 );
   gcg::VarPartition* classifier;

   SCIP_Bool onlycontsub;
   SCIP_Bool onlybinmaster;

   SCIPgetBoolParam(scip, "detection/benders/onlycontsubpr", &onlycontsub);
   SCIPgetBoolParam(scip, "detection/benders/onlybinmaster", &onlybinmaster);

   /* firstly, assign all variables to classindices */
   for( int i = 0; i < detprobdata->getNVars(); ++ i )
   {
      SCIP_VAR* var;
      bool found = false;
      var = detprobdata->getVar(i);
      SCIP_VARTYPE vT = SCIPvarGetType( var );
      size_t vartype;

      if( onlycontsub )
      {
         if ( vT == SCIP_VARTYPE_BINARY )
            vT = SCIP_VARTYPE_INTEGER;
         if( vT == SCIP_VARTYPE_IMPLINT )
            vT = SCIP_VARTYPE_CONTINUOUS;
      }

      /* check whether the variable's vartype is new */
      for( vartype = 0; vartype < foundVartypes.size(); ++ vartype )
      {
         if( foundVartypes[vartype] == vT )
         {
            found = true;
            break;
         }
      }
      /* if it is new, create a new class index */
      if( ! found )
      {
         foundVartypes.push_back( vT );
         classForVars[i] = foundVartypes.size() - 1;
      }
      else
         classForVars[i] = vartype;
   }

   /* secondly, use these information to create a VarPartition */
   classifier = new gcg::VarPartition(scip, "vartypes", (int) foundVartypes.size(), detprobdata->getNVars() );

   /* set class names and descriptions of every class */
   for( int c = 0; c < classifier->getNClasses(); ++ c )
   {
      std::string name;
      std::stringstream text;
      switch( foundVartypes[c] )
      {
         case SCIP_VARTYPE_BINARY:
            name = "bin";
            if( onlybinmaster )
               classifier->setClassDecompInfo(c, gcg::LINKING);
            break;
         case SCIP_VARTYPE_INTEGER:
            name = "int";
            if( onlycontsub )
               classifier->setClassDecompInfo(c, gcg::LINKING);
            if( onlybinmaster )
               classifier->setClassDecompInfo(c, gcg::BLOCK);
            break;
         case SCIP_VARTYPE_IMPLINT:
            name = "impl";
            if( onlybinmaster )
               classifier->setClassDecompInfo(c, gcg::BLOCK);
            break;
         case SCIP_VARTYPE_CONTINUOUS:
            name = "cont";
            if( onlycontsub )
               classifier->setClassDecompInfo(c, gcg::BLOCK);
            if( onlybinmaster )
               classifier->setClassDecompInfo(c, gcg::BLOCK);
            break;
         default:
            name = "newVartype";
            break;
      }
      classifier->setClassName( c, name.c_str() );
      text << "This class contains all variables that are of (SCIP) vartype \"" << name << "\".";
      classifier->setClassDescription( c, text.str().c_str() );
   }

   /* copy the variable assignment information found in first step */
   for( int i = 0; i < classifier->getNVars(); ++ i )
   {
      classifier->assignVarToClass( i, classForVars[i] );
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Varclassifier \"%s\" yields a classification with %d different variable classes\n", classifier->getName(), classifier->getNClasses() ) ;

   detprobdata->addVarPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeVarClassifierScipVartypes(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( DECincludeVarClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
