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

/**@file   clsvar_objvalues.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies variables according to their objective function values
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clsvar_objvalues.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>
#include <iomanip>

#include "class_detprobdata.h"

#include "class_varpartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "objectivevalues"       /**< name of classifier */
#define DEC_DESC                  "objective function values"     /**< short description of classification*/
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
   std::vector<SCIP_Real> foundobjvals; /* all found objective function values */
   std::vector<int> classforvars(detprobdata->getNVars(), -1); /* vector assigning a class index to each variable */
   int curclassindex; /* stores a var's classindex if the objective value of a var has already been found for another var */
   SCIP_Real curobjval;
   gcg::VarPartition* classifier; /* new VarPartition */

   for( int v = 0; v < detprobdata->getNVars(); ++v )
   {
      assert( detprobdata->getVar(v) != NULL );
      curobjval = SCIPvarGetObj(detprobdata->getVar(v));
      curclassindex = -1;

      /* check whether current objective funtion value already exists */
      for( size_t c = 0; c < foundobjvals.size(); ++c )
      {
         if( SCIPisEQ(scip, curobjval, foundobjvals[c]) )
         {
            curclassindex = c;
            break;
         }
      }

      /* assign var to class and save objective function value, if it is new */
      if( curclassindex == -1 )
      {
         foundobjvals.push_back(curobjval);
         classforvars[v] = foundobjvals.size() - 1;
      }
      else
      {
         classforvars[v] = curclassindex;
      }
   }

   classifier = new gcg::VarPartition(scip, "varobjvals", (int) foundobjvals.size(), detprobdata->getNVars());

   /* set up class information */
   for ( int c = 0; c < classifier->getNClasses(); ++c )
   {
      std::stringstream name;
      std::stringstream text;

      name << std::setprecision( 5 ) << foundobjvals[c];
      text << "This class contains all variables with objective function value " << name.str() << ".";

      classifier->setClassName(c, name.str().c_str());
      classifier->setClassDescription(c, text.str().c_str());
   }

   /* assign vars according to classforvars vactor */
   for ( int v = 0; v < classifier->getNVars(); ++v )
   {
      classifier->assignVarToClass(v, classforvars[v]);
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Varclassifier \"%s\" yields a classification with %d different variable classes\n", classifier->getName(), classifier->getNClasses()) ;

   detprobdata->addVarPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeVarClassifierObjValues(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( DECincludeVarClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
