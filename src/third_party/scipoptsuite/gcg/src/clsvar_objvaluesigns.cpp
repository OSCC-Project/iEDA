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

/**@file   clsvar_objvaluesigns.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies variables according to their objective function value signs
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clsvar_objvaluesigns.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_varpartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "objectivevaluesigns"       /**< name of classifier */
#define DEC_DESC                  "objective function value signs"     /**< short description of classification*/
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
   gcg::VarPartition* classifier= new gcg::VarPartition(scip, "varobjvalsigns", 3, detprobdata->getNVars() ); /* new VarPartition */
   SCIP_Real curobjval;

   /* set up class information */
   classifier->setClassName( 0, "zero" );
   classifier->setClassDescription( 0, "This class contains all variables with objective function value zero." );
   classifier->setClassDecompInfo( 0, gcg::MASTER );
   classifier->setClassName( 1, "positive" );
   classifier->setClassDescription( 1, "This class contains all variables with positive objective function value." );
   classifier->setClassDecompInfo( 1, gcg::ALL );
   classifier->setClassName( 2, "negative" );
   classifier->setClassDescription( 2, "This class contains all variables with negative objective function value." );
   classifier->setClassDecompInfo( 2, gcg::ALL );

   /* assign vars */
   for( int v = 0; v < detprobdata->getNVars(); ++v )
   {
      assert( detprobdata->getVar(v) != NULL );
      curobjval = SCIPvarGetObj(detprobdata->getVar(v));

      if( SCIPisZero(scip, curobjval) )
      {
         classifier->assignVarToClass(v, 0);
      }
      else if ( SCIPisPositive(scip, curobjval) )
      {
         classifier->assignVarToClass(v, 1);
      }
      else
      {
         classifier->assignVarToClass(v, 2);
      }
   }

   /* remove a class if there is no variable with the respective sign */
   classifier->removeEmptyClasses();

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Varclassifier \"%s\" yields a classification with %d different variable classes\n", classifier->getName(), classifier->getNClasses()) ;


   detprobdata->addVarPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

SCIP_RETCODE SCIPincludeVarClassifierObjValueSigns(
   SCIP*                 scip                /**< SCIP data structure */
)
{
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL( DECincludeVarClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata, classifierFree, classifierClassify) );

   return SCIP_OKAY;
}
