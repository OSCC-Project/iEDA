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

/**@file   clscons_scipconstypes.cpp
 * @ingroup CLASSIFIERS
 * @brief classifies constraints according to their scip constypes
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "clscons_scipconstypes.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "class_detprobdata.h"

#include "class_conspartition.h"
#include "scip_misc.h"

/* classifier properties */
#define DEC_CLASSIFIERNAME        "scipconstype"       /**< name of classifier */
#define DEC_DESC                  "scip constypes"     /**< short description of classification*/
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
DEC_DECL_CONSCLASSIFY(classifierClassify) {
   gcg::DETPROBDATA *detprobdata;
   if( transformed )
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataPresolved(scip);
   }
   else
   {
      detprobdata = GCGconshdlrDecompGetDetprobdataOrig(scip);
   }

   std::vector<consType> foundConstypes(0);
   std::vector<int> constypesIndices(0);
   std::vector<int> classForCons = std::vector<int>(detprobdata->getNConss(), -1);
   gcg::ConsPartition *classifier;

   /* firstly, assign all constraints to classindices */
   for (int i = 0; i < detprobdata->getNConss(); ++i) {
      SCIP_CONS *cons;
      bool found = false;
      cons = detprobdata->getCons(i);
      consType cT = GCGconsGetType(scip, cons);
      size_t constype;

      /* check whether the constraint's constype is new */
      for (constype = 0; constype < foundConstypes.size(); ++constype) {
         if (foundConstypes[constype] == cT) {
            found = true;
            break;
         }
      }
      /* if it is new, create a new classindex */
      if (!found) {
         foundConstypes.push_back(GCGconsGetType(scip, cons));
         classForCons[i] = foundConstypes.size() - 1;
      } else
         classForCons[i] = constype;
   }

   /* secondly, use these information to create a ConsPartition */
   classifier = new gcg::ConsPartition(scip, "constypes", (int) foundConstypes.size(), detprobdata->getNConss());

   /* set class names and descriptions of every class */
   for (int c = 0; c < classifier->getNClasses(); ++c) {
      std::string name;
      std::stringstream text;
      switch (foundConstypes[c]) {
         case linear:
            name = "linear";
            break;
         case knapsack:
            name = "knapsack";
            break;
         case varbound:
            name = "varbound";
            break;
         case setpacking:
            name = "setpacking";
            break;
         case setcovering:
            name = "setcovering";
            break;
         case setpartitioning:
            name = "setpartitioning";
            break;
         case logicor:
            name = "logicor";
            break;
         case sos1:
            name = "sos1";
            break;
         case sos2:
            name = "sos2";
            break;
         case unknown:
            name = "unknown";
            break;
         case nconsTypeItems:
            name = "nconsTypeItems";
            break;
         default:
            name = "newConstype";
            break;
      }
      classifier->setClassName(c, name.c_str());
      text << "This class contains all constraints that are of (SCIP) constype \"" << name << "\".";
      classifier->setClassDescription(c, text.str().c_str());
   }

   /* copy the constraint assignment information found in first step */
   for (int i = 0; i < classifier->getNConss(); ++i) {
      classifier->assignConsToClass(i, classForCons[i]);
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL,
                   " Consclassifier \"%s\" yields a classification with %d different constraint classes \n",
                   classifier->getName(), (int) foundConstypes.size());

   detprobdata->addConsPartition(classifier);
   return SCIP_OKAY;
}

/*
 * classifier specific interface methods
 */

/** creates the handler for XYZ classifier and includes it in SCIP */
SCIP_RETCODE SCIPincludeConsClassifierScipConstypes(
   SCIP *scip                /**< SCIP data structure */
) {
   DEC_CLASSIFIERDATA* classifierdata = NULL;

   SCIP_CALL(
      DECincludeConsClassifier(scip, DEC_CLASSIFIERNAME, DEC_DESC, DEC_PRIORITY, DEC_ENABLED, classifierdata,
         classifierFree, classifierClassify));

   return SCIP_OKAY;
}
