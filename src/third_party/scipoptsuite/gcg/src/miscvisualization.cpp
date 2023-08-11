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

/**@file   miscvisualization.cpp
 * @brief  miscellaneous methods for visualizations
 * @author Hanna Franzen
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "cons_decomp.h"
#include "miscvisualization.h"
#include "class_detprobdata.h"
#include "class_partialdecomp.h"

#include "scip/scip.h"

#include <unistd.h>
#include <stdlib.h>
#include <sstream>

using namespace gcg;


/* Gives a consistent filename for a (single) partialdec visualization that includes the probname and partialdecID
 *
 * @returns standardized filename
 */
void GCGgetVisualizationFilename(
   SCIP* scip,             /* scip data structure */
   PARTIALDECOMP* partialdec,         /* partialdec that is to be visualized */
   const char* extension,  /* file extension (to be included in the name) */
   char* filename          /* filename output */
   )
{
   char* name;
   char detectorchainstring[SCIP_MAXSTRLEN];
   char probname[SCIP_MAXSTRLEN];

   (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s", SCIPgetProbName(scip));
   SCIPsplitFilename(probname, NULL, &name, NULL, NULL);

   /* get detector chain string*/
   partialdec->buildDecChainString(detectorchainstring);

   assert( partialdec != NULL );

   /* print header */
   if(strlen(detectorchainstring) > 0)
   {
      /* if there is a PARTIALDECOMP that was detected in GCG */
      (void) SCIPsnprintf(filename, SCIP_MAXSTRLEN, "%s-%s-%d-%d%s", name, detectorchainstring, partialdec->getID(),
         partialdec->getNBlocks(), extension);
   }
   else
   {
      /* if there is a PARTIALDECOMP but it was not detected in GCG */
      (void) SCIPsnprintf(filename, SCIP_MAXSTRLEN, "%s-%d-%d%s", name, partialdec->getID(),
         partialdec->getNBlocks(), extension);
   }

   /* some filenames can still have dots in them (usually from prob name) which can cause confusion.
    * Does not replace characters in the file extension. */
   for(size_t i = 0; i < strlen(filename) - strlen(extension); i++)
   {
      if(filename[i] == '.')
         filename[i] = '-';

      if(filename[i] == '(')
         filename[i] = '-';

      if(filename[i] == ')')
        filename[i] = '-';
   }
}


/* Gives the path of the provided file */
void GCGgetFilePath(
   FILE* file,       /* file */
   char* path        /* buffer containing the path afterward, must be of length PATH_MAX! */
   )
{
   char sympath[SCIP_MAXSTRLEN];
   int filedesc;

   filedesc = fileno(file); /* get link to file descriptor */
   if( filedesc < 0 )
   {
      SCIPerrorMessage("File reading error, no fileno!\n");
      return;
   }
   SCIPsnprintf(sympath, SCIP_MAXSTRLEN, "/proc/self/fd/%d", filedesc); /* set symbolic link to file */
   realpath(sympath, path);
}
