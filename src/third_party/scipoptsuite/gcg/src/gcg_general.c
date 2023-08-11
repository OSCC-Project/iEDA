/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Colum Generation                                 */
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

/**@file   gcg_general.c
 * @brief  gcg general public methods
 * @author Steffan Schlein
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "gcg_general.h"

#include "gcggithash.h"

#include "scip/pub_message.h"
#include "scip/scip_message.h"


/** Gets GCG major version
 * @returns GCG major version */
static
int GCGmajorVersion(void)
{
   return GCG_VERSION/100;
}

/** Gets GCG minor version
 * @returns GCG minor version */
static
int GCGminorVersion(void)
{
   return (GCG_VERSION/10) % 10; /*lint !e778*/
}

/** Gets GCG technical version
 * @returns GCG technical version */
static
int GCGtechVersion(void)
{
   return GCG_VERSION % 10; /*lint !e778*/
}
#if GCG_SUBVERSION > 0
/** Gets GCG sub version number
 * @returns GCG sub version number */
static
int GCGsubversion(void)
{
   return GCG_SUBVERSION;
}
#endif

/** prints out GCG version
 * @returns nothing */
void GCGprintVersion(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   assert(scip != NULL);

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "GCG version %d.%d.%d",
      GCGmajorVersion(), GCGminorVersion(), GCGtechVersion());
#if GCG_SUBVERSION > 0
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, ".%d", GCGsubversion());
#endif
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, " [GitHash: %s]", GCGgetGitHash());
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Copyright (C) 2010-2022 Operations Research, RWTH Aachen University\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "                        Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)\n\n");
}
