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

/**@file   scoretype.h
 * @brief  miscellaneous methods for working with SCORETYPE
 * @author Erik Muehmer
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scoretype.h"

const char* scoretype_shortnames[] =
{
   "maxwhi",   /* MAX_WHITE */
   "border",   /* BORDER_AREA */
   "classi",   /* CLASSIC */
   "forswh",   /* MAX_FORESSEEING_WHITE */
   "spfwh",    /* SETPART_FWHITE */
   "fawh",     /* MAX_FORESEEING_AGG_WHITE */
   "spfawh",   /* SETPART_AGG_FWHITE */
   "bender",   /* BENDERS */
   "strode"    /* STRONG_DECOMP */
};

const char* scoretype_descriptions[] =
{
   /* MAX_WHITE */
   "maximum white area score (white area is nonblock and nonborder area)",
   /* BORDER_AREA */
   "minimum border score (i.e. minimizes fraction of border area score)",
   /* CLASSIC */
   "classical score",
   /* MAX_FORESSEEING_WHITE */
   "maximum foreseeing white area score (considering copied linking vars and their master conss; white area is nonblock and nonborder area)",
   /* SETPART_FWHITE */
   "setpartitioning maximum foreseeing white area score (convex combination of maximum foreseeing white area score and rewarding if master contains only setppc and cardinality constraints)",
   /* MAX_FORESEEING_AGG_WHITE */
   "maximum foreseeing white area score with aggregation infos (considering copied linking vars and their master conss; white area is nonblock and nonborder area)",
   /* SETPART_AGG_FWHITE */
   "setpartitioning maximum foreseeing white area score with aggregation information (convex combination of maximum foreseeing white area score and rewarding if a master contains only setppc and cardinality constraints)",
   /* BENDERS */
   "experimental score to evaluate benders decompositions",
   /* STRONG_DECOMP */
   "strong decomposition score",
};

const char* GCGscoretypeGetDescription(
   SCORETYPE   sctype
   )
{
   return scoretype_descriptions[sctype];
}

const char* GCGscoretypeGetShortName(
   SCORETYPE   sctype
   )
{
   return scoretype_shortnames[sctype];
}
