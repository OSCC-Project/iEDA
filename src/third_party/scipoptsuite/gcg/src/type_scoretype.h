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

/**@file   type_scoretype.h
 * @ingroup DECOMP
 * @brief  type definition for score type
 * @author Michael Bastubbe
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GCG_TYPE_SCORETYPE_H__
#define __GCG_TYPE_SCORETYPE_H__

#ifdef __cplusplus
extern "C" {
#endif

/** GCG score type for (partial) decomposition evaluation */
/*!
 * \brief possible scores to evaluate founds decompositions
 * \sa GCGscoretypeGetDescription for a description of this score
 * \sa GCGscoretypeGetShortName
 * \sa class_partialdecomp:getScore()
 */
enum scoretype {
   /* Note: Please ensure that this enum is compatible with the arrays in scoretype.c! */
   MAX_WHITE                  = 0,
   BORDER_AREA                = 1,
   CLASSIC                    = 2,
   MAX_FORESSEEING_WHITE      = 3,
   SETPART_FWHITE             = 4,
   MAX_FORESEEING_AGG_WHITE   = 5,
   SETPART_AGG_FWHITE         = 6,
   BENDERS                    = 7,
   STRONG_DECOMP              = 8
};
typedef enum scoretype SCORETYPE;

#ifdef __cplusplus
}
#endif

#endif
