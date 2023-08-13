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

/**@file   wrapper_partialdecomp.h
 * @ingroup DECOMP
 * @brief  Provides wrapping to have Partialdecomps as parameters in C-conform function headers with C++
 *         implementations.
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_WRAPPER_PARTIALDECOMP_H__
#define GCG_WRAPPER_PARTIALDECOMP_H__

#include "cons_decomp.h"
#include "class_partialdecomp.h"

/** Wrapper class body to be included in .cpp files only.
 * To pass a wrapper object through C headers include the wrapper forward declaration in cons_decomp.h
 *
 * The struct can contain a PARTIALDECOMP object.*/
struct Partialdecomp_Wrapper
{
   gcg::PARTIALDECOMP* partialdec;            /**< partialdec pointer */
};

#endif   /* GCG_WRAPPER_PARTIALDECOMP_H__ */
