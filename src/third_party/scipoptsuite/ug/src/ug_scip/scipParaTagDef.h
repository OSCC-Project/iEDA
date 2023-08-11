/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
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
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file    paraTagDef.h
 * @brief   Fundamental Tag definitions
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_TAG_DEF_H__
#define __SCIP_PARA_TAG_DEF_H__

#include "ug_bb/bbParaTagDef.h"

// #define TAG_STR(tag) #tag

namespace ParaSCIP
{
static const int TAG_SCIP_FIRST                         = UG::TAG_BB_LAST + 1;
//------------------------------------------------------------------------------------------------
static const int TagInitialStat                         = TAG_SCIP_FIRST + 0;
//------------------------------------------------------------------------------------------------
static const int TAG_SCIP_BASE_LAST                     = TAG_SCIP_FIRST + 0;
static const int N_SCIP_BASE_TAGS                       = TAG_SCIP_BASE_LAST - UG::TAG_UG_FIRST + 1;

#ifdef _COMM_MPI_WORLD

static const int TAG_SCIP_MPI_FIRST                     = TAG_SCIP_BASE_LAST + 1;
//------------------------------------------------------------------------------------------------
static const int TagSolverDiffParamSet1                 = TAG_SCIP_MPI_FIRST + 0;
static const int TagDiffSubproblem1                     = TAG_SCIP_MPI_FIRST + 1;
static const int TagDiffSubproblem2                     = TAG_SCIP_MPI_FIRST + 2;
static const int TagDiffSubproblem3                     = TAG_SCIP_MPI_FIRST + 3;
static const int TagDiffSubproblem4                     = TAG_SCIP_MPI_FIRST + 4;
static const int TagDiffSubproblem5                     = TAG_SCIP_MPI_FIRST + 5;
static const int TagDiffSubproblem6                     = TAG_SCIP_MPI_FIRST + 6;
static const int TagDiffSubproblem7                     = TAG_SCIP_MPI_FIRST + 7;
static const int TagDiffSubproblem8                     = TAG_SCIP_MPI_FIRST + 8;
static const int TagDiffSubproblem9                     = TAG_SCIP_MPI_FIRST + 9;
static const int TagDiffSubproblem10                    = TAG_SCIP_MPI_FIRST + 10;
static const int TagDiffSubproblem11                    = TAG_SCIP_MPI_FIRST + 11;
static const int TagDiffSubproblem12                    = TAG_SCIP_MPI_FIRST + 12;
static const int TagDiffSubproblem13                    = TAG_SCIP_MPI_FIRST + 13;
static const int TagDiffSubproblem14                    = TAG_SCIP_MPI_FIRST + 14;
static const int TagSolution1                           = TAG_SCIP_MPI_FIRST + 15;
//------------------------------------------------------------------------------------------------
static const int TAG_SCIP_MPI_LAST                      = TAG_SCIP_MPI_FIRST + 15;
static const int N_SCIP_MPI_TAGS                        = TAG_SCIP_MPI_LAST - UG::TAG_UG_FIRST + 1;
//------------------------------------------------------------------------------------------------
static const int TAG_SCIP_LAST                          = TAG_SCIP_MPI_LAST;
static const int N_SCIP_TAGS                            = TAG_SCIP_LAST - UG::TAG_UG_FIRST + 1;

#endif
#if defined(_COMM_PTH) || defined (_COMM_CPP11)

static const int TAG_SCIP_TH_FIRST                      = TAG_SCIP_BASE_LAST + 1;
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
static const int TAG_SCIP_TH_LAST                       = TAG_SCIP_TH_FIRST - 1;
static const int N_SCIP_TH_TAGS                         = TAG_SCIP_TH_LAST - UG::TAG_UG_FIRST + 1;
//------------------------------------------------------------------------------------------------
static const int TAG_SCIP_LAST                          = TAG_SCIP_TH_LAST;
static const int N_SCIP_TAGS                            = TAG_SCIP_LAST - UG::TAG_UG_FIRST + 1;
#endif

}

#endif // __SCIP_PARA_TAG_DEF_H__
