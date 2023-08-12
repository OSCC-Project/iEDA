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


#ifndef __PARA_TAG_DEF_H__
#define __PARA_TAG_DEF_H__

#define TAG_STR(tag) #tag

namespace UG
{
static const int TagAny                                 = -1;
static const int TAG_UG_FIRST = 0;
//------------------------------------------------------------------------------------------------
static const int TagTask                                = TAG_UG_FIRST +  0;
static const int TagTaskReceived                        = TAG_UG_FIRST +  1;
static const int TagDiffSubproblem                      = TAG_UG_FIRST +  2;
static const int TagRampUp                              = TAG_UG_FIRST +  3;
static const int TagSolution                            = TAG_UG_FIRST +  4;
static const int TagIncumbentValue                      = TAG_UG_FIRST +  5;
static const int TagSolverState                         = TAG_UG_FIRST +  6;
static const int TagCompletionOfCalculation             = TAG_UG_FIRST +  7;
static const int TagNotificationId                      = TAG_UG_FIRST +  8;
static const int TagTerminateRequest                    = TAG_UG_FIRST +  9;
static const int TagInterruptRequest                    = TAG_UG_FIRST + 10;
static const int TagTerminated                          = TAG_UG_FIRST + 11;
static const int TagRacingRampUpParamSet                = TAG_UG_FIRST + 12;
static const int TagWinner                              = TAG_UG_FIRST + 13;
static const int TagHardTimeLimit                       = TAG_UG_FIRST + 14;
static const int TagAckCompletion                       = TAG_UG_FIRST + 15;
static const int TagToken                               = TAG_UG_FIRST + 16;
//-----------------------------------------------------------------------------------------------
static const int TAG_UG_BASE_LAST                       = TAG_UG_FIRST + 16;
static const int N_UG_BASE_TAGS                         = TAG_UG_BASE_LAST - TAG_UG_FIRST + 1;

#ifdef _COMM_MPI_WORLD
static const int TAG_MPI_FIRST                          = TAG_UG_BASE_LAST + 1;
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
static const int TAG_MPI_LAST                           = TAG_MPI_FIRST - 1;     /// -1 : no tag
static const int N_MPI_TAGS                             = TAG_MPI_LAST - TAG_UG_FIRST + 1;

static const int TAG_UG_LAST                            = TAG_MPI_LAST;
static const int N_UG_TAGS                              = TAG_UG_LAST - TAG_UG_FIRST + 1;
#endif

#if defined(_COMM_PTH) || defined (_COMM_CPP11)
static const int TAG_TH_FIRST                           = TAG_UG_BASE_LAST + 1;
//-----------------------------------------------------------------------------------------------
static const int TagParaInstance                        = TAG_TH_FIRST + 0;
//-----------------------------------------------------------------------------------------------
static const int TAG_TH_LAST                            = TAG_TH_FIRST + 0;
static const int N_TH_TAGS                              = TAG_TH_LAST - TAG_UG_FIRST + 1;

static const int TAG_UG_LAST                            = TAG_TH_LAST;
static const int N_UG_TAGS                              = TAG_UG_LAST - TAG_UG_FIRST + 1;
#endif

}



#endif // __PARA_TAG_DEF_H__
