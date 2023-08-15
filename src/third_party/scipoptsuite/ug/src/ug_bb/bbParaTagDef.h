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

/**@file    bbParaTagDef.h
 * @brief   ug_bb Tag definitions
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_TAG_DEF_H__
#define __BB_PARA_TAG_DEF_H__

#include "ug/paraTagDef.h"

namespace UG
{
static const int TAG_BB_FIRST = TAG_UG_LAST + 1;
//------------------------------------------------------------------------------------------------
static const int TagRetryRampUp                         = TAG_BB_FIRST +  0;
static const int TagGlobalBestDualBoundValueAtWarmStart = TAG_BB_FIRST +  1;
static const int TagAnotherNodeRequest                  = TAG_BB_FIRST +  2;
static const int TagNoNodes                             = TAG_BB_FIRST +  3;
static const int TagInCollectingMode                    = TAG_BB_FIRST +  4;
static const int TagCollectAllNodes                     = TAG_BB_FIRST +  5;
static const int TagOutCollectingMode                   = TAG_BB_FIRST +  6;
static const int TagLCBestBoundValue                    = TAG_BB_FIRST +  7;
static const int TagLightWeightRootNodeProcess          = TAG_BB_FIRST +  8;
static const int TagBreaking                            = TAG_BB_FIRST +  9;
static const int TagGivenGapIsReached                   = TAG_BB_FIRST + 10;
static const int TagAllowToBeInCollectingMode           = TAG_BB_FIRST + 11;
static const int TagTestDualBoundGain                   = TAG_BB_FIRST + 12;
static const int TagNoTestDualBoundGain                 = TAG_BB_FIRST + 13;
static const int TagNoWaitModeSend                      = TAG_BB_FIRST + 14;
static const int TagRestart                             = TAG_BB_FIRST + 15;
static const int TagLbBoundTightenedIndex               = TAG_BB_FIRST + 16;
static const int TagLbBoundTightenedBound               = TAG_BB_FIRST + 17;
static const int TagUbBoundTightenedIndex               = TAG_BB_FIRST + 18;
static const int TagUbBoundTightenedBound               = TAG_BB_FIRST + 19;
static const int TagCutOffValue                         = TAG_BB_FIRST + 20;
static const int TagChangeSearchStrategy                = TAG_BB_FIRST + 21;
static const int TagSolverDiffParamSet                  = TAG_BB_FIRST + 22;
static const int TagKeepRacing                          = TAG_BB_FIRST + 23;
static const int TagTerminateSolvingToRestart           = TAG_BB_FIRST + 24;
static const int TagSelfSplitFinished                   = TAG_BB_FIRST + 25;
static const int TagNewSubtreeRootNode                  = TAG_BB_FIRST + 26;
static const int TagSubtreeRootNodeStartComputation     = TAG_BB_FIRST + 27;
static const int TagSubtreeRootNodeToBeRemoved          = TAG_BB_FIRST + 28;
static const int TagReassignSelfSplitSubtreeRootNode    = TAG_BB_FIRST + 29;
static const int TagSelfSlpitNodeCalcuationState        = TAG_BB_FIRST + 30;
static const int TagTermStateForInterruption            = TAG_BB_FIRST + 31;
static const int TagSelfSplitTermStateForInterruption   = TAG_BB_FIRST + 32;
//-----------------------------------------------------------------------------------------------
static const int TAG_BB_BASE_LAST                       = TAG_BB_FIRST + 32;
static const int N_BB_BASE_TAGS                         = TAG_BB_BASE_LAST - TAG_UG_FIRST + 1;

#ifdef _COMM_MPI_WORLD

static const int TAG_BB_MPI_FIRST                       = TAG_BB_BASE_LAST + 1;
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
static const int TAG_BB_MPI_LAST                        = TAG_BB_MPI_FIRST - 1;    // no tag
static const int N_BB_MPI_TAGS                          = TAG_BB_MPI_LAST - TAG_UG_FIRST + 1;
//-----------------------------------------------------------------------------------------------
static const int TAG_BB_LAST                            = TAG_BB_MPI_LAST;
static const int N_BB_TAGS                              = TAG_BB_LAST - TAG_UG_FIRST + 1;

#endif

#if defined(_COMM_PTH) || defined (_COMM_CPP11)

static const int TAG_BB_TH_FIRST                        = TAG_BB_BASE_LAST + 1;
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
static const int TAG_BB_TH_LAST                         = TAG_BB_TH_FIRST - 1;     //  no tag
static const int N_BB_TH_TAGS                           = TAG_BB_TH_LAST - TAG_UG_FIRST + 1;
//-----------------------------------------------------------------------------------------------
static const int TAG_BB_LAST                            = TAG_BB_TH_LAST;
static const int N_BB_TAGS                              = TAG_BB_LAST - TAG_UG_FIRST + 1;

#endif
}

#endif // __BB_PARA_TAG_DEF_H__
