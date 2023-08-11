/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*  Copyright 1996-2022 Zuse Institute Berlin                                */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SoPlex; see the file LICENSE. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file  spxfileio.h
 * @brief declaration of types for file output
 *
 * This is to make the use of compressed input files transparent
 * when programming.
 *
 * @todo maybe rename this file (it is unrelated to spxfileio.cpp)
 */
#ifndef _SPXFILEIO_H_
#define _SPXFILEIO_H_

#include <iostream>
#include <fstream>

/*-----------------------------------------------------------------------------
 * compressed file support
 *-----------------------------------------------------------------------------
 */
#ifdef SOPLEX_WITH_ZLIB
#include "soplex/gzstream.h"
#endif // WITH_GSZSTREAM

namespace soplex
{
#ifdef SOPLEX_WITH_ZLIB
typedef gzstream::igzstream spxifstream;
#else
typedef std::ifstream spxifstream;
#endif // SOPLEX_WITH_ZLIB

} // namespace soplex

#endif // _SPXFILEIO_H_
