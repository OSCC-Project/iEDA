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

/**@file  spxlp.h
 * @brief Saving LPs in a form suitable for SoPlex.
 */

#ifndef _SPXLP_H_
#define _SPXLP_H_

#include "soplex/spxdefines.h"
#include "soplex/spxlpbase.h"
#include "soplex/vector.h" // for compatibility
#include "soplex/svector.h" // for compatibility
#include "soplex/svset.h" // for compatibility
#include "soplex/lprowset.h" // for compatibility
#include "soplex/lpcolset.h" // for compatibility
#include "soplex/lprow.h" // for compatibility
#include "soplex/lpcol.h" // for compatibility

namespace soplex
{
typedef SPxLPBase< Real > SPxLP;
typedef SPxLPBase< Real > SPxLPReal;
typedef SPxLPBase< Rational > SPxLPRational;
} // namespace soplex
#endif // _SPXLP_H_
