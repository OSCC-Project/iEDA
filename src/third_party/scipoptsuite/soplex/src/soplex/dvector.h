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

/**@file  dvector.h
 * @brief Dynamic vectors.
 */
#ifndef _DVECTOR_H_
#define _DVECTOR_H_

#include "soplex/spxdefines.h"
#include "soplex/basevectors.h"
#include "soplex/vector.h" // for compatibility

// This file exists for reverse compatibility with SCIP. This isn't currently
// needed in SoPlex. DVector used to be typedefs from DVectorBase<T>, but
// DVectorBase has been replaced by VectorBase.
namespace soplex
{
typedef VectorBase< Real > DVector;
typedef VectorBase< Real > DVectorReal;
typedef VectorBase< Rational > DVectorRational;
} // namespace soplex
#endif // _DVECTOR_H_
