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

/**@file  spxalloc.h
 * @brief Memory allocation routines.
 */
#ifndef _SPXALLOC_H_
#define _SPXALLOC_H_

#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxout.h"

#include "soplex/exceptions.h"

namespace soplex
{
/**@name    Memory allocation routines
 * @ingroup Elementary
 * Here we have cover functions for malloc/realloc/free, to make sure
 * that we allays succeed. Otherwise an exception is thrown.
 *
 * We use templates to get the types right, otherwise casts would have
 * been neccessary.
 */
///@{
/**@brief Allocate memory.
 * @param p some pointer
 * @param n the number of elements \p p will point to.
 * @throw SPxMemoryException if memory could not be allocated.
 */

template <class T>
inline void spx_alloc(T& p, int n = 1)
{
   assert(p == 0);
   assert(n >= 0);

   if(n == 0)
      n = 1;

   try
   {
      p = reinterpret_cast<T>(malloc(sizeof(*p) * (unsigned int) n));
   }
   catch(const std::bad_alloc&)
   {
      throw(SPxMemoryException("Error allocating memory"));
   }

   if(nullptr == p)
   {
      // coverity[suspicious_sizeof]
      std::cerr << "EMALLC01 malloc: Out of memory - cannot allocate "
                << sizeof(*p) * (unsigned int) n << " bytes" << std::endl;
      throw(SPxMemoryException("XMALLC01 malloc: Could not allocate enough memory"));
   }
}

/**@brief Change amount of allocated memory.
 * @param p some pointer
 * @param n the number of elements p should point to.
 * @throw SPxMemoryException if memory could not be allocated.
 */
template <class T>
inline void spx_realloc(T& p, int n)
{
   assert(n >= 0);

   /* new pointer to not lose old one in case of problems */
   T pp;

   if(n == 0)
      n = 1;

   try
   {
      pp = reinterpret_cast<T>(realloc(p, sizeof(*p) * (unsigned int) n));
   }
   catch(const std::bad_alloc&)
   {
      throw(SPxMemoryException("Error reallocating memory"));
   }

   if(0 == pp)
   {
      std::cerr << "EMALLC02 realloc: Out of memory - cannot allocate "
                << sizeof(*p) * (unsigned int) n << " bytes" << std::endl;
      throw(SPxMemoryException("XMALLC02 realloc: Could not allocate enough memory"));
   }

   p = pp;
}

/// Release memory
template <class T>
inline void spx_free(T& p)
{
   assert(p != 0);
   free(p);

   p = 0;
}

///@}
} // namespace soplex


#endif // _SPXALLOC_H_
