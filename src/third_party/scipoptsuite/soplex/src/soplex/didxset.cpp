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

#include "soplex/didxset.h"
#include "soplex/spxalloc.h"

namespace soplex
{

void DIdxSet::setMax(int newmax)
{
   assert(idx   != 0);
   assert(max() >  0);

   len = (newmax < size()) ? size() : newmax;
   len = (len < 1) ? 1 : len;

   assert(len > 0);

   spx_realloc(idx, len);
}

DIdxSet::DIdxSet(const IdxSet& old)
   : IdxSet()
{
   len = old.size();
   len = (len < 1) ? 1 : len;
   spx_alloc(idx, len);

   IdxSet::operator= (old);
}

DIdxSet::DIdxSet(const DIdxSet& old)
   : IdxSet()
{
   len = old.size();
   len = (len < 1) ? 1 : len;
   spx_alloc(idx, len);

   IdxSet::operator= (old);
}

DIdxSet::DIdxSet(int n)
   : IdxSet()
{
   len = (n < 1) ? 1 : n;
   spx_alloc(idx, len);
}

DIdxSet::~DIdxSet()
{
   if(idx)
      spx_free(idx);
}
} // namespace soplex
