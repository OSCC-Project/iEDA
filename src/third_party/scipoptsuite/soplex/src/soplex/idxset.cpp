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

#include "soplex/idxset.h"

namespace soplex
{

int IdxSet::dim() const
{
   int ddim = -1;

   for(int i = 0; i < size(); i++)
      if(ddim < idx[i])
         ddim = idx[i];

   return ddim;
}

int IdxSet::pos(int i) const
{
   for(int n = 0; n < size(); n++)
      if(idx[n] == i)
         return n;

   return -1;
}

void IdxSet::add(int n, const int i[])
{
   assert(n >= 0 && size() + n <= max());

   for(int j = 0; j < n; j++)
      idx[size() + j] = i[j];

   add(n);
}

void IdxSet::remove(int n, int m)
{
   assert(n <= m && m < size() && n >= 0);
   ++m;

   int cpy = m - n;
   int newnum = num - cpy;
   cpy = (size() - m >= cpy) ? cpy : size() - m;

   do
   {
      --num;
      --cpy;
      idx[n + cpy] = idx[num];
   }
   while(cpy > 0);

   num = newnum;
}

IdxSet& IdxSet::operator=(const IdxSet& rhs)
{
   if(this != &rhs)
   {
      if(idx != 0 && max() < rhs.size())
      {
         if(freeArray)
            spx_free(idx);

         idx = 0;
      }

      if(idx == 0)
      {
         len = rhs.size();
         spx_alloc(idx, len);
         freeArray = true;
      }

      for(num = 0; num < rhs.size(); ++num)
         idx[num] = rhs.idx[num];
   }

   assert(size() == rhs.size());
   assert(size() <= max());
   assert(isConsistent());

   return *this;
}

IdxSet::IdxSet(const IdxSet& old)
   : len(old.len)
   , idx(0)
{
   spx_alloc(idx, len);

   for(num = 0; num < old.num; num++)
      idx[num] = old.idx[num];

   freeArray = true;

   assert(size() == old.size());
   assert(size() <= max());
   assert(isConsistent());
}

bool IdxSet::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   int i, j;

   if(len > 0 && idx == 0)
      return MSGinconsistent("IdxSet");

   for(i = 0; i < size(); ++i)
   {
      if(index(i) < 0)
         return MSGinconsistent("IdxSet");

      for(j = 0; j < i; j++)
         if(index(i) == index(j))
            return MSGinconsistent("IdxSet");
   }

#endif

   return true;
}
} // namespace soplex
