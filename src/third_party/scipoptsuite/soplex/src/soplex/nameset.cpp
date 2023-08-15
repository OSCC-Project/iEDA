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

#include <string.h>
#include "soplex/spxdefines.h"
#include "soplex/nameset.h"
#include "soplex/spxalloc.h"

namespace soplex
{
const char NameSet::Name::deflt = '\0';

void NameSet::add(const char* str)
{
   DataKey k;
   add(k, str);
}

void NameSet::add(DataKey& p_key, const char* str)
{
   const Name nstr(str);

   if(!hashtab.has(nstr))
   {
      if(size() + 1 > max() * HASHTABLE_FILLFACTOR)
      {
         assert(factor >= 1);
         reMax(int(factor * max() + 8));
      }

      if(memSize() + int(strlen(str)) >= memMax())
      {
         memPack();

         if(memSize() + int(strlen(str)) >= memMax())
         {
            assert(memFactor >= 1);
            memRemax(int(memFactor * memMax()) + 9 + int(strlen(str)));
            assert(memSize() + int(strlen(str)) < memMax());
         }
      }

      int   idx = memused;
      char* tmp = &(mem[idx]);
      memused  += int(strlen(str)) + 1;

      spxSnprintf(tmp, SPX_MAXSTRLEN, "%s", str);
      *(set.create(p_key)) = idx;
      Name memname(tmp);
      hashtab.add(memname, p_key);
   }
}

void NameSet::add(const NameSet& p_set)
{
   for(int i = 0; i < p_set.num(); ++i)
   {
      Name iname(p_set[i]);

      if(!hashtab.has(iname))
         add(p_set[i]);
   }
}

void NameSet::add(DataKey p_key[], const NameSet& p_set)
{
   for(int i = 0; i < p_set.num(); ++i)
   {
      Name iname = Name(p_set[i]);

      if(!hashtab.has(iname))
         add(p_key[i], p_set[i]);
   }
}

void NameSet::remove(const char* str)
{
   const Name nam(str);

   if(hashtab.has(nam))
   {
      const DataKey* hkey = hashtab.get(nam);
      assert(hkey != 0);
      hashtab.remove(nam);
      set.remove(*hkey);
   }
}

void NameSet::remove(const DataKey& p_key)
{
   assert(has(p_key));

   hashtab.remove(Name(&mem[set[p_key]]));
   set.remove(p_key);
}

void NameSet::remove(const DataKey keys[], int n)
{
   for(int i = 0; i < n; ++i)
      remove(keys[i]);
}

void NameSet::remove(const int nums[], int n)
{
   for(int i = 0; i < n; ++i)
      remove(nums[i]);
}

void NameSet::remove(int dstat[])
{
   for(int i = 0; i < set.num(); i++)
   {
      if(dstat[i] < 0)
      {
         const Name nam = &mem[set[i]];
         hashtab.remove(nam);
      }
   }

   set.remove(dstat);

   assert(isConsistent());
}

void NameSet::clear()
{
   set.clear();
   hashtab.clear();
   memused = 0;
}

void NameSet::reMax(int newmax)
{
   hashtab.reMax(newmax);
   set.reMax(newmax);
}

void NameSet::memRemax(int newmax)
{
   memmax = (newmax < memSize()) ? memSize() : newmax;
   spx_realloc(mem, memmax);

   hashtab.clear();

   for(int i = num() - 1; i >= 0; --i)
      hashtab.add(Name(&mem[set[key(i)]]), key(i));
}

void NameSet::memPack()
{
   char* newmem = 0;
   int   newlast = 0;
   int   i;

   hashtab.clear();

   spx_alloc(newmem, memSize());

   for(i = 0; i < num(); i++)
   {
      const char* t = &mem[set[i]];
      spxSnprintf(&newmem[newlast], SPX_MAXSTRLEN, "%s", t);
      set[i] = newlast;
      newlast += int(strlen(t)) + 1;
   }

   memcpy(mem, newmem, static_cast<size_t>(newlast));
   memused = newlast;

   assert(memSize() <= memMax());

   spx_free(newmem);

   for(i = 0; i < num(); i++)
      hashtab.add(Name(&mem[set[key(i)]]), key(i));
}

/// returns the hash value of the name.
static int NameSetNameHashFunction(const NameSet::Name* str)
{
   unsigned int res = 37;
   const char* sptr = str->name;

   while(*sptr != '\0')
   {
      res *= 11;
      res += (unsigned int)(*sptr++);

   }

   res %= 0x0fffffff;
   return ((int) res);
}

NameSet::NameSet(int p_max, int mmax, Real fac, Real memFac)
   : set(p_max)
   , mem(0)
   , hashtab(NameSetNameHashFunction, set.max(), 0, fac)
   , factor(fac)
   , memFactor(memFac)
{
   memused = 0;
   memmax = (mmax < 1) ? (8 * set.max() + 1) : mmax;
   spx_alloc(mem, memmax);
}

NameSet::~NameSet()
{
   spx_free(mem);
}

bool NameSet::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS

   if(memused > memmax)
      return MSGinconsistent("NameSet");

   int i;

   for(i = 0; i < num(); i++)
   {
      const char* t = &mem[set[i]];

      if(!has(t))
         return MSGinconsistent("NameSet");

      if(strcmp(t, operator[](key(t))))
         return MSGinconsistent("NameSet");
   }

   return set.isConsistent() && hashtab.isConsistent();
#else
   return true;
#endif
}

std::ostream& operator<<(std::ostream& s, const NameSet& nset)
{
   for(int i = 0; i < nset.num(); i++)
   {
      s << i << " "
        << nset.key(i).info << "."
        << nset.key(i).idx << "= "
        << nset[i]
        << std::endl;
   }

   return s;
}


} // namespace soplex
