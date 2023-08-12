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

/**@file  didxset.h
 * @brief Dymnamic index set.
 */
#ifndef _DIDXSET_H_
#define _DIDXSET_H_

#include <assert.h>

#include "soplex/idxset.h"

namespace soplex
{

/**@brief   Dynamic index set.
   @ingroup Elementary

   Class DIdxSet provides dynamic IdxSet in the sense, that no
   restrictions are posed on the use of methods add(). However, method
   indexMem() has been moved to the private members. This is because
   DIdxSet adds its own memory management to class IdxSet and the user must
   not interfere with it.

   Upon construction of an DIdxSet, memory is allocated automatically. The
   memory consumption can be controlled with methods max() and setMax().
   Finally, the destructor will release all allocated memory.
*/
class DIdxSet : public IdxSet
{
public:

   //-----------------------------------
   /**@name Adding */
   ///@{
   /// adds \p n uninitialized indices.
   void add(int n)
   {
      if(max() - size() < n)
         setMax(size() + n);

      IdxSet::add(n);
   }

   /// adds all indices from \p sv.
   void add(const IdxSet& sv)
   {
      int n = sv.size();

      if(max() - size() < n)
         setMax(size() + n);

      IdxSet::add(sv);
   }

   /// adds \p n indices from \p i.
   void add(int n, const int* i)
   {
      if(max() - size() < n)
         setMax(size() + n);

      IdxSet::add(n, i);
   }

   /// adds index \p i to the index set
   void addIdx(int i)
   {
      if(max() <= size())
         setMax(size() + 1);

      IdxSet::addIdx(i);
   }

   /// sets the maximum number of indices.
   /** This methods resets the memory consumption of the DIdxSet to
    *  \p newmax. However, if \p newmax < size(), it is reset to size()
    *  only.
    */
   void setMax(int newmax = 1);
   ///@}

   //-----------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor. \p n gives the initial size of the index space.
   explicit DIdxSet(int n = 8);

   /// copy constructor from IdxSet.
   explicit DIdxSet(const IdxSet& old);

   /// copy constructor from DIdxSet.
   DIdxSet(const DIdxSet& old);

   /// assignment operator from IdxSet
   DIdxSet& operator=(const IdxSet& sv)
   {
      if(this != &sv)
      {
         setMax(sv.size());
         IdxSet::operator=(sv);
      }

      return *this;
   }
   /// assignment operator from DIdxSet
   DIdxSet& operator=(const DIdxSet& sv)
   {
      if(this != &sv)
      {
         setMax(sv.size());
         IdxSet::operator=(sv);
      }

      return *this;
   }
   /// destructor.
   virtual ~DIdxSet();
   ///@}
};

} // namespace soplex
#endif // _DIDXSET_H_
