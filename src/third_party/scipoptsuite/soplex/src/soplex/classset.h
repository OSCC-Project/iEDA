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

/**@file  classset.h
 * @brief Set of class objects.
 */
#ifndef _CLASSSET_H_
#define _CLASSSET_H_


#include <string.h>
#include <assert.h>

#include "soplex/array.h"
#include "soplex/dataarray.h"
#include "soplex/classarray.h"
#include "soplex/datakey.h"
#include "soplex/spxalloc.h"
#include "soplex/exceptions.h"
#include "soplex/svectorbase.h"

namespace soplex
{
/**@class ClassSet
   @brief   Set of class objects.
   @ingroup Elementary

   Class ClassSet manages of sets of class objects  of a
   template type T. For constructing a ClassSet the maximum number
   of entries must be given. The current maximum number may be inquired
   with method max().

   Adding more then max() elements to a ClassSet will core dump. However,
   method reMax() allows to reset max() without loss of elements currently
   in the ClassSet. The current number of elements in a ClassSet is returned
   by method num().

   Adding elements to a ClassSet is done via methods add() or create(),
   while remove() removes elements from a ClassSet. When adding an element
   to a ClassSet the new element is assigned a DataKey. DataKeys serve to
   access CLASS elements in a set via a version of the subscript
   operator[](DataKey).

   For convenience all elements in a ClassSet are implicitely numbered
   from 0 through num()-1 and can be accessed with these numbers
   using a 2nd subscript operator[](int). The reason for providing
   DataKeys to access elements of a ClassSet is that the Key of an
   element remains unchanged as long as the element is a member of the
   ClassSet, while the numbers will change in an undefined way, if
   other elements are added to or removed from the ClassSet.

   The elements in a ClassSet and their DataKeys are stored in two arrays:
   - theitem keeps the objects along with their number stored in item.
   - thekey  keeps the DataKey::idx's of the elements in a ClassSet.

   Both arrays have size themax.

   In #thekey only elements 0 thru thenum-1 contain DataKey::idx%'s of
   valid elements, i.e., elements currently in the ClassSet.
   The current number of elements in the ClassSet is counted in thenum.

   In #theitem only elements 0 thru thesize-1 are used, but only some of
   them actually contain real class elements of the ClassSet. They are
   recognized by having info >= 0, which gives the number of that
   element. Otherwise info < 0 indicates an unused element. Unused
   elements are linked in a single linked list: starting with element
   <tt>-firstfree-1</tt>, the next free element is given by
   <tt>-info-1.</tt> The last free element in the list is marked by
   <tt>info == -themax-1.</tt> Finally all elements in theitem with
   <tt>index >= thesize</tt> are unused as well.
*/
template<class T>
class ClassSet
{
protected:

   //-----------------------------------
   /**@name Types */
   ///@{
   ///
   struct Item
   {
      T data;           ///< T element
      int  info;       ///< element number. info \f$\in\f$ [0,thesize-1]
      ///< iff element is used
   }* theitem;         ///< array of elements in the ClassSet
   ///@}

   //-----------------------------------
   /**@name Data */
   ///@{
   DataKey* thekey;    ///< DataKey::idx's of elements
   int themax;         ///< length of arrays theitem and thekey
   int thesize;        ///< highest used element in theitem
   int thenum;         ///< number of elements in ClassSet
   int firstfree;      ///< first unused element in theitem
   ///@}

public:

   //-----------------------------------
   /**@name Extension
    *  Whenever a new element is added to a ClassSet, the latter assigns it a
    *  DataKey. For this all methods that extend a ClassSet by one ore more
    *  elements are provided with two signatures, one of them having a
    *  parameter for returning the assigned DataKey(s).
    */
   ///@{
   /// adds an element.
   void add(DataKey& newkey, const T& item)
   {
      T* newelem = create(newkey);

      assert(newelem != 0);

      *newelem = item;
   }
   /// adds element \p item.
   /**@return 0 on success and non-zero, if an error occured.
    */
   void add(const T& item)
   {
      T* newelem = create();


      assert(newelem != 0);

      *newelem = item;
   }

   /// add several items.
   void add(DataKey newkey[], const T* item, int n)
   {
      assert(n         >= 0);
      assert(num() + n <= max());

      for(int i = 0; i < n; ++i)
         add(newkey[i], item[i]);
   }

   /// adds \p n elements from \p items.
   void add(const T* items, int n)
   {
      assert(n         >= 0);
      assert(num() + n <= max());

      for(int i = 0; i < n; ++i)
         add(items[i]);
   }

   /// adds several new items.
   void add(DataKey newkey[], const ClassSet < T >& set)
   {
      assert(num() + set.num() <= max());

      for(int i = 0; i < set.num(); ++i)
         add(newkey[i], set[i]);
   }

   /// adds all elements of \p set.
   void add(const ClassSet < T >& set)
   {
      assert(num() + set.num() <= max());

      for(int i = 0; i < set.num(); ++i)
         add(set[i]);
   }

   /// creates new class element in ClassSet.
   /**@return Pointer to the newly created element.
    */
   T* create(DataKey& newkey)
   {
      assert(num() < max());

      if(firstfree != -themax - 1)
      {
         newkey.idx = -firstfree - 1;
         firstfree = theitem[newkey.idx].info;
      }
      else
         newkey.idx = thesize++;

      thekey[thenum] = newkey;
      theitem[newkey.idx].info = thenum;
      ++thenum;

      return &(theitem[newkey.idx].data);
   }
   /// creates new (uninitialized) class element in ClassSet.
   /**@return Pointer to the newly created element.
    */
   T* create()
   {
      DataKey tmp;
      return create(tmp);
   }
   ///@}

   //-----------------------------------
   /**@name Shrinkage
    * When elements are removed from a ClassSet, the remaining ones are
    * renumbered from 0 through the new size()-1. How this renumbering is
    * performed will not be revealed, since it might be target of future
    * changes. However, some methods provide a parameter
    * <tt>int* perm</tt>, which
    * returns the new order after the removal: If <tt>perm[i] < 0</tt>,
    * the element numbered i prior to the removal operation has been removed
    * from the set. Otherwise, <tt>perm[i] = j >= 0</tt> means that the
    * element with number i prior to the removal operation has been
    * renumbered to j. Removing a single element from a ClassSet yields a
    * simple renumbering of the elements: The last element in the set
    * (i.e., element num()-1) is moved to the index of the removed element.
    */
   ///@{
   /// removes the \p removenum 'th element.
   void remove(int removenum)
   {
      if(has(removenum))
      {
         int idx = thekey[removenum].idx;

         theitem[idx].info = firstfree;
         firstfree = -idx - 1;

         while(-firstfree == thesize)
         {
            firstfree = theitem[ -firstfree - 1].info;
            --thesize;
         }

         --thenum;

         if(removenum != thenum)
         {
            thekey[removenum] = thekey[thenum];
            theitem[thekey[removenum].idx].info = removenum;
         }
      }
   }

   /// removes element with key \p removekey.
   void remove(const DataKey& removekey)
   {
      remove(number(removekey));
   }

   /// remove multiple elements.
   /** This method removes all elements for the ClassSet with an
    *  index i such that \p perm[i] < 0. Upon completion, \p perm contains
    *  the new numbering of elements.
    */
   void remove(int perm[])
   {
      int k, j, first = -1;

      // setup permutation and remove items
      for(k = j = 0; k < num(); ++k)
      {
         if(perm[k] >= 0)       // j has not been removed ...
            perm[k] = j++;
         else
         {
            int idx = thekey[k].idx;
            theitem[idx].info = firstfree;
            firstfree = -idx - 1;

            if(first < 0)
               first = k;
         }
      }

      if(first >= 0)         // move remaining items
      {
         for(k = first, j = num(); k < j; ++k)
         {
            if(perm[k] >= 0)
            {
               thekey[perm[k]] = thekey[k];
               theitem[thekey[k].idx].info = perm[k];
               thekey[k].idx = -1;
            }
            else
               --thenum;
         }
      }
   }

   /// remove \p n elements given by \p keys and \p perm.
   void remove(const DataKey* keys, int n, int* perm)
   {
      assert(perm != 0);

      for(int i = num() - 1; i >= 0; --i)
         perm[i] = i;

      while(--n >= 0)
         perm[number(keys[n])] = -1;

      remove(perm);
   }
   /// remove \p n elements given by \p keys.
   void remove(const DataKey* keys, int n)
   {
      DataArray<int> perm(num());
      remove(keys, n, perm.get_ptr());
   }
   /// remove \p n elements given by \p nums and \p perm.
   void remove(const int* nums, int n, int* perm)
   {
      assert(perm != 0);

      for(int i = num() - 1; i >= 0; --i)
         perm[i] = i;

      while(--n >= 0)
         perm[nums[n]] = -1;

      remove(perm);
   }
   /// remove \p n elements with numbers \p nums.
   void remove(const int* nums, int n)
   {
      DataArray<int> perm(num());
      remove(nums, n, perm.get_ptr());
   }

   /// remove all elements.
   void clear()
   {
      thesize = 0;
      thenum = 0;
      firstfree = -themax - 1;
   }
   ///@}

   //-----------------------------------
   /**@name Access
    * When accessing elements from a ClassSet with one of the index
    * operators, it must be ensured that the index is valid for the
    * ClassSet. If this is not known afore, it is the programmers
    * responsability to ensure this using the inquiry methods below.
    */
   ///@{
   ///
   T& operator[](int n)
   {
      assert(n >= 0 && n < thenum);
      return theitem[thekey[n].idx].data;
   }
   /// returns element number \p n.
   const T& operator[](int n) const
   {
      assert(n >= 0 && n < thenum);
      return theitem[thekey[n].idx].data;
   }

   ///
   T& operator[](const DataKey& k)
   {
      assert(k.idx < thesize);
      return theitem[k.idx].data;
   }
   /// returns element with DataKey \p k.
   const T& operator[](const DataKey& k) const
   {
      assert(k.idx < thesize);
      return theitem[k.idx].data;
   }
   ///@}

   //-----------------------------------
   /**@name Inquiry */
   ///@{
   /// returns maximum number of elements that would fit into ClassSet.
   int max() const
   {
      return themax;
   }

   /// returns number of elements currently in ClassSet.
   int num() const
   {
      return thenum;
   }

   /// returns the maximum DataKey::idx currently in ClassSet.
   int size() const
   {
      return thesize;
   }

   /// returns DataKey of \p n 'th element in ClassSet.
   DataKey key(int n) const
   {
      assert(n >= 0 && n < num());
      return thekey[n];
   }

   /// returns DataKey of element \p item in ClassSet.
   DataKey key(const T* item) const
   {
      assert(number(item) >= 0);
      return thekey[number(item)];
   }

   /// returns the number of the element with DataKey \p k in ClassSet or -1,
   /// if it doesn't exist.
   int number(const DataKey& k) const
   {
      if(k.idx < 0 || k.idx >= size())
         throw SPxException("Invalid index");

      return theitem[k.idx].info;
   }

   /**@todo Please check whether this is correctly implemented! */
   /// returns the number of element \p item in ClassSet,
   /// throws exception if it doesn't exist.
   int number(const T* item) const
   {
      ptrdiff_t idx = reinterpret_cast<const struct Item*>(item) - theitem;

      if(idx < 0 || idx >= size())
         throw SPxException("Invalid index");

      return theitem[idx].info;
   }

   /// Is \p k a valid DataKey of an element in ClassSet?
   bool has(const DataKey& k) const
   {
      return theitem[k.idx].info >= 0;
   }

   /// Is \p n a valid number of an element in ClassSet?
   bool has(int n) const
   {
      return (n >= 0 && n < num());
   }

   /// Does \p item belong to ClassSet?
   bool has(const T* item) const
   {
      int n;

      try
      {
         n = number(item);
      }
      catch(...)
      {
         return false;
      }

      return n >= 0;
   }
   ///@}

   //-----------------------------------
   /**@name Miscellaneous */
   ///@{
   /// resets max() to \p newmax.
   /** This method will not succeed if \p newmax < size(), in which case
    *  \p newmax == size() will be taken. As generally this method involves
    *  copying the #ClassSet%s elements in memory, reMax() returns the
    *  number of bytes the addresses of elements in the ClassSet have been
    *  moved. Note, that this is identical for all elements in the
    *  ClassSet.
    */
   ptrdiff_t reMax(int newmax = 0)
   {
      int i;
      Item* newMem = 0;
      newmax = (newmax < size()) ? size() : newmax;

      int* lastfree = &firstfree;

      while(*lastfree != -themax - 1)
         lastfree = &(theitem[ -1 - *lastfree].info);

      *lastfree = -newmax - 1;

      spx_alloc(newMem, newmax);

      /* call copy constructor for first elements */
      for(i = 0; i < max(); i++)
      {
         newMem[i].data = std::move(theitem[i].data);
         newMem[i].info = theitem[i].info;
      }

      /* call default constructor for remaining elements */
      for(; i < newmax; i++)
         new(&(newMem[i])) Item();

      /* compute pointer difference */
      ptrdiff_t pshift = reinterpret_cast<char*>(newMem) - reinterpret_cast<char*>(theitem);

      spx_free(theitem);

      theitem = newMem;
      themax = newmax;

      spx_realloc(thekey,  themax);

      return pshift;
   }

   /// consistency check.
   bool isConsistent() const
   {
#ifdef ENABLE_CONSISTENCY_CHECKS

      if(theitem == 0 || thekey == 0)
         return MSGinconsistent("ClassSet");

      if(thesize > themax || thenum > themax || thenum > thesize)
         return MSGinconsistent("ClassSet");

      if(thesize == thenum && firstfree != -themax - 1)
         return MSGinconsistent("ClassSet");

      if(thesize != thenum && firstfree == -themax - 1)
         return MSGinconsistent("ClassSet");

      for(int i = 0; i < thenum; ++i)
         if(theitem[thekey[i].idx].info != i)
            return MSGinconsistent("ClassSet");

#endif

      return true;
   }
   ///@}

   //-----------------------------------
   /**@name Constructors / Destructors */
   ///@{
   /// default constructor.
   explicit
   ClassSet(int pmax = 8)
      : theitem(0)
      , thekey(0)
      , themax(pmax < 1 ? 8 : pmax)
      , thesize(0)
      , thenum(0)

   {
      firstfree = -themax - 1;

      spx_alloc(theitem, themax);

      /* call default constructor for each element */
      for(int i = 0; i < themax; i++)
         new(&(theitem[i])) Item();

      try
      {
         spx_alloc(thekey, themax);
      }
      catch(const SPxMemoryException& x)
      {
         spx_free(theitem);
         throw x;
      }

      assert(isConsistent());
   }

   /// copy constructor.
   ClassSet(const ClassSet& old)
      : theitem(0)
      , thekey(0)
      , themax(old.themax)
      , thesize(old.thesize)
      , thenum(old.thenum)
   {
      firstfree = (old.firstfree == -old.themax - 1)
                  ? -themax - 1
                  : old.firstfree;

      spx_alloc(theitem, themax);

      /* call copy constructor for first elements */
      int i;

      for(i = 0; i < old.thenum; i++)
         new(&(theitem[i])) Item(old.theitem[i]);

      /* call default constructor for remaining elements */
      for(; i < old.themax; i++)
         new(&(theitem[i])) Item();

      try
      {
         spx_alloc(thekey, themax);
      }
      catch(const SPxMemoryException& x)
      {
         spx_free(theitem);
         throw x;
      }

      memcpy(thekey,  old.thekey,  themax * sizeof(*thekey));

      assert(isConsistent());
   }

   /// assignment operator.
   /** The assignment operator involves #reMax()%ing the lvalue ClassSet
    *  to the size needed for copying all elements of the rvalue. After the
    *  assignment all DataKeys from the lvalue are valid for the rvalue as
    *  well. They refer to a copy of the corresponding class elements.
    */
   ClassSet < T >& operator=(const ClassSet < T >& rhs)
   {
      if(this != &rhs)
      {
         int i;

         if(rhs.size() > max())
            reMax(rhs.size());

         clear();

         for(i = 0; i < rhs.size(); ++i)
            theitem[i] = std::move(rhs.theitem[i]);

         for(i = 0; i < rhs.num(); ++i)
            thekey[i] = rhs.thekey[i];

         if(rhs.firstfree == -rhs.themax - 1)
            firstfree = -themax - 1;
         else
         {
            firstfree = rhs.firstfree;
            i = rhs.firstfree;

            while(rhs.theitem[ -i - 1].info != -rhs.themax - 1)
               i = rhs.theitem[ -i - 1].info;

            theitem[ -i - 1].info = -themax - 1;
         }

         thenum = rhs.thenum;
         thesize = rhs.thesize;

         assert(isConsistent());
      }

      return *this;
   }

   /// destructor.
   ~ClassSet()
   {
      if(theitem)
         spx_free(theitem);

      if(thekey)
         spx_free(thekey);
   }
   ///@}
};

} // namespace soplex
#endif // _CLASSSET_H_
