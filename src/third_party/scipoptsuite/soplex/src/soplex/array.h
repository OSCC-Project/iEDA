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

/**@file  array.h
 * @brief Save arrays of arbitrary types.
 */
#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <assert.h>
#include <string.h>
#include <vector>
#include "soplex/spxalloc.h"

namespace soplex
{
/**@brief   Safe arrays of arbitrary types.
   @ingroup Elementary

   Class Array provides safe arrays of arbitrary type. Array elements are
   accessed just like ordinary C++ array elements by means of the index
   operator[](). Safety is provided by

    - automatic memory management in constructor and destructor
      preventing memory leaks
    - checking of array bound when accessing elements with the
      indexing operator[]() (only when compiled without \c -DNDEBUG).

    Moreover, #Array%s may easily be extended by #insert%ing or
    #append%ing elements to the Array or shrunken by
    \ref remove() "removing"
    elements. Method reSize(int n) resets the Array's length to \p n,
    thereby appending elements or truncating the Array to the
    required size.

    An Array is implemented in a C++-compliant way with respect to
    how memory is managed: Only operators new and delete are
    used for allocating memory. This involves some overhead for all
    methods effecting the length of an Array, i.e., all methods
    insert(), append(), remove() and reSize(). This involves
    allocating a new C++ array of the new size and copying all
    elements with the template parameters operator=().

    For this reason, it is not convenient to use class Array if its elements
    are \ref DataObjects "Data Objects". In this case use class DataArray
    instead.

    @see DataArray, \ref DataObjects "Data Objects"
*/
template < class T >
class Array
{
   static_assert(!std::is_same<T, bool>::value,
                 "Since Array wraps std::vector, bool is not allowed to avoid unallowed behavior");
protected:

   //----------------------------------------
   /**@name Data */
   ///@{
   std::vector<T> data;
   ///@}

public:

   //----------------------------------------
   /**@name Access / modification */
   ///@{
   /// reference \p n 'th element.
   T& operator[](int n)
   {
      assert(n >= 0 && n < int(data.capacity()));
      return data[n];
   }
   /// reference \p n 'th element.
   const T& operator[](int n) const
   {
      assert(n >= 0 && n < int(data.capacity()));
      return data[n];
   }

   /** This function serves for using a Vector in an C-style
    *  function. It returns a pointer to the first value of the array.
    */
   T* get_ptr()
   {
      return data.data();
   }
   /// get a const C pointer to the data.
   const T* get_const_ptr() const
   {
      return data.data();
   }

   /// append 1 elements with value \p t.
   void append(const T& t)
   {
      data.push_back(t);
   }
   /// append \p n uninitialized elements.
   void append(int n)
   {
      T newt = T();
      this->append(n, newt);
   }
   /// append \p n elements with value \p t.
   void append(int n, const T& t)
   {
      data.insert(data.end(), n, t);
   }
   /// append \p n elements from \p t.
   void append(int n, const T t[])
   {
      data.insert(data.end(), t, t + n);
   }
   /// append all elements from \p p_array.
   void append(const Array<T>& t)
   {
      data.insert(data.end(), t.data.begin(), t.data.end());
   }

   /// insert \p n uninitialized elements before \p i 'th element.
   void insert(int i, int n)
   {
      T newt = T();

      if(n > 0)
         data.insert(data.begin() + i - 1, n, newt);
   }

   /// insert \p n elements with value \p t before \p i 'the element.
   void insert(int i, int n, const T& t)
   {
      if(n > 0)
      {
         data.insert(data.begin() + i - 1, n, t);
      }
   }

   /// insert \p n elements from \p p_array before \p i 'th element.
   void insert(int i, int n, const T t[])
   {
      if(n > 0)
      {
         data.insert(data.begin() + i - 1, t, t + n);
      }
   }

   /// insert all elements from \p p_array before \p i 'th element.
   void insert(int i, const Array<T>& t)
   {
      if(t.size())
      {
         data.insert(data.begin() + i - 1, t.data.begin(), t.data.end());
      }
   }

   /// remove \p m elements starting at \p n.
   void remove(int n = 0, int m = 1)
   {
      assert(n < size() && n >= 0);

      if(n + m < size())
      {
         data.erase(data.begin() + n, data.begin() + n + m);
      }
      else
      {
         data.erase(data.begin() + n, data.end());
      }
   }

   /// remove all elements.
   void clear()
   {
      data.clear();
   }

   /// return the number of elements.
   int size() const
   {
      return int(data.size());
   }

   /// reset the number of elements.
   void reSize(int newsize)
   {
      data.resize(newsize);
   }

   ///@}

   //----------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// assignment operator.
   Array<T>& operator=(const Array<T>& rhs)
   {
      if(this != &rhs)
      {
         reSize(rhs.size());
         data = rhs.data;
      }

      return *this;
   }

   // Move assignment for Array
   Array& operator=(const Array&& rhs)
   {
      data = std::move(rhs.data);
      return *this;
   }

   /// default constructor.
   /** The constructor allocates an Array of \p n uninitialized elements.
    */
   explicit
   Array(int n = 0)
   {
      data.resize(n);
   }

   /// copy constructor
   Array(const Array& old)
   {
      data = old.data;
   }

   /// destructor
   ~Array()
   {
      ;
   }

   void push_back(const T& val)
   {
      data.push_back(val);
   }

   void push_back(T&& val)
   {
      data.push_back(val);
   }

   /// Consistency check.
   bool isConsistent() const
   {
      return true;
   }

   ///@}
};
} // namespace soplex
#endif // _ARRAY_H_
