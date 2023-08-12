/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file    paraPthLock.h
 * @brief   Pthread lock for UG Framework
 * @author  Yuji Shinano
 *
 * Many thanks to Daniel Junglas.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __PARA_PTH_LOCK_H__
#define __PARA_PTH_LOCK_H__
#include <pthread.h>
#include <iostream>

namespace UG
{

// #define LOCK_VERBOSE 1 // undef or define to 0 to disable output
#ifndef LOCK_VERBOSE
#   define LOCK_VERBOSE 0
#endif

///
/// Exception that is thrown whenever something goes wrong with a lock.
///
struct LockException : public std::exception {

   int code;             ///< error code

   ///
   /// constructor
   ///
   LockException(
         int c = 0      ///< no error is assumed
         )
         : code(c)
   {
   }

   ///
   /// get error code
   /// @return error code
   ///
   int getCode(
         )
   {
      return code;
   }

};

///
/// Class that implements a lock.
/// The class wraps around pthread_mutex_t and adds some safeguards.
///
class Lock {

   friend class ConditionVariable;

   pthread_mutex_t mtx;      ///< The low-level mutex that implements this lock.
   char const *file;         ///< File in which the lock was last acquired (debugging).
   int line;                 ///< Line at which the lock was last acquired (debugging).

   /// Code like this
   ///   pthread_mutex_t m1, m2;
   ///   pthread_mutex_init(&m1, 0);
   ///   m2 = m1;
   /// results in undefined behavior. So we must never assign one instance
   /// of pthread_mutex_t to another. To do that we make the copy constructor
   /// and the assignment operator private and do not implement it. This way
   /// locks can be passed around only by reference or pointer.
   Lock(
         Lock const&
         );
   Lock& operator=(
         Lock const&
         );

public:
   ///
   ///  Initialize this lock.
   ///
   Lock(
         )
         : file(0),
           line(-1)
   {
      int const error = pthread_mutex_init(&mtx, 0);
      if ( error )
         throw LockException(error);
   }

   ///
   /// Destroy this lock.
   ///
   ~Lock(
         )
   {
      pthread_mutex_destroy(&mtx);
   }

   ///
   /// Acquire this lock.
   /// The function sets the internal file/line (debugging) fields to
   /// generic values.
   ///
   void lock(
         )
   {
      lock("?", 0);
   }

   ///
   /// Acquire this lock.
   /// The function sets the internal file/line (debugging) fields to
   /// the values specified by f and l.
   ///
   void lock(
         char const *f,     ///< file name
         int l              ///< line number
         )
   {
      int const error = pthread_mutex_lock(&mtx);
      if ( error )
         throw LockException(error);
      file = f;
      line = l;
      if ( LOCK_VERBOSE )
         std::cout << "locked:" << &mtx << std::endl;
   }

   ///
   /// Release this lock.
   ///
   void unlock(
         )
   {
      if ( LOCK_VERBOSE )
         std::cout << "unlocked:" << &mtx << std::endl;
      file = 0;
      line = -1;
      int const error = pthread_mutex_unlock(&mtx);
      if ( error )
         throw LockException(error);
   }
};

///
/// Class to do RAII with a lock.
/// The constructor will acquire the lock and the destructor will delete it.
///
class LockRAII {

   Lock *const lck;       ///< lock

   /// No copying or assignment for instances of this class.
   LockRAII(
         LockRAII const &
         );
   LockRAII const &operator=(
         LockRAII const &
         );

public:

   ///
   /// Constructor.
   /// The constructor calls l->lock() to acquire the lock.
   ///
   LockRAII(
         Lock *l
         )
         : lck(l)
   {
      if ( !lck )
         throw LockException(-1);
      lck->lock();
   }

   ///
   /// Constructor.
   /// The constructor calls l->lock(file,line) to acquire the lock.
   ///
   LockRAII(
         Lock *l,
         char const *file,
         int line
         )
         : lck(l)
   {
      if ( !lck )
         throw LockException(-1);
      lck->lock(file, line);
   }
   ///
   /// Destructor.
   /// Releases the lock that was acquired in the constructor.
   ///
   ~LockRAII(
         )
   {
      lck->unlock();
   }

};

///
/// Same as LockRAII but with additional members to implement the LOCKED
/// macro.
///
class LockRAIIHelper : public LockRAII {

   int count;

   LockRAIIHelper(
         LockRAIIHelper const &
         );
   LockRAIIHelper const &operator=(
         LockRAIIHelper const &
         );

public:

   LockRAIIHelper(
         Lock *l
         )
         : LockRAII(l),
           count(0)
   {
   }

   LockRAIIHelper(
         Lock *l,
         char const *file,
         int line
         )
         : LockRAII(l, file, line),
           count(0)
   {
   }

   operator bool(
         ) const
   {
      return count == 0;
   }

   LockRAIIHelper &operator++(
         )
   {
      ++count; return *this;
   }

};

/** Macro to execute code that is guarded by a lock.
 * Usage is like this:
 * <code>
   Lock l;
   LOCKED(&l) CODE
   </code>
 * where CODE is either a single statement or a block.
 * The macro acquires l, executes CODE and then releases l. Acquisition
 * and release of the lock are exception safe.
 */
#define LOCKED(lck)                                     \
   for (LockRAIIHelper raii_(lck,__FILE__, __LINE__); raii_; ++raii_)

#define LOCK_RAII(lck)                                     \
   LockRAII raii_(lck, __FILE__, __LINE__)

}

#endif  // __PARA_PTH_LOCK_H__
