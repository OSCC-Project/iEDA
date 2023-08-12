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

/**@file  spxout.h
 * @brief Wrapper for different output streams and verbosity levels.
 */
#ifndef _SPXOUT_H_
#define _SPXOUT_H_

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "soplex/spxdefines.h"

// ----------------------------------------------------------------------
//    class SPxOut
// ----------------------------------------------------------------------

namespace soplex
{

/**@class SPxOut
   @ingroup Elementary

   @brief Wrapper for several output streams.
   A verbosity level is used to decide which stream to use and whether to
   really print a given message. Regardless of whether the verbosity level
   is set via a manipulator or via the member function, it is persistent
   until a new value is set.

   Most ostream member functions are not provided here; use the corresponding
   stream manipulators (e.g., @c setprecision()) instead. These are passed on
   to the <em>current</em> ostream, which is chosen according to the verbosity
   level. In particular, this means that the first element in an output stream
   should always be the verbosity. For instance, use
   @code
      spxout << verb( SPxOut::WARNING ) << std::setw( 15 ) << 42 << std::endl;
   @endcode
   or
   @code
      spxout.setVerbosity( SPxOut::WARNING );
      spxout << std::setw( 15 ) << 42 << std::endl;
   @endcode
   instead of
   @code
      spxout << std::setw( 15 ) << verb( SPxOut::WARNING ) << 42 << std::endl;
   @endcode
   in order to make sure that @c std::setw( 15 ) is applied to the warning stream.
*/
class SPxOut
{
public:


   //-----------------------------------
   /**@name Output control types */
   ///@{
   /// Verbosity level
   typedef enum
   {
      // Note: the implementation uses the fact that ERROR == 0
      // and that the verbosity levels are subsequent numbers.
      // If you change this, change the implementation as well.
      ERROR    = 0,
      WARNING  = 1,
      DEBUG    = 2,
      INFO1    = 3,
      INFO2    = 4,
      INFO3    = 5
   } Verbosity;

   /// helper struct for the output operator
   struct struct_Verbosity
   {
      /// verbosity level
      Verbosity v_;
   };
   ///@}

   //-----------------------------------
   /**@name Construction / destruction */
   ///@{
   /// constructor
   SPxOut();
   /// destructor
   virtual ~SPxOut();
   /// copy constructor
   SPxOut(const SPxOut&);
   /// assignment operator
   SPxOut& operator=(const SPxOut&);
   ///@}

   //-----------------------------------
   /**@name Verbosity */
   ///@{
   ///
   virtual void
   setVerbosity(const Verbosity& v)
   {
      m_verbosity = v;
   }
   ///
   inline Verbosity
   getVerbosity()
   const
   {
      return m_verbosity;
   }

   ///@}

   //----------------------------------------
   /**@name Wrappers for the current stream */
   ///@{
   ///
   inline bool good() const
   {
      return getCurrentStream().good();
   }
   ///
   inline bool operator !() const
   {
      return ! getCurrentStream();
   }
   ///
   inline std::streamsize precision() const
   {
      return getCurrentStream().precision();
   }
   ///@}

   //-----------------------------------
   /**@name Getting / setting streams */
   ///@{
   /// Sets the stream for the specified verbosity level.
   virtual void
   setStream(const Verbosity& verbosity,
             std::ostream&   stream)
   {
      m_streams[ verbosity ] = &stream;
   }
   /// Returns the stream for the specified verbosity level.
   inline std::ostream&
   getStream(const Verbosity& verbosity)
   const
   {
      return *(m_streams[ verbosity ]);
   }
   /// Returns the stream for the current verbosity.
   inline std::ostream&
   getCurrentStream()
   const
   {
      return getStream(getVerbosity());
   }

   /// Sets the precision of the stream to 16 and the floatfield to scientifix.
   static inline void setScientific(std::ostream& stream, int precision = 8)
   {
      stream << std::setprecision(precision) << std::scientific;
   }

   /// Sets the precision of the stream to 8 and the floatfield to fixed.
   static inline void setFixed(std::ostream& stream, int precision = 8)
   {
      stream << std::setprecision(precision) << std::fixed;
   }
   ///@}

private:

   //-----------------------------------
   /**@name Private data */
   ///@{
   /// verbosity level
   Verbosity               m_verbosity;
   /// array of pointers to internal streams, indexed by verbosity level
   std::ostream**          m_streams;
   ///@}
};

// ---------------------------------------------------------
//    Manipulators
// ---------------------------------------------------------


//-------------------------------------------
/**@name Verbosity manipulator
    Manipulators are implemented in a similar way as done for @c setw(),
    @c setprecision(), etc. in the standard library file iomanip. For
    instance, the non-member function \ref verb() "verb(v)" returns a
    struct struct_Severity which contains only the verbosity level.
    Calling
    @code
         SPxOut spxout;
         spxout << verb( SPxOut::ERROR ) << "This is an error!" << std::endl;
    @endcode
    passes such a struct to the output operator defined below, which
    extracts the verbosity level from the struct and passes it to the
    member function SPxOut::setVerbosity().
*/
//@{
/// manipulator to be used in an output statement
inline SPxOut::struct_Verbosity
verb(const SPxOut::Verbosity&  v)
{
   SPxOut::struct_Verbosity verbosity;
   verbosity.v_ = v;
   return verbosity;
}

/// output operator with verbosity level struct
inline SPxOut&
operator<< (SPxOut& stream,
            const SPxOut::struct_Verbosity&  verbosity)
{
   stream.setVerbosity(verbosity.v_);
   return stream;
}
///@}

//--------------------------------------------------------
/**@name Output of standard manipulators and other types
 *
 * We have to define an output operator for many kinds of numeric
 * types here because they can all be more or less casted into each
 * other. When using only a template type, it is not clear what the
 * compiler makes out of it (according to lint).
 */
//@{
///
/// Passes instances of type \p Type to the current stream.
inline SPxOut& operator<< (SPxOut& _spxout, long t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, unsigned long t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, bool t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, short t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, unsigned short t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, int t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, unsigned int t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, double t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, float t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, long double t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

inline SPxOut& operator<< (SPxOut& _spxout, const void* t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}

/// Passes standard manipulators without arguments, like @c std::endl
/// or @c std::ios::right to the current stream.
inline SPxOut&
operator<< (SPxOut&       _spxout,
            std::ostream & (*manip)(std::ostream&))
{
   _spxout.getCurrentStream() << manip;
   return _spxout;
}

//lint -e{818} (pointer could be made const; this is ok.)
/// Passes everything else to the current stream. In particular,
/// this includes structs corresponding to manipulators with arguments,
/// such as the struct @c _Setw for the @c setw() manipulator.
template <typename T>
inline SPxOut& operator<< (SPxOut& _spxout, T  t)
{
   _spxout.getCurrentStream() << t;
   return _spxout;
}
//@}

}  // namespace soplex

#endif // _SPXOUT_H_
