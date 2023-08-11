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

/**@file    paraDef.h
 * @brief   Defines for UG Framework
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_DEF_H__
#define __PARA_DEF_H__
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <cfloat>

namespace UG
{

#define UG_VERSION      100  /**< UG version number (multiplied by 100 to get integer number) */
#define DEFAULT_NUM_EPSILON          1e-9  /**< default upper bound for floating points to be considered zero */
#define MINEPSILON                   1e-20  /**< minimum value for any numerical epsilon */

#define THROW_LOGICAL_ERROR1( msg1 ) \
   { \
   std::ostringstream s; \
   s << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 );  \
   throw std::logic_error( s.str() ); \
   }

#define ABORT_LOGICAL_ERROR1( msg1 ) \
   { \
   std::cerr << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 );  \
   abort(); \
   }

#define THROW_LOGICAL_ERROR2( msg1, msg2 ) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 );  \
   throw std::logic_error( s_.str() ); \
   }

#define ABORT_LOGICAL_ERROR2( msg1, msg2 ) \
   { \
   std::cerr << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 );  \
   abort(); \
   }

#define THROW_LOGICAL_ERROR3( msg1, msg2, msg3 ) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 );  \
   throw std::logic_error( s_.str() ); \
   }

#define ABORT_LOGICAL_ERROR3( msg1, msg2, msg3 ) \
   { \
   std::cerr << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 );  \
   abort(); \
   }

#define THROW_LOGICAL_ERROR4( msg1, msg2, msg3, msg4 ) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ <<  "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 );  \
   throw std::logic_error( s_.str() ); \
   }

#define THROW_LOGICAL_ERROR5( msg1, msg2, msg3, msg4, msg5) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 ) << ( msg5 );  \
   throw std::logic_error( s_.str() ); \
   }

#define THROW_LOGICAL_ERROR6( msg1, msg2, msg3, msg4, msg5, msg6) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 ) << ( msg5 ) << ( msg6 );  \
   throw std::logic_error( s_.str() ); \
   }

#define THROW_LOGICAL_ERROR7( msg1, msg2, msg3, msg4, msg5, msg6, msg7) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 ) << ( msg5 ) << ( msg6 ) << ( msg7 );  \
   throw std::logic_error( s_.str() ); \
   }

#define THROW_LOGICAL_ERROR8( msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 ) << ( msg5 ) << ( msg6 ) << ( msg7 ) << ( msg8 );  \
   throw std::logic_error( s_.str() ); \
   }

#define THROW_LOGICAL_ERROR9( msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9) \
   { \
   std::ostringstream s_; \
   s_ << "[LOGICAL ERROR:" <<  __FILE__ << "] func = " \
     << __func__ << ", line = " << __LINE__ << " - " \
     << ( msg1 ) << ( msg2 ) << ( msg3 ) << ( msg4 ) << ( msg5 ) << ( msg6 ) << ( msg7 ) << ( msg8 ) << ( msg9 );  \
   throw std::logic_error( s_.str() ); \
   }

#ifdef _COMM_MPI_WORLD
#define DELETE_TRANSFER_OBJECT_IN_THREADED_SOLVER( object ) \

#else
#define DELETE_TRANSFER_OBJECT_IN_THREADED_SOLVER( object ) \
   if( object ) delete object
#endif

#define REALABS(x)        (fabs(x))
#define EPSEQ(x,y,eps)    (REALABS((x)-(y)) <= (eps))
#define EPSLT(x,y,eps)    ((x)-(y) < -(eps))
#define EPSLE(x,y,eps)    ((x)-(y) <= (eps))
#define EPSGT(x,y,eps)    ((x)-(y) > (eps))
#define EPSGE(x,y,eps)    ((x)-(y) >= -(eps))
#define EPSZ(x,eps)       (REALABS(x) <= (eps))
#define EPSP(x,eps)       ((x) > (eps))
#define EPSN(x,eps)       ((x) < -(eps))
#define EPSFLOOR(x,eps)   (floor((x)+(eps)))
#define EPSCEIL(x,eps)    (ceil((x)-(eps)))
#define EPSFRAC(x,eps)    ((x)-EPSFLOOR(x,eps))
#define EPSISINT(x,eps)   (EPSFRAC(x,eps) <= (eps))

static const int MaxStrLen = 1024;
static const int LpMaxNamelen = 1024;

static const int CompTerminatedNormally           = 0;
static const int CompTerminatedByAnotherTask      = 1;
static const int CompTerminatedByInterruptRequest = 2;
static const int CompTerminatedInRacingStage      = 3;
static const int CompInterruptedInRacingStage     = 4;
static const int CompInterruptedInMerging         = 5;
static const int CompTerminatedByTimeLimit        = 6;
static const int CompTerminatedByMemoryLimit      = 7;

}

#endif // __PARA_DEF_H__
