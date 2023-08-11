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

#include "soplex/spxout.h"
#include "soplex/exceptions.h"
#include "soplex/spxalloc.h"

namespace soplex
{
/// constructor
SPxOut::SPxOut()
   : m_verbosity(ERROR)
   , m_streams(0)
{
   spx_alloc(m_streams, INFO3 + 1);
   m_streams = new(m_streams) std::ostream*[INFO3 + 1];
   m_streams[ ERROR ] = m_streams[ WARNING ] = &std::cerr;

   for(int i = DEBUG; i <= INFO3; ++i)
      m_streams[ i ] = &std::cout;
}

//---------------------------------------------------

// destructor
SPxOut::~SPxOut()
{
   spx_free(m_streams);
}

SPxOut& SPxOut::operator=(const SPxOut& base)
{
   if(this != &base)
      m_verbosity = base.m_verbosity;

   for(int i = DEBUG; i <= INFO3; ++i)
      m_streams[ i ] = base.m_streams[ i ];

   return *this;
}

SPxOut::SPxOut(const SPxOut& rhs)
{
   m_verbosity = rhs.m_verbosity;
   m_streams = 0;
   spx_alloc(m_streams, INFO3 + 1);
   m_streams = new(m_streams) std::ostream*[INFO3 + 1];
   m_streams[ ERROR ] = m_streams[ WARNING ] = rhs.m_streams[ERROR];

   for(int i = DEBUG; i <= INFO3; ++i)
      m_streams[ i ] = rhs.m_streams[ i ];
}

} // namespace soplex
