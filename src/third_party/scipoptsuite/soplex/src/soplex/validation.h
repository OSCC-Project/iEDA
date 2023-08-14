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

/**@file  validation.h
 * @brief Validation object for soplex solutions
 */

#ifndef SRC_VALIDATION_H_
#define SRC_VALIDATION_H_

#include "soplex.h"

namespace soplex
{

template <class R>
class Validation
{
public:

   /// should the soplex solution be validated?
   bool           validate;

   /// external solution used for validation
   std::string          validatesolution;

   /// tolerance used for validation
   R         validatetolerance;

   /// default constructor
   Validation()
   {
      validate = false;
      validatetolerance = 1e-5;
   }

   /// default destructor
   ~Validation()
   {
      ;
   }

   /// updates the external solution used for validation
   bool updateExternalSolution(const std::string& solution);

   /// updates the tolerance used for validation
   bool updateValidationTolerance(const std::string& tolerance);

   /// validates the soplex solution using the external solution
   void validateSolveReal(SoPlexBase<R>& soplex);
};

} /* namespace soplex */

// For general templated functions
#include "validation.hpp"

#endif /* SRC_VALIDATION_H_ */
