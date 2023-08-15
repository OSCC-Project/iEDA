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

/**@file    paraParamSet.h
 * @brief   Parameter set for UG framework.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_PARAM_SET_H__
#define __PARA_PARAM_SET_H__
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include "paraComm.h"

#define OUTPUT_PARAM_VALUE_ERROR( msg1, msg2, msg3, msg4 ) \
   std::cout << "[PARAM VALUE ERROR] Param type = " << msg1 << ", Param name = " << msg2 \
     << ", Param value = " <<  msg3 <<  ": Param comment is as follows: " << std::endl \
     << msg4 << std::endl;  \
   return (-1)

namespace UG
{

///
///  Types of parameters
///
static const int ParaParamTypeBool    = 1;     ///< bool values: true or false
static const int ParaParamTypeInt     = 2;     ///< integer values
static const int ParaParamTypeLongint = 3;     ///< long integer values
static const int ParaParamTypeReal    = 4;     ///< real values
static const int ParaParamTypeChar    = 5;     ///< characters
static const int ParaParamTypeString  = 6;     ///< arrays of characters

///
///  Bool parameters
///
static const int ParaParamsFirst                    = 0;
static const int ParaParamsBoolFirst                = ParaParamsFirst;
//-------------------------------------------------------------------------
static const int Quiet                               = ParaParamsBoolFirst + 0;
static const int TagTrace                            = ParaParamsBoolFirst + 1;
static const int LogSolvingStatus                    = ParaParamsBoolFirst + 2;
static const int LogTasksTransfer                    = ParaParamsBoolFirst + 3;
static const int Checkpoint                          = ParaParamsBoolFirst + 4;
static const int Deterministic                       = ParaParamsBoolFirst + 5;
static const int StatisticsToStdout                  = ParaParamsBoolFirst + 6;
static const int DynamicAdjustNotificationInterval   = ParaParamsBoolFirst + 7;
//-------------------------------------------------------------------------
static const int ParaParamsBoolLast                  = ParaParamsBoolFirst + 7;
static const int ParaParamsBoolN                     = ParaParamsBoolLast - ParaParamsBoolFirst + 1;
///
/// Int parameters
///
static const int ParaParamsIntFirst                  = ParaParamsBoolLast  + 1;
//-------------------------------------------------------------------------
static const int OutputParaParams                    = ParaParamsIntFirst + 0;
static const int NotificationSynchronization         = ParaParamsIntFirst + 1;
//-------------------------------------------------------------------------
static const int ParaParamsIntLast                   = ParaParamsIntFirst + 1;
static const int ParaParamsIntN                      = ParaParamsIntLast - ParaParamsIntFirst + 1;
///
/// Longint parameters
///
static const int ParaParamsLongintFirst              = ParaParamsIntLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int ParaParamsLongintLast               = ParaParamsLongintFirst - 1;  // No params -1
static const int ParaParamsLongintN                  = ParaParamsLongintLast - ParaParamsLongintFirst + 1;
///
/// Real parameters
///
static const int ParaParamsRealFirst                 = ParaParamsLongintLast + 1;
//-------------------------------------------------------------------------
static const int NotificationInterval                  = ParaParamsRealFirst  + 0;
static const int TimeLimit                             = ParaParamsRealFirst  + 1;
static const int CheckpointInterval                    = ParaParamsRealFirst  + 2;
static const int FinalCheckpointGeneratingTime         = ParaParamsRealFirst  + 3;
//-------------------------------------------------------------------------
static const int ParaParamsRealLast                  = ParaParamsRealFirst  + 3;
static const int ParaParamsRealN                     = ParaParamsRealLast - ParaParamsRealFirst + 1;
///
/// Char parameters
///
static const int ParaParamsCharFirst                 = ParaParamsRealLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int ParaParamsCharLast                  = ParaParamsCharFirst - 1;   // No params -1
static const int ParaParamsCharN                     = ParaParamsCharLast - ParaParamsCharFirst + 1;
///
/// String parameters
///
static const int ParaParamsStringFirst               = ParaParamsCharLast      +1;
//-------------------------------------------------------------------------
static const int TempFilePath                        = ParaParamsStringFirst + 0;
static const int TagTraceFileName                    = ParaParamsStringFirst + 1;
static const int LogSolvingStatusFilePath            = ParaParamsStringFirst + 2;
static const int LogTasksTransferFilePath            = ParaParamsStringFirst + 3;
static const int SolutionFilePath                    = ParaParamsStringFirst + 4;
static const int CheckpointFilePath                  = ParaParamsStringFirst + 5;
static const int RacingParamsDirPath                 = ParaParamsStringFirst + 6;
//-------------------------------------------------------------------------
static const int ParaParamsStringLast                = ParaParamsStringFirst + 6;
static const int ParaParamsStringN                   = ParaParamsStringLast - ParaParamsStringFirst + 1;
static const int ParaParamsLast                      = ParaParamsStringLast;
static const int ParaParamsSize                      = ParaParamsLast + 1;

///
/// class ParaParam
///
class ParaParam {

   const char *paramName;        ///< parameter name
   const char *comment;          ///< comments for this parameter

public:

   ///
   /// constructor
   ///
   ParaParam(
         const char *inParamName,
         const char *inComment
         )
         : paramName(inParamName),
           comment(inComment)
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaParam(
         )
   {
   }

   ///
   /// getter of parameter name
   /// @return string of parameter name
   ///
   const char *getParamName(
         ) const
   {
      return paramName;
   }

   ///
   /// getter of comments string
   /// @return string of comments
   ///
   const char *getComment(
         ) const
   {
      return comment;
   }

   ///
   /// get parameter type
   /// @return parameter type value
   ///
   virtual int getType(
         ) const = 0;

};

///
/// class ParaParamBool
///
class ParaParamBool : public ParaParam {

   const bool defaultValue;        ///< default bool parameter value
   bool       currentValue;        ///< current bool parameter value

public:

   ///
   /// constructor
   ///
   ParaParamBool(
         const char *name,             ///< parameter name
         const char *inComment,        ///< comments string of this parameter
         bool value                    ///< default bool value of this parameter
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamBool(
         )
   {
   }

   ///
   /// get parameter type value
   /// @return 1: Bool
   ///
   int getType(
         ) const
   {
      return ParaParamTypeBool;
   }

   ///
   /// get default parameter value
   /// @return default parameter value
   ///
   bool getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current parameter value
   /// @return current parameter value
   ///
   bool getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default parameter value
   ///
   void setDefaultValue(
         )
   {
      currentValue = defaultValue;
   }

   ///
   /// set parameter value
   ///
   void setValue(
         bool value          ///< value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         )  const
   {
      return defaultValue == currentValue;
   }

};

///
/// class ParaParamInt
///
class ParaParamInt : public ParaParam {

   const int defaultValue;              ///< default int parameter value
   int       currentValue;              ///< current int parameter value
   const int minValue;                  ///< minimum int parameter value
   const int maxValue;                  ///< maximum int parameter value

public:

   ///
   /// contractor
   ///
   ParaParamInt(
         const char *name,              ///< int parameter name
         const char *inComment,         ///< comment string of this int parameter
         int value,                     ///< default value of this int parameter
         const int min,                 ///< minimum value of this int parameter
         const int max                  ///< maximum value of this int parameter
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value),
           minValue(min),
           maxValue(max)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamInt(
         )
   {
   }

   ///
   /// get parameter type
   /// @return 2: Int
   ///
   int getType(
         ) const
   {
      return ParaParamTypeInt;
   }

   ///
   /// get default value of this int parameter
   /// @return default value
   ///
   int getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current value of this int parameter
   /// @return current value
   ///
   int getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default value
   ///
   void setDefaultValue(
         )
   {
      currentValue = defaultValue;
   }

   ///
   /// set current value
   ///
   void setValue(
         int value                ///< int value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         ) const
   {
      return defaultValue == currentValue;
   }

   ///
   /// get minimum value of this int parameter
   /// @return minimum value
   ///
   int getMinValue(
         ) const
   {
      return minValue;
   }

   ///
   /// get maximum value of this int parameter
   /// @return maximum value
   ///
   int getMaxValue(
         ) const
   {
      return maxValue;
   }

};

///
/// class ParaParamLongint
///
class ParaParamLongint : public ParaParam {

   const long long defaultValue;            ///< default long int parameter value
   long long       currentValue;            ///< current long int parameter value
   const long long minValue;                ///< minimum long int parameter value
   const long long maxValue;                ///< maximum long int parameter value

public:

   ///
   /// constructor
   ///
   ParaParamLongint(
         const char *name,                  ///< long int parameter name
         const char *inComment,             ///< comment string of this long int parameter
         long long value,                   ///< default value of this long int parameter
         const long long min,               ///< minimum value of this long int parameter
         const long long max                ///< maximum value of this long int parameter
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value),
           minValue(min),
           maxValue(max)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamLongint(
         )
   {
   }

   ///
   /// get parameter type
   /// @return 3: Long int
   ///
   int getType(
         ) const
   {
      return ParaParamTypeLongint;
   }

   ///
   /// get default value of this long int parameter
   /// @return default value
   ///
   long long getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current value of this long int parameter
   /// @return current value
   ///
   long long getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default value of this long int parameter
   ///
   void setDefaultValue(
         )
   {
      currentValue = defaultValue;
   }

   ///
   /// set current value of this long int parameter
   ///
   void setValue(
         long long value            ///< value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         ) const
   {
      return defaultValue == currentValue;
   }

   ///
   /// get minimum value of this long int parameter
   /// @return minimum value
   ///
   long long getMinValue(
         ) const
   {
      return minValue;
   }

   ///
   /// get maximum value of this long
   ///
   long long getMaxValue(
         ) const
   {
      return maxValue;
   }

};

///
/// class ParaParamReal
///
class ParaParamReal : public ParaParam {

   const double defaultValue;                    ///< default real parameter value
   double       currentValue;                    ///< current real parameter value
   const double minValue;                        ///< minimum real parameter value
   const double maxValue;                        ///< maximum real parameter value

public:

   ///
   /// constructor
   ///
   ParaParamReal(
         const char *name,                       ///< real parameter name
         const char *inComment,                  ///< comment string of this real parameter
         double value,                           ///< default value of this real parameter
         const double min,                       ///< minimum value of this real parameter
         const double max                        ///< maximum value of this real parameter
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value),
           minValue(min),
           maxValue(max)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamReal(
         )
   {
   }

   ///
   /// get parameter type
   /// @return 4: real
   ///
   int getType(
         ) const
   {
      return ParaParamTypeReal;
   }

   ///
   /// get default value of this real parameter
   /// @return default value
   ///
   double getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current value of this real parameter
   /// @return current value
   ///
   double getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default value of this real parameter
   ///
   void setDefaultValue(
         )
   {
      currentValue = defaultValue;
   }

   ///
   /// set current value of this real parameter
   ///
   void setValue(
         double value         ///< value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         ) const
   {
      return ( fabs( defaultValue - currentValue ) < 1e-20 );
   }

   ///
   /// get minimum value of this long int parameter
   /// @return minimum value
   ///
   double getMinValue(
         ) const
   {
      return minValue;
   }

   ///
   /// get maximum value of this long
   ///
   double getMaxValue(
         ) const
   {
      return maxValue;
   }

};

///
/// class ParaParamChar
///
class ParaParamChar : public ParaParam {

   const char defaultValue;                         ///< default char parameter value
   char       currentValue;                         ///< current char parameter value
   const char *allowedValues;                       ///< allowed char parameter values

public:

   ///
   /// constructor
   ///
   ParaParamChar(
         const char *name,                          ///< char parameter name
         const char *inComment,                     ///< comment string of this char parameter
         char value,                                ///< default value of this char parameter
         const char *inAllowedValues                ///< allowed char parameter values
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value),
           allowedValues(inAllowedValues)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamChar(
         )
   {
   }

   ///
   /// get parameter type
   /// @return 5: char
   ///
   int getType(
         ) const
   {
      return ParaParamTypeChar;
   }

   ///
   /// get default value of this char parameter
   /// @return default value
   ///
   char getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current value of this char parameter
   /// @return current value
   ///
   char getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default value of this char parameter
   ///
   void setDefaultValue(
         )
   {
      currentValue = defaultValue;
   }

   ///
   /// set current value of this char parameter
   ///
   void setValue(
         char value                ///< value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         ) const
   {
      return defaultValue == currentValue;
   }

   ///
   /// get all allowed char parameter values
   /// @return sting of allowed chars
   ///
   const char *getAllowedValues(
         ) const
   {
      return allowedValues;
   }

};

///
/// class ParaParamString
///
class ParaParamString : public ParaParam {

   const char *defaultValue;                      ///< default string parameter value
   const char *currentValue;                      ///< current string parameter value

public:

   ///
   /// constructor
   ///
   ParaParamString(
         const char *name,                        ///< string parameter name
         const char *inComment,                   ///< comment string of this string parameter
         const char *value                        ///< default value of this string parameter
         )
         : ParaParam(name, inComment),
           defaultValue(value),
           currentValue(value)
   {
   }

   ///
   /// destructor
   ///
   ~ParaParamString(
         )
   {
      if( currentValue != defaultValue ) delete [] currentValue;
   }

   ///
   /// get parameter type
   /// @return 6: string
   ///
   int getType(
         ) const
   {
      return ParaParamTypeString;
   }

   ///
   /// get default value of this string parameter
   /// @return default value
   ///
   const char *getDefaultValue(
         ) const
   {
      return defaultValue;
   }

   ///
   /// get current value of this string parameter
   /// @return current value
   ///
   const char *getValue(
         ) const
   {
      return currentValue;
   }

   ///
   /// set default value of this string parameter
   ///
   void setDefaultValue(
         )
   {
      if( currentValue != defaultValue ) delete [] currentValue;
      currentValue = defaultValue;
   }

   ///
   /// set current value of this sting parameter
   ///
   void setValue(
         const char *value              ///< value to be set
         )
   {
      currentValue = value;
   }

   ///
   /// check if current value is default value or not
   /// @return true if current value is default value
   ///
   bool isDefaultValue(
         ) const
   {
      return ( std::string(defaultValue) == std::string(currentValue) );
   }

};

class ParaComm;
///
/// class ParaParamSet
///
class ParaParamSet {

protected:

   // static ParaParam *paraParams[ParaParamsSize];                     ///< array of ParaParms
   size_t    nParaParams;                                             ///< number of ParaParams
   ParaParam **paraParams;                                            ///< array of ParaParams

   ///
   /// parse bool parameter
   /// @return 0, if the bool parameter value is valid, -1: error
   ///
   int paramParaseBool(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse int parameter
   /// @return 0, if the int parameter value is valid, -1: error
   ///
   int paramParaseInt(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse long int parameter
   /// @return 0, if the long int parameter value is valid, -1: error
   ///
   int paramParaseLongint(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse real parameter
   /// @return 0, if the real parameter value is valid, -1: error
   ///
   int paramParaseReal(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse real parameter
   /// @return 0, if the real parameter value is valid, -1: error
   ///
   int paramParaseChar(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse real parameter
   /// @return 0, if the real parameter value is valid, -1: error
   ///
   int paramParaseString(
         ParaParam *paraParam,        ///< pointer to ParaParam object
         char *valuestr               ///< value string
         );

   ///
   /// parse parameter
   /// (this routine is almost copy from paramset.c of SCIP code)
   /// @return 0, if parameter in a line is valid, -1: error
   ///
   int parameterParse(
         char *line,                                ///< parameter line
         std::map<std::string, int> &mapStringToId  ///< map of parameter sting to parameter id
         );

public:

   ///
   /// constructor
   ///
   ParaParamSet(
         )
         : nParaParams(0), paraParams(0)
   {
   }

   ///
   /// constructor
   ///
   ParaParamSet(
         size_t nInParaParams
         );

   ///
   /// destructor
   ///
   virtual ~ParaParamSet(
         )
   {
      for( size_t i = 0; i < nParaParams; i++ )
      {
         delete paraParams[i];
      }
      delete [] paraParams;
   }

   ///--------------------
   /// for bool parameters
   ///--------------------

   ///
   /// get bool parameter value
   /// @return value of the bool parameter specified
   ///
   bool getBoolParamValue(
         int param        ///< bool parameter id
         );

   ///
   /// set bool parameter value
   ///
   void setBoolParamValue(
         int param,       ///< bool parameter id
         bool value       ///< value to be set
         );

   ///
   /// get default value of bool parameter
   /// @return default bool parameter value
   ///
   bool getBoolParamDefaultValue(
         int param        ///< bool parameter id
         );

   ///
   /// set bool parameter default value
   ///
   void setBoolParamDefaultValue(
         int param        ///< bool parameter id
         );

   ///
   /// check if bool parameter is default value or not
   /// @return true if bool parameter is default value
   ///
   bool isBoolParamDefaultValue(
         int param        ///< bool parameter id
         );


   ///--------------------
   /// for int parameters
   ///--------------------

   ///
   /// get int parameter value
   /// @return value of the int parameter specified
   ///
   int getIntParamValue(
         int param        ///< int parameter id
         );

   ///
   /// set int parameter value
   ///
   void setIntParamValue(
         int param,       ///< int parameter id
         int value        ///< value to be set
         );

   ///
   /// get default value of int parameter
   /// @return default int parameter value
   ///
   int getIntParamDefaultValue(
         int param        ///< int parameter id
         );

   ///
   /// set int parameter default value
   ///
   void setIntParamDefaultValue(
         int param        ///< int parameter id
         );

   ///
   /// check if int parameter is default value or not
   /// @return true if int parameter is default value
   ///
   bool isIntParamDefaultValue(
         int param        ///< int parameter id
         );

   ///------------------------
   /// for long int parameters
   ///------------------------

   ///
   /// get long int parameter value
   /// @return value of the long int parameter specified
   ///
   long long getLongintParamValue(
         int param        ///< long int parameter id
         );

   ///
   /// set long int parameter value
   ///
   void setLongintParamValue(
         int param,       ///< long int parameter id
         long long value  ///< value to be set
         );

   ///
   /// get default value of long int parameter
   /// @return default long int parameter value
   ///
   long long getLongintParamDefaultValue(
         int param        ///< long int parameter id
         );

   ///
   /// set long int parameter default value
   ///
   void setLongintParamDefaultValue(
         int param        ///< long int parameter id
         );

   ///
   /// check if long int parameter is default value or not
   /// @return true if long int parameter is default value
   ///
   bool isLongintParamDefaultValue(
         int param        ///< long int parameter id
         );

   ///------------------------
   /// for real parameters
   ///------------------------

   ///
   /// get real parameter value
   /// @return value of the real parameter specified
   ///
   double getRealParamValue(
         int param        ///< real parameter id
         );

   ///
   /// set real parameter value
   ///
   void setRealParamValue(
         int param,       ///< real parameter id
         double value     ///< value to be set
         );

   ///
   /// get default value of real parameter
   /// @return default real parameter value
   ///
   double getRealParamDefaultValue(
         int param        ///< real parameter id
         );

   ///
   /// set real parameter default value
   ///
   void setRealParamDefaultValue(
         int param        ///< real parameter id
         );

   ///
   /// check if real parameter is default value or not
   /// @return true if real parameter is default value
   ///
   bool isRealParamDefaultValue(
         int param        ///< real parameter id
         );

   ///--------------------
   /// for char parameters
   ///--------------------

   ///
   /// get char parameter value
   /// @return value of the char parameter specified
   ///
   char getCharParamValue(
         int param        ///< char parameter id
         );

   ///
   /// set char parameter value
   ///
   void setCharParamValue(
         int param,       ///< char parameter id
         char value       ///< value to be set
         );

   ///
   /// get default value of char parameter
   /// @return default char parameter value
   ///
   char getCharParamDefaultValue(
         int param        ///< char parameter id
         );

   ///
   /// set char parameter default value
   ///
   void setCharParamDefaultValue(
         int param        ///< char parameter id
         );

   ///
   /// check if char parameter is default value or not
   /// @return true if char parameter is default value
   ///
   bool isCharParamDefaultValue(
         int param        ///< char parameter id
         );

   ///--------------------
   /// for char parameters
   ///--------------------

   ///
   /// get string parameter value
   /// @return value of the string parameter specified
   ///
   const char *getStringParamValue(
         int param          ///< string parameter id
         );

   ///
   /// set string parameter value
   ///
   void setStringParamValue(
         int param,         ///< string parameter id
         const char *value  ///< value to be set
         );

   ///
   /// get default value of string parameter
   /// @return default string parameter value
   ///
   const char *getStringParamDefaultValue(
         int param          ///< string parameter id
         );

   ///
   /// set string parameter default value
   ///
   void setStringParamDefaultValue(
         int param          ///< string parameter id
         );

   ///
   /// check if string parameter is default value or not
   /// @return true if string parameter is default value
   ///
   bool isStringParamDefaultValue(
         int param          ///< string parameter id
         );


   ///
   /// read ParaParams from file
   ///
   virtual void read(
         ParaComm *comm,       ///< communicator used
         const char* filename  ///< reading file name
         );

   ///
   /// write ParaParams to output stream
   ///
   void write(
         std::ostream *os      ///< ostream for writing
         );

   ///
   /// get parameter table size
   /// @return size of parameter table
   ///
   size_t getParaParamsSize(
         )
   {
      return nParaParams;
   }

   ///
   /// get number of bool parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumBoolParams(
         ) = 0;

   ///
   /// get number of int parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumIntParams(
         ) = 0;

   ///
   /// get number of longint parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumLongintParams(
         ) = 0;

   ///
   /// get number of real parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumRealParams(
         ) = 0;

   ///
   /// get number of char parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumCharParams(
         ) = 0;

   ///
   /// get number of string parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumStringParams(
         ) = 0;

   ///
   /// broadcast ParaParams
   /// @return always 0 (for future extensions)
   ///
   virtual int bcast(
         ParaComm *comm,       ///< communicator used
         int root              ///< root rank for broadcast
         ) = 0;

};

}

#endif  // __PARA_PARAM_SET_H__
