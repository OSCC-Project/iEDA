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

/**@file    paraParamSet.cpp
 * @brief   Parameter set for UG framework.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <string>
#include <map>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cfloat>
#include <climits>
#include <cassert>
#include "paraComm.h"
#include "paraParamSet.h"

using namespace UG;

ParaParamSet::ParaParamSet(
      size_t nInParaParams
      )
      : nParaParams(nInParaParams)
{
   paraParams = new ParaParam*[nParaParams];
   for(size_t i = 0; i <  nParaParams; i++ )
   {
      paraParams[i] = 0;
   }

  /** bool params */
   paraParams[Quiet] = new ParaParamBool(
         "Quiet",
         "# Set log output to minimal information [Default value: TRUE]",
         true);
   paraParams[TagTrace] = new ParaParamBool(
         "TagTrace",
         "# Control log output of communication tags [Default value: FALSE]",
         false);
   paraParams[LogSolvingStatus] = new ParaParamBool(
         "LogSolvingStatus",
         "# Control of solving status log [Default value: FALSE]",
         false);
   paraParams[LogTasksTransfer] = new ParaParamBool(
         "LogTasksTransfer",
         "# Control output of tasks transfer log [Default value: FALSE]",
         false);
   paraParams[Checkpoint] = new ParaParamBool(
         "Checkpoint",
         "# Control checkpointing functionality [Default value: FALSE]",
         false);
   paraParams[Deterministic] = new ParaParamBool(
         "Deterministic",
         "# Control deterministic mode [Default value: FALSE]",
         false);
   paraParams[StatisticsToStdout] = new ParaParamBool(
         "StatisticsToStdout",
         "# Control output of statistics to stdout [Default value: FALSE]",
         false);
   paraParams[DynamicAdjustNotificationInterval] = new ParaParamBool(
          "DynamicAdjustNotificationInterval",
          "# Control dynamic adjustment of notification interval time [Default value: FALSE]",
          false);

   /** int params */
   paraParams[OutputParaParams] = new ParaParamInt(
         "OutputParaParams",
         "# Control output of ParaParams: 0 - no output, 1 - output only non-default values, 2 - output non-default values with comments, 3 - output all values, 4 - output all values with comments [Default value: 4]",
         4,
         0,
         4);
   paraParams[NotificationSynchronization] = new ParaParamInt(
         "NotificationSynchronization",
         "# Set notification synchronization strategy: 0 - always synchronize, 1 - collect in every iteration, 2 - no synchronization [Default value: 0]",
         0,
         0,
         2);

   /** longint params */

   /** real params */
   paraParams[NotificationInterval] = new ParaParamReal(
         "NotificationInterval",
         "# Set interval between notifications from active solver of its solvers status to LoadCoordinator. [Default: 1.0][0.0, DBL_MAX]",
         1.0,
         0.0,
         DBL_MAX);
   paraParams[TimeLimit] = new ParaParamReal(
         "TimeLimit",
         "# Time limit for computation. -1.0 means no time limit. [Default: -1.0] [-1.0, DBL_MAX]",
         -1.0,
         -1.0,
         DBL_MAX);
   paraParams[CheckpointInterval] = new ParaParamReal(
         "CheckpointInterval",
         "# Time interval between checkpoints. [Default: 3600.0] [5.0, DBL_MAX]",
         3600.0,
         5.0,
         DBL_MAX);
   paraParams[FinalCheckpointGeneratingTime] = new ParaParamReal(
         "FinalCheckpointGeneratingTime",
         "# Time until start of generation of final checkpointing files. When this parameter is specified, TimeLimit is ignored. -1.0 means no specification. [Default: -1.0] [-1.0, DBL_MAX]",
         -1.0,
         -1.0,
         DBL_MAX);

   /** char params */

   /** string params */
   paraParams[TempFilePath] = new ParaParamString(
         "TempFilePath",
         "# Path for temporary files [Default: /tmp/]",
         "/tmp/");
   paraParams[TagTraceFileName] = new ParaParamString(
         "TagTraceFileName",
         "# Filename for tag trace log [Default: std::cout]",
         "std::cout");
   paraParams[LogSolvingStatusFilePath] = new ParaParamString(
         "LogSolvingStatusFilePath",
         "# Path to solving statuses log [Default: ./]",
         "./");
   paraParams[LogTasksTransferFilePath] = new ParaParamString(
         "LogTasksTransferFilePath",
         "# Path to tasks transfer log [Default: ./]",
         "./");
   paraParams[SolutionFilePath] = new ParaParamString(
         "SolutionFilePath",
         "# Path to output solution [Default: ./]",
         "./");
   paraParams[CheckpointFilePath] = new ParaParamString(
         "CheckpointFilePath",
         "# Path to checkpoint files [Default: ./]",
         "./");
   paraParams[RacingParamsDirPath] = new ParaParamString(
         "RacingParamsDirPath",
         "# Path racing parameter configuration files. \"\" means to use default racing set. [Default: \"\"]",
         "");
}

/*********************
 * for bool params  *
 ********************/
bool
ParaParamSet::getBoolParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeBool);
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool *>(paraParams[param]);
   return paraParamBool->getValue();
}

void
ParaParamSet::setBoolParamValue(
      int param,
      bool value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeBool);
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool *>(paraParams[param]);
   return paraParamBool->setValue(value);
}

bool
ParaParamSet::getBoolParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeBool);
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool *>(paraParams[param]);
   return paraParamBool->getDefaultValue();
}

void
ParaParamSet::setBoolParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeBool);
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool *>(paraParams[param]);
   return paraParamBool->setDefaultValue();
}

bool
ParaParamSet::isBoolParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeBool);
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool *>(paraParams[param]);
   return paraParamBool->isDefaultValue();
}

/********************
 * for int params   *
 *******************/
int
ParaParamSet::getIntParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeInt);
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt *>(paraParams[param]);
   return paraParamInt->getValue();
}

void
ParaParamSet::setIntParamValue(
      int param,
      int value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeInt);
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt *>(paraParams[param]);
   return paraParamInt->setValue(value);
}

int
ParaParamSet::getIntParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeInt);
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt *>(paraParams[param]);
   return paraParamInt->getDefaultValue();
}

void
ParaParamSet::setIntParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeInt);
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt *>(paraParams[param]);
   paraParamInt->setDefaultValue();
}

bool
ParaParamSet::isIntParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeInt);
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt *>(paraParams[param]);
   return paraParamInt->isDefaultValue();
}

/*********************
 * for longint params  *
 ********************/
long long
ParaParamSet::getLongintParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeLongint);
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint *>(paraParams[param]);
   return paraParamLongint->getValue();
}

void
ParaParamSet::setLongintParamValue(
      int param,
      long long value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeLongint);
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint *>(paraParams[param]);
   paraParamLongint->setValue(value);
}

long long
ParaParamSet::getLongintParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeLongint);
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint *>(paraParams[param]);
   return paraParamLongint->getDefaultValue();
}

void
ParaParamSet::setLongintParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeLongint);
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint *>(paraParams[param]);
   paraParamLongint->setDefaultValue();
}

bool
ParaParamSet::isLongintParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeLongint);
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint *>(paraParams[param]);
   return paraParamLongint->isDefaultValue();
}

/*********************
 * for real params  *
 ********************/
double
ParaParamSet::getRealParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeReal);
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal *>(paraParams[param]);
   return paraParamReal->getValue();
}

void
ParaParamSet::setRealParamValue(
      int param,
      double value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeReal);
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal *>(paraParams[param]);
   paraParamReal->setValue(value);
}

double
ParaParamSet::getRealParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeReal);
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal *>(paraParams[param]);
   return paraParamReal->getDefaultValue();
}

void
ParaParamSet::setRealParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeReal);
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal *>(paraParams[param]);
   paraParamReal->setDefaultValue();
}

bool
ParaParamSet::isRealParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeReal);
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal *>(paraParams[param]);
   return paraParamReal->isDefaultValue();
}

/*********************
 * for char params  *
 ********************/
char
ParaParamSet::getCharParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeChar);
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar *>(paraParams[param]);
   return paraParamChar->getValue();
}

void
ParaParamSet::setCharParamValue(
      int param,
      char value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeChar);
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar *>(paraParams[param]);
   paraParamChar->setValue(value);
}

char
ParaParamSet::getCharParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeChar);
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar *>(paraParams[param]);
   return paraParamChar->getDefaultValue();
}

void
ParaParamSet::setCharParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeChar);
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar *>(paraParams[param]);
   paraParamChar->setDefaultValue();
}

bool
ParaParamSet::isCharParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeChar);
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar *>(paraParams[param]);
   return paraParamChar->isDefaultValue();
}

/**********************
 * for string params  *
 *********************/
const char *
ParaParamSet::getStringParamValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeString);
   ParaParamString *paraParamString = dynamic_cast< ParaParamString *>(paraParams[param]);
   return paraParamString->getValue();
}

void
ParaParamSet::setStringParamValue(
      int param,
      const char *value
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeString);
   ParaParamString *paraParamString = dynamic_cast< ParaParamString *>(paraParams[param]);
   char *str = new char[std::strlen(value) + 1];
   std::strcpy(str, value);
   paraParamString->setValue(value);
}

const char *
ParaParamSet::getStringParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeString);
   ParaParamString *paraParamString = dynamic_cast< ParaParamString *>(paraParams[param]);
   return paraParamString->getDefaultValue();
}

void
ParaParamSet::setStringParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeString);
   ParaParamString *paraParamString = dynamic_cast< ParaParamString *>(paraParams[param]);
   paraParamString->setDefaultValue();
}

bool
ParaParamSet::isStringParamDefaultValue(
      int param
      )
{
   assert(paraParams[param]->getType() == ParaParamTypeString);
   ParaParamString *paraParamString = dynamic_cast< ParaParamString *>(paraParams[param]);
   return paraParamString->isDefaultValue();
}

int
ParaParamSet::paramParaseBool(
      ParaParam *paraParam,
      char *valuestr
      )
{
   ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool * >(paraParam);
   assert(valuestr != NULL);
   if( std::string(valuestr) == std::string("TRUE") )
   {
      paraParamBool->setValue(true);
   }
   else if( std::string(valuestr) == std::string("FALSE") )
   {
      paraParamBool->setValue(false);
   }
   else
   {
      std::cout << "Invalid parameter value <" << valuestr
                <<  "> for ParaParam_Bool parameter <"
                << paraParamBool->getParamName() << ">" << std::endl;
      return -1;
   }
   return 0;
}

int
ParaParamSet::paramParaseInt(
      ParaParam *paraParam,
      char *valuestr
      )
{
   int value;
   ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt * >(paraParam);
   assert(valuestr != NULL);
   if( std::sscanf(valuestr, "%d", &value) == 1 )
   {
      if( value >= paraParamInt->getMinValue() && value <= paraParamInt->getMaxValue() )
      {
         paraParamInt->setValue(value);
      }
      else
      {
         OUTPUT_PARAM_VALUE_ERROR("Int", paraParamInt->getParamName(), value, paraParamInt->getComment() );
      }
   }
   else
   {
      std::cout << "Invalid parameter value <" << valuestr
                << "> for int parameter <"
                << paraParamInt->getParamName() << ">" << std::endl;
      return -1;
   }
   return 0;
}

int
ParaParamSet::paramParaseLongint(
      ParaParam *paraParam,
      char *valuestr
      )
{
   long long value;
   ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint * >(paraParam);
   assert(valuestr != NULL);
   if( std::sscanf(valuestr, "%lld", &value) == 1 )
   {
      if( value >= paraParamLongint->getMinValue() && value <= paraParamLongint->getMaxValue() )
      {
         paraParamLongint->setValue(value);
      }
      else
      {
         OUTPUT_PARAM_VALUE_ERROR("Longint", paraParamLongint->getParamName(), value, paraParamLongint->getComment() );
      }
   }
   else
   {
      std::cout << "Invalid parameter value <" << valuestr
                << "> for longint parameter <"
                << paraParamLongint->getParamName() << ">" << std::endl;
      return -1;
   }
   return 0;
}

int
ParaParamSet::paramParaseReal(
      ParaParam *paraParam,
      char *valuestr
      )
{
   double value;
   ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal * >(paraParam);
   assert(valuestr != NULL);
   if( std::sscanf(valuestr, "%lf", &value) == 1 )
   {
      if( value >= paraParamReal->getMinValue() && value <= paraParamReal->getMaxValue() )
      {
         paraParamReal->setValue(value);
      }
      else
      {
         OUTPUT_PARAM_VALUE_ERROR("Real", paraParamReal->getParamName(), value, paraParamReal->getComment() );
      }
   }
   else
   {
      std::cout << "Invalid parameter value <" << valuestr
                << "> for real parameter <"
                << paraParamReal->getParamName() << ">" << std::endl;
      return -1;
   }
   return 0;
}

int
ParaParamSet::paramParaseChar(
      ParaParam *paraParam,
      char *valuestr
      )
{
   char value;
   ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar * >(paraParam);
   assert(valuestr != NULL);
   if( std::sscanf(valuestr, "%c", &value) == 1 )
   {
      std::string allowedString(paraParamChar->getAllowedValues());
      char cString[2]; cString[0] = value; cString[1] = '\0';
      if( allowedString.find(cString) != std::string::npos )
      {
         paraParamChar->setValue(value);
      }
      else
      {
         OUTPUT_PARAM_VALUE_ERROR("Char", paraParamChar->getParamName(), value, paraParamChar->getComment() );
      }
   }
   else
   {
      std::cout << "Invalid parameter value <" << valuestr
                << "> for char parameter <"
                << paraParamChar->getParamName() << ">" << std::endl;
      return -1;
   }
   return 0;
}

int
ParaParamSet::paramParaseString(
      ParaParam *paraParam,
      char *valuestr
      )
{
   ParaParamString *paraParamString = dynamic_cast< ParaParamString * >(paraParam);
   assert(valuestr != NULL);

   /* check for quotes */
   size_t len = std::strlen(valuestr);
   if( len <= 1 || valuestr[0] != '"' || valuestr[len-1] != '"' )
   {
      std::cout << "Invalid parameter value <" << valuestr
                << "> for string parameter <"
                << paraParamString->getParamName() << ">" << std::endl;
      return -1;
   }
   /* remove the quotes */
   valuestr[len-1] = '\0';
   valuestr++;
   char *paramValue = new char[strlen(valuestr) + 1 ];
   strcpy(paramValue, valuestr);
   paraParamString->setValue(paramValue);
   return 0;
}

// -1: error
/** the parameterParse routine is almost copy from paramset.c of SCIP code */
int
ParaParamSet::parameterParse(
      char *line,
      std::map<std::string, int> &mapStringToId
      )
{
   char* paramname;
   char* paramvaluestr;
   char* lastquote;
   unsigned int quoted;

   /* find the start of the parameter name */
   while( *line == ' ' || *line == '\t' || *line == '\r' )
      line++;
   if( *line == '\0' || *line == '\n' || *line == '#' )
      return 0;
   paramname = line;

   /* find the end of the parameter name */
   while( *line != ' ' && *line != '\t' && *line != '\r' && *line != '\n' && *line != '#' && *line != '\0' && *line != '=' )
      line++;
   if( *line == '=' )
   {
      *line = '\0';
      line++;
   }
   else
   {
      *line = '\0';
      line++;

      /* search for the '=' char in the line */
      while( *line == ' ' || *line == '\t' || *line == '\r' )
         line++;
      if( *line != '=' )
      {
         std::cout << "Character '=' was expected after the parameter name" << std::endl;
         return -1;
      }
      line++;
   }

   /* find the start of the parameter value string */
   while( *line == ' ' || *line == '\t' || *line == '\r' )
      line++;
   if( *line == '\0' || *line == '\n' || *line == '#' )
   {
      std::cout << "Parameter value is missing" << std::endl;
      return -1;
   }
   paramvaluestr = line;

   /* find the end of the parameter value string */
   quoted = (*paramvaluestr == '"');
   lastquote = NULL;
   while( (quoted || (*line != ' ' && *line != '\t' && *line != '\r' && *line != '\n' && *line != '#')) && *line != '\0' )
   {
      if( *line == '"' )
         lastquote = line;
      line++;
   }
   if( lastquote != NULL )
      line = lastquote+1;
   if( *line == '#' )
      *line = '\0';
   else if( *line != '\0' )
   {
      /* check, if the rest of the line is clean */
      *line = '\0';
      line++;
      while( *line == ' ' || *line == '\t' || *line == '\r' )
         line++;
      if( *line != '\0' && *line != '\n' && *line != '#' )
      {
         std::cout << "Additional characters after parameter value" << std::endl;
         return -1;
      }
   }

   std::map<std::string, int>::iterator pos;
   pos = mapStringToId.find(paramname);
   if( pos == mapStringToId.end() )
   {
      std::cout << "Unknown parameter <" << paramname << ">" << std::endl;
      return -1;
   }
   int paramId = pos->second;
   switch ( paraParams[paramId]->getType() )
   {
   case ParaParamTypeBool:
      return paramParaseBool(paraParams[paramId], paramvaluestr);
   case ParaParamTypeInt:
      return paramParaseInt(paraParams[paramId], paramvaluestr);
   case ParaParamTypeLongint:
      return paramParaseLongint(paraParams[paramId], paramvaluestr);
   case ParaParamTypeReal:
      return paramParaseReal(paraParams[paramId], paramvaluestr);
   case ParaParamTypeChar:
      return paramParaseChar(paraParams[paramId], paramvaluestr);
   case ParaParamTypeString:
      return paramParaseString(paraParams[paramId], paramvaluestr);
   default:
      std::cout << "Unknown parameter type" << std::endl;
      return -1;
   }
}

void
ParaParamSet::read(
      ParaComm *comm,
      const char* filename
      )
{
   const int MaxLineSize = 1024;
   char line[MaxLineSize];

   std::ifstream ifs;
   ifs.open(filename);
   if( !ifs ){
      std::cout << "Cannot open ParaParams read file: file name = " << filename << std::endl;
      exit(1);
   }

   std::map<std::string, int> mapStringToId;
   // for( int i = ParaParamsFirst; i < ParaParamsSize; i ++ )
   for( size_t i = 0; i < getParaParamsSize(); i ++ )
   {
      assert( paraParams[i] );
      mapStringToId.insert(std::make_pair(paraParams[i]->getParamName(),i));
   }

   int lineNo = 0;
   while( ifs.getline(line, MaxLineSize) )
   {
      lineNo++;
      int retCode = parameterParse(line, mapStringToId);
      if( retCode ){
         ifs.close();
         std::cout << "Input error in file <" << filename << "> line " << lineNo << std::endl;
         exit(1);
      }
   }
   ifs.close();

#ifndef UG_WITH_ZLIB
   if( getBoolParamValue(Checkpoint) )
   {
      std::cout << "Cannot compile checkpointing without zlib. Checkpoint must be FALSE." << std::endl;
      exit(1);
   }
#endif
}

void
ParaParamSet::write(
      std::ostream *os
      )
{
   bool comments = false;
   bool onlyChanged = false;

   if( this->getIntParamValue(OutputParaParams) == 1 ||
         this->getIntParamValue(OutputParaParams) == 2
         )
   {
      onlyChanged = true;
   }

   if( this->getIntParamValue(OutputParaParams) == 2 ||
         this->getIntParamValue(OutputParaParams) == 4
         )
   {
      comments = true;
   }

   // for( int i = 0; i <  ParaParamsSize; i++ )
   for( size_t i = 0; i <  getParaParamsSize(); i++ )
   {
      assert( paraParams[i] );
      switch ( paraParams[i]->getType() )
      {
      case ParaParamTypeBool:
      {
         ParaParamBool *paraParamBool = dynamic_cast< ParaParamBool * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamBool->isDefaultValue()){
               if( comments )
               {
                  *os << paraParamBool->getComment() << std::endl;
               }
               *os << paraParamBool->getParamName() << " = " << ( ( paraParamBool->getValue() == true ) ? "TRUE" : "FALSE" ) << std::endl << std::endl;
            }
         } else {
            if( comments )
            {
               *os << paraParamBool->getComment() << std::endl;
            }
            *os << paraParamBool->getParamName() << " = " << ( ( paraParamBool->getValue() == true ) ? "TRUE" : "FALSE" ) << std::endl << std::endl;
         }
         break;
      }
      case ParaParamTypeInt:
      {
         ParaParamInt *paraParamInt = dynamic_cast< ParaParamInt * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamInt->isDefaultValue()){
               if( comments )
               {
                  *os << paraParamInt->getComment() << std::endl;
               }
               *os << paraParamInt->getParamName() << " = " << paraParamInt->getValue() << std::endl<< std::endl;
            }
         }
         else
         {
            if( comments )
            {
               *os << paraParamInt->getComment() << std::endl;
            }
            *os << paraParamInt->getParamName() << " = " << paraParamInt->getValue() << std::endl << std::endl;
         }
         break;
      }
      case ParaParamTypeLongint:
      {
         ParaParamLongint *paraParamLongint = dynamic_cast< ParaParamLongint * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamLongint->isDefaultValue()){
               if( comments )
               {
                  *os << paraParamLongint->getComment() << std::endl;
               }
               *os << paraParamLongint->getParamName() << " = " << paraParamLongint->getValue() << std::endl << std::endl;
            }
         }
         else
         {
            if( comments )
            {
               *os << paraParamLongint->getComment() << std::endl;
            }
            *os << paraParamLongint->getParamName() << " = " << paraParamLongint->getValue() << std::endl << std::endl;
         }
         break;
      }
      case ParaParamTypeReal:
      {
         ParaParamReal *paraParamReal = dynamic_cast< ParaParamReal * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamReal->isDefaultValue()){
               if( comments )
               {
                  *os << paraParamReal->getComment() << std::endl;
               }
               *os << paraParamReal->getParamName() << " = " << paraParamReal->getValue() << std::endl << std::endl;
            }
         }
         else
         {
            if( comments )
            {
               *os << paraParamReal->getComment() << std::endl;
            }
            *os << paraParamReal->getParamName() << " = " << paraParamReal->getValue() << std::endl << std::endl;
         }
         break;
      }
      case ParaParamTypeChar:
      {
         ParaParamChar *paraParamChar = dynamic_cast< ParaParamChar * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamChar->isDefaultValue()){
               if( comments )
               {
                  *os << paraParamChar->getComment() << std::endl;
               }
               *os << paraParamChar->getParamName() << " = " << paraParamChar->getValue() << std::endl << std::endl;
            }
         }
         else
         {
            if( comments )
            {
               *os << paraParamChar->getComment() << std::endl;
            }
            *os << paraParamChar->getParamName() << " = " << paraParamChar->getValue() << std::endl << std::endl;
         }
         break;
      }
      case ParaParamTypeString:
      {
         ParaParamString *paraParamString = dynamic_cast< ParaParamString * >(paraParams[i]);
         if( onlyChanged )
         {
            if (!paraParamString->isDefaultValue())
            {
               if( comments )
               {
                  *os << paraParamString->getComment() << std::endl;
               }
               *os << paraParamString->getParamName() << " = \"" << paraParamString->getValue() << "\"" << std::endl << std::endl;
            }
         }
         else
         {
            if( comments )
            {
               *os << paraParamString->getComment() << std::endl;
            }
            *os << paraParamString->getParamName() << " = \"" << paraParamString->getValue() << "\"" << std::endl << std::endl;
         }
         break;
      }
      default:
         std::cout << "Unknown parameter type" << std::endl;
         throw "Unknown parameter type";
      }
   }
   (*os).clear();
}
