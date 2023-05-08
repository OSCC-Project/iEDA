/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

#ifndef YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED
# define YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED
/* Debug traces.  */
#ifndef VERILOG_DEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define VERILOG_DEBUG 1
#  else
#   define VERILOG_DEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define VERILOG_DEBUG 1
# endif /* ! defined YYDEBUG */
#endif  /* ! defined VERILOG_DEBUG */
#if VERILOG_DEBUG
extern int verilog_debug;
#endif
/* "%code requires" blocks.  */
#line 1 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.y"


#include <vector>

#include "log/Log.hh"
#include "string/Str.hh"
#include "VerilogReader.hh"

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;


#line 72 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.hh"

/* Token type.  */
#ifndef VERILOG_TOKENTYPE
# define VERILOG_TOKENTYPE
  enum verilog_tokentype
  {
    INT = 258,
    CONSTANT = 259,
    ID = 260,
    STRING = 261,
    MODULE = 262,
    ENDMODULE = 263,
    ASSIGN = 264,
    PARAMETER = 265,
    DEFPARAM = 266,
    WIRE = 267,
    WAND = 268,
    WOR = 269,
    TRI = 270,
    INPUT = 271,
    OUTPUT = 272,
    INOUT = 273,
    SUPPLY1 = 274,
    SUPPLY0 = 275,
    REG = 276,
    NEG = 277
  };
#endif

/* Value type.  */
#if ! defined VERILOG_STYPE && ! defined VERILOG_STYPE_IS_DECLARED
union VERILOG_STYPE
{
#line 25 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.y"

  int integer;
  char* string;
  const char* constant;
  void*  obj;

#line 113 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.hh"

};
typedef union VERILOG_STYPE VERILOG_STYPE;
# define VERILOG_STYPE_IS_TRIVIAL 1
# define VERILOG_STYPE_IS_DECLARED 1
#endif



int verilog_parse (yyscan_t yyscanner, ista::VerilogReader* verilog_reader);
/* "%code provides" blocks.  */
#line 17 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.y"

#undef  YY_DECL
#define YY_DECL int verilog_lex(VERILOG_STYPE *yylval_param, yyscan_t yyscanner, ista::VerilogReader *verilog_reader)
YY_DECL;

void yyerror(yyscan_t scanner,ista::VerilogReader *verilog_reader, const char *str);

#line 133 "/home/taosimin/irefactor/src/database/manager/parser/verilog/VerilogParse.hh"

#endif /* !YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED  */
