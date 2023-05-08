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

#ifndef YY_SDF_PROJECT_HOME_CHENSHIJIAN_IEDA_SRC_ISTA_SDF_PARSER_SDFPARSE_HH_INCLUDED
# define YY_SDF_PROJECT_HOME_CHENSHIJIAN_IEDA_SRC_ISTA_SDF_PARSER_SDFPARSE_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int Sdf_debug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    DELAYFILE = 258,
    SDFVERSION = 259,
    DESIGN = 260,
    DATE = 261,
    VENDOR = 262,
    PROGRAM = 263,
    PVERSION = 264,
    DIVIDER = 265,
    VOLTAGE = 266,
    PROCESS = 267,
    TEMPERATURE = 268,
    TIMESCALE = 269,
    CELL = 270,
    CELLTYPE = 271,
    INSTANCE = 272,
    DELAY = 273,
    ABSOLUTE = 274,
    INCREMENTAL = 275,
    INTERCONNECT = 276,
    PORT = 277,
    DEVICE = 278,
    RETAIN = 279,
    IOPATH = 280,
    TIMINGCHECK = 281,
    SETUP = 282,
    HOLD = 283,
    SETUPHOLD = 284,
    RECOVERY = 285,
    REMOVAL = 286,
    RECREM = 287,
    WIDTH = 288,
    PERIOD = 289,
    SKEW = 290,
    NOCHANGE = 291,
    POSEDGE = 292,
    NEGEDGE = 293,
    COND = 294,
    CONDELSE = 295,
    QSTRING = 296,
    ID = 297,
    PATH = 298,
    NUMBER = 299,
    EXPR_OPEN_IOPATH = 300,
    EXPR_OPEN = 301,
    EXPR_ID_CLOSE = 302
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 42 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"

  char character;
  char *string;
  int integer;
  float number;
  void *obj;

#line 113 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.hh"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE Sdf_lval;

int Sdf_parse (void);

#endif /* !YY_SDF_PROJECT_HOME_CHENSHIJIAN_IEDA_SRC_ISTA_SDF_PARSER_SDFPARSE_HH_INCLUDED  */
