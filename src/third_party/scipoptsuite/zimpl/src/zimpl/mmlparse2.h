/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

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

#ifndef YY_YY_SRC_ZIMPL_MMLPARSE2_H_INCLUDED
# define YY_YY_SRC_ZIMPL_MMLPARSE2_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    DECLSET = 258,
    DECLPAR = 259,
    DECLVAR = 260,
    DECLMIN = 261,
    DECLMAX = 262,
    DECLSUB = 263,
    DECLSOS = 264,
    DEFNUMB = 265,
    DEFSTRG = 266,
    DEFBOOL = 267,
    DEFSET = 268,
    PRINT = 269,
    CHECK = 270,
    BINARY = 271,
    INTEGER = 272,
    REAL = 273,
    IMPLICIT = 274,
    ASGN = 275,
    DO = 276,
    WITH = 277,
    IN = 278,
    TO = 279,
    UNTIL = 280,
    BY = 281,
    FORALL = 282,
    EXISTS = 283,
    PRIORITY = 284,
    STARTVAL = 285,
    DEFAULT = 286,
    CMP_LE = 287,
    CMP_GE = 288,
    CMP_EQ = 289,
    CMP_LT = 290,
    CMP_GT = 291,
    CMP_NE = 292,
    INFTY = 293,
    AND = 294,
    OR = 295,
    XOR = 296,
    NOT = 297,
    SUM = 298,
    MIN = 299,
    MAX = 300,
    ARGMIN = 301,
    ARGMAX = 302,
    PROD = 303,
    IF = 304,
    THEN = 305,
    ELSE = 306,
    END = 307,
    INTER = 308,
    UNION = 309,
    CROSS = 310,
    SYMDIFF = 311,
    WITHOUT = 312,
    PROJ = 313,
    MOD = 314,
    DIV = 315,
    POW = 316,
    FAC = 317,
    CARD = 318,
    ROUND = 319,
    FLOOR = 320,
    CEIL = 321,
    RANDOM = 322,
    ORD = 323,
    ABS = 324,
    SGN = 325,
    LOG = 326,
    LN = 327,
    EXP = 328,
    SQRT = 329,
    SIN = 330,
    COS = 331,
    TAN = 332,
    ASIN = 333,
    ACOS = 334,
    ATAN = 335,
    POWER = 336,
    SGNPOW = 337,
    READ = 338,
    AS = 339,
    SKIP = 340,
    USE = 341,
    COMMENT = 342,
    MATCH = 343,
    SUBSETS = 344,
    INDEXSET = 345,
    POWERSET = 346,
    VIF = 347,
    VABS = 348,
    TYPE1 = 349,
    TYPE2 = 350,
    LENGTH = 351,
    SUBSTR = 352,
    NUMBSYM = 353,
    STRGSYM = 354,
    VARSYM = 355,
    SETSYM = 356,
    NUMBDEF = 357,
    STRGDEF = 358,
    BOOLDEF = 359,
    SETDEF = 360,
    DEFNAME = 361,
    NAME = 362,
    STRG = 363,
    NUMB = 364,
    SCALE = 365,
    SEPARATE = 366,
    CHECKONLY = 367,
    INDICATOR = 368,
    QUBO = 369,
    PENALTY1 = 370,
    PENALTY2 = 371,
    PENALTY3 = 372,
    PENALTY4 = 373,
    PENALTY5 = 374,
    PENALTY6 = 375
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 87 "src/zimpl/mmlparse2.y" /* yacc.c:1909  */

   unsigned int bits;
   Numb*        numb;
   const char*  strg;
   const char*  name;
   Symbol*      sym;
   Define*      def;
   CodeNode*    code;

#line 185 "src/zimpl/mmlparse2.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int yyparse (void);

#endif /* !YY_YY_SRC_ZIMPL_MMLPARSE2_H_INCLUDED  */
