/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "src/zimpl/mmlparse2.y" /* yacc.c:339  */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mmlparse2.y                                                   */
/*   Name....: MML Parser                                                    */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wunused-macros"
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif
   
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC   diagnostic ignored "-Wstrict-prototypes"
#endif
   
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
   
#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/idxset.h"
#include "zimpl/rdefpar.h"
#include "zimpl/bound.h"
#include "zimpl/define.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/list.h"
#include "zimpl/stmt.h"
#include "zimpl/local.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"
        
#define YYERROR_VERBOSE 1

/* the function is actually getting a YYSTYPE* as argument, but the
 * type isn't available here, so it is decalred to accept any number of
 * arguments, i.e. yylex() and not yylex(void).
 */
extern int yylex();

/*lint -sem(yyerror, 1p, r_no) */ 
extern void yyerror(const char* s) is_NORETURN;
 

#line 149 "src/zimpl/mmlparse2.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "mmlparse2.h".  */
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
#line 87 "src/zimpl/mmlparse2.y" /* yacc.c:355  */

   unsigned int bits;
   Numb*        numb;
   const char*  strg;
   const char*  name;
   Symbol*      sym;
   Define*      def;
   CodeNode*    code;

#line 320 "src/zimpl/mmlparse2.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int yyparse (void);

#endif /* !YY_YY_SRC_ZIMPL_MMLPARSE2_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 336 "src/zimpl/mmlparse2.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  40
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   3352

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  133
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  59
/* YYNRULES -- Number of rules.  */
#define YYNRULES  316
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  932

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   375

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     128,   129,   123,   121,   127,   122,     2,   130,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,   124,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   125,     2,   126,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   131,     2,   132,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   156,   156,   157,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   174,   181,   187,   193,   203,   204,   207,
     210,   213,   219,   228,   237,   246,   255,   264,   267,   277,
     280,   283,   286,   293,   297,   298,   305,   306,   314,   321,
     330,   340,   351,   360,   370,   374,   384,   385,   386,   390,
     393,   394,   395,   400,   408,   409,   410,   411,   416,   424,
     425,   429,   430,   438,   439,   442,   443,   447,   451,   455,
     458,   470,   473,   483,   489,   492,   495,   500,   505,   513,
     516,   521,   526,   533,   537,   542,   546,   552,   555,   560,
     565,   570,   574,   581,   588,   594,   600,   606,   611,   619,
     628,   637,   645,   656,   659,   663,   668,   676,   677,   680,
     683,   684,   687,   690,   691,   694,   697,   698,   701,   704,
     705,   708,   711,   712,   715,   718,   719,   720,   721,   722,
     726,   727,   731,   732,   733,   734,   735,   736,   737,   738,
     739,   740,   741,   745,   746,   747,   751,   752,   753,   754,
     755,   758,   759,   767,   768,   769,   773,   774,   778,   779,
     780,   786,   787,   790,   796,   799,   800,   801,   802,   803,
     804,   805,   806,   807,   808,   809,   812,   815,   818,   826,
     832,   835,   841,   842,   843,   851,   855,   856,   857,   858,
     859,   860,   861,   871,   872,   879,   882,   888,   889,   890,
     893,   894,   897,   898,   901,   902,   906,   907,   908,   911,
     915,   918,   923,   924,   927,   930,   933,   936,   939,   942,
     945,   948,   949,   950,   951,   952,   953,   954,   957,   960,
     966,   967,   971,   972,   973,   974,   978,   981,   984,   988,
     989,   990,   991,   992,   993,   994,   995,   996,   997,   998,
     999,  1000,  1001,  1002,  1003,  1004,  1005,  1006,  1007,  1012,
    1018,  1019,  1023,  1026,  1032,  1035,  1041,  1042,  1043,  1047,
    1048,  1049,  1050,  1051,  1052,  1058,  1059,  1060,  1064,  1065,
    1066,  1069,  1072,  1075,  1078,  1084,  1085,  1086,  1089,  1092,
    1095,  1100,  1105,  1106,  1107,  1108,  1109,  1110,  1111,  1112,
    1113,  1114,  1115,  1116,  1117,  1118,  1119,  1120,  1121,  1123,
    1124,  1125,  1128,  1131,  1134,  1137,  1140
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "DECLSET", "DECLPAR", "DECLVAR",
  "DECLMIN", "DECLMAX", "DECLSUB", "DECLSOS", "DEFNUMB", "DEFSTRG",
  "DEFBOOL", "DEFSET", "PRINT", "CHECK", "BINARY", "INTEGER", "REAL",
  "IMPLICIT", "ASGN", "DO", "WITH", "IN", "TO", "UNTIL", "BY", "FORALL",
  "EXISTS", "PRIORITY", "STARTVAL", "DEFAULT", "CMP_LE", "CMP_GE",
  "CMP_EQ", "CMP_LT", "CMP_GT", "CMP_NE", "INFTY", "AND", "OR", "XOR",
  "NOT", "SUM", "MIN", "MAX", "ARGMIN", "ARGMAX", "PROD", "IF", "THEN",
  "ELSE", "END", "INTER", "UNION", "CROSS", "SYMDIFF", "WITHOUT", "PROJ",
  "MOD", "DIV", "POW", "FAC", "CARD", "ROUND", "FLOOR", "CEIL", "RANDOM",
  "ORD", "ABS", "SGN", "LOG", "LN", "EXP", "SQRT", "SIN", "COS", "TAN",
  "ASIN", "ACOS", "ATAN", "POWER", "SGNPOW", "READ", "AS", "SKIP", "USE",
  "COMMENT", "MATCH", "SUBSETS", "INDEXSET", "POWERSET", "VIF", "VABS",
  "TYPE1", "TYPE2", "LENGTH", "SUBSTR", "NUMBSYM", "STRGSYM", "VARSYM",
  "SETSYM", "NUMBDEF", "STRGDEF", "BOOLDEF", "SETDEF", "DEFNAME", "NAME",
  "STRG", "NUMB", "SCALE", "SEPARATE", "CHECKONLY", "INDICATOR", "QUBO",
  "PENALTY1", "PENALTY2", "PENALTY3", "PENALTY4", "PENALTY5", "PENALTY6",
  "'+'", "'-'", "'*'", "';'", "'['", "']'", "','", "'('", "')'", "'/'",
  "'{'", "'}'", "$accept", "stmt", "decl_set", "set_entry_list",
  "set_entry", "def_numb", "def_strg", "def_bool", "def_set", "name_list",
  "decl_par", "par_singleton", "par_default", "decl_var", "var_type",
  "lower", "upper", "priority", "startval", "cexpr_entry_list",
  "cexpr_entry", "matrix_head", "matrix_body", "decl_obj", "decl_sub",
  "constraint_list", "constraint", "vbool", "con_attr_list", "con_attr",
  "con_type", "vexpr", "vproduct", "vfactor", "vexpo", "vval", "decl_sos",
  "soset", "sos_type", "exec_do", "command", "idxset", "pure_idxset",
  "sexpr", "sunion", "sproduct", "sval", "read", "read_par", "tuple_list",
  "lexpr", "tuple", "symidx", "cexpr_list", "cexpr", "cproduct", "cfactor",
  "cexpo", "cval", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,    43,    45,    42,    59,    91,    93,    44,    40,    41,
      47,   123,   125
};
# endif

#define YYPACT_NINF -540

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-540)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    1450,   -66,   -46,   -20,    -4,    53,    78,    91,     5,   103,
     115,   146,    75,   248,  -540,  -540,  -540,  -540,  -540,  -540,
    -540,  -540,  -540,  -540,  -540,    19,    22,    12,   303,   328,
     340,   366,   288,   310,   318,   327,  1520,  1610,   685,   375,
    -540,  1190,   947,  1430,   685,   422,   500,  -540,   109,   685,
     500,  2650,  2650,  1699,   213,   444,   444,   444,   444,   430,
    1978,  1610,   685,    -1,     2,   894,  1091,   685,  1610,   685,
     685,   439,   450,   455,   470,   491,   542,   550,   562,   572,
     575,   607,   626,   646,   665,   671,   684,   699,   723,   726,
     733,   746,   757,   477,   477,  -540,   477,   767,   771,   772,
     799,  -540,  -540,  -540,  3186,  3186,  1610,  1770,   376,  -540,
      -5,  -540,   721,   583,   658,   143,    99,  -540,  -540,   281,
     376,   721,   583,   143,  1610,  1190,   865,  -540,  1130,   891,
    -540,   902,   916,   813,  2851,  1610,  2851,  2851,   818,   823,
    -540,   923,  1118,  2851,   156,   825,  2851,   924,  2918,   925,
     839,  -540,   838,   925,   685,  1610,   847,   853,   858,   859,
     869,   870,   874,   878,   883,   887,   888,   893,   477,  2717,
    2717,  2650,   179,   -67,  -540,  -540,   952,   518,   338,   392,
     685,  1610,  2516,    44,  -540,   184,   198,   685,  -540,  -540,
     903,   422,  -540,     9,   107,   228,   251,   685,  -540,     4,
     156,  -540,  1008,  1841,  1014,  1841,  1018,  2447,  1020,  2447,
    1033,  1036,   323,  1038,  1040,  1190,  1190,  2851,  2851,  2851,
    2851,  1190,  2851,  2851,  2851,  2851,  2851,  2851,  2851,  2851,
    2851,  2851,  2851,  2851,   962,  2851,  2851,  2851,  -540,  -540,
    -540,  2851,  2851,  2851,  2851,  -540,  -540,   335,    33,    32,
    1610,  2447,  -540,   -11,  1118,   148,   891,   249,    35,  1190,
    1190,  1190,  1190,  1190,  1190,  1190,  1190,  1190,  1190,  1190,
    1190,   927,   927,  1610,  1610,  1610,  1190,  2851,  2851,  2851,
    2851,  2851,  2851,  2851,  2851,  2851,  3119,  3119,  3119,  3119,
    3119,  -540,   623,   272,    75,  1190,  -540,     3,  1045,    21,
    1003,   -40,   141,  -540,  1034,  2851,   923,  2851,  2851,  2851,
    2851,  -540,   156,  1048,   156,  2851,   948,  1610,  2246,   156,
    2045,   422,  -540,  1711,   950,  1056,  1006,  2650,  2650,  2650,
    2650,  2650,  2650,  2650,  2650,  2650,  2650,  2650,  2650,  -540,
    1233,  1233,  -540,  -540,   355,   358,  2650,  2650,  -540,  2784,
    3119,  3119,  2650,  2650,  2784,  -540,  1060,  1010,  2516,  2516,
    1093,   661,   789,  2583,  -540,  -540,  -540,  -540,  2650,  2650,
    1063,  -540,  1064,   979,  1084,  1087,  1100,  1105,   984,  -540,
    2851,   998,   333,  3119,  1007,   379,  3119,   426,  2851,   514,
    2851,  3119,  1610,   927,  3221,   388,   349,   560,   655,   675,
     238,   588,   768,   780,   791,   909,   958,  1069,  1111,  1120,
    1149,  1152,  1163,  1165,  1012,  1195,   374,   540,   484,   498,
     803,   833,  -540,  -540,  -540,  1107,  1910,  -540,  1034,  -540,
    -540,  2851,  2851,  1130,  1130,  1130,  1130,  1130,  1130,  -540,
    -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  1103,  1103,
    1130,   156,   156,   156,   156,   156,   156,   156,    99,    99,
    -540,  -540,  -540,  -540,  -540,  1190,  -540,     1,  1011,  1023,
     307,  -540,  1190,   503,  -540,  2851,  2851,  -540,    23,  2851,
     156,   156,   156,   156,  1322,   156,  -540,  1128,  -540,  -540,
    1610,   156,   924,   422,   500,   729,   500,  -540,  2650,  2650,
    1201,  1210,  1213,  1219,  1230,  1252,  1311,  1316,  1320,  1347,
    1390,  1394,  1413,  1424,  1428,  1480,  1482,  1484,   458,   474,
    1486,   685,  1610,  1026,  1046,  1047,  1052,  1057,  1065,  1066,
    1072,  1079,  2650,  -540,   -67,   338,   -67,   338,  -540,  -540,
    -540,  -540,   -67,   338,   -67,   338,  -540,  1699,  1699,  -540,
      45,   253,   278,  2516,  2516,  2516,  2650,  2650,  2650,  2650,
    2650,  2650,  2650,  2650,  2650,  2650,  2650,  2650,  2650,  -540,
     696,   518,    43,   218,   213,  2650,  -540,  2851,  2851,  1610,
    1190,  -540,    99,  -540,  -540,  -540,  -540,  -540,  -540,   685,
     156,   685,   156,  -540,   651,   295,   796,  -540,    -5,  1034,
    -540,  -540,  -540,  -540,  2851,  2851,  -540,  -540,  -540,  -540,
    -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,
    2851,  -540,  -540,  -540,  -540,  -540,  2447,  1032,   -70,  -540,
      -8,     7,   738,  1610,  1190,  1190,  -540,  1034,  1130,   367,
    1055,   104,   156,  -540,    25,  2851,   -12,   441,  2985,  1185,
    1094,   924,   925,  1110,   925,   -67,   338,   293,   315,  -540,
    -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  2851,  2851,
    -540,  1167,  1188,  2650,  2650,  2650,  2650,  2650,  2650,  2650,
    2650,  2650,  1178,    65,   449,   495,  -540,  -540,  1206,  1206,
     184,   198,   696,   518,   696,   518,   696,   518,   696,   518,
     696,   518,   696,   518,   696,   696,   696,   696,   696,   696,
    1119,  1119,  2851,  2851,  1119,  2851,  2851,  1119,  -540,   696,
     657,   713,   227,  1099,  1229,  1248,  1190,  1610,  2851,  1143,
    1510,   601,   856,  -540,  -540,  2851,  -540,  2851,  -540,   721,
     615,   478,  -540,  -540,  -540,  -540,  1049,  2851,  1151,  -540,
    2313,   415,  2112,  -540,  1155,   422,  -540,  1159,  2650,  1514,
    1540,  2650,  2650,  1699,  -540,  2650,  2650,  1639,   156,   156,
     156,   156,  -540,  -540,  -540,  -540,  2851,  2851,  1109,    56,
      82,  -540,  -540,  2851,  2851,   209,   216,  2851,  -540,  -540,
     156,  -540,  1237,  3052,  1238,   418,  -540,   924,  -540,   161,
    -540,  -540,   338,    41,   -16,   493,   502,   508,  -540,  -540,
    -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,
    1119,  1119,  1119,  1119,   156,   156,  -540,  -540,  -540,  1570,
    1572,  -540,  -540,   592,  2851,  2380,  2851,  2179,  1166,  -540,
    -540,  2650,  -540,  2650,  -540,  2650,  -540,  2650,  -540,  -540,
    -540,  2851,  -540,   352,  1247,   451,  1260,  -540,   184,   198,
    1119,   184,   198,  1119,   184,   198,  1119,   184,   198,  1119,
    1574,  -540,  -540,  -540,  -540,  2650,  2650,  2650,  2650,  2650,
    2650,  2650,  2650,  -540,   453,   539,   553,   666,   681,   728,
     754,   756,   758,   762,   784,   786,   867,   882,   886,   897,
    -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,  -540,
    -540,  -540,  -540,  -540,  -540,  -540,  1119,  1119,  1119,  1119,
    1119,  1119,  1119,  1119,  1119,  1119,  1119,  1119,  1119,  1119,
    1119,  1119
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       0,     0,    33,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     2,     8,     9,    10,    11,     3,
       4,     5,     6,     7,    12,     0,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       1,     0,     0,     0,     0,    59,    49,    47,     0,     0,
      49,     0,     0,     0,   182,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   262,   262,   190,   262,     0,     0,     0,
       0,   287,   286,   285,     0,     0,     0,     0,   188,   197,
     204,   206,   189,   187,   186,   264,   266,   269,   275,   278,
       0,   191,     0,     0,     0,     0,     0,   193,   194,     0,
     185,     0,     0,     0,     0,     0,     0,     0,     0,    34,
      63,     0,    65,     0,    35,     0,     0,    61,     0,    54,
       0,    48,     0,    54,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   262,     0,
       0,     0,     0,   146,   153,   158,   161,     0,   266,     0,
       0,     0,     0,     0,    74,     0,     0,     0,   183,   184,
       0,    59,    27,     0,     0,     0,     0,     0,   260,     0,
     264,   254,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   288,   289,
     210,     0,     0,     0,     0,   276,   277,     0,     0,     0,
       0,     0,   212,     0,   238,     0,   236,     0,   264,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   292,     0,     0,     0,     0,    13,     0,     0,     0,
       0,     0,     0,    32,     0,     0,    66,     0,     0,     0,
       0,   231,    67,     0,    60,     0,     0,     0,     0,    50,
       0,    59,    41,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   164,
       0,     0,   159,   160,     0,     0,     0,     0,    71,     0,
       0,     0,     0,     0,     0,    72,     0,     0,     0,     0,
       0,     0,     0,     0,    73,   143,   144,   145,     0,     0,
       0,   179,     0,     0,     0,     0,     0,     0,     0,   261,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   221,   255,   309,     0,     0,   224,     0,   222,
     223,     0,     0,   250,   248,   245,   249,   247,   246,   203,
     198,   200,   202,   199,   201,   207,   208,   251,   252,   253,
     256,   265,   244,   242,   239,   243,   241,   240,   267,   268,
     272,   273,   270,   271,   279,     0,   192,   196,     0,     0,
       0,    17,     0,     0,    68,     0,     0,    64,     0,     0,
     232,   233,   234,   235,     0,    62,    43,     0,    51,    56,
       0,    55,    61,    59,    49,     0,    49,    39,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   178,   147,   149,   148,   150,   157,   154,
     155,   162,   151,   267,   152,   268,   156,     0,     0,   128,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    75,
     130,   130,   130,   130,   182,     0,    28,     0,     0,     0,
       0,   257,   280,   283,   315,   281,   284,   316,   282,     0,
     217,     0,   219,   274,     0,     0,     0,   209,   205,     0,
     293,   296,   297,   298,     0,     0,   294,   295,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   228,   310,
       0,   263,   290,   291,   258,   211,     0,     0,     0,   237,
       0,     0,     0,     0,     0,     0,    16,     0,    22,     0,
       0,     0,   230,    69,     0,     0,    36,     0,     0,     0,
       0,    61,    54,     0,    54,   163,   280,     0,     0,   173,
     174,   167,   169,   168,   166,   170,   171,   172,     0,     0,
     165,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    76,     0,     0,     0,   129,   125,   126,   127,
       0,     0,   113,   115,   116,   118,   110,   112,   119,   121,
     122,   124,   107,   109,   114,   117,   111,   120,   123,   108,
      79,    80,     0,     0,    81,     0,     0,    82,   181,   180,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   226,   225,     0,   214,     0,   216,   195,
       0,     0,    18,    15,    14,    70,     0,     0,     0,    30,
       0,     0,     0,    45,     0,    59,    40,     0,     0,     0,
       0,     0,     0,     0,    77,     0,     0,     0,   130,   130,
     130,   130,    23,    24,    25,    26,     0,     0,     0,     0,
       0,   227,   312,     0,     0,     0,     0,     0,    21,    31,
      37,    29,     0,     0,     0,     0,    42,    61,    38,     0,
     175,   176,     0,     0,     0,     0,     0,     0,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   131,
      83,    85,    84,    86,   218,   220,   229,   259,   313,     0,
       0,   213,   215,     0,     0,     0,     0,     0,     0,   177,
      78,     0,   130,     0,   130,     0,   130,     0,   130,   314,
     311,     0,    19,     0,     0,     0,     0,    44,     0,     0,
     103,     0,     0,   105,     0,     0,   104,     0,     0,   106,
       0,    53,    52,    58,    57,     0,     0,     0,     0,     0,
       0,     0,     0,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,    87,    91,    90,    97,
      89,    96,    95,   101,    88,    94,    93,   100,    92,    99,
      98,   102
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -540,  -540,  -540,   846,   688,  -540,  -540,  -540,  -540,   668,
    -540,  -540,  -540,  -540,  1013,   -44,  -150,  -189,  -488,   845,
    1039,  -136,  -540,  -540,  -540,  -539,   974,  -343,   353,  -540,
    -109,    90,  -326,  -121,  -540,  -540,  -540,   764,  -540,  -540,
    1053,   423,   794,   855,  1455,   955,  -260,  1239,  -540,  -540,
     -15,   885,    55,   142,   -36,   535,   -33,   -91,  -540
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    13,    14,   470,   471,    15,    16,    17,    18,   193,
      19,   138,   748,    20,    50,   149,   321,   147,   316,   139,
     140,   141,   306,    21,    22,   183,   184,   360,   710,   819,
     368,   361,   173,   174,   175,   176,    23,   190,   191,    24,
      39,   126,   127,   128,   109,   110,   111,   142,   311,   255,
     112,   129,   238,   114,   177,   116,   117,   118,   119
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint16 yytable[] =
{
     115,   123,   372,   324,   650,   305,   153,   144,   682,   683,
     426,   445,   446,   245,   246,   549,   550,   186,   735,   747,
     534,   536,   121,   633,   200,   123,   542,   544,    45,    46,
      47,    48,   123,   737,    60,   841,   842,    60,    60,    41,
     379,    25,    43,   474,   476,   643,   201,   745,   342,   343,
     271,   284,   285,   212,   265,   266,   349,   267,   268,   431,
     432,    26,   734,   350,   278,   279,   280,   281,   282,   283,
     249,   258,   273,   274,   275,   712,   713,   369,   245,   246,
     363,   284,   285,   363,   553,   554,   555,    27,   123,    36,
      37,   248,   468,   840,   469,   273,   274,   275,   200,   123,
     301,   302,    38,    28,   363,   346,   347,   312,   827,   292,
     314,    32,   319,   284,   285,   304,   763,   764,   272,   123,
     300,   427,   269,   270,   736,   150,   151,   203,   284,   285,
     205,   277,   492,   597,   828,   345,   373,    49,   374,   738,
     326,   172,   179,   185,    42,   123,   362,    44,   277,   239,
     277,   240,   277,   284,   285,   728,   284,   285,   286,   287,
      29,   424,   423,   754,   346,   347,   357,   200,   364,   200,
     479,   387,   655,   389,   686,   278,   279,   280,   281,   282,
     283,   397,   398,   399,   400,    30,   402,   403,   404,   405,
     406,   407,   408,   409,   410,   411,   412,   413,    31,   415,
     416,   200,   199,   284,   285,   200,   200,   200,   200,    33,
     687,   688,   689,   839,   123,   302,   365,   366,   367,   342,
     343,    34,   288,   339,   803,   284,   285,   246,   538,   289,
     365,   366,   367,   546,   373,   425,   375,   123,   123,   123,
     187,   451,   452,   453,   454,   455,   456,   457,    40,   257,
     715,   716,    35,   460,   461,   462,   463,   464,   447,   448,
     449,   344,   284,   285,   284,   285,   273,   274,   275,   200,
     424,   480,   481,   482,   483,   428,   299,   284,   285,   485,
     429,   123,   346,   347,   491,   557,   558,   559,   560,   561,
     562,   501,   503,   505,   507,   509,   511,   513,   515,   517,
     346,   347,   487,   348,   651,   346,   347,   188,   189,   838,
     563,   564,   565,   566,   567,   568,   539,   540,   541,   352,
     353,   462,   362,   552,    51,   265,   266,   186,   267,   268,
     284,   285,   571,   573,   273,   274,   275,   284,   285,   352,
     353,   831,   290,   291,   758,   382,   727,   385,   832,    52,
     585,   774,   590,   588,   592,   373,   596,   376,   593,   284,
     285,    53,   273,   274,   275,   604,   728,   259,   260,   261,
     262,   263,   264,   392,   346,   347,   277,   595,   373,   417,
     377,   430,   533,   418,   419,   420,   421,    54,   265,   266,
     628,   267,   268,   269,   270,   630,   631,   286,   287,   352,
     353,   422,   265,   266,   871,   267,   268,   424,   259,   260,
     261,   262,   263,   264,   346,   347,    55,   500,   502,   504,
     506,   508,   510,   512,   514,   516,   518,   519,   520,   265,
     266,   636,   267,   268,   637,   655,   352,   353,    56,   641,
     642,   265,   266,   200,   267,   268,    57,   478,   647,   551,
     652,   146,   654,   185,   123,    58,   269,   270,   570,   572,
     277,   354,   584,   658,   422,   133,   793,   145,   289,   837,
     269,   270,   152,   284,   285,   649,   346,   347,   600,   352,
     353,   365,   366,   367,   533,   202,   123,   424,   208,   210,
     211,   743,   213,   214,   637,   284,   285,   269,   270,   130,
     758,   620,   755,   873,   757,   900,   277,   672,   587,   269,
     270,   186,   685,   346,   347,   599,   355,   362,   362,   362,
     691,   693,   695,   697,   699,   701,   703,   365,   366,   367,
     253,   265,   266,   148,   267,   268,   284,   285,    60,   284,
     285,   720,   721,   123,   843,   844,   728,   284,   285,    65,
      66,   192,   124,   845,   846,   589,    69,    70,   197,   847,
     848,    71,   284,   285,   722,   749,   797,   215,   730,   731,
     346,   347,   284,   285,   346,   347,   369,   325,   216,   346,
     347,   765,   766,   217,   732,   668,   178,   178,   178,   657,
     641,   901,   468,    90,   469,   346,   347,   123,   218,   269,
     270,   669,   237,   356,    96,   902,   276,   788,   100,   746,
     370,   277,   751,   622,   352,   353,   352,   353,   739,   219,
     378,   644,   344,   346,   347,   277,   381,   623,   384,   352,
     353,   125,   759,   760,   107,   284,   285,   185,   684,   352,
     353,   265,   266,   591,   267,   268,   690,   692,   694,   696,
     698,   700,   702,   704,   705,   706,   707,   708,   709,   246,
     352,   353,   273,   274,   275,   719,   621,   277,   265,   266,
     220,   267,   268,   465,   346,   347,   768,   769,   221,   770,
     771,   284,   285,   259,   260,   261,   262,   263,   264,   601,
     222,   123,   780,   557,   558,   559,   560,   561,   562,   785,
     223,   786,   726,   224,   265,   266,   178,   267,   268,   269,
     270,   790,   779,   284,   285,   605,   795,   178,   903,   851,
      60,   852,   284,   285,   194,   195,   196,   186,   783,   805,
     807,    65,    66,   904,   124,   225,   269,   270,    69,    70,
     824,   825,   787,    71,   246,   653,   151,   829,   830,   875,
     876,   833,   877,   878,   226,   879,   880,   780,   881,   882,
     273,   274,   275,   500,   502,   504,   506,   508,   510,   512,
     514,   516,   269,   270,   227,    90,   284,   285,   284,   285,
     905,   772,   346,   347,   602,   277,    96,   352,   353,   726,
     100,   265,   266,   228,   267,   268,   284,   285,   853,   229,
     855,   780,   346,   347,   603,   859,   906,   862,   907,   865,
     908,   868,   230,   125,   909,   870,   107,   346,   347,   458,
     459,   563,   564,   565,   566,   567,   568,   231,   278,   279,
     280,   281,   282,   283,   284,   285,   910,   773,   911,   885,
     887,   889,   891,   893,   895,   897,   899,   728,   799,   352,
     353,   232,   657,   185,   233,   804,   806,   204,   206,   269,
     270,   234,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   235,   346,   347,   352,   353,   346,
     347,   535,   537,   352,   353,   236,   294,   543,   545,   284,
     285,   108,   120,   178,   178,   241,   131,   606,   178,   242,
     243,   284,   285,   178,   178,   346,   347,   352,   353,   607,
     352,   353,   284,   285,   295,   582,   120,   284,   285,   912,
     608,   113,   122,   120,   711,   714,   717,   244,   143,    60,
     277,   858,   624,   861,   913,   864,   297,   867,   914,   298,
      65,    66,   303,   124,   671,   134,   122,    69,    70,   915,
     304,   313,    71,   122,   315,   265,   266,   320,   267,   268,
     277,   247,   625,   322,   323,   884,   886,   888,   890,   892,
     894,   896,   898,    65,    66,   327,   124,   284,   285,   120,
     293,   328,    60,   784,    90,    71,   329,   330,   346,   347,
     120,   122,   256,    65,    66,    96,   124,   331,   332,   100,
      69,    70,   333,   352,   353,    71,   334,   346,   347,   122,
     120,   335,   724,   351,   725,   336,   337,    90,   352,   353,
     122,   338,   207,   269,   270,   107,   296,   371,    96,   380,
     284,   285,   100,   656,   178,   383,   120,    90,   609,   386,
     122,   388,   273,   274,   275,   273,   274,   275,    96,   273,
     274,   275,   100,   475,   390,   125,   499,   391,   107,   393,
     548,   394,   293,   414,   293,   473,   122,   178,   484,    60,
     395,   396,   486,   132,   497,   125,   401,   498,   107,   284,
     285,   547,   178,   178,   574,   575,   576,   610,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   577,   120,   293,   578,   265,   266,
     178,   267,   268,   581,   433,   434,   435,   436,   437,   438,
     579,   820,   821,   822,   823,   580,    60,   583,   120,   120,
     120,   450,   553,   554,   555,   122,   586,    65,    66,   634,
     124,   618,   273,   556,    69,    70,   273,   274,   275,    71,
     467,   635,   265,   266,   673,   267,   268,   626,   122,   122,
     122,   826,   265,   266,   733,   267,   268,   273,   274,   275,
     284,   285,   120,   789,   674,   675,   269,   270,   648,   744,
     676,    90,   472,   265,   266,   677,   267,   268,   761,   143,
     284,   285,    96,   678,   679,   860,   100,   863,   611,   866,
     680,   869,   122,   307,   308,   309,   310,   681,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   363,   753,   209,
     269,   270,   107,   775,   273,   274,   275,   273,   274,   275,
     269,   270,   284,   285,   756,   752,    65,    66,   762,   124,
     612,   284,   285,    69,    70,   553,   767,   594,    71,   613,
     776,   269,   270,   916,   917,   918,   919,   920,   921,   922,
     923,   924,   925,   926,   927,   928,   929,   930,   931,   777,
     284,   285,   781,   284,   285,   791,   521,   122,   614,   796,
      90,   615,   522,   798,   284,   285,   284,   285,   834,   836,
     857,    96,   616,   178,   617,   100,   802,   178,   178,   872,
     178,   178,   523,   524,   525,   526,   527,   528,   529,   530,
     531,   627,   874,   629,   165,   166,   284,   285,   125,   639,
     632,   107,   346,   347,   619,   742,   167,   638,   640,   646,
     659,   352,   353,   168,   346,   347,   496,   569,   718,   606,
     352,   353,   660,   477,   134,   120,   254,   466,   607,   598,
       0,   346,   347,   645,   340,   341,     0,    60,   472,   661,
       0,   532,     0,     0,     0,    62,    63,    64,     0,   143,
      67,   135,     0,   352,   353,   122,   178,   120,   178,     0,
     178,   608,   178,     0,     0,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,     0,     0,   136,     0,   122,     0,     0,
     178,   178,   178,   178,   178,   178,   178,   178,    91,    92,
      93,    94,     0,     0,    97,    98,     0,     0,     0,   101,
     102,   103,   346,   347,   120,   723,     0,   352,   353,     0,
     662,   346,   347,   104,   105,   609,     0,     0,     0,   663,
     137,     0,   134,     1,     2,     3,     4,     5,     6,     7,
       8,     9,    10,    11,   122,    60,     0,     0,   352,   353,
       0,    12,     0,    62,    63,    64,   610,     0,    67,   135,
       0,   632,     0,     0,   729,     0,     0,     0,   120,   740,
     741,     0,     0,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,   346,   347,   136,     0,   352,   353,     0,   122,   664,
       0,     0,   472,   611,     0,     0,    91,    92,    93,    94,
       0,     0,    97,    98,   346,   347,     0,   101,   102,   103,
       0,     0,   665,     0,     0,   352,   353,     0,    59,   346,
     347,   104,   105,   612,     0,    60,     0,   666,   137,     0,
       0,     0,    61,    62,    63,    64,    65,    66,    67,    68,
       0,     0,     0,    69,    70,     0,     0,     0,    71,     0,
       0,   778,   120,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,   352,   353,   346,   347,   352,   353,   346,   347,   613,
      90,   667,   122,   614,     0,   670,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,     0,   101,   102,   103,
       0,   284,   285,     0,     0,   284,   285,     0,    59,   782,
       0,   104,   105,   800,     0,    60,     0,     0,   106,     0,
       0,   107,    61,    62,    63,    64,    65,    66,    67,    68,
       0,   284,   285,    69,    70,     0,     0,     0,    71,   801,
       0,     0,     0,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,   284,   285,   284,   285,   284,   285,     0,     0,   849,
      90,   850,     0,   883,     0,     0,    91,    92,    93,    94,
       0,    96,    97,    98,    99,   100,     0,   101,   102,   103,
     439,   440,   441,   442,   443,   444,   180,   493,   494,    47,
     495,   104,   105,     0,     0,     0,     0,     0,   106,     0,
       0,   107,   154,    63,    64,     0,     0,    67,   181,   808,
     809,   810,   811,   812,   813,   814,   815,   816,   817,   818,
       0,     0,    72,    73,    74,    75,    76,    77,   156,   157,
     158,   159,   160,   161,   162,   163,   164,    87,    88,    89,
     165,   166,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   182,   167,     0,     0,    91,    92,    93,    94,   168,
       0,    97,    98,     0,     0,    60,   101,   102,   103,     0,
       0,     0,     0,    62,    63,    64,    65,    66,    67,   250,
     169,   170,     0,    69,    70,     0,     0,   171,    71,     0,
       0,     0,     0,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,     0,     0,   136,     0,     0,     0,     0,     0,     0,
      90,     0,     0,     0,     0,     0,    91,    92,    93,    94,
       0,    96,    97,    98,     0,   100,    60,   101,   102,   103,
       0,     0,     0,     0,    62,    63,    64,    65,    66,    67,
     250,   104,   105,     0,    69,    70,     0,     0,   251,    71,
       0,   107,   252,     0,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    90,     0,     0,     0,     0,     0,    91,    92,    93,
      94,     0,    96,    97,    98,    60,   100,     0,   101,   102,
     103,     0,     0,    62,    63,    64,     0,     0,    67,   135,
       0,     0,   104,   105,     0,     0,     0,     0,     0,   251,
       0,     0,   107,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    91,    92,    93,    94,
       0,     0,    97,    98,   198,     0,     0,   101,   102,   103,
       0,    62,    63,    64,     0,     0,    67,   135,     0,     0,
       0,   104,   105,     0,     0,     0,     0,     0,   137,     0,
       0,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    91,    92,    93,    94,     0,     0,
      97,    98,     0,   489,     0,   101,   102,   103,    62,    63,
      64,     0,     0,    67,   490,     0,     0,     0,     0,   104,
     105,     0,     0,     0,     0,     0,   137,     0,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    91,    92,    93,    94,     0,     0,    97,    98,     0,
     794,     0,   101,   102,   103,    62,    63,    64,     0,     0,
      67,   135,     0,     0,     0,     0,   104,   105,     0,     0,
       0,     0,     0,   137,     0,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    91,    92,
      93,    94,     0,     0,    97,    98,     0,   856,     0,   101,
     102,   103,    62,    63,    64,     0,     0,    67,   135,     0,
       0,     0,     0,   104,   105,     0,     0,     0,     0,     0,
     137,     0,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    91,    92,    93,    94,     0,
       0,    97,    98,     0,   488,     0,   101,   102,   103,    62,
      63,    64,     0,     0,     0,   135,     0,     0,     0,     0,
     104,   105,     0,     0,     0,     0,     0,   137,     0,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    91,    92,    93,    94,     0,     0,    97,    98,
       0,   792,     0,   101,   102,   103,    62,    63,    64,     0,
       0,     0,   135,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   137,     0,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    91,
      92,    93,    94,     0,     0,    97,    98,     0,   854,     0,
     101,   102,   103,    62,    63,    64,     0,     0,     0,   135,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   137,     0,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    91,    92,    93,    94,
       0,     0,    97,    98,     0,     0,     0,   101,   102,   103,
      62,    63,    64,    65,    66,    67,   250,     0,     0,     0,
      69,    70,     0,     0,     0,    71,     0,     0,   137,     0,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    90,     0,     0,
       0,     0,     0,    91,    92,    93,    94,     0,    96,    97,
      98,     0,   100,     0,   101,   102,   103,     0,   358,   154,
      63,    64,     0,     0,    67,   155,     0,     0,   104,   105,
       0,     0,     0,     0,     0,   251,     0,     0,   107,    72,
      73,    74,    75,    76,    77,   156,   157,   158,   159,   160,
     161,   162,   163,   164,    87,    88,    89,   165,   166,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   167,
       0,     0,    91,    92,    93,    94,   168,     0,    97,    98,
       0,     0,     0,   101,   102,   103,   154,    63,    64,     0,
       0,    67,   155,     0,     0,     0,     0,   169,   170,     0,
       0,     0,     0,     0,   359,     0,    72,    73,    74,    75,
      76,    77,   156,   157,   158,   159,   160,   161,   162,   163,
     164,    87,    88,    89,   165,   166,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   182,   167,     0,     0,    91,
      92,    93,    94,   168,     0,    97,    98,     0,     0,     0,
     101,   102,   103,   154,    63,    64,     0,     0,    67,   155,
       0,     0,     0,     0,   169,   170,     0,     0,     0,     0,
       0,   171,     0,    72,    73,    74,    75,    76,    77,   156,
     157,   158,   159,   160,   161,   162,   163,   164,    87,    88,
      89,   165,   166,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   167,     0,     0,    91,    92,    93,    94,
     168,     0,    97,    98,     0,     0,     0,   101,   102,   103,
     154,    63,    64,     0,     0,     0,   155,     0,     0,     0,
       0,   169,   170,     0,     0,     0,     0,     0,   171,     0,
      72,    73,    74,    75,    76,    77,   156,   157,   158,   159,
     160,   161,   162,   163,   164,    87,    88,    89,   165,   166,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     167,     0,     0,    91,    92,    93,    94,   168,     0,    97,
      98,     0,     0,     0,   101,   102,   103,   154,    63,    64,
       0,     0,     0,   155,     0,     0,     0,     0,   340,   341,
       0,     0,     0,     0,     0,   171,     0,    72,    73,    74,
      75,    76,    77,   156,   157,   158,   159,   160,   161,   162,
     163,   164,    87,    88,    89,   165,   166,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   167,     0,     0,
      91,    92,    93,    94,   168,     0,    97,    98,     0,     0,
       0,   101,   102,   103,    62,    63,    64,     0,     0,    67,
     135,     0,     0,     0,     0,   169,   170,     0,     0,     0,
       0,     0,   171,     0,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    91,    92,    93,
      94,     0,     0,    97,    98,     0,     0,     0,   101,   102,
     103,    62,    63,    64,     0,     0,    67,   317,     0,     0,
       0,     0,   104,   105,     0,     0,     0,     0,     0,   137,
       0,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    91,    92,    93,    94,     0,     0,
      97,    98,     0,     0,     0,   101,   102,   103,    62,    63,
      64,     0,     0,    67,   135,     0,     0,     0,     0,   104,
     318,     0,     0,     0,     0,     0,   137,     0,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    91,    92,    93,    94,     0,     0,    97,    98,     0,
       0,     0,   101,   102,   103,    62,    63,    64,     0,     0,
      67,   135,     0,     0,     0,     0,   104,   750,     0,     0,
       0,     0,     0,   137,     0,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    91,    92,
      93,    94,     0,     0,    97,    98,     0,     0,     0,   101,
     102,   103,    62,    63,    64,     0,     0,     0,   135,     0,
       0,     0,     0,   104,   835,     0,     0,     0,     0,     0,
     137,     0,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    91,    92,    93,    94,     0,
       0,    97,    98,     0,     0,     0,   101,   102,   103,    62,
      63,    64,     0,     0,     0,   135,     0,     0,     0,     0,
     104,   105,     0,     0,     0,     0,     0,   137,     0,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    65,    66,     0,
     124,     0,     0,     0,    69,     0,     0,     0,     0,    71,
       0,     0,    91,    92,    93,    94,     0,     0,    97,    98,
       0,     0,     0,   101,   102,   103,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    90,     0,     0,   137,     0,     0,     0,     0,     0,
       0,     0,    96,     0,     0,     0,   100,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   125,
       0,     0,   107
};

static const yytype_int16 yycheck[] =
{
      36,    37,   191,   153,   492,   141,    50,    43,   547,   548,
      21,   271,   272,   104,   105,   358,   359,    53,    26,    31,
     346,   347,    37,    22,    60,    61,   352,   353,    16,    17,
      18,    19,    68,    26,    35,    51,    52,    35,    35,    20,
      36,   107,    20,    22,    84,    22,    61,    22,   169,   170,
      55,   121,   122,    68,    53,    54,   123,    56,    57,    24,
      25,   107,   132,   130,    32,    33,    34,    35,    36,    37,
     106,   107,    39,    40,    41,    32,    33,   186,   169,   170,
      39,   121,   122,    39,    39,    40,    41,   107,   124,    14,
      15,   106,    89,    52,    91,    39,    40,    41,   134,   135,
     136,   137,    27,   107,    39,   121,   122,   143,    52,   124,
     146,   106,   148,   121,   122,   127,    51,    52,   123,   155,
     135,   132,   121,   122,   132,    16,    17,   128,   121,   122,
     128,   127,   321,   393,    52,   171,   127,   125,   129,   132,
     155,    51,    52,    53,   125,   181,   182,   125,   127,    94,
     127,    96,   127,   121,   122,    51,   121,   122,    59,    60,
     107,   129,   129,   651,   121,   122,   181,   203,   124,   205,
     306,   207,   498,   209,   129,    32,    33,    34,    35,    36,
      37,   217,   218,   219,   220,   107,   222,   223,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   107,   235,
     236,   237,    60,   121,   122,   241,   242,   243,   244,   106,
     553,   554,   555,    52,   250,   251,    32,    33,    34,   340,
     341,   106,   123,   168,   763,   121,   122,   318,   349,   130,
      32,    33,    34,   354,   127,   250,   129,   273,   274,   275,
      27,   277,   278,   279,   280,   281,   282,   283,     0,   107,
      32,    33,   106,   286,   287,   288,   289,   290,   273,   274,
     275,   171,   121,   122,   121,   122,    39,    40,    41,   305,
     129,   307,   308,   309,   310,   127,   134,   121,   122,   315,
     132,   317,   121,   122,   320,    32,    33,    34,    35,    36,
      37,   327,   328,   329,   330,   331,   332,   333,   334,   335,
     121,   122,   317,   124,   493,   121,   122,    94,    95,   797,
      32,    33,    34,    35,    36,    37,   349,   350,   351,   121,
     122,   354,   358,   359,    21,    53,    54,   363,    56,    57,
     121,   122,   368,   369,    39,    40,    41,   121,   122,   121,
     122,   132,    61,    62,    51,   203,    51,   205,   132,    21,
     383,   124,   388,   386,   390,   127,   392,   129,   391,   121,
     122,    21,    39,    40,    41,   127,    51,    32,    33,    34,
      35,    36,    37,    50,   121,   122,   127,   392,   127,   237,
     129,   132,   129,   241,   242,   243,   244,    21,    53,    54,
     426,    56,    57,   121,   122,   431,   432,    59,    60,   121,
     122,   129,    53,    54,    52,    56,    57,   129,    32,    33,
      34,    35,    36,    37,   121,   122,   128,   327,   328,   329,
     330,   331,   332,   333,   334,   335,   336,   337,   338,    53,
      54,   124,    56,    57,   127,   761,   121,   122,   128,   475,
     476,    53,    54,   479,    56,    57,   128,   305,   484,   359,
     494,    29,   496,   363,   490,   128,   121,   122,   368,   369,
     127,   123,   129,   499,   129,    42,    51,    44,   130,    51,
     121,   122,    49,   121,   122,   490,   121,   122,   129,   121,
     122,    32,    33,    34,   129,    62,   522,   129,    65,    66,
      67,   124,    69,    70,   127,   121,   122,   121,   122,   124,
      51,   127,   652,    52,   654,    52,   127,   522,   129,   121,
     122,   547,   548,   121,   122,   127,   124,   553,   554,   555,
     556,   557,   558,   559,   560,   561,   562,    32,    33,    34,
     107,    53,    54,    33,    56,    57,   121,   122,    35,   121,
     122,   577,   578,   579,    51,    52,    51,   121,   122,    46,
      47,   107,    49,    51,    52,   129,    53,    54,   128,    51,
      52,    58,   121,   122,   579,   124,   755,   128,   604,   605,
     121,   122,   121,   122,   121,   122,   685,   154,   128,   121,
     122,   690,   691,   128,   620,   127,    51,    52,    53,   499,
     626,    52,    89,    90,    91,   121,   122,   633,   128,   121,
     122,   127,   125,   180,   101,    52,    23,   129,   105,   645,
     187,   127,   648,   129,   121,   122,   121,   122,   633,   128,
     197,   479,   532,   121,   122,   127,   203,   129,   205,   121,
     122,   128,   668,   669,   131,   121,   122,   547,   548,   121,
     122,    53,    54,   129,    56,    57,   556,   557,   558,   559,
     560,   561,   562,   563,   564,   565,   566,   567,   568,   750,
     121,   122,    39,    40,    41,   575,   126,   127,    53,    54,
     128,    56,    57,    50,   121,   122,   712,   713,   128,   715,
     716,   121,   122,    32,    33,    34,    35,    36,    37,   129,
     128,   727,   728,    32,    33,    34,    35,    36,    37,   735,
     128,   737,    51,   128,    53,    54,   171,    56,    57,   121,
     122,   747,   727,   121,   122,   127,   752,   182,    52,   127,
      35,   129,   121,   122,    56,    57,    58,   763,   127,   765,
     766,    46,    47,    52,    49,   128,   121,   122,    53,    54,
     776,   777,   127,    58,   835,    16,    17,   783,   784,   858,
     859,   787,   861,   862,   128,   864,   865,   793,   867,   868,
      39,    40,    41,   673,   674,   675,   676,   677,   678,   679,
     680,   681,   121,   122,   128,    90,   121,   122,   121,   122,
      52,   124,   121,   122,   129,   127,   101,   121,   122,    51,
     105,    53,    54,   128,    56,    57,   121,   122,   834,   128,
     836,   837,   121,   122,   129,   841,    52,   843,    52,   845,
      52,   847,   128,   128,    52,   851,   131,   121,   122,   284,
     285,    32,    33,    34,    35,    36,    37,   128,    32,    33,
      34,    35,    36,    37,   121,   122,    52,   124,    52,   875,
     876,   877,   878,   879,   880,   881,   882,    51,   758,   121,
     122,   128,   762,   763,   128,   765,   766,    63,    64,   121,
     122,   128,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   128,   121,   122,   121,   122,   121,
     122,   346,   347,   121,   122,   128,    21,   352,   353,   121,
     122,    36,    37,   358,   359,   128,    41,   129,   363,   128,
     128,   121,   122,   368,   369,   121,   122,   121,   122,   129,
     121,   122,   121,   122,    23,   380,    61,   121,   122,    52,
     129,    36,    37,    68,   571,   572,   573,   128,    43,    35,
     127,   841,   129,   843,    52,   845,    20,   847,    52,   126,
      46,    47,   124,    49,   521,    22,    61,    53,    54,    52,
     127,   126,    58,    68,    30,    53,    54,    32,    56,    57,
     127,   106,   129,   124,   126,   875,   876,   877,   878,   879,
     880,   881,   882,    46,    47,   128,    49,   121,   122,   124,
     125,   128,    35,   127,    90,    58,   128,   128,   121,   122,
     135,   106,   107,    46,    47,   101,    49,   128,   128,   105,
      53,    54,   128,   121,   122,    58,   128,   121,   122,   124,
     155,   128,   589,    61,   591,   128,   128,    90,   121,   122,
     135,   128,   128,   121,   122,   131,   124,   124,   101,    21,
     121,   122,   105,   498,   499,    21,   181,    90,   129,    21,
     155,    21,    39,    40,    41,    39,    40,    41,   101,    39,
      40,    41,   105,    50,    21,   128,    50,    21,   131,    21,
      50,    21,   207,   101,   209,    20,   181,   532,    20,    35,
     215,   216,   124,   126,   124,   128,   221,    21,   131,   121,
     122,    21,   547,   548,    21,    21,   107,   129,   553,   554,
     555,   556,   557,   558,   559,   560,   561,   562,   563,   564,
     565,   566,   567,   568,    20,   250,   251,    20,    53,    54,
     575,    56,    57,   129,   259,   260,   261,   262,   263,   264,
      20,   768,   769,   770,   771,    20,    35,   129,   273,   274,
     275,   276,    39,    40,    41,   250,   129,    46,    47,   128,
      49,   129,    39,    50,    53,    54,    39,    40,    41,    58,
     295,   128,    53,    54,   128,    56,    57,    50,   273,   274,
     275,    52,    53,    54,   132,    56,    57,    39,    40,    41,
     121,   122,   317,   124,   128,   128,   121,   122,    50,   124,
     128,    90,   297,    53,    54,   128,    56,    57,    21,   304,
     121,   122,   101,   128,   128,   842,   105,   844,   129,   846,
     128,   848,   317,    85,    86,    87,    88,   128,   673,   674,
     675,   676,   677,   678,   679,   680,   681,    39,   124,   128,
     121,   122,   131,   124,    39,    40,    41,    39,    40,    41,
     121,   122,   121,   122,   124,    50,    46,    47,    50,    49,
     129,   121,   122,    53,    54,    39,   127,   392,    58,   129,
      21,   121,   122,   900,   901,   902,   903,   904,   905,   906,
     907,   908,   909,   910,   911,   912,   913,   914,   915,    21,
     121,   122,   129,   121,   122,   124,    43,   392,   129,   124,
      90,   129,    49,   124,   121,   122,   121,   122,    51,    51,
     124,   101,   129,   758,   129,   105,   761,   762,   763,    52,
     765,   766,    69,    70,    71,    72,    73,    74,    75,    76,
      77,   426,    52,   428,    81,    82,   121,   122,   128,   473,
     465,   131,   121,   122,   129,   637,    93,   472,   473,   484,
     129,   121,   122,   100,   121,   122,   323,   363,   574,   129,
     121,   122,   129,   304,    22,   490,   107,   294,   129,   394,
      -1,   121,   122,    31,   121,   122,    -1,    35,   473,   129,
      -1,   128,    -1,    -1,    -1,    43,    44,    45,    -1,   484,
      48,    49,    -1,   121,   122,   490,   841,   522,   843,    -1,
     845,   129,   847,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    -1,    -1,    83,    -1,   522,    -1,    -1,
     875,   876,   877,   878,   879,   880,   881,   882,    96,    97,
      98,    99,    -1,    -1,   102,   103,    -1,    -1,    -1,   107,
     108,   109,   121,   122,   579,   580,    -1,   121,   122,    -1,
     129,   121,   122,   121,   122,   129,    -1,    -1,    -1,   129,
     128,    -1,    22,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,   579,    35,    -1,    -1,   121,   122,
      -1,    21,    -1,    43,    44,    45,   129,    -1,    48,    49,
      -1,   626,    -1,    -1,   599,    -1,    -1,    -1,   633,   634,
     635,    -1,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,   121,   122,    83,    -1,   121,   122,    -1,   633,   129,
      -1,    -1,   637,   129,    -1,    -1,    96,    97,    98,    99,
      -1,    -1,   102,   103,   121,   122,    -1,   107,   108,   109,
      -1,    -1,   129,    -1,    -1,   121,   122,    -1,    28,   121,
     122,   121,   122,   129,    -1,    35,    -1,   129,   128,    -1,
      -1,    -1,    42,    43,    44,    45,    46,    47,    48,    49,
      -1,    -1,    -1,    53,    54,    -1,    -1,    -1,    58,    -1,
      -1,   726,   727,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,   121,   122,   121,   122,   121,   122,   121,   122,   129,
      90,   129,   727,   129,    -1,   129,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,   107,   108,   109,
      -1,   121,   122,    -1,    -1,   121,   122,    -1,    28,   129,
      -1,   121,   122,   129,    -1,    35,    -1,    -1,   128,    -1,
      -1,   131,    42,    43,    44,    45,    46,    47,    48,    49,
      -1,   121,   122,    53,    54,    -1,    -1,    -1,    58,   129,
      -1,    -1,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,   121,   122,   121,   122,   121,   122,    -1,    -1,   129,
      90,   129,    -1,   129,    -1,    -1,    96,    97,    98,    99,
      -1,   101,   102,   103,   104,   105,    -1,   107,   108,   109,
     265,   266,   267,   268,   269,   270,    27,    16,    17,    18,
      19,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,
      -1,   131,    43,    44,    45,    -1,    -1,    48,    49,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
      -1,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    96,    97,    98,    99,   100,
      -1,   102,   103,    -1,    -1,    35,   107,   108,   109,    -1,
      -1,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
     121,   122,    -1,    53,    54,    -1,    -1,   128,    58,    -1,
      -1,    -1,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      90,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,
      -1,   101,   102,   103,    -1,   105,    35,   107,   108,   109,
      -1,    -1,    -1,    -1,    43,    44,    45,    46,    47,    48,
      49,   121,   122,    -1,    53,    54,    -1,    -1,   128,    58,
      -1,   131,   132,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    90,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,
      99,    -1,   101,   102,   103,    35,   105,    -1,   107,   108,
     109,    -1,    -1,    43,    44,    45,    -1,    -1,    48,    49,
      -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,
      -1,    -1,   131,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,
      -1,    -1,   102,   103,    36,    -1,    -1,   107,   108,   109,
      -1,    43,    44,    45,    -1,    -1,    48,    49,    -1,    -1,
      -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    96,    97,    98,    99,    -1,    -1,
     102,   103,    -1,    38,    -1,   107,   108,   109,    43,    44,
      45,    -1,    -1,    48,    49,    -1,    -1,    -1,    -1,   121,
     122,    -1,    -1,    -1,    -1,    -1,   128,    -1,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    96,    97,    98,    99,    -1,    -1,   102,   103,    -1,
      38,    -1,   107,   108,   109,    43,    44,    45,    -1,    -1,
      48,    49,    -1,    -1,    -1,    -1,   121,   122,    -1,    -1,
      -1,    -1,    -1,   128,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    96,    97,
      98,    99,    -1,    -1,   102,   103,    -1,    38,    -1,   107,
     108,   109,    43,    44,    45,    -1,    -1,    48,    49,    -1,
      -1,    -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,
     128,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,    -1,
      -1,   102,   103,    -1,    38,    -1,   107,   108,   109,    43,
      44,    45,    -1,    -1,    -1,    49,    -1,    -1,    -1,    -1,
     121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    96,    97,    98,    99,    -1,    -1,   102,   103,
      -1,    38,    -1,   107,   108,   109,    43,    44,    45,    -1,
      -1,    -1,    49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   128,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    96,
      97,    98,    99,    -1,    -1,   102,   103,    -1,    38,    -1,
     107,   108,   109,    43,    44,    45,    -1,    -1,    -1,    49,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   128,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,
      -1,    -1,   102,   103,    -1,    -1,    -1,   107,   108,   109,
      43,    44,    45,    46,    47,    48,    49,    -1,    -1,    -1,
      53,    54,    -1,    -1,    -1,    58,    -1,    -1,   128,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    90,    -1,    -1,
      -1,    -1,    -1,    96,    97,    98,    99,    -1,   101,   102,
     103,    -1,   105,    -1,   107,   108,   109,    -1,    42,    43,
      44,    45,    -1,    -1,    48,    49,    -1,    -1,   121,   122,
      -1,    -1,    -1,    -1,    -1,   128,    -1,    -1,   131,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      -1,    -1,    96,    97,    98,    99,   100,    -1,   102,   103,
      -1,    -1,    -1,   107,   108,   109,    43,    44,    45,    -1,
      -1,    48,    49,    -1,    -1,    -1,    -1,   121,   122,    -1,
      -1,    -1,    -1,    -1,   128,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,    -1,   102,   103,    -1,    -1,    -1,
     107,   108,   109,    43,    44,    45,    -1,    -1,    48,    49,
      -1,    -1,    -1,    -1,   121,   122,    -1,    -1,    -1,    -1,
      -1,   128,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    -1,    -1,    96,    97,    98,    99,
     100,    -1,   102,   103,    -1,    -1,    -1,   107,   108,   109,
      43,    44,    45,    -1,    -1,    -1,    49,    -1,    -1,    -1,
      -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    -1,    -1,    96,    97,    98,    99,   100,    -1,   102,
     103,    -1,    -1,    -1,   107,   108,   109,    43,    44,    45,
      -1,    -1,    -1,    49,    -1,    -1,    -1,    -1,   121,   122,
      -1,    -1,    -1,    -1,    -1,   128,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    -1,    -1,
      96,    97,    98,    99,   100,    -1,   102,   103,    -1,    -1,
      -1,   107,   108,   109,    43,    44,    45,    -1,    -1,    48,
      49,    -1,    -1,    -1,    -1,   121,   122,    -1,    -1,    -1,
      -1,    -1,   128,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,
      99,    -1,    -1,   102,   103,    -1,    -1,    -1,   107,   108,
     109,    43,    44,    45,    -1,    -1,    48,    49,    -1,    -1,
      -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    96,    97,    98,    99,    -1,    -1,
     102,   103,    -1,    -1,    -1,   107,   108,   109,    43,    44,
      45,    -1,    -1,    48,    49,    -1,    -1,    -1,    -1,   121,
     122,    -1,    -1,    -1,    -1,    -1,   128,    -1,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    96,    97,    98,    99,    -1,    -1,   102,   103,    -1,
      -1,    -1,   107,   108,   109,    43,    44,    45,    -1,    -1,
      48,    49,    -1,    -1,    -1,    -1,   121,   122,    -1,    -1,
      -1,    -1,    -1,   128,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    96,    97,
      98,    99,    -1,    -1,   102,   103,    -1,    -1,    -1,   107,
     108,   109,    43,    44,    45,    -1,    -1,    -1,    49,    -1,
      -1,    -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,
     128,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,    -1,
      -1,   102,   103,    -1,    -1,    -1,   107,   108,   109,    43,
      44,    45,    -1,    -1,    -1,    49,    -1,    -1,    -1,    -1,
     121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    46,    47,    -1,
      49,    -1,    -1,    -1,    53,    -1,    -1,    -1,    -1,    58,
      -1,    -1,    96,    97,    98,    99,    -1,    -1,   102,   103,
      -1,    -1,    -1,   107,   108,   109,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    90,    -1,    -1,   128,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   101,    -1,    -1,    -1,   105,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,
      -1,    -1,   131
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    21,   134,   135,   138,   139,   140,   141,   143,
     146,   156,   157,   169,   172,   107,   107,   107,   107,   107,
     107,   107,   106,   106,   106,   106,    14,    15,    27,   173,
       0,    20,   125,    20,   125,    16,    17,    18,    19,   125,
     147,    21,    21,    21,    21,   128,   128,   128,   128,    28,
      35,    42,    43,    44,    45,    46,    47,    48,    49,    53,
      54,    58,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      90,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   107,   108,   109,   121,   122,   128,   131,   176,   177,
     178,   179,   183,   184,   186,   187,   188,   189,   190,   191,
     176,   183,   184,   187,    49,   128,   174,   175,   176,   184,
     124,   176,   126,   174,    22,    49,    83,   128,   144,   152,
     153,   154,   180,   184,   187,   174,    29,   150,    33,   148,
      16,    17,   174,   148,    43,    49,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    81,    82,    93,   100,   121,
     122,   128,   164,   165,   166,   167,   168,   187,   188,   164,
      27,    49,    92,   158,   159,   164,   187,    27,    94,    95,
     170,   171,   107,   142,   142,   142,   142,   128,    36,   186,
     187,   183,   174,   128,   175,   128,   175,   128,   174,   128,
     174,   174,   183,   174,   174,   128,   128,   128,   128,   128,
     128,   128,   128,   128,   128,   128,   128,   128,   128,   128,
     128,   128,   128,   128,   128,   128,   128,   125,   185,   185,
     185,   128,   128,   128,   128,   190,   190,   176,   183,   187,
      49,   128,   132,   174,   180,   182,   184,   186,   187,    32,
      33,    34,    35,    36,    37,    53,    54,    56,    57,   121,
     122,    55,   123,    39,    40,    41,    23,   127,    32,    33,
      34,    35,    36,    37,   121,   122,    59,    60,   123,   130,
      61,    62,   183,   176,    21,    23,   124,    20,   126,   186,
     183,   187,   187,   124,   127,   154,   155,    85,    86,    87,
      88,   181,   187,   126,   187,    30,   151,    49,   122,   187,
      32,   149,   124,   126,   149,   174,   183,   128,   128,   128,
     128,   128,   128,   128,   128,   128,   128,   128,   128,   185,
     121,   122,   166,   166,   164,   187,   121,   122,   124,   123,
     130,    61,   121,   122,   123,   124,   174,   183,    42,   128,
     160,   164,   187,    39,   124,    32,    33,    34,   163,   163,
     174,   124,   150,   127,   129,   129,   129,   129,   174,    36,
      21,   174,   186,    21,   174,   186,    21,   187,    21,   187,
      21,    21,    50,    21,    21,   176,   176,   187,   187,   187,
     187,   176,   187,   187,   187,   187,   187,   187,   187,   187,
     187,   187,   187,   187,   101,   187,   187,   186,   186,   186,
     186,   186,   129,   129,   129,   183,    21,   132,   127,   132,
     132,    24,    25,   176,   176,   176,   176,   176,   176,   177,
     177,   177,   177,   177,   177,   179,   179,   183,   183,   183,
     176,   187,   187,   187,   187,   187,   187,   187,   188,   188,
     189,   189,   189,   189,   189,    50,   173,   176,    89,    91,
     136,   137,   184,    20,    22,    50,    84,   153,   186,   154,
     187,   187,   187,   187,    20,   187,   124,   183,    38,    38,
      49,   187,   150,    16,    17,    19,   147,   124,    21,    50,
     164,   187,   164,   187,   164,   187,   164,   187,   164,   187,
     164,   187,   164,   187,   164,   187,   164,   187,   164,   164,
     164,    43,    49,    69,    70,    71,    72,    73,    74,    75,
      76,    77,   128,   129,   165,   188,   165,   188,   166,   189,
     189,   189,   165,   188,   165,   188,   166,    21,    50,   160,
     160,   164,   187,    39,    40,    41,    50,    32,    33,    34,
      35,    36,    37,    32,    33,    34,    35,    36,    37,   159,
     164,   187,   164,   187,    21,    21,   107,    20,    20,    20,
      20,   129,   188,   129,   129,   189,   129,   129,   189,   129,
     187,   129,   187,   189,   176,   183,   187,   179,   178,   127,
     129,   129,   129,   129,   127,   127,   129,   129,   129,   129,
     129,   129,   129,   129,   129,   129,   129,   129,   129,   129,
     127,   126,   129,   129,   129,   129,    50,   184,   187,   184,
     187,   187,   176,    22,   128,   128,   124,   127,   176,   136,
     176,   187,   187,    22,   186,    31,   152,   187,    50,   183,
     151,   150,   148,    16,   148,   165,   188,   164,   187,   129,
     129,   129,   129,   129,   129,   129,   129,   129,   127,   127,
     129,   174,   183,   128,   128,   128,   128,   128,   128,   128,
     128,   128,   158,   158,   164,   187,   129,   160,   160,   160,
     164,   187,   164,   187,   164,   187,   164,   187,   164,   187,
     164,   187,   164,   187,   164,   164,   164,   164,   164,   164,
     161,   161,    32,    33,   161,    32,    33,   161,   170,   164,
     187,   187,   183,   176,   174,   174,    51,    51,    51,   184,
     187,   187,   187,   132,   132,    26,   132,    26,   132,   183,
     176,   176,   137,   124,   124,    22,   187,    31,   145,   124,
     122,   187,    50,   124,   151,   149,   124,   149,    51,   187,
     187,    21,    50,    51,    52,   163,   163,   127,   187,   187,
     187,   187,   124,   124,   124,   124,    21,    21,   176,   183,
     187,   129,   129,   127,   127,   187,   187,   127,   129,   124,
     187,   124,    38,    51,    38,   187,   124,   150,   124,   164,
     129,   129,   188,   158,   164,   187,   164,   187,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   162,
     161,   161,   161,   161,   187,   187,    52,    52,    52,   187,
     187,   132,   132,   187,    51,   122,    51,    51,   151,    52,
      52,    51,    52,    51,    52,    51,    52,    51,    52,   129,
     129,   127,   129,   187,    38,   187,    38,   124,   164,   187,
     161,   164,   187,   161,   164,   187,   161,   164,   187,   161,
     187,    52,    52,    52,    52,   163,   163,   163,   163,   163,
     163,   163,   163,   129,   164,   187,   164,   187,   164,   187,
     164,   187,   164,   187,   164,   187,   164,   187,   164,   187,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,   161,   161,   161,   161,
     161,   161,   161,   161,   161,   161,   161,   161,   161,   161,
     161,   161
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   133,   134,   134,   134,   134,   134,   134,   134,   134,
     134,   134,   134,   135,   135,   135,   135,   136,   136,   136,
     136,   136,   137,   138,   139,   140,   141,   142,   142,   143,
     143,   143,   143,   143,   144,   144,   145,   145,   146,   146,
     146,   146,   146,   146,   146,   146,   147,   147,   147,   148,
     148,   148,   148,   148,   149,   149,   149,   149,   149,   150,
     150,   151,   151,   152,   152,   152,   152,   153,   154,   155,
     155,   156,   156,   157,   158,   158,   158,   158,   158,   159,
     159,   159,   159,   159,   159,   159,   159,   159,   159,   159,
     159,   159,   159,   159,   159,   159,   159,   159,   159,   159,
     159,   159,   159,   159,   159,   159,   159,   160,   160,   160,
     160,   160,   160,   160,   160,   160,   160,   160,   160,   160,
     160,   160,   160,   160,   160,   160,   160,   160,   160,   160,
     161,   161,   162,   162,   162,   162,   162,   162,   162,   162,
     162,   162,   162,   163,   163,   163,   164,   164,   164,   164,
     164,   164,   164,   165,   165,   165,   165,   165,   166,   166,
     166,   167,   167,   167,   168,   168,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   169,
     170,   170,   171,   171,   171,   172,   173,   173,   173,   173,
     173,   173,   173,   174,   174,   175,   175,   176,   176,   176,
     176,   176,   176,   176,   177,   177,   178,   178,   178,   178,
     179,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     180,   180,   181,   181,   181,   181,   182,   182,   182,   183,
     183,   183,   183,   183,   183,   183,   183,   183,   183,   183,
     183,   183,   183,   183,   183,   183,   183,   183,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   187,   188,
     188,   188,   188,   188,   188,   189,   189,   189,   190,   190,
     190,   190,   190,   190,   190,   191,   191,   191,   191,   191,
     191,   191,   191,   191,   191,   191,   191,   191,   191,   191,
     191,   191,   191,   191,   191,   191,   191,   191,   191,   191,
     191,   191,   191,   191,   191,   191,   191
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     5,     8,     8,     7,     1,     3,     6,
       8,     4,     2,     8,     8,     8,     8,     1,     3,     9,
       8,     9,     5,     1,     1,     1,     0,     2,     9,     6,
       8,     5,     9,     6,    11,     8,     0,     1,     2,     0,
       2,     3,     9,     9,     0,     2,     2,     8,     8,     0,
       2,     0,     2,     1,     3,     1,     2,     2,     3,     3,
       4,     5,     5,     5,     1,     3,     4,     5,     7,     4,
       4,     4,     4,     6,     6,     6,     6,    12,    12,    12,
      12,    12,    12,    12,    12,    12,    12,    12,    12,    12,
      12,    12,    12,     8,     8,     8,     8,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       0,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     3,     1,     3,     3,     3,     3,     1,     2,
       2,     1,     3,     4,     2,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     6,     6,     7,     3,     5,
       4,     4,     0,     1,     1,     3,     2,     2,     2,     2,
       2,     2,     4,     1,     1,     5,     3,     1,     3,     3,
       3,     3,     3,     3,     1,     4,     1,     3,     3,     4,
       2,     4,     2,     7,     5,     7,     5,     4,     7,     4,
       7,     3,     3,     3,     3,     5,     5,     6,     4,     7,
       4,     2,     2,     2,     2,     2,     1,     3,     1,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     2,     3,     3,     4,     4,     7,
       2,     3,     0,     3,     1,     3,     1,     3,     3,     1,
       3,     3,     3,     3,     4,     1,     2,     2,     1,     3,
       4,     4,     4,     4,     4,     1,     1,     1,     2,     2,
       4,     4,     2,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     3,
       4,     8,     6,     7,     8,     4,     4
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 156 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2508 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 3:
#line 157 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2514 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 4:
#line 158 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2520 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 5:
#line 159 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2526 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 6:
#line 160 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2532 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 7:
#line 161 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2538 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 8:
#line 162 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2544 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 9:
#line 163 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2550 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 10:
#line 164 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2556 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 11:
#line 165 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2562 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 12:
#line 166 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { code_set_root((yyvsp[0].code)); }
#line 2568 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 13:
#line 174 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_set1, 3,
            code_new_name((yyvsp[-3].name)),                                       /* Name */
            code_new_inst(i_idxset_pseudo_new, 1,               /* index set */
               code_new_inst(i_bool_true, 0)),              
            (yyvsp[-1].code));                                              /* initial set */
      }
#line 2580 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 14:
#line 181 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_set1, 3,
            code_new_name((yyvsp[-6].name)),                                       /* Name */
            (yyvsp[-4].code),                                                 /* index set */
            (yyvsp[-1].code));                                                      /* set */
      }
#line 2591 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 15:
#line 187 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_set2, 3,
            code_new_name((yyvsp[-6].name)),                                       /* Name */
            (yyvsp[-4].code),                                                 /* index set */
            (yyvsp[-1].code));                                   /* initial set_entry_list */
      }
#line 2602 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 16:
#line 193 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_set2, 3,
            code_new_name((yyvsp[-5].name)),                                       /* Name */
            code_new_inst(i_idxset_pseudo_new, 1,               /* index set */
               code_new_inst(i_bool_true, 0)),              
            (yyvsp[-1].code));                                   /* initial set_entry_list */
      }
#line 2614 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 17:
#line 203 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_entry_list_new, 1, (yyvsp[0].code)); }
#line 2620 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 18:
#line 204 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_add, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 2628 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 19:
#line 207 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_subsets, 3, (yyvsp[-3].code), (yyvsp[-1].code), code_new_numb(numb_new_integer(-1)));
      }
#line 2636 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 20:
#line 210 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_subsets, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 2644 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 21:
#line 213 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_powerset, 1, (yyvsp[-1].code));
      }
#line 2652 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 22:
#line 219 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_entry, 2, (yyvsp[-1].code), (yyvsp[0].code)); }
#line 2658 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 23:
#line 228 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newdef, 3,
            code_new_define((yyvsp[-6].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-4].code)),
            (yyvsp[-1].code));
      }
#line 2669 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 24:
#line 237 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newdef, 3,
            code_new_define((yyvsp[-6].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-4].code)),
            (yyvsp[-1].code));
      }
#line 2680 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 25:
#line 246 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newdef, 3,
            code_new_define((yyvsp[-6].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-4].code)),
            (yyvsp[-1].code));
      }
#line 2691 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 26:
#line 255 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newdef, 3,
            code_new_define((yyvsp[-6].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-4].code)),
            (yyvsp[-1].code));
      }
#line 2702 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 27:
#line 264 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_elem_list_new, 1, code_new_name((yyvsp[0].name)));
      }
#line 2710 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 28:
#line 267 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_elem_list_add, 2, (yyvsp[-2].code), code_new_name((yyvsp[0].name)));
      }
#line 2718 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 29:
#line 277 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_para1, 4, code_new_name((yyvsp[-7].name)), (yyvsp[-5].code), (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 2726 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 30:
#line 280 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_para2, 4, code_new_name((yyvsp[-6].name)), (yyvsp[-4].code), (yyvsp[-1].code), code_new_inst(i_nop, 0));
      }
#line 2734 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 31:
#line 283 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_para2, 4, code_new_name((yyvsp[-7].name)), (yyvsp[-5].code), (yyvsp[-1].code), code_new_inst(i_nop, 0));
      }
#line 2742 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 32:
#line 286 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_para1, 4,
            code_new_name((yyvsp[-3].name)),
            code_new_inst(i_idxset_pseudo_new, 1, code_new_inst(i_bool_true, 0)),
            (yyvsp[-1].code),
            code_new_inst(i_nop, 0));
      }
#line 2754 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 33:
#line 293 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_nop, 0); }
#line 2760 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 34:
#line 297 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 2766 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 35:
#line 298 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_new, 1,
            code_new_inst(i_entry, 2, code_new_inst(i_tuple_empty, 0), (yyvsp[0].code)));
      }
#line 2775 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 36:
#line 305 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_nop, 0); }
#line 2781 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 37:
#line 306 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_entry, 2, code_new_inst(i_tuple_empty, 0), (yyvsp[0].code)); }
#line 2787 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 38:
#line 314 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-7].name)),
            (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
#line 2799 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 39:
#line 321 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-4].name)),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
#line 2813 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 40:
#line 330 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-6].name)),
            (yyvsp[-4].code),
            code_new_varclass(VAR_IMP),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
#line 2828 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 41:
#line 340 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-3].name)),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_IMP),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
#line 2844 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 42:
#line 351 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-7].name)),
            (yyvsp[-5].code),
            code_new_varclass(VAR_INT),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 2858 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 43:
#line 360 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-4].name)),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_INT),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 2873 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 44:
#line 370 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-9].name)), (yyvsp[-7].code), code_new_varclass(VAR_INT), (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 2882 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 45:
#line 374 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_newsym_var, 7,
            code_new_name((yyvsp[-6].name)),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_INT), (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 2894 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 46:
#line 384 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_varclass(VAR_CON); }
#line 2900 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 47:
#line 385 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_varclass(VAR_CON); }
#line 2906 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 48:
#line 386 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_varclass(VAR_IMP); }
#line 2912 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 49:
#line 390 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0)));
      }
#line 2920 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 50:
#line 393 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bound_new, 1, (yyvsp[0].code)); }
#line 2926 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 51:
#line 394 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_bound(BOUND_MINUS_INFTY); }
#line 2932 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 52:
#line 395 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-6].code),
            code_new_inst(i_bound_new, 1, (yyvsp[-4].code)),
            code_new_bound(BOUND_MINUS_INFTY));
      }
#line 2942 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 53:
#line 400 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-6].code),
            code_new_bound(BOUND_MINUS_INFTY),
            code_new_inst(i_bound_new, 1, (yyvsp[-1].code)));
      }
#line 2952 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 54:
#line 408 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_bound(BOUND_INFTY); }
#line 2958 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 55:
#line 409 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bound_new, 1, (yyvsp[0].code)); }
#line 2964 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 56:
#line 410 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_bound(BOUND_INFTY); }
#line 2970 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 57:
#line 411 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code),
            code_new_inst(i_bound_new, 1, (yyvsp[-3].code)),
            code_new_bound(BOUND_INFTY));
      }
#line 2980 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 58:
#line 416 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code),
            code_new_bound(BOUND_INFTY),
            code_new_inst(i_bound_new, 1, (yyvsp[-1].code)));
      }
#line 2990 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 59:
#line 424 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb(numb_new_integer(0)); }
#line 2996 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 60:
#line 425 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3002 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 61:
#line 429 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb(numb_copy(numb_unknown())); }
#line 3008 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 62:
#line 430 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3014 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 63:
#line 438 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_entry_list_new, 1, (yyvsp[0].code)); }
#line 3020 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 64:
#line 439 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_entry_list_add, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 3028 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 65:
#line 442 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read, 1, (yyvsp[0].code)); }
#line 3034 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 66:
#line 443 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_list_matrix, 2, (yyvsp[-1].code), (yyvsp[0].code)); }
#line 3040 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 67:
#line 447 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_entry, 2, (yyvsp[-1].code), (yyvsp[0].code)); }
#line 3046 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 68:
#line 451 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 3052 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 69:
#line 455 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_matrix_list_new, 2, (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 3060 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 70:
#line 458 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_matrix_list_add, 3, (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code));
      }
#line 3068 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 71:
#line 470 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_object_min, 2, code_new_name((yyvsp[-3].name)), (yyvsp[-1].code));
      }
#line 3076 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 72:
#line 473 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_object_max, 2, code_new_name((yyvsp[-3].name)), (yyvsp[-1].code));
      }
#line 3084 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 73:
#line 483 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_subto, 2, code_new_name((yyvsp[-3].name)), (yyvsp[-1].code));
     }
#line 3092 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 74:
#line 489 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint_list, 2, (yyvsp[0].code), code_new_inst(i_nop, 0));
     }
#line 3100 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 75:
#line 492 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint_list, 2, (yyvsp[-2].code), (yyvsp[0].code));
     }
#line 3108 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 76:
#line 495 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_forall, 2, (yyvsp[-2].code), (yyvsp[0].code)),
           code_new_inst(i_nop, 0));
     }
#line 3118 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 77:
#line 500 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_expr_if_else, 3, (yyvsp[-3].code), (yyvsp[-1].code), code_new_inst(i_nop, 0)),
           code_new_inst(i_nop, 0));
      }
#line 3128 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 78:
#line 505 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code)),
           code_new_inst(i_nop, 0));
      }
#line 3138 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 79:
#line 513 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint, 4, (yyvsp[-3].code), (yyvsp[-2].code), (yyvsp[-1].code), code_new_bits((yyvsp[0].bits)));
     }
#line 3146 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 80:
#line 516 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint, 4, (yyvsp[-3].code), (yyvsp[-2].code),
           code_new_inst(i_term_expr, 1, (yyvsp[-1].code)),
           code_new_bits((yyvsp[0].bits)));
     }
#line 3156 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 81:
#line 521 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_constraint, 4,
           code_new_inst(i_term_expr, 1, (yyvsp[-3].code)),
           (yyvsp[-2].code), (yyvsp[-1].code), code_new_bits((yyvsp[0].bits)));
     }
#line 3166 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 82:
#line 526 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
        (yyval.code) = code_new_inst(i_constraint, 4,
           code_new_inst(i_term_expr, 1, (yyvsp[-3].code)),
           (yyvsp[-2].code),
           code_new_inst(i_term_expr, 1, (yyvsp[-1].code)),
           code_new_bits((yyvsp[0].bits)));
     }
#line 3178 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 83:
#line 533 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_rangeconst, 6, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code), (yyvsp[-4].code),
           code_new_contype(CON_RHS), code_new_bits((yyvsp[0].bits))); 
     }
#line 3187 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 84:
#line 537 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_rangeconst, 6, (yyvsp[-5].code),
           code_new_inst(i_term_expr, 1, (yyvsp[-3].code)), (yyvsp[-1].code), (yyvsp[-4].code),
           code_new_contype(CON_RHS), code_new_bits((yyvsp[0].bits))); 
     }
#line 3197 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 85:
#line 542 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_rangeconst, 6, (yyvsp[-1].code), (yyvsp[-3].code), (yyvsp[-5].code), (yyvsp[-4].code),
           code_new_contype(CON_LHS), code_new_bits((yyvsp[0].bits))); 
     }
#line 3206 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 86:
#line 546 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_rangeconst, 6, (yyvsp[-1].code),
           code_new_inst(i_term_expr, 1, (yyvsp[-3].code)),
           (yyvsp[-5].code), (yyvsp[-4].code),
           code_new_contype(CON_LHS), code_new_bits((yyvsp[0].bits))); 
     }
#line 3217 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 87:
#line 552 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code), (yyvsp[-6].code), (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3225 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 88:
#line 555 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code), (yyvsp[-6].code), (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3235 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 89:
#line 560 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3245 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 90:
#line 565 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code), (yyvsp[-6].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3255 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 91:
#line 570 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code), (yyvsp[-6].code), (yyvsp[-4].code), (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), code_new_bits((yyvsp[0].bits)));
      }
#line 3264 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 92:
#line 574 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3276 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 93:
#line 581 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code), (yyvsp[-6].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3288 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 94:
#line 588 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code), (yyvsp[-6].code), (yyvsp[-4].code), (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), code_new_bits((yyvsp[0].bits)));
      }
#line 3299 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 95:
#line 594 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3310 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 96:
#line 600 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            (yyvsp[-4].code), (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), code_new_bits((yyvsp[0].bits)));
      }
#line 3321 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 97:
#line 606 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code), (yyvsp[-6].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)), (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), code_new_bits((yyvsp[0].bits)));
      }
#line 3331 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 98:
#line 611 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3344 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 99:
#line 619 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            (yyvsp[-4].code), (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), 
            code_new_bits((yyvsp[0].bits)));
      }
#line 3358 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 100:
#line 628 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code), (yyvsp[-6].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), 
            code_new_bits((yyvsp[0].bits)));
      }
#line 3372 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 101:
#line 637 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code), (yyvsp[-8].code), (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), 
            code_new_bits((yyvsp[0].bits)));
      }
#line 3385 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 102:
#line 645 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif_else, 8, (yyvsp[-10].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-8].code)),
            (yyvsp[-7].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-6].code)),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)),
            (yyvsp[-3].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), 
            code_new_bits((yyvsp[0].bits)));
      }
#line 3400 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 103:
#line 656 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vif, 5, (yyvsp[-6].code), (yyvsp[-4].code), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3408 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 104:
#line 659 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vif, 5, (yyvsp[-6].code), 
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)), (yyvsp[-3].code), (yyvsp[-2].code), code_new_bits((yyvsp[0].bits)));
      }
#line 3417 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 105:
#line 663 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vif, 5, (yyvsp[-6].code), 
            (yyvsp[-4].code), (yyvsp[-3].code), code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), 
            code_new_bits((yyvsp[0].bits)));
      }
#line 3427 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 106:
#line 668 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { /* ??? This is an error */
         (yyval.code) = code_new_inst(i_vif, 5, (yyvsp[-6].code),
            code_new_inst(i_term_expr, 1, (yyvsp[-4].code)), (yyvsp[-3].code), 
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), code_new_bits((yyvsp[0].bits)));
      }
#line 3437 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 107:
#line 676 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_ne, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3443 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 108:
#line 677 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_ne, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3451 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 109:
#line 680 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_ne, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3459 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 110:
#line 683 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_eq, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3465 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 111:
#line 684 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_eq, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3473 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 112:
#line 687 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_eq, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3481 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 113:
#line 690 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_le, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3487 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 114:
#line 691 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_le, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3495 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 115:
#line 694 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_le, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3503 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 116:
#line 697 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_ge, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3509 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 117:
#line 698 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_ge, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3517 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 118:
#line 701 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_ge, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3525 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 119:
#line 704 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_lt, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3531 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 120:
#line 705 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_lt, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3539 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 121:
#line 708 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_lt, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3547 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 122:
#line 711 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_gt, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3553 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 123:
#line 712 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_gt, 2, code_new_inst(i_term_expr, 1, (yyvsp[-2].code)), (yyvsp[0].code));
      }
#line 3561 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 124:
#line 715 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vbool_gt, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3569 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 125:
#line 718 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_and, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3575 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 126:
#line 719 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_or,  2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3581 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 127:
#line 720 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_xor, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3587 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 128:
#line 721 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vbool_not, 1, (yyvsp[0].code)); }
#line 3593 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 129:
#line 722 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 3599 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 130:
#line 726 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = 0; }
#line 3605 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 131:
#line 727 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = (yyvsp[-2].bits) | (yyvsp[0].bits); }
#line 3611 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 132:
#line 731 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_SCALE; }
#line 3617 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 133:
#line 732 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_SEPAR; }
#line 3623 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 134:
#line 733 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_CHECK; }
#line 3629 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 135:
#line 734 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_INDIC; }
#line 3635 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 136:
#line 735 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_QUBO;  }
#line 3641 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 137:
#line 736 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY1; }
#line 3647 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 138:
#line 737 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY2; }
#line 3653 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 139:
#line 738 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY3; }
#line 3659 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 140:
#line 739 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY4; }
#line 3665 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 141:
#line 740 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY5; }
#line 3671 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 142:
#line 741 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.bits) = LP_FLAG_CON_PENALTY6; }
#line 3677 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 143:
#line 745 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_contype(CON_RHS); }
#line 3683 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 144:
#line 746 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_contype(CON_LHS); }
#line 3689 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 145:
#line 747 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_contype(CON_EQUAL); }
#line 3695 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 146:
#line 751 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3701 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 147:
#line 752 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_add, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3707 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 148:
#line 753 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_sub, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3713 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 149:
#line 754 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_const, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3719 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 150:
#line 755 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_term_sub, 2, (yyvsp[-2].code), code_new_inst(i_term_expr, 1, (yyvsp[0].code)));
      }
#line 3727 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 151:
#line 758 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_const, 2, (yyvsp[0].code), (yyvsp[-2].code)); }
#line 3733 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 152:
#line 759 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_term_sub, 2,
            code_new_inst(i_term_expr, 1, (yyvsp[-2].code)),
            (yyvsp[0].code));
      }
#line 3743 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 153:
#line 767 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3749 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 154:
#line 768 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_coeff, 2, (yyvsp[-2].code), (yyvsp[0].code));  }
#line 3755 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 155:
#line 769 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_term_coeff, 2, (yyvsp[-2].code),
            code_new_inst(i_expr_div, 2, code_new_numb(numb_new_integer(1)), (yyvsp[0].code)));
      }
#line 3764 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 156:
#line 773 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_coeff, 2, (yyvsp[0].code), (yyvsp[-2].code)); }
#line 3770 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 157:
#line 774 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_term_mul, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 3776 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 159:
#line 779 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3782 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 160:
#line 780 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_term_coeff, 2, (yyvsp[0].code), code_new_numb(numb_new_integer(-1)));
      }
#line 3790 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 161:
#line 786 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 3796 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 162:
#line 787 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_term_power, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 3804 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 163:
#line 790 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_term_sum, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 3812 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 164:
#line 796 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_symbol_deref, 2, code_new_symbol((yyvsp[-1].sym)), (yyvsp[0].code));
      }
#line 3820 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 165:
#line 799 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vabs, 1, (yyvsp[-1].code)); }
#line 3826 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 166:
#line 800 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(-2)), (yyvsp[-1].code)); }
#line 3832 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 167:
#line 801 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(3)), (yyvsp[-1].code)); }
#line 3838 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 168:
#line 802 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(4)), (yyvsp[-1].code)); }
#line 3844 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 169:
#line 803 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(5)), (yyvsp[-1].code)); }
#line 3850 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 170:
#line 804 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(6)), (yyvsp[-1].code)); }
#line 3856 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 171:
#line 805 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(7)), (yyvsp[-1].code)); }
#line 3862 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 172:
#line 806 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(8)), (yyvsp[-1].code)); }
#line 3868 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 173:
#line 807 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(9)), (yyvsp[-1].code)); }
#line 3874 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 174:
#line 808 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(10)), (yyvsp[-1].code)); }
#line 3880 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 175:
#line 809 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vexpr_fun, 3, code_new_numb(numb_new_integer(11)), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 3888 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 176:
#line 812 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_vexpr_fun, 3, code_new_numb(numb_new_integer(12)), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 3896 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 177:
#line 815 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 3904 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 178:
#line 818 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 3910 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 179:
#line 826 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_sos, 2, code_new_name((yyvsp[-3].name)), (yyvsp[-1].code));
     }
#line 3918 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 180:
#line 832 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_soset, 3, (yyvsp[0].code), (yyvsp[-3].code), (yyvsp[-2].code));
     }
#line 3926 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 181:
#line 835 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_forall, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 3934 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 182:
#line 841 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb(numb_new_integer(1)); }
#line 3940 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 183:
#line 842 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb(numb_new_integer(1)); }
#line 3946 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 184:
#line 843 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb(numb_new_integer(2)); }
#line 3952 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 185:
#line 851 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 3958 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 186:
#line 855 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_print, 1, (yyvsp[0].code)); }
#line 3964 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 187:
#line 856 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_print, 1, (yyvsp[0].code)); }
#line 3970 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 188:
#line 857 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_print, 1, (yyvsp[0].code)); }
#line 3976 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 189:
#line 858 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_print, 1, (yyvsp[0].code)); }
#line 3982 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 190:
#line 859 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_print, 1, code_new_symbol((yyvsp[0].sym))); }
#line 3988 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 191:
#line 860 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_check, 1, (yyvsp[0].code)); }
#line 3994 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 192:
#line 861 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_forall, 2, (yyvsp[-2].code), (yyvsp[0].code));
     }
#line 4002 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 193:
#line 871 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 4008 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 194:
#line 872 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_idxset_new, 3,
            code_new_inst(i_tuple_empty, 0), (yyvsp[0].code), code_new_inst(i_bool_true, 0));
      }
#line 4017 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 195:
#line 879 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_idxset_new, 3, (yyvsp[-4].code), (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4025 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 196:
#line 882 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_idxset_new, 3, (yyvsp[-2].code), (yyvsp[0].code), code_new_inst(i_bool_true, 0));
      }
#line 4033 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 198:
#line 889 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_union, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4039 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 199:
#line 890 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_union, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4047 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 200:
#line 893 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_sdiff, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4053 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 201:
#line 894 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_minus, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4061 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 202:
#line 897 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_minus, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4067 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 203:
#line 898 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_inter, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4073 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 205:
#line 902 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_union2, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4079 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 207:
#line 907 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_cross, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4085 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 208:
#line 908 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_cross, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4093 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 209:
#line 911 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_inter2, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4099 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 210:
#line 915 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_symbol_deref, 2, code_new_symbol((yyvsp[-1].sym)), (yyvsp[0].code));
      }
#line 4107 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 211:
#line 918 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_define_deref, 2,
            code_new_define((yyvsp[-3].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-1].code)));
      }
#line 4117 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 212:
#line 923 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_empty, 1, code_new_size(0)); }
#line 4123 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 213:
#line 924 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_range2, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4131 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 214:
#line 927 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_range2, 3, (yyvsp[-3].code), (yyvsp[-1].code), code_new_numb(numb_new_integer(1)));
      }
#line 4139 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 215:
#line 930 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_range, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4147 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 216:
#line 933 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_range, 3, (yyvsp[-3].code), (yyvsp[-1].code), code_new_numb(numb_new_integer(1)));
      }
#line 4155 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 217:
#line 936 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_argmin, 3, code_new_numb(numb_new_integer(1)), (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4163 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 218:
#line 939 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_argmin, 3, (yyvsp[-4].code), (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4171 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 219:
#line 942 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_argmax, 3, code_new_numb(numb_new_integer(1)), (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4179 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 220:
#line 945 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_argmax, 3, (yyvsp[-4].code), (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4187 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 221:
#line 948 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 4193 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 222:
#line 949 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_new_tuple, 1, (yyvsp[-1].code)); }
#line 4199 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 223:
#line 950 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_new_elem, 1, (yyvsp[-1].code)); }
#line 4205 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 224:
#line 951 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_idxset, 1, (yyvsp[-1].code)); }
#line 4211 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 225:
#line 952 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_expr, 2, (yyvsp[-3].code), (yyvsp[-1].code)); }
#line 4217 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 226:
#line 953 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_set_expr, 2, (yyvsp[-3].code), (yyvsp[-1].code)); }
#line 4223 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 227:
#line 954 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_set_proj, 2, (yyvsp[-3].code), (yyvsp[-1].code));
       }
#line 4231 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 228:
#line 957 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
          (yyval.code) = code_new_inst(i_set_indexset, 1, code_new_symbol((yyvsp[-1].sym)));
       }
#line 4239 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 229:
#line 960 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4247 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 230:
#line 966 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_new, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4253 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 231:
#line 967 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_param, 2, (yyvsp[-1].code), (yyvsp[0].code)); }
#line 4259 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 232:
#line 971 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_skip, 1, (yyvsp[0].code)); }
#line 4265 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 233:
#line 972 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_use, 1, (yyvsp[0].code)); }
#line 4271 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 234:
#line 973 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_comment, 1, (yyvsp[0].code)); }
#line 4277 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 235:
#line 974 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read_match, 1, (yyvsp[0].code)); }
#line 4283 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 236:
#line 978 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_tuple_list_new, 1, (yyvsp[0].code));
      }
#line 4291 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 237:
#line 981 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_tuple_list_add, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4299 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 238:
#line 984 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_read, 1, (yyvsp[0].code)); }
#line 4305 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 239:
#line 988 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_eq, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4311 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 240:
#line 989 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_ne, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4317 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 241:
#line 990 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_gt, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4323 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 242:
#line 991 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_ge, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4329 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 243:
#line 992 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_lt, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4335 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 244:
#line 993 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_le, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4341 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 245:
#line 994 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_seq, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4347 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 246:
#line 995 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_sneq, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4353 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 247:
#line 996 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_subs, 2, (yyvsp[0].code), (yyvsp[-2].code)); }
#line 4359 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 248:
#line 997 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_sseq, 2, (yyvsp[0].code), (yyvsp[-2].code)); }
#line 4365 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 249:
#line 998 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_subs, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4371 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 250:
#line 999 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_sseq, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4377 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 251:
#line 1000 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_and, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4383 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 252:
#line 1001 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_or,  2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4389 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 253:
#line 1002 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_xor, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4395 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 254:
#line 1003 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_not, 1, (yyvsp[0].code)); }
#line 4401 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 255:
#line 1004 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 4407 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 256:
#line 1005 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_is_elem, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4413 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 257:
#line 1006 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_bool_exists, 1, (yyvsp[-1].code)); }
#line 4419 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 258:
#line 1007 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_define_deref, 2,
            code_new_define((yyvsp[-3].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-1].code)));
      }
#line 4429 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 259:
#line 1012 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
        (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
     }
#line 4437 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 260:
#line 1018 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_tuple_empty, 0); }
#line 4443 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 261:
#line 1019 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_tuple_new, 1, (yyvsp[-1].code));  }
#line 4449 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 262:
#line 1023 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_tuple_empty, 0);
      }
#line 4457 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 263:
#line 1026 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_tuple_new, 1, (yyvsp[-1].code));
      }
#line 4465 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 264:
#line 1032 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_elem_list_new, 1, (yyvsp[0].code));
      }
#line 4473 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 265:
#line 1035 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_elem_list_add, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4481 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 266:
#line 1041 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 4487 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 267:
#line 1042 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_add, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4493 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 268:
#line 1043 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_sub, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4499 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 269:
#line 1047 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 4505 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 270:
#line 1048 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_mul, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4511 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 271:
#line 1049 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_div, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4517 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 272:
#line 1050 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_mod, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4523 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 273:
#line 1051 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_intdiv, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4529 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 274:
#line 1052 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_prod, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4537 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 276:
#line 1059 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[0].code); }
#line 4543 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 277:
#line 1060 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_neg, 1, (yyvsp[0].code)); }
#line 4549 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1065 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_pow, 2, (yyvsp[-2].code), (yyvsp[0].code)); }
#line 4555 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1066 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_sum, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4563 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1069 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_min, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4571 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1072 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_max, 2, (yyvsp[-2].code), (yyvsp[0].code));
      }
#line 4579 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1075 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_sglmin, 1, (yyvsp[-1].code));
         }
#line 4587 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1078 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_sglmax, 1, (yyvsp[-1].code));
      }
#line 4595 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1084 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_numb((yyvsp[0].numb)); }
#line 4601 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1085 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_strg((yyvsp[0].strg));  }
#line 4607 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1086 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_local_deref, 1, code_new_name((yyvsp[0].name)));
      }
#line 4615 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1089 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_symbol_deref, 2, code_new_symbol((yyvsp[-1].sym)), (yyvsp[0].code));
      }
#line 4623 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1092 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { 
         (yyval.code) = code_new_inst(i_symbol_deref, 2, code_new_symbol((yyvsp[-1].sym)), (yyvsp[0].code));
      }
#line 4631 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1095 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_define_deref, 2,
            code_new_define((yyvsp[-3].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-1].code)));
      }
#line 4641 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1100 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_define_deref, 2,
            code_new_define((yyvsp[-3].def)),
            code_new_inst(i_tuple_new, 1, (yyvsp[-1].code)));
      }
#line 4651 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1105 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_fac, 1, (yyvsp[-1].code)); }
#line 4657 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1106 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_card, 1, (yyvsp[-1].code)); }
#line 4663 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1107 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_abs, 1, (yyvsp[-1].code)); }
#line 4669 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1108 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_sgn, 1, (yyvsp[-1].code)); }
#line 4675 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1109 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_round, 1, (yyvsp[-1].code)); }
#line 4681 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1110 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_floor, 1, (yyvsp[-1].code)); }
#line 4687 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1111 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_ceil, 1, (yyvsp[-1].code)); }
#line 4693 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1112 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_log, 1, (yyvsp[-1].code)); }
#line 4699 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1113 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_ln, 1, (yyvsp[-1].code)); }
#line 4705 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1114 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_exp, 1, (yyvsp[-1].code)); }
#line 4711 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1115 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_sqrt, 1, (yyvsp[-1].code)); }
#line 4717 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1116 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_sin, 1, (yyvsp[-1].code)); }
#line 4723 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1117 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_cos, 1, (yyvsp[-1].code)); }
#line 4729 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1118 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_tan, 1, (yyvsp[-1].code)); }
#line 4735 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1119 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_asin, 1, (yyvsp[-1].code)); }
#line 4741 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1120 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_acos, 1, (yyvsp[-1].code)); }
#line 4747 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1121 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_atan, 1, (yyvsp[-1].code)); }
#line 4753 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1123 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = (yyvsp[-1].code); }
#line 4759 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1124 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    { (yyval.code) = code_new_inst(i_expr_length, 1, (yyvsp[-1].code)); }
#line 4765 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1125 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_substr, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4773 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1128 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_rand, 2, (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4781 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1131 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_if_else, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4789 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1134 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_ord, 3, (yyvsp[-5].code), (yyvsp[-3].code), (yyvsp[-1].code));
      }
#line 4797 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1137 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_min2, 1, (yyvsp[-1].code));
      }
#line 4805 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1140 "src/zimpl/mmlparse2.y" /* yacc.c:1646  */
    {
         (yyval.code) = code_new_inst(i_expr_max2, 1, (yyvsp[-1].code));
      }
#line 4813 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
    break;


#line 4817 "src/zimpl/mmlparse2.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
