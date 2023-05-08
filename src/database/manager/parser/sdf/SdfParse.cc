/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.5.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         Sdf_parse
#define yylex           Sdf_lex
#define yyerror         Sdf_error
#define yydebug         Sdf_debug
#define yynerrs         Sdf_nerrs
#define yylval          Sdf_lval
#define yychar          Sdf_char

/* First part of user prologue.  */
#line 1 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"



#include <ctype.h>

#include "SdfReader.hh"
#include "string/Str.hh"
#include "log/Log.hh"

using namespace ista;

int Sdf_lex();

extern int g_sdf_line;
SdfReader* g_sdf_reader = nullptr;

// use yacc generated parser errors
#define YYERROR_VERBOSE

void yyerror(const char* s);


#line 115 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
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

#line 223 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE Sdf_lval;

int Sdf_parse (void);

#endif /* !YY_SDF_PROJECT_HOME_CHENSHIJIAN_IEDA_SRC_ISTA_SDF_PARSER_SDFPARSE_HH_INCLUDED  */

/* Second part of user prologue.  */
#line 73 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"


#line 242 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))

/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

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

#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
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

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

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
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
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
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   241

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  40
/* YYNRULES -- Number of rules.  */
#define YYNRULES  94
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  241

#define YYUNDEFTOK  2
#define YYMAXUTOK   302


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      48,    49,    52,     2,     2,     2,    51,    50,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    53,     2,
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
      45,    46,    47
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    79,    79,    83,    84,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   109,   111,   115,   116,   120,   121,   125,   130,   135,
     136,   137,   141,   143,   147,   149,   160,   161,   166,   165,
     169,   168,   173,   174,   178,   179,   183,   185,   188,   191,
     193,   195,   197,   203,   204,   208,   213,   220,   231,   232,
     247,   247,   261,   261,   275,   275,   291,   291,   305,   305,
     319,   319,   335,   335,   350,   350,   364,   364,   378,   378,
     394,   398,   403,   405,   410,   411,   416,   418,   423,   431,
     436,   442,   446,   458,   470
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "DELAYFILE", "SDFVERSION", "DESIGN",
  "DATE", "VENDOR", "PROGRAM", "PVERSION", "DIVIDER", "VOLTAGE", "PROCESS",
  "TEMPERATURE", "TIMESCALE", "CELL", "CELLTYPE", "INSTANCE", "DELAY",
  "ABSOLUTE", "INCREMENTAL", "INTERCONNECT", "PORT", "DEVICE", "RETAIN",
  "IOPATH", "TIMINGCHECK", "SETUP", "HOLD", "SETUPHOLD", "RECOVERY",
  "REMOVAL", "RECREM", "WIDTH", "PERIOD", "SKEW", "NOCHANGE", "POSEDGE",
  "NEGEDGE", "COND", "CONDELSE", "QSTRING", "ID", "PATH", "NUMBER",
  "EXPR_OPEN_IOPATH", "EXPR_OPEN", "EXPR_ID_CLOSE", "'('", "')'", "'/'",
  "'.'", "'*'", "':'", "$accept", "file", "header", "header_stmt", "hchar",
  "number_opt", "cells", "cell", "celltype", "cell_instance",
  "timing_specs", "timing_spec", "deltypes", "deltype", "$@1", "$@2",
  "del_defs", "path", "del_def", "retains", "retain", "delval_list",
  "tchk_defs", "tchk_def", "$@3", "$@4", "$@5", "$@6", "$@7", "$@8", "$@9",
  "$@10", "$@11", "$@12", "port_instance", "port_spec", "port_transition",
  "port_tchk", "value", "triple", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,    40,    41,
      47,    46,    42,    58
};
# endif

#define YYPACT_NINF (-181)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -37,    23,    48,     3,  -181,   133,    16,  -181,    54,    56,
      58,   107,   128,   130,    36,    26,   -21,    27,    70,   121,
    -181,    40,  -181,    35,   125,   132,   139,   140,   141,  -181,
    -181,   142,     8,  -181,   117,   143,   144,  -181,    28,  -181,
     145,   153,   148,   182,  -181,  -181,  -181,  -181,  -181,  -181,
    -181,  -181,  -181,  -181,   154,   155,  -181,  -181,  -181,  -181,
     151,   185,   156,  -181,   149,   150,   152,  -181,   165,   190,
    -181,   154,   154,   164,   160,    13,    43,  -181,  -181,  -181,
    -181,  -181,  -181,  -181,   161,   162,    12,  -181,  -181,  -181,
    -181,  -181,  -181,    45,    62,    93,  -181,  -181,   122,  -181,
    -181,  -181,  -181,  -181,  -181,  -181,  -181,  -181,  -181,  -181,
    -181,  -181,  -181,  -181,  -181,   -19,   -19,   -19,   -19,   -19,
     -19,   -19,   -19,   -19,   -19,    73,   111,  -181,   -29,  -181,
     -19,   -19,   -19,   -19,   -19,   -19,   166,   166,   -19,   -19,
      20,  -181,  -181,  -181,  -181,  -181,   115,   170,   166,   166,
     166,   166,   166,   166,    30,   167,   168,   166,   166,   123,
     123,    25,    -2,   173,   171,   126,  -181,   172,   174,   175,
     166,   176,   177,   166,    29,  -181,   178,  -181,  -181,   179,
     166,  -181,  -181,   123,   166,   119,   166,  -181,   126,   123,
      -2,   188,   180,  -181,  -181,  -181,   181,  -181,  -181,   183,
    -181,  -181,  -181,   184,   166,   124,  -181,  -181,   127,  -181,
     123,    -2,   186,  -181,  -181,  -181,   129,  -181,  -181,   189,
    -181,   123,   187,  -181,   -17,  -181,   131,   189,  -181,  -181,
     166,  -181,   134,   189,   136,   191,   138,  -181,  -181,   192,
    -181
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     0,     1,     0,     0,     3,     0,     0,
       0,     0,     0,     0,     0,    23,     0,    23,     0,     0,
       4,     0,    25,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    14,     0,     0,     0,    16,     0,    19,
       0,     0,     0,     0,     2,    26,     5,     6,     7,     8,
       9,    10,    11,    13,    23,    23,    12,    15,    17,    18,
       0,     0,     0,    24,     0,     0,     0,    20,     0,     0,
      32,    23,    23,     0,     0,     0,     0,    92,    93,    94,
      28,    44,    45,    29,     0,     0,     0,    27,    33,    30,
      31,    36,    58,     0,     0,     0,    34,    37,     0,    35,
      59,    38,    40,    60,    62,    64,    66,    68,    70,    74,
      76,    72,    78,    42,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    82,     0,    86,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    39,    43,    41,    84,    85,     0,     0,     0,     0,
       0,     0,     0,     0,    23,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    87,     0,     0,     0,
       0,     0,     0,     0,     0,    89,     0,    75,    77,     0,
       0,    80,    81,     0,     0,     0,     0,    56,     0,     0,
       0,     0,     0,    83,    61,    63,     0,    67,    69,     0,
      90,    91,    73,     0,     0,     0,    51,    57,     0,    53,
       0,     0,     0,    65,    71,    79,     0,    50,    52,     0,
      53,     0,     0,    49,    23,    54,     0,     0,    53,    88,
       0,    46,     0,     0,     0,     0,     0,    55,    48,     0,
      47
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -181,  -181,  -181,   209,  -181,   -36,  -181,   199,  -181,  -181,
    -181,  -181,  -181,  -181,  -181,  -181,   120,  -181,  -181,  -162,
    -181,  -180,  -181,  -181,  -181,  -181,  -181,  -181,  -181,  -181,
    -181,  -181,  -181,  -181,  -158,  -157,    64,   -15,  -136,    68
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     2,     6,     7,    31,    34,    21,    22,    62,    70,
      76,    88,    93,    97,   113,   114,   125,    85,   142,   219,
     225,   185,    94,   100,   115,   116,   117,   118,   119,   120,
     123,   121,   122,   124,   183,   129,   147,   130,   187,   176
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
     155,   156,   184,   186,   205,   189,   208,   230,   144,   145,
     146,     1,   168,   169,   170,   171,   172,   173,    64,    66,
      36,   179,   180,   127,   216,   204,     3,   174,    37,   128,
      91,   209,   175,   210,   196,    77,    78,   199,    92,   226,
     127,   159,   160,   161,   203,   162,   188,   232,     4,   207,
     234,     5,   220,   236,   221,    81,    82,    53,   227,   163,
     164,    54,    83,   228,    19,    84,   233,   181,   182,   207,
      32,    38,   207,   154,   174,    33,    39,    58,   200,   175,
     207,    54,    54,    35,    46,    40,    29,    30,    43,    44,
     207,    86,    87,    95,    96,    23,   207,    24,   207,    25,
     207,   131,   132,   133,   134,   135,   136,   137,   138,   139,
      98,    99,   101,   102,    41,   148,   149,   150,   151,   152,
     153,   140,   141,   157,   158,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    42,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    26,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   140,
     143,   165,   166,   144,   145,   181,   182,   154,   206,    27,
      55,    28,   154,   217,    47,   154,   218,   154,   223,   154,
     231,    48,   154,   235,   154,   237,   154,   239,    49,    50,
      51,    52,    56,    57,    59,    60,    61,    42,    63,    65,
      67,    68,    71,    72,    69,    73,    74,    75,    79,    80,
      89,    90,   167,   211,   154,    20,   177,   178,   190,   191,
      45,   193,   212,   194,   195,   197,   198,   201,   202,   192,
     213,     0,   214,   215,   126,   222,   229,   224,     0,     0,
     238,   240
};

static const yytype_int16 yycheck[] =
{
     136,   137,   160,   161,   184,   162,   186,    24,    37,    38,
      39,    48,   148,   149,   150,   151,   152,   153,    54,    55,
      41,   157,   158,    42,   204,   183,     3,    44,    49,    48,
      18,   189,    49,   190,   170,    71,    72,   173,    26,   219,
      42,    21,    22,    23,   180,    25,    48,   227,     0,   185,
     230,    48,   210,   233,   211,    42,    43,    49,   220,    39,
      40,    53,    49,   221,    48,    52,   228,    42,    43,   205,
      44,    44,   208,    48,    44,    49,    49,    49,    49,    49,
     216,    53,    53,    15,    49,    17,    50,    51,    48,    49,
     226,    48,    49,    48,    49,    41,   232,    41,   234,    41,
     236,   116,   117,   118,   119,   120,   121,   122,   123,   124,
      48,    49,    19,    20,    44,   130,   131,   132,   133,   134,
     135,    48,    49,   138,   139,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    41,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    48,
      49,    46,    47,    37,    38,    42,    43,    48,    49,    41,
      53,    41,    48,    49,    49,    48,    49,    48,    49,    48,
      49,    49,    48,    49,    48,    49,    48,    49,    49,    49,
      49,    49,    49,    49,    49,    42,    48,    15,    44,    44,
      49,    16,    53,    53,    48,    53,    41,    17,    44,    49,
      49,    49,    42,    25,    48,     6,    49,    49,    45,    48,
      21,    49,    42,    49,    49,    49,    49,    49,    49,   165,
      49,    -1,    49,    49,   114,    49,    49,    48,    -1,    -1,
      49,    49
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    48,    55,     3,     0,    48,    56,    57,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    48,
      57,    60,    61,    41,    41,    41,    41,    41,    41,    50,
      51,    58,    44,    49,    59,    93,    41,    49,    44,    49,
      93,    44,    15,    48,    49,    61,    49,    49,    49,    49,
      49,    49,    49,    49,    53,    53,    49,    49,    49,    49,
      42,    48,    62,    44,    59,    44,    59,    49,    16,    48,
      63,    53,    53,    53,    41,    17,    64,    59,    59,    44,
      49,    42,    43,    49,    52,    71,    48,    49,    65,    49,
      49,    18,    26,    66,    76,    48,    49,    67,    48,    49,
      77,    19,    20,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    68,    69,    78,    79,    80,    81,    82,
      83,    85,    86,    84,    87,    70,    70,    42,    48,    89,
      91,    91,    91,    91,    91,    91,    91,    91,    91,    91,
      48,    49,    72,    49,    37,    38,    39,    90,    91,    91,
      91,    91,    91,    91,    48,    92,    92,    91,    91,    21,
      22,    23,    25,    39,    40,    46,    47,    42,    92,    92,
      92,    92,    92,    92,    44,    49,    93,    49,    49,    92,
      92,    42,    43,    88,    88,    75,    88,    92,    48,    89,
      45,    48,    90,    49,    49,    49,    92,    49,    49,    92,
      49,    49,    49,    92,    88,    75,    49,    92,    75,    88,
      89,    25,    42,    49,    49,    49,    75,    49,    49,    73,
      88,    89,    49,    49,    48,    74,    75,    73,    88,    49,
      24,    49,    75,    73,    75,    49,    75,    49,    49,    49,
      49
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    54,    55,    56,    56,    57,    57,    57,    57,    57,
      57,    57,    57,    57,    57,    57,    57,    57,    57,    57,
      57,    58,    58,    59,    59,    60,    60,    61,    62,    63,
      63,    63,    64,    64,    65,    65,    66,    66,    68,    67,
      69,    67,    70,    70,    71,    71,    72,    72,    72,    72,
      72,    72,    72,    73,    73,    74,    75,    75,    76,    76,
      78,    77,    79,    77,    80,    77,    81,    77,    82,    77,
      83,    77,    84,    77,    85,    77,    86,    77,    87,    77,
      88,    88,    89,    89,    90,    90,    91,    91,    91,    92,
      92,    92,    93,    93,    93
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     5,     1,     2,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     3,     4,     3,     4,     4,     3,
       5,     1,     1,     0,     1,     1,     2,     6,     4,     3,
       4,     4,     0,     2,     4,     4,     0,     2,     0,     5,
       0,     5,     0,     2,     1,     1,     7,    10,     9,     6,
       5,     4,     5,     0,     2,     4,     1,     2,     0,     2,
       0,     7,     0,     7,     0,     8,     0,     7,     0,     7,
       0,     8,     0,     7,     0,     6,     0,     6,     0,     8,
       1,     1,     1,     4,     1,     1,     1,     3,     7,     2,
       3,     3,     5,     5,     5
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
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


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yytype], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyo, yytype, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
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
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[+yyssp[yyi + 1 - yynrhs]],
                       &yyvsp[(yyi + 1) - (yynrhs)]
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
#   define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
#  else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
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
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
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
            else
              goto append;

          append:
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

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
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
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                yy_state_t *yyssp, int yytoken)
{
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Actual size of YYARG. */
  int yycount = 0;
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

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
      int yyn = yypact[+*yyssp];
      YYPTRDIFF_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
      yysize = yysize0;
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
                  YYPTRDIFF_T yysize1
                    = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
                    yysize = yysize1;
                  else
                    return 2;
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
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    YYPTRDIFF_T yysize1 = yysize + (yystrlen (yyformat) - 2 * yycount) + 1;
    if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
      yysize = yysize1;
    else
      return 2;
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
          ++yyp;
          ++yyformat;
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




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYPTRDIFF_T yystacksize;

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
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;
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
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
# undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

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
      yychar = yylex ();
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
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
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
| yyreduce -- do a reduction.  |
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
#line 79 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                                    {}
#line 1566 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 5:
#line 89 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                               { Str::free((char*)(yyvsp[-1].string)); }
#line 1572 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 6:
#line 90 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                           { Str::free((char*)(yyvsp[-1].string)); }
#line 1578 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 7:
#line 91 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                         { Str::free((char*)(yyvsp[-1].string)); }
#line 1584 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 8:
#line 92 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                           { Str::free((char*)(yyvsp[-1].string)); }
#line 1590 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 9:
#line 93 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                            { Str::free((char*)(yyvsp[-1].string)); }
#line 1596 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 10:
#line 94 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                             { Str::free((char*)(yyvsp[-1].string)); }
#line 1602 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 11:
#line 95 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                          {  }
#line 1608 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 12:
#line 96 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                           {  }
#line 1614 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 15:
#line 99 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                            { Str::free((yyvsp[-1].string)); }
#line 1620 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 18:
#line 102 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                               {  }
#line 1626 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 20:
#line 105 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1632 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 21:
#line 110 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.character) = '/'; }
#line 1638 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 22:
#line 112 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.character) = '.'; }
#line 1644 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 23:
#line 115 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
            { (yyval.obj) = nullptr; }
#line 1650 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 24:
#line 116 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
             { (yyval.obj) = new float((yyvsp[0].number)); }
#line 1656 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 27:
#line 126 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1662 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 28:
#line 131 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1668 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 29:
#line 135 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                       {  }
#line 1674 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 30:
#line 136 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                           {  }
#line 1680 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 31:
#line 138 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1686 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 34:
#line 148 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
     { (yyval.obj) = nullptr; }
#line 1692 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 35:
#line 150 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {
        auto* tchk_defs = static_cast<std::vector<std::unique_ptr<SdfTimingCheckDef>>*>((yyvsp[-1].obj));
        auto* timing_spec = g_sdf_reader->makeTimingSpec(std::move(*tchk_defs));

        delete tchk_defs;

        (yyval.obj) = timing_spec;
    }
#line 1705 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 38:
#line 166 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1711 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 40:
#line 169 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1717 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 46:
#line 184 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1723 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 47:
#line 187 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1729 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 48:
#line 190 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1735 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 49:
#line 192 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1741 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 50:
#line 194 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1747 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 51:
#line 196 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1753 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 52:
#line 198 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1759 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 53:
#line 203 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = nullptr; }
#line 1765 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 55:
#line 209 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {  }
#line 1771 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 56:
#line 214 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
        auto* delay_values = new std::vector<std::unique_ptr<SdfTripleValue>>();
        auto* delay_value = static_cast<SdfTripleValue*>((yyvsp[0].obj));
        delay_values->emplace_back(delay_value);
        (yyval.obj) = delay_values; 
    }
#line 1782 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 57:
#line 221 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {   
        auto* delay_values = static_cast<std::vector<std::unique_ptr<SdfTripleValue>>*>((yyvsp[-1].obj));
        auto* delay_value = static_cast<SdfTripleValue*>((yyvsp[0].obj));
        delay_values->emplace_back(delay_value);
        (yyval.obj) = delay_values; 
    }
#line 1793 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 58:
#line 231 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = nullptr; }
#line 1799 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 59:
#line 233 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
        auto* timing_check_defs = static_cast<std::vector<std::unique_ptr<SdfTimingCheckDef>>*>((yyvsp[-1].obj));
        if (!timing_check_defs) {
            timing_check_defs = new std::vector<std::unique_ptr<SdfTimingCheckDef>>();
        }

        auto* timing_check_def = static_cast<SdfTimingCheckDef*>((yyvsp[0].obj));
        timing_check_defs->emplace_back(timing_check_def);

        (yyval.obj) = timing_check_defs;
    }
#line 1815 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 60:
#line 247 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
              { g_sdf_reader->set_parse_timing_check(true); }
#line 1821 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 61:
#line 249 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSetup, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 1838 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 62:
#line 261 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
             { g_sdf_reader->set_parse_timing_check(true); }
#line 1844 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 63:
#line 263 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kHold, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 1861 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 64:
#line 275 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                  { g_sdf_reader->set_parse_timing_check(true); }
#line 1867 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 65:
#line 277 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-4].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-2].obj));
      SdfTripleValue* value1 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSetupHold, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::move(*value1));
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      delete value1;
      (yyval.obj) = timing_check_def;
      
    }
#line 1886 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 66:
#line 291 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                 { g_sdf_reader->set_parse_timing_check(true); }
#line 1892 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 67:
#line 293 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRecovery, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 1909 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 68:
#line 305 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                { g_sdf_reader->set_parse_timing_check(true); }
#line 1915 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 69:
#line 307 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRemoval, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 1932 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 70:
#line 319 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
               { g_sdf_reader->set_parse_timing_check(true); }
#line 1938 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 71:
#line 321 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-4].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-2].obj));
      SdfTripleValue* value1 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRecRem, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::move(*value1));
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      delete value1;
      (yyval.obj) = timing_check_def;
      
    }
#line 1957 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 72:
#line 335 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
             { g_sdf_reader->set_parse_timing_check(true); }
#line 1963 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 73:
#line 338 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSkew, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 1980 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 74:
#line 350 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
              { g_sdf_reader->set_parse_timing_check(true); }
#line 1986 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 75:
#line 352 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kWidth, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 2003 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 76:
#line 364 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
               { g_sdf_reader->set_parse_timing_check(true); }
#line 2009 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 77:
#line 366 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-2].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-1].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kPeriod, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
      
    }
#line 2026 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 78:
#line 378 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                 { g_sdf_reader->set_parse_timing_check(true); }
#line 2032 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 79:
#line 380 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-4].obj));
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>((yyvsp[-3].obj));
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>((yyvsp[-2].obj));
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kNoChange, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      (yyval.obj) = timing_check_def;
    }
#line 2048 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 80:
#line 395 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
        (yyval.obj) = g_sdf_reader->makePortInstance(static_cast<const char*>((yyvsp[0].string)));
    }
#line 2056 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 81:
#line 399 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = g_sdf_reader->makePortInstance(static_cast<const char*>((yyvsp[0].string)));}
#line 2062 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 82:
#line 404 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = g_sdf_reader->makePortSpec(static_cast<const char*>((yyvsp[0].string))); }
#line 2068 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 83:
#line 406 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = g_sdf_reader->makePortSpec(static_cast<SdfPortSpec::TransitionType>((yyvsp[-2].integer)), static_cast<const char*>((yyvsp[-1].string))); }
#line 2074 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 84:
#line 410 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
              { (yyval.integer) = static_cast<int>(SdfPortSpec::TransitionType::kPOSEDGE); }
#line 2080 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 85:
#line 411 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
              { (yyval.integer) = static_cast<int>(SdfPortSpec::TransitionType::kNEGEDGE); }
#line 2086 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 86:
#line 417 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 2092 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 87:
#line 419 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
        /* (COND expr port) */
        (yyval.obj) = nullptr; 
    }
#line 2101 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 88:
#line 424 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    { 
        /* (COND expr (posedge port)) */
        (yyval.obj) = nullptr; 
    }
#line 2110 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 89:
#line 432 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {
      std::array<std::optional<float>, 3> triple_value;
      (yyval.obj) = g_sdf_reader->makeTriple(triple_value);
    }
#line 2119 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 90:
#line 437 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {
      float n = (float)(yyvsp[-1].number);
      std::array<std::optional<float>, 3> triple_value = {n, n, n};
      (yyval.obj) = g_sdf_reader->makeTriple(triple_value);
    }
#line 2129 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 91:
#line 442 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
                   { (yyval.obj) = (yyvsp[-1].obj); }
#line 2135 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 92:
#line 447 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {      
      float n1 = (float)(yyvsp[-4].number);
      std::optional<float> n2 = (yyvsp[-2].obj) ? std::optional<float>(*((float*)(yyvsp[-2].obj))) : std::nullopt; 
      std::optional<float> n3 = (yyvsp[0].obj) ? std::optional<float>(*((float*)(yyvsp[0].obj))) : std::nullopt; 

      delete (float*)(yyvsp[-2].obj);
      delete (float*)(yyvsp[0].obj);

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      (yyval.obj) = g_sdf_reader->makeTriple(triple_value);
    }
#line 2151 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 93:
#line 459 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {
      std::optional<float> n1 = (yyvsp[-4].obj) ? std::optional<float>(*((float*)(yyvsp[-4].obj))) : std::nullopt; 
      float n2 = (float)(yyvsp[-2].number);      
      std::optional<float> n3 = (yyvsp[0].obj) ? std::optional<float>(*((float*)(yyvsp[0].obj))) : std::nullopt; 

      delete (float*)(yyvsp[-4].obj);
      delete (float*)(yyvsp[0].obj);

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      (yyval.obj) = g_sdf_reader->makeTriple(triple_value);
    }
#line 2167 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;

  case 94:
#line 471 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"
    {      
      std::optional<float> n1 = (yyvsp[-4].obj) ? std::optional<float>(*((float*)(yyvsp[-4].obj))) : std::nullopt; 
      std::optional<float> n2 = (yyvsp[-2].obj) ? std::optional<float>(*((float*)(yyvsp[-2].obj))) : std::nullopt; 
      float n3 = (float)(yyvsp[0].number);

      delete (float*)(yyvsp[-4].obj);
      delete (float*)(yyvsp[-2].obj);

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      (yyval.obj) = g_sdf_reader->makeTriple(triple_value);
    }
#line 2183 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"
    break;


#line 2187 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.cc"

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
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

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
            yymsg = YY_CAST (char *, YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
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
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

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


/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
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
                  yystos[+*yyssp], yyvsp);
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
#line 484 "/Project/home/chenshijian/iEDA/src/iSTA/sdf-parser/SdfParse.y"


// Global namespace

void sdfFlushBuffer();

void yyerror(const char *msg) {

  sdfFlushBuffer();

  LOG_ERROR << "error line " << g_sdf_line << " " << msg;

}
