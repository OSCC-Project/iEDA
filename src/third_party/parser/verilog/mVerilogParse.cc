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
#define YYPURE 2

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         VERILOG_STYPE
/* Substitute the variable and function names.  */
#define yyparse         verilog_parse
#define yylex           verilog_lex
#define yyerror         verilog_error
#define yydebug         verilog_debug
#define yynerrs         verilog_nerrs


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
#ifndef YY_VERILOG_HOME_LONGSHUAIYING_IEDA_SRC_THIRD_PARTY_PARSER_VERILOG_MVERILOGPARSE_HH_INCLUDED
# define YY_VERILOG_HOME_LONGSHUAIYING_IEDA_SRC_THIRD_PARTY_PARSER_VERILOG_MVERILOGPARSE_HH_INCLUDED
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
#line 1 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"

// OpenSTA, Static Timing Analyzer
// Copyright (c) 2021, Parallax Software, Inc.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <vector>

#include "log/Log.hh"
#include "string/Str.hh"
#include "VerilogReader.hh"

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;


#line 156 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"

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
#line 40 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"

  int integer;
  char* string;
  const char* constant;
  void*  obj;

#line 197 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"

};
typedef union VERILOG_STYPE VERILOG_STYPE;
# define VERILOG_STYPE_IS_TRIVIAL 1
# define VERILOG_STYPE_IS_DECLARED 1
#endif



int verilog_parse (yyscan_t yyscanner, ista::VerilogReader* verilog_reader);
/* "%code provides" blocks.  */
#line 32 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"

#undef  YY_DECL
#define YY_DECL int verilog_lex(VERILOG_STYPE *yylval_param, yyscan_t yyscanner, ista::VerilogReader *verilog_reader)
YY_DECL;

void yyerror(yyscan_t scanner,ista::VerilogReader *verilog_reader, const char *str);

#line 217 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"

#endif /* !YY_VERILOG_HOME_LONGSHUAIYING_IEDA_SRC_THIRD_PARTY_PARSER_VERILOG_MVERILOGPARSE_HH_INCLUDED  */

/* Second part of user prologue.  */
#line 81 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"


#line 225 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"


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
typedef yytype_int16 yy_state_t;

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
         || (defined VERILOG_STYPE_IS_TRIVIAL && VERILOG_STYPE_IS_TRIVIAL)))

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
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   393

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  40
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  57
/* YYNRULES -- Number of rules.  */
#define YYNRULES  138
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  270

#define YYUNDEFTOK  2
#define YYMAXUTOK   277


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
       2,     2,     2,     2,     2,    30,     2,     2,     2,     2,
      28,    29,    24,    23,    31,    22,    32,    25,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    36,    27,
       2,    38,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    35,     2,    37,     2,     2,    39,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    33,     2,    34,     2,     2,     2,     2,
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
      15,    16,    17,    18,    19,    20,    21,    26
};

#if VERILOG_DEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    87,    87,    90,    92,    96,    96,   101,   103,   109,
     118,   121,   127,   133,   142,   144,   146,   151,   153,   157,
     160,   165,   167,   168,   172,   175,   178,   185,   185,   187,
     187,   195,   196,   197,   198,   199,   200,   201,   202,   207,
     211,   220,   233,   234,   235,   237,   239,   244,   250,   252,
     257,   258,   262,   264,   269,   271,   276,   280,   287,   291,
     295,   299,   301,   303,   305,   307,   309,   311,   313,   318,
     323,   325,   330,   334,   341,   341,   348,   348,   360,   361,
     362,   363,   364,   365,   366,   367,   368,   372,   378,   387,
     389,   394,   399,   402,   407,   407,   418,   420,   425,   425,
     436,   436,   443,   447,   449,   451,   457,   458,   459,   465,
     468,   474,   480,   492,   497,   504,   511,   516,   518,   521,
     523,   528,   534,   536,   541,   543,   545,   550,   555,   560,
     565,   570,   581,   588,   598,   600,   602,   604,   606
};
#endif

#if VERILOG_DEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "INT", "CONSTANT", "ID", "STRING",
  "MODULE", "ENDMODULE", "ASSIGN", "PARAMETER", "DEFPARAM", "WIRE", "WAND",
  "WOR", "TRI", "INPUT", "OUTPUT", "INOUT", "SUPPLY1", "SUPPLY0", "REG",
  "'-'", "'+'", "'*'", "'/'", "NEG", "';'", "'('", "')'", "'#'", "','",
  "'.'", "'{'", "'}'", "'['", "':'", "']'", "'='", "'`'", "$accept",
  "file", "modules", "module_begin", "@1", "module", "port_list", "port",
  "port_expr", "port_refs", "port_ref", "port_dcls", "port_dcl", "@2",
  "@3", "port_dcl_type", "stmts", "stmt", "stmt_seq", "parameter",
  "module_parameters", "module_parameter", "parameter_dcls",
  "parameter_dcl", "parameter_expr", "defparam", "param_values",
  "param_value", "declaration", "@4", "@5", "dcl_type", "dcl_args",
  "dcl_arg", "continuous_assign", "net_assignments", "net_assignment",
  "@6", "net_assign_lhs", "instance", "@7", "@8", "parameter_values",
  "parameter_exprs", "inst_pins", "inst_ordered_pins", "inst_named_pins",
  "inst_named_pin", "named_pin_net_expr", "net_named", "net_scalar",
  "net_bit_select", "net_part_select", "net_constant", "net_expr_concat",
  "net_exprs", "net_expr", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,    45,    43,    42,    47,   277,    59,    40,    41,
      35,    44,    46,   123,   125,    91,    58,    93,    61,    96
};
# endif

#define YYPACT_NINF (-101)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-128)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -101,    22,    39,  -101,  -101,    34,  -101,  -101,   143,  -101,
     195,    -1,   221,    50,    65,     7,     8,    27,    89,    94,
     246,  -101,  -101,  -101,   266,  -101,    74,  -101,  -101,  -101,
     104,   140,   108,  -101,    20,     3,   126,  -101,  -101,  -101,
    -101,  -101,  -101,  -101,  -101,  -101,  -101,  -101,  -101,  -101,
    -101,   154,  -101,  -101,   199,  -101,  -101,  -101,  -101,  -101,
    -101,   175,    44,  -101,   182,    30,   188,   160,    38,   215,
       5,   286,  -101,  -101,   205,   193,    92,    42,  -101,  -101,
    -101,  -101,  -101,  -101,  -101,   214,   257,    45,  -101,   238,
      95,  -101,    38,   244,    83,   253,    12,    94,  -101,  -101,
    -101,  -101,   144,  -101,  -101,  -101,  -101,   102,   277,  -101,
     268,   104,   271,   272,   311,  -101,  -101,  -101,  -101,  -101,
    -101,   149,  -101,  -101,    20,   281,    28,   284,  -101,   334,
     107,  -101,   126,    96,  -101,   343,   347,  -101,  -101,  -101,
     322,  -101,   273,   293,  -101,  -101,  -101,   102,   102,   102,
     348,   220,   316,   230,  -101,    19,   159,   326,    92,  -101,
    -101,    92,  -101,   172,   352,  -101,  -101,   172,  -101,  -101,
      38,   321,   319,  -101,  -101,  -101,  -101,   196,   320,  -101,
     102,   102,   102,   102,   102,   355,   309,   354,   331,   330,
     332,  -101,  -101,   159,   172,   318,    19,  -101,  -101,   325,
    -101,   361,  -101,  -101,   102,   118,   118,  -101,  -101,   135,
     328,   339,   -14,   340,    92,   336,   183,  -101,   102,   341,
     334,   335,   312,    38,   334,  -101,    32,   366,  -101,  -101,
    -101,  -101,   172,   344,   101,    38,  -101,  -101,  -101,   313,
      29,  -101,   345,  -101,  -101,  -101,   138,  -101,  -101,   106,
    -101,  -101,   370,  -101,   372,   349,  -101,   148,   342,    88,
     351,   350,  -101,   353,  -101,   111,  -101,  -101,   356,  -101
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     2,     1,     5,     0,     4,     6,     0,    39,
       0,     0,     0,   127,    31,    36,    33,     0,     0,     0,
       0,    12,    14,    17,     0,    24,    27,    21,    22,    23,
       0,     0,    98,     7,     0,     0,     0,    85,    84,    86,
      83,    78,    80,    79,    82,    81,    40,    41,    42,    43,
      44,    74,    47,    45,     0,    32,    37,    38,    35,    34,
      39,     0,     0,    19,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    46,     0,     0,     0,     0,    92,    94,
      96,   124,   125,   126,    97,     0,     0,     0,    54,     0,
       0,    70,     0,     0,     0,     0,     0,     0,    18,    39,
      13,    39,    89,    25,    26,    90,    28,     0,     0,    52,
       0,     0,     0,     0,     0,   130,   134,   135,   136,   137,
     138,     0,   132,    91,     0,     0,     0,     0,    48,     0,
       0,    69,     0,     0,    87,     0,     0,   128,     8,    15,
       0,    20,     0,     0,    61,    60,    58,     0,     0,     0,
       0,     0,     0,     0,    51,   106,     0,     0,     0,   131,
      93,     0,    57,    56,     0,    55,    73,    72,    71,    75,
       0,     0,     0,    16,     9,    10,    62,     0,     0,    59,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   107,
     108,   111,   109,     0,   103,     0,   106,   133,    95,     0,
      88,     0,   129,    67,     0,    64,    63,    65,    66,     0,
       0,     0,     0,     0,     0,     0,     0,   102,     0,     0,
       0,     0,     0,     0,     0,    39,     0,     0,    99,   110,
     112,   104,   105,     0,     0,     0,    68,    30,    53,     0,
       0,   113,     0,   121,   122,   123,     0,   101,    49,     0,
      11,   114,     0,   116,     0,     0,    77,     0,     0,     0,
       0,     0,   117,     0,   115,     0,   118,   119,     0,   120
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -101,  -101,  -101,  -101,  -101,  -101,  -101,   323,   280,  -101,
     -18,   228,   317,  -101,  -101,  -101,   -57,  -101,  -101,  -101,
    -101,   275,   163,   -55,  -100,  -101,  -101,   255,  -101,  -101,
    -101,  -101,   155,   -62,  -101,  -101,   -22,  -101,  -101,  -101,
    -101,  -101,  -101,   198,   197,  -101,  -101,   174,  -101,  -101,
      -8,    -6,   -10,   166,   -60,  -101,   -58
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,     5,     7,     6,    20,    21,    22,    62,
      23,    24,    25,    68,    69,    26,    12,    46,    47,    48,
      71,    72,    87,    88,   194,    49,    90,    91,    50,    92,
      93,    51,   133,   134,    52,    77,   105,   125,    79,    53,
      74,    75,   114,   195,   188,   189,   190,   191,   242,    80,
      81,    82,    83,   119,    84,   121,   192
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      29,    63,    27,    95,    28,   104,   106,   151,    85,    29,
      85,    27,    78,    28,   226,   109,   120,    13,   122,    56,
      58,   227,     3,   115,    13,    13,   163,    30,    57,    59,
     167,   144,   145,   146,   162,    13,   115,   240,    86,     8,
     108,   139,   142,   102,   143,    19,     4,   176,   177,   178,
     147,   187,    76,    76,    60,    29,   148,    27,   251,    28,
     149,   241,    18,    19,   252,    76,   118,   150,   116,   123,
     117,    76,   128,   124,   165,    97,   129,    55,    98,   141,
     205,   206,   207,   208,   209,    54,    29,    29,    27,    27,
      28,    28,   115,    13,    61,   120,   115,    13,   120,    13,
     197,   120,   160,   198,   222,   144,   145,   146,   200,   -29,
     144,   145,   146,   166,    70,   115,    13,   262,   232,   136,
     137,    76,   131,   169,   147,    76,   132,   170,   248,   147,
     148,    89,   129,   256,   149,   148,   120,   170,  -100,   149,
     267,   150,   182,   183,    76,   118,   150,   116,   118,   117,
     116,   118,   117,   116,   120,   117,   229,   180,   181,   182,
     183,   237,   144,   145,   146,   102,   245,    73,   239,   238,
       9,    10,   223,    11,   254,   255,    14,    15,    16,    54,
     158,   147,  -127,   159,   136,   260,   118,   148,   116,   -76,
     117,   149,   193,    76,   180,   181,   182,   183,   150,   120,
      13,   263,    94,    96,   118,   120,   116,   268,   117,    99,
     112,    14,    15,    16,   218,   101,   243,   231,   180,   181,
     182,   183,    31,   113,    17,   203,    32,    18,    19,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,   180,   181,   182,   183,    14,    15,    16,   118,
     107,   116,   126,   117,    31,   118,   184,   116,    32,   117,
     127,   138,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    31,    64,   130,    65,    32,   135,
     152,   174,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    31,    66,   153,    67,    32,   155,
     156,   175,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    31,   110,   157,   111,    32,   161,
     164,   250,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,   180,   181,   182,   183,   211,    85,
      67,   236,   180,   181,   182,   183,   171,   217,   204,   218,
     172,   173,   185,   179,   196,   199,   202,   201,   210,   212,
     213,   214,   220,   215,   221,   224,   225,   228,   187,   246,
     233,   247,   235,   257,   253,   258,   140,   259,   265,   261,
     264,   186,   266,   234,   103,   269,   154,   168,   100,   230,
     249,   216,   244,   219
};

static const yytype_int16 yycheck[] =
{
      10,    19,    10,    60,    10,    67,    68,   107,     5,    19,
       5,    19,    34,    19,    28,    70,    76,     5,    76,    12,
      12,    35,     0,     4,     5,     5,   126,    28,    21,    21,
     130,     3,     4,     5,     6,     5,     4,     5,    35,     5,
      35,    29,    99,     5,   101,    33,     7,   147,   148,   149,
      22,    32,    33,    33,    27,    65,    28,    65,    29,    65,
      32,    29,    32,    33,    35,    33,    76,    39,    76,    27,
      76,    33,    27,    31,   129,    31,    31,    12,    34,    97,
     180,   181,   182,   183,   184,    35,    96,    97,    96,    97,
      96,    97,     4,     5,     5,   155,     4,     5,   158,     5,
     158,   161,   124,   161,   204,     3,     4,     5,   170,    35,
       3,     4,     5,     6,    10,     4,     5,    29,   218,    36,
      37,    33,    27,    27,    22,    33,    31,    31,    27,    22,
      28,     5,    31,    27,    32,    28,   196,    31,    30,    32,
      29,    39,    24,    25,    33,   155,    39,   155,   158,   155,
     158,   161,   158,   161,   214,   161,   214,    22,    23,    24,
      25,   223,     3,     4,     5,     5,   226,    27,   225,   224,
      27,    28,    37,    30,    36,    37,    16,    17,    18,    35,
      31,    22,    38,    34,    36,    37,   196,    28,   196,    35,
     196,    32,    33,    33,    22,    23,    24,    25,    39,   259,
       5,   259,     3,    28,   214,   265,   214,   265,   214,    27,
       5,    16,    17,    18,    31,    27,   226,    34,    22,    23,
      24,    25,     1,    30,    29,    29,     5,    32,    33,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    22,    23,    24,    25,    16,    17,    18,   259,
      35,   259,    38,   259,     1,   265,    36,   265,     5,   265,
       3,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     1,    29,    38,    31,     5,    35,
       3,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     1,    29,    28,    31,     5,    28,
      28,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,     1,    29,     5,    31,     5,    38,
      36,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    22,    23,    24,    25,    29,     5,
      31,    29,    22,    23,    24,    25,     3,    29,    28,    31,
       3,    29,    36,     5,    28,     3,    37,    36,     3,     5,
      29,    31,    37,    31,     3,    37,    27,    27,    32,     3,
      29,    27,    37,     3,    29,     3,    96,    28,    28,    37,
      29,   153,    29,   220,    67,    29,   111,   132,    65,   215,
     235,   193,   226,   196
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    41,    42,     0,     7,    43,    45,    44,     5,    27,
      28,    30,    56,     5,    16,    17,    18,    29,    32,    33,
      46,    47,    48,    50,    51,    52,    55,    90,    91,    92,
      28,     1,     5,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    57,    58,    59,    65,
      68,    71,    74,    79,    35,    12,    12,    21,    12,    21,
      27,     5,    49,    50,    29,    31,    29,    31,    53,    54,
      10,    60,    61,    27,    80,    81,    33,    75,    76,    78,
      89,    90,    91,    92,    94,     5,    35,    62,    63,     5,
      66,    67,    69,    70,     3,    56,    28,    31,    34,    27,
      47,    27,     5,    52,    73,    76,    73,    35,    35,    63,
      29,    31,     5,    30,    82,     4,    90,    91,    92,    93,
      94,    95,    96,    27,    31,    77,    38,     3,    27,    31,
      38,    27,    31,    72,    73,    35,    36,    37,     8,    29,
      48,    50,    56,    56,     3,     4,     5,    22,    28,    32,
      39,    64,     3,    28,    61,    28,    28,     5,    31,    34,
      76,    38,     6,    64,    36,    63,     6,    64,    67,    27,
      31,     3,     3,    29,     8,     8,    64,    64,    64,     5,
      22,    23,    24,    25,    36,    36,    51,    32,    84,    85,
      86,    87,    96,    33,    64,    83,    28,    96,    96,     3,
      73,    36,    37,    29,    28,    64,    64,    64,    64,    64,
       3,    29,     5,    29,    31,    31,    83,    29,    31,    84,
      37,     3,    64,    37,    37,    27,    28,    35,    27,    96,
      87,    34,    64,    29,    62,    37,    29,    73,    63,    56,
       5,    29,    88,    92,    93,    94,     3,    27,    27,    72,
       8,    29,    35,    29,    36,    37,    27,     3,     3,    28,
      37,    37,    29,    96,    29,    28,    29,    29,    96,    29
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    40,    41,    42,    42,    44,    43,    45,    45,    45,
      45,    45,    46,    46,    47,    47,    47,    48,    48,    49,
      49,    50,    50,    50,    51,    51,    51,    53,    52,    54,
      52,    55,    55,    55,    55,    55,    55,    55,    55,    56,
      56,    56,    57,    57,    57,    57,    57,    58,    59,    59,
      60,    60,    61,    61,    62,    62,    63,    63,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    65,
      66,    66,    67,    67,    69,    68,    70,    68,    71,    71,
      71,    71,    71,    71,    71,    71,    71,    72,    72,    73,
      73,    74,    75,    75,    77,    76,    78,    78,    80,    79,
      81,    79,    82,    83,    83,    83,    84,    84,    84,    85,
      85,    86,    86,    87,    87,    87,    87,    87,    87,    87,
      87,    88,    88,    88,    89,    89,    89,    90,    91,    92,
      93,    94,    95,    95,    96,    96,    96,    96,    96
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     2,     5,     7,     8,
       8,    12,     1,     3,     1,     4,     5,     1,     3,     1,
       3,     1,     1,     1,     1,     3,     3,     0,     3,     0,
       8,     1,     2,     1,     2,     2,     1,     2,     2,     0,
       2,     2,     1,     1,     1,     1,     2,     1,     3,     8,
       1,     3,     2,     7,     1,     3,     3,     3,     1,     2,
       1,     1,     2,     3,     3,     3,     3,     3,     5,     3,
       1,     3,     3,     3,     0,     4,     0,     9,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     3,     1,     3,     0,     4,     1,     1,     0,     7,
       0,     8,     4,     1,     3,     3,     0,     1,     1,     1,
       3,     1,     3,     4,     5,     8,     5,     7,     8,     9,
      10,     1,     1,     1,     1,     1,     1,     1,     4,     6,
       1,     3,     1,     3,     1,     1,     1,     1,     1
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
        yyerror (yyscanner, verilog_reader, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if VERILOG_DEBUG

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
                  Type, Value, yyscanner, verilog_reader); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (yyscanner);
  YYUSE (verilog_reader);
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
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyo, yytype, yyvaluep, yyscanner, verilog_reader);
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
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, int yyrule, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
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
                                              , yyscanner, verilog_reader);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, yyscanner, verilog_reader); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !VERILOG_DEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !VERILOG_DEBUG */


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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  YYUSE (yyvaluep);
  YYUSE (yyscanner);
  YYUSE (verilog_reader);
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
yyparse (yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
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
      yychar = yylex (&yylval, yyscanner, verilog_reader);
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
  case 5:
#line 96 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
           { (yyval.integer) = verilog_reader->get_line_no(); }
#line 1606 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 6:
#line 97 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.integer) = (yyvsp[0].integer); }
#line 1612 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 7:
#line 102 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { LOG_FATAL << "module 1"; }
#line 1618 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 8:
#line 104 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj)); 
        verilog_reader->makeModule(static_cast<const char*>((yyvsp[-5].string)), std::move(*module_stmts), static_cast<int>((yyvsp[-6].integer)));
        delete module_stmts;
    }
#line 1628 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 9:
#line 110 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {         
        auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>((yyvsp[-4].obj));
        auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj)); 
        verilog_reader->makeModule(static_cast<const char*>((yyvsp[-6].string)), std::move(*port_list), std::move(*module_stmts), static_cast<int>((yyvsp[-7].integer)));
        
        delete port_list;
        delete module_stmts; 
    }
#line 1641 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 10:
#line 119 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { LOG_FATAL << "module 3"; }
#line 1647 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 11:
#line 122 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {   
     }
#line 1654 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 12:
#line 128 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* port_list = new std::vector<std::unique_ptr<VerilogID>>;
        port_list->emplace_back(static_cast<VerilogID*>((yyvsp[0].obj)));
        (yyval.obj) = port_list;
    }
#line 1664 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 13:
#line 134 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {         
        auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>((yyvsp[-2].obj));
        port_list->emplace_back(static_cast<VerilogID*>((yyvsp[0].obj)));
        (yyval.obj) = port_list; 
    }
#line 1674 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 14:
#line 143 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1680 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 15:
#line 145 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj)=nullptr;}
#line 1686 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 16:
#line 147 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj)=nullptr;}
#line 1692 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 17:
#line 152 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1698 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 18:
#line 154 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1704 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 19:
#line 158 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr;
    }
#line 1711 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 20:
#line 161 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[-2].obj); }
#line 1717 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 21:
#line 166 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1723 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 24:
#line 173 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr;;
    }
#line 1730 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 25:
#line 176 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[-2].obj);
    }
#line 1737 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 26:
#line 179 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 1745 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 27:
#line 185 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
                  { (yyval.integer) = verilog_reader->get_line_no(); }
#line 1751 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 28:
#line 186 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1757 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 29:
#line 187 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
                  { (yyval.integer) = verilog_reader->get_line_no(); }
#line 1763 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 30:
#line 189 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1769 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 31:
#line 195 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
          { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInput); }
#line 1775 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 32:
#line 196 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
               { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInputWire); }
#line 1781 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 33:
#line 197 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
          { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInout); }
#line 1787 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 34:
#line 198 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
              { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInoutReg); }
#line 1793 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 35:
#line 199 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
               { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInoutWire); }
#line 1799 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 36:
#line 200 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
           { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOutput); }
#line 1805 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 37:
#line 201 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
                { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOputputWire); }
#line 1811 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 38:
#line 202 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
               { (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOutputReg); }
#line 1817 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 39:
#line 207 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_stmts = new std::vector<std::unique_ptr<VerilogStmt>>;
        (yyval.obj) = verilog_stmts;
    }
#line 1826 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 40:
#line 212 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj)); 
        if ((yyvsp[0].obj)) {
            auto* verilog_stmt = static_cast<VerilogStmt*>((yyvsp[0].obj));
            verilog_stmts->emplace_back(verilog_stmt);
        } 
        (yyval.obj) = (yyvsp[-1].obj); 
    }
#line 1839 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 41:
#line 222 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj)); 
        if ((yyvsp[0].obj)) {
            auto* verilog_stmt = static_cast<VerilogStmt*>((yyvsp[0].obj));
            verilog_stmts->emplace_back(verilog_stmt);
        } 
        (yyval.obj) = (yyvsp[-1].obj); 
    }
#line 1852 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 44:
#line 236 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1858 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 45:
#line 238 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1864 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 46:
#line 240 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { yyerrok; (yyval.obj) = nullptr; }
#line 1870 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 47:
#line 245 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 1876 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 48:
#line 251 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1882 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 49:
#line 253 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1888 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 52:
#line 263 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1894 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 53:
#line 265 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1900 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 54:
#line 270 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1906 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 55:
#line 272 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 1912 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 56:
#line 277 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.obj) = nullptr;
    }
#line 1920 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 57:
#line 281 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.obj) = nullptr;
    }
#line 1928 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 58:
#line 288 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.integer) = 0;
    }
#line 1936 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 59:
#line 292 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.integer) = 0;
    }
#line 1944 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 60:
#line 296 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.integer) = 0;
    }
#line 1952 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 61:
#line 300 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {   }
#line 1958 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 62:
#line 302 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.integer) = - (yyvsp[0].integer); }
#line 1964 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 63:
#line 304 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  }
#line 1970 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 64:
#line 306 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  }
#line 1976 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 65:
#line 308 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  }
#line 1982 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 66:
#line 310 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  }
#line 1988 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 67:
#line 312 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.integer) = (yyvsp[-1].integer); }
#line 1994 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 68:
#line 314 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {}
#line 2000 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 69:
#line 319 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2006 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 70:
#line 324 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2012 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 71:
#line 326 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2018 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 72:
#line 331 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
      (yyval.obj) = nullptr;
    }
#line 2026 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 73:
#line 335 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2034 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 74:
#line 341 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
             { (yyval.integer) = verilog_reader->get_line_no(); }
#line 2040 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 75:
#line 342 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-1].obj));
        auto* declaration = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>((yyvsp[-3].integer)), std::move(*dcl_args), (yyvsp[-2].integer));
        delete dcl_args;
        (yyval.obj) = declaration;
    }
#line 2051 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 76:
#line 348 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
             { (yyval.integer) = verilog_reader->get_line_no(); }
#line 2057 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 77:
#line 350 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-1].obj));
        std::pair<int, int> range = std::make_pair((yyvsp[-5].integer), (yyvsp[-3].integer));
        auto* declaration = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>((yyvsp[-8].integer)), std::move(*dcl_args), (yyvsp[-7].integer), range);
        delete dcl_args;
        (yyval.obj) = declaration; 
    }
#line 2069 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 78:
#line 360 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
          { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kInput); }
#line 2075 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 79:
#line 361 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
          { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kInout); }
#line 2081 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 80:
#line 362 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
           { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kOutput); }
#line 2087 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 81:
#line 363 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
            { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kSupply0); }
#line 2093 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 82:
#line 364 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
            { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kSupply1); }
#line 2099 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 83:
#line 365 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
        { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kTri); }
#line 2105 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 84:
#line 366 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
         { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWand); }
#line 2111 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 85:
#line 367 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
         { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWire); }
#line 2117 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 86:
#line 368 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
        { (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWor); }
#line 2123 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 87:
#line 373 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* dcl_args = new std::vector<const char*>;
        dcl_args->push_back(static_cast<const char*>((yyvsp[0].obj)));
        (yyval.obj) = dcl_args;
    }
#line 2133 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 88:
#line 379 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-2].obj));
        dcl_args->push_back(static_cast<const char*>((yyvsp[0].obj)));
        (yyval.obj) = dcl_args;
    }
#line 2143 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 89:
#line 388 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].string); }
#line 2149 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 90:
#line 390 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2155 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 91:
#line 395 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[-1].obj); }
#line 2161 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 92:
#line 400 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj);
    }
#line 2168 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 93:
#line 403 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[-2].obj); }
#line 2174 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 94:
#line 407 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
                   { (yyval.integer) = verilog_reader->get_line_no(); }
#line 2180 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 95:
#line 408 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  
        auto* lhs = static_cast<VerilogNetExpr*>((yyvsp[-3].obj));
        auto* rhs = static_cast<VerilogNetExpr*>((yyvsp[0].obj));

        auto* module_assign = verilog_reader->makeModuleAssign(lhs, rhs,  (yyvsp[-2].integer)); 
        (yyval.obj) = module_assign; 
        }
#line 2192 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 96:
#line 419 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
        { (yyval.obj) = (yyvsp[0].obj); }
#line 2198 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 97:
#line 421 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
        { (yyval.obj) = (yyvsp[0].obj); }
#line 2204 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 98:
#line 425 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
       { (yyval.integer) = verilog_reader->get_line_no(); }
#line 2210 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 99:
#line 426 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        std::vector<std::unique_ptr<VerilogPortRefPortConnect>> inst_port_connection;
        if(auto* port_connection = static_cast<std::vector<std::unique_ptr<VerilogPortRefPortConnect>>*>((yyvsp[-2].obj));port_connection) {
            inst_port_connection = std::move(*port_connection);
            delete port_connection;        
        }
        
        auto* module_inst = verilog_reader->makeModuleInst(static_cast<const char*>((yyvsp[-6].string)), static_cast<const char*>((yyvsp[-4].string)), std::move(inst_port_connection), (yyvsp[-5].integer)); 
        (yyval.obj) = module_inst;
    }
#line 2225 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 100:
#line 436 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
          { (yyval.integer) = verilog_reader->get_line_no(); }
#line 2231 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 101:
#line 438 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2237 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 103:
#line 448 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { }
#line 2243 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 104:
#line 450 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.integer) = (yyvsp[-1].integer); }
#line 2249 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 105:
#line 452 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {}
#line 2255 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 106:
#line 457 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2261 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 108:
#line 460 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj);}
#line 2267 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 109:
#line 466 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj);
    }
#line 2274 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 110:
#line 469 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[-2].obj); }
#line 2280 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 111:
#line 475 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* port_connect_vec = new std::vector<std::unique_ptr<VerilogPortConnect>>;
        port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>((yyvsp[0].obj)));
        (yyval.obj) = port_connect_vec;
    }
#line 2290 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 112:
#line 481 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* port_connect_vec = static_cast<std::vector<std::unique_ptr<VerilogPortConnect>>*>((yyvsp[-2].obj));
        port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>((yyvsp[0].obj)));
        (yyval.obj) = port_connect_vec;
    }
#line 2300 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 113:
#line 493 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-2].string)));
        (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, nullptr); 
    }
#line 2309 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 114:
#line 498 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)));
        VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-1].string)));
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, net_expr); 
    }
#line 2320 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 115:
#line 505 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        VerilogID* inst_port_id = new VerilogID(static_cast<char*>((yyvsp[-6].string)));
        VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-4].string)), static_cast<int>((yyvsp[-2].integer)));
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, net_expr); 
    }
#line 2331 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 116:
#line 512 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {   VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)));
        (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, static_cast<VerilogNetExpr*>((yyvsp[-1].obj)));  
    }
#line 2339 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 117:
#line 517 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2345 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 118:
#line 519 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2351 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 119:
#line 522 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2357 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 120:
#line 524 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = nullptr; }
#line 2363 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 121:
#line 529 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {
        VerilogID* net_id = static_cast<VerilogID*>((yyvsp[0].obj));
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        (yyval.obj) = net_expr;
    }
#line 2373 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 122:
#line 535 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 2379 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 123:
#line 537 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 2385 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 124:
#line 542 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());  }
#line 2391 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 125:
#line 544 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no()); }
#line 2397 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 126:
#line 546 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no()); }
#line 2403 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 127:
#line 551 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    {  (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[0].string)));  }
#line 2409 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 128:
#line 556 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)), static_cast<int>((yyvsp[-1].integer))); }
#line 2415 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 129:
#line 561 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-5].string)), static_cast<int>((yyvsp[-3].integer)), static_cast<int>((yyvsp[-1].integer))); }
#line 2421 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 130:
#line 566 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<const char*>((yyvsp[0].constant)), verilog_reader->get_line_no()); }
#line 2427 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 131:
#line 571 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>((yyvsp[-1].obj));
        auto* verilog_net_expr_concat = verilog_reader->makeVerilogNetExpr(std::move(*verilog_id_concat), verilog_reader->get_line_no()); 
        delete verilog_id_concat;

        (yyval.obj) = verilog_net_expr_concat;
    }
#line 2439 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 132:
#line 582 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_id_concat = new Vector<std::unique_ptr<VerilogNetExpr>>();
        verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>((yyvsp[0].obj)));

        (yyval.obj) = verilog_id_concat;
    }
#line 2450 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 133:
#line 589 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { 
        auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>((yyvsp[-2].obj));
        verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>((yyvsp[0].obj)));

        (yyval.obj) = verilog_id_concat;
    }
#line 2461 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 134:
#line 599 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no()); }
#line 2467 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 135:
#line 601 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no()); }
#line 2473 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 136:
#line 603 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no()); }
#line 2479 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 137:
#line 605 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 2485 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;

  case 138:
#line 607 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"
    { (yyval.obj) = (yyvsp[0].obj); }
#line 2491 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"
    break;


#line 2495 "/iEDA/src/third_party/parser/verilog/mVerilogParse.cc"

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
      yyerror (yyscanner, verilog_reader, YY_("syntax error"));
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
        yyerror (yyscanner, verilog_reader, yymsgp);
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
                      yytoken, &yylval, yyscanner, verilog_reader);
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
                  yystos[yystate], yyvsp, yyscanner, verilog_reader);
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
  yyerror (yyscanner, verilog_reader, YY_("memory exhausted"));
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
                  yytoken, &yylval, yyscanner, verilog_reader);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[+*yyssp], yyvsp, yyscanner, verilog_reader);
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
#line 610 "/iEDA/src/third_party/parser/verilog/mVerilogParse.y"



void verilog_error(yyscan_t scanner,ista::VerilogReader *verilog_reader, const char *str)
{
   char* error_msg = Str::printf("Error found in line %lu in verilog file %s\n", 
   verilog_reader->get_line_no(), verilog_reader->get_file_name().c_str()); 
   LOG_ERROR << error_msg;
}
