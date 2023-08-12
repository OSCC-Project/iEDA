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
#define YYSTYPE VERILOG_STYPE
/* Substitute the variable and function names.  */
#define yyparse verilog_parse
#define yylex verilog_lex
#define yyerror verilog_error
#define yydebug verilog_debug
#define yynerrs verilog_nerrs

#ifndef YY_CAST
#ifdef __cplusplus
#define YY_CAST(Type, Val) static_cast<Type>(Val)
#define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type>(Val)
#else
#define YY_CAST(Type, Val) ((Type)(Val))
#define YY_REINTERPRET_CAST(Type, Val) ((Type)(Val))
#endif
#endif
#ifndef YY_NULLPTR
#if defined __cplusplus
#if 201103L <= __cplusplus
#define YY_NULLPTR nullptr
#else
#define YY_NULLPTR 0
#endif
#else
#define YY_NULLPTR ((void*) 0)
#endif
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
#undef YYERROR_VERBOSE
#define YYERROR_VERBOSE 1
#else
#define YYERROR_VERBOSE 0
#endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED
#define YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED
/* Debug traces.  */
#ifndef VERILOG_DEBUG
#if defined YYDEBUG
#if YYDEBUG
#define VERILOG_DEBUG 1
#else
#define VERILOG_DEBUG 0
#endif
#else /* ! defined YYDEBUG */
#define VERILOG_DEBUG 1
#endif /* ! defined YYDEBUG */
#endif /* ! defined VERILOG_DEBUG */
#if VERILOG_DEBUG
extern int verilog_debug;
#endif
/* "%code requires" blocks.  */
#line 1 "verilog/VerilogParse.y"

#include <vector>

#include "VerilogReader.hh"
#include "log/Log.hh"
#include "string/Str.hh"

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;

#line 141 "verilog/VerilogParse.cc"

/* Token type.  */
#ifndef VERILOG_TOKENTYPE
#define VERILOG_TOKENTYPE
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
#if !defined VERILOG_STYPE && !defined VERILOG_STYPE_IS_DECLARED
union VERILOG_STYPE
{
#line 25 "verilog/VerilogParse.y"

  int integer;
  char* string;
  const char* constant;
  void* obj;

#line 182 "verilog/VerilogParse.cc"
};
typedef union VERILOG_STYPE VERILOG_STYPE;
#define VERILOG_STYPE_IS_TRIVIAL 1
#define VERILOG_STYPE_IS_DECLARED 1
#endif

int verilog_parse(yyscan_t yyscanner, ista::VerilogReader* verilog_reader);
/* "%code provides" blocks.  */
#line 17 "verilog/VerilogParse.y"

#undef YY_DECL
#define YY_DECL int verilog_lex(VERILOG_STYPE* yylval_param, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
YY_DECL;

void yyerror(yyscan_t scanner, ista::VerilogReader* verilog_reader, const char* str);

#line 202 "verilog/VerilogParse.cc"

#endif /* !YY_VERILOG_HOME_TAOSIMIN_IREFACTOR_SRC_DATABASE_MANAGER_PARSER_VERILOG_VERILOGPARSE_HH_INCLUDED  */

/* Second part of user prologue.  */
#line 65 "verilog/VerilogParse.y"

#line 210 "verilog/VerilogParse.cc"

#ifdef short
#undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
#include <limits.h> /* INFRINGES ON USER NAME SPACE */
#if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#define YY_STDINT_H
#endif
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
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
#if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#define YYPTRDIFF_T __PTRDIFF_TYPE__
#define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
#elif defined PTRDIFF_MAX
#ifndef ptrdiff_t
#include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#endif
#define YYPTRDIFF_T ptrdiff_t
#define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
#else
#define YYPTRDIFF_T long
#define YYPTRDIFF_MAXIMUM LONG_MAX
#endif
#endif

#ifndef YYSIZE_T
#ifdef __SIZE_TYPE__
#define YYSIZE_T __SIZE_TYPE__
#elif defined size_t
#define YYSIZE_T size_t
#elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#define YYSIZE_T size_t
#else
#define YYSIZE_T unsigned
#endif
#endif

#define YYSIZE_MAXIMUM YY_CAST(YYPTRDIFF_T, (YYPTRDIFF_MAXIMUM < YY_CAST(YYSIZE_T, -1) ? YYPTRDIFF_MAXIMUM : YY_CAST(YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST(YYPTRDIFF_T, sizeof(X))

/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
#if defined YYENABLE_NLS && YYENABLE_NLS
#if ENABLE_NLS
#include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#define YY_(Msgid) dgettext("bison-runtime", Msgid)
#endif
#endif
#ifndef YY_
#define YY_(Msgid) Msgid
#endif
#endif

#ifndef YY_ATTRIBUTE_PURE
#if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#define YY_ATTRIBUTE_PURE __attribute__((__pure__))
#else
#define YY_ATTRIBUTE_PURE
#endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
#if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#define YY_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define YY_ATTRIBUTE_UNUSED
#endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if !defined lint || defined __GNUC__
#define YYUSE(E) ((void) (E))
#else
#define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && !defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                                            \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wuninitialized\"") \
      _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define YY_IGNORE_MAYBE_UNINITIALIZED_END _Pragma("GCC diagnostic pop")
#else
#define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
#define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && !defined __ICC && 6 <= __GNUC__
#define YY_IGNORE_USELESS_CAST_BEGIN _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wuseless-cast\"")
#define YY_IGNORE_USELESS_CAST_END _Pragma("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
#define YY_IGNORE_USELESS_CAST_BEGIN
#define YY_IGNORE_USELESS_CAST_END
#endif

#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

#ifdef YYSTACK_USE_ALLOCA
#if YYSTACK_USE_ALLOCA
#ifdef __GNUC__
#define YYSTACK_ALLOC __builtin_alloca
#elif defined __BUILTIN_VA_ARG_INCR
#include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#elif defined _AIX
#define YYSTACK_ALLOC __alloca
#elif defined _MSC_VER
#include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#define alloca _alloca
#else
#define YYSTACK_ALLOC alloca
#if !defined _ALLOCA_H && !defined EXIT_SUCCESS
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
/* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#endif
#endif
#endif

#ifdef YYSTACK_ALLOC
/* Pacify GCC's 'empty if-body' warning.  */
#define YYSTACK_FREE(Ptr) \
  do { /* empty */        \
    ;                     \
  } while (0)
#ifndef YYSTACK_ALLOC_MAXIMUM
/* The OS might guarantee only one guard page at the bottom of the stack,
   and a page size can be as small as 4096 bytes.  So we cannot safely
   invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
   to allow for a few compiler-allocated temporary stack slots.  */
#define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#endif
#else
#define YYSTACK_ALLOC YYMALLOC
#define YYSTACK_FREE YYFREE
#ifndef YYSTACK_ALLOC_MAXIMUM
#define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#endif
#if (defined __cplusplus && !defined EXIT_SUCCESS && !((defined YYMALLOC || defined malloc) && (defined YYFREE || defined free)))
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#ifndef YYMALLOC
#define YYMALLOC malloc
#if !defined malloc && !defined EXIT_SUCCESS
void* malloc(YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#ifndef YYFREE
#define YYFREE free
#if !defined free && !defined EXIT_SUCCESS
void free(void*);       /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */

#if (!defined yyoverflow && (!defined __cplusplus || (defined VERILOG_STYPE_IS_TRIVIAL && VERILOG_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
#define YYSTACK_GAP_MAXIMUM (YYSIZEOF(union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
#define YYSTACK_BYTES(N) ((N) * (YYSIZEOF(yy_state_t) + YYSIZEOF(YYSTYPE)) + YYSTACK_GAP_MAXIMUM)

#define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
#define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
  do {                                                                 \
    YYPTRDIFF_T yynewbytes;                                            \
    YYCOPY(&yyptr->Stack_alloc, Stack, yysize);                        \
    Stack = &yyptr->Stack_alloc;                                       \
    yynewbytes = yystacksize * YYSIZEOF(*Stack) + YYSTACK_GAP_MAXIMUM; \
    yyptr += yynewbytes / YYSIZEOF(*yyptr);                            \
  } while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
#ifndef YYCOPY
#if defined __GNUC__ && 1 < __GNUC__
#define YYCOPY(Dst, Src, Count) __builtin_memcpy(Dst, Src, YY_CAST(YYSIZE_T, (Count)) * sizeof(*(Src)))
#else
#define YYCOPY(Dst, Src, Count)         \
  do {                                  \
    YYPTRDIFF_T yyi;                    \
    for (yyi = 0; yyi < (Count); yyi++) \
      (Dst)[yyi] = (Src)[yyi];          \
  } while (0)
#endif
#endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL 3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST 332

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS 40
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS 55
/* YYNRULES -- Number of rules.  */
#define YYNRULES 132
/* YYNSTATES -- Number of states.  */
#define YYNSTATES 244

#define YYUNDEFTOK 2
#define YYMAXUTOK 277

/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX) (0 <= (YYX) && (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[]
    = {0,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2,
       39, 2, 2, 2, 2, 28, 29, 24, 23, 30, 22, 31, 25, 2, 2, 2, 2, 2, 2,  2, 2,  2,  2,  35, 27, 2,  37, 2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2,  2, 2,  34, 2,  36, 2,  2,  38, 2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 32, 2, 33, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2, 2, 2, 2,  2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2, 2,
       2,  2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  1,  2,  3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26};

#if VERILOG_DEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] = {
    0,   71,  71,  74,  76,  80,  80,  85,  87,  93,  102, 107, 113, 122, 124, 126, 131, 133, 137, 140, 145, 147, 148, 152, 155, 158, 165,
    165, 167, 167, 173, 174, 175, 176, 177, 178, 179, 180, 185, 189, 198, 211, 212, 213, 215, 217, 222, 228, 230, 235, 237, 242, 246, 253,
    257, 261, 265, 266, 268, 270, 272, 274, 276, 281, 286, 288, 293, 297, 304, 304, 311, 311, 323, 324, 325, 326, 327, 328, 329, 330, 331,
    335, 341, 350, 352, 357, 362, 365, 370, 370, 381, 383, 388, 388, 399, 399, 406, 410, 411, 413, 418, 419, 420, 426, 429, 435, 441, 453,
    458, 465, 472, 477, 479, 482, 484, 489, 495, 497, 502, 504, 506, 511, 516, 521, 526, 531, 542, 549, 559, 561, 563, 565, 567};
#endif

#if VERILOG_DEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char* const yytname[] = {"$end",
                                      "error",
                                      "$undefined",
                                      "INT",
                                      "CONSTANT",
                                      "ID",
                                      "STRING",
                                      "MODULE",
                                      "ENDMODULE",
                                      "ASSIGN",
                                      "PARAMETER",
                                      "DEFPARAM",
                                      "WIRE",
                                      "WAND",
                                      "WOR",
                                      "TRI",
                                      "INPUT",
                                      "OUTPUT",
                                      "INOUT",
                                      "SUPPLY1",
                                      "SUPPLY0",
                                      "REG",
                                      "'-'",
                                      "'+'",
                                      "'*'",
                                      "'/'",
                                      "NEG",
                                      "';'",
                                      "'('",
                                      "')'",
                                      "','",
                                      "'.'",
                                      "'{'",
                                      "'}'",
                                      "'['",
                                      "':'",
                                      "']'",
                                      "'='",
                                      "'`'",
                                      "'#'",
                                      "$accept",
                                      "file",
                                      "modules",
                                      "module_begin",
                                      "@1",
                                      "module",
                                      "port_list",
                                      "port",
                                      "port_expr",
                                      "port_refs",
                                      "port_ref",
                                      "port_dcls",
                                      "port_dcl",
                                      "@2",
                                      "@3",
                                      "port_dcl_type",
                                      "stmts",
                                      "stmt",
                                      "stmt_seq",
                                      "parameter",
                                      "parameter_dcls",
                                      "parameter_dcl",
                                      "parameter_expr",
                                      "defparam",
                                      "param_values",
                                      "param_value",
                                      "declaration",
                                      "@4",
                                      "@5",
                                      "dcl_type",
                                      "dcl_args",
                                      "dcl_arg",
                                      "continuous_assign",
                                      "net_assignments",
                                      "net_assignment",
                                      "@6",
                                      "net_assign_lhs",
                                      "instance",
                                      "@7",
                                      "@8",
                                      "parameter_values",
                                      "parameter_exprs",
                                      "inst_pins",
                                      "inst_ordered_pins",
                                      "inst_named_pins",
                                      "inst_named_pin",
                                      "named_pin_net_expr",
                                      "net_named",
                                      "net_scalar",
                                      "net_bit_select",
                                      "net_part_select",
                                      "net_constant",
                                      "net_expr_concat",
                                      "net_exprs",
                                      "net_expr",
                                      YY_NULLPTR};
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] = {0,   256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
                                        275, 276, 45,  43,  42,  47,  277, 59,  40,  41,  44,  46,  123, 125, 91,  58,  93,  61,  96,  35};
#endif

#define YYPACT_NINF (-82)

#define yypact_value_is_default(Yyn) ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-122)

#define yytable_value_is_error(Yyn) 0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[]
    = {-82, 11,  28,  -82, -82, 45,  -82, -82, 87,  -82, 140, 187, 34,  61,  39,  84,  51,  49,  86,  73,  -82, -82, -82, 184, -82,
       55,  -82, -82, -82, 68,  58,  -82, 15,  4,   105, -82, -82, -82, -82, -82, -82, -82, -82, -82, -82, -82, -82, -82, -82, 82,
       -82, -82, 121, -82, -82, -82, -82, -82, -82, 107, 79,  -82, 113, 2,   128, 143, 32,  108, -82, 157, 125, 17,  114, -82, -82,
       -82, -82, -82, -82, -82, 129, 188, 119, -82, 148, 124, -82, 32,  196, 214, 227, 40,  86,  -82, -82, -82, -82, 136, -82, -82,
       -82, -82, 215, 198, 206, 219, -82, -82, -82, -82, -82, -82, 156, -82, -82, 15,  251, 20,  216, -82, 266, 24,  -82, 105, 163,
       -82, 286, 287, -82, -82, -82, 262, -82, 247, 267, 257, 12,  89,  265, 17,  -82, -82, 17,  -82, -82, -82, -82, 115, 115, 289,
       186, 292, -82, -82, 186, -82, -82, 32,  261, 263, -82, -82, -82, 294, 293, 271, 272, 273, -82, -82, 89,  186, 192, 12,  -82,
       -82, -82, 154, -82, 115, 115, 115, 115, 268, -82, 298, -82, 269, 42,  279, 17,  276, 182, -82, 115, 280, -82, 229, 229, -82,
       -82, 266, 274, 32,  27,  305, -82, -82, -82, -82, 186, 284, 189, 32,  -82, -16, -82, 283, -82, -82, -82, 234, -82, -82, 190,
       -82, 310, -82, 311, 288, -82, 238, 281, 75,  290, 295, -82, 291, -82, 118, -82, -82, 296, -82};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[]
    = {3,   0,   2,  1,   5,  0,   4,   6,  0,  38, 0,  0,   121, 30, 35,  32, 0,   0,   0,   0,  11,  13,  16,  0,   23,  26,  20,  21,
       22,  0,   92, 7,   0,  0,   0,   79, 78, 80, 77, 72,  74,  73, 76,  75, 39,  40,  41,  42, 43,  68,  46,  44,  0,   31,  36,  37,
       34,  33,  38, 0,   0,  18,  0,   0,  0,  0,  0,  0,   45,  0,  0,   0,  0,   86,  88,  90, 118, 119, 120, 91,  0,   0,   0,   49,
       0,   0,   64, 0,   0,  0,   0,   0,  0,  17, 38, 12,  38,  83, 24,  25, 84,  27,  0,   0,  0,   0,   124, 128, 129, 130, 131, 132,
       0,   126, 85, 0,   0,  0,   0,   47, 0,  0,  63, 0,   0,   81, 0,   0,  122, 8,   14,  0,  19,  0,   0,   0,   100, 0,   0,   0,
       125, 87,  0,  56,  55, 53,  52,  0,  0,  0,  51, 0,   50,  67, 66,  65, 69,  0,   0,   0,  15,  9,   10,  0,   0,   0,   101, 102,
       105, 103, 0,  97,  0,  100, 127, 89, 57, 0,  54, 0,   0,   0,  0,   0,  82,  0,   123, 0,  0,   0,   0,   0,   0,   96,  0,   0,
       62,  59,  58, 60,  61, 0,   0,   0,  0,  0,  93, 104, 106, 98, 99,  0,  0,   0,   29,  0,  107, 0,   115, 116, 117, 0,   95,  48,
       0,   108, 0,  110, 0,  0,   71,  0,  0,  0,  0,  0,   111, 0,  109, 0,  112, 113, 0,   114};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] = {-82, -82, -82, -82, -82, -82, -82, 252, 230, -82, -17, -82, 253, -82, -82, -82, -55, -82, -82,
                                       -82, 123, 202, -81, -82, -82, 203, -82, -82, -82, -82, 116, -51, -82, -82, -27, -82, -82, -82,
                                       -82, -82, -82, 158, 159, -82, -82, 139, -82, -82, -8,  -6,  -10, 127, -65, -82, -52};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[]
    = {-1, 1,  2,   5,   7,  6,  19,  20,  21, 60, 22, 23, 24,  66,  67,  25,  11,  44,  45,  46, 82, 83, 171, 47,  85, 86,  48, 87,
       88, 49, 124, 125, 50, 72, 100, 116, 74, 51, 69, 70, 105, 172, 165, 166, 167, 168, 217, 75, 76, 77, 78,  110, 79, 112, 169};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[]
    = {28,  61,  26,  90,  27,  73,  111, 12,  28,  80,  26,  3,   27,  225, 99,  101, 106, 12,   226, 113, 12,  106, 12,  143, 144, 145,
       146, 143, 144, 145, 153, 106, 215, 17,  18,  4,   150, 97,  81,  133, 154, 134, 147, 164,  71,  12,  147, 71,  148, 71,  8,   54,
       148, 28,  59,  26,  216, 27,  149, 71,  55,  109, 149, 107, 71,  108, 176, 177, 52,  130,  204, 111, 18,  53,  111, 132, 205, 111,
       58,  106, 12,  28,  28,  26,  26,  27,  27,  174, 141, -28, 175, 12,  143, 144, 145, 68,   56,  -94, 197, 198, 199, 200, 62,  63,
       236, 57,  184, 71,  111, 92,  84,  147, 93,  210, 9,   10,  -70, 148, 143, 144, 145, 170,  106, 12,  89,  111, 109, 149, 107, 109,
       108, 107, 109, 108, 107, 91,  108, 147, 207, 220, 94,  114, 102, 148, 115, 12,  119, 241,  97,  120, 71,  122, 214, 149, 123, 96,
       13,  14,  15,  13,  14,  15,  103, 109, 104, 107, 117, 108, 111, 16,  52,  17,  18,  -121, 111, 71,  179, 180, 181, 182, 109, 237,
       107, 196, 108, 121, 139, 242, 29,  140, 156, 118, 30,  157, 218, 31,  32,  33,  34,  35,   36,  37,  38,  39,  40,  41,  42,  43,
       179, 180, 181, 182, 194, 64,  65,  209, 223, 230, 135, 120, 157, 193, 194, 109, 138, 107,  136, 108, 29,  109, 126, 107, 30,  108,
       137, 129, 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  29,  127, 128, 151,  30,  181, 182, 161, 32,  33,  34,  35,
       36,  37,  38,  39,  40,  41,  42,  43,  29,  228, 229, 80,  30,  127, 234, 162, 32,  33,   34,  35,  36,  37,  38,  39,  40,  41,
       42,  43,  142, 158, 159, 160, 163, 173, 178, 183, 185, 187, 188, 186, 189, 202, 190, 191,  201, 203, 206, 164, 221, 211, 213, 222,
       227, 231, 232, 95,  233, 235, 98,  238, 240, 131, 152, 239, 212, 243, 155, 0,   192, 224,  208, 219, 195};

static const yytype_int16 yycheck[]
    = {10,  18,  10,  58,  10,  32,  71,  5,   18,  5,   18,  0,   18,  29, 65,  66,  4,   5,   34,  71,  5,   4,   5,   3,   4,   5,
       6,   3,   4,   5,   6,   4,   5,   31,  32,  7,   117, 5,   34,  94, 121, 96,  22,  31,  32,  5,   22,  32,  28,  32,  5,   12,
       28,  63,  5,   63,  29,  63,  38,  32,  21,  71,  38,  71,  32,  71, 147, 148, 34,  29,  28,  136, 32,  12,  139, 92,  34,  142,
       27,  4,   5,   91,  92,  91,  92,  91,  92,  139, 115, 34,  142, 5,  3,   4,   5,   27,  12,  39,  179, 180, 181, 182, 29,  30,
       29,  21,  157, 32,  173, 30,  5,   22,  33,  194, 27,  28,  34,  28, 3,   4,   5,   32,  4,   5,   3,   190, 136, 38,  136, 139,
       136, 139, 142, 139, 142, 28,  142, 22,  190, 204, 27,  27,  34,  28, 30,  5,   27,  29,  5,   30,  32,  27,  203, 38,  30,  27,
       16,  17,  18,  16,  17,  18,  5,   173, 39,  173, 37,  173, 233, 29, 34,  31,  32,  37,  239, 32,  22,  23,  24,  25,  190, 233,
       190, 29,  190, 37,  30,  239, 1,   33,  27,  3,   5,   30,  204, 8,  9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
       22,  23,  24,  25,  30,  29,  30,  33,  27,  27,  3,   30,  30,  29, 30,  233, 5,   233, 28,  233, 1,   239, 34,  239, 5,   239,
       28,  8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20, 1,   35,  36,  35,  5,   24,  25,  8,   9,   10,  11,  12,
       13,  14,  15,  16,  17,  18,  19,  20,  1,   35,  36,  5,   5,   35, 36,  8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,
       19,  20,  37,  3,   3,   29,  35,  28,  5,   3,   35,  3,   5,   36, 29,  3,   30,  30,  36,  36,  27,  31,  3,   29,  36,  27,
       29,  3,   3,   63,  28,  36,  65,  29,  29,  91,  120, 28,  201, 29, 123, -1,  170, 213, 191, 204, 173};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_int8 yystos[]
    = {0,  41, 42, 0,  7,  43, 45, 44, 5,  27, 28, 56, 5,  16, 17, 18, 29, 31, 32, 46, 47, 48, 50, 51, 52, 55, 88, 89, 90, 1,  5,
       8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 57, 58, 59, 63, 66, 69, 72, 77, 34, 12, 12, 21, 12, 21, 27, 5,  49, 50,
       29, 30, 29, 30, 53, 54, 27, 78, 79, 32, 73, 74, 76, 87, 88, 89, 90, 92, 5,  34, 60, 61, 5,  64, 65, 67, 68, 3,  56, 28, 30,
       33, 27, 47, 27, 5,  52, 71, 74, 71, 34, 5,  39, 80, 4,  88, 89, 90, 91, 92, 93, 94, 27, 30, 75, 37, 3,  27, 30, 37, 27, 30,
       70, 71, 34, 35, 36, 8,  29, 48, 50, 56, 56, 3,  28, 28, 5,  30, 33, 74, 37, 3,  4,  5,  6,  22, 28, 38, 62, 35, 61, 6,  62,
       65, 27, 30, 3,  3,  29, 8,  8,  35, 31, 82, 83, 84, 85, 94, 32, 62, 81, 28, 94, 94, 62, 62, 5,  22, 23, 24, 25, 3,  71, 35,
       36, 3,  5,  29, 30, 30, 81, 29, 30, 82, 29, 62, 62, 62, 62, 36, 3,  36, 28, 34, 27, 94, 85, 33, 62, 29, 60, 36, 71, 5,  29,
       86, 90, 91, 92, 3,  27, 27, 70, 29, 34, 29, 35, 36, 27, 3,  3,  28, 36, 36, 29, 94, 29, 28, 29, 29, 94, 29};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] = {
    0,  40, 41, 42, 42, 44, 43, 45, 45, 45, 45, 46, 46, 47, 47, 47, 48, 48, 49, 49, 50, 50, 50, 51, 51, 51, 53, 52, 54, 52, 55, 55, 55, 55,
    55, 55, 55, 55, 56, 56, 56, 57, 57, 57, 57, 57, 58, 59, 59, 60, 60, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 64, 64, 65, 65,
    67, 66, 68, 66, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 71, 71, 72, 73, 73, 75, 74, 76, 76, 78, 77, 79, 77, 80, 81, 81, 81, 82, 82,
    82, 83, 83, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 89, 90, 91, 92, 93, 93, 94, 94, 94, 94, 94};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] = {0, 2, 1, 0, 2, 0, 2, 5, 7, 8, 8, 1, 3,  1, 4, 5, 1, 3, 1, 3, 1, 1, 1, 1, 3, 3, 0, 3, 0, 8, 1, 2, 1, 2,
                                   2, 1, 2, 2, 0, 2, 2, 1, 1, 1, 1, 2, 1,  3, 8, 1, 3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3,
                                   0, 4, 0, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 3, 1, 1, 3, 1, 3, 0, 4, 1, 1, 0, 7, 0, 8, 4, 1, 3, 3, 0, 1,
                                   1, 1, 3, 1, 3, 4, 5, 8, 5, 7, 8, 9, 10, 1, 1, 1, 1, 1, 1, 1, 4, 6, 1, 3, 1, 3, 1, 1, 1, 1, 1};

#define yyerrok (yyerrstatus = 0)
#define yyclearin (yychar = YYEMPTY)
#define YYEMPTY (-2)
#define YYEOF 0

#define YYACCEPT goto yyacceptlab
#define YYABORT goto yyabortlab
#define YYERROR goto yyerrorlab

#define YYRECOVERING() (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                                 \
  do                                                                           \
    if (yychar == YYEMPTY) {                                                   \
      yychar = (Token);                                                        \
      yylval = (Value);                                                        \
      YYPOPSTACK(yylen);                                                       \
      yystate = *yyssp;                                                        \
      goto yybackup;                                                           \
    } else {                                                                   \
      yyerror(yyscanner, verilog_reader, YY_("syntax error: cannot back up")); \
      YYERROR;                                                                 \
    }                                                                          \
  while (0)

/* Error token number */
#define YYTERROR 1
#define YYERRCODE 256

/* Enable debugging if requested.  */
#if VERILOG_DEBUG

#ifndef YYFPRINTF
#include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#define YYFPRINTF fprintf
#endif

#define YYDPRINTF(Args) \
  do {                  \
    if (yydebug)        \
      YYFPRINTF Args;   \
  } while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
#define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif

#define YY_SYMBOL_PRINT(Title, Type, Value, Location)                  \
  do {                                                                 \
    if (yydebug) {                                                     \
      YYFPRINTF(stderr, "%s ", Title);                                 \
      yy_symbol_print(stderr, Type, Value, yyscanner, verilog_reader); \
      YYFPRINTF(stderr, "\n");                                         \
    }                                                                  \
  } while (0)

/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void yy_symbol_value_print(FILE* yyo, int yytype, YYSTYPE const* const yyvaluep, yyscan_t yyscanner,
                                  ista::VerilogReader* verilog_reader)
{
  FILE* yyoutput = yyo;
  YYUSE(yyoutput);
  YYUSE(yyscanner);
  YYUSE(verilog_reader);
  if (!yyvaluep)
    return;
#ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT(yyo, yytoknum[yytype], *yyvaluep);
#endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE(yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}

/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void yy_symbol_print(FILE* yyo, int yytype, YYSTYPE const* const yyvaluep, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  YYFPRINTF(yyo, "%s %s (", yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print(yyo, yytype, yyvaluep, yyscanner, verilog_reader);
  YYFPRINTF(yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void yy_stack_print(yy_state_t* yybottom, yy_state_t* yytop)
{
  YYFPRINTF(stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++) {
    int yybot = *yybottom;
    YYFPRINTF(stderr, " %d", yybot);
  }
  YYFPRINTF(stderr, "\n");
}

#define YY_STACK_PRINT(Bottom, Top)    \
  do {                                 \
    if (yydebug)                       \
      yy_stack_print((Bottom), (Top)); \
  } while (0)

/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void yy_reduce_print(yy_state_t* yyssp, YYSTYPE* yyvsp, int yyrule, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF(stderr, "Reducing stack by rule %d (line %d):\n", yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++) {
    YYFPRINTF(stderr, "   $%d = ", yyi + 1);
    yy_symbol_print(stderr, yystos[+yyssp[yyi + 1 - yynrhs]], &yyvsp[(yyi + 1) - (yynrhs)], yyscanner, verilog_reader);
    YYFPRINTF(stderr, "\n");
  }
}

#define YY_REDUCE_PRINT(Rule)                                         \
  do {                                                                \
    if (yydebug)                                                      \
      yy_reduce_print(yyssp, yyvsp, Rule, yyscanner, verilog_reader); \
  } while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !VERILOG_DEBUG */
#define YYDPRINTF(Args)
#define YY_SYMBOL_PRINT(Title, Type, Value, Location)
#define YY_STACK_PRINT(Bottom, Top)
#define YY_REDUCE_PRINT(Rule)
#endif /* !VERILOG_DEBUG */

/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

#if YYERROR_VERBOSE

#ifndef yystrlen
#if defined __GLIBC__ && defined _STRING_H
#define yystrlen(S) (YY_CAST(YYPTRDIFF_T, strlen(S)))
#else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T yystrlen(const char* yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#endif
#endif

#ifndef yystpcpy
#if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#define yystpcpy stpcpy
#else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char* yystpcpy(char* yydest, const char* yysrc)
{
  char* yyd = yydest;
  const char* yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T yytnamerr(char* yyres, const char* yystr)
{
  if (*yystr == '"') {
    YYPTRDIFF_T yyn = 0;
    char const* yyp = yystr;

    for (;;)
      switch (*++yyp) {
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
  do_not_strip_quotes:;
  }

  if (yyres)
    return yystpcpy(yyres, yystr) - yyres;
  else
    return yystrlen(yystr);
}
#endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int yysyntax_error(YYPTRDIFF_T* yymsg_alloc, char** yymsg, yy_state_t* yyssp, int yytoken)
{
  enum
  {
    YYERROR_VERBOSE_ARGS_MAXIMUM = 5
  };
  /* Internationalized format string. */
  const char* yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const* yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
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
  if (yytoken != YYEMPTY) {
    int yyn = yypact[+*yyssp];
    YYPTRDIFF_T yysize0 = yytnamerr(YY_NULLPTR, yytname[yytoken]);
    yysize = yysize0;
    yyarg[yycount++] = yytname[yytoken];
    if (!yypact_value_is_default(yyn)) {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR && !yytable_value_is_error(yytable[yyx + yyn])) {
          if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM) {
            yycount = 1;
            yysize = yysize0;
            break;
          }
          yyarg[yycount++] = yytname[yyx];
          {
            YYPTRDIFF_T yysize1 = yysize + yytnamerr(YY_NULLPTR, yytname[yyx]);
            if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
              yysize = yysize1;
            else
              return 2;
          }
        }
    }
  }

  switch (yycount) {
#define YYCASE_(N, S) \
  case N:             \
    yyformat = S;     \
    break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
  }

  {
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    YYPTRDIFF_T yysize1 = yysize + (yystrlen(yyformat) - 2 * yycount) + 1;
    if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
      yysize = yysize1;
    else
      return 2;
  }

  if (*yymsg_alloc < yysize) {
    *yymsg_alloc = 2 * yysize;
    if (!(yysize <= *yymsg_alloc && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
      *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
    return 1;
  }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char* yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount) {
        yyp += yytnamerr(yyp, yyarg[yyi++]);
        yyformat += 2;
      } else {
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

static void yydestruct(const char* yymsg, int yytype, YYSTYPE* yyvaluep, yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  YYUSE(yyvaluep);
  YYUSE(yyscanner);
  YYUSE(verilog_reader);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT(yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE(yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}

/*----------.
| yyparse.  |
`----------*/

int yyparse(yyscan_t yyscanner, ista::VerilogReader* verilog_reader)
{
  /* The lookahead symbol.  */
  int yychar;

  /* The semantic value of the lookahead symbol.  */
  /* Default value used for initialization, for pacifying older GCCs
     or non-GCC compilers.  */
  YY_INITIAL_VALUE(static YYSTYPE yyval_default;)
  YYSTYPE yylval YY_INITIAL_VALUE(= yyval_default);

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
  yy_state_t* yyss;
  yy_state_t* yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE* yyvs;
  YYSTYPE* yyvsp;

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
  char* yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N) (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF((stderr, "Starting parse\n"));

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
  YYDPRINTF((stderr, "Entering state %d\n", yystate));
  YY_ASSERT(0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST(yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
  {
    /* Get the current used size of the three stacks, in elements.  */
    YYPTRDIFF_T yysize = yyssp - yyss + 1;

#if defined yyoverflow
    {
      /* Give user a chance to reallocate the stack.  Use copies of
         these so that the &'s don't force the real ones into
         memory.  */
      yy_state_t* yyss1 = yyss;
      YYSTYPE* yyvs1 = yyvs;

      /* Each stack pointer address is followed by the size of the
         data in use in that stack, in bytes.  This used to be a
         conditional around just the two extra args, but that might
         be undefined if yyoverflow is a macro.  */
      yyoverflow(YY_("memory exhausted"), &yyss1, yysize * YYSIZEOF(*yyssp), &yyvs1, yysize * YYSIZEOF(*yyvsp), &yystacksize);
      yyss = yyss1;
      yyvs = yyvs1;
    }
#else /* defined YYSTACK_RELOCATE */
    /* Extend the stack our own way.  */
    if (YYMAXDEPTH <= yystacksize)
      goto yyexhaustedlab;
    yystacksize *= 2;
    if (YYMAXDEPTH < yystacksize)
      yystacksize = YYMAXDEPTH;

    {
      yy_state_t* yyss1 = yyss;
      union yyalloc* yyptr = YY_CAST(union yyalloc*, YYSTACK_ALLOC(YY_CAST(YYSIZE_T, YYSTACK_BYTES(yystacksize))));
      if (!yyptr)
        goto yyexhaustedlab;
      YYSTACK_RELOCATE(yyss_alloc, yyss);
      YYSTACK_RELOCATE(yyvs_alloc, yyvs);
#undef YYSTACK_RELOCATE
      if (yyss1 != yyssa)
        YYSTACK_FREE(yyss1);
    }
#endif

    yyssp = yyss + yysize - 1;
    yyvsp = yyvs + yysize - 1;

    YY_IGNORE_USELESS_CAST_BEGIN
    YYDPRINTF((stderr, "Stack size increased to %ld\n", YY_CAST(long, yystacksize)));
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
  if (yypact_value_is_default(yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY) {
    YYDPRINTF((stderr, "Reading a token: "));
    yychar = yylex(&yylval, yyscanner, verilog_reader);
  }

  if (yychar <= YYEOF) {
    yychar = yytoken = YYEOF;
    YYDPRINTF((stderr, "Now at end of input.\n"));
  } else {
    yytoken = YYTRANSLATE(yychar);
    YY_SYMBOL_PRINT("Next token is", yytoken, &yylval, &yylloc);
  }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0) {
    if (yytable_value_is_error(yyn))
      goto yyerrlab;
    yyn = -yyn;
    goto yyreduce;
  }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT("Shifting", yytoken, &yylval, &yylloc);
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
  yyval = yyvsp[1 - yylen];

  YY_REDUCE_PRINT(yyn);
  switch (yyn) {
    case 5:
#line 80 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 1572 "verilog/VerilogParse.cc"
    break;

    case 6:
#line 81 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[0].integer);
    }
#line 1578 "verilog/VerilogParse.cc"
    break;

    case 7:
#line 86 "verilog/VerilogParse.y"
    {
      LOG_FATAL << "module 1";
    }
#line 1584 "verilog/VerilogParse.cc"
    break;

    case 8:
#line 88 "verilog/VerilogParse.y"
    {
      auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj));
      verilog_reader->makeModule(static_cast<const char*>((yyvsp[-5].string)), std::move(*module_stmts),
                                 static_cast<int>((yyvsp[-6].integer)));
      delete module_stmts;
    }
#line 1594 "verilog/VerilogParse.cc"
    break;

    case 9:
#line 94 "verilog/VerilogParse.y"
    {
      auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>((yyvsp[-4].obj));
      auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj));
      verilog_reader->makeModule(static_cast<const char*>((yyvsp[-6].string)), std::move(*port_list), std::move(*module_stmts),
                                 static_cast<int>((yyvsp[-7].integer)));

      delete port_list;
      delete module_stmts;
    }
#line 1607 "verilog/VerilogParse.cc"
    break;

    case 10:
#line 103 "verilog/VerilogParse.y"
    {
      LOG_FATAL << "module 3";
    }
#line 1613 "verilog/VerilogParse.cc"
    break;

    case 11:
#line 108 "verilog/VerilogParse.y"
    {
      auto* port_list = new std::vector<std::unique_ptr<VerilogID>>;
      port_list->emplace_back(static_cast<VerilogID*>((yyvsp[0].obj)));
      (yyval.obj) = port_list;
    }
#line 1623 "verilog/VerilogParse.cc"
    break;

    case 12:
#line 114 "verilog/VerilogParse.y"
    {
      auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>((yyvsp[-2].obj));
      port_list->emplace_back(static_cast<VerilogID*>((yyvsp[0].obj)));
      (yyval.obj) = port_list;
    }
#line 1633 "verilog/VerilogParse.cc"
    break;

    case 13:
#line 123 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1639 "verilog/VerilogParse.cc"
    break;

    case 14:
#line 125 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1645 "verilog/VerilogParse.cc"
    break;

    case 15:
#line 127 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1651 "verilog/VerilogParse.cc"
    break;

    case 16:
#line 132 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1657 "verilog/VerilogParse.cc"
    break;

    case 17:
#line 134 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1663 "verilog/VerilogParse.cc"
    break;

    case 18:
#line 138 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1670 "verilog/VerilogParse.cc"
    break;

    case 19:
#line 141 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 1676 "verilog/VerilogParse.cc"
    break;

    case 20:
#line 146 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1682 "verilog/VerilogParse.cc"
    break;

    case 23:
#line 153 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
      ;
    }
#line 1689 "verilog/VerilogParse.cc"
    break;

    case 24:
#line 156 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 1696 "verilog/VerilogParse.cc"
    break;

    case 25:
#line 159 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 1704 "verilog/VerilogParse.cc"
    break;

    case 26:
#line 165 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 1710 "verilog/VerilogParse.cc"
    break;

    case 27:
#line 166 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1716 "verilog/VerilogParse.cc"
    break;

    case 28:
#line 167 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 1722 "verilog/VerilogParse.cc"
    break;

    case 29:
#line 169 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1728 "verilog/VerilogParse.cc"
    break;

    case 30:
#line 173 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInput);
    }
#line 1734 "verilog/VerilogParse.cc"
    break;

    case 31:
#line 174 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInputWire);
    }
#line 1740 "verilog/VerilogParse.cc"
    break;

    case 32:
#line 175 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInout);
    }
#line 1746 "verilog/VerilogParse.cc"
    break;

    case 33:
#line 176 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInoutReg);
    }
#line 1752 "verilog/VerilogParse.cc"
    break;

    case 34:
#line 177 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kInoutWire);
    }
#line 1758 "verilog/VerilogParse.cc"
    break;

    case 35:
#line 178 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOutput);
    }
#line 1764 "verilog/VerilogParse.cc"
    break;

    case 36:
#line 179 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOputputWire);
    }
#line 1770 "verilog/VerilogParse.cc"
    break;

    case 37:
#line 180 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogModule::PortDclType::kOutputReg);
    }
#line 1776 "verilog/VerilogParse.cc"
    break;

    case 38:
#line 185 "verilog/VerilogParse.y"
    {
      auto* verilog_stmts = new std::vector<std::unique_ptr<VerilogStmt>>;
      (yyval.obj) = verilog_stmts;
    }
#line 1785 "verilog/VerilogParse.cc"
    break;

    case 39:
#line 190 "verilog/VerilogParse.y"
    {
      auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj));
      if ((yyvsp[0].obj)) {
        auto* verilog_stmt = static_cast<VerilogStmt*>((yyvsp[0].obj));
        verilog_stmts->emplace_back(verilog_stmt);
      }
      (yyval.obj) = (yyvsp[-1].obj);
    }
#line 1798 "verilog/VerilogParse.cc"
    break;

    case 40:
#line 200 "verilog/VerilogParse.y"
    {
      auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>((yyvsp[-1].obj));
      if ((yyvsp[0].obj)) {
        auto* verilog_stmt = static_cast<VerilogStmt*>((yyvsp[0].obj));
        verilog_stmts->emplace_back(verilog_stmt);
      }
      (yyval.obj) = (yyvsp[-1].obj);
    }
#line 1811 "verilog/VerilogParse.cc"
    break;

    case 43:
#line 214 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1817 "verilog/VerilogParse.cc"
    break;

    case 44:
#line 216 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1823 "verilog/VerilogParse.cc"
    break;

    case 45:
#line 218 "verilog/VerilogParse.y"
    {
      yyerrok;
      (yyval.obj) = nullptr;
    }
#line 1829 "verilog/VerilogParse.cc"
    break;

    case 46:
#line 223 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 1835 "verilog/VerilogParse.cc"
    break;

    case 47:
#line 229 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1841 "verilog/VerilogParse.cc"
    break;

    case 48:
#line 231 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1847 "verilog/VerilogParse.cc"
    break;

    case 49:
#line 236 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1853 "verilog/VerilogParse.cc"
    break;

    case 50:
#line 238 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1859 "verilog/VerilogParse.cc"
    break;

    case 51:
#line 243 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1867 "verilog/VerilogParse.cc"
    break;

    case 52:
#line 247 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1875 "verilog/VerilogParse.cc"
    break;

    case 53:
#line 254 "verilog/VerilogParse.y"
    {
      (yyval.integer) = 0;
    }
#line 1883 "verilog/VerilogParse.cc"
    break;

    case 54:
#line 258 "verilog/VerilogParse.y"
    {
      (yyval.integer) = 0;
    }
#line 1891 "verilog/VerilogParse.cc"
    break;

    case 55:
#line 262 "verilog/VerilogParse.y"
    {
      (yyval.integer) = 0;
    }
#line 1899 "verilog/VerilogParse.cc"
    break;

    case 57:
#line 267 "verilog/VerilogParse.y"
    {
      (yyval.integer) = -(yyvsp[0].integer);
    }
#line 1905 "verilog/VerilogParse.cc"
    break;

    case 58:
#line 269 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-2].integer) + (yyvsp[0].integer);
    }
#line 1911 "verilog/VerilogParse.cc"
    break;

    case 59:
#line 271 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-2].integer) - (yyvsp[0].integer);
    }
#line 1917 "verilog/VerilogParse.cc"
    break;

    case 60:
#line 273 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-2].integer) * (yyvsp[0].integer);
    }
#line 1923 "verilog/VerilogParse.cc"
    break;

    case 61:
#line 275 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-2].integer) / (yyvsp[0].integer);
    }
#line 1929 "verilog/VerilogParse.cc"
    break;

    case 62:
#line 277 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-1].integer);
    }
#line 1935 "verilog/VerilogParse.cc"
    break;

    case 63:
#line 282 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1941 "verilog/VerilogParse.cc"
    break;

    case 64:
#line 287 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1947 "verilog/VerilogParse.cc"
    break;

    case 65:
#line 289 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1953 "verilog/VerilogParse.cc"
    break;

    case 66:
#line 294 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1961 "verilog/VerilogParse.cc"
    break;

    case 67:
#line 298 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 1969 "verilog/VerilogParse.cc"
    break;

    case 68:
#line 304 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 1975 "verilog/VerilogParse.cc"
    break;

    case 69:
#line 305 "verilog/VerilogParse.y"
    {
      auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-1].obj));
      auto* declaration
          = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>((yyvsp[-3].integer)), std::move(*dcl_args), (yyvsp[-2].integer));
      delete dcl_args;
      (yyval.obj) = declaration;
    }
#line 1986 "verilog/VerilogParse.cc"
    break;

    case 70:
#line 311 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 1992 "verilog/VerilogParse.cc"
    break;

    case 71:
#line 313 "verilog/VerilogParse.y"
    {
      auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-1].obj));
      std::pair<int, int> range = std::make_pair((yyvsp[-5].integer), (yyvsp[-3].integer));
      auto* declaration = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>((yyvsp[-8].integer)), std::move(*dcl_args),
                                                  (yyvsp[-7].integer), range);
      delete dcl_args;
      (yyval.obj) = declaration;
    }
#line 2004 "verilog/VerilogParse.cc"
    break;

    case 72:
#line 323 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kInput);
    }
#line 2010 "verilog/VerilogParse.cc"
    break;

    case 73:
#line 324 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kInout);
    }
#line 2016 "verilog/VerilogParse.cc"
    break;

    case 74:
#line 325 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kOutput);
    }
#line 2022 "verilog/VerilogParse.cc"
    break;

    case 75:
#line 326 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kSupply0);
    }
#line 2028 "verilog/VerilogParse.cc"
    break;

    case 76:
#line 327 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kSupply1);
    }
#line 2034 "verilog/VerilogParse.cc"
    break;

    case 77:
#line 328 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kTri);
    }
#line 2040 "verilog/VerilogParse.cc"
    break;

    case 78:
#line 329 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWand);
    }
#line 2046 "verilog/VerilogParse.cc"
    break;

    case 79:
#line 330 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWire);
    }
#line 2052 "verilog/VerilogParse.cc"
    break;

    case 80:
#line 331 "verilog/VerilogParse.y"
    {
      (yyval.integer) = static_cast<int>(VerilogDcl::DclType::kWor);
    }
#line 2058 "verilog/VerilogParse.cc"
    break;

    case 81:
#line 336 "verilog/VerilogParse.y"
    {
      auto* dcl_args = new std::vector<const char*>;
      dcl_args->push_back(static_cast<const char*>((yyvsp[0].obj)));
      (yyval.obj) = dcl_args;
    }
#line 2068 "verilog/VerilogParse.cc"
    break;

    case 82:
#line 342 "verilog/VerilogParse.y"
    {
      auto* dcl_args = static_cast<std::vector<const char*>*>((yyvsp[-2].obj));
      dcl_args->push_back(static_cast<const char*>((yyvsp[0].obj)));
      (yyval.obj) = dcl_args;
    }
#line 2078 "verilog/VerilogParse.cc"
    break;

    case 83:
#line 351 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].string);
    }
#line 2084 "verilog/VerilogParse.cc"
    break;

    case 84:
#line 353 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2090 "verilog/VerilogParse.cc"
    break;

    case 85:
#line 358 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-1].obj);
    }
#line 2096 "verilog/VerilogParse.cc"
    break;

    case 86:
#line 363 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2103 "verilog/VerilogParse.cc"
    break;

    case 87:
#line 366 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 2109 "verilog/VerilogParse.cc"
    break;

    case 88:
#line 370 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 2115 "verilog/VerilogParse.cc"
    break;

    case 89:
#line 371 "verilog/VerilogParse.y"
    {
      auto* lhs = static_cast<VerilogNetExpr*>((yyvsp[-3].obj));
      auto* rhs = static_cast<VerilogNetExpr*>((yyvsp[0].obj));

      auto* module_assign = verilog_reader->makeModuleAssign(lhs, rhs, (yyvsp[-2].integer));
      (yyval.obj) = module_assign;
    }
#line 2127 "verilog/VerilogParse.cc"
    break;

    case 90:
#line 382 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2133 "verilog/VerilogParse.cc"
    break;

    case 91:
#line 384 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2139 "verilog/VerilogParse.cc"
    break;

    case 92:
#line 388 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 2145 "verilog/VerilogParse.cc"
    break;

    case 93:
#line 389 "verilog/VerilogParse.y"
    {
      std::vector<std::unique_ptr<VerilogPortRefPortConnect>> inst_port_connection;
      if (auto* port_connection = static_cast<std::vector<std::unique_ptr<VerilogPortRefPortConnect>>*>((yyvsp[-2].obj)); port_connection) {
        inst_port_connection = std::move(*port_connection);
        delete port_connection;
      }

      auto* module_inst
          = verilog_reader->makeModuleInst(static_cast<const char*>((yyvsp[-6].string)), static_cast<const char*>((yyvsp[-4].string)),
                                           std::move(inst_port_connection), (yyvsp[-5].integer));
      (yyval.obj) = module_inst;
    }
#line 2160 "verilog/VerilogParse.cc"
    break;

    case 94:
#line 399 "verilog/VerilogParse.y"
    {
      (yyval.integer) = verilog_reader->get_line_no();
    }
#line 2166 "verilog/VerilogParse.cc"
    break;

    case 95:
#line 401 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2172 "verilog/VerilogParse.cc"
    break;

    case 98:
#line 412 "verilog/VerilogParse.y"
    {
      (yyval.integer) = (yyvsp[-1].integer);
    }
#line 2178 "verilog/VerilogParse.cc"
    break;

    case 100:
#line 418 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2184 "verilog/VerilogParse.cc"
    break;

    case 102:
#line 421 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2190 "verilog/VerilogParse.cc"
    break;

    case 103:
#line 427 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2197 "verilog/VerilogParse.cc"
    break;

    case 104:
#line 430 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[-2].obj);
    }
#line 2203 "verilog/VerilogParse.cc"
    break;

    case 105:
#line 436 "verilog/VerilogParse.y"
    {
      auto* port_connect_vec = new std::vector<std::unique_ptr<VerilogPortConnect>>;
      port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>((yyvsp[0].obj)));
      (yyval.obj) = port_connect_vec;
    }
#line 2213 "verilog/VerilogParse.cc"
    break;

    case 106:
#line 442 "verilog/VerilogParse.y"
    {
      auto* port_connect_vec = static_cast<std::vector<std::unique_ptr<VerilogPortConnect>>*>((yyvsp[-2].obj));
      port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>((yyvsp[0].obj)));
      (yyval.obj) = port_connect_vec;
    }
#line 2223 "verilog/VerilogParse.cc"
    break;

    case 107:
#line 454 "verilog/VerilogParse.y"
    {
      VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-2].string)));
      (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, nullptr);
    }
#line 2232 "verilog/VerilogParse.cc"
    break;

    case 108:
#line 459 "verilog/VerilogParse.y"
    {
      VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)));
      VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-1].string)));
      auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
      (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, net_expr);
    }
#line 2243 "verilog/VerilogParse.cc"
    break;

    case 109:
#line 466 "verilog/VerilogParse.y"
    {
      VerilogID* inst_port_id = new VerilogID(static_cast<char*>((yyvsp[-6].string)));
      VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-4].string)), static_cast<int>((yyvsp[-2].integer)));
      auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
      (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, net_expr);
    }
#line 2254 "verilog/VerilogParse.cc"
    break;

    case 110:
#line 473 "verilog/VerilogParse.y"
    {
      VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)));
      (yyval.obj) = verilog_reader->makePortConnect(inst_port_id, static_cast<VerilogNetExpr*>((yyvsp[-1].obj)));
    }
#line 2262 "verilog/VerilogParse.cc"
    break;

    case 111:
#line 478 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2268 "verilog/VerilogParse.cc"
    break;

    case 112:
#line 480 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2274 "verilog/VerilogParse.cc"
    break;

    case 113:
#line 483 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2280 "verilog/VerilogParse.cc"
    break;

    case 114:
#line 485 "verilog/VerilogParse.y"
    {
      (yyval.obj) = nullptr;
    }
#line 2286 "verilog/VerilogParse.cc"
    break;

    case 115:
#line 490 "verilog/VerilogParse.y"
    {
      VerilogID* net_id = static_cast<VerilogID*>((yyvsp[0].obj));
      auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
      (yyval.obj) = net_expr;
    }
#line 2296 "verilog/VerilogParse.cc"
    break;

    case 116:
#line 496 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2302 "verilog/VerilogParse.cc"
    break;

    case 117:
#line 498 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2308 "verilog/VerilogParse.cc"
    break;

    case 118:
#line 503 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2314 "verilog/VerilogParse.cc"
    break;

    case 119:
#line 505 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2320 "verilog/VerilogParse.cc"
    break;

    case 120:
#line 507 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2326 "verilog/VerilogParse.cc"
    break;

    case 121:
#line 512 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[0].string)));
    }
#line 2332 "verilog/VerilogParse.cc"
    break;

    case 122:
#line 517 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-3].string)), static_cast<int>((yyvsp[-1].integer)));
    }
#line 2338 "verilog/VerilogParse.cc"
    break;

    case 123:
#line 522 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogID(static_cast<char*>((yyvsp[-5].string)), static_cast<int>((yyvsp[-3].integer)),
                                                  static_cast<int>((yyvsp[-1].integer)));
    }
#line 2344 "verilog/VerilogParse.cc"
    break;

    case 124:
#line 527 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<const char*>((yyvsp[0].constant)), verilog_reader->get_line_no());
    }
#line 2350 "verilog/VerilogParse.cc"
    break;

    case 125:
#line 532 "verilog/VerilogParse.y"
    {
      auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>((yyvsp[-1].obj));
      auto* verilog_net_expr_concat = verilog_reader->makeVerilogNetExpr(std::move(*verilog_id_concat), verilog_reader->get_line_no());
      delete verilog_id_concat;

      (yyval.obj) = verilog_net_expr_concat;
    }
#line 2362 "verilog/VerilogParse.cc"
    break;

    case 126:
#line 543 "verilog/VerilogParse.y"
    {
      auto* verilog_id_concat = new Vector<std::unique_ptr<VerilogNetExpr>>();
      verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>((yyvsp[0].obj)));

      (yyval.obj) = verilog_id_concat;
    }
#line 2373 "verilog/VerilogParse.cc"
    break;

    case 127:
#line 550 "verilog/VerilogParse.y"
    {
      auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>((yyvsp[-2].obj));
      verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>((yyvsp[0].obj)));

      (yyval.obj) = verilog_id_concat;
    }
#line 2384 "verilog/VerilogParse.cc"
    break;

    case 128:
#line 560 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2390 "verilog/VerilogParse.cc"
    break;

    case 129:
#line 562 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2396 "verilog/VerilogParse.cc"
    break;

    case 130:
#line 564 "verilog/VerilogParse.y"
    {
      (yyval.obj) = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>((yyvsp[0].obj)), verilog_reader->get_line_no());
    }
#line 2402 "verilog/VerilogParse.cc"
    break;

    case 131:
#line 566 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2408 "verilog/VerilogParse.cc"
    break;

    case 132:
#line 568 "verilog/VerilogParse.y"
    {
      (yyval.obj) = (yyvsp[0].obj);
    }
#line 2414 "verilog/VerilogParse.cc"
    break;

#line 2418 "verilog/VerilogParse.cc"

    default:
      break;
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
  YY_SYMBOL_PRINT("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp ? yytable[yyi] : yydefgoto[yylhs]);
  }

  goto yynewstate;

/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE(yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus) {
    ++yynerrs;
#if !YYERROR_VERBOSE
    yyerror(yyscanner, verilog_reader, YY_("syntax error"));
#else
#define YYSYNTAX_ERROR yysyntax_error(&yymsg_alloc, &yymsg, yyssp, yytoken)
    {
      char const* yymsgp = YY_("syntax error");
      int yysyntax_error_status;
      yysyntax_error_status = YYSYNTAX_ERROR;
      if (yysyntax_error_status == 0)
        yymsgp = yymsg;
      else if (yysyntax_error_status == 1) {
        if (yymsg != yymsgbuf)
          YYSTACK_FREE(yymsg);
        yymsg = YY_CAST(char*, YYSTACK_ALLOC(YY_CAST(YYSIZE_T, yymsg_alloc)));
        if (!yymsg) {
          yymsg = yymsgbuf;
          yymsg_alloc = sizeof yymsgbuf;
          yysyntax_error_status = 2;
        } else {
          yysyntax_error_status = YYSYNTAX_ERROR;
          yymsgp = yymsg;
        }
      }
      yyerror(yyscanner, verilog_reader, yymsgp);
      if (yysyntax_error_status == 2)
        goto yyexhaustedlab;
    }
#undef YYSYNTAX_ERROR
#endif
  }

  if (yyerrstatus == 3) {
    /* If just tried and failed to reuse lookahead token after an
       error, discard it.  */

    if (yychar <= YYEOF) {
      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        YYABORT;
    } else {
      yydestruct("Error: discarding", yytoken, &yylval, yyscanner, verilog_reader);
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
  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;

/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3; /* Each real token shifted decrements this.  */

  for (;;) {
    yyn = yypact[yystate];
    if (!yypact_value_is_default(yyn)) {
      yyn += YYTERROR;
      if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR) {
        yyn = yytable[yyn];
        if (0 < yyn)
          break;
      }
    }

    /* Pop the current state because it cannot handle the error token.  */
    if (yyssp == yyss)
      YYABORT;

    yydestruct("Error: popping", yystos[yystate], yyvsp, yyscanner, verilog_reader);
    YYPOPSTACK(1);
    yystate = *yyssp;
    YY_STACK_PRINT(yyss, yyssp);
  }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Shift the error token.  */
  YY_SYMBOL_PRINT("Shifting", yystos[yyn], yyvsp, yylsp);

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
  yyerror(yyscanner, verilog_reader, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY) {
    /* Make sure we have latest lookahead translation.  See comments at
       user semantic actions for why this is necessary.  */
    yytoken = YYTRANSLATE(yychar);
    yydestruct("Cleanup: discarding lookahead", yytoken, &yylval, yyscanner, verilog_reader);
  }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK(yylen);
  YY_STACK_PRINT(yyss, yyssp);
  while (yyssp != yyss) {
    yydestruct("Cleanup: popping", yystos[+*yyssp], yyvsp, yyscanner, verilog_reader);
    YYPOPSTACK(1);
  }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE(yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE(yymsg);
#endif
  return yyresult;
}
#line 571 "verilog/VerilogParse.y"

void verilog_error(yyscan_t scanner, ista::VerilogReader* verilog_reader, const char* str)
{
  char* error_msg
      = Str::printf("Error found in line %lu in verilog file %s\n", verilog_reader->get_line_no(), verilog_reader->get_file_name().c_str());
  LOG_ERROR << error_msg;
}
