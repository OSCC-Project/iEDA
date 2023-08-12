/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mme.h                                                         */
/*   Name....: Mathematical Modelling Engine                                 */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#ifndef _MME_H_
#define _MME_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ZIMPL_VERSION  353

/* the following is not in code.h because code.h needs mme.h anyway,
 * but we also need these declaratons.
 */
enum code_type
{
   CODE_ERR = 0, CODE_NUMB, CODE_STRG, CODE_NAME, CODE_TUPLE,
   CODE_SET, CODE_TERM, CODE_BOOL, CODE_SIZE,
   CODE_IDXSET, CODE_LIST, CODE_VOID, CODE_ENTRY, CODE_VARCLASS, CODE_CONTYPE,
   CODE_RDEF, CODE_RPAR, CODE_BITS, CODE_SYM, CODE_DEF, CODE_BOUND
};

enum symbol_type     { SYM_ERR = 0, SYM_NUMB, SYM_STRG, SYM_SET, SYM_VAR };

typedef enum symbol_type         SymbolType;
typedef struct symbol            Symbol;
typedef enum code_type           CodeType;
typedef struct code_node         CodeNode;

typedef CodeNode*              (*Inst)(CodeNode* self);

typedef struct entry             Entry;

typedef struct list_element      ListElem;
typedef struct list              List;

typedef enum   var_type      VarType;  /* From ratlptypes.h */
typedef struct mono          Mono;     /* From mono.h */


#define SYMBOL_NAME_INTERNAL  "@@"

#define VERB_QUIET    0
#define VERB_NORMAL   1
#define VERB_VERBOSE  2
#define VERB_CHATTER  3
#define VERB_DEBUG    5

/* zimpllib.c
 */
extern int          verbose;
//lint -function(exit, zpl_exit) 
extern void zpl_exit(int retval) is_NORETURN;

/* source.c
 */
//lint -sem(show_source, inout(1), 1p == 1, 2p, chneg(3)) 
extern void show_source(FILE* fp, char const* text, int column) expects_NONNULL;

/* vinst.c
 */
extern void interns_init(void);
extern void interns_exit(void);

#define Min(a, b)    (((a) <= (b)) ? (a) : (b))
#define Sgn(a)       (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))

/* Directory separator, so we could redefine it for Windoof.
 */
#ifndef DIRSEP
#define DIRSEP '/'
#endif /* DIRSEP */

#ifndef NDEBUG
#define SID unsigned int sid;
#define SID_set(p, id)  (p->sid = id)
#define SID_del(p)      (p->sid = 0xdeadda7a)
#define SID_ok(p, id)   (p->sid == id)
#define SID_set2(p, id) (p.sid = id)
#define SID_del2(p)     (p.sid = 0xdeadda7a)
#define SID_ok2(p, id)  (p.sid == id)
#else /* NDEBUG */
#define SID              /* */
#define SID_set(p, sid)  /* */
#define SID_del(p)       /* */
#define SID_ok(p, id)    true
#define SID_set2(p, sid) /* */
#define SID_del2(p)      /* */
#define SID_ok2(p, id)   true
#endif /* NDEBUG */

#define DISPERSE(x) (1664525U * (x) + 1013904223U)

#ifdef TRACE
#define Trace(fname) fprintf(stderr, "Trace: %s\n", fname);
#else
#define Trace(fname) /* */
#endif /* TRACE */

// Here is a macro to switch off a particular warning from clang in the code
// Somebody should burn in hell for making it so difficult
#ifdef __clang__
#define MAKE_STRING(x) #x
#define CLANG_WARN_OFF(warningoption) \
   _Pragma("clang diagnostic push") \
   _Pragma(MAKE_STRING(clang diagnostic ignored #warningoption))
#define CLANG_WARN_ON _Pragma("clang diagnostic pop")
#else
#define CLANG_WARN_OFF(warningoption) /* */
#define CLANG_WARN_ON /* */
#endif // __clang__   
   
#ifdef __cplusplus
}
#endif

#endif /* _MME_H_ */
