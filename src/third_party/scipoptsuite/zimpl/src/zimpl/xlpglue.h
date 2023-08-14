/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: xlpglue.h                                                     */
/*   Name....: Glue between numb/term and ratlp                              */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2003-2022 by Thorsten Koch <koch@zib.de>
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
#ifndef _XLPGLUE_H_
#define _XLPGLUE_H_

#ifndef _RATLPTYPES_H_
#error "Need to include ratlptypes.h before xlpglue.h"
#endif

#ifndef _MME_H_
#error "Need to include mme.h before xlpglue.h"
#endif

#ifndef ATTRIBUTE_H_
#error "Need to include attribute.h before xlpglue.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(    xlp_alloc, 1p, @P >= malloc(1)) 
extern Lps*     xlp_alloc(char const* name, bool need_startval, void* user_data) expects_NONNULL1 returns_NONNULL;
//lint -sem(    xlp_free, custodial(1), inout(1), 1p == 1) 
extern void     xlp_free(Lps* lp) expects_NONNULL;
//lint -sem(    xlp_conname_exists, 1p == 1, 2p) 
extern bool     xlp_conname_exists(Lps const* lp, char const* conname) expects_NONNULL is_PURE;
//lint -sem(    xlp_addcon_term, inout(1), 1p == 1, 2p, 4p == 1, 5p == 1, 7p == 1) 
extern bool     xlp_addcon_term(Lps* lp, char const* name, ConType type,
   Numb const* lhs, Numb const* rhs, unsigned int flags, Term const* term) expects_NONNULL;
//lint -sem(    xlp_addvar, inout(1), 1p == 1, 2p, 4p == 1, 5p == 1, 6p == 1, 7p == 1, @p) 
extern Var*     xlp_addvar(Lps* lp, char const* name, VarClass usevarclass,
   Bound const* lower, Bound const* upper, Numb const* priority, Numb const* startval) expects_NONNULL returns_NONNULL;
//lint -sem(    xlp_addsos_term, inout(1), 1p == 1, 2p, 4p == 1, 5p == 1) 
extern int      xlp_addsos_term(Lps* lp, char const* name, SosType type, Numb const* priority, Term const* term) expects_NONNULL;
//lint -sem(    xlp_getvarname, 1p == 1, 2p == 1, @p) 
char const*     xlp_getvarname(Lps const* lp, Var const* var) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(    xlp_getclass, 1p == 1, 2p == 1) 
extern VarClass xlp_getclass(Lps const* lp, Var const* var) expects_NONNULL is_PURE;
//lint -sem(    xlp_getlower, 1p == 1, 2p == 1, @p == 1) 
extern Bound*   xlp_getlower(Lps const* lp, Var const* var) expects_NONNULL returns_NONNULL;
//lint -sem(    xlp_getupper, 1p == 1, 2p == 1, @p == 1) 
extern Bound*   xlp_getupper(Lps const* lp, Var const* var) expects_NONNULL returns_NONNULL;
//lint -sem(    xlp_setobj, inout(1), 1p == 1, 2p) 
extern bool     xlp_setobj(Lps* lp, char const* name, bool minimize) expects_NONNULL;
//lint -sem(    xlp_addtermtoobj, inout(1), 1p == 1, 2p == 1) 
extern void     xlp_addtoobj(Lps* lp, Term const* term) expects_NONNULL;
   
#ifdef __cplusplus
}
#endif
#endif // _XLPGLUE_H 








