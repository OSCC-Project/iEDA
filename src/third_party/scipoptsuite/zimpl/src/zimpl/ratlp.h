/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: ratlp.h                                                       */
/*   Name....: Rational Number LP Storage Library                            */
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
#ifndef _RATLP_H_
#define _RATLP_H_

#ifndef __GMP_H__
#error "Need to include gmp.h before ratlp.h"
#endif
#ifndef _RATLPTYPES_H_
#error "Need to include ratlptypes.h before ratlp.h"
#endif
#ifndef _MME_H_
#error "Need to include mme.h before ratlp.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* lpstore.c
 */
//lint -sem(        lps_alloc, 1p, @p == 1) 
extern Lps*         lps_alloc(char const* name) expects_NONNULL returns_NONNULL;
//lint -sem(        lps_free, custodial(1), inout(1), 1p == 1) 
extern void         lps_free(Lps* lp) expects_NONNULL;
//lint -sem(        lps_getsos, 1p == 1, 2p, r_null) 
extern Sos*         lps_getsos(Lps const* lp, char const* name) expects_NONNULL is_PURE;
//lint -sem(        lps_getvar, 1p == 1, 2p, r_null) 
extern Var*         lps_getvar(Lps const* lp, char const* name) expects_NONNULL is_PURE;
//lint -sem(        lps_getcon, 1p == 1, 2p, r_null) 
extern Con*         lps_getcon(Lps const* lp, char const* name) expects_NONNULL is_PURE;
//lint -sem(        lps_getnzo, 1p == 1, 2p == 1, 3p == 1, r_null) 
extern Nzo*         lps_getnzo(Lps const* lp, Con const* con, Var const* var) expects_NONNULL;
//lint -sem(        lps_addsos, inout(1), 1p == 1, 2p, chneg(4), @p) 
extern Sos*         lps_addsos(Lps* lp, char const* name, SosType type, int priority) expects_NONNULL returns_NONNULL; 
//lint -sem(        lps_addsse, inout(1), 1p == 1, inout(2), 2p == 1, 3p == 1) 
extern void         lps_addsse(Sos* sos, Var* var, const mpq_t weight) expects_NONNULL; 
//lint -sem(        lps_addvar, inout(1), 1p == 1, 2p, @p) 
extern Var*         lps_addvar(Lps* lp, char const* name) expects_NONNULL returns_NONNULL; 
//lint -sem(        lps_delvar, inout(1), 1p == 1, 2p == 1) 
extern void         lps_delvar(Lps* lp, Var* var) expects_NONNULL;
//lint -sem(        lps_addcon, inout(1), 1p == 1, 2p, @p) 
extern Con*         lps_addcon(Lps* lp, char const* name) expects_NONNULL returns_NONNULL;
//lint -sem(        lps_delcon, inout(1), 1p == 1, 2p == 1) 
extern void         lps_delcon(Lps* lp, Con* con) expects_NONNULL;
//lint -sem(        lps_addnzo, inout(1), 1p == 1, inout(2), 2p == 1, inout(3), 3p == 1) 
extern void         lps_addnzo(Lps* lp, Con* con, Var* var, const mpq_t value) expects_NONNULL;
//lint -sem(        lps_delnzo, inout(1), 1p == 1, inout(2), 2p == 1) 
extern void         lps_delnzo(Lps* lp, Nzo* nzo) expects_NONNULL;
//lint -sem(        lps_addtoobjterm, inout(1), 1p == 1, 2p == 1) 
extern void         lps_addtoobjterm(Lps* lp, Term const* term) expects_NONNULL;
//lint -sem(        lps_addterm, inout(1), 1p == 1, inout(2), 2p == 1, 3p == 1) 
extern void         lps_addterm(Lps* lp, Con* con, Term const* term) expects_NONNULL;
//lint -sem(        lps_stat, 1p == 1) 
extern void         lps_stat(Lps const* lp) expects_NONNULL;
//lint -sem(        lps_setval, 1p == 1) 
extern void         lps_setval(Nzo* nzo, const mpq_t value) expects_NONNULL;
//lint -sem(        lps_getval, 1p == 1) 
extern void         lps_getval(Nzo const* nzo, mpq_t value) expects_NONNULL;
//lint -sem(        lps_setdir, inout(1), 1p == 1) 
extern void         lps_setdir(Lps* lp, LpDirect direct) expects_NONNULL;
//lint -sem(        lps_getdir, inout(1), 1p == 1) 
extern LpDirect     lps_getdir(Lps* lp) expects_NONNULL;
//lint -sem(        lps_setprobname, inout(1), 1p == 1, 2p) 
extern void         lps_setprobname(Lps* lp, char const* name) expects_NONNULL;
//lint -sem(        lps_setobjname, inout(1), 1p == 1, 2p) 
extern bool         lps_setobjname(Lps* lp, char const* name) expects_NONNULL;
//lint -sem(        lps_setrhsname, inout(1), 1p == 1, 2p) 
extern void         lps_setrhsname(Lps* lp, char const* name) expects_NONNULL;
//lint -sem(        lps_setbndname, inout(1), 1p == 1, 2p) 
extern void         lps_setbndname(Lps* lp, char const* name) expects_NONNULL;
//lint -sem(        lps_setrngname, inout(1), 1p == 1, 2p) 
extern void         lps_setrngname(Lps* lp, char const* name) expects_NONNULL;
//lint -sem(        lps_getcost, 1p == 1) 
extern void         lps_getcost(Var const* var, mpq_t cost) expects_NONNULL;
//lint -sem(        lps_setcost, inout(1), 1p == 1) 
extern void         lps_setcost(Var* var, const mpq_t cost) expects_NONNULL;
//lint -sem(        lps_haslower, 1p == 1) 
extern bool         lps_haslower(Var const* var) expects_NONNULL is_PURE;
//lint -sem(        lps_getlower, 1p == 1) 
extern void         lps_getlower(Var const* var, mpq_t lower) expects_NONNULL;
//lint -sem(        lps_setlower, inout(1), 1p == 1) 
extern void         lps_setlower(Var* var, const mpq_t lower) expects_NONNULL;
//lint -sem(        lps_hasupper, 1p == 1) 
extern bool         lps_hasupper(Var const* var) expects_NONNULL is_PURE;
//lint -sem(        lps_getupper, 1p == 1) 
extern void         lps_getupper(Var const* var, mpq_t upper) expects_NONNULL;
//lint -sem(        lps_setupper, inout(1), 1p == 1) 
extern void         lps_setupper(Var* var, const mpq_t upper) expects_NONNULL;
//lint -sem(        lps_getclass, 1p == 1) 
extern VarClass     lps_getclass(Var const* var) expects_NONNULL is_PURE;
//lint -sem(        lps_setclass, inout(1), 1p == 1) 
extern void         lps_setclass(Var* var, VarClass vclass) expects_NONNULL;
//lint -sem(        lps_setlhs, inout(1), 1p == 1) 
extern void         lps_setlhs(Con* con, const mpq_t lhs) expects_NONNULL;
//lint -sem(        lps_setrhs, inout(1), 1p == 1) 
extern void         lps_setrhs(Con* con, const mpq_t rhs) expects_NONNULL;
//lint -sem(        lps_setcontype, inout(1), 1p == 1) 
extern void         lps_setcontype(Con* con, ConType type) expects_NONNULL;
//lint -sem(        lps_contype, 1p == 1) 
extern ConType      lps_contype(Con const* con) expects_NONNULL is_PURE;
//lint -sem(        lps_vartype, 1p == 1) 
extern VarType      lps_vartype(Var const* var) expects_NONNULL is_PURE;
//lint -sem(        lps_is_binary, 1p == 1) 
extern bool         lps_is_binary(Var const* var) expects_NONNULL is_PURE;
//lint -sem(        lps_getlhs, 1p == 1) 
extern void         lps_getlhs(Con const* con, mpq_t lhs) expects_NONNULL;
//lint -sem(        lps_getrhs, 1p == 1) 
extern void         lps_getrhs(Con const* con, mpq_t rhs) expects_NONNULL;
//lint -sem(        lps_varname, 1p == 1, @p) 
extern char const*  lps_varname(Var const* var) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        lps_setvartype, inout(1), 1p == 1) 
extern void         lps_setvartype(Var* con, VarType type) expects_NONNULL;
//lint -sem(        lps_flags, 1p == 1) 
extern unsigned int lps_flags(Con const* con) expects_NONNULL is_PURE;
//lint -sem(        lps_addflags, inout(1), 1p == 1) 
extern void         lps_addflags(Con* con, unsigned int flags) expects_NONNULL;
//lint -sem(        lps_setscale, inout(1), 1p == 1) 
extern void         lps_setscale(Con* con, const mpq_t scale) expects_NONNULL;
//lint -sem(        lps_setpriority, inout(1), 1p == 1) 
extern void         lps_setpriority(Var* var, int priority) expects_NONNULL;
//lint -sem(        lps_setvalue, inout(1), 1p == 1) 
extern void         lps_setvalue(Var* var, const mpq_t value) expects_NONNULL;
//lint -sem(        lps_setstartval, inout(1), 1p == 1) 
extern void         lps_setstartval(Var* var, const mpq_t startval) expects_NONNULL;
//lint -sem(        lps_setnamelen, inout(1), 1p == 1, chneg(2)) 
extern void         lps_setnamelen(Lps* lp, int name_len) expects_NONNULL;
//lint -sem(        lps_setindictaor, inout(1), 1p == 1, inout(2), 2p == 1) 
extern void         lps_setindicator(Con* con, Var* var, bool on_true) expects_NONNULL;
//lint -sem(        lps_write, 1p == 1, inout(2), 2p == 1, 4p) 
extern void         lps_write(Lps const* lp, FILE* fp, LpFormat format, char const* format_options, char const* text) expects_NONNULL12;
//lint -sem(        lps_transtable, 1p == 1, inout(2), 2p == 1, 4p) 
extern void         lps_transtable(Lps const* lp, FILE* fp, LpFormat format, char const* head) expects_NONNULL;
//lint -sem(        lps_clearobj, inout(1), 1p == 1) 
extern void         lps_clearobj(Lps const* lp) expects_NONNULL;
//lint -sem(        lps_scale, inout(1), 1p == 1) 
extern void         lps_scale(Lps const* lp) expects_NONNULL;
//lint -sem(        lps_has_sos, 1p == 1) 
extern bool         lps_has_sos(Lps const* lp) expects_NONNULL is_PURE;
//lint -sem(        lps_con_sumup, 1p == 1) 
extern bool         lps_con_sumup(Con const* con, mpq_t sum) expects_NONNULL;

/* ratlpfwrite.c
 */
//lint -sem(        lpf_write, 1p == 1, inout(2), 2p == 1) 
extern void         lpf_write(Lps const* lp, FILE* fp, LpFormat format, char const* text) expects_NONNULL12;

/* ratmpswrite.c
 */
//lint -sem(        lpf_write, 1p == 1, inout(2), 2p == 1) 
extern void         mps_write(Lps const* lp, FILE* fp, char const* text) expects_NONNULL12;

/* ratordwrite.c
*/
//lint -sem(        lps_ordwrite, 1p == 1, inout(2), 2p == 1) 
extern void         lps_orderfile(Lps const* lp, FILE* fp, LpFormat format, char const* text) expects_NONNULL12;

/* ratmstwrite.c
*/
//lint -sem(        lps_mstwrite, 1p == 1, inout(2), 2p == 1) 
extern void         lps_mstfile(Lps const* lp, FILE* fp, LpFormat format, char const* text) expects_NONNULL12;

/* ratqubowrite.c
 */
//lint -sem(        qubo_write, 1p == 1, inout(2), 2p == 1, 4p) 
extern void         qbo_write(Lps const* lp, FILE* fp, LpFormat format, char const* format_options, char const* text) expects_NONNULL12;

#if 0 // Not used anymore ??? 
/* ratpresolve.c
 */
//lint -sem(        lps_presolve, 1p == 1, chneg(2)) 
extern PSResult     lps_presolve(Lps* lp, int verbose_level) expects_NONNULL;
#endif
   
#ifdef __cplusplus
}
#endif
#endif // _RATLP_H_ 



