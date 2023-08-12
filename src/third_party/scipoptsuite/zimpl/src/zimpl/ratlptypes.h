/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: ratlptypes.h                                                  */
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
#ifndef _RATLPTYPES_H_
#define _RATLPTYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

enum con_type        { CON_FREE    = 0, CON_LHS,   CON_RHS,   CON_RANGE, CON_EQUAL };
enum var_type        { VAR_FREE    = 0, VAR_LOWER, VAR_UPPER, VAR_BOXED, VAR_FIXED };
enum sos_type        { SOS_ERR     = 0, SOS_TYPE1, SOS_TYPE2 };
enum var_class       { VAR_CON     = 0, VAR_IMP,   VAR_INT };
enum lp_direct       { LP_MIN      = 0, LP_MAX };
enum lp_type         { LP_ERR      = 0, LP_LP, LP_IP };
enum lp_format       { LP_FORM_ERR = 0, LP_FORM_LPF, LP_FORM_HUM, LP_FORM_MPS, LP_FORM_RLP, LP_FORM_PIP, LP_FORM_QBO };

#if 0 /* not used anymore ??? */
enum presolve_result
{
   PRESOLVE_ERROR = 0, PRESOLVE_OKAY, PRESOLVE_INFEASIBLE,
   PRESOLVE_UNBOUNDED, PRESOLVE_VANISHED
};
typedef enum presolve_result PSResult;

#endif
   
typedef struct nonzero       Nzo;
typedef struct variable      Var;
typedef struct constraint    Con;
typedef struct soset         Sos;
typedef struct soselement    Sse;
typedef struct lpstorage     Lps;

typedef enum   con_type      ConType;
typedef enum   sos_type      SosType;
typedef enum   var_class     VarClass;
typedef enum   lp_direct     LpDirect;
typedef enum   lp_type       LpType;
typedef enum   lp_format     LpFormat;

#define LP_FLAG_CON_SCALE     0x0001
#define LP_FLAG_CON_SEPAR     0x0002
#define LP_FLAG_CON_CHECK     0x0004
#define LP_FLAG_CON_INDIC     0x0008
#define LP_FLAG_CON_QUBO      0x0010
#define LP_FLAG_CON_PENALTY1  0x0020
#define LP_FLAG_CON_PENALTY2  0x0040
#define LP_FLAG_CON_PENALTY3  0x0080
#define LP_FLAG_CON_PENALTY4  0x0100
#define LP_FLAG_CON_PENALTY5  0x0200
#define LP_FLAG_CON_PENALTY6  0x0400

#define HAS_LOWER(var)  ((var)->type != VAR_FREE && (var)->type != VAR_UPPER)
#define HAS_UPPER(var)  ((var)->type != VAR_FREE && (var)->type != VAR_LOWER)
#define HAS_LHS(con)    ((con)->type != CON_FREE && (con)->type != CON_RHS)
#define HAS_RHS(con)    ((con)->type != CON_FREE && (con)->type != CON_LHS)

#ifdef __cplusplus
}
#endif
#endif /* _RATLPTYPES_H_ */
