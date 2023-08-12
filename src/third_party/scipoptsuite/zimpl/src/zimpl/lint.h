/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: lint.h                                                        */
/*   Name....: Lint defines                                                  */
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
#ifndef _LINT_H_
#define _LINT_H_
  
#if 0  // not needed anymore for pc-lint 1.2 ?

#ifdef __cplusplus
extern "C" {
#endif

/* Unfortunately strdup() is not a POSIX function.
 */
/*lint -sem(strdup, 1p, @P >= malloc(1)) */ 
extern char* strdup(char const* s);

/* It is not clear if isinf() and isnan() are already POSIX
 * or only in the next Draft.
 */
extern int isinf(double);
extern int isnan(double);
extern int isfinite(double);
extern int finite(double); /* This is probably not POSIX */

/*lint -esym(757, optarg, optind, opterr, optopt) */
/*lint -sem(getopt, 1n > 0 && 2p && 3p) */
extern int getopt(int argc, char* const argv[], char const* optstring);
extern char* optarg;
extern int optind;
extern int opterr;
extern int optopt;

/*lint -function(fopen, popen) */
extern FILE* popen(const char *command, const char *type);
/*lint -function(fclose, pclose) */
/*lint -esym(534, pclose) */
extern int   pclose(FILE *stream);
/*lint -sem(fsync, 1n >= 0, @n <= 0) */
extern int   fsync(int fd);

/* zlib support
 */
/*lint -esym(534,gzclose) */

typedef void* gzFile;

/*lint -sem(  gzopen, 1p && 2p, r_null) */
extern gzFile gzopen(const char *path, const char *mode);
/*lint -sem(  gzread, 1p == 1 && 2p) */
extern int    gzread(gzFile file, void* buf, unsigned len);
/*lint -sem(  gzwrite, 1p == 1 && 2p) */
extern int    gzwrite(gzFile file, void const* buf, unsigned len);
/*lint -sem(  gzputs, 1p == 1 && 2p) */
extern int    gzputs(gzFile file, const char *s);
/*lintx -sem(  gzgets, 1p == 1 && 2p > 0 && 2P <= 3n && 3n > 0, @P == 2P || @P == 0) */
/*lint -function(fgets(1), gzgets(2)) */
/*lint -function(fgets(2), gzgets(3)) */
/*lint -function(fgets(3), gzgets(1)) */
/*lint -function(fgets(r), gzgets(r)) */
extern char*  gzgets(gzFile file, char *buf, int len);
/*lint -function(fgetc, gzgetc) */
extern int    gzgetc(gzFile file);
/*lint -sem(  gzclose, 1p == 1) */
extern int    gzclose(gzFile file);

#ifdef __cplusplus
}
#endif

#endif // 0

#endif /* _LINT_H_ */



