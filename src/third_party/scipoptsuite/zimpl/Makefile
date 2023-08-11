#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*   File....: Makefile                                                      *
#*   Name....: Zimpl Makefile                                                *
#*   Author..: Thorsten Koch                                                 *
#*   Copyright by Author, All rights reserved                                *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*
#* Copyright (C) 2005-2022 by Thorsten Koch <koch@zib.de>
#*
#* This program is free software; you can redistribute it and/or
#* modify it under the terms of the GNU General Public License
#* as published by the Free Software Foundation; either version 2
#* of the License, or (at your option) any later version.
#*
#* This program is distributed in the hope that it will be useful,
#* but WITHOUT ANY WARRANTY; without even the implied warranty of
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#* GNU General Public License for more details.
#*
#* You should have received a copy of the GNU General Public License
#* along with this program; if not, write to the Free Software
#* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#*
#
.PHONY:		all depend clean lint doc doxygen check valgrind libdbl coverage

ARCH		:=	$(shell uname -m | \
			sed \
			-e 's/sun../sparc/' \
			-e 's/i.86/x86/' \
			-e 's/i86pc/x86/' \
			-e 's/[0-9]86/x86/' \
			-e 's/amd64/x86_64/' \
			-e 's/9000..../hppa/' \
			-e 's/00........../pwr4/' \
			-e 's/arm.*/arm/' \
			-e 's/aarch64/arm/')
OSTYPE          :=      $(shell uname -s | \
                        tr '[:upper:]' '[:lower:]' | \
			tr '/' '_' | \
                        sed \
			-e s/cygwin.*/cygwin/ \
                        -e s/irix../irix/ \
                        -e s/windows.*/windows/ \
                        -e s/mingw.*/mingw/)

HOSTNAME	:=      $(shell uname -n | tr '[:upper:]' '[:lower:]')

VERSION		=  3.5.3
VERBOSE		=	false
SHARED		=	false
STATIC		=	false
ZLIB		=	true
LINK		=	static
OPT		=	dbg
COMP		=	gnu
CC		=	gcc
CC_o		= 	-o #
LINKCC   	=	gcc
LINKCC_o 	=	-o #the white space is important
LIBEXT		= 	.a
YACC		=	bison
LEX		=	flex
DCC		=	gcc
LINT		=	pclp64_linux # _debug
CPPCHECK	=	cppcheck
AR		=	ar cr
AR_o	   	=
RANLIB		=	ranlib
DOXY		=	doxygen
VALGRIND	=	valgrind --tool=memcheck --leak-check=full \
			--leak-resolution=high --show-reachable=yes
ANALYZER	=	scan-build

SRCDIR		=	src/zimpl
BINDIR		=	bin
LIBDIR		=	lib
LINTCONF	=	/opt/pclint/config

CPPFLAGS	=	-I$(SRCDIR)/.. -DVERSION='"$(VERSION)"'
CFLAGS		=	-O
LDFLAGS		=	-lgmp -lm
YFLAGS		=	-d -t -v
LFLAGS		=	-d
ARFLAGS		=
DFLAGS		=	-MM
# if changing these flags, also update ADD_C_FLAGS in CMakeLists.txt
GCCWARN		=	-Wall \
			-Wextra \
			-Wno-unknown-pragmas \
			-Wno-nonnull-compare \
			-Wno-cast-qual \
			-Wpointer-arith \
			-Wcast-align \
			-Wwrite-strings \
			-Winline \
			-Wshadow \
			-Wstrict-prototypes \
			-Wmissing-prototypes \
			-Wmissing-declarations \
			-Wmissing-noreturn \
			-Wstrict-overflow=4 \
			-Wduplicated-branches \
			-Wsuggest-attribute=pure \
			-Wsuggest-attribute=const \
			-Wsuggest-attribute=noreturn \
			-Wsuggest-attribute=format \
			-Wno-attributes \
			-Wno-unused-function \
			-Wno-unused-parameter \
			-Wno-nonnull-compare \
			-fno-omit-frame-pointer \
			-fstack-protector-strong \
			-fsanitize=address \
			-fsanitize=undefined \
			-fsanitize=shift \
			-fsanitize=shift-exponent \
			-fsanitize=shift-base \
			-fsanitize=integer-divide-by-zero \
			-fsanitize=unreachable \
			-fsanitize=vla-bound \
			-fsanitize=null \
			-fsanitize=bounds \
			-fsanitize=alignment \
			-fsanitize=object-size \
			-fsanitize=float-cast-overflow \
			-fsanitize=nonnull-attribute \
			-fsanitize=returns-nonnull-attribute \
			-fsanitize=bool \
			-fsanitize=enum \
			-fsanitize=signed-integer-overflow \
#			-Wsuggest-attribute=malloc \
#			-Wsuggest-attribute=cold \
#			-fsanitize=pointer-overflow \
#			-fsanitize=builtin 
#			-fsanitize=float-divide-by-zero \
#			-fsanitize=bounds-strict \

ifeq ($(ZLIB),true)
LDFLAGS		+=	-lz
else
CPPFLAGS	+=	-DWITHOUT_ZLIB
endif

ifeq ($(SHARED),true)
LINK		=	shared
endif
ifeq ($(STATIC),true)
LINK		=	static
endif

BASE		=	$(OSTYPE).$(ARCH).$(COMP).$(OPT)
OBJDIR		=	obj/O.$(OSTYPE).$(ARCH).$(COMP).$(LINK).$(OPT)
NAME		=	zimpl
BINNAME		=	$(NAME)-$(VERSION).$(OSTYPE).$(ARCH).$(COMP).$(LINK).$(OPT)
LIBNAME		=	$(NAME)-$(VERSION).$(BASE)

LIBRARY		=	$(LIBDIR)/lib$(LIBNAME)$(LIBEXT)
LIBRARYDBL	=	$(LIBDIR)/lib$(LIBNAME).dbl$(LIBEXT)
LIBLINK		=	$(LIBDIR)/lib$(NAME).$(BASE)$(LIBEXT)
LIBDBLLINK	=	$(LIBDIR)/lib$(NAME).$(BASE).dbl$(LIBEXT)

BINARY		=	$(BINDIR)/$(BINNAME)
BINARYDBL	=	$(BINDIR)/$(BINNAME).dbl
BINLINK		=	$(BINDIR)/$(NAME).$(BASE)
BINSHORTLINK	=	$(BINDIR)/$(NAME)
DEPEND		=	$(SRCDIR)/depend

#-----------------------------------------------------------------------------

OBJECT  	=       zimpl.o xlpglue.o zlpglue.o \
			ratlpstore.o ratlpfwrite.o ratmpswrite.o ratmstwrite.o \
			ratordwrite.o ratpresolve.o ratqbowrite.o
LIBBASE		=	blkmem.o bound.o code.o conname.o define.o elem.o entry.o \
			hash.o heap.o idxset.o inst.o iread.o list.o \
			load.o local.o metaio.o mmlparse2.o mmlscan.o mono.o \
			mshell.o prog.o random.o rdefpar.o source.o \
			setempty.o setpseudo.o setlist.o setrange.o setprod.o \
			setmulti.o set4.o stmt.o stkchk.o strstore2.o symbol.o \
			term2.o tuple.o vinst.o zimpllib.o
LIBOBJ		=	$(LIBBASE) gmpmisc.o numbgmp.o
LIBDBLOBJ	=	$(LIBBASE) numbdbl.o
OBJXXX		=	$(addprefix $(OBJDIR)/,$(OBJECT))
LIBXXX		=	$(addprefix $(OBJDIR)/,$(LIBOBJ))
LIBDBLXXX	=	$(addprefix $(OBJDIR)/,$(LIBDBLOBJ))
OBJSRC		=	$(addprefix $(SRCDIR)/,$(OBJECT:.o=.c))
LIBSRC		=	$(addprefix $(SRCDIR)/,$(LIBOBJ:.o=.c)) #(SRCDIR)/numbdbl.c

#-----------------------------------------------------------------------------
include make/make.$(OSTYPE).$(ARCH).$(COMP).$(OPT)
-include make/local/make.$(HOSTNAME)
-include make/local/make.$(HOSTNAME).$(COMP)
-include make/local/make.$(HOSTNAME).$(COMP).$(OPT)
#-----------------------------------------------------------------------------

FLAGS           +=      $(USRFLAGS)
OFLAGS          +=      $(USROFLAGS)
CFLAGS          +=      $(USRCFLAGS) $(USROFLAGS)
LDFLAGS         +=      $(USRLDFLAGS)
ARFLAGS         +=      $(USRARFLAGS)
DFLAGS          +=      $(USRDFLAGS)


ifeq ($(VERBOSE),false)
.SILENT:	$(LIBRARY) $(LIBLINK) $(BINARY) $(BINLINK) $(BINSHORTLINK) \
		$(SRCDIR)/mmlparse2.c $(SRCDIR)/mmlscan.c $(OBJXXX) $(LIBXXX) $(LIBDBLXXX)
endif

all:		$(LIBRARY) $(LIBLINK) $(BINARY) $(BINLINK) $(BINSHORTLINK)

double:		$(LIBRARYDBL) $(LIBDBLLINK) $(BINARYDBL) # $(BINLINK) $(BINSHORTLINK)

$(LIBLINK):	$(LIBRARY)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(LIBRARY)) $(notdir $@)

$(LIBDBLLINK):	$(LIBRARYDBL)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(LIBRARYDBL)) $(notdir $@)

$(BINLINK) $(BINSHORTLINK):	$(BINARY)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(BINARY)) $(notdir $@)

$(BINARY):	$(OBJDIR) $(BINDIR) $(OBJXXX) $(LIBRARY)
		@echo "-> linking $@"
ifeq ($(COMP), msvc)
		$(LINKCC) $(CFLAGS) $(OBJXXX) $(LIBRARY) $(LDFLAGS) $(LINKCC_o)$@
else
		$(LINKCC) $(CFLAGS) $(OBJXXX) -L$(LIBDIR) -l$(LIBNAME) $(LDFLAGS) $(LINKCC_o)$@
endif

$(BINARYDBL):	$(OBJDIR) $(BINDIR) $(OBJXXX) $(LIBRARYDBL)
		@echo "-> linking $@"
		$(LINKCC) $(CFLAGS) $(OBJXXX) -L$(LIBDIR) -l$(LIBNAME).dbl $(LDFLAGS) $(CC_o)$@

$(LIBRARY):	$(OBJDIR) $(LIBDIR) $(LIBXXX)
		@echo "-> generating library $@"
		-rm -f $(LIBRARY)
		$(AR) $(AR_o)$@ $(LIBXXX) $(ARFLAGS)
ifneq ($(RANLIB),)
		$(RANLIB) $@
endif

libdbl:		$(LIBRARYDBL) $(LIBDBLLINK)

$(LIBRARYDBL):	$(OBJDIR) $(LIBDIR) $(LIBDBLXXX)
		@echo "-> generating library $@"
		-rm -f $(LIBRARYDBL)
		$(AR) $@ $(LIBDBLXXX) $(ARFLAGS)
		$(RANLIB) $@

$(SRCDIR)/mmlparse2.c:	$(SRCDIR)/mmlparse2.y $(SRCDIR)/mme.h
		@echo "-> generating yacc parser $@"
		$(YACC) $(YFLAGS) -o $@ $<

$(SRCDIR)/mmlscan.c:	$(SRCDIR)/mmlscan.l $(SRCDIR)/mme.h
		@echo "-> generating lex scanner $@"
		$(LEX) $(LFLAGS) -o$@ $<

lint:		$(OBJSRC) $(LIBSRC)
		$(LINT) $(LINTCONF)/co-gcc.lnt $(SRCDIR)/project2.lnt \
		-I$(LINTCONF) -I$(SRCDIR)/.. -dNO_MSHELL -dVERSION='"$(VERSION)"' $^

cppcheck:	$(OBJSRC) $(LIBSRC)
		$(CPPCHECK) $(CPPFLAGS) -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/include/x86_64-linux-gnu/bits -I/usr/include/x86_64-linux-gnu/sys -I/usr/include/linux -I/usr/lib/gcc/x86_64-linux-gnu/5/include --inline-suppr --suppressions-list=src/cppcheck.txt --enable=warning,style,performance,portability,information $^

doc:
		cd doc; make -f Makefile

doxygen:
		cd doc; $(DOXY) $(NAME).dxy

check:
		export GLIBC_TUNABLES=glibc.malloc.check=3
		cd check; \
		/bin/sh ./check.sh ../$(BINARY)

valgrind:
		cd check; \
		/bin/sh ./check.sh "$(VALGRIND) ../$(BINARY)"

analyze:
		make clean
		$(ANALYZER) make

coverage:
		-ln -s ../../src $(OBJDIR)
		-mkdir -p gcov
		lcov -d $(OBJDIR) -z
		make OPT=gcov check
		lcov -d $(OBJDIR) -c >gcov/z.capture
		lcov -d $(OBJDIR) -r gcov/z.capture "*mmlscan.c" "*mmlparse2.c" >gcov/zimpl.capture
		genhtml -o gcov gcov/zimpl.capture
		-rm gcov/z.capture

$(OBJDIR):
		@echo "** creating directory \"$@\""
		@-mkdir -p $(OBJDIR)

$(LIBDIR):
		@echo "** creating directory \"$@\""
		@-mkdir -p $(LIBDIR)

$(BINDIR):
		@echo "** creating directory \"$@\""
		@-mkdir -p $(BINDIR)

clean:
		-rm -rf $(OBJDIR)/* $(BINARY) $(LIBRARY) $(LIBRARYDBL) $(LIBLINK) $(BINLINK) $(BINSHORTLINK)

depend:
		$(SHELL) -ec '$(DCC) $(DFLAGS) $(CPPFLAGS) $(OBJSRC) $(LIBSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o|$$\(OBJDIR\)/\1.o|g'\'' \
		>$(DEPEND)'

-include	$(DEPEND)

$(OBJDIR)/%.o:	$(SRCDIR)/%.c
		@echo "-> compiling $@"
		$(CC) $(CPPFLAGS) $(CFLAGS) -c $< $(CC_o)$@

# --- EOF ---------------------------------------------------------------------
