#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*          This file is part of the program and software framework          *#
#*                    UG --- Ubquity Generator Framework                     *#
#*                                                                           *#
#*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      *#
#*            Copyright (C) 2021 by Zuse Institute Berlin,                   *#
#*            licensed under LGPL version 3 or later.                        *#
#*            Commercial licenses are available through <licenses@zib.de>    *#
#*                                                                           *#
#* This code is free software; you can redistribute it and/or                *#
#* modify it under the terms of the GNU Lesser General Public License        *#
#* as published by the Free Software Foundation; either version 3            *#
#* of the License, or (at your option) any later version.                    *#
#*                                                                           *#
#* This program is distributed in the hope that it will be useful,           *#
#* but WITHOUT ANY WARRANTY; without even the implied warranty of            *#
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *#
#* GNU Lesser General Public License for more details.                       *#
#*                                                                           *#
#* You should have received a copy of the GNU Lesser General Public License  *#
#* along with this program.  If not, see <http://www.gnu.org/licenses/>.     *#
#*                                                                           *#
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#

#@file    Makefile
#@brief   Makefile for parallelization of SCIP-SDP
#@author  Stefan Heinz
#@author	 Tristan Gally

UG_VERSION		:=	1.0.0
#-----------------------------------------------------------------------------
# COMM
#-----------------------------------------------------------------------------
#COMM            = pth
COMM            = cpp11
THREADS		= 4
#-----------------------------------------------------------------------------
# paths
#-----------------------------------------------------------------------------
SCIPDIR         =       ../../lib/scip
UGDIR           =       ../..
UGLIBDIR			 =			lib
UGMISDPDIR		 =			$(realpath .)
#-----------------------------------------------------------------------------
# include default project Makefile from SCIP
#-----------------------------------------------------------------------------
include $(SCIPDIR)/make/make.project
#-----------------------------------------------------------------------------
# set the UGLIB
#-----------------------------------------------------------------------------
UGLIB           =       ugscip-$(UG_VERSION).$(BASE).$(COMM)
#-----------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------

ifeq ($(COMM), pth)
MAINNAME	=	fscip
endif
ifeq ($(COMM), cpp11)
MAINNAME	=	fscip
endif
ifeq ($(COMM), mpi)
MAINNAME	=	parascip
endif
UGOBJ		=	misdp_plugins.o
MAINOBJ	=	scipsdp/SdpVarmapper.o \
			scipsdp/SdpVarfixer.o \
			scipsdp/cons_sdp.o \
			scipsdp/cons_savedsdpsettings.o \
			scipsdp/cons_savesdpsol.o \
			scipsdp/relax_sdp.o \
			scipsdp/disp_sdpiterations.o \
			scipsdp/disp_sdpavgiterations.o \
			scipsdp/disp_sdpfastsettings.o \
			scipsdp/disp_sdppenalty.o \
			scipsdp/disp_sdpunsolved.o \
			scipsdp/prop_sdpredcost.o \
			scipsdp/branch_sdpmostfrac.o \
			scipsdp/branch_sdpmostinf.o \
			scipsdp/branch_sdpobjective.o \
			scipsdp/branch_sdpinfobjective.o \
			scipsdp/heur_sdpfracdiving.o \
			scipsdp/heur_sdprand.o \
			scipsdp/reader_cbf.o \
			scipsdp/prop_sdpobbt.o \
			scipsdp/prop_companalcent.o \
			sdpi/sdpi.o \
			sdpi/sdpi_mosek \
			sdpi/sdpi_sdpa \
			sdpi/sdpi_dsdp \
			sdpi/lapack_dsdp \
			sdpi/lapack_sdpa \
			sdpi/sdpsolchecker.o \
			scipsdpgithash.o

MAINCXXOBJ 	=	scipsdp/main.o \
			scipsdp/objreader_sdpa.o \
			scipsdp/ScipStreamBuffer.o

UGSRC		=	$(addprefix $(SRCDIR)/,$(UGOBJ:.o=.cpp))
MAINSRC		=	$(addprefix $(UGLIBDIR)/include/scipsdpsrc/,$(MAINOBJ:.o=.c))
MAINSRC         +=      $(addprefix $(UGLIBDIR)/include/scipsdpsrc/,$(MAINCXXOBJ:.o=.cpp))
MAINDEP		=	$(SRCDIR)/depend.cmain.$(OPT)

MAIN		=	$(MAINNAME).$(BASE).$(LPS).$(SDPS)$(EXEEXTENSION)
MAINFILE	=	$(BINDIR)/$(MAIN)
MAINSHORTLINK	=	$(BINDIR)/$(MAINNAME)
UGOBJFILES	=	$(addprefix $(OBJDIR)/,$(UGOBJ))

# ---------------------------------------------------------------------
# SCIPSDP-linkings
# ---------------------------------------------------------------------
MAKESOFTLINKS	=	true

SOFTLINKS	 =	$(UGLIBDIR)/include/scipsdpsrc
SOFTLINKS	+=	$(UGLIBDIR)/static/libscipsdp.$(SDPS).$(BASE).$(STATICLIBEXT)
SCIPSDPFLAGS	 = $(UGLIBDIR)/static/libscipsdp.$(SDPS).$(BASE).$(STATICLIBEXT)
SCIPSDPINCLUDE	 = -I$(UGLIBDIR)/include/scipsdpsrc
SDPIINSTMSG	+=	"  -> \"scipsdpsrc\" is the path to the SCIP-SDP source directory, e.g., \"<SCIPSDP-path>/src\".\n"
SDPIINSTMSG	+=	" -> \"libscipsdp.*\" is the path to the SCIP-SDP library, e.g., \"<SCIPSDP-path>/lib/static/libdsdp-3.1.0.$(SDPS).linux.x64_64.$(BASE).$(STATICLIBEXT)\".\n"

ifeq ($(SDPS), msk)
BITEXT     =  $(word 2, $(subst _, ,$(ARCH)))
SCIPSDPFLAGS	+= $(UGLIBDIR)/shared/libmosek$(BITEXT).$(SHAREDLIBEXT)
SCIPSDPFLAGS	+= -Wl,-rpath=$(dir $(realpath $(UGLIBDIR)/shared/libmosek$(BITEXT).$(SHAREDLIBEXT)))
SCIPSDPFLAGS	+= -llapack -lblas
SOFTLINKS	+=	$(UGLIBDIR)/shared/libmosek$(BITEXT).$(SHAREDLIBEXT)
SDPIINSTMSG	+=	" -> \"libmosek$(BITEXT).*\" is the path to the MOSEK library, e.g., \"<MOSEK-path>/8/tools/platform/linux64x86/bin/libmosek$(BITEXT).$(SHAREDLIBEXT)\".\n"
endif

ifeq ($(SDPS), dsdp)
SCIPSDPFLAGS	 = $(UGLIBDIR)/static/libscipsdp.$(SDPS).$(BASE).$(STATICLIBEXT)
SCIPSDPFLAGS	+= $(UGLIBDIR)/static/libdsdp.$(STATICLIBEXT)
SCIPSDPFLAGS	+= -llapack -lblas
SOFTLINKS	+=	$(UGLIBDIR)/static/libdsdp.$(STATICLIBEXT)
SDPIINSTMSG	+=	" -> \"libdsdp.*\" is the path to the DSDP library, e.g., \"<DSDP-path>/lib/libdsdp.$(STATICLIBEXT)\""
endif

ifeq ($(SDPS), sdpa)
SOFTLINKS	+=	$(UGLIBDIR)/static/libsdpa.$(STATICLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/shared/libsdpa.$(SHAREDLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/static/libdmumps.$(STATICLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/static/libmumps_common.$(STATICLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/static/libpord.$(STATICLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/static/libmpiseq.$(STATICLIBEXT)
SOFTLINKS	+=	$(UGLIBDIR)/shared/libopenblas.$(SHAREDLIBEXT).0
SDPIINSTMSG	+=	" -> \"libsdpa.*\" is the path to the SDPA library, e.g., \"<SDPA-path>/lib/libsdpa.a\".\n"
SDPIINSTMSG	+=	" -> \"libdmumps.*\" is the path to the dmumps library, e.g., \"<SDPA-path>/mumps/build/lib/libdmumps.$(STATICLIBEXT)\".\n"
SDPIINSTMSG	+=	" -> \"libdmumps_common.*\" is the path to the mumps_common library, e.g., \"<SDPA-path>/mumps/build/lib/libmumps_common.$(STATICLIBEXT)\".\n"
SDPIINSTMSG	+=	" -> \"libdmumps_common.*\" is the path to the pord library, e.g., \"<SDPA-path>/mumps/build/lib/libpord.$(STATICLIBEXT)\".\n"
SDPIINSTMSG	+=	" -> \"libdmumps.*\" is the path to the mpiseq library, e.g., \"<SDPA-path>/mumps/build/libseq/libmpiseq.$(STATICLIBEXT)\".\n"
SCIPSDPFLAGS	+= -L$(UGLIBDIR)/$(LIBEXTTYPE) -lsdpa
SCIPSDPFLAGS	+= $(UGLIBDIR)/static/libdmumps.$(STATICLIBEXT)
SCIPSDPFLAGS	+= $(UGLIBDIR)/static/libmumps_common.$(STATICLIBEXT)
SCIPSDPFLAGS	+= $(UGLIBDIR)/static/libpord.$(STATICLIBEXT)
SCIPSDPFLAGS	+= $(UGLIBDIR)/static/libmpiseq.$(STATICLIBEXT)
SCIPSDPFLAGS	+= $(UGLIBDIR)/shared/libopenblas.$(SHAREDLIBEXT).0
SCIPSDPFLAGS	+= -lgfortran
SCIPSDPFLAGS	+= -lgomp
endif

ifeq ($(SDPS), none)
SCIPSDPFLAGS	 = $(UGLIBDIR)/static/libscipsdp.none.a
SCIPSDPFLAGS	+= -llapack -lblas
endif

LINKSMARKERFILE	=	$(UGLIBDIR)/linkscreated.$(LPS)-$(LPSOPT).$(SDPS).$(OSTYPE).$(ARCH).$(COMP)$(LINKLIBSUFFIX).$(ZIMPL)-$(ZIMPLOPT).$(IPOPT)-$(IPOPTOPT).$(GAMS)

LASTSETTINGS    =       $(OBJDIR)/make.lastsettings

CFLAGS		+=	-I$(SCIPDIR)/src
ifeq ($(COMM), pth)
ifeq ($(COMP), intel)
CXX             =       icpc
LINKCXX         =       icpc
else
CXX             =       g++
LINKCXX         =       g++
endif # comp
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_PTH -g
endif
ifeq ($(COMM), cpp11)
ifeq ($(COMP), intel)
CXX             =       icpc
LINKCXX         =       icpc
else
CXX             =       g++
LINKCXX         =       g++
endif # comp
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_CPP11 -g
endif
ifeq ($(COMM), mpi)
CC              =       mpicc
CXX             =       mpicxx
LINKCXX         =       mpicxx
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_MPI_WORLD -DMPICH_IGNORE_CXX_SEEK -g
endif

LDFLAGS		+=	-lpthread

#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

ifeq ($(VERBOSE),false)
.SILENT:	$(MAINFILE) $(UGOBJFILES) $(MAINOBJFILES) $(MAINCXXOBJFILES) $(MAINSHORTLINK)
endif

.PHONY: all
all:            $(SCIPDIR) $(MAINFILE) $(MAINSHORTLINK)

.PHONY: preprocess
preprocess:
		@$(SHELL) -ec 'if test ! -e $(LINKSMARKERFILE) ; \
			then \
				echo "-> generating necessary links" ; \
				$(MAKE) -j1 $(LINKSMARKERFILE) ; \
			fi'
		@$(MAKE) touchexternal

.PHONY: lint
lint:		$(MAINSRC)
		-rm -f lint.out
		$(SHELL) -ec 'for i in $^; \
			do \
			echo $$i; \
			$(LINT) $(SCIPDIR)/lint/scip.lnt +os\(lint.out\) -u -zero \
			$(FLAGS) -UNDEBUG -UWITH_READLINE -UROUNDING_FE $$i; \
			done'

.PHONY: scip
scip:
		@$(MAKE) -C $(SCIPDIR) libs $^

.PHONY: doc
doc:
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/scip.css);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/pictures/scippy.png);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/pictures/miniscippy.png);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/scipfooter.html footer.html);
		cd doc; $(DOXY) scheduler.dxy

$(MAINSHORTLINK):	$(MAINFILE)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(MAINFILE)) $(notdir $@)

$(OBJDIR):
		@-mkdir -p $(OBJDIR)

$(BINDIR):
		@-mkdir -p $(BINDIR)

$(UGLIBDIR):
		@-mkdir -p $(UGLIBDIR)

$(UGLIBDIR)/include: $(UGLIBDIR)
		@-mkdir -p $(UGLIBDIR)/include

$(UGLIBDIR)/static: $(UGLIBDIR)
		@-mkdir -p $(UGLIBDIR)/static

$(UGLIBDIR)/shared: $(UGLIBDIR)
		@-mkdir -p $(UGLIBDIR)/shared

$(CHECKDIR):
		@-mkdir -p $(CHECKDIR)

$(LINKSMARKERFILE):
		@$(MAKE) links

.PHONY: links
links:		| $(UGLIBDIR) $(UGLIBDIR)/include $(UGLIBDIR)/static $(UGLIBDIR)/shared echosoftlinks $(SOFTLINKS)
		@rm -f $(LINKSMARKERFILE)
		@echo "this is only a marker" > $(LINKSMARKERFILE)

.PHONY: echosoftlinks
echosoftlinks:
		@echo
		@echo "- Current settings: SDPS=$(SDPS) LPS=$(LPS) SUFFIX=$(LINKLIBSUFFIX) OSTYPE=$(OSTYPE) ARCH=$(ARCH) COMP=$(COMP)"
		@echo
		@echo "* UG-SCIPSDP needs some softlinks to external programs, in particular, SDP-solvers."
		@echo "* Please insert the paths to the corresponding directories/libraries below."
		@echo "* The links will be installed in the 'lib' directory."
		@echo "* For more information and if you experience problems see the INSTALL file."
		@echo
		@echo -e $(SDPIINSTMSG)

.PHONY: $(SOFTLINKS)
$(SOFTLINKS):
ifeq ($(MAKESOFTLINKS), true)
		@$(SHELL) -ec 'if test ! -e $@ ; \
			then \
				DIRNAME=`dirname $@` ; \
				BASENAMEA=`basename $@ .$(STATICLIBEXT)` ; \
				BASENAMESO=`basename $@ .$(SHAREDLIBEXT)` ; \
				echo ; \
				echo "- preparing missing soft-link \"$@\":" ; \
				if test -e $$DIRNAME/$$BASENAMEA.$(SHAREDLIBEXT) ; \
				then \
					echo "* this soft-link is not necessarily needed since \"$$DIRNAME/$$BASENAMEA.$(SHAREDLIBEXT)\" already exists - press return to skip" ; \
				fi ; \
				if test -e $$DIRNAME/$$BASENAMESO.$(STATICLIBEXT) ; \
				then \
					echo "* this soft-link is not necessarily needed since \"$$DIRNAME/$$BASENAMESO.$(STATICLIBEXT)\" already exists - press return to skip" ; \
				fi ; \
				echo "> Enter soft-link target file or directory for \"$@\" (return if not needed): " ; \
				echo -n "> " ; \
				cd $$DIRNAME ; \
				eval $(READ) TARGET ; \
				cd $(UGMISDPDIR) ; \
				if test "$$TARGET" != "" ; \
				then \
					echo "-> creating softlink \"$@\" -> \"$$TARGET\"" ; \
					rm -f $@ ; \
					$(LN_s) $$TARGET $@ ; \
				else \
					echo "* skipped creation of softlink \"$@\". Call \"make links\" if needed later." ; \
				fi ; \
				echo ; \
			fi'
endif

.PHONY: test
test:		$(CHECKDIR)
		-@test -d check/testset || { \
		echo "-> Creating testset directory"; \
		cd check && mkdir testset; }
		@-(cd check/testset && ln -fs $(realpath $(UGLIBDIR)/include/scipsdpsrc)/../check/testset/short.test);
		@-(cd check/testset && ln -fs $(realpath $(UGLIBDIR)/include/scipsdpsrc)/../check/testset/short.solu);
		@-(ln -fs $(realpath $(UGLIBDIR)/include/scipsdpsrc)/../instances/);
		@-(cd check && ln -fs $(realpath $(SCIPDIR))/check/check.sh);
		@-(cd check && ln -fs $(realpath $(SCIPDIR))/check/evalcheck.sh);
		@-(cd check && ln -fs $(realpath $(SCIPDIR))/check/check.awk);
		@-(cd check && ln -fs $(realpath $(SCIPDIR))/check/getlastprob.awk);
		@-(cd check && ln -fs $(realpath $(SCIPDIR))/check/configuration_set.sh);
ifeq ($(COMM),pth)
		cd check; \
		$(SHELL) ./check_fscip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS) $(SDPSETTINGS);
endif
ifeq ($(COMM),cpp11)
		cd check; \
		$(SHELL) ./check_fscip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS) $(SDPSETTINGS);
endif
ifeq ($(COMM),mpi)
		cd check; \
		$(SHELL) ./check_parascip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS) $(SDPSETTINGS);
endif

# include local targets
-include $(SCIPDIR)/make/local/make.targets

.PHONY: clean
clean:		$(OBJDIR)
ifneq ($(OBJDIR),)
		@-(rm -f $(OBJDIR)/*.o && rmdir $(OBJDIR));
		@echo "-> remove main objective files"
endif
		@-rm -f $(MAINFILE) $(MAINLINK) $(MAINSHORTLINK)
		@echo "-> remove binary"

.PHONY: tags
tags:
		rm -f TAGS; ctags -e src/*.c src/*.h $(SCIPDIR)/src/scip/*.c $(SCIPDIR)/src/scip/*.h;

.PHONY: depend
depend:		$(SCIPDIR)
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(MAINSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z\_]*\).c|$$\(OBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(MAINDEP)'

-include $(LASTSETTINGS)

.PHONY: touchexternal
touchexternal:  $(CPOPTDEP) $(OBJDIR)
ifneq ($(CPOPT),$(LAST_CPOPT))
		@-touch $(CPOPTSRC)
endif
		@-rm -f $(LASTSETTINGS)
		@echo "LAST_CPOPT=$(CPOPT)" >> $(LASTSETTINGS)

-include	$(MAINDEP)

$(MAINFILE):	preprocess $(BINDIR) $(OBJDIR) $(UGOBJFILES) $(MAINOBJFILES) $(MAINCXXOBJFILES)
		@echo "-> linking $@"
		$(LINKCXX) $(UGOBJFILES) $(MAINOBJFILES) $(MAINCXXOBJFILES)  \
		$(LINKCXX_L)$(UGDIR)/lib $(LINKCXX_l)$ $(UGLIB) \
		$(LINKCXX_L)$(SCIPDIR)/lib/static $(LINKCXX_l)$(SCIPLIB)$(LINKLIBSUFFIX) \
                $(LINKCXX_l)$(OBJSCIPLIB)$(LINKLIBSUFFIX) $(LINKCXX_l)$(LPILIB)$(LINKLIBSUFFIX) $(LINKCXXSCIPALL) $(LINKCXX_l)$(NLPILIB)$(LINKLIBSUFFIX) \
                $(OFLAGS) $(LPSLDFLAGS) \
		$(LDFLAGS) $(SCIPSDPFLAGS) $(LINKCXX_o)$@

$(OBJDIR)/%.o:	$(SRCDIR)/%.c
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CFLAGS) $(SCIPSDPINCLUDE) -c $< $(CC_o)$@

$(OBJDIR)/%.o:	$(SRCDIR)/%.cpp
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) $(SCIPSDPINCLUDE) -c $< $(CXX_o)$@

#---- EOF --------------------------------------------------------------------
