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
#@brief   Makefile for a parallel steiner tree problem
#@author  Yuji Shinano

UG_VERSION		:=	1.0.0
#-----------------------------------------------------------------------------
# COMM
#-----------------------------------------------------------------------------
COMM		= cpp11
TSUG            = false
#-----------------------------------------------------------------------------
# paths
#-----------------------------------------------------------------------------
SCIPDIR         =       ../../lib/scip
UGDIR		=	../..
STPDIR		= 	../../lib/scip/applications/STP
#-----------------------------------------------------------------------------
# include default project Makefile from SCIP
#-----------------------------------------------------------------------------
include $(SCIPDIR)/make/make.project
#-----------------------------------------------------------------------------
# set the UGLIB
#-----------------------------------------------------------------------------
ifeq ($(TSUG), true)
UGLIB		=	tsugscip-$(UG_VERSION).$(BASE).$(COMM)
else
UGLIB		=	ugscip-$(UG_VERSION).$(BASE).$(COMM)
endif # tsug
#-----------------------------------------------------------------------------
# default settings
#-----------------------------------------------------------------------------
TEST            =      short
TIME            =      3600
SETTINGS        =      default
VERBOSE         =      false
OPT             =      opt
COMP            =      gnu
LPS             =      spx2
DET             =      false
ZLIB            =      true
THREADS         =      4
#-----------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------

ifeq ($(COMM), pth)
MAINNAME	=	fscip
endif
ifeq ($(COMM), cpp11)
ifeq ($(TSUG), true)
MAINNAME	=	tsfscip
else
MAINNAME	=	fscip
endif
endif
ifeq ($(COMM), mpi)
ifeq ($(TSUG), true)
MAINNAME	=	tsparascip
else
MAINNAME	=	parascip
endif
endif
ifeq ($(TSUG), true)
UGOBJ		=	stp_ts_plugins.o
else
UGOBJ		=	stp_plugins.o
endif
MAINOBJ         =       bidecomposition.o \
			completegraph.o \
			cons_stp.o \
			cons_stpcomponents.o \
			dpborder_base.o \
			dpborder_core.o \
			dpborder_util.o \
			dpterms_base.o \
			dpterms_core.o \
			dpterms_util.o \
			dualascent.o \
			enumeration.o \
			extreduce_base.o \
			extreduce_bottleneck.o \
			extreduce_contract.o \
			extreduce_core.o \
			extreduce_data.o \
			extreduce_dbg.o \
			extreduce_dist.o \
			extreduce_extmst.o \
			extreduce_extmstbiased.o \
			extreduce_extspg.o \
			extreduce_mldists.o \
			extreduce_redcosts.o \
			extreduce_util.o \
			graph_history.o \
			graph_base.o \
			graph_delpseudo.o \
			graph_edge.o \
			graph_grid.o \
			graph_load.o \
			graph_mcut.o \
			graph_node.o \
			graph_path.o \
			graph_sdpath.o \
			graph_pcbase.o \
			graph_save.o \
			graph_stats.o \
			graph_sub.o \
			graph_tpath.o \
			graph_trans.o \
			graph_util.o \
			graph_vnoi.o \
			heur_ascendprune.o \
			heur_lurkprune.o \
			heur_slackprune.o \
			heur_tm.o \
			heur_local.o \
			heur_prune.o \
			heur_rec.o \
			mincut.o \
			misc_stp.o \
			mst.o \
			probdata_stp.o \
			event_bestsol.o \
			dialog_stp.o \
			reader_stp.o \
			reader_gr.o \
			redcosts.o \
			pricer_stp.o \
			reduce_base.o \
			reduce_alt.o \
			reduce_bnd.o \
			reduce_da.o \
			reduce_ext.o \
			reduce_path.o \
			reduce_pcsimple.o \
			reduce_sd.o \
			reduce_sdcomp.o \
			reduce_sdgraph.o \
			reduce_sepa.o \
			reduce_sdutil.o \
			reduce_simple.o \
			reduce_sol.o \
			reduce_termsepa.o \
			reduce_termsepada.o \
			reduce_termsepafull.o \
			reduce_util.o \
			relax_stp.o \
			relax_stpenum.o \
			relax_stpdp.o \
			sepaspecial.o \
			shortestpath.o \
			solpool.o \
			solstp.o \
			stpprioqueue.o \
			stptest_base.o \
			stptest_da.o \
			stptest_extreduce.o \
			stptest_extutils.o \
			stptest_graph.o \
			stptest_graphpath.o \
			stptest_heurlocal.o \
			stptest_heurtm.o \
			stptest_misc.o \
			stptest_pcreduce.o \
			stptest_reducesd.o \
			stptest_reducesepa.o \
			stptest_reduceutils.o \
			solhistory.o \
			substpsolver.o \
			prop_stp.o \
			branch_stp.o \
			validate.o

UGSRC		=	$(addprefix $(SRCDIR)/,$(UGOBJ:.o=.cpp))
MAINSRC		=	$(addprefix $(STPDIR)/src/,$(MAINOBJ:.o=.c))
MAINSRC         +=      $(addprefix $(SRCDIR)/,$(MAINCXXOBJ:.o=.cpp))
MAINDEP		=	$(SRCDIR)/depend.cmain.$(OPT)

MAIN		=	$(MAINNAME).$(BASE).$(LPS)$(EXEEXTENSION)
MAINFILE	=	$(BINDIR)/$(MAIN)
MAINSHORTLINK	=	$(BINDIR)/$(MAINNAME)
UGOBJFILES	=	$(addprefix $(OBJDIR)/,$(UGOBJ))
MAINOBJFILES	=	$(addprefix $(OBJDIR)/,$(MAINOBJ))
MAINCXXOBJFILES =	$(addprefix $(OBJDIR)/,$(MAINCXXOBJ))

ifeq ($(CPOPT),true)
FLAGS           +=      -DWITH_CPOPTIMIZER -DIL_STD -I$(LIBDIR)/cpinc -I$(LIBDIR)/cpxinc
FLAGS           +=      -I$(LIBDIR)/ilconcertinc
LDFLAGS         +=       $(LIBDIR)/libcp.a $(LIBDIR)/libconcert.a  $(LIBDIR)/libcplex.a -lpthread
endif

# LASTSETTINGS    =       $(OBJDIR)/make.lastsettings

# maybe setting the include path to SCIP's application directory is enough
CFLAGS		+=	-I$(SCIPDIR)/src -DWITH_UG -I$(STPDIR)/src
ifeq ($(COMM), pth)
CXX             =       g++
LINKCXX         =       g++
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_PTH -I$(STPDIR)/src
endif
ifeq ($(COMM), cpp11)
CXX             =       g++
LINKCXX         =       g++
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_CPP11 -I$(STPDIR)/src
endif
ifeq ($(COMM), mpi)
CC              =       mpicc
CXX             =       mpicxx
LINKCXX         =       mpicxx
CXXFLAGS	+=	-I$(UGDIR)/src -D_COMM_MPI_WORLD -DMPICH_IGNORE_CXX_SEEK -I$(STPDIR)/src -D_MUTEX_CPP11
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

$(CHECKDIR):
		@-mkdir -p $(CHECKDIR)

.PHONY: test
test:		$(STPDIR)
		@-(ln -fs $(STPDIR)/data);
		@-(cd check && ln -fs ../$(STPDIR)/check/testset);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_solufile.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/evalcheck.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/evaluate.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/check.awk);

ifeq ($(COMM),pth)
		cd check; \
		$(SHELL) ./check_fscip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS);
endif
ifeq ($(COMM),cpp11)
ifeq ($(TSUG), true)
		cd check; \
		$(SHELL) ./check_tsfscip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS);
else
		cd check; \
		$(SHELL) ./check_fscip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS);
endif
endif
ifeq ($(COMM),mpi)
		cd check; \
		$(SHELL) ./check_parascip.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)).$(HOSTNAME) $(TIME) $(MEM) $(CONTINUE) $(LOCK) $(LPS) $(THREADS);
endif

# include local targets
-include $(SCIPDIR)/make/local/make.targets

.PHONY: testclusterf
testclusterf:	$(STPDIR)
		@-(cd check && ln -fs ../$(STPDIR)/check/testset);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_cluster.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_logfiles.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/evalcheck_cluster.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/check_cluster.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_tmpfile_setup_scip.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/allcmpres.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/check.awk);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/cmpres.awk);
		cd check; \
		$(SHELL) ./check_cluster_fscip.sh $(TEST) $(notdir $(MAINFILE)) $(SETTINGS) \
		$(notdir $(MAINFILE)) $(OUTPUTDIR) $(TIME) $(NODES) $(MEM) \
		$(THREADS) $(FEASTOL) $(LPS) $(DISPFREQ) $(CONTINUE) \
		$(QUEUETYPE) $(QUEUE) $(PPN) $(CLIENTTMPDIR) \
		$(NOWAITCLUSTER) $(EXCLUSIVE) $(PERMUTE) $(SEEDS) $(GLBSEEDSHIFT) $(STARTPERM) $(DEBUGTOOL) $(REOPT) $(OPTCOMMAND) $(SETCUTOFF) $(VISUALIZE) $(CLUSTERNODES) $(SLURMACCOUNT) $(PYTHON);

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
touchexternal:  $(CPOPTDEP)
ifneq ($(CPOPT),$(LAST_CPOPT))
		@-touch $(CPOPTSRC)
endif
		@-rm -f $(LASTSETTINGS)
		@echo "LAST_CPOPT=$(CPOPT)" >> $(LASTSETTINGS)

-include	$(MAINDEP)

# $(MAINFILE):	$(BINDIR) $(OBJDIR) $(SCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) $(MAINOBJFILES) $(MAINCXXOBJFILES)
$(MAINFILE):	$(BINDIR) $(OBJDIR) $(UGOBJFILES) $(MAINOBJFILES) $(MAINCXXOBJFILES)
		@echo "-> linking $@"
		$(LINKCXX) $(UGOBJFILES) $(MAINOBJFILES) $(MAINCXXOBJFILES) $(LINKCXXSCIPALL) \
		$(LINKCXX_L)$(UGDIR)/lib $(LINKCXX_l)$ $(UGLIB) \
		$(LINKCXXSCIPALL) \
		$(LINKCXX_l)$(LPILIB)$(LINKLIBSUFFIX) \
                $(OFLAGS) $(LPSLDFLAGS) \
		$(LDFLAGS) $(LINKCXX_o)$@

$(OBJDIR)/%.o:	$(STPDIR)/src/%.c
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CFLAGS) -c $< $(CC_o)$@

$(OBJDIR)/%.o:	$(SRCDIR)/%.cpp
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) -c $< $(CXX_o)$@

#---- EOF --------------------------------------------------------------------
