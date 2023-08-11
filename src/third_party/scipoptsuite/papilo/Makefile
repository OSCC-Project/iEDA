#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2022 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# description -> see check/README_PERFORMANCE.md

#-----------------------------------------------------------------------------
# paths variables
#-----------------------------------------------------------------------------

PAPILODIR=$(shell pwd -P)

.DEFAULT_GOAL := help

#-----------------------------------------------------------------------------
# include make.project file
#-----------------------------------------------------------------------------

# load default settings and detect host architecture
include $(PAPILODIR)/make/make.project

# include local targets
-include make/local/make.targets

#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

.PHONY: help
help:
		@echo "  Main targets:"
		@echo "  - test: start Papilo testrun locally."
		@echo "  - testcluster: start Papilo testrun on the cluster."

.PHONY: test
test:
		cd check; \
		$(SHELL) ./check.sh $(TEST) $(EXECUTABLE) $(SETTINGS) $(BINID) $(OUTPUTDIR) $(TIME) $(NODES) $(MEM) $(FEASTOL) $(DISPFREQ) \
		$(CONTINUE) $(LOCK) $(VERSION) $(LPS) $(DEBUGTOOL) $(CLIENTTMPDIR) $(REOPT) $(PAPILO_OPT_COMMAND) $(SETCUTOFF) $(MAXJOBS) $(VISUALIZE) $(PERMUTE) \
                $(SEEDS) $(GLBSEEDSHIFT) $(STARTPERM);

# --- EOF ---------------------------------------------------------------------
# DO NOT DELETE
