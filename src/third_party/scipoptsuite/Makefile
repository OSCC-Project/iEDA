#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*                  This file is part of the program and library             *#
#*         SCIP --- Solving Constraint Integer Programs                      *#
#*                                                                           *#
#*  Copyright 2002-2022 Zuse Institute Berlin                                *#
#*                                                                           *#
#*  Licensed under the Apache License, Version 2.0 (the "License");          *#
#*  you may not use this file except in compliance with the License.         *#
#*  You may obtain a copy of the License at                                  *#
#*                                                                           *#
#*      http://www.apache.org/licenses/LICENSE-2.0                           *#
#*                                                                           *#
#*  Unless required by applicable law or agreed to in writing, software      *#
#*  distributed under the License is distributed on an "AS IS" BASIS,        *#
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *#
#*  See the License for the specific language governing permissions and      *#
#*  limitations under the License.                                           *#
#*                                                                           *#
#*  You should have received a copy of the Apache-2.0 license                *#
#*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         *#
#*                                                                           *#
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#

#@file    Makefile
#@brief   SCIP Optimization Suite Makefile
#@author  Tobias Achterberg
#@author  Ambros Gleixner
#@author  Gerald Gamrath
#@author  Matthias Miltenberger

# version of the SCIP Optimization Suite
OPTSUITEVERSION=$(awk '/^SCIP_VERSION/ { print $3 }' scip/make/make.project)

TAR		=	tar

BASEDIR		:=	$(CURDIR)

INSTALLDIR	=	/usr/local
LIBDIR		=	lib
LIBOBJDIR	=       obj
LIBOBJZIMPLDIR	=	obj/zimpl
SCIPDIR		=	$(BASEDIR)/scip
SOPLEXDIR	=	$(BASEDIR)/soplex
ZIMPLDIR	=	$(BASEDIR)/zimpl
GCGDIR		=	$(BASEDIR)/gcg
UGDIR		=	$(BASEDIR)/ug
PAPILODIR	=	$(BASEDIR)/papilo

SHARED		=	false
OPT		=	opt
LPSOPT		=	opt
ZIMPL		=	true
ZIMPLOPT	=	opt
IPOPT		=	false
SYM		=	none
PAPILO		=	false
ZLIB		=	true

GTEST		=	false

SPX_LEGACY	=	false
LEGACY		= 	false
ifeq ($(LEGACY),true)
SPX_LEGACY 	= 	true
endif

.PHONY:		all
all:		scipbinary

.PHONY:		clean
clean:
		@$(MAKE) -C $(SCIPDIR) clean
		@$(MAKE) -C $(SOPLEXDIR) clean OPT=$(LPSOPT)
ifneq ($(ZIMPL),false)
		@$(MAKE) -C $(ZIMPLDIR) clean OPT=$(ZIMPLOPT)
endif

.PHONY:		cleanlib
cleanlib:
		@$(MAKE) -C $(SCIPDIR) cleanlib
		@$(MAKE) -C $(SOPLEXDIR) cleanlib OPT=$(LPSOPT)
ifneq ($(ZIMPL),false)
		@$(MAKE) -C $(ZIMPLDIR) clean OPT=$(ZIMPLOPT)
endif

.PHONY:		check
check:		test

.PHONY:		test
test:
		@$(MAKE) -C $(SCIPDIR) test $^

.PHONY:		scipbinary
scipbinary:	$(SOPLEXDIR) $(ZIMPLDIR) $(SCIPDIR)
		@$(MAKE) -f Makefile.doit SCIPDIR=$(SCIPDIR) SOPLEXDIR=$(SOPLEXDIR) ZIMPL=$(ZIMPL) PAPILO=$(PAPILO) ZIMPLDIR=$(ZIMPLDIR) PAPILODIR=$(PAPILODIR) SPX_LEGACY=$(SPX_LEGACY)

.PHONY:		scipoptlib
scipoptlib:	$(LIBDIR) $(LIBOBJDIR) $(LIBOBJZIMPLDIR) $(LIBOBJSUBDIRS) $(SOPLEXDIR) $(ZIMPLDIR) $(SCIPDIR) $(PAPILODIR)
ifeq ($(SHARED),true)
		@echo "** compile shared libraries with compiler flag -fPIC"
else
		@echo "** compile libraries statically with compiler flag -fPIC"
endif
ifneq ($(ZIMPL),false)
		@$(MAKE) -C $(ZIMPLDIR) clean SHARED=$(SHARED) OPT=$(ZIMPLOPT)
endif
		@$(MAKE) scipbinary SHARED=$(SHARED) IPOPT=$(IPOPT) SPX_LEGACY=$(SPX_LEGACY)
		@$(MAKE) -f Makefile.doit $@ VERSION=$(OPTSUITEVERSION) SCIPDIR=$(SCIPDIR) SOPLEXDIR=$(SOPLEXDIR) ZIMPL=$(ZIMPL) PAPILO=$(PAPILO) ZIMPLDIR=$(ZIMPLDIR) PAPILODIR=$(PAPILODIR)

.PHONY:		install
install:	scipoptlib
		@$(MAKE) -C $(SCIPDIR) INSTALLDIR=$(abspath $(INSTALLDIR)) install
		@$(MAKE) -C $(SOPLEXDIR) OPT=$(LPSOPT) ZLIB=$(ZLIB) INSTALLDIR=$(abspath $(INSTALLDIR)) install
		@$(MAKE) -f Makefile.doit installscipopt VERSION=$(OPTSUITEVERSION) SCIPDIR=$(SCIPDIR) INSTALLDIR=$(abspath $(INSTALLDIR)) LIBDIR=$(LIBDIR)

.PHONY:		soplex
soplex:		$(SOPLEXDIR) $(SCIPDIR)
		@$(MAKE) -f Makefile.doit soplex SCIPDIR=$(SCIPDIR) SOPLEXDIR=$(SOPLEXDIR)

.PHONY:		testsoplex
testsoplex:		$(SOPLEXDIR)
		@$(MAKE) -C $(SOPLEXDIR) check $^

.PHONY:		gcg
gcg:		$(SOPLEXDIR) $(ZIMPLDIR) $(SCIPDIR) $(GCGDIR)
		@$(MAKE) -f Makefile.doit gcg SCIPDIR=$(SCIPDIR) SOPLEXDIR=$(SOPLEXDIR) ZIMPL=$(ZIMPL) PAPILO=$(PAPILO) ZIMPLDIR=$(ZIMPLDIR) PAPILODIR=$(PAPILODIR) GCGDIR=$(GCGDIR) SPX_LEGACY=$(SPX_LEGACY) GTEST=${GTEST}

.PHONY:		testgcg
testgcg:
		@$(MAKE) -C $(GCGDIR) test $^

.PHONY:		ug
ug:		$(SOPLEXDIR) $(ZIMPLDIR) $(SCIPDIR) $(UGDIR)
		@$(MAKE) -f Makefile.doit ug SCIPDIR=$(SCIPDIR) SOPLEXDIR=$(SOPLEXDIR) ZIMPL=$(ZIMPL) PAPILO=$(PAPILO) ZIMPLDIR=$(ZIMPLDIR) PAPILODIR=$(PAPILODIR) UGDIR=$(UGDIR) PARASCIP=true SPX_LEGACY=$(SPX_LEGACY)

.PHONY:		testug
testug:
		@$(MAKE) -C $(UGDIR) test $^

$(LIBDIR):
		@-mkdir -p $(LIBDIR)

$(LIBOBJDIR):
		@-mkdir -p $(LIBOBJDIR)

$(LIBOBJZIMPLDIR):
		@-mkdir -p $(LIBOBJZIMPLDIR)
