#  $Source: /usr1/mfg/3.4C/solaris_bld/group/util/makefiles/RCS/dir.mk,v $
#
#  $Author: wanda $
#  $Revision: #3 $
#  $Date: 2004/09/29 $
#  $State: Exp $
#

.PHONY: all
all: install release

BUILD_ORDER	= \
			def \
			cdef \
			cdefzlib \
			defzlib \
			defrw \
			defwrite \
			defdiff

## HP-UX 9.0.X
OS_TYPE := $(shell uname)
ifeq ($(OS_TYPE),HP-UX)
OPTIMIZE_FLAG = +O2
else
OS_VER := $(shell uname -r)
ifeq ($(findstring 4.1,$(OS_VER)),4.1)
OPTIMIZE_FLAG = -O
else
OPTIMIZE_FLAG = -O
endif
endif

install:
	@$(MAKE) $(MFLAGS) installhdrs installlib installbin

release:
	@$(MAKE) "DEBUG=$(OPTIMIZE_FLAG)" install

test:
	@$(MAKE) "BUILD_ORDER=TEST" dotest

.PHONY: clean
clean:
	@$(MAKE) "BUILD_ORDER += TEST" doclean;
	echo $(BUILD_ORDER);
	@$(MAKE) doclean;

.DEFAULT:
	@for i in $(BUILD_ORDER) ;do \
		echo $(MAKE) $@ in $$i ; \
		cd $$i ; \
		$(MAKE) $(MFLAGS) $@ || exit ; \
		cd .. ; \
	done

.DELETE_ON_ERROR:
