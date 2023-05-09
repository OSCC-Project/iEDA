SHELL=/bin/sh

OS_TYPE	:= $(shell uname -s)
ifeq ($(OS_TYPE),AIX)
    ARCH=ibmrs
    CXX=/usr/bin/xlC_r -+ -Dibmrs
    CC=/usr/bin/cc_r -Dibmrs
endif
ifeq ($(OS_TYPE),HP-UX)
    ARCH=hppa
    CXX=/opt/aCC/bin/aCC
    CXXFLAGS=+DAportable -AA
    BIN_LINK_FLAGS=-lPW
endif

ifeq ($(OS_TYPE),SunOS)
    ARCH=sun4v
    CXX=CC
    CXXFLAGS=-g
endif

ifeq ($(OS_TYPE),Linux)
    ARCH=lnx86 
    CXX=g++
    CC=gcc
endif

.SUFFIXES: $(SUFFIXES) .cpp


TMP1	=	$(LIBSRCS:.cpp=.o)
TMP2	=	$(TMP1:.cc=.o)
LIBOBJS =	$(TMP2:.c=.o)

TMP3	=	$(BINSRCS:.cpp=.o)
TMP4	=	$(TMP3:.cc=.o)
BINOBJS =	$(TMP4:.c=.o)

all: $(LIBTARGET) $(BINTARGET)

ifdef BINTARGET
INSTALLED_BIN = ../bin/$(BINTARGET)
endif


$(INSTALLED_BIN): $(BINTARGET)
	@mkdir -p ../bin
	@echo $< =\> $@
	@rm -f $(INSTALLED_BIN)
	@cp $(BINTARGET) $(INSTALLED_BIN)

installbin: $(INSTALLED_BIN)

ifdef LIBTARGET
INSTALLED_LIB = ../lib/$(LIBTARGET)
endif


$(INSTALLED_LIB): $(LIBTARGET)
	@mkdir -p ../lib
	@echo $< =\> $@
	@rm -f $(INSTALLED_LIB)
	@cp $(LIBTARGET) $(INSTALLED_LIB)

installlib: $(INSTALLED_LIB)

install release: all installhdrs installlib installbin

INSTALLED_HDRS = $(PUBLIC_HDRS:%=../include/%)
$(INSTALLED_HDRS):	../include/%:	%
	@mkdir -p ../include
	@echo $< =\> $@
	@rm -f $@
	@cp -p $< $@

installhdrs: $(INSTALLED_HDRS)

.cpp.o:
	$(COMPILE.cc) -I../include $<

.c.o:
	$(COMPILE.c) -I../include $<

$(LIBTARGET): $(LIBOBJS)
	rm -f $(LIBTARGET)
	$(AR) $(ARFLAGS) $(LIBTARGET) $^

$(BINTARGET): $(BINOBJS) $(LIBTARGET) $(LDLIBS)
	rm -f $(BINTARGET)
	$(LINK.cc) -o $(BINTARGET) $(BINOBJS) $(LIBTARGET) $(LDLIBS) $(BIN_LINK_FLAGS)

clean doclean:
	rm -f $(LIBTARGET) $(LIBOBJS) $(BINTARGET) $(BINOBJS) $(INSTALLED_LIB) \
			$(INSTALLED_BIN) $(INSTALLED_HDRS) lef.tab.c* lef.tab.h
