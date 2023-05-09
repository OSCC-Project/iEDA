# This make file is for lefwrite.cpp only.  lefwrite.cpp is an example 
# which calls directly to the defwWrite calls.  It does not use the callback
# mechanism.  The user is responsible for the correct sequence of the output.
#
# There is another example (lefwriteCbs.cpp) in this directory.  This example
# uses the callback mechanism.  The user does not have to worry about the
# correct sequence of the lef output file.
# To compile lefwriteCbs.cpp, you need to create another Makefile which
# is identical to this one, but change the line:
# BINSRCS = lefwriteCbs.o

FAKE_ALL: all

LIBTARGET =	
BINTARGET = lefwrite

LDLIBS =	../lib/liblef.a

BINSRCS =	lefwrite.cpp

include ../template.mk
