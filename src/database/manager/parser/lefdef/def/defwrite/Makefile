# This make file is for defwrite.cpp only.  defwrite.cpp is an example
# which calls directly to the defwWrite calls.  It does not use the callback
# mechanism.  The user is responsible for the correct sequence of the output.
#
# There is another example (defwriteCbs.cpp) in this directory.  This example
# uses the callback mechanism.  The user does not have to worry about the
# correct sequence of the def output file.
# To compile defwriteCbs.cpp, you need to create another Makefile which
# is identical to this one, but change the line:
# BINSRCS = defwriteCbs.cpp

FAKE_ALL: all

BINTARGET =	defwrite

HEADERS =

PUBLIC_HDRS = 

LDLIBS = ../lib/libdef.a

BINSRCS =	defwrite.cpp

include ../template.mk
