LEF_TABNAME     = lef
LEF_BISON_SRCS  = lef.y

FAKE_ALL: all

DIRNAME = lef
LEF_BISON_SRCS  = lef.y

LIBTARGET =	liblef.a

HEADERS =	\
		    lef.tab.h \
		    lex.h \
		    crypt.hpp \
		    lex.cpph

PUBLIC_HDRS =       lefiArray.hpp \
		    lefiCrossTalk.hpp \
		    lefiDebug.hpp \
		    lefiDefs.hpp \
		    lefiEncryptInt.hpp \
		    lefiKRDefs.hpp \
		    lefiLayer.hpp \
		    lefiMacro.hpp \
		    lefiMisc.hpp \
		    lefiNonDefault.hpp \
		    lefiProp.hpp \
		    lefiPropType.hpp \
		    lefiUnits.hpp \
		    lefiUser.hpp \
		    lefiUtil.hpp \
		    lefiVia.hpp \
		    lefiViaRule.hpp \
		    lefrCallBacks.hpp \
                    lefrData.hpp \
		    lefrReader.hpp \
                    lefrSettings.hpp \
		    lefwWriter.hpp \
		    lefwWriterCalls.hpp                    

LIBSRCS =  \
		    crypt.cpp \
		    lef.tab.cpp \
		    lef_keywords.cpp \
		    lefiArray.cpp \
		    lefiCrossTalk.cpp \
		    lefiDebug.cpp \
		    lefiEncryptInt.cpp \
		    lefiLayer.cpp \
		    lefiMacro.cpp \
		    lefiMisc.cpp \
		    lefiNonDefault.cpp \
		    lefiProp.cpp \
		    lefiPropType.cpp \
		    lefiTBExt.cpp \
		    lefiUnits.cpp \
		    lefiVia.cpp \
		    lefiViaRule.cpp \
                    lefrCallbacks.cpp \
                    lefrData.cpp \
		    lefrReader.cpp \
                    lefrSettings.cpp \
		    lefwWriter.cpp \
		    lefwWriterCalls.cpp

# For lef, create '.cpp' object
lef.tab.cpp : $(LEF_BISON_SRCS)
	bison -v -p$(LEF_TABNAME)yy -d $(LEF_BISON_SRCS)
	mv $(LEF_TABNAME).tab.c $(LEF_TABNAME).tab.cpp ;

include ../template.mk
