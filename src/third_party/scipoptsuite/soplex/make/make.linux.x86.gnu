CPPFLAGS	+=	-m32
LDFLAGS		+=	-m32

ifneq ($(OPT),opt-gccold)
ifneq ($(OPT),dbg)
OFLAGS          +=      -mtune=native  # -malign-double -mcpu=pentium4
endif
endif

ifeq ($(SHARED),true)
LIBBUILDFLAGS	+=     	-m32
endif
