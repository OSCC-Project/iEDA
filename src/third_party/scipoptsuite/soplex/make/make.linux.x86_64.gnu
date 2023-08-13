CPPFLAGS	+=	-m64
LDFLAGS		+=	-m64

ifneq ($(OPT),opt-gccold)
ifneq ($(OPT),dbg)
OFLAGS          +=      -mtune=native  # -malign-double -mcpu=pentium4
endif
endif

ifeq ($(SHARED),true)
LIBBUILDFLAGS	+=     	-m64
endif
