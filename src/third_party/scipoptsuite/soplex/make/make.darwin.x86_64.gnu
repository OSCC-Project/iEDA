CPPFLAGS	+=	-m64
LDFLAGS		+=	-m64

ifeq ($(SHARED),true)
LIBBUILDFLAGS	+=     	-m64
endif
