CPPFLAGS	+=	-m32
LDFLAGS		+=	-m32

ifeq ($(SHARED),true)
LIBBUILDFLAGS	+=     	-m32
endif
