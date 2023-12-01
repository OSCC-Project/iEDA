.KEEP_STATE:

.SUFFIXES:        .o .cxx

CC = g++  -Wall 
#CC = CC 


#CFLAGS= -O5 
CFLAGS= -g 

LIB_OBJECT= facility.o bst.o  bst_sub1.o  bst_sub2.o bst_sub3.o IME_code.o sink_move.o

OBJECT= main.o 

CLEANOBJECT= $(OBJECT)  $(LIB_OBJECT)

test : ${OBJECT} libbst.a 
	${CC} ${CFLAGS} -o bst \
		${OBJECT}  -L. -lbst -lm

.cxx.o :  bst_header.h
	 ${CC} ${CFLAGS} -c $<

clean :
	\rm -f ${CLEANOBJECT} libbst.a core

print :
	enscript -Ppub1 -h  $(SRC) 

all : lib

lib : libbst.a

libbst.a: $(LIB_OBJECT)
	ar cr libbst.a $(LIB_OBJECT) 



