# Remove default rules
.SUFFIXES:

# detect operating system
OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN := $(strip $(findstring darwin, $(OSLOWER)))

# C compiler
ifneq ($(DARWIN),)
  # C compiler for osx
  CC := clang
  CPPFLAGS :=
  CFLAGS := -arch x86_64 -fPIC
else
  # C compiler for linux
  CC := gcc
  CPPFLAGS :=
  CFLAGS := -m64 -fPIC
endif

# Fortran optimization level
FOPT := -O3

# Fortran compilers
ifneq ($(DARWIN),)
  # Fortran 90 compiler
  F90C := gfortran
  F90FLAGS := -m64 -fPIC -Jsrc $(FOPT)
  # Fortran 77 compiler
  F77C := gfortran
  F77FLAGS := -m64 -fPIC -fdefault-integer-8 $(FOPT)
else
  # Fortran 90 compiler
  F90C := gfortran
  F90FLAGS :=  -m64 -fPIC -Jsrc $(FOPT)
  # Fortran 77 compiler
  F77C := gfortran
  F77FLAGS := -m64 -fPIC -fdefault-integer-8 $(FOPT)
endif

# Matlab
ML := matlab
MLFLAGS := -nojvm -nodisplay
ifneq ($(DARWIN),)
  # settings for mac os x
  MLARCH := maci64
else
  # settins for linux
  MLARCH := glnxa64
endif

# Linker
ifneq ($(DARWIN),)
  # settings for mac os x
  LD := clang
  LIB_SUFFIX := dylib
  EXPORT_SYMBOLS := src/symbols.osx
  LDFLAGS := -dynamiclib
  LDFLAGS += -arch x86_64
  LDFLAGS += -Wl,-twolevel_namespace
  LDFLAGS += -Wl,-no_compact_unwind
  LDFLAGS += -undefined error
  LDFLAGS += -bind_at_load
  LDFLAGS += -Wl,-exported_symbols_list,$(EXPORT_SYMBOLS)
  LDLIBS :=
  # static libraries
  LDLIBS += /usr/local/opt/gcc/lib/gcc/5/libgfortran.a
  LDLIBS += /usr/local/opt/gcc/lib/gcc/5/libquadmath.a
  LDLIBS += /usr/local/Cellar/gcc/5.3.0/lib/gcc/5/gcc/x86_64-apple-darwin15.0.0/5.3.0/libgcc.a
  # get blas from Matlab
  LDLIBS += -L/Applications/MATLAB_R2015b.app/bin/maci64 -lmwblas
else
  # settins for linux
  LD := gcc
  LIB_SUFFIX := so
  EXPORT_SYMBOLS := src/symbols.map
  LDFLAGS := -m64 -shared
  LDFLAGS += -Wl,--version-script,$(EXPORT_SYMBOLS)
  # libraries
  LDLIBS :=
  LDLIBS += -Wl,-rpath,/usr/lib -lgfortran
endif

# list of files required by matlab
MATLAB_FILES := \
  matlab/libclusol.$(LIB_SUFFIX) \
  matlab/clusol.h \
  matlab/libclusol_proto_$(MLARCH).m \
  matlab/libclusol_thunk_$(MLARCH).$(LIB_SUFFIX)

# list of interface specification files
INTERFACE_FILES := \
  gen/interface.py \
  gen/interface_files.org \
  gen/lu1fac.org \
  gen/lu6mul.org \
  gen/lu6sol.org \
  gen/lu8adc.org \
  gen/lu8adr.org \
  gen/lu8dlc.org \
  gen/lu8dlr.org \
  gen/lu8mod.org \
  gen/lu8rpc.org \
  gen/lu8rpr.org

# list of F77 code files
F77_FILES := \
  src/lusol_util.f \
  src/lusol6b.f \
  src/lusol7b.f \
  src/lusol8b.f

F77_OBJ := $(patsubst %.f,%.o,$(filter %.f,$(F77_FILES)))

# list of F90 code files
F90_FILES := \
  src/lusol_precision.f90 \
  src/lusol.f90

F90_OBJ := $(patsubst %.f90,%.o,$(filter %.f90,$(F90_FILES)))
F90_MOD := $(patsubst %.f90,%.mod,$(filter %.f90,$(F90_FILES)))

# list of object files
OBJ := src/clusol.o
OBJ += $(F90_OBJ)
OBJ += $(F77_OBJ)

# set the default goal
.DEFAULT_GOAL := all

# default target to build everything
.PHONY: all
all: src/libclusol.$(LIB_SUFFIX) src/clusol.h

# pattern to compile fortran 77 files
$(F77_OBJ) : %.o : %.f
	$(F77C) $(F77FLAGS) -c $< -o $@

# pattern to compile fortran 90 files
$(F90_OBJ) : %.o : %.f90
	$(F90C) $(F90FLAGS) -c $< -o $@

# dependencies for F90 module files
$(F90_MOD) : %.mod : %.o

# extra fortran dependencies
lusol.o : lusol_precision.mod

# C code generation
src/clusol.h: $(INTERFACE_FILES)
	./gen/interface.py -i gen/interface_files.org -o $@ -t header

src/clusol.c: $(INTERFACE_FILES)
	./gen/interface.py -i gen/interface_files.org -o $@ -t source

# C compilation
src/clusol.o: src/clusol.c src/clusol.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Link the dynamic library
src/libclusol.$(LIB_SUFFIX): $(OBJ) $(EXPORT_SYMBOLS)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ $(LDLIBS)

# file copying to matlab directory
$(MATLAB_FILES): src/libclusol.$(LIB_SUFFIX) src/clusol.h
	cp src/libclusol.$(LIB_SUFFIX) src/clusol.h ./matlab/
	$(ML) $(MLFLAGS) -r "cd matlab; lusol_build; exit"

.PHONY: matlab
matlab: $(MATLAB_FILES)

.PHONY: matlab_test
matlab_test: $(MATLAB_FILES)
	$(ML) $(MLFLAGS) -r "cd matlab; lusol_test; exit"

.PHONY: clean
clean:
	$(RM) src/*.o
	$(RM) src/*.$(LIB_SUFFIX)
	$(RM) src/*.mod
	$(RM) $(MATLAB_FILES)

.PHONY: clean_gen
clean_gen:
	$(RM) src/clusol.h
	$(RM) src/clusol.c

# print helper
#print-%:
#	@echo $* := $($*)
