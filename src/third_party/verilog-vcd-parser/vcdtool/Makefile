
SRC_DIR         ?= ../src
BUILD_DIR       ?= ../build

DOCS_CFG        ?= ../.doxygen

LEX_SRC         ?= $(SRC_DIR)/VCDScanner.l
LEX_OUT         ?= $(BUILD_DIR)/VCDScanner.cpp
LEX_HEADER      ?= $(BUILD_DIR)/VCDScanner.hpp
LEX_OBJ         ?= $(BUILD_DIR)/VCDScanner.o

YAC_SRC         ?= $(SRC_DIR)/VCDParser.ypp
YAC_OUT         ?= $(BUILD_DIR)/VCDParser.cpp
YAC_HEADER      ?= $(BUILD_DIR)/VCDParser.hpp
YAC_OBJ         ?= $(BUILD_DIR)/VCDParser.o

CXXFLAGS        += -I$(BUILD_DIR) -I$(SRC_DIR) -g -std=c++0x

VCD_SRC         ?= $(SRC_DIR)/VCDFile.cpp \
                   $(SRC_DIR)/VCDValue.cpp \
                   $(SRC_DIR)/VCDFileParser.cpp

VCD_PARSER        ?= $(BUILD_DIR)/vcd-parse

VCDTOOL        ?= $(BUILD_DIR)/../vcdtool/vcdtool
VCDTOOL_SRC    ?= $(BUILD_DIR)/../vcdtool/vcdtool.cpp \
                  $(BUILD_DIR)/../vcdtool/gitversion.h

all : $(VCDTOOL) docs

$(BUILD_DIR)/../vcdtool/gitversion.h: ../.git/HEAD ../.git/index
	echo "const char *gitversion = \"$(shell git rev-parse HEAD)\";" > $@

parser-srcs: $(YAC_OUT) $(LEX_OUT)

docs: $(DOCS_CFG) $(VCD_SRC) $(TEST_FILE)
	doxygen $(DOCS_CFG)

$(YAC_OUT) : $(YAC_SRC)
	bison -v --defines=$(YAC_HEADER) $(YAC_SRC) -o $(YAC_OUT)

$(LEX_OUT) : $(LEX_SRC) $(YAC_OUT)
	flex  -P VCDParser --header-file=$(LEX_HEADER) -o $(LEX_OUT) $(LEX_SRC)

$(VCDTOOL) : $(VCDTOOL_SRC) $(VCD_SRC) $(LEX_OBJ) $(YAC_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(LEX_OUT) $(LEX_HEADER) $(LEX_OBJ) \
           $(YAC_OUT) $(YAC_HEADER) $(YAC_OBJ) \
           position.hh stack.hh location.hh VCDParser.output $(VCD_PARSER)
