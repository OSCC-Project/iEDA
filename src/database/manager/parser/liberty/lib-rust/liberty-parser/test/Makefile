# 指定编译器
CXX = g++

# 源文件名
SRC = test.cpp

# 目标文件(可执行文件)
TARGET = test

# 外部库文件目录
LIBDIR = /home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/target/debug

# 需要链接的库
LIBS = -lliberty_parser -lpthread

# 包含头文件目录
INCLUDES = -I/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/dist 

# 编译参数
CXXFLAGS = -g -O0 -Wall $(INCLUDES)
CXXFLAGS += -std=c++20

# 链接参数
LDFLAGS = -L$(LIBDIR) $(LIBS)

# 编译命令
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) -O0 -g

# 清理命令
clean: 
	rm -f $(TARGET)