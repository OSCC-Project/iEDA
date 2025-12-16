# Changelog for salt

## Reason

To streamline project construction, we retained only the primary source files (./src/salt) of [SALT](https://github.com/chengengjie/salt) while removing redundant assessment files. To ensure smooth compilation, certain adjustments were made to the namespace, primarily due to the third-party's utilization of Flute. Additionally, to facilitate the construction of data for the iCTS net, we incorporated additional conversion interfaces.


## Diff

### Redundant assessment files

```
Only in {ORIGIN REPOSITORY}: .clang-format
Only in {ORIGIN REPOSITORY}: .git
Only in {ORIGIN REPOSITORY}: .gitignore
Only in {ORIGIN REPOSITORY}: scripts
Only in {ORIGIN REPOSITORY}: src
Only in {ORIGIN REPOSITORY}: toys
```

### For ensure compilation
namespace "flute" -> "Flute"
```
Only in {ORIGIN REPOSITORY}/src/salt/base: flute
diff -r ./base/flute.cpp {ORIGIN REPOSITORY}/src/salt/base/flute.cpp
6,7c6,7
< #include "flute3/flute.h" // should be included after boost/functional/hash.hpp
< #define MAXD 100000          // max. degree that can be handled
---
> #include "flute/flute.h" // should be included after boost/functional/hash.hpp
> 
12,13c12,13
<       Flute::readLUT();
<       once = true;
---
>         flute::readLUT();
>         once = true;
17c17
<     Flute::Tree fluteTree;
---
>     flute::Tree fluteTree;
27c27
<     fluteTree = Flute::flute(d, x, y, FLUTE_ACCURACY);
---
>     fluteTree = flute::flute(d, x, y, ACCURACY);
diff -r ./base/net.h {ORIGIN REPOSITORY}/src/salt/base/net.h
2a3,4
> #include "salt/utils/utils.h"
> 
4d5
< #include <memory>
6,7c7
< 
< #include "salt/utils/utils.h"
---
> #include <memory>
11,12c11
< // #define DTYPE int  // same as flute.h, will overwrite it
< typedef int DTYPE;
---
> #define DTYPE int  // same as flute.h, will overwrite it
Only in ./: CHANGELOG.md
diff -r ./CMakeLists.txt {ORIGIN REPOSITORY}/src/salt/CMakeLists.txt
1,8c1,3
< file(GLOB SALT_SRCS *.cpp */*.cpp)
< set(CMAKE_CXX_STANDARD 14)
< set(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
< set(POWV9_DAT ${FLUTE_HOME}/etc/POWV9.dat)
< set(POST9_DAT ${FLUTE_HOME}/etc/POST9.dat)
< add_library(salt STATIC ${SALT_SRCS})
< target_link_libraries(salt PRIVATE flute)
< target_include_directories(salt PRIVATE ${FLUTE_HOME})
---
> file(GLOB SALT_SRCS *.cpp */*.cpp base/flute/*.c)
> 
> add_library(salt STATIC ${SALT_SRCS})
\ No newline at end of file
```
### iCTS Interfaces

```
diff -r ./base/net.h {ORIGIN REPOSITORY}/src/salt/base/net.h
33,72c31,57
< class Net
< {
<  public:
<   int id;
<   string name;
<   bool withCap = false;
< 
<   vector<shared_ptr<Pin>> pins;  // source is always the first one
< 
<   void init(const int& id, const string& name, const vector<shared_ptr<Pin>>& pins)
<   {
<     this->id = id;
<     this->name = name;
<     this->pins = pins;
<     this->withCap = true;
<   }
< 
<   void RanInit(int i, int numPin, DTYPE width = 100, DTYPE height = 100);  // random initialization
< 
<   shared_ptr<Pin> source() const { return pins[0]; }
< 
<   // File read/write
<   // ------
<   // Format:
<   // Net <net_id> <net_name> <pin_num> [-cap]
<   // 0 x0 y0 [cap0]
<   // 1 x1 y1 [cap1]
<   // ...
<   // ------
<   bool Read(istream& is);
<   void Read(const string& fileName);
<   string GetHeader() const;
<   void Write(ostream& os) const;
<   void Write(const string& prefix, bool withNetInfo = true) const;
< 
<   friend ostream& operator<<(ostream& os, const Net& net)
<   {
<     net.Write(os);
<     return os;
<   }
---
> class Net {
> public:
>     int id;
>     string name;
>     bool withCap = false;
> 
>     vector<shared_ptr<Pin>> pins;  // source is always the first one
> 
>     void RanInit(int i, int numPin, DTYPE width = 100, DTYPE height = 100);  // random initialization
> 
>     shared_ptr<Pin> source() const { return pins[0]; }
>     
>     // File read/write
>     // ------
>     // Format:
>     // Net <net_id> <net_name> <pin_num> [-cap]
>     // 0 x0 y0 [cap0]
>     // 1 x1 y1 [cap1]
>     // ...
>     // ------
>     bool Read(istream& is);
>     void Read(const string& fileName);
>     string GetHeader() const;
>     void Write(ostream& os) const;
>     void Write(const string& prefix, bool withNetInfo = true) const;
> 
>     friend ostream& operator<<(ostream& os, const Net& net) { net.Write(os); return os; }
```

### Tree::SetParentFromUndirectedAdjList function

Fixed the non-tree topology error in SALT, caused by undirected cyclic graph of third_party:flute3

### Refine::removeRedundantCoincident function

Fixed a bug caused by redundancy after flip and ushit

### Rename variable

Changing some variable names and function names for the uniform naming convention does not change the code logic

### Comparator const-correctness in `rsa.h`

Added the `const` qualifier to the comparator `CompInnerNode::operator()` to ensure const-correctness, and added the missing newline at the end of the `rsa.h` file.

```
diff --git a/src/third_party/salt/base/rsa.h b/src/third_party/salt/base/rsa.h
index 172829fa7..41dd6222b 100644
--- a/src/third_party/salt/base/rsa.h
+++ b/src/third_party/salt/base/rsa.h
@@ -37,7 +37,7 @@ class InnerNode
 class CompInnerNode
 {
	public:
-  bool operator()(const InnerNode* a, const InnerNode* b)
+  bool operator()(const InnerNode* a, const InnerNode* b) const
	 {
		 return a->dist > b->dist ||  // prefer fathest one
						(a->dist == b->dist
@@ -88,4 +88,4 @@ class RsaBuilder : public RSABase
	 void PrintOuterNodes();
 };
 
-}  // namespace salt
\ No newline at end of file
+}  // namespace salt
```

Reason: Fixes a missing `const` on the comparator that could trigger compiler warnings or prevent use on const objects; also adds the POSIX-required newline at EOF.
