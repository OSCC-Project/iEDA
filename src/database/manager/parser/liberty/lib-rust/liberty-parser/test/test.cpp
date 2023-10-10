#include <iostream>
#include <string>

extern "C" {
void* rust_parse_lib(const char* s);
}
int main()
{
  std::string s = "/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/example/example1_slow.lib";
  std::cout << s << "\n";
  auto* lib_file = rust_parse_lib(s.c_str());

  std::cout << "lib file :" << lib_file << "\n";

  return 0;
}