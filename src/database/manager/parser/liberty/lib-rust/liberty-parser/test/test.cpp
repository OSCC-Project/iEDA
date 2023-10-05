extern "C" void rust_parse_lib(const char* s);

int main()
{
  const char* lib_file = "/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/example/example1_slow.lib";
  rust_parse_lib(lib_file);
  return 0;
}