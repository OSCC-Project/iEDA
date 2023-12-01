#include <iostream>
#include <string>
#include <string_view>

typedef struct RustVec {
  void* data;
  uintptr_t len;
  uintptr_t cap;
} RustVec;

template <typename T>
class RustVecIterator {
 public:
  explicit RustVecIterator(RustVec* rust_vec) : _rust_vec(rust_vec) {}
  ~RustVecIterator() = default;

  bool hasNext() { return _index < _rust_vec->len; }
  T* next() {
    auto* ret_value = static_cast<T*>(_rust_vec->data) + _index++;
    return ret_value;
  }

 private:
  RustVec* _rust_vec;
  uintptr_t _index = 0;
};

/**
 * @brief usage:
 * RustVec* vec;
 * T* elem;
 * FOREACH_VEC_ELEM(vec, T, elem)
 * {
 *    do_something_for_elem();
 * }
 *
 */
#define FOREACH_VEC_ELEM(vec, T, elem) \
  for (RustVecIterator<T> iter(vec);   \
       iter.hasNext() ? elem = iter.next(), true : false;)

extern "C" {

typedef struct RustLibertyGroupStmt {
  char* file_name;
  uint32_t line_no;
  char* group_name;
  struct RustVec attri_values;
  struct RustVec stmts;
} RustLibertyGroupStmt;

typedef struct RustLibertySimpleAttrStmt {
  char* file_name;
  uint32_t line_no;
  char* attri_name;
  const void* attri_value;
} RustLibertySimpleAttrStmt;

typedef struct RustLibertyComplexAttrStmt {
  char* file_name;
  uint32_t line_no;
  char* attri_name;
  struct RustVec attri_values;
} RustLibertyComplexAttrStmt;

typedef struct RustLibertyStringValue {
  char* value;
} RustLibertyStringValue;

typedef struct RustLibertyFloatValue {
  double value;
} RustLibertyFloatValue;

void* rust_parse_lib(const char* s);

uintptr_t rust_vec_len(const struct RustVec* vec);

void free_c_char(char* s);

struct RustLibertyGroupStmt* rust_convert_group_stmt(void* group_stmt);

struct RustLibertySimpleAttrStmt* rust_convert_simple_attribute_stmt(
    void* simple_attri_stmt);

struct RustLibertyComplexAttrStmt* rust_convert_complex_attribute_stmt(
    void* complex_attri_stmt);

bool rust_is_float_value(void* c_attribute_value);

bool rust_is_string_value(void* c_attribute_value);

struct RustLibertyStringValue* rust_convert_string_value(void* string_value);

struct RustLibertyFloatValue* rust_convert_float_value(void* float_value);

typedef struct RustStingView {
  const uint8_t* data;
  uintptr_t len;
} RustStingView;

struct RustStingView test_string_to_view(void);
}

int main() {
  // std::string s =
  //     "/home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/"
  //     "liberty-parser/example/example1_slow.lib";
  // std::cout << s << "\n";

  // auto* lib_file = rust_parse_lib(s.c_str());
  // auto* lib_group = rust_convert_group_stmt(lib_file);

  // auto attribute_values = lib_group->attri_values;

  // void* attribute_value;
  // FOREACH_VEC_ELEM(&attribute_values, void, attribute_value) {
  //   if (rust_is_string_value(attribute_value)) {
  //     auto* string_val = rust_convert_string_value(attribute_value);
  //     std::cout << string_val->value << "\n";
  //   }
  // }

  // std::cout << "lib file :" << lib_file << "\n";

  auto str_view = test_string_to_view();
  std::string_view sv((const char*)str_view.data, str_view.len);
  std::cout << sv << "\n";

  return 0;
}