#pragma once

#include <cstdint>
#include <type_traits>

extern "C" {

/**
 * @brief Rust C vector.
 *
 */
typedef struct RustVec
{
  void* data;           //!< vec elem data storage
  uintptr_t len;        //!< vec elem num
  uintptr_t cap;        //!< vec elem capacitance
  uintptr_t type_size;  //!< vec elem type size
} RustVec;
}

/**
 * @brief Rust C vector iterator.
 *
 * @tparam T vector element type.
 */
template <typename T>
class RustVecIterator
{
 public:
  explicit RustVecIterator(RustVec* rust_vec) : _rust_vec(rust_vec) {}
  ~RustVecIterator() = default;

  bool hasNext() { return _index < _rust_vec->len; }
  T* next()
  {
    uintptr_t ptr_move = std::is_same_v<T, void> ? _index * _rust_vec->type_size : _index;
    auto* ret_value = static_cast<T*>(_rust_vec->data) + ptr_move;

    ++_index;
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
#define FOREACH_VEC_ELEM(vec, T, elem) for (RustVecIterator<T> iter(vec); iter.hasNext() ? elem = iter.next(), true : false;)

/**
 * @brief Get the Rust Vec Elem object
 *
 * @tparam T
 * @param rust_vec
 * @param index
 * @return T*
 */
template <typename T>
T* GetRustVecElem(RustVec* rust_vec, uintptr_t index)
{
  uintptr_t ptr_move = std::is_same_v<T, void> ? index * rust_vec->type_size : index;
  auto* ret_value = static_cast<T*>(rust_vec->data) + ptr_move;
  return ret_value;
}