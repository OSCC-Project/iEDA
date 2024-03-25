// THIS FILE HAS BEEN GENERATED AUTOMATICALLY. DO NOT EDIT DIRECTLY.
// Generated: 2019-03-08 09:59:35.864870255
// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
#ifndef CX_HPP
#define CX_HPP 
#define CX_VERSION_MAJOR 1
#define CX_VERSION_MINOR 3
#define CX_VERSION_PATCH 1
#ifndef CX_ALGORITHM_HPP
#define CX_ALGORITHM_HPP 
#ifndef CONFIG_HPP
#define CONFIG_HPP 
#ifndef QUICKCPPLIB_HAS_FEATURE_H
#define QUICKCPPLIB_HAS_FEATURE_H 
#if __cplusplus >= 201103L
#if !defined(__cpp_alias_templates)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr)
#if __cplusplus >= 201402L
#define __cpp_constexpr 201304
#else
#define __cpp_constexpr 190000
#endif
#endif
#if !defined(__cpp_decltype)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for)
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers)
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates)
#define __cpp_variadic_templates 190000
#endif
#endif
#if __cplusplus >= 201402L
#if !defined(__cpp_aggregate_nsdmi)
#define __cpp_aggregate_nsdmi 190000
#endif
#if !defined(__cpp_binary_literals)
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto)
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas)
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures)
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction)
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation)
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates)
#define __cpp_variable_templates 190000
#endif
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#if !defined(__cpp_exceptions) && defined(_CPPUNWIND)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(_CPPRTTI)
#define __cpp_rtti 190000
#endif
#if !defined(__cpp_alias_templates) && _MSC_VER >= 1800
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && _MSC_FULL_VER >= 190023506
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && _MSC_VER >= 1600
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && _MSC_VER >= 1800
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && _MSC_VER >= 1800
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && _MSC_VER >= 1900
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && _MSC_VER >= 1900
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && _MSC_VER >= 1600
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && _MSC_VER >= 1900
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && _MSC_VER >= 1700
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && _MSC_VER >= 1800
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && _MSC_VER >= 1900
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references) && _MSC_VER >= 1600
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && _MSC_VER >= 1600
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_user_defined_literals) && _MSC_VER >= 1900
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && _MSC_VER >= 1800
#define __cpp_variadic_templates 190000
#endif
#if !defined(__cpp_binary_literals) && _MSC_VER >= 1900
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto) && _MSC_VER >= 1900
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas) && _MSC_VER >= 1900
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures) && _MSC_VER >= 1900
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction) && _MSC_VER >= 1900
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation) && _MSC_VER >= 1900
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates) && _MSC_FULL_VER >= 190023506
#define __cpp_variable_templates 190000
#endif
#endif
#if(defined(__GNUC__) && !defined(__clang__))
#define QUICKCPPLIB_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if !defined(__cpp_exceptions) && defined(__EXCEPTIONS)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(__GXX_RTTI)
#define __cpp_rtti 190000
#endif
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_GCC >= 40801)
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_GCC >= 40400)
#define __cpp_variadic_templates 190000
#endif
#endif
#endif
#if defined(__clang__)
#define QUICKCPPLIB_CLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#if !defined(__cpp_exceptions) && (defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && (defined(__GXX_RTTI) || defined(_CPPRTTI))
#define __cpp_rtti 190000
#endif
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && defined(__cpp_raw_string_literals)
#define __cpp_raw_strings __cpp_raw_string_literals
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_rvalue_references) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals) && defined(__cpp_user_literals)
#define __cpp_user_defined_literals __cpp_user_literals
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_variadic_templates 190000
#endif
#endif
#endif
#endif
#define cfg_HAS_CONSTEXPR14 (__cpp_constexpr >= 201304)
#if cfg_HAS_CONSTEXPR14
#define cfg_constexpr14 constexpr
#else
#define cfg_constexpr14 
#endif
#if cfg_HAS_CONSTEXPR14 && defined(__clang__)
#define cfg_HAS_FULL_FEATURED_CONSTEXPR14 1
#else
#define cfg_HAS_FULL_FEATURED_CONSTEXPR14 0
#endif
#endif
namespace cx
{
template <typename Iterator1, typename Iterator2>
constexpr bool equal(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
{
#if cfg_HAS_CONSTEXPR14
while (first1 != last1 && first2 != last2 && *first1 == *first2)
{
++first1, ++first2;
}
return first1 == last1 && first2 == last2;
#else
return first1 != last1 && first2 != last2 && *first1 == *first2
? equal(first1 + 1, last1, first2 + 1, last2)
: first1 == last1 && first2 == last2;
#endif
}
}
#endif
#ifndef CX_ARRAY_HPP
#define CX_ARRAY_HPP 
#include <cstddef>
#include <stdexcept>
namespace cx
{
template <typename T, std::size_t N>
struct array
{
using value_type = T;
constexpr std::size_t size() const
{
return N;
}
constexpr bool empty() const
{
return size() == 0;
}
constexpr const T* begin() const
{
return &data[0];
}
cfg_constexpr14 T* begin()
{
return &data[0];
}
constexpr const T* cbegin() const
{
return begin();
}
constexpr const T* end() const
{
return &data[N];
}
cfg_constexpr14 T* end()
{
return &data[N];
}
constexpr const T* cend() const
{
return end();
}
constexpr const T& operator[](std::size_t pos) const
{
return data[pos];
}
cfg_constexpr14 T& operator[](std::size_t pos)
{
return data[pos];
}
constexpr const T& at(std::size_t pos) const
{
return pos < size() ? data[pos] : throw std::out_of_range("The given position is out of range");
}
cfg_constexpr14 T& at(std::size_t pos)
{
return pos < size() ? data[pos] : throw std::out_of_range("The given position is out of range");
}
T data[N > 0 ? N : 1];
};
template <typename T, std::size_t N>
constexpr bool operator==(const array<T, N>& lhs, const array<T, N>& rhs)
{
return equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}
template <typename T, std::size_t N>
constexpr bool operator!=(const array<T, N>& lhs, const array<T, N>& rhs)
{
return !(lhs == rhs);
}
template <std::size_t Index, typename T>
struct tuple_element;
template <std::size_t Index, typename T, std::size_t N>
struct tuple_element<Index, array<T, N>>
{
static_assert(Index < N, "Index out of bounds in cx::tuple_element<>");
using type = T;
constexpr static const T& get(const array<T, N>& a)
{
return a[Index];
}
constexpr static T& get(array<T, N>& a)
{
return a[Index];
}
};
template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;
template <std::size_t Index, typename T, std::size_t N>
constexpr const T& get(const array<T, N>& a)
{
static_assert(Index < N, "Index out of bounds in cx::get<>(const cx::array<>&)");
return a[Index];
}
template <std::size_t Index, typename T, std::size_t N>
constexpr T& get(array<T, N>& a)
{
static_assert(Index < N, "Index out of bounds in cx::get<>(cx::array<>&)");
return a[Index];
}
}
#endif
#ifndef CX_FUNCTIONAL_HPP
#define CX_FUNCTIONAL_HPP 
#include <utility>
namespace cx
{
template <typename T>
struct less
{
constexpr auto operator()(const T& lhs, const T& rhs) const -> decltype(lhs < rhs)
{
return lhs < rhs;
}
};
template <>
struct less<void>
{
template <typename T, typename U>
constexpr auto operator()(T&& lhs, U&& rhs) const
-> decltype(std::forward<T>(lhs) < std::forward<U>(rhs))
{
return std::forward<T>(lhs) < std::forward<U>(rhs);
}
};
template <typename T>
struct equal_to
{
constexpr auto operator()(const T& lhs, const T& rhs) const -> decltype(lhs == rhs)
{
return lhs == rhs;
}
};
template <>
struct equal_to<void>
{
template <typename T, typename U>
constexpr auto operator()(T&& lhs, U&& rhs) const
-> decltype(std::forward<T>(lhs) == std::forward<U>(rhs))
{
return std::forward<T>(lhs) == std::forward<U>(rhs);
}
};
}
#endif
#ifndef CX_ITERATOR_HPP
#define CX_ITERATOR_HPP 
#include <cstddef>
#include <initializer_list>
namespace cx
{
template <typename It>
constexpr It next(It it)
{
return it + 1;
}
template <typename It>
constexpr It prev(It it)
{
return it - 1;
}
template <typename C>
constexpr auto size(const C& c) -> decltype(c.size())
{
return c.size();
}
template <typename T, std::size_t N>
constexpr std::size_t size(const T (&)[N])
{
return N;
}
template <typename C>
constexpr auto empty(const C& c) -> decltype(c.empty())
{
return c.empty();
}
template <typename T, std::size_t N>
constexpr bool empty(const T (&)[N])
{
return false;
}
template <typename E>
constexpr bool empty(std::initializer_list<E> il)
{
return il.size() == 0;
}
template <typename C>
constexpr auto begin(const C& c) -> decltype(c.begin())
{
return c.begin();
}
template <typename C>
constexpr auto begin(C& c) -> decltype(c.begin())
{
return c.begin();
}
template <typename T, std::size_t N>
constexpr T* begin(T (&array)[N])
{
return &array[0];
}
template <typename E>
constexpr const E* begin(std::initializer_list<E> il)
{
return il.begin();
}
template <typename C>
constexpr auto cbegin(const C& c) -> decltype(cx::begin(c))
{
return cx::begin(c);
}
template <typename C>
constexpr auto end(const C& c) -> decltype(c.end())
{
return c.end();
}
template <typename C>
constexpr auto end(C& c) -> decltype(c.end())
{
return c.end();
}
template <typename T, std::size_t N>
constexpr T* end(T (&array)[N])
{
return &array[N];
}
template <typename E>
constexpr const E* end(std::initializer_list<E> il)
{
return il.end();
}
template <typename C>
constexpr auto cend(const C& c) -> decltype(cx::end(c))
{
return cx::end(c);
}
}
#endif
#ifndef CX_STRING_VIEW_HPP
#define CX_STRING_VIEW_HPP 
#include <cstddef>
#include <ostream>
namespace cx
{
template <typename T>
class basic_string_view
{
public:
using value_type = T;
constexpr basic_string_view() = default;
template <std::size_t N>
constexpr basic_string_view(const T (&str)[N]) : data_{&str[0]}, size_{N - 1}
{
}
constexpr basic_string_view(const T* str, std::size_t s) : data_{str}, size_{s}
{
}
constexpr const T* data() const
{
return data_;
}
constexpr std::size_t size() const
{
return size_;
}
constexpr bool empty() const
{
return size() == 0;
}
constexpr const T* begin() const
{
return data_;
}
constexpr const T* cbegin() const
{
return begin();
}
constexpr const T* end() const
{
return data_ + size_;
}
constexpr const T* cend() const
{
return end();
}
private:
const T* data_{nullptr};
std::size_t size_{0};
};
template <typename T>
constexpr bool operator==(const basic_string_view<T>& lhs, const basic_string_view<T>& rhs)
{
return equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}
template <typename T>
constexpr bool operator!=(const basic_string_view<T>& lhs, const basic_string_view<T>& rhs)
{
return !(lhs == rhs);
}
template <typename T>
std::basic_ostream<T>& operator<<(std::basic_ostream<T>& o, const basic_string_view<T>& s)
{
o << s.data();
return o;
}
template <typename T, std::size_t N>
constexpr basic_string_view<T> make_string_view(const T (&str)[N])
{
return {str, N - 1};
}
template <typename T>
constexpr basic_string_view<T> make_string_view(const T* str, std::size_t s)
{
return {str, s};
}
using string_view = basic_string_view<char>;
using u16string_view = basic_string_view<char16_t>;
using u32string_view = basic_string_view<char32_t>;
using wstring_view = basic_string_view<wchar_t>;
namespace literals
{
constexpr string_view operator"" _sv(const char* str, std::size_t s)
{
return {str, s};
}
constexpr u16string_view operator"" _sv(const char16_t* str, std::size_t s)
{
return {str, s};
}
constexpr u32string_view operator"" _sv(const char32_t* str, std::size_t s)
{
return {str, s};
}
constexpr wstring_view operator"" _sv(const wchar_t* str, std::size_t s)
{
return {str, s};
}
}
}
#endif
#ifndef CX_TUPLE_HPP
#define CX_TUPLE_HPP 
#include <cstddef>
#include <type_traits>
#include <utility>
namespace cx
{
template <typename... Types>
struct tuple;
template <typename First, typename... Rest>
struct tuple<First, Rest...> : public tuple<Rest...>
{
template <std::size_t Index, typename T>
friend struct tuple_element;
constexpr tuple() = default;
constexpr tuple(First first, Rest... rest)
: tuple<Rest...>{std::move(rest)...}, first_{std::move(first)}
{
}
private:
First first_;
};
template <>
struct tuple<>
{
};
template <typename... Types>
constexpr tuple<Types...> make_tuple(Types&&... types)
{
return {std::forward<Types>(types)...};
}
template <typename T>
struct tuple_size;
template <typename... Types>
struct tuple_size<tuple<Types...>> : std::integral_constant<std::size_t, sizeof...(Types)>
{
};
template <typename... Types>
struct tuple_size<const tuple<Types...>>
: std::integral_constant<std::size_t, tuple_size<tuple<Types...>>::value>
{
};
template <std::size_t Index, typename T>
struct tuple_element;
template <std::size_t Index, typename First, typename... Rest>
struct tuple_element<Index, tuple<First, Rest...>>
{
using type = typename tuple_element<Index - 1, tuple<Rest...>>::type;
constexpr static const typename tuple_element<Index - 1, tuple<Rest...>>::type& get(
const tuple<First, Rest...>& t)
{
return tuple_element<Index - 1, tuple<Rest...>>::get(t);
}
constexpr static typename tuple_element<Index - 1, tuple<Rest...>>::type& get(
tuple<First, Rest...>& t)
{
return tuple_element<Index - 1, tuple<Rest...>>::get(t);
}
};
template <typename First, typename... Rest>
struct tuple_element<0, tuple<First, Rest...>>
{
using type = First;
constexpr static const First& get(const tuple<First, Rest...>& t)
{
return t.first_;
}
constexpr static First& get(tuple<First, Rest...>& t)
{
return t.first_;
}
};
template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;
template <std::size_t Index, typename First, typename... Rest>
constexpr const tuple_element_t<Index, tuple<First, Rest...>>& get(const tuple<First, Rest...>& t)
{
return tuple_element<Index, tuple<First, Rest...>>::get(t);
}
template <std::size_t Index, typename First, typename... Rest>
constexpr tuple_element_t<Index, tuple<First, Rest...>>& get(tuple<First, Rest...>& t)
{
return tuple_element<Index, tuple<First, Rest...>>::get(t);
}
namespace detail
{
template <std::size_t Index>
struct tuples
{
template <typename... Types1, typename... Types2>
constexpr static bool equal(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
return get<Index>(lhs) == get<Index>(rhs) && tuples<Index - 1>::equal(lhs, rhs);
}
};
template <>
struct tuples<0>
{
template <typename... Types1, typename... Types2>
constexpr static bool equal(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
return get<0>(lhs) == get<0>(rhs);
}
};
}
template <typename... Types1, typename... Types2>
constexpr bool operator==(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
static_assert(
sizeof...(Types1) == sizeof...(Types2),
"Tuples size is not equal in cx::operator==(const cx::tuple<>&, const cx::tuple<>&)");
return detail::tuples<sizeof...(Types1) -1>::equal(lhs, rhs);
}
template <>
constexpr bool operator==(const tuple<>&, const tuple<>&)
{
return true;
}
template <typename... Types1, typename... Types2>
constexpr bool operator!=(const tuple<Types1...>& lhs, const tuple<Types2...>& rhs)
{
return !(lhs == rhs);
}
}
#endif
#ifndef CX_UTILITY_HPP
#define CX_UTILITY_HPP 
#include <cstddef>
#include <type_traits>
#include <utility>
namespace cx
{
template <typename First, typename Second>
struct pair
{
using first_type = First;
using second_type = Second;
constexpr pair() = default;
constexpr pair(First frst, Second scnd) : first{std::move(frst)}, second{std::move(scnd)}
{
}
First first;
Second second;
};
template <typename First, typename Second>
constexpr bool operator==(const pair<First, Second>& lhs, const pair<First, Second>& rhs)
{
return lhs.first == rhs.first && lhs.second == rhs.second;
}
template <typename First, typename Second>
constexpr bool operator!=(const pair<First, Second>& lhs, const pair<First, Second>& rhs)
{
return !(lhs == rhs);
}
template <typename First, typename Second>
constexpr pair<First, Second> make_pair(First&& first, Second&& second)
{
return {std::forward<First>(first), std::forward<Second>(second)};
}
template <typename T>
struct tuple_size;
template <typename First, typename Second>
struct tuple_size<pair<First, Second>> : std::integral_constant<std::size_t, 2>
{
};
template <typename First, typename Second>
struct tuple_size<const pair<First, Second>>
: std::integral_constant<std::size_t, tuple_size<pair<First, Second>>::value>
{
};
template <std::size_t Index, typename T>
struct tuple_element;
template <typename First, typename Second>
struct tuple_element<0, pair<First, Second>>
{
using type = First;
constexpr static const First& get(const pair<First, Second>& p)
{
return p.first;
}
constexpr static First& get(pair<First, Second>& p)
{
return p.first;
}
};
template <typename First, typename Second>
struct tuple_element<1, pair<First, Second>>
{
using type = Second;
constexpr static const Second& get(const pair<First, Second>& p)
{
return p.second;
}
constexpr static Second& get(pair<First, Second>& p)
{
return p.second;
}
};
template <std::size_t Index, typename T>
using tuple_element_t = typename tuple_element<Index, T>::type;
template <std::size_t Index, typename First, typename Second>
constexpr const tuple_element_t<Index, pair<First, Second>>& get(const pair<First, Second>& p)
{
return tuple_element<Index, pair<First, Second>>::get(p);
}
template <std::size_t Index, typename First, typename Second>
constexpr tuple_element_t<Index, pair<First, Second>>& get(pair<First, Second>& p)
{
return tuple_element<Index, pair<First, Second>>::get(p);
}
template <typename First, typename Second>
constexpr const First& get(const pair<First, Second>& p)
{
return p.first;
}
template <typename First, typename Second>
constexpr First& get(pair<First, Second>& p)
{
return p.first;
}
template <typename Second, typename First>
constexpr const Second& get(const pair<First, Second>& p)
{
return p.second;
}
template <typename Second, typename First>
constexpr Second& get(pair<First, Second>& p)
{
return p.second;
}
}
#endif
#endif
