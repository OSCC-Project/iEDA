//
// Pretty printing for C++ STL containers
//
// Usage: operator<< will "just work" for most STL containers, for example,
//      std::vector<int> nums = {1, 2, 10};
//      std::cout << nums << std::endl;
//     This code piece will show "1 2 10".
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>

namespace utils {

// SFINAE HasBeginEnd

template <typename T, typename = void>
struct HasBeginEnd : std::false_type
{
};
template <typename T>
struct HasBeginEnd<T, decltype((void) std::begin(std::declval<T>()), (void) std::end(std::declval<T>()))> : std::true_type
{
};

// Holds the delimiter values for a specific character type

template <typename TChar>
struct DelimitersValues
{
  using char_type = TChar;
  const char_type* prefix;
  const char_type* delimiter;
  const char_type* postfix;
};

// Defines the delimiter values for a specific container and character type

template <typename T, typename TChar = char>
struct Delimiters
{
  using type = DelimitersValues<TChar>;
  static const type kValues;
};

// Functor to print containers. You can use this directly if you want
// to specificy a non-default Delimiters type. The printing logic can
// be customized by specializing the nested template.

template <typename T, typename TChar = char, typename TCharTraits = std::char_traits<TChar>, typename TDelimiters = Delimiters<T, TChar>>
struct PrintContainerHelper
{
  using delimiters_type = TDelimiters;
  using ostream_type = std::basic_ostream<TChar, TCharTraits>;

  template <typename U>
  struct Printer
  {
    static void printBody(const U& c, ostream_type& stream)
    {
      auto it = std::begin(c);
      const auto the_end = std::end(c);

      if (it != the_end) {
        for (;;) {
          stream << *it;

          if (++it == the_end)
            break;

          if (delimiters_type::values.delimiter != NULL)
            stream << delimiters_type::values.delimiter;
        }
      }
    }
  };

  PrintContainerHelper(const T& container) : container_(container) {}

  inline void operator()(ostream_type& stream) const
  {
    if (delimiters_type::values.prefix != NULL)
      stream << delimiters_type::values.prefix;

    Printer<T>::printBody(container_, stream);

    if (delimiters_type::values.postfix != NULL)
      stream << delimiters_type::values.postfix;
  }

 private:
  const T& container_;
};

// Specialization for pairs

template <typename T, typename TChar, typename TCharTraits, typename TDelimiters>
template <typename T1, typename T2>
struct PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::Printer<std::pair<T1, T2>>
{
  using ostream_type = typename PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::ostream_type;

  static void printBody(const std::pair<T1, T2>& c, ostream_type& stream)
  {
    stream << c.first;
    if (PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::delimiters_type::values.delimiter != NULL)
      stream << PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::delimiters_type::values.delimiter;
    stream << c.second;
  }
};

// Specialization for tuples

template <typename T, typename TChar, typename TCharTraits, typename TDelimiters>
template <typename... Args>
struct PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::Printer<std::tuple<Args...>>
{
  using ostream_type = typename PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::ostream_type;
  using element_type = std::tuple<Args...>;

  template <std::size_t I>
  struct Int
  {
  };

  static void printBody(const element_type& c, ostream_type& stream) { tuplePrint(c, stream, Int<0>()); }

  static void tuplePrint(const element_type&, ostream_type&, Int<sizeof...(Args)>) {}

  static void tuplePrint(const element_type& c, ostream_type& stream,
                         typename std::conditional<sizeof...(Args) != 0, Int<0>, std::nullptr_t>::type)
  {
    stream << std::get<0>(c);
    tuplePrint(c, stream, Int<1>());
  }

  template <std::size_t N>
  static void tuplePrint(const element_type& c, ostream_type& stream, Int<N>)
  {
    if (PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::delimiters_type::values.delimiter != NULL)
      stream << PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>::delimiters_type::values.delimiter;

    stream << std::get<N>(c);

    tuplePrint(c, stream, Int<N + 1>());
  }
};

// Prints a PrintContainerHelper to the specified stream.

template <typename T, typename TChar, typename TCharTraits, typename TDelimiters>
inline std::basic_ostream<TChar, TCharTraits>& operator<<(std::basic_ostream<TChar, TCharTraits>& stream,
                                                          const PrintContainerHelper<T, TChar, TCharTraits, TDelimiters>& helper)
{
  helper(stream);
  return stream;
}

// Basic IsContainer template; specialize to derive from std::true_type for all desired container types

template <typename T>
struct IsContainer : std::integral_constant<bool, HasBeginEnd<T>::value>
{
};

template <typename... T>
struct IsContainer<std::pair<T...>> : std::true_type
{
};

template <typename... T>
struct IsContainer<std::tuple<T...>> : std::true_type
{
};

// Default Delimiters

template <typename T>
struct Delimiters<T, char>
{
  static constexpr DelimitersValues<char> kValues = {"[", ", ", "]"};
};

// Delimiters for (unordered_)(multi)set

template <typename... T>
struct Delimiters<std::set<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"{", ", ", "}"};
};

template <typename... T>
struct Delimiters<std::multiset<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"{", ", ", "}"};
};

template <typename... T>
struct Delimiters<std::unordered_set<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"{", ", ", "}"};
};

template <typename... T>
struct Delimiters<std::unordered_multiset<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"{", ", ", "}"};
};

// Delimiters for pair and tuple

template <typename... T>
struct Delimiters<std::pair<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"(", ", ", ")"};
};

template <typename... T>
struct Delimiters<std::tuple<T...>>
{
  static constexpr DelimitersValues<char> kValues = {"(", ", ", ")"};
};

// Type-erasing helper class for easy use of custom Delimiters.
// Requires TCharTraits = std::char_traits<TChar> and TChar = char or wchar_t, and MyDelims needs to be defined for
// TChar. Usage: "cout << pretty_print::CustomDelims<MyDelims>(x)".

struct CustomDelimsInterface
{
  virtual ~CustomDelimsInterface() {}
  virtual std::ostream& stream(std::ostream&) = 0;
};

template <typename T, typename Delims>
struct CustomDelimsWrapper : CustomDelimsInterface
{
  CustomDelimsWrapper(const T& t_) : t(t_) {}

  std::ostream& stream(std::ostream& s) { return s << PrintContainerHelper<T, char, std::char_traits<char>, Delims>(t); }

 private:
  const T& t;
};

template <typename Delims>
struct CustomDelims
{
  template <typename Container>
  CustomDelims(const Container& c) : base(new CustomDelimsWrapper<Container, Delims>(c))
  {
  }

  std::unique_ptr<CustomDelimsInterface> base;
};

template <typename TChar, typename TCharTraits, typename Delims>
inline std::basic_ostream<TChar, TCharTraits>& operator<<(std::basic_ostream<TChar, TCharTraits>& s, const CustomDelims<Delims>& p)
{
  return p.base->stream(s);
}

}  // namespace utils

// Main magic entry point: An overload snuck into namespace std.
// Can we do better?

namespace std {
// Prints a container to the stream using default Delimiters

template <typename T, typename TChar, typename TCharTraits>
inline typename enable_if<::utils::IsContainer<T>::value, basic_ostream<TChar, TCharTraits>&>::type operator<<(
    basic_ostream<TChar, TCharTraits>& stream, const T& container)
{
  return stream << ::utils::PrintContainerHelper<T, TChar, TCharTraits>(container);
}

}  // namespace std