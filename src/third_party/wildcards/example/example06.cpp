// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "cx/string_view.hpp"
#include "wildcards/match.hpp"

#ifndef __DJGPP__
using std::round;
#endif

enum class nucleobase
{
  adenine,
  cytosine,
  guanine,
  thymine
};

nucleobase to_base(int n)
{
  switch (n)
  {
    case 0:
      return nucleobase::adenine;
    case 1:
      return nucleobase::cytosine;
    case 2:
      return nucleobase::guanine;
    case 3:
      return nucleobase::thymine;
  }

  throw std::out_of_range("Input out of range");
}

char to_char(nucleobase base)
{
  switch (base)
  {
    case nucleobase::adenine:
      return 'A';
    case nucleobase::cytosine:
      return 'C';
    case nucleobase::guanine:
      return 'G';
    case nucleobase::thymine:
      return 'T';
  }

  throw std::out_of_range("Input out of range");
}

std::ostream& operator<<(std::ostream& o, const std::vector<nucleobase>& sequence)
{
  for (auto base : sequence)
  {
    o << to_char(base);
  }

  return o;
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "usage: " << argv[0] << " pattern" << std::endl;
    return EXIT_SUCCESS;
  }

  if (argc != 2)
  {
    std::cerr << "invalid arguments" << std::endl;
    return EXIT_FAILURE;
  }

  auto pattern = cx::make_string_view(argv[1], std::strlen(argv[1]));

  auto generator = std::default_random_engine{};
  auto n_distribution = std::normal_distribution<double>{5.0, 2.0};
  auto u_distribution = std::uniform_int_distribution<int>{0, 3};

  auto count = 0;
  auto found = 0;
  do
  {
    auto length = n_distribution(generator);
    if (length < 0.0 || length > 10.0)
    {
      continue;
    }

    auto size = static_cast<std::vector<nucleobase>::size_type>(round(length));

    auto sequence = std::vector<nucleobase>{};
    sequence.reserve(size);
    for (decltype(size) n = 0; n < size; ++n)
    {
      sequence.push_back(to_base(u_distribution(generator)));
    }

    if (wildcards::match(sequence, pattern,
                         [](nucleobase base, char c) { return to_char(base) == c; }))
    {
      std::cout << sequence << std::endl;
      ++found;
    }

    ++count;
  } while (count < 1000000);

  std::cout << "Generated : " << count << std::endl;
  std::cout << "Found     : " << found << std::endl;

  return EXIT_SUCCESS;
}
