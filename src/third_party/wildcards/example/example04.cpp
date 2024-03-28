// Copyright Tomas Zeman 2018.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>
#include <string>

#include "cx/string_view.hpp"
#include "wildcards/match.hpp"

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "usage: " << argv[0] << " [-v] sequence pattern" << std::endl;
    return EXIT_SUCCESS;
  }

  auto sequence_index = 1;
  auto pattern_index = 2;
  auto verbose = false;

  if (argc == 4 && std::strcmp(argv[1], "-v") == 0)
  {
    ++sequence_index;
    ++pattern_index;
    verbose = true;
  }
  else if (argc != 3)
  {
    std::cerr << "invalid arguments" << std::endl;
    return EXIT_FAILURE;
  }

  auto sequence = cx::make_string_view(argv[sequence_index], std::strlen(argv[sequence_index]));
  auto pattern = cx::make_string_view(argv[pattern_index], std::strlen(argv[pattern_index]));
  auto result = wildcards::match(sequence, pattern);

  if (!result)
  {
    if (verbose)
    {
      std::cout << std::string{result.s, result.send} << '\n';

      for (decltype(result.s1 - result.s) n = 0; n < result.s1 - result.s; ++n)
      {
        std::cout << ' ';
      }
      std::cout << '^' << '\n';

      std::cout << std::string{result.p, result.pend} << '\n';

      for (decltype(result.p1 - result.p) n = 0; n < result.p1 - result.p; ++n)
      {
        std::cout << ' ';
      }
      std::cout << '^' << '\n';
    }

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
