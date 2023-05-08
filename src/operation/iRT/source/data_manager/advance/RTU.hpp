#pragma once

#include <assert.h>
#include <libgen.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <any>
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <experimental/source_location>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <ostream>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "libfort/fort.hpp"
#include "omp.h"

using irt_int = int32_t;
#define IRT_INT_MIN (INT32_MIN);
#define IRT_INT_MAX (INT32_MAX);
#define DBL_ERROR 1E-5

template <class... Fs>
struct Overload : Fs...
{
  using Fs::operator()...;
};
template <class... Fs>
Overload(Fs...) -> Overload<Fs...>;
