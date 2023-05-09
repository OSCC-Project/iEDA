// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "Utility.h"

namespace ito {

constexpr static float float_equal_tolerance = 1E-15F;

int metersToDbu(double dist, int dbu) {
  return dist * dbu * 1e+6;
}


double dbuToMeters(int dist, int dbu) {
  return dist / (dbu * 1e+6);
}


bool fuzzyEqual(float f1, float f2) {
  if (f1 == f2) {
    return true;
  } else if (f1 == 0.0) {
    return abs(f2) < float_equal_tolerance;
  } else if (f2 == 0.0) {
    return abs(f1) < float_equal_tolerance;
  } else {
    return abs(f1 - f2) < 1E-6F * max(abs(f1), abs(f2));
  }
}


bool fuzzyLess(float f1, float f2) {
  return f1 < f2 && !fuzzyEqual(f1, f2);
}


bool fuzzyLessEqual(float f1, float f2) {
  return f1 < f2 || fuzzyEqual(f1, f2);
}


bool fuzzyGreater(float f1, float f2) {
  return f1 > f2 && !fuzzyEqual(f1, f2);
}


bool fuzzyGreaterEqual(float f1, float f2) {
  return f1 > f2 || fuzzyEqual(f1, f2);
}

bool stringLess(const char *str1, const char *str2) {
  return strcmp(str1, str2) < 0;
}

bool stringEqual(const char *str1, const char *str2) {
  return (str1 == nullptr && str2 == nullptr)
    || (str1 && str2 && strcasecmp(str1, str2) == 0);
}

void hashIncr(size_t &hash, size_t add) {
  hash = ((hash << 5) + hash) ^ add;
}
} // namespace ito
