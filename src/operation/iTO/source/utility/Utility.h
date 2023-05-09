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
#pragma once

#include <ostream>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <vector>

using std::max;
using std::string;
using std::vector;

namespace ito {
using std::max;

int metersToDbu(double dist, int dbu);

double dbuToMeters(int dist, int dbu);

// "Fuzzy" floating point comparisons that allow some tolerance.
bool fuzzyEqual(float f1, float f2);

bool fuzzyLess(float f1, float f2);

bool fuzzyLessEqual(float f1, float f2);

bool fuzzyGreater(float f1, float f2);

bool fuzzyGreaterEqual(float f1, float f2);

bool stringLess(const char *s1, const char *s2);
bool stringEqual(const char *str1, const char *str2);

void hashIncr(size_t &hash, size_t add);
} // namespace ito
