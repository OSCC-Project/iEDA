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
