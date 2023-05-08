#include <array>
#include <cstring>
#include <ctime>
#include <iostream>

#include "absl/time/clock.h"
#include "absl/time/time.h"
//#include "Array.h"
#include "gtest/gtest.h"

TEST(arrayCompare, addSTL) {
  absl::Time startTime1, endTime1;
  constexpr size_t array_size = 99000000;
  startTime1 = absl::Now();
  std::array<size_t, array_size>* ar2 = new std::array<size_t, array_size>;
  for (size_t i = 0; i < array_size; i++) {
    (*ar2)[i] = i + 1;
  }
  endTime1 = absl::Now();
  absl::Duration duration = endTime1 - startTime1;
  std::cout << "the STL array run time is = " << duration / absl::Nanoseconds(1)
            << "ns" << std::endl;
}
