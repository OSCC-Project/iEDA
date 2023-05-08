//#include <array>
#include <ctime>
#include <iostream>
//#include <vector>
#include "Array.hh"
#include "absl/time/clock.h"
#include "absl/time/time.h"
//#include "Vector.h"
#include "gtest/gtest.h"
TEST(arrayCompare, add) {
  absl::Time startTime, endTime;
  startTime = absl::Now();

  constexpr size_t array_size = 99000000;
  ieda::Array<size_t, 9900000>* ar1 =
      new ieda::Array<size_t, 9900000>(array_size);
  for (size_t i = 0; i < array_size; i++) {
    (*ar1)[i] = i + 1;
  }
  endTime = absl::Now();
  absl::Duration duration = endTime - startTime;
  int64_t duration_time = duration / absl::Nanoseconds(1);
  std::cout << "the array run time is = " << duration_time << "ns" << std::endl;
}