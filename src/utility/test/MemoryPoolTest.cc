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
#include <memory>

#include "MemoryPool.hh"
#include "Vector.hh"
#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"

using namespace testing;

using ieda::FastPoolAllocator;
using ieda::ObjectPool;
using ieda::PoolAllocator;
using ieda::SingletonPool;
using ieda::Vector;

namespace {

class A {
  std::string _str;

 public:
  explicit A(std::string str) : _str(str) {}
  A(const A& other) : _str(other._str) {
    std::cout << "copy"
              << "\n";
  }

  A(A&& other) : _str(std::move(other._str)) {
    std::cout << "move"
              << "\n";
  }

  ~A() {
    std::cout << "desctructor"
              << "\n";
  }

  static void* operator new(size_t size) {
    std::cout << "operator new size " << size << std::endl;
    return PoolAllocator<A>().allocate(size);
  }

  static void operator delete(void* pointee) {
    std::cout << "operator delete" << std::endl;
    PoolAllocator<A>().destroy(static_cast<A*>(pointee));
  }

  static void* operator new(std::size_t count, void* ptr) {
    std::cout << "placement new size " << std::endl;
    return ptr;
  }

  A& operator=(const A& other) {
    _str = other._str;
    std::cout << "copy"
              << "\n";
    return *this;
  }

  A& operator=(A&& other) {
    _str = std::move(other._str);
    std::cout << "move"
              << "\n";
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os, A const& rhs) {
    os << rhs._str;
    return os;
  }
};

A func(const A& test) {
  A test1 = std::move(test);
  return test1;
}

A func(A&& test) {
  A test1 = test;
  return test1;
}

auto timeit = [](std::function<int()> set_test, std::string what = "") {
  auto start = std::chrono::system_clock::now();
  int setsize = set_test();
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> time = stop - start;
  if (what.size() > 0 && setsize > 0) {
    std::cout << std::fixed << std::setprecision(2) << time.count()
              << "  ms for " << what << '\n';
  }
};

TEST(ObjectPoolTest, construct1) {
  //   A test("test");

  //   // auto test1 = test;
  //   auto& test2 = test;
  //   auto&& test3 = test;

  //   auto test4 = func(test);
  //   auto& test5 = func(std::move(test));
  //   auto&& test6 = func(std::move(test));

  constexpr size_t count = 1;

  auto boost_func = []() -> int {
    ObjectPool<A> p(count);
    for (size_t i = 0; i < count; ++i) {
      A* const t = p.construct("test");
      // std::cout << *t << std::endl;
      // p.destroy(t);
      // Do something with t; don't take the time to free() it
    }
    return 1;
  };  // on function exit, p is destroyed, and all destructors for the X objects
      // are called

  auto stl_func = []() -> int {
    for (size_t i = 0; i < count; ++i) {
      A* const t = new A("test");
      // std::cout << *t << std::endl;
      // delete t;
      // Do something with t; don't take the time to free() it
    }
    return 1;
  };

  timeit(boost_func, "boost");
  timeit(stl_func, "stl");
  timeit(boost_func, "boost");
  timeit(stl_func, "stl");
}

TEST(ObjectPoolTest, construct2) {
  constexpr size_t count = 10000;
  ObjectPool<A> p(count);
  auto boost_func = [&p]() -> int {
    for (size_t i = 0; i < count; ++i) {
      A* const t = p.construct("test");
      // std::cout << *t << std::endl;
      p.destroy(t);
      // Do something with t; don't take the time to free() it
    }
    return 1;
  };  // on function exit, p is destroyed, and all destructors for the X objects
      // are called

  std::allocator<A> alloc;
  auto p1 = alloc.allocate(count);
  auto stl_func = [&alloc, &p1]() -> int {
    auto q = p1;
    for (size_t i = 0; i < count; ++i) {
      alloc.construct(q, "test");
      // std::cout << *q << std::endl;
      alloc.destroy(q);
      q++;
      // Do something with t; don't take the time to free() it
    }
    return 1;
  };

  timeit(boost_func, "boost");
  timeit(stl_func, "stl");
  timeit(boost_func, "boost");
  timeit(stl_func, "stl");
}

struct pool_tag {};  // for tag
using spl = SingletonPool<pool_tag, sizeof(int)>;

TEST(SingtonPool, construct) {
  int* p = static_cast<int*>(spl::malloc());
  *p = 3;
  std::cout << *p << std::endl;
  ASSERT_TRUE(spl::is_from(p));

  spl::free(p);

  spl::purge_memory();
}

TEST(SingtonPool, allocator) {
  constexpr size_t count = 1;
  auto boost_alloc = []() -> int {
    std::vector<A, PoolAllocator<A>> v;
    for (int i = 0; i < count; ++i) v.emplace_back("test");

    // for (auto p : v) {
    //   std::cout << p << std::endl;
    // }

    return 1;
  };

  auto stl_alloc = []() -> int {
    std::vector<A> v;
    for (int i = 0; i < count; ++i) v.emplace_back("test");

    // for (auto p : v) {
    //   std::cout << p << std::endl;
    // }

    return 1;
  };

  auto abseil_alloc = []() -> int {
    Vector<A> v;
    for (int i = 0; i < count; ++i) v.emplace_back("test");

    // for (auto p : v) {
    //   std::cout << p << std::endl;
    // }

    return 1;
  };

  timeit(boost_alloc, "boost");
  timeit(stl_alloc, "stl");
  timeit(abseil_alloc, "google");
  timeit(boost_alloc, "boost");
  timeit(stl_alloc, "stl");
  timeit(abseil_alloc, "google");

  boost::singleton_pool<boost::pool_allocator_tag,
                        sizeof(int)>::release_memory();
}
}  // namespace