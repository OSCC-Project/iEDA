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
#include <chrono>
#include <functional>
#include <memory>
#include <utility>

#include "Set.hh"
#include "gtest/gtest.h"

using ieda::Multiset;
using ieda::BTreeSet;

namespace {

TEST(SetTest, Ctor) {
  BTreeSet<int> bset = {1, 2, 3};

  EXPECT_TRUE(bset.hasKey(1));
}

TEST(SetTest, Subtract) {
  BTreeSet<int> bset1 = {1, 2, 3};
  BTreeSet<int> bset2 = {2, 3, 4};

  bset1.subtract(bset2);

  BTreeSet<int> bset3 = {1};

  EXPECT_TRUE(BTreeSet<int>::equal(&bset1, &bset3));
}

TEST(SetTest, less) {
  BTreeSet<int> bset1 = {1, 2, 3};
  BTreeSet<int> bset2 = {2, 3, 4};

  bset1 < bset2;
}

TEST(SetTest, swap) {
  BTreeSet<int> bset1 = {1, 2, 3};
  BTreeSet<int> bset2 = {2, 3, 4};

  swap(bset1, bset2);

  EXPECT_FALSE(bset1.hasKey(1));
}

TEST(SetTest, insert_emplace) {
  class Dew {
   private:
    int a;
    int b;
    int c;

   public:
    Dew(int _a, int _b, int _c) : a(_a), b(_b), c(_c) {}

    bool operator<(const Dew& other) const {
      if (a < other.a) return true;
      if (a == other.a && b < other.b) return true;
      return (a == other.a && b == other.b && c < other.c);
    }
  };

  const int nof_operations = 120;

  auto set_emplace = [=]() -> int {
    BTreeSet<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.emplace(i, j, k);

    return set.size();
  };

  auto set_insert = [=]() -> int {
    BTreeSet<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

    return set.size();
  };

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

  timeit(set_insert, "insert");
  timeit(set_emplace, "emplace");
  timeit(set_insert, "insert");
  timeit(set_emplace, "emplace");
}

TEST(SetTest, extract) {
  std::unique_ptr<int> p = std::make_unique<int>(2);
  BTreeSet<std::unique_ptr<int>> cont;

  cont.insert(std::move(p));

  auto print = [](std::unique_ptr<int>& p) { std::cout << " " << *p; };

  std::cout << "Start:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Extract node handle and change key
  auto nh = cont.extract(cont.begin());
  *(nh.value()) = 4;

  std::cout << "After extract and before insert:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Insert node handle back
  cont.insert(move(nh));

  std::cout << "End:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  std::unique_ptr<int> q = std::make_unique<int>(4);
  BTreeSet<std::unique_ptr<int>> cont1;
  cont1.insert(std::move(q));

  EXPECT_TRUE(BTreeSet<std::unique_ptr<int>>::equal(&cont, &cont));
}

TEST(SetTest, equal) {
  BTreeSet<int> cont = {1, 2, 3};
  BTreeSet<int> cont1 = {1, 2, 3};

  EXPECT_TRUE(BTreeSet<int>::equal(&cont, &cont1));
}

TEST(SetTest, operator1) {
  BTreeSet<int> cont = {1, 2, 3};
  cont << 4;
  for (auto& p : cont) {
    std::cout << p << std::endl;
  }
}

TEST(SetTest, operator2) {
  BTreeSet<int> cont = {1, 2, 3};
  BTreeSet<int> cont1 = {4, 5, 6};

  cont |= cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }
}

TEST(SetTest, operator3) {
  BTreeSet<int> cont = {1, 2, 3};
  BTreeSet<int> cont1 = {4, 5, 6};

  cont |= std::move(cont1);

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator4) {
  BTreeSet<int> cont = {1, 2, 3};

  cont |= 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator5) {
  BTreeSet<int> cont = {1, 2, 3};
  BTreeSet<int> cont1 = {2, 3, 4};

  cont &= cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {2, 3};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator6) {
  BTreeSet<int> cont = {1, 2, 3};

  cont &= 3;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {3};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator7) {
  BTreeSet<int> cont = {1, 2, 3};

  cont += 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator8) {
  BTreeSet<int> cont = {1, 2, 3};

  cont += {4};

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator9) {
  BTreeSet<int> cont = {1, 2, 3, 4};

  cont -= {4};

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator10) {
  BTreeSet<int> cont = {1, 2, 3, 4};

  cont -= 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator11) {
  BTreeSet<int> cont = {1, 2, 3, 4};
  BTreeSet<int> cont1 = {5};

  cont = cont | cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4, 5};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator12) {
  BTreeSet<int> cont = {1, 2, 3, 4};
  BTreeSet<int> cont1 = {4};

  cont = cont & cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {4};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, operator13) {
  BTreeSet<int> cont = {1, 2, 3, 4};
  BTreeSet<int> cont1 = {5};

  cont = cont + cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4, 5};
  EXPECT_EQ(cont, result);
  EXPECT_TRUE(cont1.empty());
}

TEST(SetTest, operator14) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {5};

  cont = cont - cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  BTreeSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, haskey) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  EXPECT_TRUE(cont.hasKey(4));
}

TEST(SetTest, issubset) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3};
  EXPECT_TRUE(cont.isSubset(&cont1));
}

TEST(SetTest, insertset) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3, 6};
  cont.insertSet(&cont1);
  BTreeSet<int> result = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(cont, result);
}

TEST(SetTest, intersects) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3, 6};
  EXPECT_TRUE(BTreeSet<int>::intersects(&cont, &cont1));
}

TEST(SetTest, nonmember1) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3};

  EXPECT_FALSE(cont == cont1);
}

TEST(SetTest, nonmember2) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3, 6};

  EXPECT_TRUE(cont != cont1);
}

TEST(SetTest, nonmember3) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3, 5, 6};

  EXPECT_TRUE(cont < cont1);
}

TEST(SetTest, nonmember4) {
  BTreeSet<int> cont = {1, 2, 3, 4, 5};
  BTreeSet<int> cont1 = {1, 2, 3, 5, 6};

  EXPECT_TRUE(cont <= cont1);
}

TEST(SetTest, nonmember5) {
  BTreeSet<int> cont = {1, 2, 3, 5, 7};
  BTreeSet<int> cont1 = {1, 2, 3, 5, 6};

  EXPECT_TRUE(cont >= cont1);
}

TEST(SetTest, nonmember6) {
  BTreeSet<int> cont = {1, 2, 3, 5, 7};
  BTreeSet<int> cont1 = {1, 2, 3, 5, 6};

  EXPECT_TRUE(cont > cont1);
}

TEST(SetTest, nonmember7) {
  BTreeSet<int> cont = {1, 2, 3, 5, 7};
  BTreeSet<int> cont1 = {1, 2, 3, 5, 6};

  swap(cont, cont1);

  BTreeSet<int> result_cont = {1, 2, 3, 5, 6};

  EXPECT_EQ(cont, result_cont);
}

TEST(MultisetTest, ctor) {
  Multiset<int> bmultiset = {1, 1, 3};

  for (auto& p : bmultiset) {
    std::cout << p << std::endl;
  }
}

TEST(MultisetTest, capacity) {
  Multiset<int> bmultiset;

  EXPECT_TRUE(bmultiset.empty());

  std::cout << "max size : " << bmultiset.max_size() << std::endl;
}

TEST(MultisetTest, modifier1) {
  Multiset<std::unique_ptr<int, std::function<void(int*)>>> bmultiset;
  auto deleter = [](int* p) {
    std::cout << "delete " << *p << std::endl;
    delete p;
  };

  bmultiset.emplace(new int(2), deleter);
  bmultiset.emplace_hint(bmultiset.begin(), new int(1), deleter);

  std::cout << "remove one element" << std::endl;
  bmultiset.erase(++bmultiset.begin());
  std::cout << "add one element" << std::endl;
  bmultiset.emplace(new int(3), deleter);

  auto nh = bmultiset.extract(bmultiset.begin());
  bmultiset.insert(std::move(nh));
}

TEST(MultisetTest, modifier2) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  bmultiset1.merge(bmultiset2);

  for (auto& p : bmultiset1) {
    std::cout << p << std::endl;
  }

  Multiset<int> result = {1, 2, 3, 2, 3, 4};
  EXPECT_EQ(bmultiset1, result);
  EXPECT_TRUE(bmultiset2.empty());
}

TEST(MultisetTest, swap) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  bmultiset1.swap(bmultiset2);

  for (auto& p : bmultiset1) {
    std::cout << p << std::endl;
  }

  Multiset<int> result = {2, 3, 4};
  EXPECT_EQ(bmultiset1, result);
}

TEST(MultisetTest, lookup) {
  Multiset<int> bmultiset1 = {1, 1, 3};
  EXPECT_TRUE(bmultiset1.contains(1));
  EXPECT_EQ(bmultiset1.count(1), 2);

  auto range = bmultiset1.equal_range(1);
  for (auto p = range.first; p != range.second; p++) {
    std::cout << *p << std::endl;
  }
}

TEST(MultisetTest, nonmember1) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_FALSE(bmultiset1 == bmultiset2);
}

TEST(MultisetTest, nonmember2) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_TRUE(bmultiset1 != bmultiset2);
}

TEST(MultisetTest, nonmember3) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_TRUE(bmultiset1 < bmultiset2);
}

TEST(MultisetTest, nonmember4) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_TRUE(bmultiset1 <= bmultiset2);
}

TEST(MultisetTest, nonmember5) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_FALSE(bmultiset1 >= bmultiset2);
}

TEST(MultisetTest, nonmember6) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  EXPECT_FALSE(bmultiset1 > bmultiset2);
}

TEST(MultisetTest, nonmember7) {
  Multiset<int> bmultiset1 = {1, 2, 3};
  Multiset<int> bmultiset2 = {2, 3, 4};

  swap(bmultiset1, bmultiset2);

  Multiset<int> result = {2, 3, 4};

  EXPECT_EQ(bmultiset1, result);
}

class Dew {
 private:
  int _a;
  int _b;
  int _c;

 public:
  Dew(int a, int b, int c) : _a(a), _b(b), _c(c) {}

  bool operator<(const Dew& other) const {
    if (_a < other._a) return true;
    if (_a == other._a && _b < other._b) return true;
    return (_a == other._a && _b == other._b && _c < other._c);
  }
};

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

TEST(SetTest, perf1) {
  const int nof_operations = 200;

  auto set_emplace = [=]() -> int {
    BTreeSet<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.emplace(i, j, k);

    return set.size();
  };

  auto stl_set_emplace = [=]() -> int {
    std::set<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.emplace(i, j, k);

    return set.size();
  };

  timeit(stl_set_emplace, "stl emplace");
  timeit(set_emplace, "emplace");
  timeit(stl_set_emplace, "stl emplace");
  timeit(set_emplace, "emplace");
}

TEST(SetTest, perf2) {
  const int nof_operations = 200;

  auto set_insert = [=]() -> int {
    BTreeSet<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

    return set.size();
  };

  auto stl_set_insert = [=]() -> int {
    std::set<Dew> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

    return set.size();
  };

  timeit(stl_set_insert, "stl insert");
  timeit(set_insert, "insert");
  timeit(stl_set_insert, "stl insert");
  timeit(set_insert, "insert");
}

TEST(SetTest, perf3) {
  const int nof_operations = 200;

  BTreeSet<Dew> set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

  std::set<Dew> stl_set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) stl_set.insert(Dew(i, j, k));

  auto set_find = [=, &set]() -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.find(Dew(i, j, k));

    return 1;
  };

  auto stl_set_find = [=, &stl_set]() -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) stl_set.find(Dew(i, j, k));
    return 1;
  };

  timeit(stl_set_find, "stl find");
  timeit(set_find, "find");
  timeit(stl_set_find, "stl find");
  timeit(set_find, "find");
}

TEST(SetTest, perf4) {
  const int nof_operations = 200;

  BTreeSet<Dew> set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

  std::set<Dew> stl_set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) stl_set.insert(Dew(i, j, k));

  auto set_erase = [=]() mutable -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.erase(Dew(i, j, k));

    return 1;
  };

  auto stl_set_erase = [=]() mutable -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) stl_set.erase(Dew(i, j, k));
    return 1;
  };

  timeit(stl_set_erase, "stl erase");
  timeit(set_erase, "erase");
  timeit(stl_set_erase, "stl erase");
  timeit(set_erase, "erase");
}

}  // namespace