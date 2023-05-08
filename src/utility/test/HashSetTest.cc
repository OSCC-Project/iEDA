#include <chrono>
#include <functional>
#include <memory>
#include <utility>

#include "HashSet.hh"
#include "gtest/gtest.h"

using ieda::HashMultiset;
using ieda::HashSet;

namespace {

TEST(HashSetTest, Ctor) {
  HashSet<int> hset = {1, 2, 3};

  EXPECT_TRUE(hset.hasKey(1));
}

TEST(HashSetTest, Subtract) {
  HashSet<int> hset1 = {1, 2, 3};
  HashSet<int> hset2 = {2, 3, 4};

  hset1.subtract(hset2);

  HashSet<int> hset3 = {1};

  EXPECT_TRUE(HashSet<int>::equal(&hset1, &hset3));
}

TEST(HashSetTest, swap) {
  HashSet<int> hset1 = {1, 2, 3};
  HashSet<int> hset2 = {2, 3, 4};

  swap(hset1, hset2);

  EXPECT_FALSE(hset1.hasKey(1));
}

TEST(HashSetTest, extract) {
  HashSet<int> cont = {1, 2, 3};

  auto print = [](int p) { std::cout << " " << p; };

  std::cout << "Start:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Extract node handle and change key
  auto nh = cont.extract(cont.begin());

  std::cout << "After extract and before insert:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Insert node handle back
  cont.insert(move(nh));

  std::cout << "End:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  EXPECT_TRUE(HashSet<int>::equal(&cont, &cont));
}

TEST(HashSetTest, equal) {
  HashSet<int> cont = {1, 2, 3};
  HashSet<int> cont1 = {1, 2, 3};

  EXPECT_TRUE(HashSet<int>::equal(&cont, &cont1));
}

TEST(HashSetTest, operator1) {
  HashSet<int> cont = {1, 2, 3};
  cont << 4;
  for (auto& p : cont) {
    std::cout << p << std::endl;
  }
}

TEST(HashSetTest, operator2) {
  HashSet<int> cont = {1, 2, 3};
  HashSet<int> cont1 = {4, 5, 6};

  cont |= cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }
}

TEST(HashSetTest, operator3) {
  HashSet<int> cont = {1, 2, 3};
  HashSet<int> cont1 = {4, 5, 6};

  cont |= std::move(cont1);

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator4) {
  HashSet<int> cont = {1, 2, 3};

  cont |= 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator5) {
  HashSet<int> cont = {1, 2, 3};
  HashSet<int> cont1 = {2, 3, 4};

  cont &= cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {2, 3};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator6) {
  HashSet<int> cont = {1, 2, 3};

  cont &= 3;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {3};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator7) {
  HashSet<int> cont = {1, 2, 3};

  cont += 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator8) {
  HashSet<int> cont = {1, 2, 3};

  cont += {4};

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator9) {
  HashSet<int> cont = {1, 2, 3, 4};

  cont -= {4};

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator10) {
  HashSet<int> cont = {1, 2, 3, 4};

  cont -= 4;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator11) {
  HashSet<int> cont = {1, 2, 3, 4};
  HashSet<int> cont1 = {5};

  cont = cont | cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4, 5};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator12) {
  HashSet<int> cont = {1, 2, 3, 4};
  HashSet<int> cont1 = {4};

  cont = cont & cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {4};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, operator13) {
  HashSet<int> cont = {1, 2, 3, 4};
  HashSet<int> cont1 = {5};

  cont = cont + cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4, 5};
  EXPECT_EQ(cont, result);
  EXPECT_TRUE(cont1.empty());
}

TEST(HashSetTest, operator14) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {5};

  cont = cont - cont1;

  for (auto& p : cont) {
    std::cout << p << std::endl;
  }

  HashSet<int> result = {1, 2, 3, 4};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, haskey) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  EXPECT_TRUE(cont.hasKey(4));
}

TEST(HashSetTest, issuhset) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {1, 2, 3};
  EXPECT_TRUE(cont.isSubset(&cont1));
}

TEST(HashSetTest, insertset) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {1, 2, 3, 6};
  cont.insertSet(&cont1);
  HashSet<int> result = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(cont, result);
}

TEST(HashSetTest, intersects) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {1, 2, 3, 6};
  EXPECT_TRUE(HashSet<int>::intersects(&cont, &cont1));
}

TEST(HashSetTest, nonmember1) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {1, 2, 3};

  EXPECT_FALSE(cont == cont1);
}

TEST(HashSetTest, nonmember2) {
  HashSet<int> cont = {1, 2, 3, 4, 5};
  HashSet<int> cont1 = {1, 2, 3, 6};

  EXPECT_TRUE(cont != cont1);
}

TEST(HashSetTest, nonmember7) {
  HashSet<int> cont = {1, 2, 3, 5, 7};
  HashSet<int> cont1 = {1, 2, 3, 5, 6};

  swap(cont, cont1);

  HashSet<int> result_cont = {1, 2, 3, 5, 6};

  EXPECT_EQ(cont, result_cont);
}

TEST(HashMultisetTest, ctor) {
  HashMultiset<int> hmultiset = {1, 2, 2, 3};
  for (auto item : hmultiset) {
    std::cout << item << std::endl;
  }
}

TEST(HashMultisetTest, capacity) {
  HashMultiset<int> hmultiset;

  EXPECT_TRUE(hmultiset.empty());

  std::cout << "max size : " << hmultiset.max_size() << std::endl;
}

TEST(HashMultisetTest, modifier1) {
  HashMultiset<std::unique_ptr<int, std::function<void(int*)>>> hmultiset;
  auto deleter = [](int* p) {
    std::cout << "delete " << *p << std::endl;
    delete p;
  };

  hmultiset.emplace(new int(2), deleter);
  hmultiset.emplace_hint(hmultiset.begin(), new int(1), deleter);

  std::cout << "remove one element" << std::endl;
  hmultiset.erase(++hmultiset.begin());
  std::cout << "add one element" << std::endl;
  hmultiset.emplace(new int(3), deleter);
}

TEST(HashMultisetTest, swap) {
  HashMultiset<int> hmultiset1 = {1, 2, 3};
  HashMultiset<int> hmultiset2 = {2, 3, 4};

  hmultiset1.swap(hmultiset2);

  for (auto& p : hmultiset1) {
    std::cout << p << std::endl;
  }

  HashMultiset<int> result = {2, 3, 4};
  EXPECT_EQ(hmultiset1, result);
}

TEST(HashMultisetTest, lookup) {
  HashMultiset<int> hmultiset1 = {1, 1, 3};

  auto range = hmultiset1.equal_range(1);
  for (auto p = range.first; p != range.second; p++) {
    std::cout << *p << std::endl;
  }
}

TEST(HashMultisetTest, nonmember1) {
  HashMultiset<int> hmultiset1 = {1, 2, 3};
  HashMultiset<int> hmultiset2 = {2, 3, 4};

  EXPECT_FALSE(hmultiset1 == hmultiset2);
}

TEST(HashMultisetTest, nonmember2) {
  HashMultiset<int> hmultiset1 = {1, 2, 3};
  HashMultiset<int> hmultiset2 = {2, 3, 4};

  EXPECT_TRUE(hmultiset1 != hmultiset2);
}

TEST(HashMultisetTest, nonmember7) {
  HashMultiset<int> hmultiset1 = {1, 2, 3};
  HashMultiset<int> hmultiset2 = {2, 3, 4};

  swap(hmultiset1, hmultiset2);

  HashMultiset<int> result = {2, 3, 4};

  EXPECT_EQ(hmultiset1, result);
}

class Dew {
 public:
  int _a;
  int _b;
  int _c;

  Dew(int a, int b, int c) : _a(a), _b(b), _c(c) {}

  bool operator<(const Dew& other) const {
    if (_a < other._a) return true;
    if (_a == other._a && _b < other._b) return true;
    return (_a == other._a && _b == other._b && _c < other._c);
  }
};

struct DewHash {
  size_t operator()(const Dew& rhs) const {
    return std::hash<int>()(rhs._a) ^ std::hash<int>()(rhs._b) ^
           std::hash<int>()(rhs._c);
  }
};

struct DewCmp {
  bool operator()(const Dew& lhs, const Dew& rhs) const {
    return lhs._a == rhs._a && lhs._b == rhs._b && lhs._b == rhs._b;
  }
};

struct DewGHash {
  size_t operator()(const Dew& rhs) const {
    return HashSet<int>::hash()(rhs._a) ^ HashSet<int>::hash()(rhs._b) ^
           HashSet<int>::hash()(rhs._c);
  }
};

struct DewGCmp {
  bool operator()(const Dew& lhs, const Dew& rhs) const {
    return lhs._a == rhs._a && lhs._b == rhs._b && lhs._b == rhs._b;
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

TEST(HashSetTest, perf1) {
  const int nof_operations = 10;

  auto set_insert = [=]() -> int {
    HashSet<Dew, DewGHash, DewGCmp> set;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

    return set.size();
  };

  auto stl_set_insert = [=]() -> int {
    std::unordered_set<Dew, DewHash, DewCmp> set;
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

TEST(HashSetTest, perf2) {
  const int nof_operations = 10;

  HashSet<Dew, DewGHash, DewGCmp> set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

  std::unordered_set<Dew, DewHash, DewCmp> stl_set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) stl_set.insert(Dew(i, j, k));

  auto set_find = [&]() -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) set.find(Dew(i, j, k));

    return 1;
  };

  auto stl_set_find = [&]() -> int {
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

TEST(HashSetTest, perf3) {
  const int nof_operations = 10;

  HashSet<Dew, DewGHash, DewGCmp> set;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k) set.insert(Dew(i, j, k));

  std::unordered_set<Dew, DewHash, DewCmp> stl_set;
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