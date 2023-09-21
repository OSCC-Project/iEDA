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
#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <utility>

#include "Map.hh"
#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"

using ieda::BTreeMap;
using ieda::Multimap;

using namespace testing;

namespace {
TEST(MapTest, initializer_list) {
  BTreeMap<int, const char *> bmap = {{1, "test"}};

  EXPECT_STREQ(bmap.value(1), "test");
}

TEST(MapTest, copy_constructor) {
  BTreeMap<int, const char *> bmap = {{1, "test"}};
  BTreeMap<int, const char *> bmap1(bmap);

  EXPECT_STREQ(bmap1.value(1), "test");
}

TEST(MapTest, assignmen_operator) {
  BTreeMap<int, const char *> bmap = {{1, "test"}};
  BTreeMap<int, const char *> bmap1 = bmap;

  EXPECT_STREQ(bmap1.value(1), "test");
}

TEST(MapTest, move_constructor) {
  BTreeMap<int, const char *> bmap = {{1, "test"}};
  BTreeMap<int, const char *> bmap1(std::move(bmap));

  EXPECT_STREQ(bmap1.value(1), "test");
}

TEST(MapTest, move_assignment) {
  BTreeMap<int, const char *> bmap = {{1, "test"}};
  BTreeMap<int, const char *> bmap1;
  bmap1 = std::move(bmap);

  EXPECT_STREQ(bmap1.value(1), "test");
}

TEST(MapTest, range_constructor) {
  std::vector<std::pair<int, const char *>> v = {{1, "a"}, {2, "b"}};
  BTreeMap<int, const char *> bmap1(v.begin(), v.end());

  EXPECT_STREQ(bmap1.value(1), "a");
}

TEST(MapTest, at) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  const char *value = bmap.at(1);

  EXPECT_STREQ(bmap.at(1), "a");
}

TEST(MapTest, at_exception) {
  BTreeMap<int, const char *> bmap;
  ASSERT_THROW(bmap.at(1), std::out_of_range);
}

TEST(MapTest, begin) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  EXPECT_STREQ(bmap.begin()->second, "a");
}

TEST(MapTest, cbegin) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  EXPECT_STREQ(bmap.cbegin()->second, "a");
}

TEST(MapTest, end) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  EXPECT_TRUE(bmap.find(3) == bmap.end());
}

TEST(MapTest, cend) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  EXPECT_TRUE(bmap.find(3) == bmap.cend());
}

TEST(MapTest, rbegin) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  EXPECT_STREQ(bmap.rbegin()->second, "b");
}

TEST(MapTest, rend) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {2, "b"}};
  for (auto p = bmap.rbegin(); p != bmap.rend(); p++) {
    std::cout << p->first << " " << p->second << std::endl;
  }
}

TEST(MapTest, crend) {
  BTreeMap<int, const char *> bmap{{1, "a"}, {1, "b"}};
  for (auto p = bmap.crbegin(); p != bmap.crend(); p++) {
    std::cout << p->first << " " << p->second << std::endl;
  }
}

TEST(MapTest, emplace) {
  // emplace would construct the object in place.
  BTreeMap<std::string, std::string> m;

  // uses pair's move constructor
  m.emplace(std::make_pair(std::string("a"), std::string("a")));

  // uses pair's converting move constructor
  m.emplace(std::make_pair("b", "abcd"));

  // uses pair's template constructor
  m.emplace("d", "ddd");

  // uses pair's piecewise constructor
  m.emplace(std::piecewise_construct, std::forward_as_tuple("c"),
            std::forward_as_tuple(10, 'c'));
  // as of C++17, m.try_emplace("c", 10, 'c'); can be used

  for (const auto &p : m) {
    std::cout << p.first << " => " << p.second << '\n';
  }
}

TEST(MapTest, emplace_hint) {
  // emplace_hint would construct the object in place and insert the object
  // before the hint.
  BTreeMap<std::string, std::string> m;
  auto hint = m.begin();

  // uses pair's move constructor
  m.emplace_hint(hint, std::make_pair(std::string("a"), std::string("a")));

  // uses pair's converting move constructor
  m.emplace_hint(hint, std::make_pair("b", "abcd"));

  // uses pair's template constructor
  m.emplace_hint(hint, "d", "ddd");

  // uses pair's piecewise constructor
  m.emplace_hint(hint, std::piecewise_construct, std::forward_as_tuple("c"),
                 std::forward_as_tuple(10, 'c'));
  // as of C++17, m.try_emplace("c", 10, 'c'); can be used

  for (const auto &p : m) {
    std::cout << p.first << " => " << p.second << '\n';
  }
}

TEST(MapTest, empty) {
  BTreeMap<int, std::string> m;

  // auto IsEmpty = [](BTreeMap<std::string, std::string> &m) { return m.empty(); };

  EXPECT_THAT(m, IsEmpty());
  // EXPECT_TRUE(m.empty());
}

TEST(MapTest, erase) {
  BTreeMap<int, std::string> c = {{1, "one"},  {2, "two"},  {3, "three"},
                             {4, "four"}, {5, "five"}, {6, "six"}};

  // erase all odd numbers from c
  for (auto it = c.begin(); it != c.end();) {
    if (it->first % 2 == 1)
      it = c.erase(it);
    else
      ++it;
  }

  for (auto &p : c) {
    std::cout << p.first << "=>" << p.second << std::endl;
  }

  // use range erase
  c.erase(c.begin(), --c.end());

  for (auto &p : c) {
    std::cout << p.first << "=>" << p.second << std::endl;
  }

  // use erase key
  c.erase(6);

  EXPECT_TRUE(c.empty());
}

TEST(MapTest, extract) {
  BTreeMap<int, char> cont{{1, 'a'}, {2, 'b'}, {3, 'c'}};

  auto print = [](std::pair<const int, char> &n) {
    std::cout << " " << n.first << '(' << n.second << ')';
  };

  std::cout << "Start:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Extract node handle and change key
  auto nh = cont.extract(1);

  std::cout << "After extract and before insert:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';

  // Insert node handle back
  cont.insert(move(nh));
  cont.extract(cont.begin());

  std::cout << "End:";
  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';
}

TEST(MapTest, insert) {
  auto print = [](std::pair<const int, char> &n) {
    std::cout << " " << n.first << '(' << n.second << ')';
  };

  BTreeMap<int, char> cont{{1, 'a'}, {2, 'b'}, {3, 'c'}};

  // value type insert
  auto result = cont.insert(std::make_pair(4, 'd'));
  EXPECT_TRUE(result.second);

  // hint insert
  cont.insert(cont.end(), std::make_pair(5, 'e'));
  EXPECT_EQ(cont[5], 'e');

  BTreeMap<int, char> cont1{{6, 'f'}};
  // range insert
  cont.insert(cont1.begin(), cont1.end());

  // initializer list insert
  cont.insert({7, 'g'});

  // node type insert
  auto nh = cont.extract(5);
  cont1.insert(move(nh));

  // hint node type insert
  nh = cont.extract(7);
  cont1.insert(cont.begin(), move(nh));

  std::for_each(cont.begin(), cont.end(), print);
  std::cout << '\n';
  std::for_each(cont1.begin(), cont1.end(), print);
}

TEST(MapTest, insert_or_assign) {
  BTreeMap<std::string, std::string> myMap;
  myMap.insert_or_assign("a", "apple");
  myMap.insert_or_assign("b", "bannana");
  myMap.insert_or_assign("c", "cherry");
  myMap.insert_or_assign("c", "clementine");

  myMap.insert_or_assign(myMap.begin(), "d", "duck");

  for (const auto &pair : myMap) {
    std::cout << pair.first << " : " << pair.second << '\n';
  }
}

TEST(MapTest, merge) {
  BTreeMap<int, std::string> ma{{1, "apple"}, {5, "pear"}, {10, "banana"}};
  BTreeMap<int, std::string> mb{
      {2, "zorro"}, {4, "batman"}, {5, "X"}, {8, "alpaca"}};
  BTreeMap<int, std::string> u;
  u.merge(ma);
  std::cout << "ma.size(): " << ma.size() << '\n';
  u.merge(mb);
  std::cout << "mb.size(): " << mb.size() << '\n';
  std::cout << "mb.at(5): " << mb.at(5) << '\n';
  for (auto const kv : u) std::cout << kv.first << ", " << kv.second << '\n';

  std::cout << std::begin(u)->first;

  auto const &kv1 = std::begin(u);

  std::cout << kv1->first << '\n';
}

// print out a std::pair
template <class Os, class U, class V>
Os &operator<<(Os &os, const std::pair<U, V> &p) {
  return os << p.first << ":" << p.second;
}

// print out a container
template <class Os, class Co>
Os &operator<<(Os &os, const Co &co) {
  os << "{";
  for (auto const &i : co) {
    os << ' ' << i;
  }
  return os << " }\n";
}
TEST(MapTest, swap) {
  BTreeMap<std::string, std::string> m1{
      {"γ", "gamma"},
      {"β", "beta"},
      {"α", "alpha"},
      {"γ", "gamma"},
  },
      m2{{"ε", "epsilon"}, {"δ", "delta"}, {"ε", "epsilon"}};

  const auto &ref = *(m1.begin());
  const auto iter = std::next(m1.cbegin());

  std::cout << "──────── before swap ────────\n"
            << "m1: " << m1 << "m2: " << m2 << "ref: " << ref
            << "\niter: " << *iter << '\n';

  m1.swap(m2);

  std::cout << "──────── after swap ────────\n"
            << "m1: " << m1 << "m2: " << m2 << "ref: " << ref
            << "\niter: " << *iter << '\n';
}

TEST(MapTest, try_emplace) {
  BTreeMap<const char *, std::string> m;

  m.try_emplace("a", "a");
  m.try_emplace("b", "abcd");
  m.try_emplace("c", 10, 'c');
  m.try_emplace("c", "Won't be inserted");

  for (const auto &p : m) {
    std::cout << p.first << " => " << p.second << '\n';
  }
}

TEST(MapTest, contains) {
  BTreeMap<int, std::string> m{{1, "a"}};
  EXPECT_TRUE(m.contains(1));
}

TEST(MapTest, count) {
  BTreeMap<int, std::string> m{{1, "a"}, {1, "b"}};
  for (const auto &p : m) {
    std::cout << p.first << " => " << p.second << '\n';
  }
  EXPECT_EQ(m.count(1), 1);
}

TEST(MapTest, equal_range) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  {
    auto p = m.equal_range(1);

    EXPECT_EQ(p.first, m.find(1));
    EXPECT_EQ(p.second, m.find(3));
  }

  {
    auto pp = m.equal_range(-1);

    EXPECT_EQ(pp.first, m.begin());
    EXPECT_EQ(pp.second, m.begin());
  }

  {
    auto ppp = m.equal_range(2);

    EXPECT_EQ(ppp.first, m.find(3));
    EXPECT_EQ(ppp.second, m.find(3));
  }

  {
    auto ppp = m.equal_range(4);

    EXPECT_EQ(ppp.first, m.end());
    EXPECT_EQ(ppp.second, m.end());
  }
}

TEST(MapTest, find) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  EXPECT_EQ(m.find(1), m.equal_range(1).first);
  EXPECT_EQ(m.find(2), m.end());
}

TEST(MapTest, bound) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  EXPECT_EQ(m.equal_range(2).first, m.lower_bound(2));
  EXPECT_EQ(m.equal_range(2).second, m.upper_bound(2));
}

TEST(MapTest, key_comp) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  auto cmp = m.key_comp();
}

TEST(MapTest, value_comp) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  auto cmp = m.value_comp();
}

TEST(MapTest, keys) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  std::list<int> keys = {0, 1, 3};

  EXPECT_EQ(m.keys(), keys);
}

TEST(MapTest, values) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  std::list<const char *> values = {"zero", "one", "three"};

  EXPECT_EQ(m.values(), values);
}

TEST(MapTest, haskey) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  EXPECT_TRUE(m.hasKey(1));
  EXPECT_FALSE(m.hasKey(4));
}

TEST(MapTest, value) {
  const BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  EXPECT_STREQ(m.value(1, ""), "one");
  EXPECT_STREQ(m.value(4, ""), "");
}

// interface refer to the qt interface.
TEST(MapTest, insert_qt) {
  BTreeMap<int, const char *> m{
      {0, "zero"},
      {1, "one"},
      {3, "three"},
  };

  m.insert(4, "four");

  EXPECT_STREQ(m.value(4, ""), "four");
}

TEST(MapTest, clear) {
  BTreeMap<int, char> container{{1, 'x'}, {2, 'y'}, {3, 'z'}};

  container.clear();

  EXPECT_EQ(container.size(), 0);
  EXPECT_TRUE(container.empty());
}

TEST(MapTest, max_size) {
  BTreeMap<int, char> container{{1, 'x'}, {2, 'y'}, {3, 'z'}};
  BTreeMap<int, char>::size_type max_size = container.max_size();
}

TEST(MapTest, Iterator) {
  BTreeMap<int, char> container{{1, 'x'}, {2, 'y'}, {3, 'z'}};
  BTreeMap<int, char>::Iterator iter(&container);
  while (iter.hasNext()) {
    std::cout << iter.value() << std::endl;
    iter = iter.next();
  }
}

TEST(MapTest, ConstIterator) {
  BTreeMap<int, char> container{{1, 'x'}, {2, 'y'}, {3, 'z'}};
  BTreeMap<int, char>::ConstIterator iter(&container);
  while (iter.hasNext()) {
    int key;
    char value;

    iter.next(&key, &value);
    std::cout << key << "=>" << value << std::endl;
  }
}

TEST(MapTest, NoMemberOperator1) {
  BTreeMap<int, std::string> container1{{1, "x"}, {2, "y"}, {3, "z"}};
  BTreeMap<int, std::string> container2{{1, "x"}, {2, "y"}, {3, "z"}};

  EXPECT_TRUE(container1 == container2);
}

TEST(MapTest, NoMemberOperator2) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {2, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {2, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 < container2);
}

TEST(MapTest, NoMemberOperator3) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {2, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {2, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 != container2);
}

TEST(MapTest, NoMemberOperator4) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {2, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {2, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 <= container2);
}

TEST(MapTest, NoMemberOperator5) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {2, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {2, "y2"}, {3, "z2"}};

  EXPECT_FALSE(container1 >= container2);
}

TEST(MapTest, NoMemberOperator6) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {2, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {2, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 < container2);
  swap(container1, container2);
  EXPECT_FALSE(container1 < container2);
}

TEST(MultimapTest, initializer_list) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};

  EXPECT_STREQ(bmap.values(1).front(), "test");
  EXPECT_STREQ(bmap.values(1).back(), "test1");
}

TEST(MultimapTest, copy_constructor) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};
  Multimap<int, const char *> bmap1(bmap);

  EXPECT_STREQ(bmap1.values(1).front(), "test");
  EXPECT_STREQ(bmap1.values(1).back(), "test1");
}

TEST(MultimapTest, assignmen_operator) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};
  Multimap<int, const char *> bmap1 = bmap;

  EXPECT_STREQ(bmap1.values(1).front(), "test");
  EXPECT_STREQ(bmap1.values(1).back(), "test1");
}

TEST(MultimapTest, move_constructor) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};
  Multimap<int, const char *> bmap1(std::move(bmap));

  EXPECT_STREQ(bmap1.values(1).front(), "test");
  EXPECT_STREQ(bmap1.values(1).back(), "test1");
}

TEST(MultimapTest, move_assignment) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};
  Multimap<int, const char *> bmap1;
  bmap1 = std::move(bmap);

  EXPECT_STREQ(bmap1.values(1).front(), "test");
  EXPECT_STREQ(bmap1.values(1).back(), "test1");
}

TEST(MultimapTest, range_constructor) {
  std::vector<std::pair<int, const char *>> v = {{1, "a"}, {1, "b"}};
  Multimap<int, const char *> bmap1(v.begin(), v.end());

  EXPECT_STREQ(bmap1.values(1).front(), "a");
  EXPECT_STREQ(bmap1.values(1).back(), "b");
}

TEST(MultimapTest, count) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};

  EXPECT_EQ(bmap.count(1), 2);
}

TEST(MultimapTest, begin) {
  Multimap<int, const char *> bmap = {{1, "test"}, {1, "test1"}};

  for (auto p : bmap) {
    std::cout << p.second << std::endl;
  }

  Multimap<int, const char *>::const_reverse_iterator it = bmap.crbegin();
  for (; it != bmap.crend(); it++) {
    std::cout << it->second << std::endl;
  }
}

TEST(MultimapTest, Iterator) {
  Multimap<const char *, int> bmap = {{"test", 1}, {"test", 2}};
  Multimap<const char *, int>::Iterator p(&bmap);

  while (p.hasNext()) {
    std::cout << p.key() << "=>" << p.value() << std::endl;
    p = p.next();
  }

  p.init(&bmap);
  Multimap<const char *, int>::Iterator q = p;
  while (q.hasNext()) {
    const char *key;
    int value;
    q.next(&key, &value);
    std::cout << key << "=>" << value << std::endl;
  }

  Multimap<const char *, int>::ConstIterator q1(&bmap);
  while (q1.hasNext()) {
    const char *key;
    int value;
    q1.next(&key, &value);
    std::cout << key << "=>" << value << std::endl;
  }
}

TEST(MultimapTest, NoMemberOperator1) {
  BTreeMap<int, std::string> container1{{1, "x"}, {1, "y"}, {3, "z"}};
  BTreeMap<int, std::string> container2{{1, "x"}, {1, "y"}, {3, "z"}};

  EXPECT_TRUE(container1 == container2);
}

TEST(MultimapTest, NoMemberOperator2) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {1, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 < container2);
}

TEST(MultimapTest, NoMemberOperator3) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {1, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 != container2);
}

TEST(MultimapTest, NoMemberOperator4) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {1, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 <= container2);
}

TEST(MultimapTest, NoMemberOperator5) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {1, "y2"}, {3, "z2"}};

  EXPECT_FALSE(container1 >= container2);
}

TEST(MultimapTest, NoMemberOperator6) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x2"}, {1, "y2"}, {3, "z2"}};

  EXPECT_TRUE(container1 < container2);
  swap(container1, container2);
  EXPECT_FALSE(container1 < container2);
}

TEST(MultimapTest, swap) {
  BTreeMap<int, std::string> container1{{1, "x1"}, {1, "y1"}, {3, "z1"}};
  BTreeMap<int, std::string> container2{{1, "x1"}, {1, "y1"}, {3, "z1"}};

  EXPECT_FALSE(container1 < container2);
  swap(container1, container2);
  EXPECT_FALSE(container1 < container2);
}

class Dew {
 private:
  int _a;
  int _b;
  int _c;

 public:
  Dew(int a, int b, int c) : _a(a), _b(b), _c(c) {}

  Dew() = default;
  Dew(Dew &&other) = default;
  Dew(const Dew &other) = default;

  Dew &operator=(const Dew &other) = default;
  Dew &operator=(Dew &&other) = default;

  bool operator<(const Dew &other) const {
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

TEST(MapTest, perf1) {
  const int nof_operations = 100;

  auto map_emplace = [=]() -> int {
    BTreeMap<Dew, Dew> map;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k)
          map.emplace(std::piecewise_construct, std::forward_as_tuple(i, j, k),
                      std::forward_as_tuple(i, j, k));

    return map.size();
  };

  auto stl_map_emplace = [=]() -> int {
    std::map<Dew, Dew> map;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k)
          map.emplace(std::piecewise_construct, std::forward_as_tuple(i, j, k),
                      std::forward_as_tuple(i, j, k));

    return map.size();
  };

  timeit(stl_map_emplace, "stl emplace");
  timeit(map_emplace, "emplace");
  timeit(stl_map_emplace, "stl emplace");
  timeit(map_emplace, "emplace");
}

TEST(MapTest, perf2) {
  const int nof_operations = 100;

  auto map_insert = [=]() -> int {
    BTreeMap<Dew, Dew> map;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k)
          map.insert(Dew(i, j, k), Dew(i, j, k));

    return map.size();
  };

  auto stl_map_insert = [=]() -> int {
    std::map<Dew, Dew> map;
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k)
          map[Dew(i, j, k)] = Dew(i, j, k);

    return map.size();
  };

  timeit(stl_map_insert, "stl insert");
  timeit(map_insert, "insert");
  timeit(stl_map_insert, "stl insert");
  timeit(map_insert, "insert");
}

TEST(MapTest, perf3) {
  const int nof_operations = 100;

  BTreeMap<Dew, Dew> map;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k)
        map.insert(Dew(i, j, k), Dew(i, j, k));

  std::map<Dew, Dew> stl_map;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k)
        stl_map[Dew(i, j, k)] = Dew(i, j, k);

  auto map_find = [=, &map]() -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) map.find(Dew(i, j, k));

    return 1;
  };

  auto stl_map_find = [=, &stl_map]() -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) stl_map.find(Dew(i, j, k));
    return 1;
  };

  timeit(stl_map_find, "stl find");
  timeit(map_find, "find");
  timeit(stl_map_find, "stl find");
  timeit(map_find, "find");
}

TEST(MapTest, perf4) {
  const int nof_operations = 100;

  BTreeMap<Dew, Dew> map;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k)
        map.insert(Dew(i, j, k), Dew(i, j, k));

  std::map<Dew, Dew> stl_map;
  for (int i = 0; i < nof_operations; ++i)
    for (int j = 0; j < nof_operations; ++j)
      for (int k = 0; k < nof_operations; ++k)
        stl_map.insert(std::make_pair(Dew(i, j, k), Dew(i, j, k)));

  auto set_erase = [=]() mutable -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) map.erase(Dew(i, j, k));

    return 1;
  };

  auto stl_set_erase = [=]() mutable -> int {
    for (int i = 0; i < nof_operations; ++i)
      for (int j = 0; j < nof_operations; ++j)
        for (int k = 0; k < nof_operations; ++k) stl_map.erase(Dew(i, j, k));
    return 1;
  };

  timeit(stl_set_erase, "stl erase");
  timeit(set_erase, "erase");
  timeit(stl_set_erase, "stl erase");
  timeit(set_erase, "erase");
}

}  // namespace
;
}

}  // namespace
