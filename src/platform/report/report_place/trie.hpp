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

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
namespace iplf {
/**
 * @brief  split string by "\/" or "/"
 * @param str
 * @return std::vector<std::string>
 */
std::vector<std::string> split(const std::string& str)
{
  std::vector<std::string> vec;
  std::string s;
  for (size_t i = 0; i < str.size(); ++i) {
    switch (str[i]) {
      case '\\':
        continue;
        // fallthrough
      case '/': {
        std::string tmp;
        std::swap(tmp, s);
        vec.push_back(std::move(tmp));
        break;
      }
      default:
        s.push_back(str[i]);
    }
  }
  if (!s.empty()) {
    vec.push_back(std::move(s));
  }
  return vec;
}

template <typename T>
struct Node
{
  int cnt{0};
  std::map<std::string, Node*> next;
  bool ok{false};
  T val;
};
template <typename T>
struct Trie
{
  using Nodep = Node<T>*;
  Node<T>* root;
  Trie() : root(new Node<T>) {}
  Trie(const Trie&) = delete;
  Trie(Trie&& rhs) noexcept
  {
    this->root = rhs.root;
    rhs.root = nullptr;
  };
  Trie& operator=(Trie&) = delete;
  Trie& operator=(Trie&& rhs) noexcept
  {
    std::swap(this->root, rhs.root);
    return *this;
  };

  void insert(const std::string& key, T val)
  {
    auto splited_key = split(key);
    Nodep p = root;
    for (auto& ikey : splited_key) {
      p->cnt++;
      auto it = p->next.find(ikey);
      if (it == p->next.end()) {
        Nodep next = new Node<T>;
        p->next[ikey] = next;
        p = next;
      } else {
        p = it->second;
      }
    }
    p->cnt++;
    p->val = val;
    p->ok = true;
  }

  int getCount(const std::string& prefix)
  {
    auto splited_key = split(prefix);
    Nodep p = findPrefix(splited_key);
    return p ? p->cnt : 0;
  }
  // find a node with prefix;
  Nodep findPrefix(const std::vector<std::string>& splited_keys)
  {
    Nodep p = root;
    for (auto& ikey : splited_keys) {
      auto it = p->next.find(ikey);
      if (it == p->next.end()) {
        return nullptr;
      }
      p = it->second;
    }
    return p;
  }

  /**
   * @brief find next level prefix with input prefix and level, with count limit cnt
   *    example :   raw data [a/b/c, a/b/d, a/a, b, b/c], nextLevel("a",1) will return {"a/b":2, "a/a":1}
   *
   * @param prefix
   * @param level
   * @param cnt
   * @return std::map<std::string, int>
   */
  std::map<std::string, int> nextLevel(const std::string& prefix, int level = 1, int cnt_threshold = 1)
  {
    auto splited_key = split(prefix);
    Nodep p = findPrefix(splited_key);
    if (p == nullptr) {
      return {};
    }
    std::map<std::string, int> ans;
    std::string name;
    for (auto x : splited_key) {
      if (!name.empty()) {
        name += "/";
      }
      name += x;
    }

    std::function<void(Nodep, std::string&, int)> dfs = [&dfs, &ans, cnt_threshold](Nodep root, std::string& name, int level) {
      if (!root || root->cnt < cnt_threshold || level < 0) {
        return;
      }

      if (level == 0) {
        ans[name] = root->cnt;
        return;
      }
      --level;
      for (auto& [k, v] : root->next) {
        auto sz = name.size();
        name += "/";
        name += k;
        dfs(v, name, level);
        name.resize(sz);
      }
    };
    dfs(p, name, level);
    return ans;
  }
  std::vector<T> findPrefixVals(const std::string& prefix)
  {
    auto splited_key = split(prefix);
    Nodep p = findPrefix(splited_key);
    if (p == nullptr) {
      return {};
    }
    std::vector<T> result;
    std::function<void(Nodep)> dfs = [&dfs, &result](Nodep root) {
      if (root->ok) {
        result.push_back(root->val);
      }
      for (auto& [_, v] : root->next) {
        dfs(v);
      }
    };
    dfs(p);
    return result;
  }

  void operateOnPrefix(const std::string& prefix, std::function<void(T)> func)
  {
    auto splited_key = split(prefix);
    Nodep p = findPrefix(splited_key);
    if (p == nullptr) {
      return;
    }
    std::function<void(Nodep)> dfs = [&dfs, &func](Nodep root) {
      assert(root);
      if (root->ok) {
        func(root->val);
      }
      for (auto& [_, next] : root->next) {
        dfs(next);
      }
    };
    dfs(p);
  }

  ~Trie()
  {
    std::function<void(Nodep)> dfs = [&dfs](Nodep root) {
      if (!root) {
        return;
      }
      for (auto& [_, node] : root->next) {
        dfs(node);
      }
      delete (root);
    };
    dfs(root);
  }
};

}  // namespace iplf