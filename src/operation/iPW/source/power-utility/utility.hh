#pragma once

#include <deque>
#include <string>

inline std::deque<std::string> split_scope(const std::string& target_scope) {
  auto scope = target_scope;
  scope.erase(std::remove(scope.begin(), scope.end(), '\\'), scope.end());
  // split scope by delimiter "/"
  if (scope.front() == '/') {
    scope.erase(0, 1);
  }
  std::deque<std::string> result;
  std::string temp = "";
  for (size_t i = 0; i < scope.size(); ++i) {
    if (scope[i] == '/') {
      result.push_back(temp);
      temp = "";
    } else {
      temp += scope[i];
    }
  }
  result.push_back(temp);
  return result;
}