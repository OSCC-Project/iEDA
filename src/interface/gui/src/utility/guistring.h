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


#ifndef GUI_STRING
#define GUI_STRING

#include <string>

class GuiString {
 public:
  GuiString()  = default;
  ~GuiString() = default;

  /// function
  bool isNumeral(std::string& str) {
    std::string::iterator it;
    for (it = str.begin(); it < str.end(); ++it) {
      if ((*it < '0' || *it > '9') && (*it != '.')) {
        return false;
      }
    }
    return true;
  }

  bool isCoordinateInDef(std::string str) {
    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);

    if (str.empty()) {
      return false;
    }

    std::string::iterator it;
    for (it = str.begin(); it < str.end(); ++it) {
      if ((*it < '0' || *it > '9') && (*it != ' ')) {
        return false;
      }
    }
    return true;
  }

  std::pair<int, int> getCoordinateInDef(std::string str) {
    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);

    if (str.empty()) {
      return std::make_pair<int, int>(0, 0);
    }

    int pos                = str.find_first_of(" ", 0);
    std::string str_first  = str.substr(0, pos);
    std::string str_second = str.erase(0, pos);

    return std::make_pair<int, int>(atoi(str_first.c_str()), atoi(str_second.c_str()));
  }

  bool isCoordinateInGui(std::string str) {
    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);

    if (str.empty()) {
      return false;
    }

    std::string::iterator it;
    for (it = str.begin(); it < str.end(); ++it) {
      if ((*it < '0' || *it > '9') && *it != '.' && *it != ' ') {
        return false;
      }
    }
    return true;
  }

  std::pair<qreal, qreal> getCoordinateInGui(std::string str) {
    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);

    if (str.empty()) {
      return std::make_pair<qreal, qreal>(0, 0);
    }

    int pos                = str.find_first_of(" ", 0);
    std::string str_first  = str.substr(0, pos);
    std::string str_second = str.erase(0, pos);

    return std::make_pair<qreal, qreal>(atof(str_first.c_str()), atof(str_second.c_str()));
  }
};

#endif  // GUI_STRING
