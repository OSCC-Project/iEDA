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
#ifndef IDB_ENCLOSURE_CHECK_H
#define IDB_ENCLOSURE_CHECK_H

namespace idb {
  class Enclosure {
   public:
    Enclosure() : _overhang1(0), _overhang2(0) { }
    Enclosure(int overHang1, int overHang2) : _overhang1(overHang1), _overhang2(overHang2) { }
    ~Enclosure() { }
    const Enclosure &operator=(const Enclosure &other) {
      _overhang1 = other._overhang1;
      _overhang2 = other._overhang2;
      return *this;
    }
    const Enclosure &operator=(Enclosure &&other) {
      _overhang1 = other._overhang1;
      _overhang2 = other._overhang2;
      return *this;
    }
    int get_overhang1() { return _overhang1; }
    int get_overhang2() { return _overhang2; }

    void set_overhang1(int overhang) { _overhang1 = overhang; }
    void set_overhang2(int overhang) { _overhang2 = overhang; }
    void setOverhang(int overhang1, int overhang2) {
      _overhang1 = overhang1;
      _overhang2 = overhang2;
    }

   private:
    int _overhang1;
    int _overhang2;
  };

  class IdbEnclosureCheck {
   public:
    IdbEnclosureCheck() : _enclosure_below(), _enclosure_above() { }
    ~IdbEnclosureCheck() { }
    // getter
    const Enclosure &get_enclosure_above() const { return _enclosure_above; }
    const Enclosure &get_enclosure_below() const { return _enclosure_below; }
    // setter
    void set_enclosure_above(const Enclosure &in) { _enclosure_above = in; }
    void setEnclosureAbove(int overhang1, int overhang2) { _enclosure_above.setOverhang(overhang1, overhang2); }
    void set_enclosure_below(const Enclosure &in) { _enclosure_below = in; }
    void setEnclosureBelow(int overhang1, int overhang2) { _enclosure_below.setOverhang(overhang1, overhang2); }
    // other
    int getBelowOverhang1() { return _enclosure_below.get_overhang1(); }
    int getBelowOverhang2() { return _enclosure_below.get_overhang2(); }
    int getAboveOverhang1() { return _enclosure_above.get_overhang1(); }
    int getAboveOverhang2() { return _enclosure_above.get_overhang2(); }

   private:
    Enclosure _enclosure_below;
    Enclosure _enclosure_above;
  };
}  // namespace idb

#endif
