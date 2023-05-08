/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 21:31:13
 * @LastEditTime: 2022-03-10 16:00:42
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/NesterovPlace/database/NesPin.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_NESTEROV_PLACE_DATABASE_NESPIN_H
#define IPL_OPERATOR_NESTEROV_PLACE_DATABASE_NESPIN_H

namespace ipl {

class NesInstance;
class NesNet;

class NesPin
{
 public:
  NesPin() = delete;
  explicit NesPin(std::string name);
  NesPin(const NesPin&) = delete;
  NesPin(NesPin&&)      = delete;
  ~NesPin()             = default;

  NesPin& operator=(const NesPin&) = delete;
  NesPin& operator=(NesPin&&) = delete;

  // getter.
  std::string  get_name() const { return _name; }
  NesInstance* get_nInstance() const { return _nInstance; }
  NesNet*      get_nNet() const { return _nNet; }

  Point<int32_t> get_offset_coordi() const { return _offset_coordi; }
  Point<int32_t> get_center_coordi() const { return _center_coordi; }

  // setter.
  void set_nInstance(NesInstance* nInst) { _nInstance = nInst; }
  void set_nNet(NesNet* nNet) { _nNet = nNet; }

  void set_offset_coordi(Point<int32_t> coordi) { _offset_coordi = std::move(coordi); }
  void set_center_coordi(Point<int32_t> coordi) { _center_coordi = std::move(coordi); }

 private:
  std::string _name;

  NesInstance* _nInstance;
  NesNet*      _nNet;

  Point<int32_t> _offset_coordi;
  Point<int32_t> _center_coordi;
};
inline NesPin::NesPin(std::string name) : _name(name), _nInstance(nullptr), _nNet(nullptr)
{
}

}  // namespace ipl

#endif