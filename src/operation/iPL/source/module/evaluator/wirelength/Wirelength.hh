/*
 * @Author: S.J Chen
 * @Date: 2022-03-07 12:10:03
 * @LastEditTime: 2022-11-23 12:12:17
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/Wirelength.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_WIRELENGTH_H
#define IPL_EVALUATOR_WIRELENGTH_H

#include "TopologyManager.hh"

namespace ipl {

class Wirelength
{
 public:
  Wirelength() = delete;
  explicit Wirelength(TopologyManager* topology_manager);
  Wirelength(const Wirelength&) = delete;
  Wirelength(Wirelength&&) = delete;
  virtual ~Wirelength() = default;

  Wirelength& operator=(const Wirelength&) = delete;
  Wirelength& operator=(Wirelength&&) = delete;

  virtual int64_t obtainTotalWirelength() = 0;
  virtual int64_t obtainNetWirelength(std::string net_name) = 0;
  virtual int64_t obtainPartOfNetWirelength(std::string net_name, std::string sink_pin_name) = 0;

 protected:
  TopologyManager* _topology_manager;
};
inline Wirelength::Wirelength(TopologyManager* topology_manager) : _topology_manager(topology_manager)
{
}

}  // namespace ipl

#endif