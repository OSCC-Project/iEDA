/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 15:08:52
 * @LastEditTime: 2022-11-23 12:11:40
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/HPWirelength.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_HPWL_H
#define IPL_EVALUATOR_HPWL_H

#include "PlacerDB.hh"
#include "Wirelength.hh"
namespace ipl {

class HPWirelength : public Wirelength
{
 public:
  HPWirelength() = delete;
  explicit HPWirelength(TopologyManager* topology_manager);
  HPWirelength(const HPWirelength&) = delete;
  HPWirelength(HPWirelength&&)      = delete;
  ~HPWirelength() override                   = default;

  HPWirelength& operator=(const HPWirelength&) = delete;
  HPWirelength& operator=(HPWirelength&&) = delete;

  int64_t obtainTotalWirelength();
  int64_t obtainNetWirelength(std::string net_name);
  int64_t obtainPartOfNetWirelength(std::string net_name, std::string sink_pin_name);
};
inline HPWirelength::HPWirelength(TopologyManager* topology_manager) : Wirelength(topology_manager)
{
}

}  // namespace ipl

#endif