#ifndef SRC_EVALUATOR_SOURCE_MANAGER_HPP_
#define SRC_EVALUATOR_SOURCE_MANAGER_HPP_

#include "Config.hpp"
#include "CongestionEval.hpp"
#include "GDSWrapper.hpp"
#include "WirelengthEval.hpp"

namespace eval {

class Manager
{
 public:
  static Manager& initInst(Config* config);
  static Manager& getInst();
  static void destroyInst();

  WirelengthEval* getWirelengthEval() { return _wirelength_eval; }
  CongestionEval* getCongestionEval() { return _congestion_eval; }
  GDSWrapper* getGDSWrapper() { return _gds_wrapper; }

 private:
  static Manager* _mg_instance;
  WirelengthEval* _wirelength_eval = nullptr;
  CongestionEval* _congestion_eval = nullptr;
  GDSWrapper* _gds_wrapper = nullptr;

  explicit Manager(Config* config);
  Manager(const Manager& other) = delete;
  Manager(Manager&& other) = delete;
  ~Manager() = default;
  Manager& operator=(const Manager& other) = delete;
  Manager& operator=(Manager&& other) = delete;
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_MANAGER_HPP_
