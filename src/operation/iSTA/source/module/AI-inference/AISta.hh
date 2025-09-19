/**
 * @file AISta.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The top layer for deploying the AI model used in sta.
 * @version 0.1
 * @date 2024-06-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <onnxruntime_cxx_api.h>

#include <map>
#include <string>
#include <vector>

#include "sta/StaPathData.hh"

namespace ista {

/**
 * @brief The class for deploying the AI model.
 *
 */
class AISta {
 public:
  virtual unsigned init() { LOG_FATAL << "The func is not implemented"; }

  virtual Ort::Value createInputTensor(StaSeqPathData* seq_path_data) {
    LOG_FATAL << "The func is not implemented";
  }
  virtual std::vector<Ort::Value> infer(Ort::Value& input_tensor) {
    LOG_FATAL << "The func is not implemented";
  }
  virtual std::vector<float> getOutputResult(
      std::vector<Ort::Value>& output_tensor) {
    LOG_FATAL << "The func is not implemented";
    return {};
  }
};

/**
 * @brief The class for deploy the calibrate path delay model.
 *
 */
class AICalibratePathDelay : public AISta {
 public:
  enum class AIModeType {
    kDefault,
    kSky130CalibratePathDelay,
    kT28CalibratePathDelay
  };

  AICalibratePathDelay(std::map<AIModeType, std::string>&& model_to_path,
                       AIModeType model_type,
                       std::string&& cell_list_path,
                       std::string&& pin_list_path)
      : _model_to_path(std::move(model_to_path)),
        _model_type(model_type),
        _cell_list_path(std::move(cell_list_path)),
        _pin_list_path(std::move(pin_list_path)) {}
  ~AICalibratePathDelay() = default;

  unsigned init() override;
  Ort::Value createInputTensor(StaSeqPathData* seq_path_data) override;
  std::vector<Ort::Value> infer(Ort::Value& input_tensor) override;
  std::vector<float> getOutputResult(
      std::vector<Ort::Value>& output_tensors) override;

 private:
  auto preprocessData(StaSeqPathData* seq_path_data);
  std::map<std::string, unsigned> tokenization(std::string file_path);

  std::map<AIModeType, std::string>
      _model_to_path;           //!< The ML model to be loaded.
  AIModeType _model_type; //!< The current model type.
  std::string _cell_list_path;  //!< The cell list for tokenization.
  std::string _pin_list_path;   //!< The pin list for tokenization.

  std::map<std::string, unsigned>
      _cell_to_id;  //!< The dictionary of cell name to id.
  std::map<std::string, unsigned>
      _pin_to_id;  //!< The dictionary of pin name to id.

  static constexpr unsigned MAX_SEQ_LEN = 128;
};

}  // namespace ista