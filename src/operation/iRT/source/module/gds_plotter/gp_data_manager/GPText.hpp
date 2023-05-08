#pragma once

#include "GPTextPresentation.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class GPText
{
 public:
  GPText(irt_int layer_idx, irt_int text_type, GPTextPresentation presentation, PlanarCoord coord, std::string message)
      : _layer_idx(layer_idx), _text_type(text_type), _presentation(presentation), _coord(coord), _message(message)
  {
  }
  GPText() = default;
  ~GPText() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_text_type() const { return _text_type; }
  GPTextPresentation& get_presentation() { return _presentation; }
  PlanarCoord& get_coord() { return _coord; }
  std::string& get_message() { return _message; }

  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_text_type(const irt_int text_type) { _text_type = text_type; }
  void set_presentation(const GPTextPresentation& presentation) { _presentation = presentation; }
  void set_coord(const PlanarCoord& coord) { _coord = coord; }
  void set_coord(const irt_int x, const irt_int y) { _coord.set_coord(x, y); }
  void set_message(const std::string& message) { _message = message; }
  // function

 private:
  irt_int _layer_idx;
  irt_int _text_type = 0;
  GPTextPresentation _presentation = GPTextPresentation::kNone;
  PlanarCoord _coord;
  std::string _message;
};

}  // namespace irt
