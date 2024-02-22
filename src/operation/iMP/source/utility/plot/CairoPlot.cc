#include "CairoPlot.hh"

#include <cairo.h>
#include <cairo-pdf.h>
namespace imp {

void CairoPlot::add_rectangle(double lx, double ly, double width, double height, CairoColor fill_color, CairoColor edge_color)
{
  _rect_list.push_back({lx, ly, width, height, fill_color, edge_color});
}

void CairoPlot::save_as_pdf(const std::string& filename)
{
  cairo_surface_t* surface = cairo_pdf_surface_create(filename.c_str(),_canvas_width, _canvas_height);
  cairo_t* cr = cairo_create(surface);
  cairo_scale(cr, 1, -1);
  cairo_translate(cr, 0, -_canvas_height);

  cairo_set_source_rgb(cr, 1, 1, 1);  
  cairo_paint(cr);

  save_rectangles(cr);

  cairo_surface_flush(surface);
  cairo_surface_finish(surface);

  cairo_destroy(cr);
  cairo_surface_destroy(surface);
}

void CairoPlot::save_as_png(const std::string& filename)
{
  cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, _canvas_width, _canvas_height);
  cairo_t* cr = cairo_create(surface);
  cairo_scale(cr, 1, -1);
  cairo_translate(cr, 0, -_canvas_height);

  cairo_set_source_rgb(cr, 1, 1, 1); 
  cairo_paint(cr);

  save_rectangles(cr);

  cairo_surface_write_to_png(surface, filename.c_str());

  cairo_destroy(cr);
  cairo_surface_destroy(surface);
}
void CairoPlot::save_rectangles(cairo_t* cr)
{
  cairo_set_line_width(cr, 1.0);
  for (auto&& [lx, ly, width, height, fcolor, ecolor] : _rect_list) {
    auto&& [fr, fg, fb, fa] = fcolor;
    cairo_set_source_rgba(cr, fr, fg, fb, fa);
    cairo_rectangle(cr, lx, ly, width, height);
    cairo_fill(cr);
    auto&& [er, eg, eb, ea] = ecolor;
    cairo_set_source_rgba(cr, er, eg, eb, ea);
    cairo_rectangle(cr, lx, ly, width, height);
    cairo_stroke(cr);
  }
}
}  // namespace imp
