/**
 * @file PyPlot.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-31
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_PYPLOT_H
#define IMP_PYPLOT_H
#include <fstream>
#include <string>
#include <vector>
namespace imp {
const std::string python = PYTHON;
template <typename T>
struct PlotRect
{
  T lx{0};
  T ly{0};
  T dx{0};
  T dy{0};
  T angle{0};
  std::string edgecolor = "red";
  std::string facecolor = "none";
};
template <typename T>
struct PlotFlyLine
{
  T x[4];
  T y[4];
};

template <typename T>
class PyPlot
{
 public:
  PyPlot() {}
  ~PyPlot() {}
  void set_limitation(int64_t xlim, int64_t ylim)
  {
    _xlim = xlim;
    _ylim = ylim;
  }
  void addTitle(const std::string& str) { _title = str; }
  void addMacro(T lx, T ly, T dx, T dy, T angle = 0)
  {
    addRectangle(lx, ly, dx, dy, angle);
    _rects[_rects.size() - 1].edgecolor = "black";
    _rects[_rects.size() - 1].facecolor = "red";
  }
  void addCluster(T lx, T ly, T dx, T dy, T angle = 0)
  {
    addRectangle(lx, ly, dx, dy, angle);
    _rects[_rects.size() - 1].edgecolor = "black";
    _rects[_rects.size() - 1].facecolor = "blue";
  }
  void addRectangle(T lx, T ly, T dx, T dy, T angle = 0) { _rects.push_back({lx, ly, dx, dy, angle}); }
  void addFlyLine(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4)
  {
    PlotFlyLine<T> flyline;
    flyline.x[0] = x1;
    flyline.y[0] = y1;
    flyline.x[1] = x2;
    flyline.y[1] = y2;
    flyline.x[2] = x3;
    flyline.y[2] = y3;
    flyline.x[3] = x4;
    flyline.y[3] = y4;
    _flylines.push_back(flyline);
  }
  bool save(std::string filename);

 private:
  void saveRectangle(std::ofstream& py, const PlotRect<T>& rect);
  void saveFlyLine(std::ofstream& py, const PlotFlyLine<T>& flyline);
  int64_t _xlim{};
  int64_t _ylim{};
  std::string _title{};
  std::vector<PlotRect<T>> _rects{};
  std::vector<PlotFlyLine<T>> _flylines{};
};

template <typename T>
inline bool PyPlot<T>::save(std::string filename)
{
  std::string py_script = filename + ".py";
  std::ofstream py(py_script);
  if (!py) {
    return false;
  }

  py << "import matplotlib.pyplot as plt" << std::endl;
  py << "import matplotlib.style as mplstyle" << std::endl;
  py << "import matplotlib" << std::endl;
  py << "matplotlib.use('agg')" << std::endl;
  py << "mplstyle.use('fast')" << std::endl;
  py << "fig, ax = plt.subplots()" << std::endl;

  for (const auto& rect : _rects) {
    saveRectangle(py, rect);
  }

  for (const auto& flyline : _flylines) {
    saveFlyLine(py, flyline);
  }

  if (_xlim > 0 && _ylim > 0) {
    py << "ax.set_xlim(0," + std::to_string(_xlim) + ")" << std::endl;
    py << "ax.set_ylim(0," + std::to_string(_ylim) + ")" << std::endl;
  }

  if (!_title.empty())
    py << "ax.set(title='" + _title + "')" << std::endl;

  py << "ax.set_aspect('equal')" << std::endl;
  py << "plt.savefig('" + filename + "', dpi=100)";

  py.close();

  std::string command = python + " " + py_script;
  int result = 0;
#ifdef BUILD_PLOT
  result = system(command.c_str());
#endif

  if (result != 0) {
    return false;
  }

  result = std::remove(py_script.c_str());

  if (result != 0) {
    return false;
  }
  return true;
}

template <typename T>
inline void PyPlot<T>::saveRectangle(std::ofstream& py, const PlotRect<T>& rect)
{
  py << "ax.add_patch(plt.Rectangle((";
  py << std::to_string(rect.lx) << ", " << std::to_string(rect.ly) << "), ";
  py << std::to_string(rect.dx) << ", " << std::to_string(rect.dy) << ", ";
  py << "angle=" << std::to_string(rect.angle) << ", ";
  py << "edgecolor='" + rect.edgecolor + "', facecolor='" + rect.facecolor + "'))" << std::endl;
}

template <typename T>
inline void PyPlot<T>::saveFlyLine(std::ofstream& py, const PlotFlyLine<T>& flyline)
{
  T cx = (flyline.x[2] + flyline.x[0]) / 2;
  T cy = (flyline.y[1] + flyline.y[3]) / 2;
  py << "ax.plot([" + std::to_string(flyline.x[0]) + ", " + std::to_string(cx) + "], [" + std::to_string(flyline.y[0]) + ", "
            + std::to_string(cy) + "],linewidth = 0.1, color = 'green')\n";
  py << "ax.plot([" + std::to_string(flyline.x[1]) + ", " + std::to_string(cx) + "], [" + std::to_string(flyline.y[1]) + ", "
            + std::to_string(cy) + "],linewidth = 0.1, color = 'green')\n";
  py << "ax.plot([" + std::to_string(flyline.x[2]) + ", " + std::to_string(cx) + "], [" + std::to_string(flyline.y[2]) + ", "
            + std::to_string(cy) + "],linewidth = 0.1, color = 'green')\n";
  py << "ax.plot([" + std::to_string(flyline.x[3]) + ", " + std::to_string(cx) + "], [" + std::to_string(flyline.y[3]) + ", "
            + std::to_string(cy) + "],linewidth = 0.1, color = 'green')\n";
}

bool makeGif(const std::vector<std::string>& imgnames, const std::string gifname, size_t fps = 5, int isloop = 0)
{
  std::string py_script = gifname + ".py";
  std::ofstream py(py_script);
  py << "import imageio.v2 as imageio" << std::endl;
  // py << "import imageio" << std::endl;
  py << "images = []" << std::endl;
  for (auto&& imgname : imgnames) {
    py << "images.append(imageio.imread('" + imgname + "'))" << std::endl;
  }
  double duration = 1 / static_cast<double>(fps);
  py << "imageio.mimsave('" + gifname + "', images, \"gif\" , duration = " + std::to_string(duration) + ", loop =" + std::to_string(isloop)
            + ")";
  py.close();
  std::string command = python + " " + py_script;
  int result = 0;
#ifdef BUILD_GIF
  result = system(command.c_str());
#endif

  // result = std::remove(py_script.c_str());
  if (result != 0) {
    return false;
  }
  return true;
}

}  // namespace imp
#endif