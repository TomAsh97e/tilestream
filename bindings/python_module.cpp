#include <tilestream/info.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_tilestream, module) {
    module.doc() = "TileStream Python bindings";
    module.attr("__version__") = tilestream::version();
    module.def("info_json", &tilestream::info_json);
}
