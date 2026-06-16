
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <proxsuite/proxgqp/dense/BaseGQP.hpp>
#include <proxsuite/proxgqp/dense/cones/LorentzCone.hpp>
#include <proxsuite/proxgqp/dense/cones/PositiveOrthantCone.hpp>

namespace proxsuite {
namespace proxgqp {
namespace python {

using proxsuite::linalg::veg::isize;

template <typename T> void exposeCones(nanobind::module_ m) {
  ::nanobind::class_<dense::Cone<T>>(m, "Cone");

  ::nanobind::class_<dense::LorentzCone<T>, dense::Cone<T>>(m, "LorentzCone")
      .def(::nanobind::init<isize>(), nanobind::arg("dim"),
           "Second-order (Lorentz) cone.");

  ::nanobind::class_<dense::PositiveOrthantCone<T>, dense::Cone<T>>(
      m, "PositiveOrthantCone")
      .def(::nanobind::init<isize>(), nanobind::arg("dim"),
           "Nonnegative orthant cone.");
}

} // namespace python
} // namespace proxgqp
} // namespace proxsuite
