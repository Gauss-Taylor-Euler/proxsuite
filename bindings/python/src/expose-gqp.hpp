
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <proxsuite/proxgqp/dense/dense.hpp>

namespace proxsuite {
namespace proxgqp {
namespace python {

using proxsuite::linalg::veg::isize;

template <typename T> void exposeGQP(nanobind::module_ m) {
  ::nanobind::enum_<dense::GQPSolverStatus>(m, "GQPSolverStatus")
      .value("SOLVED", dense::GQPSolverStatus::GQP_SOLVED)
      .value("MAX_ITER_REACHED", dense::GQPSolverStatus::GQP_MAX_ITER_REACHED)
      .value("MAX_INNER_ITER_REACHED",
             dense::GQPSolverStatus::GQP_MAX_INNER_ITER_REACHED)
      .value("NOT_RUN", dense::GQPSolverStatus::GQP_NOT_RUN)
      .export_values();

  ::nanobind::enum_<dense::GQPStrategy>(m, "Strategy")
      .value("Base", dense::GQPStrategy::Base)
      .value("BaseWithoutProduct", dense::GQPStrategy::BaseWithoutProduct)
      .export_values();

  ::nanobind::class_<dense::GQPSettings<T>, proxsuite::proxqp::Settings<T>>(
      m, "Settings")
      .def(::nanobind::init<>(), "Default constructor for GQP settings.")
      .def_rw("penaltyReduction", &dense::GQPSettings<T>::penaltyReduction,
              "mu reduction factor in the BCL outer loop.")
      .def_rw("epsNewtonInit", &dense::GQPSettings<T>::epsNewtonInit,
              "initial Newton (inner loop) accuracy factor.")
      .def_rw("epsOuterInit", &dense::GQPSettings<T>::epsOuterInit,
              "initial outer loop accuracy factor.")
      .def_rw("lineSearchReduction",
              &dense::GQPSettings<T>::lineSearchReduction,
              "backtracking step reduction factor.")
      .def_rw("armijoConstant", &dense::GQPSettings<T>::armijoConstant,
              "Armijo sufficient-decrease constant.")
      .def_rw("maxLineSearchIters", &dense::GQPSettings<T>::maxLineSearchIters,
               "maximum number of line-search backtracking steps.")
      .def_rw("min_search_step", &dense::GQPSettings<T>::min_search_step,
               "minimum allowed line-search step before triggering rho "
               "increase.")
      .def_rw(
          "stepInCaseBelowMin",
          &dense::GQPSettings<T>::stepInCaseBelowMin,
          "step to take when the line-search step falls below min_search_step.")
      .def_rw("rhoIncreaseFactor",
              &dense::GQPSettings<T>::rhoIncreaseFactor,
              "factor by which rhoIncrease is multiplied after a small step.")
      .def_rw("maxRho", &dense::GQPSettings<T>::maxRho,
              "maximum allowed value for the proximal parameter rho.");

  ::nanobind::class_<dense::GQPResult<T>>(m, "Result")
      .def(::nanobind::init<>(), "Default constructor.")
      .def_ro("x", &dense::GQPResult<T>::x, "primal solution.")
      .def_ro("y", &dense::GQPResult<T>::y, "dual equality multipliers.")
      .def_ro("z", &dense::GQPResult<T>::z, "dual inequality multipliers.")
      .def_ro("outerIters", &dense::GQPResult<T>::outerIters,
              "number of outer BCL iterations.")
      .def_ro("totalInnerIters", &dense::GQPResult<T>::totalInnerIters,
              "total number of inner Newton iterations.")
      .def_ro("status", &dense::GQPResult<T>::status,
              "solver termination status.")
      .def_ro("pri_res", &dense::GQPResult<T>::pri_res,
              "infinity-norm of final KKT residual.")
      .def_ro("dua_res", &dense::GQPResult<T>::dua_res,
              "primal infeasibility norm.")
      .def_ro("mu_eq", &dense::GQPResult<T>::mu_eq,
              "final equality penalty parameter.")
      .def_ro("mu_in", &dense::GQPResult<T>::mu_in,
              "final inequality penalty parameter.")
      .def_ro("rho", &dense::GQPResult<T>::rho, "final proximal parameter.");

  ::nanobind::class_<dense::GQP<T>>(m, "GQP")
      .def(::nanobind::init<isize>(), nanobind::arg("dim"),
           "Constructor taking the primal dimension.")
      .def_rw("settings", &dense::GQP<T>::settings,
              "solver settings (GQPSettings).")
      .def("setObjective", &dense::GQP<T>::setObjective, nanobind::arg("H"),
           nanobind::arg("g"), "Set the objective: 0.5 x^T H x + g^T x.")
      .def("addEqualityConstraint", &dense::GQP<T>::addEqualityConstraint,
           nanobind::arg("A"), nanobind::arg("b"),
           "Add an equality constraint: A x = b.")
      .def("addInequalityConstraint", &dense::GQP<T>::addInequalityConstraint,
           nanobind::arg("C"), nanobind::arg("d"), nanobind::arg("cone"),
           "Add an inequality (cone) constraint:  C x + d<=_K 0.",
           // This is needed to not have a dangling reference with the cone
           // passed as argument
           nanobind::keep_alive<4, 1>())
      .def("initSolutionWithZero", &dense::GQP<T>::initSolutionWithZero,
           "Initialize solution with zeros.")
      .def("initSolutionWithEqualitySolution",
           &dense::GQP<T>::initSolutionWithEqualitySolution,
           "Initialize solution from equality-constrained KKT solve.")
      .def("initSolutionWithWarmStart",
           &dense::GQP<T>::initSolutionWithWarmStart, nanobind::arg("x"),
           nanobind::arg("y"), nanobind::arg("z"),
           "Initialize solution from a warm-start (x,y,z).")
      .def("solve", &dense::GQP<T>::solve, nanobind::arg("debug") = false,
           nanobind::arg("strategy") = dense::GQPStrategy::Base,
           "Solve the problem and return a Result.");
}

} // namespace python
} // namespace proxgqp
} // namespace proxsuite
