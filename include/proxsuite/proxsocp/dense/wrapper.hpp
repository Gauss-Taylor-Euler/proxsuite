// TODO
#include "fwd.hpp"
#include "model.hpp"
#include "proxsuite/helpers/optional.hpp"
#include <vector>
namespace proxsuite {
namespace proxsocp {
namespace dense {
/////SOCP object
/////Barebone for now
template <typename T> struct SOCP {
private:
  // TODO
  // Add other private compatible private attribute
public:
  Model<T> model;
  // TODO
  // Add other public compatible public attribute

  SOCP<T>(isize _dim, isize _n_eq, isize _n_in,
          std::vector<isize> _dimInequality)
      : model(Model<T>(_dim, _n_eq, _n_in, _dimInequality)) {}

  // Remove most of the options from the original proxqp dense wrapper
  // Kept only compute_preconditioner(unused for now) and the model parameter
  // TODO add compatible options
  void init(optional<MatRef<T>> H, optional<VecRef<T>> g, optional<MatRef<T>> A,
            optional<VecRef<T>> b, Vec<T> dInequality, Mat<T> cInequality,
            std::vector<Mat<T>> &AInequality, std::vector<Vec<T>> &bInequality,
            bool compute_preconditioner) {
    model.setAndCheck(H, g, A, b, dInequality, cInequality, AInequality,
                      bInequality);

    if (compute_preconditioner) {
      // TODO
      // Adapt the preconditioner logic
    }
  }

  // Solve socp withouth warmstart
  void solve() {}

  // Solve socp with warmstart
  void solve(optional<VecRef<T>> x, optional<VecRef<T>> y,
             optional<VecRef<T>> z) {};
};
} // namespace dense

} // namespace proxsocp
} // namespace proxsuite
