#ifndef PROXSUITE_GQP_LDL_WRAPPER
#define PROXSUITE_GQP_LDL_WRAPPER

#include "GQPWithInit.hpp"
#include "fwd.hpp"
#include "proxsuite/linalg/dense/core.hpp"
#include "proxsuite/linalg/dense/ldlt.hpp"
#include "proxsuite/linalg/veg/memory/dynamic_stack.hpp"
#include "proxsuite/linalg/veg/vec.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T> struct GQPLDLWrapper : BaseGQPWithInitSupported<T> {
  linalg::dense::Ldlt<T> ldl;
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  Mat<T> kkt;
  Vec<T> rhs;

  GQPLDLWrapper(isize dim) : BaseGQPWithInitSupported<T>(dim) {}

  void _setupLDLT(isize kktDim, isize nDiag) {
    using namespace proxsuite::linalg::veg::dynstack;

    auto stackReq =
        ldl.factorize_req(kktDim) |
        (linalg::dense::temp_vec_req(proxsuite::linalg::veg::Tag<T>{}, nDiag) &
         StackReq::with_len(proxsuite::linalg::veg::Tag<isize>{}, nDiag) &
         ldl.diagonal_update_req(kktDim, nDiag)) |
        ldl.solve_in_place_req(kktDim);

    isize needed = stackReq.alloc_req();
    if (ldl_stack.len() < needed) {
      ldl.reserve_uninit(kktDim);
      ldl_stack.resize_for_overwrite(needed);
    }
  }

  auto _makeStack() -> proxsuite::linalg::veg::dynstack::DynStackMut {
    return {proxsuite::linalg::veg::from_slice_mut, ldl_stack.as_mut()};
  }

  void _buildKKT() {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;
    isize kktDim = n + m_eq + m_in;

    if (kkt.rows() != kktDim) {
      kkt.resize(kktDim, kktDim);
      rhs.resize(kktDim);
    }

    this->_setupLDLT(kktDim, m_eq + m_in);

    kkt.setZero();
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled;
    kkt.topLeftCorner(n, n).diagonal().array() += this->settings.default_rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-this->settings.default_mu_eq);
    kkt.diagonal()
        .segment(n + m_eq, m_in)
        .setConstant(-this->settings.default_mu_in);

    auto stack = _makeStack();
    ldl.factorize(kkt.transpose(), stack);
  }

  virtual void _updateFKBlocks(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;

    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      auto J = ineq.cone.dualJacobian(arg);
      auto JC = J * ineq.CScaled;

      kkt.block(n + m_eq + off, 0, dimC, n) = JC;
      kkt.block(0, n + m_eq + off, n, dimC) = JC.transpose();
      off += dimC;
    }

    auto stack = _makeStack();
    ldl.factorize(kkt.transpose(), stack);
  }

  void _updateBarrierParams(T rho_old, T rho_new, T r_eq_old, T r_eq_new,
                            T r_in_old, T r_in_new) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    auto stack = _makeStack();

    if (rho_old != rho_new) {
      kkt.topLeftCorner(n, n).diagonal().array() += rho_new - rho_old;
      ldl.factorize(kkt.transpose(), stack);
    }

    if (r_eq_old != r_eq_new || r_in_old != r_in_new) {
      isize r = m_eq + m_in;
      LDLT_TEMP_VEC_UNINIT(T, alpha, r, stack);
      alpha.head(m_eq).setConstant(r_eq_old - r_eq_new);
      alpha.tail(m_in).setConstant(r_in_old - r_in_new);

      auto indices_storage =
          stack.make_new_for_overwrite(proxsuite::linalg::veg::Tag<isize>{}, r);
      isize *indices = indices_storage.ptr_mut();
      for (isize k = 0; k < m_eq; ++k) {
        indices[k] = n + k;
      }
      for (isize k = 0; k < m_in; ++k) {
        indices[m_eq + k] = n + m_eq + k;
      }

      ldl.diagonal_update_clobber_indices(indices, r, alpha, stack);

      kkt.diagonal().segment(n, m_eq).setConstant(-r_eq_new);
      kkt.diagonal().segment(n + m_eq, m_in).setConstant(-r_in_new);
    }
  }

  void _solveKKT(VecRefMut<T> rhs) {
    auto stack = _makeStack();
    ldl.solve_in_place(rhs, stack);
  }
};

} // namespace dense

} // namespace proxgqp
} // namespace proxsuite

#endif
