/*++
Copyright (c) 2015 Microsoft Corporation

--*/


#pragma once

#include "ast/arith_decl_plugin.h"
#include "model/model.h"
#include "qe/mbp/mbp_plugin.h"
#include <optional>

namespace mbp {


    namespace iter {

    template<class T> struct opt_value_type;
    // template<class T> struct opt_value_type<std::optional<T>> { using type = T; };
    template<template<class> class Tmplt, class T> struct opt_value_type<Tmplt<T>> { using type = T; };

    template<class I>
    using value_type = typename opt_value_type<decltype(((I*)0)->next())>::type;


    template<class Op>
    struct IterOperator {
      Op self;
      template<class I>
      friend auto operator|(I iter, IterOperator self) -> decltype(auto) 
      { return (self.self)(iter); }

      template<class Op2>
      auto compose(IterOperator<Op2> other) {
        return iterOperator([self = std::move(this->self), other = std::move(other)](auto iter){ return (other.self)((self)(iter)); });
      }
    };

    template<class O>
    constexpr IterOperator<O> iterOperator(O o) 
    { return IterOperator<O>{ std::move(o) }; }

    constexpr auto foreach = [](auto f) {
      return iterOperator([f = std::move(f)](auto iter) {
        for (auto x = iter.next(); x; x = iter.next()) {
          f(*x);
        }
      });
    };

    template<class I, class F>
    struct MapIter {
      I i;
      F f;
      auto next() -> decltype(auto) { 
        auto next = i.next();
        return next ? std::optional<decltype(f(*next))>(f(*next))
                    : std::optional<decltype(f(*next))>();
      }
    };

    constexpr auto map = [](auto f) {
      return iterOperator([f = std::move(f)](auto i) {
        return MapIter<decltype(i), decltype(f)>{i,f};
      });
    };


    template<class I, class F>
    struct FilterIter {
      I i;
      F f;
      auto next() -> decltype(auto) { 
        auto e = i.next();
        while (e && !f(&*e))
          e = i.next();
        return e;
      }
    };

    constexpr auto filter = [](auto f) {
      return iterOperator([f = std::move(f)](auto i) {
        return FilterIter<decltype(i), decltype(f)>(i,f);
      });
    };

    template<class I, class F>
    struct TakeWhileIter {
      I i;
      F f;
      bool end;
      optional<value_type<I>> next();
      // optional<value_type<I>> next() { 
      //   if (end) {
      //     return {};
      //   } else {
      //     auto e = i.next();
      //     if (!bool(e) || !f(&*e)) {
      //       end = true;
      //       return {};
      //     } else {
      //       return e;
      //     }
      //   }
      // }
    };

    constexpr auto take_while = [](auto f) {
      return iterOperator([f = std::move(f)](auto i) {
        return TakeWhileIter<decltype(i), decltype(f)>{ std::move(i),std::move(f), false, };
      });
    };

    template<class I>
    struct FlattenIter {
      I outer;
      std::optional<value_type<I>> inner;
      bool init;

      FlattenIter(I i): outer(std::move(i)), inner(), init(false) {}

      optional<value_type<value_type<I>>> next()
      {
        if (!init) {
          inner = outer.next();
          init = true;
        }
        while (inner) {
          auto next = inner->next();
          if (next) 
            return optional(*next);
          else 
            inner = outer.next();
        }
        return {};
      }
    };

    constexpr auto flatten = iterOperator([](auto i) {
      return FlattenIter<decltype(i)>(i);
    });

    constexpr auto min = iterOperator([](auto iter) {
      using out = std::optional<value_type<decltype(iter)>>;
      auto min_ = iter.next();
      if (min_) {
        auto min = std::move(*min_);
        for (auto x = iter.next(); x; x = iter.next()) {
          if (x < min) {
            min = std::move(*x);
          }
        }
        return out(std::move(min));
      } else {
        return out();
      }
    });

    constexpr auto flat_map = [](auto f) {
      return map(f).compose(flatten);
    };

    template<class T>
    struct RangeIter {
      T lower;
      T upper;

      optional<T> next()
      {
        if (lower < upper) return optional(lower++);
        else return {};
      }
    };


    template<class L, class U>
    RangeIter<U> range(L lower, U upper)
    { return RangeIter<U>{U(std::move(lower)), std::move(upper)}; }

    template<class Array>
    auto array(Array && a)
    { return range(0, a.size())
       | map([a = std::move(a)](auto i) { return std::move(a[i]); }); }

    template<class Array> auto array(Array const& a)
    { return range(0, a.size()) | map([&](auto i) { return &a[i]; }); }

    template<class Array> auto array(Array      & a)
    { return range(0, a.size()) | map([&](auto i) { return &a[i]; }); }

    template<class F>
    struct ClosureIter {
      F fn;

      auto next() -> std::invoke_result_t<F>;
      // { return fn(); }
    };

    template<class F>
    ClosureIter<F> closure(F fn)
    { return ClosureIter<F>{std::move(fn)}; }

    } // namespace iter


    struct z3_viras_config {
      ast_manager& m;
      arith_util m_arith;
      z3_viras_config(ast_manager* m) :m(*m), m_arith(*m) {}

      using Literals = expr_ref_vector; 
      using Var      = app*; 
      using Term     = expr*;
      using Numeral  = rational;

      Numeral numeral(int);
      Numeral lcm(Numeral l, Numeral r);

      Numeral mul(Numeral l, Numeral r);
      Numeral add(Numeral l, Numeral r);
      Numeral div(Numeral l, Numeral r);

      Term mul(Numeral l, Term r);
      Term div(Term l, Numeral r);
      Term add(Term l, Term r);
      Term add(Term l, Numeral r);
      Term add(Numeral l, Term r);
      Term floor(Term t);
      Term minus(Term t);

      Term term(Numeral n);
      Term one() { return term(numeral(1)); }

      Term subs(Term term, Var var, Term by);

      /* the numerator of some rational */
      Numeral num(Numeral l);

      /* the denomiantor of some rational */
      Numeral den(Numeral l);

      template<class IfVar, class IfOne, class IfMul, class IfAdd, class IfFloor>
      auto matchTerm(Term t, 
          IfVar   if_var, 
          IfOne   if_one, 
          IfMul   if_mul, 
          IfAdd   if_add, 
          IfFloor if_floor) -> decltype(auto) {
        expr* e1;
        expr* e2;
        Numeral q;
        if (m_arith.is_mul(t, e1, e2)) {
          expr* e = nullptr;
          if (m_arith.is_numeral(e1, q)) { e = e1; }
          if (m_arith.is_numeral(e2, q)) { e = e2; }
          SASSERT(e != nullptr); // TODO what if someone passes a non-linar term
          return if_mul(q, e);
        } else if (m_arith.is_numeral(t, q)) {
          if (q.is_one()) {
            return if_one();
          } else {
            return if_mul(q, this->one());
          }
        } else if (m_arith.is_add(t, e1, e2)) {
          return if_add(e1, e2);
        } else if (m_arith.is_to_int(t, e1)) {
          return if_floor(e1);
        } else {
          SASSERT(is_app(t)); // TODO what if someone passes in a var? why don't we use vars in the first place?
          auto var = (app*)t; 
          SASSERT(var->get_num_args() == 0); // TODO what if someone passes an uninterpreted function?
          return if_var(var);
        }
      }
    };

    template<class Config>
    class Viras {
      using Term     = typename Config::Term;
      using Numeral  = typename Config::Numeral;
      using Var      = typename Config::Var;
      using Literals = typename Config::Literals;
      Config _config;
    public:

      template<class... Args>
      Viras(Args... args) : _config(args...) { };
      ~Viras() {};

      template<class C>
      struct WithConfig {
        Config* conf;
        C inner;
        operator C&() { return inner; }
        operator C const&() const { return inner; }
      };

      template<class T>
      static WithConfig<T> withConfig(Config* c, T inner)
      { return WithConfig<T> { c, std::move(inner) }; }

#define BIN_OP_WITH_CONFIG(OP, op_name) \
      template<class R> \
      friend auto operator OP(int lhs, WithConfig<R> rhs)  \
      { return rhs.conf->numeral(lhs) OP rhs; } \
      \
      template<class L> \
      friend auto operator OP(WithConfig<L> lhs, int rhs)  \
      { return lhs OP lhs.conf->numeral(rhs); } \
 \
      template<class L, class R> \
      friend auto operator OP(WithConfig<L> lhs, R rhs)  \
      { return withConfig(lhs.conf, lhs.conf->op_name(lhs.inner, rhs)); } \
 \
      template<class L, class R> \
      friend auto operator OP(L lhs, WithConfig<R> rhs)  \
      { return withConfig(rhs.conf, rhs.conf->op_name(lhs, rhs.inner)); } \
 \
      template<class L, class R> \
      friend auto operator OP(WithConfig<L> l, WithConfig<R> r)  \
      { SASSERT(&l.conf == &r.conf); return l OP r.inner; } \

      BIN_OP_WITH_CONFIG(*, mul);
      BIN_OP_WITH_CONFIG(+, add);
      BIN_OP_WITH_CONFIG(/, div);

      template<class T>
      friend auto operator-(WithConfig<T> self) 
      { return withConfig(self.conf, self.conf->minus(self.inner)); }

      template<class L, class R>
      friend auto operator-(WithConfig<L> lhs, WithConfig<R> rhs)
      { return lhs.inner + rhs; }

      template<class L, class R>
      friend auto operator-(WithConfig<R> lhs, R rhs)
      { return withConfig(lhs.conf, lhs.conf->add(lhs.inner, lhs.conf->minus(rhs))); }

      template<class L, class R>
      friend auto operator-(L lhs, WithConfig<R> rhs)
      { return withConfig(rhs.conf, rhs.conf->add(lhs, rhs.conf->minus(rhs.inner))); }


      template<class T>
      auto floor(WithConfig<T> self) { return withConfig(self.conf, self.conf->floor(self.inner)); }

      template<class T>
      auto ceil(WithConfig<T> self) { return -floor(-self); };

      WithConfig<Term> floor(Term t) { return floor(wrapConfig(t)); }
      WithConfig<Term> ceil(Term t) { return ceil(wrapConfig(t)); }

      struct Break {
        Term t;
        Numeral p;
      };

      WithConfig<Term> quot(WithConfig<Term> t, Numeral p) { return floor(t / p); }
      WithConfig<Term> rem(WithConfig<Term> t, Numeral p) { return t - p * quot(t, p); }

      WithConfig<Term> grid_ceil (WithConfig<Term> t, Break s_pZ) { return t + rem(s_pZ.t - t, s_pZ.p); }
      WithConfig<Term> grid_floor(WithConfig<Term> t, Break s_pZ) { return t - rem(t - wrapConfig(s_pZ.t), s_pZ.p); }

      WithConfig<Term> grid_ceil (Term t, Break s_pZ) { return grid_ceil (wrapConfig(t), s_pZ); }
      WithConfig<Term> grid_floor(Term t, Break s_pZ) { return grid_floor(wrapConfig(t), s_pZ); }

      struct LiraTerm {
        Term self;
        Var x;
        Term lim;
        Numeral sslp;
        Numeral oslp;
        Numeral per;
        Numeral deltaY;
        Term distYminus;
        Term distYplus();
        std::vector<Break> breaks;

        WithConfig<Term> lim_at(WithConfig<Term> x0) { return WithConfig<Term> { x0.conf, x0.conf->subs(lim, x, x0), }; }
        WithConfig<Term> dseg(WithConfig<Term> x0) { return -(sslp * x0) + lim_at(x0); }
        WithConfig<Term> zero(WithConfig<Term> x0) { return x0 - lim_at(x0) / sslp; }

      };

      template<class T> WithConfig<T> wrapConfig(T t) { return WithConfig<T> { &_config, std::move(t) }; }
      template<class T> WithConfig<T> wrapConfig(WithConfig<T> t) { return t; }


      template<class IfVar, class IfOne, class IfMul, class IfAdd, class IfFloor>
      auto matchTerm(Term t, 
          IfVar   if_var, 
          IfOne   if_one, 
          IfMul   if_mul, 
          IfAdd   if_add, 
          IfFloor if_floor
          ) -> decltype(auto) {
        return _config.matchTerm(t,
          [&](auto x) { return wrapConfig(if_var(wrapConfig(x))); }, 
          [&]() { return wrapConfig(if_one()); }, 
          [&](auto l, auto r) { return wrapConfig(if_mul(wrapConfig(l),r)); }, 
          [&](auto l, auto r) { return wrapConfig(if_add(wrapConfig(l),r)); }, 
          [&](auto x) { return wrapConfig(if_floor(wrapConfig(x))); }
             );
      }

      WithConfig<Numeral> numeral(int i) { return wrapConfig(_config.numeral(i)); }
      WithConfig<Term>    term(int i)    { return wrapConfig(_config.term(numeral(i))); }

      enum class Bound {
        Open,
        Closed,
      };

      auto intersectGrid(Break s_pZ, Bound l, Term t, Numeral k, Bound r) {
        auto f = [n = numeral(0).inner, one = numeral(1)]() mutable {
            auto out = n;
            n = n + one;
            return std::optional(out);
         };
        auto N = iter::closure(std::move(f));
        static_assert(std::is_same_v<decltype(N), iter::ClosureIter<decltype(f)>>);
        static_assert(std::is_same_v<iter::value_type<decltype(N)>, Numeral>);
        auto p  = wrapConfig(s_pZ.p);
        auto start = [&]() {
          switch(l) {
            case Bound::Open:   return grid_floor(t + p, s_pZ);
            case Bound::Closed: return grid_ceil(t, s_pZ);
          };
        }();
        return std::move(N) 
          | iter::take_while([r,p,k](auto n) -> bool { 
              switch(r) {
                case Bound::Open: return n * p < k; 
                case Bound::Closed: return n * p <= k; 
              }
            })
          | iter::map([start, p](auto n) {
            return start + p * n;
            });
      }

      LiraTerm analyse(Term self, Var x) {
        return matchTerm(self, 
          /* var v */ [&](auto y) { return LiraTerm {
            .self = self,
            .x = x,
            .lim = self,
            .sslp = _config.numeral(y == x ? 1 : 0),
            .oslp = _config.numeral(y == x ? 1 : 0),
            .per = _config.numeral(0),
            .deltaY = _config.numeral(0),
            .distYminus = y == x ? term(0) 
                                 : y,
            .breaks = std::vector<Break>(),
          }; }, 

          /* numeral 1 */ [&]() { return LiraTerm {
            .self = self,
            .x = x,
            .lim = self,
            .sslp = _config.numeral(0),
            .oslp = _config.numeral(0),
            .per = _config.numeral(0),
            .deltaY = _config.numeral(0),
            .distYminus = _config.term(_config.numeral(1)),
            .breaks = std::vector<Break>(),
          }; }, 
          /* k * t */ [&](auto k, auto t) { 
            auto rec = analyse(t, x);
            return LiraTerm {
              .self = self,
              .x = x,
              .lim = (k * rec.lim).inner,
              .sslp = (k * rec.sslp).inner,
              .oslp = (k * rec.sslp).inner,
              .per = rec.per,
              .deltaY = abs(k) * rec.deltaY,
              .distYminus = k >= 0 ? (k * rec.distYminus).inner
                                   : (k * rec.distYplus()).inner,
              .breaks = std::move(rec.breaks),
            }; 
          }, 

          /* l + r */ [&](auto l, auto r) { 
            auto rec_l = analyse(l, x);
            auto rec_r = analyse(r, x);
            auto breaks = std::move(rec_l.breaks);
            breaks.insert(breaks.end(), rec_l.breaks.begin(), rec_l.breaks.end());
            return LiraTerm {
              .self = self,
              .x = x,
              .lim = rec_l.lim + wrapConfig(rec_r.lim),
              .sslp = rec_l.sslp + wrapConfig(rec_r.sslp),
              .oslp = rec_l.sslp + wrapConfig(rec_r.sslp),
              .per = rec_l.per == 0 ? rec_r.per
                   : rec_r.per == 0 ? rec_l.per
                   : _config.lcm(rec_l.per, rec_r.per),
              .deltaY = rec_l.deltaY + rec_r.deltaY,
              .distYminus = wrapConfig(rec_l.distYminus) + rec_r.distYminus,
              .breaks = std::move(breaks),
            }; 
          }, 

          /* floor */ [&](auto t) { 
            auto rec = analyse(t, x);
            
            auto out = LiraTerm {
              .self = self,
              .x = x,
              .lim = rec.sslp >= 0 ? floor(rec.lim) 
                                   : ceil(rec.lim) - 1,
              .sslp = numeral(0),
              .oslp = rec.oslp,
              .per = rec.per == 0 && rec.oslp == 0 ? numeral(0)
                   : rec.per == 0                  ? 1 / abs(rec.oslp)
                   : _config.num(rec.per) * _config.den(rec.oslp),
              .deltaY = rec.deltaY + 1,
              .distYminus = rec.distYminus - 1,
              .breaks = std::vector<Break>(),
            }; 
            if (rec.sslp == 0) {
              out.breaks = std::move(rec.breaks);
            } else if (rec.breaks.empty()) {
              out.breaks.push_back(Break {rec.zero(term(0)).inner, out.per});
            } else {
              auto p_min = *(iter::array(out.breaks) 
                | iter::map([](auto b) -> Numeral { return b->p; })
                | iter::min);
              for ( auto b0p_pZ : rec.breaks ) {
                auto b0p = b0p_pZ.t;
                auto p   = b0p_pZ.p;
                intersectGrid(b0p_pZ, 
                              Bound::Closed, b0p, out.per, Bound::Open) 
                  | iter::foreach([&](auto b0) {
                      intersectGrid(Break{rec.zero(b0), 1/rec.sslp}, 
                                    Bound::Closed, b0, p_min, Bound::Open)
                        | iter::foreach([&](auto b) {
                            out.breaks.push_back(Break{b, out.per });
                        });
                  });
              }
              out.breaks.insert(out.breaks.end(), rec.breaks.begin(), rec.breaks.end());
            }
            return out;
          }
          );
      };


      class VirtualTerm {

      };

      class ElimSetIter {
      public:
        std::optional<VirtualTerm> next();
      };

      ElimSetIter elim_set(Var const& x, Literals const& lits);

      class VsubsIter {
      public:
        std::optional<Literals> next();
      };

      VsubsIter vsubs(Literals const& lits, Var const& x, VirtualTerm& term);

      auto quantifier_elimination(Var const& x, Literals const& lits)
      {
        Term* t = 0;
        analyse(*t, x);
        return elim_set(x, lits)
          | iter::flat_map([&](auto t) { return vsubs(lits, x, t); });
      }
    };

    /**
      VIRAS quantifier elimination
      TODO more explanation & ref paper
     */

    class viras_project_plugin : public project_plugin {
        ast_manager            m;
    public:
        viras_project_plugin(ast_manager& m) :project_plugin(m), m(m) {}
        ~viras_project_plugin() override {}
        
        bool operator()(model& model, app* var, app_ref_vector& vars, expr_ref_vector& lits) override;
        bool solve(model& model, app_ref_vector& vars, expr_ref_vector& lits) override { return false; }
        family_id get_family_id() override;
        bool operator()(model& model, app_ref_vector& vars, expr_ref_vector& lits) override;
        bool project(model& model, app_ref_vector& vars, expr_ref_vector& lits, vector<def>& defs) override;
        void saturate(model& model, func_decl_ref_vector const& shared, expr_ref_vector& lits) override { UNREACHABLE(); }

        // /**
        //  * \brief check if formulas are purified, or leave it to caller to ensure that
        //  * arithmetic variables nested under foreign functions are handled properly.
        //  */
        // void set_check_purified(bool check_purified);
        //
        // /**
        // * \brief apply projection 
        // */
        // void set_apply_projection(bool apply_project);

    };


};


