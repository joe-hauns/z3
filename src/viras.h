#include "muz/rel/dl_lazy_table.h"
#include <optional>

#pragma once

namespace viras {

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
      std::optional<value_type<I>> next() { 
        if (end) {
          return {};
        } else {
          auto e = i.next();
          if (!bool(e) || !f(&*e)) {
            end = true;
            return {};
          } else {
            return std::optional<value_type<I>>(std::move(e));
          }
        }
      }
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
    { return range(0, a.size()) | map([&](auto i) { return a[i]; }); }

    template<class Array> auto array(Array      & a)
    { return range(0, a.size()) | map([&](auto i) { return a[i]; }); }


    template<class Array> auto array_ptr(Array const& a)
    { return range(0, a.size()) | map([&](auto i) { return &a[i]; }); }

    template<class Array> auto array_ptr(Array      & a)
    { return range(0, a.size()) | map([&](auto i) { return &a[i]; }); }

    template<class F>
    struct ClosureIter {
      F fn;

      auto next() -> std::invoke_result_t<F>
      { return fn(); }
    };

    template<class F>
    ClosureIter<F> closure(F fn)
    { return ClosureIter<F>{std::move(fn)}; }

    template<class A, unsigned N>
    struct ConstSizeIter {
      A _elems[N];
      unsigned _cur;

      auto next() -> std::optional<A>
      { return _cur < N ? std::optional<A>(std::move(_elems[_cur++]))
                        : std::optional<A>(); }
    };

    template<class A>
    constexpr auto empty() 
    { return ConstSizeIter<A, 0> { ._cur = 0, }; }

    template<class A, class... As>
    constexpr auto const_size(A a, As... as) 
    { return ConstSizeIter<A, std::tuple_size_v<std::tuple<A, As...>>>{ ._elems = {std::move(a), std::move(as)...}, ._cur = 0, }; }


    template<class... Is>
    struct IfThenElseIter {
      std::variant<Is...> self;

      auto next()
      { return std::visit([](auto&& x){ return x.next(); }, self); }
    };

    template<class Conds, class... Thens>
    struct IfThen {
      Conds _conds;
      std::tuple<Thens...> _thens;

      template<class Cond, class Then>
      auto else_if(Cond c, Then t) -> IfThen<decltype(std::tuple_cat(_conds, std::make_tuple(c))), Thens..., Then>;

      template<class Else>
      auto else_(Else e) -> IfThenElseIter<
                         std::invoke_result_t<Thens>..., 
                         std::invoke_result_t<Else>>;
    };

    template<class Cond, class Then>
    auto if_then(Cond c, Then t)
    { return IfThen<std::tuple<Cond>,Then> { std::make_tuple(std::move(c)), t }; }

    template<class I, class... Is>
    struct ConcatIter {
      std::tuple<I, Is...> self;
      unsigned idx;

      std::optional<value_type<I>> next();
      // { return std::visit([](auto&& x){ return x.next(); }, self); }
    };

    template<class... Is>
    ConcatIter<Is...> concat(Is... is) 
    { return ConcatIter<Is...> { std::make_tuple(std::move(is)...) }; }

  } // namespace iter

  enum class PredSymbol { Gt, Geq, Neq, Eq, };

  template<class Config>
  class Viras {
    using Term     = typename Config::Term;
    using Numeral  = typename Config::Numeral;
    using Var      = typename Config::Var;
    using Literals = typename Config::Literals;
    using Literal  = typename Config::Literal;
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

#define BIN_OP_WITH_CONFIG(OP, op_name)                                                   \
    template<class R>                                                                     \
    friend auto operator OP(int lhs, WithConfig<R> rhs)                                   \
    { return rhs.conf->numeral(lhs) OP rhs; }                                             \
                                                                                          \
    template<class L>                                                                     \
    friend auto operator OP(WithConfig<L> lhs, int rhs)                                   \
    { return lhs OP lhs.conf->numeral(rhs); }                                             \
                                                                                          \
    template<class L, class R>                                                            \
    friend auto operator OP(WithConfig<L> lhs, R rhs)                                     \
    { return withConfig(lhs.conf, lhs.conf->op_name(lhs.inner, rhs)); }                   \
                                                                                          \
    template<class L, class R>                                                            \
    friend auto operator OP(L lhs, WithConfig<R> rhs)                                     \
    { return withConfig(rhs.conf, rhs.conf->op_name(lhs, rhs.inner)); }                   \
                                                                                          \
    template<class L, class R>                                                            \
    friend auto operator OP(WithConfig<L> l, WithConfig<R> r)                             \
    { return l OP r.inner; }                                                              \

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
      bool periodic() { return oslp == 0; }
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
      auto N = iter::closure([this, n = numeral(0).inner, one = numeral(1)]() mutable {
          auto out = n;
          n = n + one;
          return std::optional(wrapConfig(out));
       });
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
              case Bound::Open: return (*n) * p < k; 
              case Bound::Closed: return (*n) * p <= k; 
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
            auto p_min = *(iter::array_ptr(out.breaks) 
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
    public:
      static VirtualTerm minusInf();
      static VirtualTerm plusEpsilon(Term t);
      static VirtualTerm plain(Term t);
    };

    class ElimSetIter2 {
    public:
      std::optional<VirtualTerm> next();
    };

    class ElimSetIter {
    public:
      std::optional<VirtualTerm> next();
    };

#define if_then_(x, y) if_then([&]() { return x; }, [&]() { return y; }) 
#define else_if_(x, y) .else_if([&]() { return x; }, [&]() { return y; })
#define else____(x) .else_([&]() { return x; })
#define else_is_(x,y) .else_([&]() { SASSERT(x); return y; })

    auto elim_set(Var const& x, Literal const& lit)
    {
      auto t = analyse(_config.term_of_literal(lit), x);
      auto symbol = _config.symbol_of_literal(lit);

      return iter::if_then_(t.breaks.empty(), 
                           iter::if_then_(t.sslp == 0              , iter::const_size(VirtualTerm::minusInf()))
                                 else_if_(symbol == PredSymbol::Neq, iter::const_size(VirtualTerm::minusInf(), VirtualTerm::plusEpsilon(t.zero(term(0)))))
                                 else_if_(symbol == PredSymbol:: Eq, iter::const_size(VirtualTerm::plain(t.zero(term(0)))))
                                 else_if_(t.sslp < 0               , iter::const_size(VirtualTerm::minusInf()))
                                 else_if_(symbol == PredSymbol::Geq, iter::const_size(VirtualTerm::plain(t.zero(term(0)))))
                                 else_is_(symbol == PredSymbol::Gt , iter::const_size(VirtualTerm::plusEpsilon(t.zero(term(0))))))

                   else_is_(!t.breaks.empty(), [&]() { 
                       auto ebreak       = ElimSetIter{};
                       auto eseg         = ElimSetIter{};
                       auto ebound_plus  = ElimSetIter{};
                       auto ebound_minus = ElimSetIter{};
                       return iter::if_then_(t.periodic(), iter::concat(ebreak, eseg))
                                    else____(              iter::concat(ebreak, eseg, ebound_plus, ebound_minus));
                   }());

    }

    auto elim_set(Var const& x, Literals const& lits)
    {
      return iter::array(lits) 
        | iter::flat_map([&](auto* lit) { return elim_set(x, lit); });
      // return iter::if_then([&](){ return  })
      // return iter::if_then([](){ return true; }, []() { return ElimSetIter { }; })
      //               .else_if([](){ return true; }, []() { return ElimSetIter2 { }; })
      //               .else_([]() { return ElimSetIter { }; });
      // return iter::if_then([](){ return true; }, []() { return ElimSetIter { }; })
      //               .else_if([](){ return true; }, []() { return ElimSetIter2 { }; })
      //               .else_([]() { return ElimSetIter { }; });
    }

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


}
