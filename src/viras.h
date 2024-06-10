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
          inner.~decltype(inner)();
          new(&inner) decltype(inner)(outer.next());
          init = true;
        }
        while (inner) {
          auto next = inner->next();
          if (next) 
            return optional(*next);
          else {
            inner.~decltype(inner)();
            new(&inner) decltype(inner)(outer.next());
          }
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

      std::optional<T> next()
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
    struct ValsIter {
      A _vals[N];
      unsigned _cur;

      auto next() -> std::optional<A>
      { return _cur < N ? std::optional<A>(std::move(_vals[_cur++]))
                        : std::optional<A>(); }
    };

    template<class A>
    constexpr auto empty() 
    { return ValsIter<A, 0> { ._cur = 0, }; }

    template<class A, class... As>
    constexpr auto vals(A a, As... as) 
    { return ValsIter<A, std::tuple_size_v<std::tuple<A, As...>>>{ ._vals = {std::move(a), std::move(as)...}, ._cur = 0, }; }


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
      auto else_if(Cond c, Then t) -> IfThen<decltype(std::tuple_cat(_conds, std::make_tuple(c))), Thens..., Then> {
        return IfThen<decltype(std::tuple_cat(_conds, std::make_tuple(c))), Thens..., Then> {
          ._conds = std::tuple_cat(std::move(_conds), std::make_tuple(std::move(c))),
          ._thens = std::tuple_cat(std::move(_thens), std::make_tuple(std::move(t))),
        };
      }

      template<unsigned i, unsigned sz> 
      struct __TryIfThen {
        template<class Out>
        static std::optional<Out> apply(IfThen& self) {
          if (std::get<i>(self._conds)()) {
            return std::optional<Out>(Out(std::in_place_index_t<i>(), std::get<i>(self._thens)()));
          } else {
            return __TryIfThen<i + 1, sz>::template apply<Out>(self);
          }
        };
      };

      template<unsigned sz>
      struct __TryIfThen<sz, sz> {
        template<class Out>
        static std::optional<Out> apply(IfThen& self) {
          return {};
        };
      };

      template<class Else>
      auto else_(Else e) -> IfThenElseIter<
                         std::invoke_result_t<Thens>..., 
                         std::invoke_result_t<Else>> 
      {
        using Out = IfThenElseIter<std::invoke_result_t<Thens>..., 
                                  std::invoke_result_t<Else>>;
        using Var = std::variant<std::invoke_result_t<Thens>..., 
                                 std::invoke_result_t<Else>>;
        auto var = __TryIfThen<0, std::tuple_size_v<Conds>>::template apply<Var>(*this);
        if (var) {
          return Out{.self = std::move(*var)};
        } else {
          return Out{.self = Var(std::in_place_index_t<std::tuple_size_v<Conds>>(),e())};
        }
      }
    };

    template<class Cond, class Then>
    auto if_then(Cond c, Then t)
    { return IfThen<std::tuple<Cond>,Then> { std::make_tuple(std::move(c)), t }; }

    template<class I, class... Is>
    struct ConcatIter {

      std::tuple<I, Is...> self;
      unsigned idx;

      template<unsigned i, unsigned sz> 
      struct __TryConcat {
        static std::optional<value_type<I>> apply(ConcatIter& self) {
          if (self.idx == i) {
            auto out = std::get<i>(self.self).next();
            if (out) {
              return out;
            } else {
              self.idx++;
            }
          }
          return __TryConcat<i + 1, sz>::apply(self);
        }
      };

      template<unsigned sz>
      struct __TryConcat<sz, sz> {
        static std::optional<value_type<I>> apply(ConcatIter& self) {
          return {};
        };
      };

      std::optional<value_type<I>> next()
      { return __TryConcat<0, std::tuple_size_v<std::tuple<I, Is...>>>::apply(*this); }
    };

    template<class... Is>
    ConcatIter<Is...> concat(Is... is) 
    { return ConcatIter<Is...> { std::make_tuple(std::move(is)...) }; }

    template<class T>
    struct NatIter {
      T _cur;

      std::optional<T> next() {
        auto out = _cur;
        _cur = _cur + 1;
        return out;
      }
    };


    template<class T>
    NatIter<T> nat_iter(T first = T(0)) {
      return NatIter<T> { ._cur = std::move(first) };
    }
  } // namespace iter

  enum class PredSymbol { Gt, Geq, Neq, Eq, };

  template<class Config, class T>
  struct WithConfig { 
    Config* config; 
    T inner; 
    friend bool operator==(WithConfig l, WithConfig r) 
    { 
      SASSERT(l.config == r.config);
      return l.inner == r.inner;
    }
    friend bool operator!=(WithConfig l, WithConfig r) 
    { return !(l == r); }
  };

  template<class Config>
  struct CTerm : public WithConfig<Config, typename Config::Term> { };

  template<class Config>
  struct CNumeral : public WithConfig<Config, typename Config::Numeral> {};

  template<class Config>
  struct CVar : public WithConfig<Config, typename Config::Var> {};

  template<class Config>
  struct CLiteral : public WithConfig<Config, typename Config::Literal> {
    CTerm<Config> term() const
    { return CTerm<Config> {this->config, this->config->term_of_literal(this->inner)}; }

    PredSymbol symbol() const
    { return this->config->symbol_of_literal(this->inner); }
  };

  template<class Config>
  struct CLiterals : public WithConfig<Config, typename Config::Literals> {
    auto size() const { return this->config->literals_size(this->inner); }
    auto operator[](size_t idx) const { 
      return CLiteral<Config> { this->config, this->config->literals_get(this->inner, idx) }; 
    }
  };

  ///////////////////////////////////////
  // PRIMARY OPERATORS
  ///////////////////////////////////////

  template<class Config>
  CTerm<Config> operator+(CTerm<Config> lhs, CTerm<Config> rhs) 
  {
    SASSERT(lhs.config == rhs.config);
    return CTerm<Config> {lhs.config, lhs.config->add(lhs.inner, rhs.inner)};
  }

  template<class Config>
  CTerm<Config> operator*(CNumeral<Config> lhs, CTerm<Config> rhs) 
  {
    SASSERT(lhs.config == rhs.config);
    return CTerm<Config> {lhs.config, lhs.config->mul(lhs.inner, rhs.inner)};
  }

  template<class Config>
  CNumeral<Config> operator*(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  {
    SASSERT(lhs.config == rhs.config);
    return CNumeral<Config> {lhs.config, lhs.config->mul(lhs.inner, rhs.inner)};
  }

  ///////////////////////////////////////
  // LIFTED OPERATORS
  ///////////////////////////////////////

  template<class Config>
  CNumeral<Config> operator+(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return CNumeral<Config> { lhs.config, lhs.config->add(lhs.inner, rhs.inner)}; }

  template<class Config>
  CTerm<Config> operator+(CNumeral<Config> lhs, CTerm<Config> rhs) 
  { return CTerm<Config> { rhs.config, lhs.config->term(lhs.inner)} + rhs; }

  template<class Config>
  CTerm<Config> operator+(CTerm<Config> lhs, CNumeral<Config> rhs) 
  { return lhs + CTerm<Config> { rhs.config, rhs.config->term(rhs.inner), }; }

#define LIFT_NUMRAL_TO_TERM_L(function)                                            \
  template<class Config>                                                                  \
  auto function(CNumeral<Config> lhs, CTerm<Config> rhs)                                      \
  { return function(CTerm<Config> {lhs.config, lhs.config->term(lhs)}, rhs); }      \

#define LIFT_NUMRAL_TO_TERM_R(function)                                            \
  template<class Config>                                                                  \
  auto function(CTerm<Config> lhs, CNumeral<Config> rhs)                                      \
  { return function(lhs, CTerm<Config> {rhs.config, rhs.config->term(rhs)}); }      \

#define LIFT_INT_TO_NUMERAL(function, CType)                                              \
  LIFT_INT_TO_NUMERAL_L(function, CType)                                                  \
  LIFT_INT_TO_NUMERAL_R(function, CType)                                                  \


#define LIFT_INT_TO_NUMERAL_L(function, CType)                                            \
  template<class Config>                                                                  \
  auto function(int lhs, CType<Config> rhs)                                      \
  { return function(CNumeral<Config> {rhs.config, rhs.config->numeral(lhs)}, rhs); }      \

#define LIFT_INT_TO_NUMERAL_R(function, CType)                                            \
  template<class Config>                                                                  \
  auto function(CType<Config> lhs, int rhs)                                      \
  { return function(lhs, CNumeral<Config> {lhs.config, lhs.config->numeral(rhs)}); }      \

#define LIFT_INT_TO_NUMERAL(function, CType)                                              \
  LIFT_INT_TO_NUMERAL_L(function, CType)                                                  \
  LIFT_INT_TO_NUMERAL_R(function, CType)                                                  \

  LIFT_INT_TO_NUMERAL(operator+, CNumeral)
  LIFT_INT_TO_NUMERAL(operator-, CNumeral)
  LIFT_INT_TO_NUMERAL(operator*, CNumeral)

  LIFT_INT_TO_NUMERAL(operator==, CNumeral)
  LIFT_INT_TO_NUMERAL(operator!=, CNumeral)
  LIFT_INT_TO_NUMERAL(operator<=, CNumeral)
  LIFT_INT_TO_NUMERAL(operator>=, CNumeral)
  LIFT_INT_TO_NUMERAL(operator< , CNumeral)
  LIFT_INT_TO_NUMERAL(operator> , CNumeral)

  LIFT_INT_TO_NUMERAL(operator+, CTerm)
  LIFT_INT_TO_NUMERAL(operator-, CTerm)
  LIFT_INT_TO_NUMERAL_L(operator*, CTerm)


   // MULTIPLICATION
   // DIVISION

  template<class Config>
  CTerm<Config> operator/(CTerm<Config> lhs, CNumeral<Config> rhs) 
  { return (1 / rhs) * lhs; }

  template<class Config>
  CNumeral<Config> operator/(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return CNumeral<Config>{ rhs.config, rhs.config->inverse(rhs.inner) } * lhs; }

  LIFT_INT_TO_NUMERAL_R(operator/, CTerm)
  LIFT_INT_TO_NUMERAL(operator/, CNumeral)

   // MINUS

#define DEF_UMINUS(CType) \
  template<class Config> \
  CType<Config> operator-(CType<Config> x)  \
  { return -1 * x; } \

DEF_UMINUS(CNumeral)
DEF_UMINUS(CTerm)

#define DEF_BMINUS(T1, T2) \
  template<class Config> \
  auto operator-(T1<Config> x, T2<Config> y)  \
  { return x + -y; } \

DEF_BMINUS(CNumeral, CNumeral)
DEF_BMINUS(CTerm   , CNumeral)
DEF_BMINUS(CNumeral, CTerm   )
DEF_BMINUS(CTerm   , CTerm   )

   // ABS

  template<class Config>
  CNumeral<Config> abs(CNumeral<Config> x) 
  { return x < 0 ? -x : x; }

  template<class Config>
  CNumeral<Config> num(CNumeral<Config> x);

  template<class Config>
  CNumeral<Config> den(CNumeral<Config> x);

  template<class Config>
  CNumeral<Config> lcm(CNumeral<Config> l, CNumeral<Config> r);
  // { return x < 0 ? -x : x; }

   // floor

  template<class Config>
  CTerm<Config> floor(CTerm<Config> x) 
  { return CTerm<Config> { x.config, x.config->floor(x.inner), }; }

  template<class Config>
  CTerm<Config> ceil(CTerm<Config> x) 
  { return -floor(-x); }


  template<class Config>
  CNumeral<Config> floor(CNumeral<Config> x) 
  { return CNumeral<Config> { x.config, x.config->floor(x.inner), }; }

  // COMPARISIONS

  template<class Config>
  bool operator<=(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return lhs.config->leq(lhs.inner, rhs.inner); }

  template<class Config>
  bool operator<(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return lhs.config->less(lhs.inner, rhs.inner); }

  template<class Config>
  bool operator>=(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return rhs <= lhs; }

  template<class Config>
  bool operator>(CNumeral<Config> lhs, CNumeral<Config> rhs) 
  { return rhs < lhs; }

#define INT_CMP(OP)                                                                       \
  template<class Config>                                                                  \
  bool operator OP (CNumeral<Config> lhs, int rhs)                                        \
  { return lhs OP CNumeral <Config> { lhs.config, lhs.config->numeral(rhs), }; }          \
                                                                                          \
  template<class Config>                                                                  \
  bool operator OP (int lhs, CNumeral<Config> rhs)                                        \
  { return CNumeral <Config> { rhs.config, rhs.config->numeral(lhs), } OP rhs; }          \

  template<class Config>
  class Viras {
    using Term     = CTerm<Config>;
    using Numeral  = CNumeral<Config>;
    using Var      = CVar<Config>;
    using Literals = CLiterals<Config>;
    using Literal  = CLiteral<Config>;
    Config _config;
  public:

    template<class... Args>
    Viras(Args... args) : _config(args...) { };
    ~Viras() {};

    struct Break {
      Term t;
      Numeral p;
    };

    Term quot(Term t, Numeral p) { return floor(t / p); }
    Term rem(Term t, Numeral p) { return t - p * quot(t, p); }

    Term grid_ceil (Term t, Break s_pZ) { return t + rem(s_pZ.t - t, s_pZ.p); }
    Term grid_floor(Term t, Break s_pZ) { return t - rem(t - s_pZ.t, s_pZ.p); }

    struct LiraTerm {
      Term self;
      Var x;
      Term lim;
      Numeral sslp;
      Numeral oslp;
      Numeral per;
      Numeral deltaY;
      Term distYminus;
      Term distYplus() { return distYminus + deltaY; }
      std::vector<Break> breaks;
      Term distXplus();
      Term distXminus();
      Numeral deltaX();

      bool lim_pos_inf();
      bool lim_neg_inf();

      Term lim_at(Term x0) { return Term { x0.config, x0.config->subs(lim.inner, x.inner, x0.inner), }; }
      Term dseg(Term x0) { return -(sslp * x0) + lim_at(x0); }
      Term zero(Term x0) { return x0 - lim_at(x0) / sslp; }
      bool periodic() { return oslp == 0; }
    };

    template<class IfVar, class IfOne, class IfMul, class IfAdd, class IfFloor>
    auto matchTerm(Term t, 
        IfVar   if_var, 
        IfOne   if_one, 
        IfMul   if_mul, 
        IfAdd   if_add, 
        IfFloor if_floor
        ) -> decltype(auto) {
      return _config.matchTerm(t.inner,
        [&](auto x) { return if_var(CVar<Config>{ &_config, x }); }, 
        [&]() { return if_one(); }, 
        [&](auto l, auto r) { return if_mul(CNumeral<Config>{&_config, l},CTerm<Config>{&_config, r}); }, 
        [&](auto l, auto r) { return if_add(CTerm<Config>{&_config, l},CTerm<Config>{&_config, r}); }, 
        [&](auto x) { return if_floor(CTerm<Config>{&_config, x}); }
           );
    }

    Numeral numeral(int i)  { return CNumeral<Config> { &_config, _config.numeral(i)}; }
    Term    term(Numeral n) { return CTerm<Config>    { &_config, _config.term(n.inner) }; }
    Term    term(Var v)     { return CTerm<Config>    { &_config, _config.term(v.inner) }; }
    Term    term(int i)     { return term(numeral(i)); }

    enum class Bound {
      Open,
      Closed,
    };

    auto intersectGrid(Break s_pZ, Bound l, Term t, Numeral k, Bound r) {
      auto p = s_pZ.p;
      auto start = [&]() {
        switch(l) {
          case Bound::Open:   return grid_floor(t + p, s_pZ);
          case Bound::Closed: return grid_ceil(t, s_pZ);
        };
      }();
      return iter::nat_iter(numeral(0))
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
          .sslp = numeral(y == x ? 1 : 0),
          .oslp = numeral(y == x ? 1 : 0),
          .per = numeral(0),
          .deltaY = numeral(0),
          .distYminus = y == x ? term(0) 
                               : term(y),
          .breaks = std::vector<Break>(),
        }; }, 

        /* numeral 1 */ [&]() { return LiraTerm {
          .self = self,
          .x = x,
          .lim = self,
          .sslp = numeral(0),
          .oslp = numeral(0),
          .per = numeral(0),
          .deltaY = numeral(0),
          .distYminus = term(numeral(1)),
          .breaks = std::vector<Break>(),
        }; }, 
        /* k * t */ [&](auto k, auto t) { 
          auto rec = analyse(t, x);
          return LiraTerm {
            .self = self,
            .x = x,
            .lim = k * rec.lim,
            .sslp = k * rec.sslp,
            .oslp = k * rec.sslp,
            .per = rec.per,
            .deltaY = abs(k) * rec.deltaY,
            .distYminus = k >= 0 ? k * rec.distYminus
                                 : k * rec.distYplus(),
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
            .lim = rec_l.lim + rec_r.lim,
            .sslp = rec_l.sslp + rec_r.sslp,
            .oslp = rec_l.sslp + rec_r.sslp,
            .per = rec_l.per == 0 ? rec_r.per
                 : rec_r.per == 0 ? rec_l.per
                 : lcm(rec_l.per, rec_r.per),
            .deltaY = rec_l.deltaY + rec_r.deltaY,
            .distYminus = rec_l.distYminus + rec_r.distYminus,
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
                 : num(rec.per) * den(rec.oslp),
            .deltaY = rec.deltaY + 1,
            .distYminus = rec.distYminus - 1,
            .breaks = std::vector<Break>(),
          }; 
          if (rec.sslp == 0) {
            out.breaks = std::move(rec.breaks);
          } else if (rec.breaks.empty()) {
            out.breaks.push_back(Break {rec.zero(term(0)), out.per});
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
      static VirtualTerm plusEpsilon(VirtualTerm t);
      static VirtualTerm plain(Term t);
      static VirtualTerm periodic(Break b);
      static VirtualTerm periodic(Term b, Numeral p);
    };

#define if_then_(x, y) if_then([&]() { return x; }, [&]() { return y; }) 
#define else_if_(x, y) .else_if([&]() { return x; }, [&]() { return y; })
#define else____(x) .else_([&]() { return x; })
#define else_is_(x,y) .else_([&]() { SASSERT(x); return y; })

    auto elim_set(Var const& x, Literal const& lit)
    {
      auto t = analyse(lit.term(), x);
      auto symbol = lit.symbol();
      auto isIneq = [](auto symbol) { return (symbol == PredSymbol::Geq || symbol == PredSymbol::Gt); };
      using VT = VirtualTerm;

      return iter::if_then_(t.breaks.empty(), 
                           iter::if_then_(t.sslp == 0              , iter::vals(VT::minusInf()))
                                 else_if_(symbol == PredSymbol::Neq, iter::vals(VT::minusInf(), VT::plusEpsilon(t.zero(term(0)))))
                                 else_if_(symbol == PredSymbol:: Eq, iter::vals(VT::plain(t.zero(term(0)))))
                                 else_if_(t.sslp < 0               , iter::vals(VT::minusInf()))
                                 else_if_(symbol == PredSymbol::Geq, iter::vals(VT::plain(t.zero(term(0)))))
                                 else_is_(symbol == PredSymbol::Gt , iter::vals(VT::plusEpsilon(t.zero(term(0))))))

                   else_is_(!t.breaks.empty(), [&]() { 
                       auto ebreak       = [&]() { return 
                         iter::if_then_(t.periodic(), 
                                        iter::array_ptr(t.breaks) 
                                          | iter::map([&](auto* b) { return VT::periodic(*b); }) )

                               else____(iter::array_ptr(t.breaks) 
                                          | iter::flat_map([&](auto* b) { return intersectGrid(*b, Bound::Open, t.distXminus(), t.deltaX(), Bound::Open); })
                                          | iter::map([](auto t) { return VT::plain(t); }) )
                       ; };

                       auto breaks_plus_epsilon = [&]() { return iter::array_ptr(t.breaks) | iter::map([](auto* b) { return VT::plusEpsilon(VT::periodic(*b)); }); };

                       auto ezero = [&]() { return 
                          iter::if_then_(t.periodic(), 
                                         iter::array_ptr(t.breaks) 
                                           | iter::map([&](auto* b) { return VT::periodic(t.zero(b->t), b->p); }))

                                else_if_(t.oslp == t.sslp,
                                         iter::array_ptr(t.breaks) 
                                           | iter::map([&](auto* b) { return VT::plain(t.zero(b->t)); }))

                                else____(iter::array_ptr(t.breaks) 
                                           | iter::flat_map([&](auto* b) { return intersectGrid(Break { .t=t.zero(b->t), .p=(1 - t.oslp / t.sslp) },
                                                                                                Bound::Open, t.distXminus(), t.deltaX(), Bound::Open); })
                                           | iter::map([&](auto t) { return VT::plain(t); }))
                                         
                       ; };
                       auto eseg         = [&]() { return 
                           iter::if_then_(t.sslp == 0 || ( t.sslp < 0 && isIneq(symbol)), 
                                          breaks_plus_epsilon())
                                 else_if_(t.sslp >  0 && symbol == PredSymbol::Geq, iter::concat(breaks_plus_epsilon(), ezero()))
                                 else_if_(t.sslp >  0 && symbol == PredSymbol::Gt , iter::concat(breaks_plus_epsilon(), ezero() | iter::map([](auto x) { return VT::plusEpsilon(x); })))
                                 else_if_(t.sslp != 0 && symbol == PredSymbol::Neq, iter::concat(breaks_plus_epsilon(), ezero() | iter::map([](auto x) { return VT::plusEpsilon(x); })))
                                 else_is_(t.sslp != 0 && symbol == PredSymbol::Eq,  ezero())
                       ; };
                       auto ebound_plus  = [&]() { return 
                           iter::if_then_(t.lim_pos_inf(), iter::vals(VT::plain(t.distXplus()), VT::plusEpsilon(t.distXplus())))
                                 else____(                 iter::vals(VT::plain(t.distXplus())                                         )); };

                       auto ebound_minus = [&]() { return 
                           iter::if_then_(t.lim_neg_inf(), iter::vals(VT::plain(t.distXplus()), VT::minusInf()))
                                 else____(                 iter::vals(VT::plain(t.distXplus()))) ; };

                       return iter::if_then_(t.periodic(), iter::concat(ebreak(), eseg()))
                                    else____(              iter::concat(ebreak(), eseg(), ebound_plus(), ebound_minus()));
                   }());

    }

    auto elim_set(Var const& x, Literals const& lits)
    {
      return iter::array(lits) 
        | iter::flat_map([&](auto lit) { return elim_set(x, lit); });
    }

    class VsubsIter {
    public:
      std::optional<Literals> next();
    };

    VsubsIter vsubs(Literals const& lits, Var const& x, VirtualTerm& term);


    auto quantifier_elimination(Var const& x, Literals const& lits)
    {
      return elim_set(x, lits)
        | iter::flat_map([&](auto t) { return vsubs(lits, x, t); });
    }

    auto quantifier_elimination(typename Config::Var const& x, typename Config::Literals const& lits)
    {
      return quantifier_elimination(CVar<Config> { &_config, x }, CLiterals<Config> { &_config, lits });
    }
  };


}
