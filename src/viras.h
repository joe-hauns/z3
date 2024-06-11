#include <optional>
#include <iostream>

#pragma once

namespace viras {

  template<class T>
  T dummyVal();

  namespace iter {

    template<class T> struct opt_value_type;
    template<template<class> class Tmplt, class T> struct opt_value_type<Tmplt<T>> { using type = T; };

    template<class I>
    // using value_type = typename opt_value_type<decltype(((I*)0)->next())>::type;
    using value_type = typename decltype(((I*)0)->next())::value_type;


    template<class Op>
    struct IterCombinator {
      Op self;
      template<class I>
      friend auto operator|(I iter, IterCombinator self) -> decltype(auto) 
      { return (self.self)(iter); }

      template<class Op2>
      auto compose(IterCombinator<Op2> other) {
        return iterCombinator([self = std::move(this->self), other = std::move(other)](auto iter){ return (other.self)((self)(std::move(iter))); });
      }
    };

    template<class O>
    constexpr IterCombinator<O> iterCombinator(O o) 
    { return IterCombinator<O>{ std::move(o) }; }

    constexpr auto foreach = [](auto f) {
      return iterCombinator([f = std::move(f)](auto iter) {
        while (true) {
          auto x = iter.next();
          if (x) {
            f(*x);
          } else {
            break;
          }
        }
      });
    };


    constexpr auto fold = [](auto f) {
      return iterCombinator([f = std::move(f)](auto iter) -> std::optional<decltype(f(*iter.next(), *iter.next()))> {
          auto fst = iter.next();
          if (fst) {
            auto res = *fst;
            for (auto x = iter.next(); x; x = iter.next()) {
              res = f(res, *x);
            }
            return res;
          } else {
            return {};
          }
      });
    };


    constexpr auto all = [](auto f) {
      return iterCombinator([f = std::move(f)](auto iter) {
          for (auto x = iter.next(); x; x = iter.next()) {
            if (!f(*x))
              return false;
          }
          return true;
      });
    };


    constexpr auto any = [](auto f) {
      return iterCombinator([f = std::move(f)](auto iter) {
          for (auto x = iter.next(); x; x = iter.next()) {
            if (f(*x))
              return true;
          }
          return false;
      });
    };

    constexpr auto find = [](auto f) {
      return iterCombinator([f = std::move(f)](auto iter) -> std::optional<value_type<decltype(iter)>> {
          using res = std::optional<value_type<decltype(iter)>>;
          for (auto x = iter.next(); x; x = iter.next()) {
            if (f(*x))
              return res(*x);
          }
          return res();
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
      return iterCombinator([f = std::move(f)](auto i) {
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
      return iterCombinator([f = std::move(f)](auto i) {
        return FilterIter<decltype(i), decltype(f)>{std::move(i),std::move(f)};
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
      return iterCombinator([f = std::move(f)](auto i) {
        return TakeWhileIter<decltype(i), decltype(f)>{ std::move(i),std::move(f), false, };
      });
    };

    template<class I>
    struct FlattenIter {
      I outer;
      std::optional<value_type<I>> inner;
      bool init;

      FlattenIter(I i): outer(std::move(i)), inner(), init(false) {}

      std::optional<value_type<value_type<I>>> next()
      {
        if (!init) {
          inner.~decltype(inner)();
          new(&inner) decltype(inner)(outer.next());
          init = true;
        }
        while (inner) {
          auto next = inner->next();
          if (next) 
            return next;
          else {
            inner.~decltype(inner)();
            new(&inner) decltype(inner)(outer.next());
          }
        }
        return {};
      }
    };

    constexpr auto flatten = iterCombinator([](auto i) {
      return FlattenIter<decltype(i)>(i);
    });

    constexpr auto min = iterCombinator([](auto iter) {
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
        if (lower < upper) return std::optional(lower++);
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

    template<class A>
    struct DummyIter {
      auto next() -> std::optional<A>;
    };

    template<class A>
    constexpr auto dummy() 
    { return DummyIter<A>{}; }


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

  template<class Config>
  class Viras {

  // START OF SYNTAX SUGAR STUFF

    template<class T>
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

    struct CTerm : public WithConfig<typename Config::Term> { };
    struct CNumeral : public WithConfig<typename Config::Numeral> {};
    struct CVar : public WithConfig<typename Config::Var> {};
    struct CLiteral : public WithConfig<typename Config::Literal> {
      CTerm term() const
      { return CTerm {this->config, this->config->term_of_literal(this->inner)}; }

      PredSymbol symbol() const
      { return this->config->symbol_of_literal(this->inner); }
    };
    struct CLiterals : public WithConfig<typename Config::Literals> {
      auto size() const { return this->config->literals_size(this->inner); }
      auto operator[](size_t idx) const { 
        return CLiteral { this->config, this->config->literals_get(this->inner, idx) }; 
      }
    };

    ///////////////////////////////////////
    // PRIMARY OPERATORS
    ///////////////////////////////////////

    friend CTerm operator+(CTerm lhs, CTerm rhs) 
    {
      SASSERT(lhs.config == rhs.config);
      return CTerm {lhs.config, lhs.config->add(lhs.inner, rhs.inner)};
    }

    friend CTerm operator*(CNumeral lhs, CTerm rhs) 
    {
      SASSERT(lhs.config == rhs.config);
      return CTerm {lhs.config, lhs.config->mul(lhs.inner, rhs.inner)};
    }

    friend CNumeral operator*(CNumeral lhs, CNumeral rhs) 
    {
      SASSERT(lhs.config == rhs.config);
      return CNumeral {lhs.config, lhs.config->mul(lhs.inner, rhs.inner)};
    }

    ///////////////////////////////////////
    // LIFTED OPERATORS
    ///////////////////////////////////////

    friend CNumeral operator+(CNumeral lhs, CNumeral rhs) 
    { return CNumeral { lhs.config, lhs.config->add(lhs.inner, rhs.inner)}; }

    friend CTerm operator+(CNumeral lhs, CTerm rhs) 
    { return CTerm { rhs.config, lhs.config->term(lhs.inner)} + rhs; }

    friend CTerm operator+(CTerm lhs, CNumeral rhs) 
    { return lhs + CTerm { rhs.config, rhs.config->term(rhs.inner), }; }

#define LIFT_NUMRAL_TO_TERM_L(function)                                                   \
    friend auto function(CNumeral lhs, CTerm rhs)                                         \
    { return function(CTerm {lhs.config, lhs.config->term(lhs)}, rhs); }                  \

#define LIFT_NUMRAL_TO_TERM_R(function)                                                   \
    friend auto function(CTerm lhs, CNumeral rhs)                                         \
    { return function(lhs, CTerm {rhs.config, rhs.config->term(rhs)}); }                  \

#define LIFT_INT_TO_NUMERAL(function, CType)                                              \
    LIFT_INT_TO_NUMERAL_L(function, CType)                                                \
    LIFT_INT_TO_NUMERAL_R(function, CType)                                                \


#define LIFT_INT_TO_NUMERAL_L(function, CType)                                            \
    friend auto function(int lhs, CType rhs)                                              \
    { return function(CNumeral {rhs.config, rhs.config->numeral(lhs)}, rhs); }            \

#define LIFT_INT_TO_NUMERAL_R(function, CType)                                            \
    friend auto function(CType lhs, int rhs)                                              \
    { return function(lhs, CNumeral {lhs.config, lhs.config->numeral(rhs)}); }            \

#define LIFT_INT_TO_NUMERAL(function, CType)                                              \
    LIFT_INT_TO_NUMERAL_L(function, CType)                                                \
    LIFT_INT_TO_NUMERAL_R(function, CType)                                                \

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

    friend CTerm operator/(CTerm lhs, CNumeral rhs) 
    { return (1 / rhs) * lhs; }

    friend CNumeral operator/(CNumeral lhs, CNumeral rhs) 
    { return CNumeral{ rhs.config, rhs.config->inverse(rhs.inner) } * lhs; }

    LIFT_INT_TO_NUMERAL_R(operator/, CTerm)
    LIFT_INT_TO_NUMERAL(operator/, CNumeral)

     // MINUS

#define DEF_UMINUS(CType)                                                                 \
    friend CType operator-(CType x)                                                       \
    { return -1 * x; }                                                                    \

  DEF_UMINUS(CNumeral)
  DEF_UMINUS(CTerm)

#define DEF_BMINUS(T1, T2)                                                                \
    friend auto operator-(T1 x, T2 y)                                                     \
    { return x + -y; }                                                                    \

  DEF_BMINUS(CNumeral, CNumeral)
  DEF_BMINUS(CTerm   , CNumeral)
  DEF_BMINUS(CNumeral, CTerm   )
  DEF_BMINUS(CTerm   , CTerm   )

     // ABS

    friend CNumeral abs(CNumeral x) 
    { return x < 0 ? -x : x; }

    friend CNumeral num(CNumeral x)
    { return CNumeral { x.config, x.config->num(x.inner) }; }

    friend CNumeral den(CNumeral x) 
    { return CNumeral { x.config, x.config->den(x.inner) }; }

    friend CNumeral lcm(CNumeral l, CNumeral r)
    { 
      auto cn = [&](auto x) { return CNumeral { l.config, x }; };
      return cn(l.config->lcm(num(l).inner, num(r).inner)) 
           / cn(l.config->gcd(den(l).inner, den(r).inner));  
    }

     // floor
    friend CTerm floor(CTerm x) 
    { return CTerm { x.config, x.config->floor(x.inner), }; }

    friend CTerm ceil(CTerm x) 
    { return -floor(-x); }


    friend CNumeral floor(CNumeral x) 
    { return CNumeral { x.config, x.config->floor(x.inner), }; }

    // COMPARISIONS

    friend bool operator<=(CNumeral lhs, CNumeral rhs) 
    { return lhs.config->leq(lhs.inner, rhs.inner); }

    friend bool operator<(CNumeral lhs, CNumeral rhs) 
    { return lhs.config->less(lhs.inner, rhs.inner); }

    friend bool operator>=(CNumeral lhs, CNumeral rhs) 
    { return rhs <= lhs; }

    friend bool operator>(CNumeral lhs, CNumeral rhs) 
    { return rhs < lhs; }

#define INT_CMP(OP)                                                                       \
    friend bool operator OP (CNumeral lhs, int rhs)                                       \
    { return lhs OP CNumeral  { lhs.config, lhs.config->numeral(rhs), }; }                \
                                                                                          \
    friend bool operator OP (int lhs, CNumeral rhs)                                       \
    { return CNumeral  { rhs.config, rhs.config->numeral(lhs), } OP rhs; }                \

    friend CTerm quot(CTerm t, CNumeral p) { return floor(t / p); }
    friend CTerm rem(CTerm t, CNumeral p) { return t - p * quot(t, p); }

    friend CTerm subs(CTerm t, CVar v, CTerm by) {
      return Term { t.config, t.config->subs(t.inner, v.inner, by.inner), };
    }

    
    // END OF SYNTAX SUGAR STUFF


    using Term     = CTerm;
    using Numeral  = CNumeral;
    using Var      = CVar;
    using Literals = CLiterals;
    using Literal  = CLiteral;

    Config _config;
  public:

    template<class... Args>
    Viras(Args... args) : _config(args...) { };
    ~Viras() {};

    struct Break {
      Term t;
      Numeral p;
    };

    template<class L, class R> struct Add { L l; R r; };
    template<class L, class R> struct Mul { L l; R r; };

    template<class L, class R>
    Term eval(Add<L, R> x) { return _config.add(eval(x.l), eval(x.r)); };

    template<class T>
    struct Expr : public std::variant<T> { };

    friend CTerm grid_ceil (CTerm t, Break s_pZ) { return t + rem(s_pZ.t - t, s_pZ.p); }
    friend CTerm grid_floor(CTerm t, Break s_pZ) { return t - rem(t - s_pZ.t, s_pZ.p); }

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
      Term distXminus() { 
        auto distY = oslp > 0 ? distYplus() : distYminus;
        return -(1 / oslp) * distY;
      }
      Term distXplus() { return distXminus() + deltaX(); }
      Numeral deltaX() { return abs(1 / oslp) * deltaY; }
      Term lim_at(Term x0) const { return subs(lim, x, x0); }
      Term dseg(Term x0) const { return -(sslp * x0) + lim_at(x0); }
      Term zero(Term x0) const { return x0 - lim_at(x0) / sslp; }
      bool periodic() const { return oslp == 0; }
    };


    struct LiraLiteral {
      LiraTerm term;
      PredSymbol symbol;

      bool lim_pos_inf() const { return lim_inf(/* pos */ true); }
      bool lim_neg_inf() const { return lim_inf(/* pos */ false); }
      bool periodic() const { return term.periodic(); }

      bool lim_inf(bool positive) const
      { 
        SASSERT(term.oslp != 0);
        switch (symbol) {
        case PredSymbol::Gt:
        case PredSymbol::Geq: return positive ? term.oslp > 0 
                                              : term.oslp < 0;
        case PredSymbol::Neq: return true;
        case PredSymbol::Eq: return false;
        }
      }


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
        [&](auto x) { return if_var(CVar{ &_config, x }); }, 
        [&]() { return if_one(); }, 
        [&](auto l, auto r) { return if_mul(CNumeral{&_config, l},CTerm{&_config, r}); }, 
        [&](auto l, auto r) { return if_add(CTerm{&_config, l},CTerm{&_config, r}); }, 
        [&](auto x) { return if_floor(CTerm{&_config, x}); }
           );
    }

    Numeral numeral(int i)  { return CNumeral { &_config, _config.numeral(i)}; }
    Term    term(Numeral n) { return CTerm    { &_config, _config.term(n.inner) }; }
    Term    term(Var v)     { return CTerm    { &_config, _config.term(v.inner) }; }
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

    LiraLiteral analyse(Literal self, Var x) 
    { return LiraLiteral { analyse(self.term(), x), self.symbol() }; }





    enum class Infinity {
      Plus,
      Minus,
      Zero,
    };

    struct Infty {
      bool positive;
      Infty operator-() const 
      { return Infty { .positive = !positive, }; }

      friend std::ostream& operator<<(std::ostream& out, Infty const& self)
      { 
        if (self.positive) {
          return out << "∞";
        } else {
          return out << "-∞";
        }
      }
    };

    struct Epsilon {
      friend std::ostream& operator<<(std::ostream& out, Epsilon const& self)
      { return out << "ε"; }
    };
    static constexpr Infty infty { .positive = true, };
    static constexpr Epsilon epsilon;

    struct VirtualTerm {
      std::optional<Term> term;
      std::optional<Epsilon> epsilon;
      std::optional<Numeral> period;
      std::optional<Infty> infty;
    public:
      VirtualTerm(
        std::optional<Term> term,
        std::optional<Epsilon> epsilon,
        std::optional<Numeral> period,
        std::optional<Infty> infinity) 
        : term(std::move(term))
        , epsilon(std::move(epsilon))
        , period(std::move(period))
        , infty(std::move(infinity))
      {}

      VirtualTerm(Break t_pZ) : VirtualTerm(std::move(t_pZ.t), {}, std::move(t_pZ.p), {}) {}
      VirtualTerm(Infty infty) : VirtualTerm({}, {}, {}, infty) {}
      VirtualTerm(Term t) : VirtualTerm(std::move(t), {}, {}, {}) {}

      friend VirtualTerm operator+(VirtualTerm const& t, Epsilon const);

      static VirtualTerm periodic(Term b, Numeral p) { return VirtualTerm(std::move(b), {}, std::move(p), {}); }

      friend std::ostream& operator<<(std::ostream& out, VirtualTerm const& self)
      { 
        bool fst = true;
#define OUTPUT(field)                                                                     \
        if (fst) { out << field; fst = false; }                                           \
        else { out << " + " << field; }                                                   \

        if (self.term) { OUTPUT(*self.term) }
        if (self.epsilon) { OUTPUT(*self.epsilon) }
        if (self.period) { OUTPUT(*self.period << " ℤ") }
        if (self.infinity) { OUTPUT(*self.infty) }
        if (fst) { out << "0"; fst = false; }
        return out; 
      }
    };

    friend VirtualTerm operator+(Term const& t, Epsilon const) 
    { return VirtualTerm(t) + epsilon; }

    friend VirtualTerm operator+(VirtualTerm const& t, Epsilon const) 
    { 
      VirtualTerm out = t;
      out.epsilon = std::optional<Epsilon>(epsilon);
      return out; 
    }

    friend VirtualTerm operator+(Term const& t, Infty const& infty) 
    { return VirtualTerm(t) + infty; }

    friend VirtualTerm operator+(VirtualTerm const& t, Infty const& infty) 
    { 
      VirtualTerm out = t;
      out.infty = std::optional<Infty>(infty);
      return out; 
    }


#define if_then_(x, y) if_then([&]() { return x; }, [&]() { return y; }) 
#define else_if_(x, y) .else_if([&]() { return x; }, [&]() { return y; })
#define else____(x) .else_([&]() { return x; })
#define else_is_(x,y) .else_([&]() { SASSERT(x); return y; })

    auto elim_set(Var const& x, Literal const& l)
    {
      auto lit = analyse(l, x);
      auto t = lit.term;
      auto symbol = lit.symbol;
      auto isIneq = [](auto symbol) { return (symbol == PredSymbol::Geq || symbol == PredSymbol::Gt); };
      using VT = VirtualTerm;

      return iter::if_then_(t.breaks.empty(), 
                           iter::if_then_(t.sslp == 0              , iter::vals<VT>(-infty))
                                 else_if_(symbol == PredSymbol::Neq, iter::vals<VT>(-infty, t.zero(term(0)) + epsilon))
                                 else_if_(symbol == PredSymbol:: Eq, iter::vals<VT>(t.zero(term(0))))
                                 else_if_(t.sslp < 0               , iter::vals<VT>(-infty))
                                 else_if_(symbol == PredSymbol::Geq, iter::vals<VT>(t.zero(term(0))))
                                 else_is_(symbol == PredSymbol::Gt , iter::vals<VT>(t.zero(term(0)) + epsilon)))

                   else_is_(!t.breaks.empty(), [&]() { 
                       auto ebreak       = [&]() { return 
                         iter::if_then_(t.periodic(), 
                                        iter::array_ptr(t.breaks) 
                                          | iter::map([&](auto* b) { return VT(*b); }) )

                               else____(iter::array_ptr(t.breaks) 
                                          | iter::flat_map([&](auto* b) { return intersectGrid(*b, Bound::Open, t.distXminus(), t.deltaX(), Bound::Open); })
                                          | iter::map([](auto t) { return VT(t); }) )
                       ; };

                       auto breaks_plus_epsilon = [&]() { return iter::array_ptr(t.breaks) | iter::map([](auto* b) { return VT(*b) + epsilon; }); };

                       auto ezero = [&]() { return 
                          iter::if_then_(t.periodic(), 
                                         iter::array_ptr(t.breaks) 
                                           | iter::map([&](auto* b) { return VT::periodic(t.zero(b->t), b->p); }))

                                else_if_(t.oslp == t.sslp,
                                         iter::array_ptr(t.breaks) 
                                           | iter::map([&](auto* b) { return VT(t.zero(b->t)); }))

                                else____(iter::array_ptr(t.breaks) 
                                           | iter::flat_map([&](auto* b) { return intersectGrid(Break { .t=t.zero(b->t), .p=(1 - t.oslp / t.sslp) },
                                                                                                Bound::Open, t.distXminus(), t.deltaX(), Bound::Open); })
                                           | iter::map([&](auto t) { return VT(t); }))
                                         
                       ; };
                       auto eseg         = [&]() { return 
                           iter::if_then_(t.sslp == 0 || ( t.sslp < 0 && isIneq(symbol)), 
                                          breaks_plus_epsilon())
                                 else_if_(t.sslp >  0 && symbol == PredSymbol::Geq, iter::concat(breaks_plus_epsilon(), ezero()))
                                 else_if_(t.sslp >  0 && symbol == PredSymbol::Gt , iter::concat(breaks_plus_epsilon(), ezero() | iter::map([](auto x) { return x + epsilon; })))
                                 else_if_(t.sslp != 0 && symbol == PredSymbol::Neq, iter::concat(breaks_plus_epsilon(), ezero() | iter::map([](auto x) { return x + epsilon; })))
                                 else_is_(t.sslp != 0 && symbol == PredSymbol::Eq,  ezero())
                       ; };
                       auto ebound_plus  = [&]() { return 
                           iter::if_then_(lit.lim_pos_inf(), iter::vals<VT>(t.distXplus(), t.distXplus() + epsilon))
                                 else____(                   iter::vals<VT>(t.distXplus()                         )); };

                       auto ebound_minus = [&]() { return 
                           iter::if_then_(lit.lim_neg_inf(), iter::vals<VT>(t.distXplus(), -infty))
                                 else____(                   iter::vals<VT>(t.distXplus()        )); };

                       return iter::if_then_(t.periodic(), iter::concat(ebreak(), eseg()))
                                    else____(              iter::concat(ebreak(), eseg(), ebound_plus(), ebound_minus()));
                   }());

    }

    auto elim_set(Var const& x, Literals const& lits)
    {
      return iter::array(lits) 
        | iter::flat_map([&](auto lit) { return elim_set(x, lit); });
    }



    class LiteralIter {
    public:
      std::optional<Literal> next();
    };

    Literal literal(Term t, PredSymbol s) 
    { return CLiteral { &_config, _config.create_literal(t.inner, s), }; }

    Literal literal(bool b) { return CLiteral { &_config, _config.create_literal(b), }; }

    // TODO nicer?
    auto vt_term(VirtualTerm vt) { return vt.term ? *vt.term : term(0); };

    Literal vsubs_aperiodic0(LiraLiteral const& lit, Var const& x, VirtualTerm vt) {
      auto& s = lit.term;
      auto symbol = lit.symbol;
      SASSERT(!vt.period);
      if (vt.infty) {
        /* case 3 */
        if (s.periodic()) {
          vt.infty = {};
          return vsubs_aperiodic0(lit, x, vt);
        } else {
          return literal(lit.lim_inf(vt.infty->positive));
        }
      } else if (vt.epsilon) {
        switch(symbol) {
          case PredSymbol::Eq:
          case PredSymbol::Neq: {
            /* case 4 */
            if (s.sslp != 0) {
              return symbol == PredSymbol::Eq ? literal(false) : literal(true);
            } else {
              return literal(s.lim_at(vt_term(vt)), symbol);
            }
          }
          case PredSymbol::Geq:
          case PredSymbol::Gt: {
            /* case 5 */
            return literal(s.lim_at(vt_term(vt)), 
                  s.sslp > 0 ? PredSymbol::Geq
                : s.sslp < 0 ? PredSymbol::Gt
                :              symbol);
          }
        }
      } else {
        SASSERT(!vt.epsilon && !vt.infty && !vt.period);
        return literal(subs(s.self, x, vt_term(vt)), symbol);
      }
    }


    auto vsubs_aperiodic1(std::vector<LiraLiteral> const& lits, Var const& x, VirtualTerm const& term) {
      SASSERT(!term.period);
          /* case 2 */
      return iter::array(lits) 
        | iter::map([&](auto lit) { return vsubs_aperiodic0(lit, x, term); });
    }

    auto vsubs(Literals const& ls, Var const& x, VirtualTerm const& term) {
      SASSERT(!term.infty || !term.period);
      std::vector<LiraLiteral> lits;
      lits.reserve(ls.size());
      iter::array(ls) | iter::foreach([&](auto l) { return lits.push_back(analyse(l, x)); });
      return iter::if_then_(term.period, ([&](){
                              /* case 1 */
                              auto all_lits = [&](auto p) { return iter::array(lits) | iter::all(p); };
                              Numeral lambda = *(iter::array(lits)
                                             | iter::filter([&](auto L) { return L->periodic(); })
                                             | iter::map([&](auto L) { return L.term.per; })
                                             | iter::fold([](auto l, auto r) { return lcm(l, r); }));
                              auto grid = Break { vt_term(term), *term.period };
                              auto fin = 
                                iter::if_then_(all_lits([&](auto l) { return l.lim_pos_inf(); }), 
                                               intersectGrid(grid, Bound::Closed, vt_term(term), lambda, Bound::Open)
                                                | iter::map([](auto s) { return s + infty; }))
                                      else_if_(all_lits([&](auto l) { return l.lim_neg_inf(); }), 
                                               intersectGrid(grid, Bound::Closed, vt_term(term), lambda, Bound::Open)
                                                | iter::map([](auto s) { return s + -infty; }))
                                      else_if_(iter::array(lits) | iter::any([](auto l) { return l.symbol == PredSymbol::Eq; }), 
                                          [&](){
                                                // TODO which one of these to find? The one with the smallest deltaX!!
                                                auto L = *(iter::array(lits) | iter::find([](auto l) { return l.symbol == PredSymbol::Eq; }));
                                                return intersectGrid(grid, Bound::Closed, L.term.distXminus(), L.term.deltaX(), Bound::Closed)
                                                  | iter::map([](auto x) { return VirtualTerm(x); });
                                          }())
                                      else____(
                                            iter::array(lits) 
                                          | iter::filter([&](auto L) { return !L->periodic() && L->lim_neg_inf() == false; })
                                          | iter::flat_map([&](auto L) {
                                                return intersectGrid(Break { vt_term(term), *term.period }, Bound::Closed, L.term.distXminus(), L.term.deltaX() + lambda, Bound::Closed);
                                            })
                                          | iter::map([](auto t) { return VirtualTerm(t); })

                                          );

                              return std::move(fin) | iter::map([&](auto t) { return vsubs_aperiodic1(lits, x, t); });
                            }()))
                   else____(iter::vals(vsubs_aperiodic1(lits, x, term)));
    }


    auto quantifier_elimination(Var const& x, Literals const& lits)
    {
      return elim_set(x, lits)
        | iter::flat_map([&](auto t) { return vsubs(lits, x, t); });
    }

    auto quantifier_elimination(typename Config::Var const& x, typename Config::Literals const& lits)
    {
      return quantifier_elimination(CVar { &_config, x }, CLiterals { &_config, lits });
    }
  };


}
