/*++
Copyright (c) 2015 Microsoft Corporation

Module Name:

    qe_arith.cpp

Abstract:

    Interface to the implementation of the VIRAS Quantifier Elimination method from the paper TODO

Author:

    Johannes Schoisswohl (joe-hauns) 2024-6-5

--*/

#include "qe/mbp/mbp_viras.h"
#include "ast/ast_util.h"
#include "ast/arith_decl_plugin.h"
#include "ast/ast_pp.h"
#include "ast/expr_functors.h"
#include "ast/rewriter/expr_safe_replace.h"
#include "math/simplex/model_based_opt.h"
#include "model/model_evaluator.h"
#include "model/model_smt2_pp.h"
#include "model/model_v2_pp.h"
#include "util/debug.h"

namespace mbp {

    bool viras_project_plugin::operator()(model& model, app* var, app_ref_vector& vars, expr_ref_vector& lits) {
      // we can only project all variables at once
      // otherwise we'd might have to introduce a disjunction in the result `lits`
      return false;
      // ref_vector<expr, ast_manager> result(m);
      // auto viras = Viras<z3_viras_config>();
      // auto elim_set = viras.elim_set(var, lits);
      // while (elim_set.has_next()) {
      //   auto vterm = elim_set.next();
      //   auto disjuncts = viras.vsubs(lits, var, vterm);
      //   while (disjuncts.has_next()) {
      //     result.push_back(m.mk_and(disjuncts.next()));
      //   }
      // }
      // lits.shrink(0);
      // SASSERT(lits.size() == 0);
      // lits.push_back(m.mk_or(result));
      // return true;
    }

    bool viras_project_plugin::operator()(model& model, app_ref_vector& vars, expr_ref_vector& lits) {
      auto viras = Viras<z3_viras_config>(&m);
      vector<expr_ref_vector> cur_disj;
      cur_disj.push_back(lits);
      // invariant: `cur_disj` is a disjunction of conjunctions equivalent to `lits`
      {
        vector<expr_ref_vector> new_disj;
        // we eliminate var by var 
        // using the property `exists x. (A \/ B) <-> exists x.A \/ exists x.B`
        for (int i = 0; i < vars.size(); i++) {
          auto var = vars[i].get();
          for (auto conj : cur_disj) {
            viras.quantifier_elimination(var, conj)
              | iter::foreach([&](auto x) { new_disj.push_back(x); });
          }
          cur_disj.shrink(0);
          std::swap(cur_disj, new_disj);
        }
      }
      // `cur_disj` does not contain any variables from `vars` anymore
      // all other variables have a value interpreted in the model, 
      // thus we can evaluate every cur_disj to either true or false
      model_evaluator eval(model);
      eval.set_model_completion(true);
      lits.shrink(0);
      for (auto conj : cur_disj) {
        expr_ref result(m);
        auto success = eval.eval(m.mk_and(conj), result);
        SASSERT(success);
        SASSERT(m.is_true(result) || m.is_false(result));
        if (m.is_true(result)) {
          // some disjunct is true
          lits.push_back(m.mk_true());
          return true;
        }
      }
      // all disjuncts are false
      lits.push_back(m.mk_false());
      return true;
    }

    bool viras_project_plugin::project(model& model, app_ref_vector& vars, expr_ref_vector& lits, vector<def>& defs) {
      // instantiation cannot be done for virtual substitution because we'd have to put terms like `infty` or `1 + epsilon` into `defs`
      UNREACHABLE();
      return false;
    }

    family_id viras_project_plugin::get_family_id() {
      return arith_family_id;
    }

}
