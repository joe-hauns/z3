/*++
Copyright (c) 2015 Microsoft Corporation

Module Name:

    mbp_plugin.h

Abstract:

    Model-based projection utilities

Author:

    Nikolaj Bjorner (nbjorner) 2015-5-28

Revision History:


--*/

#pragma once

#include "ast/ast.h"
#include "util/params.h"
#include "model/model.h"
#include "math/simplex/model_based_opt.h"


namespace mbp {

    struct cant_project {};

    struct def {
        expr_ref var, term;
        def(const expr_ref& v, expr_ref& t): var(v), term(t) {}
    };

    // TODO joe
    class project_plugin {
        ast_manager&     m;
        expr_mark        m_visited;
        ptr_vector<expr> m_to_visit;
        expr_mark        m_bool_visited;
        expr_mark        m_non_ground;
        expr_ref_vector  m_cache, m_args, m_pure_eqs;

        bool reduce(model_evaluator& eval, model& model, expr* fml, expr_ref_vector& fmls);
        void extract_bools(model_evaluator& eval, expr_ref_vector& fmls, unsigned i, expr* fml, bool is_true);
        void visit_app(expr* e);
        bool visit_ite(model_evaluator& eval, expr* e, expr_ref_vector& fmls);
        bool visit_bool(model_evaluator& eval, expr* e, expr_ref_vector& fmls);
        bool is_true(model_evaluator& eval, expr* e);

        // over-approximation
        bool contains_uninterpreted(expr* v) { return true; }

        void push_back(expr_ref_vector& lits, expr* lit);

        void mark_non_ground(expr* e);

    public:
        project_plugin(ast_manager& m) :m(m), m_cache(m), m_args(m), m_pure_eqs(m) {}
        virtual ~project_plugin() = default;
        // Q: What is the supposed to do?
        virtual bool operator()(model& model, app* var, app_ref_vector& vars, expr_ref_vector& lits) { return false; }
        /**
           \brief partial solver.
        */
        // Q: What is the supposed to do in contrast to project?
        virtual bool solve(model& model, app_ref_vector& vars, expr_ref_vector& lits) { return false; }
        // Q: What are family ids?
        virtual family_id get_family_id() { return null_family_id; }

        // Q: What is the supposed to do? Esp in contrast to the other application operator?
        // the only implemntor is the aritmetic plugin which implements it as
        /*
        bool operator()(model& model, app_ref_vector& vars, expr_ref_vector& lits) 
        { 
          vector<def> defs;
          return m_imp->project(model, vars, lits, defs, false); // <- false = do not compute definitions
        };
         */
        virtual bool operator()(model& model, app_ref_vector& vars, expr_ref_vector& lits) { return false; };


        /**
           \brief project vars modulo model, return set of definitions for eliminated variables.
           - vars in/out: returns variables that were not eliminated
           - lits in/out: returns projected literals
           - returns set of definitions
             (TBD: in triangular form, the last definition can be substituted into definitions that come before)
        */
        virtual bool project(model& model, app_ref_vector& vars, expr_ref_vector& lits, vector<def>& defs) { return true; }

        /**
           \brief model based saturation. Saturates theory axioms to equi-satisfiable literals over EUF,
           such that 'shared' are not retained for EUF.
         */
        // Q: Which theory axioms? I don't get the explanation.
        // This is only implemented for arrays
        virtual void saturate(model& model, func_decl_ref_vector const& shared, expr_ref_vector& lits) {}


        /*
        * extract top-level literals
        */
        void extract_literals(model& model, app_ref_vector const& vars, expr_ref_vector& fmls);

        static expr_ref pick_equality(ast_manager& m, model& model, expr* t);
        static void erase(expr_ref_vector& lits, unsigned& i);

        static void mark_rec(expr_mark& visited, expr* e);
        static void mark_rec(expr_mark& visited, expr_ref_vector const& es);

        /**
        * mark sub-terms in e whether they contain a variable from vars.
        */
        void mark_non_ground(app_ref_vector const& vars, expr* e) {
            for (app* v : vars)
                m_non_ground.mark(v);
            mark_non_ground(e);
        }

        bool is_non_ground(expr* t) const { return m_non_ground.is_marked(t); }

    };
}

