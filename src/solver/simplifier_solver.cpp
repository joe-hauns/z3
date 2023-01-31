/*++
Copyright (c) 2023 Microsoft Corporation

Module Name:

    simplifier_solver.cpp

Abstract:

    Implements a solver with simplifying pre-processing.

Author:

    Nikolaj Bjorner (nbjorner) 2023-01-30

 Notes:

 - add translation for preprocess state.
   - If the pre-processors are stateful, they need to be properly translated.

--*/
#include "util/params.h"
#include "ast/ast_util.h"
#include "ast/rewriter/expr_safe_replace.h"
#include "ast/simplifiers/dependent_expr_state.h"
#include "ast/simplifiers/seq_simplifier.h"
#include "solver/solver.h"
#include "solver/simplifier_solver.h"
#include "solver/solver_preprocess.h"


class simplifier_solver : public solver {


    struct dep_expr_state : public dependent_expr_state {
        simplifier_solver& s;
        model_reconstruction_trail m_reconstruction_trail;
        dep_expr_state(simplifier_solver& s) :dependent_expr_state(s.m), s(s), m_reconstruction_trail(s.m, m_trail) {}
        ~dep_expr_state() override {}
        virtual unsigned qtail() const override { return s.m_fmls.size(); }
        dependent_expr const& operator[](unsigned i) override { return s.m_fmls[i]; }
        void update(unsigned i, dependent_expr const& j) override { 
            SASSERT(j.fml());  
            check_false(j.fml());
            s.m_fmls[i] = j; 
        }
        void add(dependent_expr const& j) override { check_false(j.fml()); s.m_fmls.push_back(j); }
        bool inconsistent() override { return s.m_inconsistent; }
        model_reconstruction_trail& model_trail() override { return m_reconstruction_trail; }
        std::ostream& display(std::ostream& out) const override {
            unsigned i = 0;
            for (auto const& d : s.m_fmls) {
                if (i > 0 && i == qhead())
                    out << "---- head ---\n";
                out << d << "\n";
                ++i;
            }
            m_reconstruction_trail.display(out);
            return out;
        }
        void check_false(expr* f) {
            if (s.m.is_false(f))
                s.set_inconsistent();
        }        
        void append(generic_model_converter& mc) { model_trail().append(mc); }
        void replay(unsigned qhead, expr_ref_vector& assumptions) { m_reconstruction_trail.replay(qhead, assumptions, *this); }
        void flatten_suffix() override {
            expr_mark seen;
            unsigned j = qhead();
            for (unsigned i = qhead(); i < qtail(); ++i) {
                expr* f = s.m_fmls[i].fml();
                if (seen.is_marked(f))
                    continue;
                seen.mark(f, true);
                if (s.m.is_true(f))
                    continue;
                if (s.m.is_and(f)) {
                    auto* d = s.m_fmls[i].dep();
                    for (expr* arg : *to_app(f))
                        add(dependent_expr(s.m, arg, nullptr, d));
                    continue;
                }
                if (i != j)
                    s.m_fmls[j] = s.m_fmls[i];
                ++j;
            }
            s.m_fmls.shrink(j);
        }
    };   

    ast_manager& m;
    solver_ref  s;
    vector<dependent_expr>      m_fmls;
    dep_expr_state              m_preprocess_state;
    seq_simplifier              m_preprocess;
    expr_ref_vector             m_assumptions;
    generic_model_converter_ref m_mc;
    bool                        m_inconsistent = false;

    void flush(expr_ref_vector& assumptions) {
        unsigned qhead = m_preprocess_state.qhead();
        if (qhead < m_fmls.size()) {
            for (expr* a : assumptions)
                m_preprocess_state.freeze(a);
            TRACE("solver", tout << "qhead " << qhead << "\n");
            m_preprocess_state.replay(qhead, assumptions);
            m_preprocess.reduce();
            if (!m.inc())
                return;
            m_preprocess_state.advance_qhead();
        }
        m_mc = alloc(generic_model_converter, m, "simplifier-model-converter");
        m_cached_mc = nullptr;
        m_preprocess_state.append(*m_mc);
        for (; qhead < m_fmls.size(); ++qhead)
            add_with_dependency(m_fmls[qhead]);
    }

    ptr_vector<expr> m_deps;
    void add_with_dependency(dependent_expr const& de) {
        if (!de.dep()) {
            s->assert_expr(de.fml());
            return;
        }
        m_deps.reset();
        m.linearize(de.dep(), m_deps);
        m_assumptions.reset();
        for (expr* d : m_deps) 
            m_assumptions.push_back(d);        
        s->assert_expr(de.fml(), mk_and(m_assumptions));
    }

    bool inconsistent() const {
        return m_inconsistent;
    }

    void set_inconsistent() {
        if (!m_inconsistent) {
            m_preprocess_state.m_trail.push(value_trail(m_inconsistent));
            m_inconsistent = true;
        }
    }

public:

    simplifier_solver(solver* s, simplifier_factory* fac) : 
        solver(s->get_manager()), 
        m(s->get_manager()), 
        s(s),
        m_preprocess_state(*this),
        m_preprocess(m, s->get_params(), m_preprocess_state),
        m_assumptions(m),
        m_proof(m)
    {
        if (fac) 
            m_preprocess.add_simplifier((*fac)(m, s->get_params(), m_preprocess_state));
        else 
            init_preprocess(m, s->get_params(), m_preprocess, m_preprocess_state);
    }

    void assert_expr_core2(expr* t, expr* a) override {
        m_cached_model = nullptr;
        m_cached_mc    = nullptr;
        proof* pr = m.proofs_enabled() ? m.mk_asserted(t) : nullptr;
        m_fmls.push_back(dependent_expr(m, t, pr, m.mk_leaf(a)));
    }

    void assert_expr_core(expr* t) override {
        m_cached_model = nullptr;
        m_cached_mc    = nullptr;
        proof* pr = m.proofs_enabled() ? m.mk_asserted(t) : nullptr;
        m_fmls.push_back(dependent_expr(m, t, pr, nullptr));
    }

    void push() override { 
        expr_ref_vector none(m);
        flush(none);
        m_preprocess_state.push();
        m_preprocess.push();
        m_preprocess_state.m_trail.push(restore_vector(m_fmls));
        s->push(); 
    }

    void pop(unsigned n) override { 
        s->pop(n);
        m_cached_model = nullptr;
        m_preprocess.pop(n);
        m_preprocess_state.pop(n);
    }

    lbool check_sat_core(unsigned num_assumptions, expr* const* assumptions) override { 
        expr_ref_vector _assumptions(m, num_assumptions, assumptions);
        flush(_assumptions);
        return s->check_sat_core(num_assumptions, assumptions); 
    }

    void collect_statistics(statistics& st) const override { 
        s->collect_statistics(st); 
        m_preprocess.collect_statistics(st);
    }

    model_ref m_cached_model;
    void get_model_core(model_ref& m) override {       
        CTRACE("simplifier", m_mc.get(), m_mc->display(tout));
        if (m_cached_model) {
            m = m_cached_model;
            return;
        }
        s->get_model(m); 
        if (m_mc)
            (*m_mc)(m);
        m_cached_model = m;        
    }

    proof_ref m_proof;
    proof* get_proof_core() { 
        proof* p = s->get_proof(); 
        m_proof = p;
        if (p) {
            expr_ref tmp(p, m);
            expr_safe_replace sub(m);
            for (auto const& d : m_fmls) {
                if (d.pr()) 
                    sub.insert(m.mk_asserted(d.fml()), d.pr());                
            }
            sub(tmp);
            SASSERT(is_app(tmp));
            m_proof = to_app(tmp);
        }
        return m_proof;
    }

    solver* translate(ast_manager& m, params_ref const& p) override { 
        solver* new_s = s->translate(m, p);
        ast_translation tr(get_manager(), m);
        simplifier_solver* result = alloc(simplifier_solver, new_s, nullptr); // factory?
        for (dependent_expr const& f : m_fmls) 
            result->m_fmls.push_back(dependent_expr(tr, f));
        if (m_mc) 
            result->m_mc = dynamic_cast<generic_model_converter*>(m_mc->translate(tr));

        // copy m_preprocess_state?
        return result;
    }    

    void updt_params(params_ref const& p) override { 
        s->updt_params(p); 
        m_preprocess.updt_params(p); 
    }

    mutable model_converter_ref m_cached_mc;
    model_converter_ref get_model_converter() const override { 
        if (!m_cached_mc) 
            m_cached_mc = concat(solver::get_model_converter().get(), m_mc.get(), s->get_model_converter().get());
        return m_cached_mc;
    }

    unsigned get_num_assertions() const override { return s->get_num_assertions(); }
    expr* get_assertion(unsigned idx) const override { return s->get_assertion(idx); }
    std::string reason_unknown() const override { return s->reason_unknown(); }
    void set_reason_unknown(char const* msg) override { s->set_reason_unknown(msg); }
    void get_labels(svector<symbol>& r) override { s->get_labels(r); }
    void get_unsat_core(expr_ref_vector& r) { s->get_unsat_core(r); }
    ast_manager& get_manager() const override { return s->get_manager(); }    
    void reset_params(params_ref const& p) override { s->reset_params(p); }
    params_ref const& get_params() const override { return s->get_params(); }
    void collect_param_descrs(param_descrs& r) override { s->collect_param_descrs(r); }
    void push_params() override { s->push_params(); }
    void pop_params() override { s->pop_params(); }
    void set_produce_models(bool f) override { s->set_produce_models(f); }
    void set_phase(expr* e) override { s->set_phase(e); }
    void move_to_front(expr* e) override { s->move_to_front(e); }
    phase* get_phase() override { return s->get_phase(); }
    void set_phase(phase* p) override { s->set_phase(p); }
    unsigned get_num_assumptions() const override { return s->get_num_assumptions(); }
    expr* get_assumption(unsigned idx) const override { return s->get_assumption(idx); }
    unsigned get_scope_level() const override { return s->get_scope_level(); }
    lbool check_sat_cc(expr_ref_vector const& cube, vector<expr_ref_vector> const& clauses) override { return check_sat_cc(cube, clauses); }
    void set_progress_callback(progress_callback* callback) override { s->set_progress_callback(callback); }
    lbool get_consequences(expr_ref_vector const& asms, expr_ref_vector const& vars, expr_ref_vector& consequences) override {
        return s->get_consequences(asms, vars, consequences);
    }
    lbool find_mutexes(expr_ref_vector const& vars, vector<expr_ref_vector>& mutexes) override { return s->find_mutexes(vars, mutexes); }
    lbool preferred_sat(expr_ref_vector const& asms, vector<expr_ref_vector>& cores) override { return s->preferred_sat(asms, cores); }
    
    expr_ref_vector cube(expr_ref_vector& vars, unsigned backtrack_level) override { return s->cube(vars, backtrack_level); }
    expr* congruence_root(expr* e) override { return s->congruence_root(e); }
    expr* congruence_next(expr* e) override { return s->congruence_next(e); }
    std::ostream& display(std::ostream& out, unsigned n, expr* const* assumptions) const override {
        return s->display(out, n, assumptions);
    }
    void get_units_core(expr_ref_vector& units) override { s->get_units_core(units); }
    expr_ref_vector get_trail(unsigned max_level) override { return s->get_trail(max_level); }
    void get_levels(ptr_vector<expr> const& vars, unsigned_vector& depth) override { s->get_levels(vars, depth); }

    void register_on_clause(void* ctx, user_propagator::on_clause_eh_t& on_clause) override {
        s->register_on_clause(ctx, on_clause);
    }
    
    void user_propagate_init(
        void*                ctx, 
        user_propagator::push_eh_t&   push_eh,
        user_propagator::pop_eh_t&    pop_eh,
        user_propagator::fresh_eh_t&  fresh_eh) override {
        s->user_propagate_init(ctx, push_eh, pop_eh, fresh_eh);
    }        
    void user_propagate_register_fixed(user_propagator::fixed_eh_t& fixed_eh) override { s->user_propagate_register_fixed(fixed_eh); }    
    void user_propagate_register_final(user_propagator::final_eh_t& final_eh) override { s->user_propagate_register_final(final_eh); }
    void user_propagate_register_eq(user_propagator::eq_eh_t& eq_eh) override { s->user_propagate_register_eq(eq_eh); }    
    void user_propagate_register_diseq(user_propagator::eq_eh_t& diseq_eh) override { s->user_propagate_register_diseq(diseq_eh); }    
    void user_propagate_register_expr(expr* e) override { /*m_preprocess.user_propagate_register_expr(e); */  s->user_propagate_register_expr(e); }
    void user_propagate_register_created(user_propagator::created_eh_t& r) override { s->user_propagate_register_created(r); }
    // void user_propagate_clear() override { m_preprocess.user_propagate_clear(); s->user_propagate_clear(); }


};

solver* mk_simplifier_solver(solver* s, simplifier_factory* fac) {
    return alloc(simplifier_solver, s, fac);
}

