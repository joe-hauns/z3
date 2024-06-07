/*++
Copyright (c) 2015 Microsoft Corporation

--*/


#pragma once

#include "ast/arith_decl_plugin.h"
#include "model/model.h"
#include "qe/mbp/mbp_plugin.h"
#include <optional>

namespace mbp {

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


