#pragma once

#include "Engine/app.h"
#include "Labs/2-FluidSimulation/CaseFluid.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::Fluid {

    class App : public Engine::IApp {
    private:
        Common::UI _ui { Common::UIOptions {} };
        CaseFluid  _case;

        std::size_t _caseId = 0;
        std::vector<std::reference_wrapper<Common::ICase>> _cases = { _case };

    public:
        App();

        void OnFrame() override;
    };

} // namespace VCX::Labs::Fluid
