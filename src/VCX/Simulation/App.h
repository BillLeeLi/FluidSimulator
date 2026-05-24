#pragma once

#include <vector>

#include "Engine/app.h"
#include "Simulation/CaseFluid.h"
#include "Common/UI.h"

namespace VCX::Fluid {
    class App : public Engine::IApp {
    private:
        Common::UI _ui;

        CaseFluid _casefluid;

        std::size_t _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = { _casefluid };

    public:
        App();

        void OnFrame() override;
    };
}