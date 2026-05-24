#pragma once

#include <vector>

#include "Engine/app.h"
#include "Simulation/MainScene.h"
#include "Common/UI.h"

namespace VCX::MainScene {
    class App : public Engine::IApp {
    private:
        Common::UI _ui;

        MainScene _mainScene;

        std::size_t _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = { _mainScene };

    public:
        App();

        void OnFrame() override;
    };
}