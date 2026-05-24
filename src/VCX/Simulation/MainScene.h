#pragma once

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/UniformBlock.hpp"
#include "Engine/Sphere.h"
#include "Simulation/FluidSimulator.h"
#include "Common/ICase.h"
#include "Common/ImageRGB.h"
#include "Common/OrbitCameraManager.h"
#include "Scene/Content.h"
#include "Scene/SceneObject.h"

namespace VCX::MainScene {

    class MainScene: public Common::ICase {
    public:
        MainScene(std::initializer_list<Assets::ExampleScene> && scenes);

        virtual std::string_view const GetName() override { return "Fluid Simulation"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        std::vector<Assets::ExampleScene> const _scenes;

        Engine::GL::UniqueProgram     _program;
        Engine::GL::UniqueProgram     _lineprogram;
        Engine::GL::UniqueRenderFrame _frame;
        Scene::SceneObject        _sceneObject;
        std::size_t                   _sceneIdx { 0 };
        bool                          _recompute { true };
        bool                          _uniformDirty { true };
        int                           _msaa { 2 };
        int                           _useBlinn { 0 };
        float                         _shininess { 32 };
        float                         _ambientScale { 1 };
        bool                          _useGammaCorrection { true };
        int                           _attenuationOrder { 2 };
        int                           _bumpMappingPercent { 20 };
        int                           _invDeltaTime { 60 };

        Engine::GL::UniqueIndexedRenderItem _BoundaryItem;
        Common::OrbitCameraManager          _cameraManager;
        float                               _BndWidth { 2.0f };
        bool                                _stopped { false };
        Engine::Model                       _sphere;
        int                                 _res { 16 };
        float                               _r;
        int                                 numofSpheres;
        FluidSimulator                           _simulation;

        char const *          GetSceneName(std::size_t const i) const { return VCX::Scene::Content::SceneNames[std::size_t(_scenes[i])].c_str(); }
        Engine::Scene const & GetScene(std::size_t const i) const { return VCX::Scene::Content::Scenes[std::size_t(_scenes[i])]; }
        void                  ResetSystem();
    };

} // namespace VCX::MainScene
