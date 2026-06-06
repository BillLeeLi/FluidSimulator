#pragma once

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Engine/GL/UniformBlock.hpp"
#include "Engine/Sphere.h"
#include "Simulation/SimulationWorld.h"
#include "Common/ICase.h"
#include "Common/ImageRGB.h"
#include "Common/OrbitCameraManager.h"
#include "Scene/Content.h"
#include "Scene/SceneObject.h"

#include <optional>
#include <utility>
#include <vector>

namespace VCX::MainScene {

    struct WorldRay {
        glm::vec3 Origin { 0.0f };
        glm::vec3 Direction { 0.0f, 0.0f, -1.0f };
    };

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
        Scene::SceneObject            _sceneObject;
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

        Engine::GL::UniqueIndexedRenderItem _BoundaryItem;
        Common::OrbitCameraManager          _cameraManager;
        float                               _BndWidth { 2.0f };
        bool                                _stopped { false };
        Engine::Model                       _sphere;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _particleItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _fluidSurfaceItem;
        std::vector<glm::vec3> _fluidSurfaceOffsets;
        std::vector<glm::vec3> _fluidSurfaceColors;

        // Rigid bodies are rendered in the same style as Lab1: solid colored mesh + white/yellow wireframe.
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidBoxItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidBoxLineItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidSphereItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidSphereLineItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidBoatItem;
        std::optional<Engine::GL::UniqueIndexedRenderItem> _rigidBoatLineItem;
        std::optional<Engine::GL::UniqueRenderItem>        _rigidContactPointItem;

        std::pair<std::uint32_t, std::uint32_t> _viewportSize { 1, 1 };
        float                               _r;
        SimulationWorld                     _world;

        bool  _drawRigidSolid     { true };
        bool  _drawRigidWireframe { true };
        bool  _showRigidContacts  { true };
        int   _fluidRenderMode    { 2 }; // 0=particles, 1=surface, 2=both
        float _rigidLineWidth     { 1.5f };
        float _rigidPointSize     { 7.0f };

        // Lab1-style interaction state.  F applies a point force at the hovered surface point.
        // Alt + left mouse drags the body on a camera-facing plane.  The drag math is the same pick-plane logic used in Lab1; Alt is kept only to avoid stealing the Lab4 camera drag.
        bool            _isDraggingRigidBody { false };
        int             _hoverRigidBodyId    { -1 };
        bool            _hoverHasHit         { false };
        Eigen::Vector3f _hoverHitPoint       { Eigen::Vector3f::Zero() };
        Eigen::Vector3f _hoverRayDir         { Eigen::Vector3f::UnitZ() };
        Eigen::Vector3f _dragPlanePoint      { Eigen::Vector3f::Zero() };
        Eigen::Vector3f _dragPlaneNormal     { Eigen::Vector3f::UnitZ() };
        Eigen::Vector3f _dragBodyOffset      { Eigen::Vector3f::Zero() };

        char const *          GetSceneName(std::size_t const i) const { return VCX::Scene::Content::SceneNames[std::size_t(_scenes[i])].c_str(); }
        Engine::Scene const & GetScene(std::size_t const i) const { return VCX::Scene::Content::Scenes[std::size_t(_scenes[i])]; }
        void                  ResetSystem();
        void                  ResetSystem(int res);
        void                  RebuildParticleRenderItem();
        void                  RebuildFluidSurfaceRenderItem();
        void                  RebuildRigidBodyRenderItem();
        void                  DrawRigidBodyControls();
        void                  ApplyRigidBodyLab1Controls();
        bool                  ComputePickRay(ImVec2 const & pos, Eigen::Vector3f & rayOrigin, Eigen::Vector3f & rayDir) const;
        bool                  PickRigidBody(ImVec2 const & pos, int & bodyId, Eigen::Vector3f & hitPoint) const;
        bool                  UpdateDraggedRigidBody(ImVec2 const & pos);
        std::vector<glm::vec3> GetRigidBoxVertices(RigidBody const & body) const;
        std::vector<glm::vec3> GetRigidSphereVertices(RigidBody const & body) const;
        std::vector<glm::vec3> GetRigidBoatVertices(RigidBody const & body) const;
        std::vector<glm::vec3> GetRigidContactVertices() const;
        WorldRay              ScreenPointToWorldRay(ImVec2 const & pos) const;
    };

} // namespace VCX::MainScene
