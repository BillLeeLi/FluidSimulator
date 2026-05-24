#include "Simulation/MainScene.h"
#include "Engine/app.h"
#include "Common/ImGuiHelper.h"
#include <chrono>

namespace VCX::MainScene {
    // 水槽边界框 — 单位立方体 [-0.5, 0.5] 的线框索引
    const std::vector<glm::vec3> vertex_pos = {
        glm::vec3(-0.5f, -0.5f, -0.5f),
        glm::vec3(0.5f, -0.5f, -0.5f),
        glm::vec3(0.5f, 0.5f, -0.5f),
        glm::vec3(-0.5f, 0.5f, -0.5f),
        glm::vec3(-0.5f, -0.5f, 0.5f),
        glm::vec3(0.5f, -0.5f, 0.5f),
        glm::vec3(0.5f, 0.5f, 0.5f),
        glm::vec3(-0.5f, 0.5f, 0.5f),
    };
    const std::vector<std::uint32_t> line_index = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 }; // line index

    MainScene::MainScene(std::initializer_list<Assets::ExampleScene> && scenes):
        _scenes(scenes),
        _program(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/fluid.vert"), Engine::GL::SharedShader("assets/shaders/fluid.frag") })),
        _lineprogram(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"), Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _sceneObject(1), // PassConstants UBO 绑定到 binding point 1
        _BoundaryItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Lines) {
        _cameraManager.AutoRotate = false;

        // 将着色器的 PassConstants 块映射到 binding point 1
        _program.BindUniformBlock("PassConstants", 1);
        _program.GetUniforms().SetByName("u_DiffuseMap", 0);
        _program.GetUniforms().SetByName("u_SpecularMap", 1);
        _program.GetUniforms().SetByName("u_HeightMap", 2);
        _lineprogram.GetUniforms().SetByName("u_Color", glm::vec3(1.0f));
        _BoundaryItem.UpdateElementBuffer(line_index);
        ResetSystem();
        _sphere = Engine::Model { Engine::Sphere(6, _r), 0 };
    }

    void MainScene::OnSetupPropsUI() {
        if (ImGui::Button("Reset System"))
            ResetSystem();
        ImGui::SameLine();
        if (ImGui::Button(_stopped ? "Start Simulation" : "Stop Simulation"))
            _stopped = ! _stopped;
        ImGui::Spacing();

        ImGui::SliderFloat("FLIP Ratio", &_simulation.m_fRatio, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = PIC (stable, dissipative)\n1 = FLIP (non-dissipative, noisy)\n0.95 = FLIP95 (recommended)");

        if (ImGui::SliderInt("Resolution", &_res, 8, 32))
            ResetSystem();

        (ImGui::SliderInt("Inv Time Step", &_invDeltaTime, 20, 240));

        ImGui::Text("Particles: %d", _simulation.m_iNumSpheres);
        ImGui::Text("Grid: %d x %d x %d", _simulation.m_iCellX, _simulation.m_iCellY, _simulation.m_iCellZ);
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Simulate Time: %.3f ms", _lastSimTime);
    }

    Common::CaseRenderResult MainScene::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        if (_recompute) {
            _recompute = false;
            _sceneObject.ReplaceScene(GetScene(_sceneIdx));
            _cameraManager.Save(_sceneObject.Camera);
        }
        if (! _stopped){
            float frameDt = ImGui::GetIO().DeltaTime;
            // 为了避免在调试过程中帧率过低导致模拟不稳定，限制最大时间步长
            if (frameDt > 0.1f) frameDt = 0.1f;
            _timeAccumulator += frameDt;
            float dt = 1.0f / float(_invDeltaTime);
            int stepCount = 0;
            // 限制每帧最多执行的物理步数，避免在性能跟不上的情况下模拟过度堆积
            while (_timeAccumulator >= dt && stepCount < 2) {
                auto startSim = std::chrono::high_resolution_clock::now();
                _simulation.SimulateTimestep(dt);
                auto endSim = std::chrono::high_resolution_clock::now();
                _lastSimTime = std::chrono::duration<float, std::milli>(endSim - startSim).count();
                
                _timeAccumulator -= dt;
                stepCount++;
            }
            // 如果堆积的时间依然太多，说明性能跟不上，丢弃剩余时间（允许物理时间慢跑来弥补显示效果）
            if (_timeAccumulator > dt) {
                _timeAccumulator = 0.0f;
            }
        }

        _BoundaryItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(vertex_pos));
        _frame.Resize(desiredSize);

        _cameraManager.Update(_sceneObject.Camera);
        _sceneObject.PassConstantsBlock.Update(&VCX::Scene::SceneObject::PassConstants::Projection, _sceneObject.Camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _sceneObject.PassConstantsBlock.Update(&VCX::Scene::SceneObject::PassConstants::View, _sceneObject.Camera.GetViewMatrix());
        _sceneObject.PassConstantsBlock.Update(&VCX::Scene::SceneObject::PassConstants::ViewPosition, _sceneObject.Camera.Eye);
        _lineprogram.GetUniforms().SetByName("u_Projection", _sceneObject.Camera.GetProjectionMatrix((float(desiredSize.first) / desiredSize.second)));
        _lineprogram.GetUniforms().SetByName("u_View", _sceneObject.Camera.GetViewMatrix());

        if (_uniformDirty) {
            _uniformDirty = false;
            _program.GetUniforms().SetByName("u_AmbientScale", _ambientScale);
            _program.GetUniforms().SetByName("u_UseBlinn", _useBlinn);
            _program.GetUniforms().SetByName("u_Shininess", _shininess);
            _program.GetUniforms().SetByName("u_UseGammaCorrection", int(_useGammaCorrection));
            _program.GetUniforms().SetByName("u_AttenuationOrder", _attenuationOrder);
            _program.GetUniforms().SetByName("u_BumpMappingBlend", _bumpMappingPercent * .01f);
        }

        gl_using(_frame);

        glEnable(GL_DEPTH_TEST);
        glLineWidth(_BndWidth);
        _BoundaryItem.Draw({ _lineprogram.Use() });
        glLineWidth(1.0f);

        if (_simulation.m_iNumSpheres > 0) {
            auto const modelObj = Scene::ModelObject(
                _sphere,
                _simulation.m_particlePos,
                _simulation.m_particleColor);

            modelObj.Mesh.Draw(
                { _program.Use() },
                _sphere.Mesh.Indices.size(),
                0,
                _simulation.m_iNumSpheres);
        }

        glDepthFunc(GL_LEQUAL);
        glDepthFunc(GL_LESS);
        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void MainScene::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_sceneObject.Camera, pos);
    }

    void MainScene::ResetSystem() {
        _simulation.setupScene(_res);
        numofSpheres = _simulation.m_iNumSpheres;
        _r           = _simulation.m_particleRadius;
        _sphere      = Engine::Model { Engine::Sphere(4, _r), 0 };
        _timeAccumulator = 0.0f;
    }
}; // namespace VCX::MainScene
