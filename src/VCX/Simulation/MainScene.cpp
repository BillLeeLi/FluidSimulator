#include "Simulation/MainScene.h"
#include "Engine/app.h"
#include "Common/ImGuiHelper.h"

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
    }

    void MainScene::OnSetupPropsUI() {
        if (ImGui::Button("Reset System"))
            ResetSystem();
        ImGui::SameLine();
        if (ImGui::Button(_stopped ? "Start Simulation" : "Stop Simulation"))
            _stopped = ! _stopped;
        ImGui::Spacing();

        float flipRatio = _world.FlipRatio();
        if (ImGui::SliderFloat("FLIP Ratio", &flipRatio, 0.0f, 1.0f))
            _world.SetFlipRatio(flipRatio);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = PIC (stable, dissipative)\n1 = FLIP (non-dissipative, noisy)\n0.95 = FLIP95 (recommended)");

        int resolution = _world.Resolution();
        if (ImGui::SliderInt("Resolution", &resolution, 8, 32))
            ResetSystem(resolution);

        int invDeltaTime = _world.InvDeltaTime();
        if (ImGui::SliderInt("Inv Time Step", &invDeltaTime, 20, 240))
            _world.SetInvDeltaTime(invDeltaTime);

        auto const & fluid = _world.GetFluid();
        ImGui::Text("Particles: %d", fluid.m_iNumSpheres);
        ImGui::Text("Grid: %d x %d x %d", fluid.m_iCellX, fluid.m_iCellY, fluid.m_iCellZ);
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Simulate Time: %.3f ms", _world.LastSimTimeMs());
    }

    Common::CaseRenderResult MainScene::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _viewportSize = desiredSize;

        if (_recompute) {
            _recompute = false;
            _sceneObject.ReplaceScene(GetScene(_sceneIdx));
            _cameraManager.Save(_sceneObject.Camera);
        }
        if (! _stopped)
            _world.StepFrame(ImGui::GetIO().DeltaTime);

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

        auto const & fluid = _world.GetFluid();
        if (fluid.m_iNumSpheres > 0 && _particleItem) {
            _particleItem->UpdateVertexBuffer("offset", Engine::make_span_bytes<glm::vec3>(fluid.m_particlePos));
            _particleItem->UpdateVertexBuffer("color", Engine::make_span_bytes<glm::vec3>(fluid.m_particleColor));
            _particleItem->Draw(
                { _program.Use() },
                _sphere.Mesh.Indices.size(),
                0,
                fluid.m_iNumSpheres);
        }

        // glDepthFunc(GL_LEQUAL);
        glDepthFunc(GL_LESS);   // 只在片段深度值小于当前深度缓冲区值时绘制
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
        _world.Reset();
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        RebuildParticleRenderItem();
    }

    void MainScene::ResetSystem(int res) {
        _world.Reset(res);
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        RebuildParticleRenderItem();
    }

    void MainScene::RebuildParticleRenderItem() {
        _particleItem.emplace(
            Engine::GL::VertexLayout()
                .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Static, 0)
                .Add<glm::vec3>("normal", Engine::GL::DrawFrequency::Static, 1)
                .Add<glm::vec3>("offset", Engine::GL::DrawFrequency::Stream, 2)
                .Add<glm::vec3>("color", Engine::GL::DrawFrequency::Stream, 3),
            Engine::GL::PrimitiveType::Triangles);
        _particleItem->UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(_sphere.Mesh.Positions));
        _particleItem->UpdateVertexBuffer("normal", Engine::make_span_bytes<glm::vec3>(_sphere.Mesh.Normals));
        _particleItem->UpdateVertexBuffer("offset", Engine::make_span_bytes<glm::vec3>(_world.GetFluid().m_particlePos));
        _particleItem->UpdateVertexBuffer("color", Engine::make_span_bytes<glm::vec3>(_world.GetFluid().m_particleColor));
        _particleItem->SetAttributeDivisor(2, 1);
        _particleItem->SetAttributeDivisor(3, 1);
        _particleItem->UpdateElementBuffer(_sphere.Mesh.Indices);
    }

    WorldRay MainScene::ScreenPointToWorldRay(ImVec2 const & pos) const {
        float const width  = float(std::max<std::uint32_t>(_viewportSize.first, 1));
        float const height = float(std::max<std::uint32_t>(_viewportSize.second, 1));
        float const aspect = width / height;

        float const x = 2.0f * pos.x / width - 1.0f;
        float const y = 1.0f - 2.0f * pos.y / height;

        glm::mat4 const invViewProj = glm::inverse(
            _sceneObject.Camera.GetProjectionMatrix(aspect) *
            _sceneObject.Camera.GetViewMatrix());

        glm::vec4 nearPoint = invViewProj * glm::vec4(x, y, -1.0f, 1.0f);
        glm::vec4 farPoint  = invViewProj * glm::vec4(x, y, 1.0f, 1.0f);
        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;

        return WorldRay {
            .Origin    = _sceneObject.Camera.Eye,
            .Direction = glm::normalize(glm::vec3(farPoint - nearPoint)),
        };
    }
}; // namespace VCX::MainScene
