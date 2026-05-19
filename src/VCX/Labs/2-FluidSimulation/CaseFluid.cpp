#include "Labs/2-FluidSimulation/CaseFluid.h"
#include "Engine/app.h"
#include "Labs/Common/ImGuiHelper.h"

namespace VCX::Labs::Fluid {

    // 水槽边界框 — 单位立方体 [-0.5, 0.5] 的线框索引
    static std::vector<glm::vec3> const kBoundaryVertices = {
        glm::vec3(-0.5f, -0.5f, -0.5f),
        glm::vec3(0.5f, -0.5f, -0.5f),
        glm::vec3(0.5f, 0.5f, -0.5f),
        glm::vec3(-0.5f, 0.5f, -0.5f),
        glm::vec3(-0.5f, -0.5f, 0.5f),
        glm::vec3(0.5f, -0.5f, 0.5f),
        glm::vec3(0.5f, 0.5f, 0.5f),
        glm::vec3(-0.5f, 0.5f, 0.5f),
    };
    static std::vector<std::uint32_t> const kBoundaryIndices = {
        0, 1, 1, 2, 2, 3, 3, 0, // 后面
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4, // 前面
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7 // 连接边
    };

    CaseFluid::CaseFluid(std::initializer_list<Assets::ExampleScene> && scenes):
        _scenes(scenes),
        _program(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/fluid.vert"), Engine::GL::SharedShader("assets/shaders/fluid.frag") })),
        _lineProgram(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"), Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _sceneObject(1), // PassConstants UBO 绑定到 binding point 1
        _boundaryItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Lines) {
        // 将着色器的 PassConstants 块映射到 binding point 1
        _program.BindUniformBlock("PassConstants", 1);
        _program.GetUniforms().SetByName("u_AmbientScale", 0.3f);
        _program.GetUniforms().SetByName("u_UseBlinn", 1);
        _program.GetUniforms().SetByName("u_Shininess", 32.0f);
        _program.GetUniforms().SetByName("u_UseGammaCorrection", 0);
        _program.GetUniforms().SetByName("u_AttenuationOrder", 2);

        _lineProgram.GetUniforms().SetByName("u_Color", glm::vec3(0.4f, 0.6f, 0.9f));
        _boundaryItem.UpdateElementBuffer(kBoundaryIndices);

        _cameraManager.AutoRotate = false;
        _cameraManager.Save(_sceneObject.Camera);

        ResetSystem();
    }

    void CaseFluid::OnSetupPropsUI() {
        if (ImGui::Button("Reset System"))
            ResetSystem();
        ImGui::SameLine();
        if (ImGui::Button(_stopped ? "Start Simulation" : "Stop Simulation"))
            _stopped = !_stopped;
        ImGui::Spacing();

        ImGui::SliderFloat("FLIP Ratio", &_simulation.m_fRatio, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = PIC (stable, dissipative)\n1 = FLIP (non-dissipative, noisy)\n0.95 = FLIP95 (recommended)");

        if (ImGui::SliderInt("Resolution", &_res, 8, 32))
            ResetSystem();

        ImGui::Text("Particles: %d", _simulation.m_iNumSpheres);
        ImGui::Text("Grid: %d x %d x %d", _simulation.m_iCellX, _simulation.m_iCellY, _simulation.m_iCellZ);
    }

    Common::CaseRenderResult CaseFluid::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        if (!_stopped)
            _simulation.SimulateTimestep(Engine::GetDeltaTime());

        _boundaryItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(kBoundaryVertices));
        _frame.Resize(desiredSize);

        _cameraManager.Update(_sceneObject.Camera);
        auto const proj = _sceneObject.Camera.GetProjectionMatrix(float(desiredSize.first) / float(desiredSize.second));
        auto const view = _sceneObject.Camera.GetViewMatrix();

        // 通过 SceneObject 内置的 PassConstantsBlock 更新 UBO
        _sceneObject.PassConstantsBlock.Update(&Rendering::SceneObject::PassConstants::Projection, proj);
        _sceneObject.PassConstantsBlock.Update(&Rendering::SceneObject::PassConstants::View, view);
        _sceneObject.PassConstantsBlock.Update(&Rendering::SceneObject::PassConstants::ViewPosition, _sceneObject.Camera.Eye);

        _lineProgram.GetUniforms().SetByName("u_Projection", proj);
        _lineProgram.GetUniforms().SetByName("u_View", view);

        gl_using(_frame);
        glEnable(GL_DEPTH_TEST);

        glLineWidth(_lineWidth);
        _boundaryItem.Draw({ _lineProgram.Use() });
        glLineWidth(1.0f);

        if (_simulation.m_iNumSpheres > 0) {
            auto const modelObj = Rendering::ModelObject(
                _sphere,
                _simulation.m_particlePos,
                _simulation.m_particleColor);

            modelObj.Mesh.Draw(
                { _program.Use() },
                _sphere.Mesh.Indices.size(),
                0,
                _simulation.m_iNumSpheres);
        }

        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void CaseFluid::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_sceneObject.Camera, pos);
    }

    void CaseFluid::ResetSystem() {
        _simulation.setupScene(_res);
        _sphere = Engine::Model { Engine::Sphere(6, _simulation.m_particleRadius), 0 };
        _cameraManager.Reset(_sceneObject.Camera);
    }

} // namespace VCX::Labs::Fluid
