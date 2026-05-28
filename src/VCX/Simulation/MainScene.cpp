#include "Simulation/MainScene.h"
#include "Engine/app.h"
#include "Common/ImGuiHelper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace VCX::MainScene {
    namespace {
        constexpr float kPi = 3.14159265358979323846f;

        const std::vector<glm::vec3> vertex_pos = {
            glm::vec3(-0.5f, -0.5f, -0.5f),
            glm::vec3( 0.5f, -0.5f, -0.5f),
            glm::vec3( 0.5f,  0.5f, -0.5f),
            glm::vec3(-0.5f,  0.5f, -0.5f),
            glm::vec3(-0.5f, -0.5f,  0.5f),
            glm::vec3( 0.5f, -0.5f,  0.5f),
            glm::vec3( 0.5f,  0.5f,  0.5f),
            glm::vec3(-0.5f,  0.5f,  0.5f),
        };
        const std::vector<std::uint32_t> line_index = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };

        std::array<std::uint32_t, 24> const kBoxLineIndex = {
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        };

        glm::vec3 MakeSphereRingPoint(RigidBody const & body, int ring, int segment, int segmentCount) {
            float const angle = 2.0f * kPi * float(segment) / float(segmentCount);
            float const c = std::cos(angle);
            float const s = std::sin(angle);
            float const r = 0.5f * body.dim.x();

            Eigen::Vector3f local;
            if (ring == 0) local = Eigen::Vector3f(r * c, r * s, 0.0f);      // xy
            else if (ring == 1) local = Eigen::Vector3f(r * c, 0.0f, r * s); // xz
            else local = Eigen::Vector3f(0.0f, r * c, r * s);                // yz
            return ToGlm(body.LocalToWorld(local));
        }
    } // namespace

    MainScene::MainScene(std::initializer_list<Assets::ExampleScene> && scenes):
        _scenes(scenes),
        _program(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/fluid.vert"), Engine::GL::SharedShader("assets/shaders/fluid.frag") })),
        _lineprogram(Engine::GL::UniqueProgram({ Engine::GL::SharedShader("assets/shaders/flat.vert"), Engine::GL::SharedShader("assets/shaders/flat.frag") })),
        _sceneObject(1),
        _BoundaryItem(Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0), Engine::GL::PrimitiveType::Lines) {
        _cameraManager.AutoRotate = false;

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

        DrawRigidBodyControls();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Simulate Time: %.3f ms", _world.LastSimTimeMs());
    }

    void MainScene::DrawRigidBodyControls() {
        auto & rigid = _world.GetRigidBodies();
        if (! ImGui::CollapsingHeader("Rigid Body Controls", ImGuiTreeNodeFlags_DefaultOpen)) return;

        char const * presetNames[] = { "Fluid Coupling Mixed", "Box Collision", "Mixed Stack" };
        int preset = int(_world.RigidPreset());
        if (ImGui::Combo("Rigid Preset", &preset, presetNames, 3)) {
            _world.SetRigidPreset(RigidBodyPreset(preset));
            RebuildRigidBodyRenderItem();
        }

        ImGui::Text("Rigid Bodies: %d", int(rigid.Bodies.size()));
        ImGui::Text("Contacts: %d", int(rigid.Contacts.size()));

        int selected = _world.SelectedRigidBody();
        if (! rigid.Bodies.empty()) {
            selected = std::clamp(selected, 0, int(rigid.Bodies.size()) - 1);
            if (ImGui::SliderInt("Selected Body", &selected, 0, int(rigid.Bodies.size()) - 1)) {
                _world.SetSelectedRigidBody(selected);
            }

            auto & body = rigid.Bodies[selected];
            ImGui::Text("Name: %s", body.name.c_str());
            ImGui::Text("Shape: %s", body.shape == RigidBodyShape::Box ? "Box" : "Sphere");

            bool isStatic = body.isStatic;
            if (ImGui::Checkbox("Static", &isStatic)) rigid.SetBodyStatic(selected, isStatic);

            bool useGravity = body.useGravity;
            if (ImGui::Checkbox("Use Gravity", &useGravity)) rigid.SetBodyGravity(selected, useGravity);

            float mass = std::isfinite(body.mass) ? body.mass : 1.0f;
            if (! body.isStatic && ImGui::SliderFloat("Mass", &mass, 0.05f, 3.0f)) {
                rigid.SetBodyMass(selected, mass);
            }

            glm::vec3 dim = ToGlm(body.dim);
            if (body.shape == RigidBodyShape::Sphere) {
                float diameter = dim.x;
                if (ImGui::SliderFloat("Sphere Diameter", &diameter, 0.04f, 0.30f)) {
                    rigid.SetBodyDim(selected, Eigen::Vector3f::Constant(diameter));
                }
            } else {
                if (ImGui::SliderFloat3("Box Size", &dim.x, 0.04f, 0.30f)) {
                    rigid.SetBodyDim(selected, ToEigen(dim));
                }
            }

            glm::vec3 pos = ToGlm(body.x);
            if (ImGui::SliderFloat3("Position", &pos.x, -0.45f, 0.45f)) {
                body.x = ToEigen(pos);
            }

            ImGui::Text("Velocity: %.3f %.3f %.3f", body.v.x(), body.v.y(), body.v.z());
            ImGui::Text("Angular Vel: %.3f %.3f %.3f", body.w.x(), body.w.y(), body.w.z());
        }

        float keyboardForce = _world.RigidKeyboardForce();
        if (ImGui::SliderFloat("Keyboard Force", &keyboardForce, 0.0f, 30.0f)) {
            _world.SetRigidKeyboardForce(keyboardForce);
        }
        ImGui::TextWrapped("Move selected rigid body: J/L = X, U/O = Y, I/K = Z. Q/E applies yaw torque. This is the rigid-body part; fluid-solid coupling can use the same bodies later.");
    }

    Common::CaseRenderResult MainScene::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _viewportSize = desiredSize;

        if (_recompute) {
            _recompute = false;
            _sceneObject.ReplaceScene(GetScene(_sceneIdx));
            _cameraManager.Save(_sceneObject.Camera);
        }

        ApplyRigidBodyKeyboardControls();
        if (! _stopped)
            _world.StepFrame(ImGui::GetIO().DeltaTime);

        _BoundaryItem.UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(vertex_pos));
        UpdateRigidBodyRenderItem();
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
        _lineprogram.GetUniforms().SetByName("u_Color", glm::vec3(1.0f));
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

        if (_rigidBodyLineItem && ! _world.GetRigidBodies().Bodies.empty()) {
            glLineWidth(2.0f);
            _lineprogram.GetUniforms().SetByName("u_Color", glm::vec3(1.0f, 0.86f, 0.25f));
            _rigidBodyLineItem->Draw({ _lineprogram.Use() });
            glLineWidth(1.0f);
        }

        glDepthFunc(GL_LESS);
        glDisable(GL_DEPTH_TEST);

        return Common::CaseRenderResult {
            .Fixed     = false,
            .Flipped   = true,
            .Image     = _frame.GetColorAttachment(),
            .ImageSize = desiredSize,
        };
    }

    void MainScene::ApplyRigidBodyKeyboardControls() {
        int const selected = _world.SelectedRigidBody();
        if (selected < 0) return;

        float const forceMag = _world.RigidKeyboardForce();
        Eigen::Vector3f f = Eigen::Vector3f::Zero();
        if (ImGui::IsKeyDown(ImGuiKey_J)) f.x() -= forceMag;
        if (ImGui::IsKeyDown(ImGuiKey_L)) f.x() += forceMag;
        if (ImGui::IsKeyDown(ImGuiKey_U)) f.y() += forceMag;
        if (ImGui::IsKeyDown(ImGuiKey_O)) f.y() -= forceMag;
        if (ImGui::IsKeyDown(ImGuiKey_I)) f.z() -= forceMag;
        if (ImGui::IsKeyDown(ImGuiKey_K)) f.z() += forceMag;
        if (f.squaredNorm() > 0.0f) {
            _world.ApplyExternalForceToBody(selected, f);
        }

        Eigen::Vector3f torque = Eigen::Vector3f::Zero();
        if (ImGui::IsKeyDown(ImGuiKey_Q)) torque.y() += forceMag * 0.08f;
        if (ImGui::IsKeyDown(ImGuiKey_E)) torque.y() -= forceMag * 0.08f;
        if (torque.squaredNorm() > 0.0f) {
            _world.ApplyExternalTorqueToBody(selected, torque);
        }
    }

    void MainScene::OnProcessInput(ImVec2 const & pos) {
        _cameraManager.ProcessInput(_sceneObject.Camera, pos);
    }

    void MainScene::ResetSystem() {
        _world.Reset();
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        RebuildParticleRenderItem();
        RebuildRigidBodyRenderItem();
    }

    void MainScene::ResetSystem(int res) {
        _world.Reset(res);
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        RebuildParticleRenderItem();
        RebuildRigidBodyRenderItem();
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

    void MainScene::RebuildRigidBodyRenderItem() {
        _rigidBodyLineItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines);
        UpdateRigidBodyRenderItem();
    }

    void MainScene::UpdateRigidBodyRenderItem() {
        if (! _rigidBodyLineItem) return;

        std::vector<glm::vec3>     vertices;
        std::vector<std::uint32_t> indices;
        auto const & bodies = _world.GetRigidBodies().Bodies;

        for (auto const & body : bodies) {
            if (body.shape == RigidBodyShape::Box) {
                auto const corners = body.GetWorldCorners();
                std::uint32_t const base = static_cast<std::uint32_t>(vertices.size());
                for (auto const & c : corners) vertices.push_back(ToGlm(c));
                for (std::uint32_t idx : kBoxLineIndex) indices.push_back(base + idx);
            } else {
                int constexpr segmentCount = 32;
                for (int ring = 0; ring < 3; ++ring) {
                    std::uint32_t const base = static_cast<std::uint32_t>(vertices.size());
                    for (int s = 0; s < segmentCount; ++s) {
                        vertices.push_back(MakeSphereRingPoint(body, ring, s, segmentCount));
                    }
                    for (int s = 0; s < segmentCount; ++s) {
                        indices.push_back(base + std::uint32_t(s));
                        indices.push_back(base + std::uint32_t((s + 1) % segmentCount));
                    }
                }
            }
        }

        if (vertices.empty() || indices.empty()) return;
        _rigidBodyLineItem->UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(vertices));
        _rigidBodyLineItem->UpdateElementBuffer(indices);
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
