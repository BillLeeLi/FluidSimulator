#include "Simulation/MainScene.h"
#include "Engine/app.h"
#include "Common/ImGuiHelper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <imgui.h>

namespace VCX::MainScene {
    namespace {
        constexpr float kEps = 1e-6f;
        constexpr float kPi  = 3.14159265358979323846f;

        struct SphereMeshData {
            std::vector<glm::vec3>     vertices;
            std::vector<std::uint32_t> triIndices;
            std::vector<std::uint32_t> lineIndices;
        };

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

        const std::vector<std::uint32_t> kBoxLineIndex = {
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        };

        const std::vector<std::uint32_t> kBoxTriIndex = {
            0, 1, 2, 0, 2, 3,
            1, 0, 4, 1, 4, 5,
            1, 5, 6, 1, 6, 2,
            2, 6, 7, 2, 7, 3,
            0, 3, 7, 0, 7, 4,
            4, 7, 6, 4, 6, 5,
        };

        SphereMeshData const & GetUnitRigidSphereMesh() {
            static SphereMeshData mesh = []() {
                SphereMeshData data;
                int constexpr stacks = 10;
                int constexpr slices = 20;

                data.vertices.reserve((stacks + 1) * (slices + 1));
                for (int stack = 0; stack <= stacks; ++stack) {
                    float const v   = static_cast<float>(stack) / static_cast<float>(stacks);
                    float const phi = kPi * v;
                    float const y   = std::cos(phi);
                    float const r   = std::sin(phi);
                    for (int slice = 0; slice <= slices; ++slice) {
                        float const u     = static_cast<float>(slice) / static_cast<float>(slices);
                        float const theta = 2.0f * kPi * u;
                        data.vertices.emplace_back(r * std::cos(theta), y, r * std::sin(theta));
                    }
                }

                auto idx = [slices](int stack, int slice) -> std::uint32_t {
                    return static_cast<std::uint32_t>(stack * (slices + 1) + slice);
                };

                for (int stack = 0; stack < stacks; ++stack) {
                    for (int slice = 0; slice < slices; ++slice) {
                        std::uint32_t const a = idx(stack, slice);
                        std::uint32_t const b = idx(stack + 1, slice);
                        std::uint32_t const c = idx(stack + 1, slice + 1);
                        std::uint32_t const d = idx(stack, slice + 1);
                        data.triIndices.insert(data.triIndices.end(), { a, b, c, a, c, d });
                        data.lineIndices.insert(data.lineIndices.end(), { a, d, a, b });
                    }
                }
                return data;
            }();
            return mesh;
        }

        Eigen::Vector3f SafeNormalized(Eigen::Vector3f const & v, Eigen::Vector3f const & fallback = Eigen::Vector3f::UnitZ()) {
            float const n = v.norm();
            return n > kEps ? (v / n) : fallback;
        }

        bool IntersectRayPlane(
            Eigen::Vector3f const & rayOrigin,
            Eigen::Vector3f const & rayDir,
            Eigen::Vector3f const & planePoint,
            Eigen::Vector3f const & planeNormal,
            Eigen::Vector3f &       hitPoint) {
            float const denom = planeNormal.dot(rayDir);
            if (std::abs(denom) < kEps) return false;
            float const t = planeNormal.dot(planePoint - rayOrigin) / denom;
            if (t < 0.0f) return false;
            hitPoint = rayOrigin + t * rayDir;
            return true;
        }

        bool IntersectRayOBB(
            Eigen::Vector3f const & rayOrigin,
            Eigen::Vector3f const & rayDir,
            RigidBody const &       body,
            float &                 tHit,
            Eigen::Vector3f &       hitPoint) {
            Eigen::Matrix3f const R      = body.GetRotationMatrix();
            Eigen::Matrix3f const invRot = R.transpose();
            Eigen::Vector3f const half   = 0.5f * body.dim;

            Eigen::Vector3f const localOrigin = invRot * (rayOrigin - body.x);
            Eigen::Vector3f const localDir    = invRot * rayDir;

            float tMin = 0.0f;
            float tMax = std::numeric_limits<float>::max();

            for (int axis = 0; axis < 3; ++axis) {
                float const o = localOrigin[axis];
                float const d = localDir[axis];
                float const h = half[axis];
                if (std::abs(d) < kEps) {
                    if (o < -h || o > h) return false;
                    continue;
                }

                float t1 = (-h - o) / d;
                float t2 = ( h - o) / d;
                if (t1 > t2) std::swap(t1, t2);
                tMin = std::max(tMin, t1);
                tMax = std::min(tMax, t2);
                if (tMin > tMax) return false;
            }

            if (tMax < 0.0f) return false;
            tHit     = tMin >= 0.0f ? tMin : tMax;
            hitPoint = rayOrigin + tHit * rayDir;
            return true;
        }


        bool IsInternalTankBoundary(RigidBody const & body) {
            return body.isStatic && body.name.rfind("tank_", 0) == 0;
        }

        bool IntersectRaySphere(
            Eigen::Vector3f const & rayOrigin,
            Eigen::Vector3f const & rayDir,
            RigidBody const &       body,
            float &                 tHit,
            Eigen::Vector3f &       hitPoint) {
            float const           r    = 0.5f * body.dim.x();
            Eigen::Vector3f const oc   = rayOrigin - body.x;
            float const           b    = oc.dot(rayDir);
            float const           c    = oc.dot(oc) - r * r;
            float const           disc = b * b - c;
            if (disc < 0.0f) return false;

            float const sqrtDisc = std::sqrt(disc);
            float       t0       = -b - sqrtDisc;
            float       t1       = -b + sqrtDisc;
            if (t1 < 0.0f) return false;

            tHit     = t0 >= 0.0f ? t0 : t1;
            hitPoint = rayOrigin + tHit * rayDir;
            return true;
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

            glm::vec3 color = ToGlm(body.color);
            if (ImGui::ColorEdit3("Rigid Color", &color.x)) {
                body.color = ToEigen(color);
            }

            ImGui::Text("Velocity: %.3f %.3f %.3f", body.v.x(), body.v.y(), body.v.z());
            ImGui::Text("Angular Vel: %.3f %.3f %.3f", body.w.x(), body.w.y(), body.w.z());
        }

        float pointForce = _world.RigidKeyboardForce();
        if (ImGui::SliderFloat("Lab1 F Point Force", &pointForce, 1.0f, 160.0f)) {
            _world.SetRigidKeyboardForce(pointForce);
        }

        ImGui::Checkbox("Draw Rigid Solid", &_drawRigidSolid);
        ImGui::Checkbox("Draw Rigid Wireframe", &_drawRigidWireframe);
        ImGui::Checkbox("Show Rigid Contacts", &_showRigidContacts);
        ImGui::SliderFloat("Rigid Line Width", &_rigidLineWidth, 1.0f, 4.0f);
        ImGui::SliderFloat("Contact Point Size", &_rigidPointSize, 2.0f, 16.0f);

        ImGui::TextWrapped("Lab1 interaction: hover a dynamic rigid body and hold F to apply a point force along the view ray at the hit point.  Hold Alt + left mouse and drag a dynamic body to reposition it on a camera-facing plane.");
        if (_hoverHasHit) {
            ImGui::Text("Hover Body: %d", _hoverRigidBodyId);
        }
    }

    Common::CaseRenderResult MainScene::OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
        _viewportSize = desiredSize;

        if (_recompute) {
            _recompute = false;
            _sceneObject.ReplaceScene(GetScene(_sceneIdx));
            _cameraManager.Save(_sceneObject.Camera);
        }

        if (! _stopped) {
            ApplyRigidBodyLab1Controls();
            _world.StepFrame(ImGui::GetIO().DeltaTime);
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

        auto const & bodies = _world.GetRigidBodies().Bodies;
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            auto const & body = bodies[i];
            if (IsInternalTankBoundary(body)) continue; // the visible white fluid frame already represents the tank

            bool const selected = (i == _world.SelectedRigidBody());
            bool const hovered  = (_hoverHasHit && i == _hoverRigidBodyId);

            if (body.shape == RigidBodyShape::Box) {
                auto verts = GetRigidBoxVertices(body);
                auto span  = Engine::make_span_bytes<glm::vec3>(verts);

                if (_drawRigidSolid && _rigidBoxItem) {
                    _lineprogram.GetUniforms().SetByName("u_Color", ToGlm(body.color));
                    _rigidBoxItem->UpdateVertexBuffer("position", span);
                    _rigidBoxItem->Draw({ _lineprogram.Use() });
                }
                if (_drawRigidWireframe && _rigidBoxLineItem) {
                    glm::vec3 lineColor = body.isStatic ? glm::vec3(0.90f, 0.90f, 0.90f) : glm::vec3(1.0f, 1.0f, 1.0f);
                    if (selected && ! body.isStatic) lineColor = glm::vec3(1.0f, 0.95f, 0.20f);
                    if (hovered && ! body.isStatic) lineColor = glm::vec3(0.20f, 1.0f, 0.35f);
                    _lineprogram.GetUniforms().SetByName("u_Color", lineColor);
                    glLineWidth(_rigidLineWidth);
                    _rigidBoxLineItem->UpdateVertexBuffer("position", span);
                    _rigidBoxLineItem->Draw({ _lineprogram.Use() });
                    glLineWidth(1.0f);
                }
            } else {
                auto verts = GetRigidSphereVertices(body);
                auto span  = Engine::make_span_bytes<glm::vec3>(verts);

                if (_drawRigidSolid && _rigidSphereItem) {
                    _lineprogram.GetUniforms().SetByName("u_Color", ToGlm(body.color));
                    _rigidSphereItem->UpdateVertexBuffer("position", span);
                    _rigidSphereItem->Draw({ _lineprogram.Use() });
                }
                if (_drawRigidWireframe && _rigidSphereLineItem) {
                    glm::vec3 lineColor = body.isStatic ? glm::vec3(0.90f, 0.90f, 0.90f) : glm::vec3(1.0f, 1.0f, 1.0f);
                    if (selected && ! body.isStatic) lineColor = glm::vec3(1.0f, 0.95f, 0.20f);
                    if (hovered && ! body.isStatic) lineColor = glm::vec3(0.20f, 1.0f, 0.35f);
                    _lineprogram.GetUniforms().SetByName("u_Color", lineColor);
                    glLineWidth(_rigidLineWidth);
                    _rigidSphereLineItem->UpdateVertexBuffer("position", span);
                    _rigidSphereLineItem->Draw({ _lineprogram.Use() });
                    glLineWidth(1.0f);
                }
            }
        }

        if (_showRigidContacts && _rigidContactPointItem && ! _world.GetRigidBodies().Contacts.empty()) {
            auto contactVerts = GetRigidContactVertices();
            glPointSize(_rigidPointSize);
            _lineprogram.GetUniforms().SetByName("u_Color", glm::vec3(1.0f, 0.15f, 0.1f));
            _rigidContactPointItem->UpdateVertexBuffer("position", Engine::make_span_bytes<glm::vec3>(contactVerts));
            _rigidContactPointItem->Draw({ _lineprogram.Use() });
            glPointSize(1.0f);
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

    void MainScene::ApplyRigidBodyLab1Controls() {
        auto & rigid = _world.GetRigidBodies();
        if (! _hoverHasHit) return;
        if (! rigid.IsValidBody(_hoverRigidBodyId)) return;
        if (rigid.Bodies[_hoverRigidBodyId].isStatic) return;
        if (! ImGui::IsKeyDown(ImGuiKey_F)) return;

        _world.SetSelectedRigidBody(_hoverRigidBodyId);
        Eigen::Vector3f const pointForce = _world.RigidKeyboardForce() * SafeNormalized(_hoverRayDir, Eigen::Vector3f::UnitZ());
        rigid.ApplyForce(_hoverRigidBodyId, pointForce, _hoverHitPoint);
    }

    void MainScene::OnProcessInput(ImVec2 const & pos) {
        ImGuiIO const & io            = ImGui::GetIO();
        bool const      canvasHovered = ImGui::IsItemHovered();
        bool const      leftDown      = ImGui::IsMouseDown(ImGuiMouseButton_Left);
        bool const      altLeftClick  = canvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.KeyAlt;

        _hoverHasHit      = false;
        _hoverRigidBodyId = -1;
        if (canvasHovered) {
            int             hoverId  = -1;
            Eigen::Vector3f hoverHit = Eigen::Vector3f::Zero();
            Eigen::Vector3f rayOrigin = Eigen::Vector3f::Zero();
            Eigen::Vector3f rayDir    = Eigen::Vector3f::UnitZ();
            if (ComputePickRay(pos, rayOrigin, rayDir) && PickRigidBody(pos, hoverId, hoverHit)) {
                _hoverHasHit      = true;
                _hoverRigidBodyId = hoverId;
                _hoverHitPoint    = hoverHit;
                _hoverRayDir      = rayDir;
            }
        }

        if (_isDraggingRigidBody && ! leftDown) {
            _isDraggingRigidBody = false;
        }

        if (altLeftClick && _hoverHasHit && _world.GetRigidBodies().IsValidBody(_hoverRigidBodyId)) {
            auto const & body = _world.GetRigidBodies().Bodies[_hoverRigidBodyId];
            if (! body.isStatic) {
                _world.SetSelectedRigidBody(_hoverRigidBodyId);
                _dragPlaneNormal     = SafeNormalized(ToEigen(glm::normalize(_sceneObject.Camera.Target - _sceneObject.Camera.Eye)), Eigen::Vector3f::UnitZ());
                _dragPlanePoint      = _hoverHitPoint;
                _dragBodyOffset      = body.x - _hoverHitPoint;
                _isDraggingRigidBody = true;
                UpdateDraggedRigidBody(pos);
            }
        }

        bool const draggingNow = _isDraggingRigidBody && leftDown;
        if (draggingNow) {
            UpdateDraggedRigidBody(pos);
        } else {
            _cameraManager.ProcessInput(_sceneObject.Camera, pos);
        }
    }

    void MainScene::ResetSystem() {
        _world.Reset();
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        _isDraggingRigidBody = false;
        _hoverHasHit         = false;
        _hoverRigidBodyId    = -1;
        RebuildParticleRenderItem();
        RebuildRigidBodyRenderItem();
    }

    void MainScene::ResetSystem(int res) {
        _world.Reset(res);
        _r      = _world.GetFluid().m_particleRadius;
        _sphere = Engine::Model { Engine::Sphere(4, _r), 0 };
        _isDraggingRigidBody = false;
        _hoverHasHit         = false;
        _hoverRigidBodyId    = -1;
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
        _rigidBoxItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Triangles);
        _rigidBoxItem->UpdateElementBuffer(kBoxTriIndex);

        _rigidBoxLineItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines);
        _rigidBoxLineItem->UpdateElementBuffer(kBoxLineIndex);

        auto const & sphereMesh = GetUnitRigidSphereMesh();
        _rigidSphereItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Triangles);
        _rigidSphereItem->UpdateElementBuffer(sphereMesh.triIndices);

        _rigidSphereLineItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Lines);
        _rigidSphereLineItem->UpdateElementBuffer(sphereMesh.lineIndices);

        _rigidContactPointItem.emplace(
            Engine::GL::VertexLayout().Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
            Engine::GL::PrimitiveType::Points);
    }

    bool MainScene::ComputePickRay(ImVec2 const & pos, Eigen::Vector3f & rayOrigin, Eigen::Vector3f & rayDir) const {
        if (_world.GetRigidBodies().Bodies.empty() || _viewportSize.first == 0 || _viewportSize.second == 0) return false;
        WorldRay const ray = ScreenPointToWorldRay(pos);
        rayOrigin = ToEigen(ray.Origin);
        rayDir    = SafeNormalized(ToEigen(ray.Direction), Eigen::Vector3f::UnitZ());
        return true;
    }

    bool MainScene::PickRigidBody(ImVec2 const & pos, int & bodyId, Eigen::Vector3f & hitPoint) const {
        Eigen::Vector3f rayOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3f rayDir    = Eigen::Vector3f::UnitZ();
        if (! ComputePickRay(pos, rayOrigin, rayDir)) return false;

        float closestT = std::numeric_limits<float>::max();
        int   picked   = -1;
        auto const & bodies = _world.GetRigidBodies().Bodies;
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            auto const & body = bodies[i];
            if (body.isStatic) continue;

            float           t   = 0.0f;
            Eigen::Vector3f hit = Eigen::Vector3f::Zero();
            bool            ok  = false;
            if (body.shape == RigidBodyShape::Sphere) {
                ok = IntersectRaySphere(rayOrigin, rayDir, body, t, hit);
            } else {
                ok = IntersectRayOBB(rayOrigin, rayDir, body, t, hit);
            }
            if (! ok) continue;
            if (t < closestT) {
                closestT = t;
                picked   = i;
                hitPoint = hit;
            }
        }

        bodyId = picked;
        return picked >= 0;
    }

    bool MainScene::UpdateDraggedRigidBody(ImVec2 const & pos) {
        int const id = _world.SelectedRigidBody();
        auto & rigid = _world.GetRigidBodies();
        if (! _isDraggingRigidBody || ! rigid.IsValidBody(id)) return false;

        Eigen::Vector3f rayOrigin = Eigen::Vector3f::Zero();
        Eigen::Vector3f rayDir    = Eigen::Vector3f::UnitZ();
        if (! ComputePickRay(pos, rayOrigin, rayDir)) return false;

        Eigen::Vector3f hitPoint = Eigen::Vector3f::Zero();
        if (! IntersectRayPlane(rayOrigin, rayDir, _dragPlanePoint, _dragPlaneNormal, hitPoint)) return false;

        auto & body = rigid.Bodies[id];
        if (body.isStatic) return false;
        body.x = hitPoint + _dragBodyOffset;
        body.v.setZero();
        body.w.setZero();
        body.force.setZero();
        body.torque.setZero();
        rigid.Contacts.clear();
        return true;
    }

    std::vector<glm::vec3> MainScene::GetRigidBoxVertices(RigidBody const & body) const {
        auto const             corners = body.GetWorldCorners();
        std::vector<glm::vec3> verts(8);
        for (int i = 0; i < 8; ++i) verts[i] = ToGlm(corners[i]);
        return verts;
    }

    std::vector<glm::vec3> MainScene::GetRigidSphereVertices(RigidBody const & body) const {
        SphereMeshData const & sphereMesh = GetUnitRigidSphereMesh();
        std::vector<glm::vec3> verts;
        verts.reserve(sphereMesh.vertices.size());
        float const           radius = 0.5f * body.dim.x();
        Eigen::Matrix3f const R      = body.GetRotationMatrix();
        for (glm::vec3 const & p : sphereMesh.vertices) {
            Eigen::Vector3f const local(p.x, p.y, p.z);
            Eigen::Vector3f const world = body.x + radius * (R * local);
            verts.push_back(ToGlm(world));
        }
        return verts;
    }

    std::vector<glm::vec3> MainScene::GetRigidContactVertices() const {
        std::vector<glm::vec3> verts;
        auto const & rigid = _world.GetRigidBodies();
        verts.reserve(rigid.Contacts.size());
        for (auto const & c : rigid.Contacts) {
            bool const touchesInternalTank = rigid.IsValidBody(c.idA) && IsInternalTankBoundary(rigid.Bodies[c.idA])
                || rigid.IsValidBody(c.idB) && IsInternalTankBoundary(rigid.Bodies[c.idB]);
            if (touchesInternalTank) continue;
            verts.push_back(ToGlm(c.position));
        }
        return verts;
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
} // namespace VCX::MainScene
