#include "Simulation/RigidBodySystem.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/narrowphase/collision.h>

namespace VCX::MainScene {

    namespace {
        constexpr float kEps = 1e-6f;

        Eigen::Vector3f SafeNormalized(Eigen::Vector3f const & v, Eigen::Vector3f const & fallback = Eigen::Vector3f::UnitY()) {
            float const n = v.norm();
            return n > kEps ? (v / n) : fallback;
        }

        using CollisionGeometryPtr = std::shared_ptr<fcl::CollisionGeometry<float>>;

        CollisionGeometryPtr MakeCollisionGeometry(RigidBody const & body) {
            switch (body.shape) {
            case RigidBodyShape::Sphere:
                return CollisionGeometryPtr(new fcl::Sphere<float>(0.5f * body.dim.x()));
            case RigidBodyShape::Box:
            default:
                return CollisionGeometryPtr(new fcl::Box<float>(body.dim.x(), body.dim.y(), body.dim.z()));
            }
        }

        Eigen::Quaternionf SmallRotationFromAngularVelocity(Eigen::Vector3f const & w, float dt, Eigen::Quaternionf const & q) {
            Eigen::Quaternionf omegaQ(0.0f, w.x(), w.y(), w.z());
            Eigen::Quaternionf qdot = omegaQ * q;
            Eigen::Quaternionf out  = q;
            out.coeffs() += 0.5f * dt * qdot.coeffs();
            out.normalize();
            return out;
        }
    } // namespace

    glm::vec3 ToGlm(Eigen::Vector3f const & v) {
        return glm::vec3(v.x(), v.y(), v.z());
    }

    Eigen::Vector3f ToEigen(glm::vec3 const & v) {
        return Eigen::Vector3f(v.x, v.y, v.z);
    }

    float RigidBody::GetBoundingSphereRadius() const {
        if (shape == RigidBodyShape::Sphere) {
            return 0.5f * dim.x();
        }
        return 0.5f * dim.norm();
    }

    void RigidBody::UpdateMassProperties() {
        if (isStatic || mass <= kEps) {
            mass    = std::numeric_limits<float>::infinity();
            invMass = 0.0f;
            inertiaBody.setZero();
            inertiaBodyInv.setZero();
            return;
        }

        invMass = 1.0f / mass;
        inertiaBody.setZero();

        switch (shape) {
        case RigidBodyShape::Sphere: {
            float const r = 0.5f * dim.x();
            float const I = 0.4f * mass * r * r;
            inertiaBody.diagonal().setConstant(I);
            break;
        }
        case RigidBodyShape::Box:
        default: {
            float const x2    = dim.x() * dim.x();
            float const y2    = dim.y() * dim.y();
            float const z2    = dim.z() * dim.z();
            inertiaBody(0, 0) = mass * (y2 + z2) / 12.0f;
            inertiaBody(1, 1) = mass * (x2 + z2) / 12.0f;
            inertiaBody(2, 2) = mass * (x2 + y2) / 12.0f;
            break;
        }
        }

        inertiaBodyInv = inertiaBody.inverse();
    }

    Eigen::Matrix3f RigidBody::GetRotationMatrix() const {
        return q.normalized().toRotationMatrix();
    }

    Eigen::Matrix3f RigidBody::GetWorldInertiaInv() const {
        if (isStatic) return Eigen::Matrix3f::Zero();
        Eigen::Matrix3f const R = GetRotationMatrix();
        return R * inertiaBodyInv * R.transpose();
    }

    std::array<Eigen::Vector3f, 8> RigidBody::GetWorldCorners() const {
        if (shape == RigidBodyShape::Sphere) {
            float const r = 0.5f * dim.x();
            return {
                x + Eigen::Vector3f(-r,  r,  r),
                x + Eigen::Vector3f( r,  r,  r),
                x + Eigen::Vector3f( r,  r, -r),
                x + Eigen::Vector3f(-r,  r, -r),
                x + Eigen::Vector3f(-r, -r,  r),
                x + Eigen::Vector3f( r, -r,  r),
                x + Eigen::Vector3f( r, -r, -r),
                x + Eigen::Vector3f(-r, -r, -r),
            };
        }

        Eigen::Matrix3f const R  = GetRotationMatrix();
        Eigen::Vector3f const hx = 0.5f * dim.x() * R.col(0);
        Eigen::Vector3f const hy = 0.5f * dim.y() * R.col(1);
        Eigen::Vector3f const hz = 0.5f * dim.z() * R.col(2);
        return {
            x - hx + hy + hz,
            x + hx + hy + hz,
            x + hx + hy - hz,
            x - hx + hy - hz,
            x - hx - hy + hz,
            x + hx - hy + hz,
            x + hx - hy - hz,
            x - hx - hy - hz,
        };
    }

    std::pair<Eigen::Vector3f, Eigen::Vector3f> RigidBody::GetWorldAABB() const {
        auto const corners = GetWorldCorners();
        Eigen::Vector3f mn = corners[0];
        Eigen::Vector3f mx = corners[0];
        for (auto const & c : corners) {
            mn = mn.cwiseMin(c);
            mx = mx.cwiseMax(c);
        }
        return { mn, mx };
    }

    Eigen::Vector3f RigidBody::WorldToLocal(Eigen::Vector3f const & p) const {
        return GetRotationMatrix().transpose() * (p - x);
    }

    Eigen::Vector3f RigidBody::LocalToWorld(Eigen::Vector3f const & p) const {
        return x + GetRotationMatrix() * p;
    }

    bool RigidBody::ContainsPoint(Eigen::Vector3f const & worldPoint) const {
        Eigen::Vector3f const local = WorldToLocal(worldPoint);
        if (shape == RigidBodyShape::Sphere) {
            float const r = 0.5f * dim.x();
            return local.squaredNorm() <= r * r;
        }
        Eigen::Vector3f const half = 0.5f * dim;
        return std::abs(local.x()) <= half.x()
            && std::abs(local.y()) <= half.y()
            && std::abs(local.z()) <= half.z();
    }

    Eigen::Vector3f RigidBody::ClosestSurfacePoint(Eigen::Vector3f const & worldPoint) const {
        Eigen::Vector3f const local = WorldToLocal(worldPoint);
        if (shape == RigidBodyShape::Sphere) {
            float const r = 0.5f * dim.x();
            return LocalToWorld(SafeNormalized(local, Eigen::Vector3f::UnitY()) * r);
        }

        Eigen::Vector3f const half = 0.5f * dim;
        Eigen::Vector3f clamped = local.cwiseMax(-half).cwiseMin(half);
        if (ContainsPoint(worldPoint)) {
            Eigen::Vector3f const distToFace = half - local.cwiseAbs();
            int axis = 0;
            if (distToFace.y() < distToFace[axis]) axis = 1;
            if (distToFace.z() < distToFace[axis]) axis = 2;
            clamped[axis] = (local[axis] >= 0.0f ? half[axis] : -half[axis]);
        }
        return LocalToWorld(clamped);
    }

    Eigen::Vector3f RigidBody::SurfaceNormalAt(Eigen::Vector3f const & worldPoint) const {
        Eigen::Vector3f const local = WorldToLocal(worldPoint);
        if (shape == RigidBodyShape::Sphere) {
            return GetRotationMatrix() * SafeNormalized(local, Eigen::Vector3f::UnitY());
        }

        Eigen::Vector3f const half       = 0.5f * dim;
        Eigen::Vector3f const distToFace = half - local.cwiseAbs();
        int axis = 0;
        if (distToFace.y() < distToFace[axis]) axis = 1;
        if (distToFace.z() < distToFace[axis]) axis = 2;
        Eigen::Vector3f n = Eigen::Vector3f::Zero();
        n[axis] = local[axis] >= 0.0f ? 1.0f : -1.0f;
        return GetRotationMatrix() * n;
    }

    void RigidBodySystem::Clear() {
        Bodies.clear();
        Contacts.clear();
    }

    void RigidBodySystem::Reset() {
        Clear();
    }

    int RigidBodySystem::AddBody(RigidBody body) {
        body.q.normalize();
        body.UpdateMassProperties();
        Bodies.push_back(std::move(body));
        return static_cast<int>(Bodies.size()) - 1;
    }

    void RigidBodySystem::SetupDefaultScene(RigidBodyPreset preset) {
        Clear();

        // These values are copied from the stable stacking setting in Lab1.
        // Compared with the earlier Lab4 version, the higher substep/iteration count,
        // contact sorting, alternating sweep, and resting stabilization are the main
        // reasons why objects do not keep shaking while resting on a static surface.
        Substeps                     = 16;
        ImpulseIterations            = 34;
        LinearDamping                = 0.01f;
        AngularDamping               = 0.015f;
        RestitutionVelocityThreshold = 0.5f;
        PositionCorrectionPercent    = 0.82f;
        PositionCorrectionSlop       = 0.0005f;
        EnableFriction               = true;
        SortContactsForStability     = true;
        EnableRestingStabilization   = true;
        AlternateIterationSweep      = true;
        UseSequentialImpulseCaching  = true;
        RestingLinearThreshold       = 0.008f;
        RestingAngularThreshold      = 0.015f;

        auto addTankWall = [&](std::string name, Eigen::Vector3f const & dim, Eigen::Vector3f const & x) {
            RigidBody wall;
            wall.name        = std::move(name);
            wall.shape       = RigidBodyShape::Box;
            wall.dim         = dim;
            wall.x           = x;
            wall.mass        = 1.0f;
            wall.isStatic    = true;
            wall.useGravity  = false;
            wall.restitution = 0.0f;
            wall.friction    = 0.95f;
            wall.color       = Eigen::Vector3f(0.36f, 0.38f, 0.43f);
            AddBody(wall);
        };

        auto addLab1StyleTankBoundary = [&]() {
            float constexpr inner = 0.43f;
            float constexpr thick = 0.08f;
            float constexpr span  = 1.08f;
            float constexpr c     = inner + 0.5f * thick;

            // Six static walls are used only for rigid-body collision response.
            // They are handled by the same FCL + sequential impulse pipeline as Lab1,
            // instead of a separate AABB post-correction.  This is the key change that
            // reduces the resting jitter seen against the tank frame.
            addTankWall("tank_neg_x", Eigen::Vector3f(thick, span,  span), Eigen::Vector3f(-c, 0.0f, 0.0f));
            addTankWall("tank_pos_x", Eigen::Vector3f(thick, span,  span), Eigen::Vector3f( c, 0.0f, 0.0f));
            addTankWall("tank_neg_y", Eigen::Vector3f(span,  thick, span), Eigen::Vector3f(0.0f, -c, 0.0f));
            addTankWall("tank_pos_y", Eigen::Vector3f(span,  thick, span), Eigen::Vector3f(0.0f,  c, 0.0f));
            addTankWall("tank_neg_z", Eigen::Vector3f(span,  span,  thick), Eigen::Vector3f(0.0f, 0.0f, -c));
            addTankWall("tank_pos_z", Eigen::Vector3f(span,  span,  thick), Eigen::Vector3f(0.0f, 0.0f,  c));
        };

        // The Lab4 fluid tank is currently [-0.5, 0.5]^3.  These presets keep the
        // Lab1 rigid-body method, but scale the demos down so that they can later
        // be rasterized into the fluid grid for solid-fluid coupling.
        switch (preset) {
        case RigidBodyPreset::BoxCollision: {
            RigidBody a;
            a.name        = "box_A";
            a.shape       = RigidBodyShape::Box;
            a.dim         = Eigen::Vector3f(0.16f, 0.11f, 0.13f);
            a.x           = Eigen::Vector3f(-0.28f, 0.02f, 0.0f);
            a.q           = Eigen::Quaternionf(Eigen::AngleAxisf(0.28f, Eigen::Vector3f::UnitY()));
            a.v           = Eigen::Vector3f(0.65f, 0.0f, 0.0f);
            a.w           = Eigen::Vector3f(0.0f, 1.0f, 0.15f);
            a.mass        = 0.45f;
            a.restitution = 0.18f;
            a.friction    = 0.55f;
            a.useGravity  = false;
            a.color       = Eigen::Vector3f(0.90f, 0.55f, 0.55f);
            AddBody(a);

            RigidBody b;
            b.name        = "box_B";
            b.shape       = RigidBodyShape::Box;
            b.dim         = Eigen::Vector3f(0.17f, 0.12f, 0.12f);
            b.x           = Eigen::Vector3f(0.25f, 0.02f, 0.0f);
            b.q           = Eigen::Quaternionf(Eigen::AngleAxisf(-0.18f, Eigen::Vector3f::UnitY()));
            b.v           = Eigen::Vector3f(-0.55f, 0.0f, 0.0f);
            b.w           = Eigen::Vector3f(0.0f, -0.8f, 0.0f);
            b.mass        = 0.50f;
            b.restitution = 0.18f;
            b.friction    = 0.55f;
            b.useGravity  = false;
            b.color       = Eigen::Vector3f(0.58f, 0.72f, 0.95f);
            AddBody(b);
            addLab1StyleTankBoundary();
            break;
        }
        case RigidBodyPreset::MixedStack: {
            RigidBody floor;
            floor.name        = "static_floor_block";
            floor.shape       = RigidBodyShape::Box;
            floor.dim         = Eigen::Vector3f(0.80f, 0.05f, 0.80f);
            floor.x           = Eigen::Vector3f(0.0f, -0.42f, 0.0f);
            floor.mass        = 1.0f;
            floor.isStatic    = true;
            floor.useGravity  = false;
            floor.restitution = 0.02f;
            floor.friction    = 0.90f;
            floor.color       = Eigen::Vector3f(0.45f, 0.48f, 0.52f);
            AddBody(floor);

            for (int i = 0; i < 4; ++i) {
                RigidBody body;
                body.name        = "mixed_body_" + std::to_string(i);
                body.useGravity  = true;
                body.mass        = 0.25f + 0.04f * float(i);
                body.restitution = 0.03f;
                body.friction    = 0.75f;
                body.x           = Eigen::Vector3f((i % 2 == 0 ? -0.07f : 0.07f), -0.33f + 0.09f * float(i), 0.0f);
                body.color       = Eigen::Vector3f(0.90f - 0.06f * float(i), 0.68f + 0.05f * float(i), 0.52f + 0.05f * float(i));
                if (i % 2 == 0) {
                    body.shape = RigidBodyShape::Box;
                    body.dim   = Eigen::Vector3f(0.13f, 0.08f, 0.13f);
                    body.q     = Eigen::Quaternionf(Eigen::AngleAxisf(0.12f * float(i), Eigen::Vector3f::UnitY()));
                } else {
                    body.shape = RigidBodyShape::Sphere;
                    body.dim   = Eigen::Vector3f::Constant(0.11f);
                }
                AddBody(body);
            }
            addLab1StyleTankBoundary();
            break;
        }
        case RigidBodyPreset::FluidCouplingMixed:
        default: {
            RigidBody box;
            box.name        = "fluid_coupling_box";
            box.shape       = RigidBodyShape::Box;
            box.dim         = Eigen::Vector3f(0.18f, 0.10f, 0.14f);
            box.x           = Eigen::Vector3f(-0.16f, 0.05f, 0.00f);
            box.q           = Eigen::Quaternionf(Eigen::AngleAxisf(0.25f, Eigen::Vector3f::UnitY()));
            box.mass        = 0.45f;
            box.restitution = 0.08f;
            box.friction    = 0.65f;
            box.useGravity  = true;
            box.color       = Eigen::Vector3f(0.92f, 0.58f, 0.50f);
            AddBody(box);

            RigidBody sphere;
            sphere.name        = "fluid_coupling_sphere";
            sphere.shape       = RigidBodyShape::Sphere;
            sphere.dim         = Eigen::Vector3f::Constant(0.13f);
            sphere.x           = Eigen::Vector3f(0.12f, 0.08f, 0.08f);
            sphere.mass        = 0.25f;
            sphere.restitution = 0.10f;
            sphere.friction    = 0.25f;
            sphere.useGravity  = true;
            sphere.color       = Eigen::Vector3f(0.55f, 0.72f, 0.95f);
            AddBody(sphere);

            RigidBody obstacle;
            obstacle.name        = "static_coupling_box";
            obstacle.shape       = RigidBodyShape::Box;
            obstacle.dim         = Eigen::Vector3f(0.12f, 0.12f, 0.20f);
            obstacle.x           = Eigen::Vector3f(0.20f, -0.18f, -0.08f);
            obstacle.q           = Eigen::Quaternionf(Eigen::AngleAxisf(-0.35f, Eigen::Vector3f::UnitY()));
            obstacle.mass        = 1.0f;
            obstacle.isStatic    = true;
            obstacle.useGravity  = false;
            obstacle.restitution = 0.05f;
            obstacle.friction    = 0.80f;
            obstacle.color       = Eigen::Vector3f(0.50f, 0.85f, 0.62f);
            AddBody(obstacle);
            addLab1StyleTankBoundary();
            break;
        }
        }
    }

    bool RigidBodySystem::IsValidBody(int id) const {
        return id >= 0 && id < static_cast<int>(Bodies.size());
    }

    void RigidBodySystem::ClearForces() {
        for (auto & body : Bodies) {
            body.force.setZero();
            body.torque.setZero();
        }
    }

    void RigidBodySystem::ApplyForce(int id, Eigen::Vector3f const & forceWorld, Eigen::Vector3f const & worldPoint) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        if (b.isStatic) return;
        b.force += forceWorld;
        b.torque += (worldPoint - b.x).cross(forceWorld);
    }

    void RigidBodySystem::ApplyForceToCenter(int id, Eigen::Vector3f const & forceWorld) {
        if (! IsValidBody(id)) return;
        ApplyForce(id, forceWorld, Bodies[id].x);
    }

    void RigidBodySystem::ApplyTorque(int id, Eigen::Vector3f const & torqueWorld) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        if (b.isStatic) return;
        b.torque += torqueWorld;
    }

    void RigidBodySystem::ApplyImpulse(int id, Eigen::Vector3f const & impulseWorld, Eigen::Vector3f const & worldPoint) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        if (b.isStatic) return;
        b.v += b.invMass * impulseWorld;
        b.w += b.GetWorldInertiaInv() * (worldPoint - b.x).cross(impulseWorld);
    }

    void RigidBodySystem::ApplyImpulseToCenter(int id, Eigen::Vector3f const & impulseWorld) {
        if (! IsValidBody(id)) return;
        ApplyImpulse(id, impulseWorld, Bodies[id].x);
    }

    int RigidBodySystem::GetFirstDynamicBody() const {
        for (int i = 0; i < static_cast<int>(Bodies.size()); ++i) {
            if (! Bodies[i].isStatic) return i;
        }
        return -1;
    }

    Eigen::Vector3f RigidBodySystem::VelocityAtPoint(int id, Eigen::Vector3f const & worldPoint) const {
        if (! IsValidBody(id)) return Eigen::Vector3f::Zero();
        auto const & b = Bodies[id];
        return b.v + b.w.cross(worldPoint - b.x);
    }


    void RigidBodySystem::CollectSurfaceSamples(int bodyId, int samplesPerAxis, std::vector<RigidSurfaceSample> & samples) const {
        if (! IsValidBody(bodyId)) return;
        auto const & body = Bodies[bodyId];

        int const n = std::max(1, samplesPerAxis);
        Eigen::Matrix3f const R = body.GetRotationMatrix();

        if (body.shape == RigidBodyShape::Sphere) {
            float const radius = 0.5f * body.dim.x();
            int const sampleCount = std::max(12, 6 * n * n);
            float const area = 4.0f * 3.14159265358979323846f * radius * radius / static_cast<float>(sampleCount);
            float const goldenAngle = 3.14159265358979323846f * (3.0f - std::sqrt(5.0f));

            for (int i = 0; i < sampleCount; ++i) {
                float const y = 1.0f - 2.0f * (float(i) + 0.5f) / float(sampleCount);
                float const r = std::sqrt(std::max(0.0f, 1.0f - y * y));
                float const theta = goldenAngle * float(i);
                Eigen::Vector3f const localNormal(r * std::cos(theta), y, r * std::sin(theta));
                Eigen::Vector3f const normal = SafeNormalized(R * localNormal);

                RigidSurfaceSample sample;
                sample.bodyId   = bodyId;
                sample.normal   = normal;
                sample.position = body.x + radius * normal;
                sample.area     = area;
                samples.push_back(sample);
            }
            return;
        }

        Eigen::Vector3f const half = 0.5f * body.dim;
        auto emitFace = [&](int axis, float sign) {
            int const uAxis = (axis + 1) % 3;
            int const vAxis = (axis + 2) % 3;
            float const faceArea = body.dim[uAxis] * body.dim[vAxis];
            float const sampleArea = faceArea / float(n * n);

            Eigen::Vector3f localNormal = Eigen::Vector3f::Zero();
            localNormal[axis] = sign;
            Eigen::Vector3f const worldNormal = SafeNormalized(R * localNormal);

            for (int iu = 0; iu < n; ++iu) {
                for (int iv = 0; iv < n; ++iv) {
                    float const fu = (float(iu) + 0.5f) / float(n) * 2.0f - 1.0f;
                    float const fv = (float(iv) + 0.5f) / float(n) * 2.0f - 1.0f;

                    Eigen::Vector3f localPoint = Eigen::Vector3f::Zero();
                    localPoint[axis]  = sign * half[axis];
                    localPoint[uAxis] = fu * half[uAxis];
                    localPoint[vAxis] = fv * half[vAxis];

                    RigidSurfaceSample sample;
                    sample.bodyId   = bodyId;
                    sample.position = body.LocalToWorld(localPoint);
                    sample.normal   = worldNormal;
                    sample.area     = sampleArea;
                    samples.push_back(sample);
                }
            }
        };

        emitFace(0,  1.0f);
        emitFace(0, -1.0f);
        emitFace(1,  1.0f);
        emitFace(1, -1.0f);
        emitFace(2,  1.0f);
        emitFace(2, -1.0f);
    }

    void RigidBodySystem::SetBodyMass(int id, float mass) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        if (b.isStatic) return;
        b.mass = std::max(mass, 1e-4f);
        b.UpdateMassProperties();
    }

    void RigidBodySystem::SetBodyDim(int id, Eigen::Vector3f const & dim) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        b.dim = dim.cwiseMax(Eigen::Vector3f::Constant(1e-4f));
        if (b.shape == RigidBodyShape::Sphere) {
            float const d = std::max({ b.dim.x(), b.dim.y(), b.dim.z() });
            b.dim = Eigen::Vector3f::Constant(d);
        }
        b.UpdateMassProperties();
    }

    void RigidBodySystem::SetBodyStatic(int id, bool isStatic) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        if (b.isStatic == isStatic) return;
        b.isStatic = isStatic;
        if (b.isStatic) {
            b.v.setZero();
            b.w.setZero();
            b.force.setZero();
            b.torque.setZero();
        } else if (! std::isfinite(b.mass)) {
            b.mass = 1.0f;
        }
        b.UpdateMassProperties();
    }

    void RigidBodySystem::SetBodyGravity(int id, bool useGravity) {
        if (! IsValidBody(id)) return;
        Bodies[id].useGravity = useGravity;
    }

    void RigidBodySystem::SetBodyShape(int id, RigidBodyShape shape) {
        if (! IsValidBody(id)) return;
        auto & b = Bodies[id];
        b.shape = shape;
        if (shape == RigidBodyShape::Sphere) {
            float const d = std::max({ b.dim.x(), b.dim.y(), b.dim.z() });
            b.dim = Eigen::Vector3f::Constant(d);
        }
        b.UpdateMassProperties();
    }

    void RigidBodySystem::Step(float dt, int substeps) {
        if (dt <= 0.0f || Bodies.empty()) return;
        int const steps = std::max(1, substeps > 0 ? substeps : Substeps);
        float const h = dt / static_cast<float>(steps);

        for (int s = 0; s < steps; ++s) {
            Integrate(h);
            Contacts.clear();
            DetectCollisionsFCL();
            if (SortContactsForStability) sortContactsForStability();
            SolveContacts();
            PositionalCorrection();
            if (EnableRestingStabilization) StabilizeRestingContacts();
        }
        ClearForces();
    }

    void RigidBodySystem::Integrate(float dt) {
        for (auto & b : Bodies) {
            if (b.isStatic) continue;

            Eigen::Vector3f gravityForce = Eigen::Vector3f::Zero();
            if (b.useGravity && std::isfinite(b.mass)) {
                gravityForce = b.mass * Gravity;
            }
            Eigen::Vector3f const linearAcc  = b.invMass * (b.force + gravityForce);
            Eigen::Vector3f const angularAcc = b.GetWorldInertiaInv() * b.torque;

            // Same explicit Euler style as Lab1: position/orientation are advanced
            // from the current velocities, then velocities are advanced by current forces.
            b.x += dt * b.v;
            b.q = SmallRotationFromAngularVelocity(b.w, dt, b.q);

            b.v += dt * linearAcc;
            b.w += dt * angularAcc;

            b.v *= std::max(0.0f, 1.0f - LinearDamping * dt);
            b.w *= std::max(0.0f, 1.0f - AngularDamping * dt);
        }
    }

    void RigidBodySystem::DetectCollisionsFCL() {
        int const n = static_cast<int>(Bodies.size());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (Bodies[i].isStatic && Bodies[j].isStatic) continue;
                collisionDetectPairFCL(i, j);
            }
        }
    }

    void RigidBodySystem::collisionDetectPairFCL(int idA, int idB) {
        RigidBody const & a = Bodies[idA];
        RigidBody const & b = Bodies[idB];

        auto geomA = MakeCollisionGeometry(a);
        auto geomB = MakeCollisionGeometry(b);

        fcl::CollisionObject<float> shapeA(geomA, fcl::Transform3f(Eigen::Translation3f(a.x) * a.q));
        fcl::CollisionObject<float> shapeB(geomB, fcl::Transform3f(Eigen::Translation3f(b.x) * b.q));

        fcl::CollisionRequest<float> request(8, true);
        fcl::CollisionResult<float>  result;
        fcl::collide(&shapeA, &shapeB, request, result);
        if (! result.isCollision()) return;

        std::vector<fcl::Contact<float>> contacts;
        result.getContacts(contacts);
        for (auto const & c : contacts) {
            RigidContact contact;
            contact.idA         = idA;
            contact.idB         = idB;
            contact.position    = c.pos;
            contact.normal      = SafeNormalized(c.normal, Eigen::Vector3f::UnitY());
            contact.penetration = std::max(0.0f, c.penetration_depth);
            Contacts.push_back(contact);
        }
    }

    void RigidBodySystem::sortContactsForStability() {
        std::stable_sort(Contacts.begin(), Contacts.end(), [&](RigidContact const & lhs, RigidContact const & rhs) {
            bool const lhsTouchesStatic = Bodies[lhs.idA].isStatic || Bodies[lhs.idB].isStatic;
            bool const rhsTouchesStatic = Bodies[rhs.idA].isStatic || Bodies[rhs.idB].isStatic;
            auto const lhsKey = std::make_tuple(lhsTouchesStatic ? 0 : 1, lhs.position.y(), -lhs.penetration);
            auto const rhsKey = std::make_tuple(rhsTouchesStatic ? 0 : 1, rhs.position.y(), -rhs.penetration);
            return lhsKey < rhsKey;
        });
    }

    void RigidBodySystem::SolveContacts() {
        if (Contacts.empty()) return;

        for (auto & contact : Contacts) {
            contact.accumulatedNormalImpulse  = 0.0f;
            contact.accumulatedTangentImpulse = 0.0f;
            contact.tangent.setZero();
            contact.tangentInitialized = false;
        }

        for (int iter = 0; iter < ImpulseIterations; ++iter) {
            bool const reverse = AlternateIterationSweep && (iter % 2 == 1);
            if (! reverse) {
                for (auto & contact : Contacts) resolveVelocityContact(contact);
            } else {
                for (auto it = Contacts.rbegin(); it != Contacts.rend(); ++it) resolveVelocityContact(*it);
            }
        }
    }

    void RigidBodySystem::resolveVelocityContact(RigidContact & contact) {
        auto & a = Bodies[contact.idA];
        auto & b = Bodies[contact.idB];
        if (a.isStatic && b.isStatic) return;

        Eigen::Vector3f const n  = SafeNormalized(contact.normal);
        Eigen::Vector3f const rA = contact.position - a.x;
        Eigen::Vector3f const rB = contact.position - b.x;

        Eigen::Matrix3f const invIA = a.GetWorldInertiaInv();
        Eigen::Matrix3f const invIB = b.GetWorldInertiaInv();

        Eigen::Vector3f const vA          = a.v + a.w.cross(rA);
        Eigen::Vector3f const vB          = b.v + b.w.cross(rB);
        Eigen::Vector3f const rv          = vB - vA;
        float const           normalSpeed = rv.dot(n);

        Eigen::Vector3f const rnA     = rA.cross(n);
        Eigen::Vector3f const rnB     = rB.cross(n);
        float const           kNormal = a.invMass + b.invMass
            + n.dot((invIA * rnA).cross(rA) + (invIB * rnB).cross(rB));
        if (kNormal < kEps) return;

        float const restitution = (std::abs(normalSpeed) < RestitutionVelocityThreshold)
            ? 0.0f
            : std::min(a.restitution, b.restitution);

        float lambdaN = -normalSpeed / kNormal;
        if (contact.accumulatedNormalImpulse <= kEps && normalSpeed < -RestitutionVelocityThreshold) {
            lambdaN += (-restitution * normalSpeed) / kNormal;
        }

        float const oldNormalImpulse = contact.accumulatedNormalImpulse;
        if (UseSequentialImpulseCaching) {
            contact.accumulatedNormalImpulse = std::max(0.0f, oldNormalImpulse + lambdaN);
            lambdaN = contact.accumulatedNormalImpulse - oldNormalImpulse;
        } else {
            lambdaN = std::max(0.0f, lambdaN);
        }

        if (lambdaN > 0.0f) {
            Eigen::Vector3f const impulseN = lambdaN * n;
            ApplyImpulse(contact.idA, -impulseN, contact.position);
            ApplyImpulse(contact.idB,  impulseN, contact.position);
        }

        if (! EnableFriction) return;

        Eigen::Vector3f const newVA       = a.v + a.w.cross(rA);
        Eigen::Vector3f const newVB       = b.v + b.w.cross(rB);
        Eigen::Vector3f const newRV       = newVB - newVA;
        Eigen::Vector3f       tangent     = newRV - newRV.dot(n) * n;
        float const           tangentNorm = tangent.norm();
        if (tangentNorm < kEps) {
            contact.accumulatedTangentImpulse = 0.0f;
            contact.tangent.setZero();
            contact.tangentInitialized = false;
            return;
        }
        tangent /= tangentNorm;

        if (! contact.tangentInitialized) {
            contact.tangent = tangent;
            contact.tangentInitialized = true;
        } else {
            float const alignment = contact.tangent.dot(tangent);
            if (alignment < 0.0f) {
                contact.tangent *= -1.0f;
                contact.accumulatedTangentImpulse *= -1.0f;
                tangent *= -1.0f;
            }
            if (std::abs(contact.tangent.dot(tangent)) < 0.85f) {
                contact.tangent = tangent;
                contact.accumulatedTangentImpulse = 0.0f;
            }
        }
        tangent = contact.tangentInitialized ? contact.tangent : tangent;

        Eigen::Vector3f const rtA = rA.cross(tangent);
        Eigen::Vector3f const rtB = rB.cross(tangent);
        float const kTangent = a.invMass + b.invMass
            + tangent.dot((invIA * rtA).cross(rA) + (invIB * rtB).cross(rB));
        if (kTangent < kEps) return;

        float lambdaT = -newRV.dot(tangent) / kTangent;
        float const mu = std::sqrt(std::max(0.0f, a.friction * b.friction));
        float const jtMax = mu * (UseSequentialImpulseCaching ? contact.accumulatedNormalImpulse : std::max(0.0f, lambdaN));

        if (UseSequentialImpulseCaching) {
            float const oldTangentImpulse = contact.accumulatedTangentImpulse;
            contact.accumulatedTangentImpulse = std::clamp(oldTangentImpulse + lambdaT, -jtMax, jtMax);
            lambdaT = contact.accumulatedTangentImpulse - oldTangentImpulse;
        } else {
            lambdaT = std::clamp(lambdaT, -jtMax, jtMax);
        }

        if (std::abs(lambdaT) < kEps) return;
        Eigen::Vector3f const impulseT = lambdaT * tangent;
        ApplyImpulse(contact.idA, -impulseT, contact.position);
        ApplyImpulse(contact.idB,  impulseT, contact.position);
    }

    void RigidBodySystem::PositionalCorrection() {
        for (auto const & c : Contacts) {
            auto & a = Bodies[c.idA];
            auto & b = Bodies[c.idB];
            float const invMassSum = a.invMass + b.invMass;
            if (invMassSum < kEps) continue;

            float const depth = std::max(0.0f, c.penetration - PositionCorrectionSlop);
            if (depth <= 0.0f) continue;

            Eigen::Vector3f const correction = (PositionCorrectionPercent * depth / invMassSum) * SafeNormalized(c.normal);
            if (! a.isStatic) a.x -= a.invMass * correction;
            if (! b.isStatic) b.x += b.invMass * correction;
        }
    }

    void RigidBodySystem::StabilizeRestingContacts() {
        // Lab1 used resting stabilization to kill tiny jitter.  In this Lab4 tank,
        // the bodies are much smaller; directly using Lab1 thresholds can freeze a
        // box while it is still tilted.  Therefore we only allow sleeping when the
        // body has enough static-wall/static-obstacle contacts and its residual
        // velocity is very small.  A box on one corner/edge should keep rotating; a
        // box lying on a face can sleep.
        std::vector<int> staticContactCount(Bodies.size(), 0);
        for (auto const & c : Contacts) {
            if (c.penetration <= 0.0f) continue;
            if (! IsValidBody(c.idA) || ! IsValidBody(c.idB)) continue;

            RigidBody const & a = Bodies[c.idA];
            RigidBody const & b = Bodies[c.idB];

            if (! a.isStatic && b.isStatic) ++staticContactCount[c.idA];
            if (! b.isStatic && a.isStatic) ++staticContactCount[c.idB];
        }

        for (int i = 0; i < static_cast<int>(Bodies.size()); ++i) {
            auto & b = Bodies[i];
            if (b.isStatic || ! b.useGravity) continue;

            int const requiredStaticContacts = (b.shape == RigidBodyShape::Box) ? 3 : 1;
            if (staticContactCount[i] < requiredStaticContacts) continue;

            if (b.v.norm() < RestingLinearThreshold && b.w.norm() < RestingAngularThreshold) {
                b.v.setZero();
                b.w.setZero();
            }
        }
    }

    void RigidBodySystem::ResolveTankBounds(float minBound, float maxBound) {
        bool hasLab1TankWalls = false;
        for (auto const & body : Bodies) {
            if (body.isStatic && body.name.rfind("tank_", 0) == 0) {
                hasLab1TankWalls = true;
                break;
            }
        }

        // When the Lab1-style tank walls exist, wall contacts are already solved by
        // FCL + sequential impulses inside Step().  Running the old AABB correction
        // afterwards would fight the contact solver and cause visible jitter, so this
        // function becomes a very conservative safety net only.
        if (hasLab1TankWalls) {
            float const hardLimit = std::max(std::abs(minBound), std::abs(maxBound)) + 0.20f;
            for (auto & body : Bodies) {
                if (body.isStatic) continue;
                bool escaped = false;
                for (int axis = 0; axis < 3; ++axis) {
                    if (body.x[axis] < -hardLimit) {
                        body.x[axis] = -hardLimit;
                        body.v[axis] = 0.0f;
                        escaped = true;
                    }
                    if (body.x[axis] > hardLimit) {
                        body.x[axis] = hardLimit;
                        body.v[axis] = 0.0f;
                        escaped = true;
                    }
                }
                if (escaped) {
                    body.w *= 0.5f;
                }
            }
            return;
        }

        // Fallback for scenes without explicit tank wall bodies.  This keeps the old
        // protective behavior, but it is not used by the default Lab4 rigid presets.
        for (auto & body : Bodies) {
            if (body.isStatic) continue;

            auto const [mn, mx] = body.GetWorldAABB();
            Eigen::Vector3f correction = Eigen::Vector3f::Zero();
            for (int axis = 0; axis < 3; ++axis) {
                if (mn[axis] < minBound) {
                    correction[axis] += minBound - mn[axis];
                    if (body.v[axis] < 0.0f) body.v[axis] = 0.0f;
                }
                if (mx[axis] > maxBound) {
                    correction[axis] -= mx[axis] - maxBound;
                    if (body.v[axis] > 0.0f) body.v[axis] = 0.0f;
                }
            }
            body.x += correction;
        }
    }


} // namespace VCX::MainScene
