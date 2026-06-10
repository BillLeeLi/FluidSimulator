#include "Simulation/RigidBodySystem.h"
#include "Simulation/KenneyBoatSpeedAMesh.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/math/bv/OBBRSS.h>
#include <fcl/narrowphase/collision.h>

namespace VCX::MainScene {

    namespace {
        constexpr float kEps = 1e-6f;

        Eigen::Vector3f SafeNormalized(Eigen::Vector3f const & v, Eigen::Vector3f const & fallback = Eigen::Vector3f::UnitY()) {
            float const n = v.norm();
            return n > kEps ? (v / n) : fallback;
        }

        using CollisionGeometryPtr = std::shared_ptr<fcl::CollisionGeometry<float>>;
        using BoatBVHModel = fcl::BVHModel<fcl::OBBRSS<float>>;

        CollisionGeometryPtr MakePrimitiveCollisionGeometry(RigidBodyShape shape, Eigen::Vector3f const & dim) {
            switch (shape) {
            case RigidBodyShape::Sphere:
                return CollisionGeometryPtr(new fcl::Sphere<float>(0.5f * dim.x()));
            case RigidBodyShape::Box:
            default:
                return CollisionGeometryPtr(new fcl::Box<float>(dim.x(), dim.y(), dim.z()));
            }
        }

        CollisionGeometryPtr MakeBoatMeshCollisionGeometry(RigidBody const & body) {
            if (body.meshVertices.empty() || body.meshTriIndices.size() < 3) {
                return MakePrimitiveCollisionGeometry(RigidBodyShape::Box, body.dim);
            }

            std::vector<fcl::Vector3<float>> vertices;
            vertices.reserve(body.meshVertices.size());
            for (auto const & v : body.meshVertices) {
                vertices.emplace_back(v.x(), v.y(), v.z());
            }

            std::vector<fcl::Triangle> triangles;
            triangles.reserve(body.meshTriIndices.size() / 3);
            for (std::size_t i = 0; i + 2 < body.meshTriIndices.size(); i += 3) {
                std::uint32_t const ia = body.meshTriIndices[i + 0];
                std::uint32_t const ib = body.meshTriIndices[i + 1];
                std::uint32_t const ic = body.meshTriIndices[i + 2];
                if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size()) continue;
                triangles.emplace_back(static_cast<int>(ia), static_cast<int>(ib), static_cast<int>(ic));
            }

            if (triangles.empty()) {
                return MakePrimitiveCollisionGeometry(RigidBodyShape::Box, body.dim);
            }

            auto mesh = std::make_shared<BoatBVHModel>();
            mesh->beginModel(static_cast<int>(triangles.size()), static_cast<int>(vertices.size()));
            mesh->addSubModel(vertices, triangles);
            mesh->endModel();
            return mesh;
        }

        CollisionGeometryPtr MakeCollisionGeometry(RigidBody const & body) {
            if (body.shape == RigidBodyShape::BoatHull) {
                return MakeBoatMeshCollisionGeometry(body);
            }
            return MakePrimitiveCollisionGeometry(body.shape, body.dim);
        }

        Eigen::Vector3f MeshLocalCenter(RigidBody const & body) {
            if (body.meshVertices.empty()) return Eigen::Vector3f::Zero();
            Eigen::Vector3f mn = body.meshVertices.front();
            Eigen::Vector3f mx = body.meshVertices.front();
            for (auto const & v : body.meshVertices) {
                mn = mn.cwiseMin(v);
                mx = mx.cwiseMax(v);
            }
            return 0.5f * (mn + mx);
        }

        Eigen::Vector3f OrientedTriangleNormalLocal(
            RigidBody const & body,
            Eigen::Vector3f const & a,
            Eigen::Vector3f const & b,
            Eigen::Vector3f const & c) {
            Eigen::Vector3f n = (b - a).cross(c - a);
            if (n.squaredNorm() <= kEps * kEps) return Eigen::Vector3f::UnitY();

            Eigen::Vector3f const triCenter  = (a + b + c) / 3.0f;
            Eigen::Vector3f const meshCenter = MeshLocalCenter(body);
            if (n.dot(triCenter - meshCenter) < 0.0f) n = -n;
            return SafeNormalized(n, Eigen::Vector3f::UnitY());
        }

        bool RayTriangleHitLocalX(
            Eigen::Vector3f const & p,
            Eigen::Vector3f const & a,
            Eigen::Vector3f const & b,
            Eigen::Vector3f const & c,
            float & t) {
            Eigen::Vector3f const dir = Eigen::Vector3f::UnitX();
            Eigen::Vector3f const e1 = b - a;
            Eigen::Vector3f const e2 = c - a;
            Eigen::Vector3f const h  = dir.cross(e2);
            float const det = e1.dot(h);
            if (std::abs(det) < 1e-7f) return false;

            float const invDet = 1.0f / det;
            Eigen::Vector3f const s = p - a;
            float const u = invDet * s.dot(h);
            if (u < -1e-5f || u > 1.0f + 1e-5f) return false;

            Eigen::Vector3f const q = s.cross(e1);
            float const v = invDet * dir.dot(q);
            if (v < -1e-5f || u + v > 1.0f + 1e-5f) return false;

            t = invDet * e2.dot(q);
            return t > 1e-5f;
        }

        bool MeshContainsLocalPoint(RigidBody const & body, Eigen::Vector3f const & p) {
            if (body.meshVertices.empty() || body.meshTriIndices.size() < 3) return false;

            std::vector<float> hits;
            hits.reserve(32);
            for (std::size_t i = 0; i + 2 < body.meshTriIndices.size(); i += 3) {
                std::uint32_t const ia = body.meshTriIndices[i + 0];
                std::uint32_t const ib = body.meshTriIndices[i + 1];
                std::uint32_t const ic = body.meshTriIndices[i + 2];
                if (ia >= body.meshVertices.size() || ib >= body.meshVertices.size() || ic >= body.meshVertices.size()) continue;

                float t = 0.0f;
                if (! RayTriangleHitLocalX(p, body.meshVertices[ia], body.meshVertices[ib], body.meshVertices[ic], t)) continue;

                bool duplicate = false;
                for (float oldT : hits) {
                    if (std::abs(oldT - t) < 1e-4f) {
                        duplicate = true;
                        break;
                    }
                }
                if (! duplicate) hits.push_back(t);
            }
            return (hits.size() % 2) == 1;
        }

        Eigen::Vector3f ClosestPointOnTriangle(
            Eigen::Vector3f const & p,
            Eigen::Vector3f const & a,
            Eigen::Vector3f const & b,
            Eigen::Vector3f const & c) {
            Eigen::Vector3f const ab = b - a;
            Eigen::Vector3f const ac = c - a;
            Eigen::Vector3f const ap = p - a;
            float const d1 = ab.dot(ap);
            float const d2 = ac.dot(ap);
            if (d1 <= 0.0f && d2 <= 0.0f) return a;

            Eigen::Vector3f const bp = p - b;
            float const d3 = ab.dot(bp);
            float const d4 = ac.dot(bp);
            if (d3 >= 0.0f && d4 <= d3) return b;

            float const vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                float const v = d1 / (d1 - d3);
                return a + v * ab;
            }

            Eigen::Vector3f const cp = p - c;
            float const d5 = ab.dot(cp);
            float const d6 = ac.dot(cp);
            if (d6 >= 0.0f && d5 <= d6) return c;

            float const vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                float const w = d2 / (d2 - d6);
                return a + w * ac;
            }

            float const va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                float const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return b + w * (c - b);
            }

            float const denom = 1.0f / (va + vb + vc);
            float const v = vb * denom;
            float const w = vc * denom;
            return a + ab * v + ac * w;
        }

        bool ClosestBoatMeshPointLocal(
            RigidBody const & body,
            Eigen::Vector3f const & localPoint,
            Eigen::Vector3f & closestPoint,
            Eigen::Vector3f & normal) {
            if (body.meshVertices.empty() || body.meshTriIndices.size() < 3) return false;

            bool found = false;
            float bestDist2 = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i + 2 < body.meshTriIndices.size(); i += 3) {
                std::uint32_t const ia = body.meshTriIndices[i + 0];
                std::uint32_t const ib = body.meshTriIndices[i + 1];
                std::uint32_t const ic = body.meshTriIndices[i + 2];
                if (ia >= body.meshVertices.size() || ib >= body.meshVertices.size() || ic >= body.meshVertices.size()) continue;

                Eigen::Vector3f const & a = body.meshVertices[ia];
                Eigen::Vector3f const & b = body.meshVertices[ib];
                Eigen::Vector3f const & c = body.meshVertices[ic];
                Eigen::Vector3f const q = ClosestPointOnTriangle(localPoint, a, b, c);
                float const dist2 = (q - localPoint).squaredNorm();
                if (! found || dist2 < bestDist2) {
                    found = true;
                    bestDist2 = dist2;
                    closestPoint = q;
                    normal = OrientedTriangleNormalLocal(body, a, b, c);
                }
            }
            return found;
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
        auto const [mn, mx] = GetWorldAABB();
        return 0.5f * (mx - mn).norm();
    }

    void RigidBody::UpdateMassProperties() {
        // 改质量、尺寸或形状后都要重新算一次。
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
        case RigidBodyShape::BoatHull:
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
        if (shape == RigidBodyShape::BoatHull && ! meshVertices.empty()) {
            Eigen::Vector3f mn = LocalToWorld(meshVertices[0]);
            Eigen::Vector3f mx = mn;
            for (auto const & v : meshVertices) {
                Eigen::Vector3f const p = LocalToWorld(v);
                mn = mn.cwiseMin(p);
                mx = mx.cwiseMax(p);
            }
            return { mn, mx };
        }


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
        if (shape == RigidBodyShape::BoatHull && ! meshVertices.empty() && meshTriIndices.size() >= 3) {
            Eigen::Vector3f const localPoint = WorldToLocal(worldPoint);
            if (MeshContainsLocalPoint(*this, localPoint)) return true;

            // Kenney 船体是薄壳。流体网格只看 cell center，完全零厚度的话很容易漏掉表面。
            // 这里给 BoatHull 一个显式的有效厚度；FCL 碰撞不受这个值影响。
            Eigen::Vector3f closestLocal = Eigen::Vector3f::Zero();
            Eigen::Vector3f normalLocal  = Eigen::Vector3f::UnitY();
            if (ClosestBoatMeshPointLocal(*this, localPoint, closestLocal, normalLocal)) {
                float const shellThickness = std::max(0.0f, solidShellThickness);
                return shellThickness > 0.0f && (closestLocal - localPoint).squaredNorm() <= shellThickness * shellThickness;
            }
            return false;
        }


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
        // 鼠标选点和粒子推出刚体时都会用到这个近似表面点。
        if (shape == RigidBodyShape::BoatHull && ! meshVertices.empty() && meshTriIndices.size() >= 3) {
            Eigen::Vector3f closestLocal = Eigen::Vector3f::Zero();
            Eigen::Vector3f normalLocal  = Eigen::Vector3f::UnitY();
            if (ClosestBoatMeshPointLocal(*this, WorldToLocal(worldPoint), closestLocal, normalLocal)) {
                return LocalToWorld(closestLocal);
            }
        }


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
        if (shape == RigidBodyShape::BoatHull && ! meshVertices.empty() && meshTriIndices.size() >= 3) {
            Eigen::Vector3f closestLocal = Eigen::Vector3f::Zero();
            Eigen::Vector3f normalLocal  = Eigen::Vector3f::UnitY();
            if (ClosestBoatMeshPointLocal(*this, WorldToLocal(worldPoint), closestLocal, normalLocal)) {
                return GetRotationMatrix() * normalLocal;
            }
        }


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
        InvalidateCollisionGeometryCache();
    }

    void RigidBodySystem::Reset() {
        Clear();
    }

    void RigidBodySystem::InvalidateCollisionGeometryCache() {
        _collisionGeometryCache.clear();
        _collisionGeometryCacheDirty = true;
    }

    void RigidBodySystem::EnsureCollisionGeometryCache() {
        if (! _collisionGeometryCacheDirty && _collisionGeometryCache.size() == Bodies.size()) return;

        _collisionGeometryCache.assign(Bodies.size(), nullptr);
        for (int i = 0; i < static_cast<int>(Bodies.size()); ++i) {
            _collisionGeometryCache[i] = MakeCollisionGeometry(Bodies[i]);
        }
        _collisionGeometryCacheDirty = false;
    }

    RigidBodySystem::CollisionGeometryPtr const & RigidBodySystem::CollisionGeometryAt(int id) {
        EnsureCollisionGeometryCache();
        return _collisionGeometryCache[id];
    }

    int RigidBodySystem::AddBody(RigidBody body) {
        body.q.normalize();
        body.UpdateMassProperties();
        Bodies.push_back(std::move(body));
        _collisionGeometryCacheDirty = true;
        return static_cast<int>(Bodies.size()) - 1;
    }

    void RigidBodySystem::SetupDefaultScene(RigidBodyPreset preset) {
        Clear();

        // 这里基本沿用 Lab1 里比较稳的那组参数。Lab4 物体更小，所以休眠阈值另外调低了。
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
            float constexpr inner     = 0.43f;
            float constexpr baseThick = 0.08f;
            float constexpr thick     = 20.0f * baseThick;
            float constexpr span      = 2.0f * (inner + thick);
            float constexpr c         = inner + 0.5f * thick;

            // 碰撞面仍然在原来的水槽内壁，墙体向外加厚，防止高速物体一帧穿出去。
            // tank_ 开头的墙只参与碰撞，不参与选择和绘制。
            addTankWall("tank_neg_x", Eigen::Vector3f(thick, span,  span), Eigen::Vector3f(-c, 0.0f, 0.0f));
            addTankWall("tank_pos_x", Eigen::Vector3f(thick, span,  span), Eigen::Vector3f( c, 0.0f, 0.0f));
            addTankWall("tank_neg_y", Eigen::Vector3f(span,  thick, span), Eigen::Vector3f(0.0f, -c, 0.0f));
            addTankWall("tank_pos_y", Eigen::Vector3f(span,  thick, span), Eigen::Vector3f(0.0f,  c, 0.0f));
            addTankWall("tank_neg_z", Eigen::Vector3f(span,  span,  thick), Eigen::Vector3f(0.0f, 0.0f, -c));
            addTankWall("tank_pos_z", Eigen::Vector3f(span,  span,  thick), Eigen::Vector3f(0.0f, 0.0f,  c));
        };


        // 下面几个预设都按 Lab4 水槽尺寸缩小过，后面写入流体网格也方便。
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
        case RigidBodyPreset::BoatInWater: {
            RigidBody boat;
            boat.name        = "kenney_speed_boat";
            boat.shape       = RigidBodyShape::BoatHull;
            boat.dim         = Eigen::Vector3f(0.47f, 0.21f, 0.25f);
            boat.x           = Eigen::Vector3f(-0.18f, -0.095f, -0.15f);
            boat.q           = Eigen::Quaternionf::Identity();
            boat.v           = Eigen::Vector3f(0.015f, 0.0f, 0.006f);
            boat.w           = Eigen::Vector3f::Zero();
            boat.mass        = 1.10f;
            boat.restitution = 0.02f;
            boat.friction    = 0.82f;
            boat.useGravity  = true;
            boat.color       = Eigen::Vector3f(0.18f, 0.42f, 0.86f);

            // Kenney boat-speed-a 的 OBJ 顶点直接进入刚体数据。
            // 渲染、FCL 碰撞、鼠标拾取、流体 solid mask、粒子投影和压力采样都使用这套三角网格。
            boat.meshVertices  = KenneyBoatSpeedA::MakePhysicsVertices();
            boat.meshTriIndices = KenneyBoatSpeedA::MakeTriangleIndices();
            // 只给流体网格一个薄壳厚度，避免 cell center 正好落不到船体表面。
            boat.solidShellThickness = 0.018f;

            // 浮力点放在 Kenney 船体的底部/两侧下方，只有真实水粒子接近时才会被使用。
            for (int ix = 0; ix < 6; ++ix) {
                float const fx = (float(ix) / 5.0f) * 2.0f - 1.0f;
                float const halfWidth = 0.030f + 0.065f * (1.0f - std::abs(fx));
                for (int iz = 0; iz < 3; ++iz) {
                    float const fz = float(iz - 1);
                    RigidBuoyancySample sample;
                    sample.localPosition = Eigen::Vector3f(0.185f * fx, -0.072f, halfWidth * fz);
                    sample.volumeWeight  = 1.0f / 18.0f;
                    sample.radius        = 0.030f;
                    boat.buoyancySamples.push_back(sample);
                }
            }

            AddBody(boat);
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
            obstacle.name        = "dynamic_coupling_box";
            obstacle.shape       = RigidBodyShape::Box;
            obstacle.dim         = Eigen::Vector3f(0.12f, 0.12f, 0.20f);
            obstacle.x           = Eigen::Vector3f(0.20f, -0.18f, -0.08f);
            obstacle.q           = Eigen::Quaternionf(Eigen::AngleAxisf(-0.35f, Eigen::Vector3f::UnitY()));
            obstacle.mass        = 1.0f;
            obstacle.isStatic    = false;
            obstacle.useGravity  = true;
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

    bool RigidBodySystem::IsInternalTankBoundary(int id) const {
        if (! IsValidBody(id)) return false;
        auto const & body = Bodies[id];
        return body.isStatic && body.name.rfind("tank_", 0) == 0;
    }

    void RigidBodySystem::ClearForces() {
        for (auto & body : Bodies) {
            body.force.setZero();
            body.torque.setZero();
        }
    }

    void RigidBodySystem::ApplyForce(int id, Eigen::Vector3f const & forceWorld, Eigen::Vector3f const & worldPoint) {
        // 力不一定打在质心上，所以这里顺手把力矩也加上。
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
        // 碰撞瞬间用冲量改速度；偏心冲量会直接改角速度。
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

    RigidGeneralizedVelocity RigidBodySystem::GetGeneralizedVelocity(int id) const {
        // 变分耦合把刚体看成一个 6 自由度速度变量：[v_x v_y v_z w_x w_y w_z]^T。
        RigidGeneralizedVelocity velocity = RigidGeneralizedVelocity::Zero();
        if (! IsValidBody(id)) return velocity;

        auto const & body = Bodies[id];
        velocity.segment<3>(0) = body.v;
        velocity.segment<3>(3) = body.w;
        return velocity;
    }

    void RigidBodySystem::SetGeneralizedVelocity(int id, RigidGeneralizedVelocity const & velocity) {
        // 求解器得到新的广义速度后写回刚体；静态物体不可被压力冲量推动。
        if (! IsValidBody(id)) return;

        auto & body = Bodies[id];
        if (body.isStatic) return;
        body.v = velocity.segment<3>(0);
        body.w = velocity.segment<3>(3);
    }

    RigidInverseMassBlock RigidBodySystem::GetInverseMassBlock(int id) const {
        // 上半块是平动逆质量，下半块是世界坐标下的转动惯量逆。
        // 没有平动-转动耦合项，所以当前是块对角矩阵。
        RigidInverseMassBlock invMass = RigidInverseMassBlock::Zero();
        if (! IsValidBody(id)) return invMass;

        auto const & body = Bodies[id];
        if (body.isStatic) return invMass;

        invMass.block<3, 3>(0, 0) = body.invMass * Eigen::Matrix3f::Identity();
        invMass.block<3, 3>(3, 3) = body.GetWorldInertiaInv();
        return invMass;
    }


    void RigidBodySystem::CollectSurfaceSamples(int bodyId, int samplesPerAxis, std::vector<RigidSurfaceSample> & samples) const {
        // 这里只采样几何表面，不碰流体；压力、浮力怎么用由 coupler 那边决定。
        if (! IsValidBody(bodyId)) return;
        auto const & body = Bodies[bodyId];

        int const n = std::max(1, samplesPerAxis);
        Eigen::Matrix3f const R = body.GetRotationMatrix();

        if (body.shape == RigidBodyShape::BoatHull && ! body.meshVertices.empty() && body.meshTriIndices.size() >= 3) {
            for (std::size_t i = 0; i + 2 < body.meshTriIndices.size(); i += 3) {
                std::uint32_t const ia = body.meshTriIndices[i + 0];
                std::uint32_t const ib = body.meshTriIndices[i + 1];
                std::uint32_t const ic = body.meshTriIndices[i + 2];
                if (ia >= body.meshVertices.size() || ib >= body.meshVertices.size() || ic >= body.meshVertices.size()) continue;

                Eigen::Vector3f const & la = body.meshVertices[ia];
                Eigen::Vector3f const & lb = body.meshVertices[ib];
                Eigen::Vector3f const & lc = body.meshVertices[ic];
                Eigen::Vector3f const localCross = (lb - la).cross(lc - la);
                float const area = 0.5f * localCross.norm();
                if (area <= kEps) continue;

                Eigen::Vector3f const a = body.LocalToWorld(la);
                Eigen::Vector3f const b = body.LocalToWorld(lb);
                Eigen::Vector3f const c = body.LocalToWorld(lc);

                RigidSurfaceSample sample;
                sample.bodyId = bodyId;
                sample.position = (a + b + c) / 3.0f;
                sample.normal = body.GetRotationMatrix() * OrientedTriangleNormalLocal(body, la, lb, lc);
                sample.area = area;
                samples.push_back(sample);
            }
            return;
        }

        auto emitBoxFaces = [&](Eigen::Vector3f const & dim, Eigen::Matrix3f const & worldR, auto const & localToWorld) {
            Eigen::Vector3f const half = 0.5f * dim;
            auto emitFace = [&](int axis, float sign) {
                int const uAxis = (axis + 1) % 3;
                int const vAxis = (axis + 2) % 3;
                float const faceArea = dim[uAxis] * dim[vAxis];
                float const sampleArea = faceArea / float(n * n);

                Eigen::Vector3f localNormal = Eigen::Vector3f::Zero();
                localNormal[axis] = sign;
                Eigen::Vector3f const worldNormal = SafeNormalized(worldR * localNormal);

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
                        sample.position = localToWorld(localPoint);
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
        };


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

        emitBoxFaces(body.dim, R, [&](Eigen::Vector3f const & p) { return body.LocalToWorld(p); });
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
        _collisionGeometryCacheDirty = true;
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
        _collisionGeometryCacheDirty = true;
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

            // 和 Lab1 一样用显式欧拉：先按当前速度走一步，再用这一帧的力更新速度。
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

        auto const & geomA = CollisionGeometryAt(idA);
        auto const & geomB = CollisionGeometryAt(idB);
        if (! geomA || ! geomB) return;

        fcl::CollisionObject<float> shapeA(geomA, fcl::Transform3f(Eigen::Translation3f(a.x) * a.q));
        fcl::CollisionObject<float> shapeB(geomB, fcl::Transform3f(Eigen::Translation3f(b.x) * b.q));

        fcl::CollisionRequest<float> request(24, true);
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
        // 一个接触点的速度约束：先法向冲量，再摩擦冲量。
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
        // 速度解算后再推开一点，防止两个刚体一直嵌在一起。
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
        // 休眠只处理很小的残余抖动。Lab4 尺寸比 Lab1 小，阈值不能设太大。
        // 盒子如果只是一角/一边碰到地面，就先不让它睡眠，避免斜着被冻住。
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

            int const requiredStaticContacts = (b.shape == RigidBodyShape::Sphere) ? 1 : 3;
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

        // 默认预设里已经有 tank_ 静态墙体了，普通撞墙交给 FCL 接触求解。
        // 这里保留成保险：真的飞出水槽很远时才拉回来。
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

        // 兼容没有 tank_ 墙体的旧场景，默认几个预设一般走不到这里。
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
