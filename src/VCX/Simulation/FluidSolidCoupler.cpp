#include "Simulation/FluidSolidCoupler.h"
#include "Simulation/FluidSimulator.h"
#include "Simulation/RigidBodySystem.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace VCX::MainScene {

    namespace {
        constexpr float kEps = 1e-6f;
        std::array<glm::ivec3, 6> const kNeighborOffsets = {
            glm::ivec3( 1,  0,  0),
            glm::ivec3(-1,  0,  0),
            glm::ivec3( 0,  1,  0),
            glm::ivec3( 0, -1,  0),
            glm::ivec3( 0,  0,  1),
            glm::ivec3( 0,  0, -1),
        };

        bool IsInternalTankBoundary(RigidBody const & body) {
            // 认为名字以"tank_"开头的静态物体（也就是水箱墙体，之后可以考虑改一下前缀名称）是内部不可穿透的边界，而不是普通的固体物体
            return body.name.rfind("tank_", 0) == 0;
        }

        Eigen::Vector3f SafeNormalized(Eigen::Vector3f const & v, Eigen::Vector3f const & fallback = Eigen::Vector3f::UnitY()) {
            // 提供安全的归一化，避免零向量导致的数值问题
            float const n = v.norm();
            return n > kEps ? (v / n) : fallback;
        }

        int AxisFromOffset(glm::ivec3 const & offset) {
            if (offset.x != 0) return 0;
            if (offset.y != 0) return 1;
            return 2;
        }

        glm::ivec3 FaceIndexForContact(glm::ivec3 const & solidCell, glm::ivec3 const & solidToFluid) {
            glm::ivec3 face = solidCell;
            int const  dir  = AxisFromOffset(solidToFluid);
            if (solidToFluid[dir] > 0) {
                face[dir] += 1;
            }
            return face;
        }

        Eigen::Vector3f FaceCenter(FluidSimulator const & fluid, glm::ivec3 const & face, int dir) {
            glm::vec3 pos(float(face.x), float(face.y), float(face.z));
            if (dir != 0) pos.x += 0.5f;
            if (dir != 1) pos.y += 0.5f;
            if (dir != 2) pos.z += 0.5f;
            return ToEigen(pos * fluid.m_h + glm::vec3(-0.5f));
        }

        glm::ivec3 ClampCell(FluidSimulator const & fluid, glm::ivec3 cell) {
            return glm::ivec3(
                std::clamp(cell.x, 0, fluid.m_iCellX - 1),
                std::clamp(cell.y, 0, fluid.m_iCellY - 1),
                std::clamp(cell.z, 0, fluid.m_iCellZ - 1));
        }

        std::pair<glm::ivec3, glm::ivec3> BodyCellRange(FluidSimulator const & fluid, RigidBody const & body) {
            auto const [mn, mx] = body.GetWorldAABB();
            Eigen::Vector3f const pad = Eigen::Vector3f::Constant(fluid.m_h);
            glm::ivec3 lo = ClampCell(fluid, fluid.worldToCell(ToGlm(mn - pad)));
            glm::ivec3 hi = ClampCell(fluid, fluid.worldToCell(ToGlm(mx + pad)));
            return {
                glm::ivec3(std::min(lo.x, hi.x), std::min(lo.y, hi.y), std::min(lo.z, hi.z)),
                glm::ivec3(std::max(lo.x, hi.x), std::max(lo.y, hi.y), std::max(lo.z, hi.z)),
            };
        }
    } // namespace

    void FluidSolidCoupler::ResetDebug() {
        _projectedParticleCount = 0;
        _rigidSolidCellCount = 0;
        _pressureContactFaceCount = 0;
        _movingBoundaryFaceCount = 0;
        std::fill(_pressureForcesByBody.begin(), _pressureForcesByBody.end(), Eigen::Vector3f::Zero());
        std::fill(_particleImpulsesByBody.begin(), _particleImpulsesByBody.end(), Eigen::Vector3f::Zero());
    }

    Eigen::Vector3f FluidSolidCoupler::PressureForceOnBody(int bodyId) const {
        if (bodyId < 0 || bodyId >= static_cast<int>(_pressureForcesByBody.size())) {
            return Eigen::Vector3f::Zero();
        }
        return _pressureForcesByBody[bodyId];
    }

    Eigen::Vector3f FluidSolidCoupler::ParticleImpulseOnBody(int bodyId) const {
        if (bodyId < 0 || bodyId >= static_cast<int>(_particleImpulsesByBody.size())) {
            return Eigen::Vector3f::Zero();
        }
        return _particleImpulsesByBody[bodyId];
    }

    int FluidSolidCoupler::ProjectParticlesOutOfRigidBodies(FluidSimulator & fluid, RigidBodySystem & rigid) {
        if (_particleImpulsesByBody.size() != rigid.Bodies.size()) {
            _particleImpulsesByBody.assign(rigid.Bodies.size(), Eigen::Vector3f::Zero());
        }

        int projectedThisCall = 0;
        float const particleMass = 1000.0f * (4.0f / 3.0f) * 3.14159265358979323846f
            * fluid.m_particleRadius * fluid.m_particleRadius * fluid.m_particleRadius;

        for (int particleId = 0; particleId < fluid.m_iNumSpheres; ++particleId) {
            Eigen::Vector3f position = ToEigen(fluid.m_particlePos[particleId]);
            Eigen::Vector3f velocity = ToEigen(fluid.m_particleVel[particleId]);
            bool            projectedParticle = false;

            for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
                RigidBody const & body = rigid.Bodies[bodyId];
                if (IsInternalTankBoundary(body)) continue;
                if (! body.ContainsPoint(position)) continue;

                Eigen::Vector3f const surfacePoint = body.ClosestSurfacePoint(position);
                Eigen::Vector3f const normal       = SafeNormalized(body.SurfaceNormalAt(surfacePoint));

                position = surfacePoint + normal * fluid.m_particleRadius;

                Eigen::Vector3f const bodyVelocity = rigid.VelocityAtPoint(bodyId, surfacePoint);
                Eigen::Vector3f const relativeVel  = velocity - bodyVelocity;
                float const           inwardSpeed  = relativeVel.dot(normal);
                if (inwardSpeed < 0.0f) {
                    Eigen::Vector3f const oldVelocity = velocity;
                    velocity -= inwardSpeed * normal;

                    if (enableParticleCollisionImpulse && particleImpulseScale > 0.0f && ! body.isStatic) {
                        Eigen::Vector3f const deltaV  = velocity - oldVelocity;
                        Eigen::Vector3f const impulse = -particleImpulseScale * particleMass * deltaV;
                        rigid.ApplyImpulse(bodyId, impulse, surfacePoint);
                        _particleImpulsesByBody[bodyId] += impulse;
                    }
                }

                projectedParticle = true;
            }

            if (projectedParticle) {
                fluid.m_particlePos[particleId] = ToGlm(position);
                fluid.m_particleVel[particleId] = ToGlm(velocity);
                ++projectedThisCall;
            }
        }

        _projectedParticleCount += projectedThisCall;
        return projectedThisCall;
    }

    int FluidSolidCoupler::RasterizeRigidBodiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid) {
        fluid.ResetSolidMaskToTank();
        if (! enableRigidSolidMask) {
            _rigidSolidCellCount = 0;
            return 0;
        }

        int solidCells = 0;
        for (RigidBody const & body : rigid.Bodies) {
            if (IsInternalTankBoundary(body)) continue;

            auto const [lo, hi] = BodyCellRange(fluid, body);
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const cell(i, j, k);
                        if (fluid.IsCellSolid(cell)) continue;

                        Eigen::Vector3f const center = ToEigen(fluid.CellCenter(cell));
                        if (! body.ContainsPoint(center)) continue;

                        fluid.SetCellSolid(cell, true);
                        ++solidCells;
                    }
                }
            }
        }

        _rigidSolidCellCount = solidCells;
        return solidCells;
    }

    int FluidSolidCoupler::ApplyRigidBoundaryVelocitiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid) {
        fluid.ResetSolidBoundaryVelocity();
        if (! enableMovingSolidVelocity || ! enableRigidSolidMask || fluid.m_type.empty()) {
            return 0;
        }

        int boundaryFaces = 0;

        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body)) continue;

            auto const [lo, hi] = BodyCellRange(fluid, body);
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const solidCell(i, j, k);
                        Eigen::Vector3f const solidCenter = ToEigen(fluid.CellCenter(solidCell));
                        if (! body.ContainsPoint(solidCenter)) continue;
                        if (! fluid.IsCellSolid(solidCell)) continue;

                        for (glm::ivec3 const & offset : kNeighborOffsets) {
                            glm::ivec3 const fluidCell = solidCell + offset;
                            if (! fluid.IsInsideGrid(fluidCell)) continue;

                            int const fluidId = fluid.GridIndex(fluidCell);
                            if (fluid.m_type[fluidId] != FLUID_CELL) continue;

                            int const         dir        = AxisFromOffset(offset);
                            glm::ivec3 const  face       = FaceIndexForContact(solidCell, offset);
                            Eigen::Vector3f const center = FaceCenter(fluid, face, dir);
                            Eigen::Vector3f const vel    = rigid.VelocityAtPoint(bodyId, center);

                            fluid.SetSolidBoundaryVelocity(face, dir, vel[dir]);
                            ++boundaryFaces;
                        }
                    }
                }
            }
        }

        _movingBoundaryFaceCount += boundaryFaces;
        return boundaryFaces;
    }

    int FluidSolidCoupler::ApplyPressureForcesFromFluid(FluidSimulator const & fluid, RigidBodySystem & rigid) {
        if (_pressureForcesByBody.size() != rigid.Bodies.size()) {
            _pressureForcesByBody.assign(rigid.Bodies.size(), Eigen::Vector3f::Zero());
        }
        if (! enablePressureForce || fluid.m_p.empty() || fluid.m_type.empty()) return 0;

        int contactFaces = 0;
        int const samplesPerAxis = std::max(2, int(std::ceil(0.18f / std::max(fluid.m_h, kEps))));

        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body)) continue;

            std::vector<RigidSurfaceSample> samples;
            rigid.CollectSurfaceSamples(bodyId, samplesPerAxis, samples);

            for (RigidSurfaceSample const & sample : samples) {
                Eigen::Vector3f const normal = SafeNormalized(sample.normal);
                glm::vec3 const      probe  = ToGlm(sample.position + normal * (0.5f * fluid.m_h));
                glm::ivec3 const     cell   = fluid.worldToCell(probe);
                if (! fluid.IsInsideGrid(cell)) continue;

                int const cellId = fluid.GridIndex(cell);
                if (fluid.m_type[cellId] != FLUID_CELL) continue;

                float const pressure = std::clamp(std::max(0.0f, fluid.SamplePressure(probe)), 0.0f, maxPressureForForce);
                if (pressure <= 0.0f || pressureForceScale <= 0.0f) continue;

                Eigen::Vector3f const force = -normal * (pressure * sample.area * pressureForceScale);

                ++contactFaces;
                _pressureForcesByBody[bodyId] += force;
                if (! body.isStatic) {
                    rigid.ApplyForce(bodyId, force, sample.position);
                }
            }
        }

        _pressureContactFaceCount += contactFaces;
        return contactFaces;
    }

} // namespace VCX::MainScene
