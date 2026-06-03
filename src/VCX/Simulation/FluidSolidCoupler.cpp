#include "Simulation/FluidSolidCoupler.h"
#include "Simulation/FluidSimulator.h"
#include "Simulation/RigidBodySystem.h"

namespace VCX::MainScene {

    namespace {
        constexpr float kEps = 1e-6f;

        bool IsInternalTankBoundary(RigidBody const & body) {
            // 认为名字以"tank_"开头的静态物体（也就是水箱墙体，之后可以考虑改一下前缀名称）是内部不可穿透的边界，而不是普通的固体物体
            return body.name.rfind("tank_", 0) == 0;
        }

        Eigen::Vector3f SafeNormalized(Eigen::Vector3f const & v, Eigen::Vector3f const & fallback = Eigen::Vector3f::UnitY()) {
            // 提供安全的归一化，避免零向量导致的数值问题
            float const n = v.norm();
            return n > kEps ? (v / n) : fallback;
        }
    } // namespace

    void FluidSolidCoupler::ResetDebug() {
        _projectedParticleCount = 0;
    }

    int FluidSolidCoupler::ProjectParticlesOutOfRigidBodies(FluidSimulator & fluid, RigidBodySystem const & rigid) {
        int projectedThisCall = 0;

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
                    velocity -= inwardSpeed * normal;
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

} // namespace VCX::MainScene
