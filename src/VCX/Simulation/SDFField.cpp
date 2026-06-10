#include "Simulation/SDFField.h"
#include "Simulation/RigidBodySystem.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace VCX::MainScene {

    namespace {
        constexpr float kEps = 1e-6f;

        glm::ivec3 ClampIndex(SDFField const & field, glm::ivec3 idx) {
            return glm::ivec3(
                std::clamp(idx.x, 0, field.nx - 1),
                std::clamp(idx.y, 0, field.ny - 1),
                std::clamp(idx.z, 0, field.nz - 1));
        }

        // 把刚体在世界坐标里的AABB包围盒范围转换成SDF网格里的整数索引范围。
        std::pair<glm::ivec3, glm::ivec3> WorldAABBToGridRange(
            SDFField const &        field,
            Eigen::Vector3f const & minWorld,
            Eigen::Vector3f const & maxWorld,
            float                   pad) {
            // 固体 SDF 不需要每次扫描整个水槽；用刚体 AABB 加一圈 pad 只更新可能影响 fraction 的窄带。
            glm::vec3 const mn = ToGlm(minWorld) - glm::vec3(pad);
            glm::vec3 const mx = ToGlm(maxWorld) + glm::vec3(pad);

            glm::ivec3 lo(
                static_cast<int>(std::floor((mn.x - field.origin.x) / field.dx)),
                static_cast<int>(std::floor((mn.y - field.origin.y) / field.dx)),
                static_cast<int>(std::floor((mn.z - field.origin.z) / field.dx)));
            glm::ivec3 hi(
                static_cast<int>(std::ceil((mx.x - field.origin.x) / field.dx)),
                static_cast<int>(std::ceil((mx.y - field.origin.y) / field.dx)),
                static_cast<int>(std::ceil((mx.z - field.origin.z) / field.dx)));

            return { ClampIndex(field, lo), ClampIndex(field, hi) };
        }
    } // namespace

    void SDFField::Resize(int x, int y, int z, float spacing, glm::vec3 const & gridOrigin, float initialValue) {
        nx     = std::max(0, x);
        ny     = std::max(0, y);
        nz     = std::max(0, z);
        dx     = spacing;
        origin = gridOrigin;

        int const count = nx * ny * nz;
        phi.assign(std::max(0, count), initialValue);
    }

    void SDFField::Clear() {
        phi.clear();
        nx = ny = nz = 0;
        dx           = 0.0f;
        origin       = glm::vec3(-0.5f);
    }

    bool SDFField::IsValid() const {
        return nx > 0
            && ny > 0
            && nz > 0
            && dx > 0.0f
            && phi.size() == static_cast<std::size_t>(nx * ny * nz);
    }

    bool SDFField::IsInside(glm::ivec3 idx) const {
        return idx.x >= 0 && idx.x < nx
            && idx.y >= 0 && idx.y < ny
            && idx.z >= 0 && idx.z < nz;
    }

    int SDFField::Index(glm::ivec3 idx) const {
        return idx.x * (ny * nz) + idx.y * nz + idx.z;
    }

    glm::vec3 SDFField::Position(glm::ivec3 idx) const {
        return origin + glm::vec3(idx) * dx;
    }

    float SampleSDF(SDFField const & field, glm::vec3 const & p) {
        if (! field.IsValid()) return std::numeric_limits<float>::max();

        // p 先转到 SDF 网格坐标，再在周围 8 个格点间三线性插值。
        glm::vec3 const gRaw = (p - field.origin) / field.dx;
        glm::vec3 const g(
            std::clamp(gRaw.x, 0.0f, float(field.nx - 1)),
            std::clamp(gRaw.y, 0.0f, float(field.ny - 1)),
            std::clamp(gRaw.z, 0.0f, float(field.nz - 1)));

        int const ix0 = std::clamp(static_cast<int>(std::floor(g.x)), 0, field.nx - 1);
        int const iy0 = std::clamp(static_cast<int>(std::floor(g.y)), 0, field.ny - 1);
        int const iz0 = std::clamp(static_cast<int>(std::floor(g.z)), 0, field.nz - 1);
        int const ix1 = std::min(ix0 + 1, field.nx - 1);
        int const iy1 = std::min(iy0 + 1, field.ny - 1);
        int const iz1 = std::min(iz0 + 1, field.nz - 1);

        float const fx = g.x - float(ix0);
        float const fy = g.y - float(iy0);
        float const fz = g.z - float(iz0);

        auto value = [&](int i, int j, int k) {
            return field.phi[field.Index(glm::ivec3(i, j, k))];
        };

        float const c000 = value(ix0, iy0, iz0);
        float const c100 = value(ix1, iy0, iz0);
        float const c010 = value(ix0, iy1, iz0);
        float const c110 = value(ix1, iy1, iz0);
        float const c001 = value(ix0, iy0, iz1);
        float const c101 = value(ix1, iy0, iz1);
        float const c011 = value(ix0, iy1, iz1);
        float const c111 = value(ix1, iy1, iz1);

        float const c00 = glm::mix(c000, c100, fx);
        float const c10 = glm::mix(c010, c110, fx);
        float const c01 = glm::mix(c001, c101, fx);
        float const c11 = glm::mix(c011, c111, fx);
        float const c0  = glm::mix(c00, c10, fy);
        float const c1  = glm::mix(c01, c11, fy);
        return glm::mix(c0, c1, fz);
    }

    glm::vec3 GradSDF(SDFField const & field, glm::vec3 const & p) {
        if (! field.IsValid()) return glm::vec3(0.0f);

        float const     h = field.dx;
        glm::vec3 const dx(h, 0.0f, 0.0f);
        glm::vec3 const dy(0.0f, h, 0.0f);
        glm::vec3 const dz(0.0f, 0.0f, h);
        return glm::vec3(
            (SampleSDF(field, p + dx) - SampleSDF(field, p - dx)) / (2.0f * h),
            (SampleSDF(field, p + dy) - SampleSDF(field, p - dy)) / (2.0f * h),
            (SampleSDF(field, p + dz) - SampleSDF(field, p - dz)) / (2.0f * h));
    }

    void BuildFluidSDFFromParticles(
        std::vector<glm::vec3> const & particles,
        SDFField &                     phiFluid,
        float                          particleRadius,
        float                          activeBand) {
        if (! phiFluid.IsValid() || particleRadius <= 0.0f) return;

        // farPhi 是窄带外的保守正值；这些远离粒子的点只需要保持“液体外部”即可。
        activeBand         = activeBand > 0.0f
            ? activeBand
            : std::max(2.5f * phiFluid.dx, 0.75f * particleRadius);
        float const farPhi = std::max(4.0f * particleRadius, 4.0f * phiFluid.dx);
        int const   reach  = std::max(1, static_cast<int>(std::ceil((particleRadius + activeBand) / phiFluid.dx)));

        std::fill(phiFluid.phi.begin(), phiFluid.phi.end(), farPhi);

        for (glm::vec3 const & particlePos : particles) {
            glm::vec3 const  grid = (particlePos - phiFluid.origin) / phiFluid.dx;
            glm::ivec3 const center(
                static_cast<int>(std::floor(grid.x)),
                static_cast<int>(std::floor(grid.y)),
                static_cast<int>(std::floor(grid.z)));

            glm::ivec3 const lo = ClampIndex(phiFluid, center - glm::ivec3(reach));
            glm::ivec3 const hi = ClampIndex(phiFluid, center + glm::ivec3(reach));

            // 每个粒子是一个隐式小球，多个粒子的并集通过对 phi 取 min 得到。
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const idx(i, j, k);
                        float const      phi = glm::length(phiFluid.Position(idx) - particlePos) - particleRadius;
                        int const        id  = phiFluid.Index(idx);
                        if (phi < phiFluid.phi[id]) {
                            phiFluid.phi[id] = phi;
                        }
                    }
                }
            }
        }
    }

    void BuildSolidSDFFromRigidBodies(
        RigidBodySystem const & rigid,
        SDFField &              phiSolid,
        float                   maxDistance,
        bool                    includeTankBoundaries) {
        if (! phiSolid.IsValid()) return;

        // maxDistance 之外不需要准确距离；fraction 计算只关心固体边界附近的符号和过渡。
        maxDistance = maxDistance > 0.0f ? maxDistance : std::max(4.0f * phiSolid.dx, 0.08f);
        std::fill(phiSolid.phi.begin(), phiSolid.phi.end(), maxDistance);

        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            if (! includeTankBoundaries && rigid.IsInternalTankBoundary(bodyId)) continue;

            RigidBody const & body        = rigid.Bodies[bodyId];
            auto const [bodyMin, bodyMax] = body.GetWorldAABB();
            auto const [lo, hi]           = WorldAABBToGridRange(phiSolid, bodyMin, bodyMax, maxDistance);

            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const      idx(i, j, k);
                        Eigen::Vector3f const p        = ToEigen(phiSolid.Position(idx));
                        // main 分支已经给 RigidBody 提供统一 SDF 查询；BoatHull 会优先走局部 SDF 缓存。
                        float signedDistance = body.SignedDistance(p);
                        if (std::abs(signedDistance) < kEps && body.ContainsPoint(p)) {
                            signedDistance = -kEps;
                        }

                        int const id = phiSolid.Index(idx);
                        if (signedDistance < phiSolid.phi[id]) {
                            phiSolid.phi[id] = signedDistance;
                        }
                    }
                }
            }
        }
    }

} // namespace VCX::MainScene
