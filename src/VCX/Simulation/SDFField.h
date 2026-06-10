#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace VCX::MainScene {

    class RigidBodySystem;

    // 规则网格上的 signed distance field。
    // phi < 0 表示几何内部，phi > 0 表示外部；origin 是 (0,0,0) 格点的世界坐标。
    struct SDFField {
        std::vector<float> phi;
        glm::vec3          origin { -0.5f };
        float              dx { 0.0f };
        int                nx { 0 };
        int                ny { 0 };
        int                nz { 0 };

        void Resize(int x, int y, int z, float spacing, glm::vec3 const & gridOrigin, float initialValue);
        void Clear();

        bool      IsValid() const;
        bool      IsInside(glm::ivec3 idx) const;
        int       Index(glm::ivec3 idx) const;
        glm::vec3 Position(glm::ivec3 idx) const;
    };

    // 三线性采样和中心差分梯度，后续 cell/face fraction、边界法线都会共用。
    float     SampleSDF(SDFField const & field, glm::vec3 const & p);
    glm::vec3 GradSDF(SDFField const & field, glm::vec3 const & p);

    // 粒子并集 SDF：phi(x)=min_p(|x-xp|-r)。只更新粒子附近窄带，避免全网格乘全粒子。
    // 构建方式类似于渲染中的粒子SDF
    void BuildFluidSDFFromParticles(
        std::vector<glm::vec3> const & particles,
        SDFField &                     phiFluid,
        float                          particleRadius,
        float                          activeBand = 0.0f);

    // 刚体 SDF 复用 RigidBody::SignedDistance。
    // box/sphere 走解析 SDF，BoatHull 会优先走 main 分支新增的局部 SDF 缓存。
    void BuildSolidSDFFromRigidBodies(
        RigidBodySystem const & rigid,
        SDFField &              phiSolid,
        float                   maxDistance           = 0.0f,
        bool                    includeTankBoundaries = true);

} // namespace VCX::MainScene
