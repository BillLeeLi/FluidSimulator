#include "Simulation/FluidSolidCoupler.h"
#include "Simulation/FluidSimulator.h"
#include "Simulation/RigidBodySystem.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>

namespace VCX::MainScene {

    namespace Detail {
        constexpr float kEps = 1e-6f;
        struct RigidRowContribution {
            // 一个压力 dof 对某个刚体的线性耦合项
            // g 表示“该压力未知量对应的面冲量”作用到刚体广义速度 [v,w] 上的方向，
            // 后面会用 g^T V* 进入右端项，用 g^T M_s^-1 g 进入矩阵 A。
            int                      dof = -1;
            RigidGeneralizedVelocity g   = RigidGeneralizedVelocity::Zero();
        };

        struct BodyProjectionCache {
            // 每个刚体本次投影的临时缓存
            // vStar 是投影前刚体速度，invMass 是 M_s^-1，rows 是它接触到哪些压力 dof。
            int                      bodyId = -1;
            RigidGeneralizedVelocity vStar  = RigidGeneralizedVelocity::Zero();
            RigidInverseMassBlock    invMass = RigidInverseMassBlock::Zero();
            std::vector<RigidRowContribution> rows;
        };

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

        bool BodyOccupiesPoint(RigidBody const & body, Eigen::Vector3f const & worldPoint) {
            // 一个 worldPoint 是否在刚体里面？
            // 这里只需要布尔内外判断；box/sphere 会走刚体侧的快速 primitive path。
            return body.ContainsPoint(worldPoint);
        }

        bool BodyOccupiesCellCenter(FluidSimulator const & fluid, RigidBody const & body, glm::ivec3 const & cell) {
            // 判断cell center 是否在当前刚体内部。当前仍是中心点 0/1 判断；primitive 会走快速布尔 inside。
            // 后续做精确变分映射时，再通过 SignedDistance 采样比例。
            return BodyOccupiesPoint(body, ToEigen(fluid.CellCenter(cell)));
        }

    } // namespace Detail

    using namespace Detail;

    bool FluidSolidCoupler::SolveVariationalProjection(FluidSimulator & fluid, RigidBodySystem & rigid, float dt) {
        if (dt <= 0.0f || fluid.m_iNumCells <= 0 || fluid.m_type.empty() || fluid.m_vel.empty()) {
            return false;
        }

        // 先清理掉旧的 solid boundary/contact cache，避免读到上一帧的过时数据。
        fluid.ResetSolidBoundaryVelocity();
        _rigidContactFaceCacheValid = false;
        _rigidContactFaces.clear();

        FluidPressureDofs const dofs = fluid.BuildPressureDofs();
        int const          numDofs = static_cast<int>(dofs.dofCell.size());
        if (numDofs == 0) return false;

        // u*为压力投影之前的 MAC 网格速度。
        std::vector<glm::vec3> const uStar = fluid.m_vel;
        fluid.m_pre_vel = uStar;

        // Eigen::Triplet(row,col,value) 用来构建稀疏矩阵
        std::vector<Eigen::Triplet<float>> trips;
        trips.reserve(static_cast<std::size_t>(numDofs) * 8);
        Eigen::VectorXf rhs = Eigen::VectorXf::Zero(numDofs);

        // 密度固定，且用最简单的均匀密度和完整 face 面积，没有搞占比权重
        // area 是 face 面积，mass 是一个 face 控制体的近似流体质量，invMass 对应 M_f^-1。
        float const h    = std::max(fluid.m_h, kEps);
        float const area = h * h;
        float const density = std::max(fluidDensity, kEps);
        float const mass = std::max(density * h * h * h, kEps);
        float const invMass = 1.0f / mass;

        auto addMatrix = [&](int row, int col, float value) {
            if (row < 0 || col < 0 || std::abs(value) <= 1e-12f) return;
            trips.emplace_back(row, col, value);
        };

        auto isSolidCell = [&](glm::ivec3 const & cell) {
            if (! fluid.IsInsideGrid(cell)) return true;
            int const id = fluid.GridIndex(cell);
            return id < 0 || id >= static_cast<int>(fluid.m_type.size()) || fluid.m_type[id] == SOLID_CELL || fluid.IsCellSolid(cell);
        };

        auto addFluidFace = [&](glm::ivec3 const & face, int dir) {
            // Assemble one MAC face into the pressure projection.
            if (! fluid.IsVelocityFaceInRange(face, dir)) return;

            auto const [lowerCell, upperCell] = fluid.FaceNeighborCells(face, dir);
            int const lowerDof = fluid.PressureDofForCell(dofs, lowerCell);
            int const upperDof = fluid.PressureDofForCell(dofs, upperCell);
            if (lowerDof < 0 && upperDof < 0) return;  // 如果面相邻的两个cell都不是流体,不需要处理

            bool const lowerSolid = isSolidCell(lowerCell);
            bool const upperSolid = isSolidCell(upperCell);
            if (lowerSolid || upperSolid) {  // 至少存在一个固体邻居，即流固界面
                int const   fluidDof = lowerDof >= 0 ? lowerDof : upperDof;
                float const coeff    = lowerDof >= 0 ? area : -area;
                rhs[fluidDof] += coeff * fluid.SolidBoundaryVelocity(face.x, face.y, face.z, dir);
                return;  // 流固界面速度使用固体速度，不作为可以由压力任意调整的流体face DOF,所以不加入矩阵项
            }

            float const u = fluid.FaceVelocity(uStar, face, dir);

            std::array<std::pair<int, float>, 2> coeffs {
                std::pair<int, float> { lowerDof,  area },
                std::pair<int, float> { upperDof, -area },
            };

            for (auto const & a : coeffs) {
                if (a.first < 0) continue;
                // b 里放的是当前预测速度 u* 带来的通量，需要用压力冲量抵消。
                rhs[a.first] += a.second * u;
                for (auto const & b : coeffs) {
                    if (b.first < 0) continue;
                    // A 的流体部分加入的是 B_f M_f^-1 B_f^T
                    addMatrix(a.first, b.first, a.second * invMass * b.second);
                }
            }
        };

        // 遍历所有 MAC face，先组装纯流体压力投影项。
        for (int i = 0; i < fluid.m_iCellX; ++i) {
            for (int j = 0; j < fluid.m_iCellY; ++j) {
                for (int k = 0; k < fluid.m_iCellZ; ++k) {
                    glm::ivec3 const face(i, j, k);
                    addFluidFace(face, 0);
                    addFluidFace(face, 1);
                    addFluidFace(face, 2);
                }
            }
        }

        std::vector<glm::ivec3> dynamicRigidFaceMask(fluid.m_iNumCells, glm::ivec3(0));  // 标记动态的刚体的接触face
        std::vector<BodyProjectionCache> bodyCaches;
        bodyCaches.reserve(rigid.Bodies.size());

        // 把刚体-流体接触面转换成压力 dof 与刚体广义速度之间的耦合关系。
        // 由刚体 AABB 提供粗范围，在范围内优先找 FLUID_CELL，
        // 再检查它的 6 邻居是否被当前刚体占据。这样避免遍历 box/船内部大量无效 solid cell。
        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body)) continue;
            // 静态刚体作为固定边界已经由 solid mask 和 u_solid=0 处理，不需要进入动态刚体矩阵项。
            if (body.isStatic) continue;

            BodyProjectionCache cache;
            cache.bodyId  = bodyId;
            cache.vStar   = rigid.GetGeneralizedVelocity(bodyId);
            cache.invMass = rigid.GetInverseMassBlock(bodyId);

            auto appendFluidCellContact = [&](glm::ivec3 const & fluidCell, glm::ivec3 const & solidCell, int fluidDof) {
                // 从一个 FLUID_CELL 出发检查相邻 solidCell 是否属于当前刚体；
                // 成功时同时生成矩阵里的刚体耦合 row 和后续边界速度写回用的 contact face。
                if (! fluid.IsInsideGrid(solidCell) || ! fluid.IsCellSolid(solidCell)) return;
                if (! BodyOccupiesCellCenter(fluid, body, solidCell)) return;

                glm::ivec3 const solidToFluid = fluidCell - solidCell;
                int const        dir          = AxisFromOffset(solidToFluid);
                glm::ivec3 const face         = FaceIndexForContact(solidCell, solidToFluid);
                if (! fluid.IsVelocityFaceInRange(face, dir)) return;

                auto const [lowerCell, upperCell] = fluid.FaceNeighborCells(face, dir);
                float fluidCoeff = 0.0f;
                if (fluidCell == lowerCell) {
                    fluidCoeff = area;
                } else if (fluidCell == upperCell) {
                    fluidCoeff = -area;
                } else {
                    return;
                }

                Eigen::Vector3f const axis = ToEigen(fluid.FaceAxis(dir));
                Eigen::Vector3f const center = ToEigen(fluid.FaceCenter(face, dir));
                Eigen::Vector3f const r = center - body.x;

                RigidGeneralizedVelocity g = RigidGeneralizedVelocity::Zero();
                // 面压力对刚体产生线冲量和角冲量：
                // 线冲量方向沿 face 法向，角冲量是 r x 线冲量。
                // g uses the same signed flux coefficient as the fluid divergence row.
                // The actual pressure impulse applied to the body is -J^T lambda.
                g.segment<3>(0) = fluidCoeff * axis;
                g.segment<3>(3) = fluidCoeff * r.cross(axis);

                cache.rows.push_back(RigidRowContribution { fluidDof, g });
                // 这份 contact face 后面会被 ApplyRigidBoundaryVelocitiesToFluid 复用，
                // 避免 solve 后再通过刚体 AABB 重新找一遍同样的接触面。
                _rigidContactFaces.push_back(RigidContactFace {
                    bodyId,
                    fluidCell,
                    solidCell,
                    face,
                    dir,
                });

                int const faceId = fluid.GridIndex(face);
                if (faceId >= 0 && faceId < static_cast<int>(dynamicRigidFaceMask.size())) {
                    dynamicRigidFaceMask[faceId][dir] = 1;
                }
            };

            auto const [lo, hi] = BodyCellRange(fluid, body);
            // 只在刚体 AABB 粗范围内找 fluid pressure dof，再检查 6 邻居是否是当前刚体；
            // 比从刚体内部所有 solid cell 出发更少扫无效体积。
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const fluidCell(i, j, k);
                        int const        fluidDof = fluid.PressureDofForCell(dofs, fluidCell);
                        if (fluidDof < 0) continue;

                        for (glm::ivec3 const & fluidToSolid : kNeighborOffsets) {
                            glm::ivec3 const solidCell = fluidCell + fluidToSolid;
                            appendFluidCellContact(fluidCell, solidCell, fluidDof);
                        }
                    }
                }
            }

            if (! cache.rows.empty()) {
                // 同一个 pressure dof 可能从多个相邻 solidCell/face 收到贡献。
                // 先合并 g，后面的 g_a^T M^-1 g_b 与 g^T V* 都保持数学等价，
                // 但 rows 数量变小后，显式组装 J M_s^-1 J^T 的二次循环会快很多。
                std::sort(cache.rows.begin(), cache.rows.end(), [](RigidRowContribution const & a, RigidRowContribution const & b) {
                    return a.dof < b.dof;
                });

                std::vector<RigidRowContribution> mergedRows;
                mergedRows.reserve(cache.rows.size());
                for (RigidRowContribution const & row : cache.rows) {
                    if (mergedRows.empty() || mergedRows.back().dof != row.dof) {
                        mergedRows.push_back(row);
                    } else {
                        mergedRows.back().g += row.g;
                    }
                }

                cache.rows = std::move(mergedRows);
                bodyCaches.push_back(std::move(cache));
            }
        }

        for (BodyProjectionCache const & cache : bodyCaches) {
            for (RigidRowContribution const & row : cache.rows) {
                // b 的刚体部分是当前刚体预测速度 V* 穿过接触面的通量：J V*。
                rhs[row.dof] += row.g.dot(cache.vStar);
            }

            if (cache.invMass.cwiseAbs().maxCoeff() <= 0.0f) continue;

            for (RigidRowContribution const & a : cache.rows) {
                RigidGeneralizedVelocity const invMassG = cache.invMass * a.g;
                for (RigidRowContribution const & b : cache.rows) {
                    // A 的刚体部分显式组装为 J M_s^-1 J^T。
                    addMatrix(a.dof, b.dof, invMassG.dot(b.g));
                }
            }
        }

        Eigen::SparseMatrix<float> system(numDofs, numDofs);
        system.setFromTriplets(trips.begin(), trips.end());

        Eigen::ConjugateGradient<
            Eigen::SparseMatrix<float>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<float>>
            solver;
        // 临时关闭 matrix-free：显式组装完整稀疏系统，便于对比求解性能。
        solver.setMaxIterations(200);
        solver.setTolerance(1e-5f);
        solver.compute(system);
        if (solver.info() != Eigen::Success) return false;

        Eigen::VectorXf lambda = solver.solve(rhs);
        if (solver.info() != Eigen::Success || lambda.size() != numDofs) return false;

        if (_pressureForcesByBody.size() != rigid.Bodies.size()) {
            _pressureForcesByBody.assign(rigid.Bodies.size(), Eigen::Vector3f::Zero());
        }

        // 用解出的 lambda 回代更新流体 face 速度：u_new = u* - M_f^-1 B_f^T lambda。
        for (int i = 0; i < fluid.m_iCellX; ++i) {
            for (int j = 0; j < fluid.m_iCellY; ++j) {
                for (int k = 0; k < fluid.m_iCellZ; ++k) {
                    glm::ivec3 const face(i, j, k);
                    for (int dir = 0; dir < 3; ++dir) {
                        if (! fluid.IsVelocityFaceInRange(face, dir)) continue;

                        auto const [lowerCell, upperCell] = fluid.FaceNeighborCells(face, dir);
                        int const lowerDof = fluid.PressureDofForCell(dofs, lowerCell);
                        int const upperDof = fluid.PressureDofForCell(dofs, upperCell);
                        if (lowerDof < 0 && upperDof < 0) continue;

                        // 更新速度时固液界面上不使用压力更新，直接使用固体的边界速度
                        bool const lowerSolid = isSolidCell(lowerCell);
                        bool const upperSolid = isSolidCell(upperCell);
                        if (lowerSolid || upperSolid) {
                            int const faceId = fluid.GridIndex(face);
                            // 回代时动态刚体界面 face 不再写 SolidBoundaryVelocity() 的旧值/零值；静态墙面仍写 0。
                            if (enableMovingSolidVelocity
                                && faceId >= 0
                                && faceId < static_cast<int>(dynamicRigidFaceMask.size())
                                && dynamicRigidFaceMask[faceId][dir] != 0) {
                                continue;
                            }
                            fluid.m_vel[faceId][dir] = 0.0f;
                            continue;
                        }

                        float const lambdaLower = lowerDof >= 0 ? lambda[lowerDof] : 0.0f;
                        float const lambdaUpper = upperDof >= 0 ? lambda[upperDof] : 0.0f;
                        float const impulse     = area * (lambdaLower - lambdaUpper);

                        int const faceId = fluid.GridIndex(face);
                        fluid.m_vel[faceId][dir] = uStar[faceId][dir] - invMass * impulse;
                    }
                }
            }
        }

        // 同一个 lambda 也要更新刚体广义速度：V_new = V* - M_s^-1 J^T lambda。
        for (BodyProjectionCache const & cache : bodyCaches) {
            if (cache.bodyId < 0 || cache.bodyId >= static_cast<int>(rigid.Bodies.size())) continue;
            if (rigid.Bodies[cache.bodyId].isStatic) continue;

            RigidGeneralizedVelocity impulse = RigidGeneralizedVelocity::Zero();
            for (RigidRowContribution const & row : cache.rows) {
                impulse += row.g * lambda[row.dof];
            }

            // 更新刚体速度之后将新的刚体边界速度写回MAC网格
            _pressureForcesByBody[cache.bodyId] += -impulse.segment<3>(0) / dt;
            rigid.SetGeneralizedVelocity(cache.bodyId, cache.vStar - cache.invMass * impulse);
        }

        // m_p 仍然保存压力用于可视化接口；lambda 是压力冲量，所以除以 dt 近似得到压力。
        _rigidContactFaceCacheValid = true;
        ApplyRigidBoundaryVelocitiesToFluid(fluid, rigid, false);

        std::fill(fluid.m_p.begin(), fluid.m_p.end(), 0.0f);
        for (int dof = 0; dof < numDofs; ++dof) {
            fluid.m_p[fluid.GridIndex(dofs.dofCell[dof])] = lambda[dof] / dt;
        }

        return true;
    }

    void FluidSolidCoupler::ResetDebug() {
        _projectedParticleCount = 0;
        _rigidSolidCellCount = 0;
        _pressureContactFaceCount = 0;
        _movingBoundaryFaceCount = 0;
        std::fill(_pressureForcesByBody.begin(), _pressureForcesByBody.end(), Eigen::Vector3f::Zero());
        std::fill(_particleImpulsesByBody.begin(), _particleImpulsesByBody.end(), Eigen::Vector3f::Zero());
        std::fill(_boatBuoyancyForcesByBody.begin(), _boatBuoyancyForcesByBody.end(), Eigen::Vector3f::Zero());
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

    Eigen::Vector3f FluidSolidCoupler::BoatBuoyancyForceOnBody(int bodyId) const {
        if (bodyId < 0 || bodyId >= static_cast<int>(_boatBuoyancyForcesByBody.size())) {
            return Eigen::Vector3f::Zero();
        }
        return _boatBuoyancyForcesByBody[bodyId];
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
                if (! BodyOccupiesPoint(body, position)) continue;

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
        _rigidContactFaceCacheValid = false;
        _rigidContactFaces.clear();
        if (! enableRigidSolidMask) {
            _rigidSolidCellCount = 0;
            return 0;
        }

        int solidCells = 0;
        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body)) continue;

            auto const [lo, hi] = BodyCellRange(fluid, body);
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const cell(i, j, k);
                        if (fluid.IsCellSolid(cell)) continue;

                        Eigen::Vector3f const center = ToEigen(fluid.CellCenter(cell));
                        if (! BodyOccupiesPoint(body, center)) continue;

                        fluid.SetCellSolid(cell, true);
                        ++solidCells;
                    }
                }
            }
        }

        _rigidSolidCellCount = solidCells;
        return solidCells;
    }

    int FluidSolidCoupler::ApplyRigidBoundaryVelocitiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid, bool updatePreVel) {
        fluid.ResetSolidBoundaryVelocity();
        if (! enableMovingSolidVelocity || ! enableRigidSolidMask || fluid.m_type.empty()) {
            return 0;
        }

        int boundaryFaces = 0;

        if (_rigidContactFaceCacheValid) {
            // 变分投影已经找到过动态刚体接触面时，直接在这些 face 上写刚体速度。
            // 这样可以避免重复扫描刚体 AABB 和 boundary cells。
            for (RigidContactFace const & contact : _rigidContactFaces) {
                if (contact.bodyId < 0 || contact.bodyId >= static_cast<int>(rigid.Bodies.size())) continue;
                RigidBody const & body = rigid.Bodies[contact.bodyId];
                if (IsInternalTankBoundary(body)) continue;
                if (! fluid.IsVelocityFaceInRange(contact.face, contact.dir)) continue;

                Eigen::Vector3f const center = ToEigen(fluid.FaceCenter(contact.face, contact.dir));
                Eigen::Vector3f const vel    = rigid.VelocityAtPoint(contact.bodyId, center);
                fluid.SetSolidBoundaryVelocity(contact.face, contact.dir, vel[contact.dir], updatePreVel);
                ++boundaryFaces;
            }

            _movingBoundaryFaceCount += boundaryFaces;
            return boundaryFaces;
        }

        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body)) continue;

            auto applySolidCell = [&](glm::ivec3 const & solidCell) {
                // fallback 路径：没有有效 contact cache 时，从固体 cell 找相邻流体 cell，
                // 再把该接触 MAC face 的速度设置为刚体表面速度。
                if (! fluid.IsInsideGrid(solidCell) || ! fluid.IsCellSolid(solidCell)) return;

                for (glm::ivec3 const & offset : kNeighborOffsets) {
                    glm::ivec3 const fluidCell = solidCell + offset;
                    if (! fluid.IsInsideGrid(fluidCell)) continue;

                    int const fluidId = fluid.GridIndex(fluidCell);
                    if (fluid.m_type[fluidId] != FLUID_CELL) continue;

                    int const         dir        = AxisFromOffset(offset);
                    glm::ivec3 const  face       = FaceIndexForContact(solidCell, offset);
                    Eigen::Vector3f const center = ToEigen(fluid.FaceCenter(face, dir));
                    Eigen::Vector3f const vel    = rigid.VelocityAtPoint(bodyId, center);

                    fluid.SetSolidBoundaryVelocity(face, dir, vel[dir], updatePreVel);
                    ++boundaryFaces;
                }
            };

            auto const [lo, hi] = BodyCellRange(fluid, body);
            // 非变分投影或求解失败时仍需要这条备用扫描路径，保证移动固体边界速度可用。
            for (int i = lo.x; i <= hi.x; ++i) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int k = lo.z; k <= hi.z; ++k) {
                        glm::ivec3 const solidCell(i, j, k);
                        if (! BodyOccupiesCellCenter(fluid, body, solidCell)) continue;
                        applySolidCell(solidCell);
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
                glm::vec3 const      probe  = ToGlm(sample.position + normal * (0.25f * fluid.m_h));
                float const pressure = std::clamp(std::max(0.0f, -fluid.SamplePressure(probe)), 0.0f, maxPressureForForce);
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


    int FluidSolidCoupler::ApplyBoatBuoyancyForces(FluidSimulator const & fluid, RigidBodySystem & rigid) {
        if (_boatBuoyancyForcesByBody.size() != rigid.Bodies.size()) {
            _boatBuoyancyForcesByBody.assign(rigid.Bodies.size(), Eigen::Vector3f::Zero());
        }
        if (! enableBoatBuoyancy || boatBuoyancyScale <= 0.0f) return 0;

        auto estimateLocalWater = [&](Eigen::Vector3f const & worldPoint, float radius, float & waterY, Eigen::Vector3f & waterVelocity) -> bool {
            if (fluid.m_iNumSpheres <= 0 || fluid.m_particlePos.empty()) return false;

            float const searchR = std::max(2.8f * fluid.m_h, 1.8f * radius);
            float const searchR2 = searchR * searchR;
            float highestY = -1.0e9f;
            Eigen::Vector3f velSum = Eigen::Vector3f::Zero();
            int count = 0;

            for (int i = 0; i < fluid.m_iNumSpheres; ++i) {
                Eigen::Vector3f const p = ToEigen(fluid.m_particlePos[i]);
                float const dx = p.x() - worldPoint.x();
                float const dz = p.z() - worldPoint.z();
                if (dx * dx + dz * dz > searchR2) continue;

                highestY = std::max(highestY, p.y() + fluid.m_particleRadius);
                if (i < static_cast<int>(fluid.m_particleVel.size())) {
                    velSum += ToEigen(fluid.m_particleVel[i]);
                }
                ++count;
            }

            if (count == 0) return false;
            waterY = highestY + boatWaterLevel;
            waterVelocity = velSum / float(count);
            return true;
        };

        int usedSamples = 0;
        float const g = 9.81f;

        for (int bodyId = 0; bodyId < static_cast<int>(rigid.Bodies.size()); ++bodyId) {
            RigidBody const & body = rigid.Bodies[bodyId];
            if (IsInternalTankBoundary(body) || body.isStatic) continue;
            if (body.shape != RigidBodyShape::BoatHull) continue;

            std::vector<RigidBuoyancySample> fallbackSamples;
            std::vector<RigidBuoyancySample> const * samples = &body.buoyancySamples;
            if (samples->empty()) {
                for (int ix = 0; ix < 3; ++ix) {
                    for (int iz = 0; iz < 3; ++iz) {
                        RigidBuoyancySample s;
                        s.localPosition = Eigen::Vector3f((ix - 1) * 0.08f, -0.06f, (iz - 1) * 0.05f);
                        s.volumeWeight = 1.0f / 9.0f;
                        s.radius = 0.035f;
                        fallbackSamples.push_back(s);
                    }
                }
                samples = &fallbackSamples;
            }

            float totalWeight = 0.0f;
            for (auto const & sample : *samples) totalWeight += std::max(0.0f, sample.volumeWeight);
            if (totalWeight <= kEps) continue;

            for (auto const & sample : *samples) {
                Eigen::Vector3f const worldPoint = body.LocalToWorld(sample.localPosition);
                float const r = std::max(0.005f, sample.radius);

                float waterY = 0.0f;
                Eigen::Vector3f waterVelocity = Eigen::Vector3f::Zero();
                if (! estimateLocalWater(worldPoint, r, waterY, waterVelocity)) continue;

                float const depth = waterY - worldPoint.y();
                float const submerge = std::clamp((depth + r) / (2.0f * r), 0.0f, 1.0f);
                if (submerge <= 0.0f) continue;

                float const weightShare = std::max(0.0f, sample.volumeWeight) / totalWeight;
                Eigen::Vector3f force = Eigen::Vector3f::UnitY() * (body.mass * g * boatBuoyancyScale * weightShare * submerge);

                Eigen::Vector3f relVel = rigid.VelocityAtPoint(bodyId, worldPoint) - waterVelocity;
                float const relNorm = relVel.norm();
                if (relNorm > 1.2f) relVel *= 1.2f / relNorm;

                Eigen::Vector3f drag = -boatWaterDrag * body.mass * weightShare * submerge * relVel;
                drag.y() *= 1.6f;
                force += drag;

                float const maxForce = 1.6f * body.mass * g * weightShare;
                float const fNorm = force.norm();
                if (fNorm > maxForce) force *= maxForce / fNorm;

                rigid.ApplyForce(bodyId, force, worldPoint);
                _boatBuoyancyForcesByBody[bodyId] += force;
                ++usedSamples;
            }
        }

        return usedSamples;
    }



} // namespace VCX::MainScene
