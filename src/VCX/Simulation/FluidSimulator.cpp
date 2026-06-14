#include "Simulation/FluidSimulator.h"
#include "Simulation/RigidBodySystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace VCX::MainScene {

    // ==================== 辅助函数 ====================

    int FluidSimulator::index2GridOffset(glm::ivec3 idx) const {
        return idx.x * (m_iCellY * m_iCellZ) + idx.y * m_iCellZ + idx.z;
    }

    int FluidSimulator::clampSlot(int v, int hi) {
        return std::max(0, std::min(v, hi));
    }

    int FluidSimulator::clampCoord(int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    }

    float FluidSimulator::weight(glm::vec3 gridPos, glm::vec3 partPos) const {
        glm::vec3 d = (partPos - gridPos) * m_fInvSpacing;
        float     w = 1.0f;
        for (int i = 0; i < 3; i++) {
            float ad = std::abs(d[i]);
            if (ad < 1.0f)
                w *= (1.0f - ad);
            else
                return 0.0f;
        }
        return w;
    }

    void FluidSimulator::buildHash() {
        std::fill(m_hashtableindex.begin(), m_hashtableindex.end(), 0);
        // 1. 统计每个格点中的粒子数
        for (int p = 0; p < m_iNumSpheres; ++p) {
            glm::ivec3 slot   = worldToCell(m_particlePos[p]);
            m_particleSlot[p] = slot;
            int const id      = index2GridOffset(slot);
            ++m_hashtableindex[id + 1];
        }
        // 2. 前缀和 → 每个格点在哈希表中的起止范围
        for (int i = 1; i <= m_iNumCells; ++i)
            m_hashtableindex[i] += m_hashtableindex[i - 1];
        // 3. 散射粒子到哈希表
        std::vector<int> writeCursor = m_hashtableindex;
        for (int p = 0; p < m_iNumSpheres; ++p) {
            int const id                   = index2GridOffset(m_particleSlot[p]);
            m_hashtable[writeCursor[id]++] = p;
        }
    }

    bool FluidSimulator::isVelocityFaceInRange(int i, int j, int k, int dir) const {
        if (i < 0 || i >= m_iCellX || j < 0 || j >= m_iCellY || k < 0 || k >= m_iCellZ)
            return false;
        switch (dir) {
        case 0: // x-face: 有效范围 i∈[1, CX-2], j∈[0, CY-2], k∈[0, CZ-2]
            if (i == 0 || i >= m_iCellX - 1 || j >= m_iCellY - 1 || k >= m_iCellZ - 1) return false;
            break;
        case 1: // y-face: 有效范围 i∈[0, CX-2], j∈[1, CY-2], k∈[0, CZ-2]
            if (j == 0 || j >= m_iCellY - 1 || i >= m_iCellX - 1 || k >= m_iCellZ - 1) return false;
            break;
        case 2: // z-face: 有效范围 i∈[0, CX-2], j∈[0, CY-2], k∈[1, CZ-2]
            if (k == 0 || k >= m_iCellZ - 1 || i >= m_iCellX - 1 || j >= m_iCellY - 1) return false;
            break;
        default: return false;
        }
        return true;
    }

    bool FluidSimulator::isValidVelocity(int i, int j, int k, int dir) const {
        if (! isVelocityFaceInRange(i, j, k, dir)) return false;

        // MAC face 代表两个 pressure cell 之间的通量。只检查 face 自己所在的 cell 会漏掉一半墙面：
        // 例如负 x 玻璃壁的边界 face 索引是 i=1，m_s(1,j,k) 是流体，于是旧代码会把它当成普通流体 face。
        // 这些未被压力投影真正修正的墙面速度之后又被 G2P 采回粒子，是玻璃边缘持续抖动的主要来源。
        // 因此普通流体速度 face 必须要求两侧 cell 都不是固体；固体边界 face 由
        // EnforceSolidBoundaryVelocities()/SolidBoundaryVelocity 单独处理。
        glm::ivec3 const face(i, j, k);
        auto const [lower, upper] = FaceNeighborCells(face, dir);
        if (! IsInsideGrid(lower) || ! IsInsideGrid(upper)) return false;
        if (IsCellSolid(lower) || IsCellSolid(upper)) return false;

        return m_s[index2GridOffset(face)] > 0.5f;
    }

    // ==================== 公共接口 =====================

    void FluidSimulator::integrateParticles(float timeStep) {
        for (int i = 0; i < m_iNumSpheres; i++) {
            if (enableGravity) m_particleVel[i] += gravity * timeStep;
            m_particlePos[i] += m_particleVel[i] * timeStep;
        }
    }

    void FluidSimulator::handleParticleCollisions() {
        float const minBound = -0.5f + m_h + m_particleRadius;
        float const maxBound = 0.5f - m_h - m_particleRadius;

        for (int i = 0; i < m_iNumSpheres; i++) {
            // —— 水槽六面边界 (Solid at i=0, i>=CX-2, etc.) ——
            // x方向
            if (m_particlePos[i].x < minBound) {
                m_particlePos[i].x = minBound;
                m_particleVel[i].x = 0.0f;
            } else if (m_particlePos[i].x > maxBound) {
                m_particlePos[i].x = maxBound;
                m_particleVel[i].x = 0.0f;
            }
            // y方向
            if (m_particlePos[i].y < minBound) {
                m_particlePos[i].y = minBound;
                m_particleVel[i].y = 0.0f;
            } else if (m_particlePos[i].y > maxBound) {
                m_particlePos[i].y = maxBound;
                m_particleVel[i].y = 0.0f;
            }
            // z方向
            if (m_particlePos[i].z < minBound) {
                m_particlePos[i].z = minBound;
                m_particleVel[i].z = 0.0f;
            } else if (m_particlePos[i].z > maxBound) {
                m_particlePos[i].z = maxBound;
                m_particleVel[i].z = 0.0f;
            }
        }
    }

    void FluidSimulator::pushParticlesApart(int numIters) {
        for (int iter = 0; iter < numIters; iter++) {
            buildHash(); // 每轮迭代重建哈希表以反映最新位置

            float const minDist = 2.0f * m_particleRadius;

            for (int i = 0; i < m_iNumSpheres; i++) {
                glm::ivec3 const slot = m_particleSlot[i];

                // 仅在当前粒子所在格点及其26个相邻格点中搜索
                for (int di = -1; di <= 1; di++) {
                    int ni = clampSlot(slot.x + di, m_iCellX - 1);
                    for (int dj = -1; dj <= 1; dj++) {
                        int nj = clampSlot(slot.y + dj, m_iCellY - 1);
                        for (int dk = -1; dk <= 1; dk++) {
                            int nk         = clampSlot(slot.z + dk, m_iCellZ - 1);
                            int neighborId = index2GridOffset(glm::ivec3(ni, nj, nk));

                            for (int idx = m_hashtableindex[neighborId];
                                 idx < m_hashtableindex[neighborId + 1];
                                 idx++) {
                                int j = m_hashtable[idx];
                                if (i >= j) continue; // 每对只处理一次

                                glm::vec3 delta = m_particlePos[i] - m_particlePos[j];
                                float     dist  = glm::length(delta);
                                if (dist >= minDist) continue;

                                glm::vec3 normal(0.0f);
                                if (dist > 1e-8f) {
                                    normal = delta / dist;
                                } else {
                                    // 完全重合时没有可用法向；给粒子对一个稳定的退化分离方向，
                                    // 避免角落/边界投影后位置完全相同时再也推不开。
                                    unsigned const key = unsigned(i) * 73856093u
                                        ^ unsigned(j) * 19349663u
                                        ^ unsigned(iter) * 83492791u;
                                    normal = glm::normalize(glm::vec3(
                                        (key & 1u) ? 1.0f : -1.0f,
                                        (key & 2u) ? 1.0f : -1.0f,
                                        (key & 4u) ? 1.0f : -1.0f));
                                }

                                glm::vec3 const corr = 0.5f * (minDist - dist) * normal;
                                m_particlePos[i] += corr;
                                m_particlePos[j] -= corr;
                            }
                        }
                    }
                }
            }
        }
    }

    void FluidSimulator::transferVelocities(bool toGrid, float flipRatio) {
        if (toGrid) {
            // ============ P2G: 粒子 → 网格 ============

            std::fill(m_vel.begin(), m_vel.end(), glm::vec3(0.0f));
            for (int d = 0; d < 3; ++d)
                std::fill(m_near_num[d].begin(), m_near_num[d].end(), 0.0f);

            buildHash();

            // 辅助lambda: 在采样点samlePos处, 搜索邻域粒子并累积dir方向速度
            auto accumulateDir = [&](int dir, glm::vec3 const & samplePos, glm::ivec3 const & sampleSlot) {
                // 邻域搜索范围: ±1个格点
                int i0 = clampSlot(sampleSlot.x - 1, m_iCellX - 1);
                int j0 = clampSlot(sampleSlot.y - 1, m_iCellY - 1);
                int k0 = clampSlot(sampleSlot.z - 1, m_iCellZ - 1);
                int i1 = clampSlot(sampleSlot.x + 1, m_iCellX - 1);
                int j1 = clampSlot(sampleSlot.y + 1, m_iCellY - 1);
                int k1 = clampSlot(sampleSlot.z + 1, m_iCellZ - 1);

                int const sampleId = index2GridOffset(sampleSlot);

                for (int ii = i0; ii <= i1; ++ii) {
                    for (int jj = j0; jj <= j1; ++jj) {
                        for (int kk = k0; kk <= k1; ++kk) {
                            int const neighborId = index2GridOffset(glm::ivec3(ii, jj, kk));
                            for (int ptr = m_hashtableindex[neighborId];
                                 ptr < m_hashtableindex[neighborId + 1];
                                 ++ptr) {
                                int   p = m_hashtable[ptr];
                                float w = weight(samplePos, m_particlePos[p]);
                                if (w <= 0.0f) continue;
                                m_vel[sampleId][dir] += w * m_particleVel[p][dir];
                                m_near_num[dir][sampleId] += w;
                            }
                        }
                    }
                }
            };

            // —— 遍历所有x方向速度面 u(i, j+1/2, k+1/2) ——
            for (int i = 0; i < m_iCellX; ++i) {
                for (int j = 0; j < m_iCellY - 1; ++j) {
                    for (int k = 0; k < m_iCellZ - 1; ++k) {
                        if (! isValidVelocity(i, j, k, 0)) continue;
                        // x-速度面的物理位置
                        glm::vec3 samplePos = glm::vec3(i, j + 0.5f, k + 0.5f) * m_h + glm::vec3(-0.5f);
                        accumulateDir(0, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // —— 遍历所有y方向速度面 v(i+1/2, j, k+1/2) ——
            for (int i = 0; i < m_iCellX - 1; ++i) {
                for (int j = 0; j < m_iCellY; ++j) {
                    for (int k = 0; k < m_iCellZ - 1; ++k) {
                        if (! isValidVelocity(i, j, k, 1)) continue;
                        glm::vec3 samplePos = glm::vec3(i + 0.5f, j, k + 0.5f) * m_h + glm::vec3(-0.5f);
                        accumulateDir(1, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // —— 遍历所有z方向速度面 w(i+1/2, j+1/2, k) ——
            for (int i = 0; i < m_iCellX - 1; ++i) {
                for (int j = 0; j < m_iCellY - 1; ++j) {
                    for (int k = 0; k < m_iCellZ; ++k) {
                        if (! isValidVelocity(i, j, k, 2)) continue;
                        glm::vec3 samplePos = glm::vec3(i + 0.5f, j + 0.5f, k) * m_h + glm::vec3(-0.5f);
                        accumulateDir(2, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // 归一化: 每个速度面除以其累积权重
            for (int id = 0; id < m_iNumCells; ++id) {
                if (m_near_num[0][id] > 1e-6f) m_vel[id].x /= m_near_num[0][id];
                if (m_near_num[1][id] > 1e-6f) m_vel[id].y /= m_near_num[1][id];
                if (m_near_num[2][id] > 1e-6f) m_vel[id].z /= m_near_num[2][id];
            }

            m_pre_vel = m_vel;
        } else {
            // ============ G2P: 网格 → 粒子 ============

            // 辅助lambda: 在samplePos处采样dir方向的速度 (使用三线性插值)
            // 调用者负责将samplePos偏移到对应的交错速度面位置
            // useFlipDelta=true → 采样速度变化量 (m_vel-m_pre_vel) for FLIP
            // useFlipDelta=false → 采样当前速度 for PIC
            auto sampleVelocity = [&](glm::vec3 samplePos, int dir, bool useFlipDelta, float fallback) -> float {
                // 直接转换到网格坐标 (调用者已预处理偏移)
                glm::vec3 gridPos = (samplePos + glm::vec3(0.5f)) * m_fInvSpacing;

                // 钳制到有效插值范围
                gridPos.x = std::max(0.0f, std::min(gridPos.x, float(m_iCellX - 1) - 1e-4f));
                gridPos.y = std::max(0.0f, std::min(gridPos.y, float(m_iCellY - 1) - 1e-4f));
                gridPos.z = std::max(0.0f, std::min(gridPos.z, float(m_iCellZ - 1) - 1e-4f));

                int   i0 = static_cast<int>(std::floor(gridPos.x));
                int   j0 = static_cast<int>(std::floor(gridPos.y));
                int   k0 = static_cast<int>(std::floor(gridPos.z));
                float fx = gridPos.x - i0;
                float fy = gridPos.y - j0;
                float fz = gridPos.z - k0;

                float accum = 0.0f;
                float wsum  = 0.0f;
                for (int di = 0; di <= 1; ++di) {
                    float wx = di ? fx : (1.0f - fx);
                    for (int dj = 0; dj <= 1; ++dj) {
                        float wy = dj ? fy : (1.0f - fy);
                        for (int dk = 0; dk <= 1; ++dk) {
                            float      wz = dk ? fz : (1.0f - fz);
                            glm::ivec3 idx(i0 + di, j0 + dj, k0 + dk);
                            bool const regularFace = isValidVelocity(idx.x, idx.y, idx.z, dir);
                            bool const boundaryFace = IsVelocityFaceSolidBoundary(idx, dir);
                            if (! regularFace && ! boundaryFace) continue;

                            int   id  = index2GridOffset(idx);
                            float w   = wx * wy * wz;
                            float val = useFlipDelta ? (m_vel[id][dir] - m_pre_vel[id][dir])
                                                     : m_vel[id][dir];
                            accum += w * val;
                            wsum += w;
                        }
                    }
                }
                return (wsum > 1e-6f) ? (accum / wsum) : fallback;
            };

            for (int i = 0; i < m_iNumSpheres; i++) {
                glm::vec3 const oldVel = m_particleVel[i];

                // 将粒子位置偏移到交错面上的对应位置:
                // x-速度面 u 位于 (i, j+1/2, k+1/2) → 偏移 (0, -0.5h, -0.5h)
                // y-速度面 v 位于 (i+1/2, j, k+1/2) → 偏移 (-0.5h, 0, -0.5h)
                // z-速度面 w 位于 (i+1/2, j+1/2, k) → 偏移 (-0.5h, -0.5h, 0)
                glm::vec3 const px = m_particlePos[i] + glm::vec3(0.0f, -0.5f, -0.5f) * m_h;
                glm::vec3 const py = m_particlePos[i] + glm::vec3(-0.5f, 0.0f, -0.5f) * m_h;
                glm::vec3 const pz = m_particlePos[i] + glm::vec3(-0.5f, -0.5f, 0.0f) * m_h;

                // PIC: 直接从网格插值得到新速度 (耗散但稳定)
                glm::vec3 pic_vel;
                pic_vel.x = sampleVelocity(px, 0, false, oldVel.x);
                pic_vel.y = sampleVelocity(py, 1, false, oldVel.y);
                pic_vel.z = sampleVelocity(pz, 2, false, oldVel.z);

                // FLIP: 旧速度 + 网格速度变化量 (非耗散但可能有噪声)
                glm::vec3 flip_vel = oldVel;
                flip_vel.x += sampleVelocity(px, 0, true, 0.0f);
                flip_vel.y += sampleVelocity(py, 1, true, 0.0f);
                flip_vel.z += sampleVelocity(pz, 2, true, 0.0f);

                // 混合: flipRatio控制FLIP比例, (1-flipRatio)控制PIC比例
                m_particleVel[i] = flipRatio * flip_vel + (1.0f - flipRatio) * pic_vel;
            }
        }
    }

    void FluidSimulator::updateParticleDensity() {
        std::fill(m_particleDensity.begin(), m_particleDensity.end(), 0.0f);

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int const       id         = index2GridOffset(glm::ivec3(i, j, k));
                    glm::vec3 const gridPos    = glm::vec3(i, j, k) * m_h + glm::vec3(-0.5f);
                    glm::ivec3      centerSlot = worldToCell(gridPos);

                    // 在3×3×3邻域格点中搜索粒子
                    int i0 = clampSlot(centerSlot.x - 1, m_iCellX - 1);
                    int j0 = clampSlot(centerSlot.y - 1, m_iCellY - 1);
                    int k0 = clampSlot(centerSlot.z - 1, m_iCellZ - 1);
                    int i1 = clampSlot(centerSlot.x + 1, m_iCellX - 1);
                    int j1 = clampSlot(centerSlot.y + 1, m_iCellY - 1);
                    int k1 = clampSlot(centerSlot.z + 1, m_iCellZ - 1);

                    float density = 0.0f;
                    for (int ii = i0; ii <= i1; ++ii) {
                        for (int jj = j0; jj <= j1; ++jj) {
                            for (int kk = k0; kk <= k1; ++kk) {
                                int neighborId = index2GridOffset(glm::ivec3(ii, jj, kk));
                                for (int ptr = m_hashtableindex[neighborId];
                                     ptr < m_hashtableindex[neighborId + 1];
                                     ++ptr) {
                                    int p = m_hashtable[ptr];
                                    density += weight(gridPos, m_particlePos[p]);
                                }
                            }
                        }
                    }
                    m_particleDensity[id] = density;
                }
            }
        }

        // 同步更新单元格类型 (供 solveIncompressibility 使用)
        for (int cell = 0; cell < m_iNumCells; ++cell) {
            if (m_s[cell] <= 0.0f) {
                m_type[cell] = SOLID_CELL;
            } else {
                int const cnt = m_hashtableindex[cell + 1] - m_hashtableindex[cell];
                m_type[cell]  = (cnt > 0) ? FLUID_CELL : EMPTY_CELL;
            }
        }
    }

    void FluidSimulator::solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift) {
        // 清零压力
        std::fill(m_p.begin(), m_p.end(), 0.0f);

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 0; i < m_iCellX; i++) {
                for (int j = 0; j < m_iCellY; j++) {
                    for (int k = 0; k < m_iCellZ; k++) {
                        int const id = index2GridOffset(glm::ivec3(i, j, k));
                        if (m_type[id] != FLUID_CELL) continue;
                        // 这里将刚体考虑在内，与固体相邻的面使用刚体边界速度; 没有刚体边界时等价于静止墙速度0.
                        glm::ivec3 const cell(i, j, k);
                        auto const       fluidWeight = [&](glm::ivec3 const & neighbor) -> float {
                            if (! IsInsideGrid(neighbor)) return 0.0f;
                            return m_s[index2GridOffset(neighbor)];
                        };
                        auto const faceVelocity = [&](glm::ivec3 const & face, glm::ivec3 const & neighbor, int dir) -> float {
                            if (! IsInsideGrid(neighbor) || IsCellSolid(neighbor)) {
                                return SolidBoundaryVelocity(face.x, face.y, face.z, dir);
                            }
                            if (! isValidVelocity(face.x, face.y, face.z, dir)) return 0.0f;
                            return m_vel[index2GridOffset(face)][dir];
                        };

                        // 1. 计算该格点的速度散度 (右-左 + 上-下 + 前-后)
                        //    与固体相邻的面使用刚体写入的运动边界速度; 未写入时等价于静止墙速度0.
                        float divergence = 0.0f;
                        // 流入面 (左/下/后) 贡献为负
                        divergence -= faceVelocity(cell, cell + glm::ivec3(-1, 0, 0), 0);
                        divergence -= faceVelocity(cell, cell + glm::ivec3(0, -1, 0), 1);
                        divergence -= faceVelocity(cell, cell + glm::ivec3(0, 0, -1), 2);
                        // 流出面 (右/上/前) 贡献为正
                        divergence += faceVelocity(cell + glm::ivec3(1, 0, 0), cell + glm::ivec3(1, 0, 0), 0);
                        divergence += faceVelocity(cell + glm::ivec3(0, 1, 0), cell + glm::ivec3(0, 1, 0), 1);
                        divergence += faceVelocity(cell + glm::ivec3(0, 0, 1), cell + glm::ivec3(0, 0, 1), 2);

                        // 2. 过松弛 (加速收敛)
                        divergence *= overRelaxation;

                        // 3. 密度漂移补偿 (补偿因插值误差导致的体积变化)
                        if (compensateDrift) {
                            float densityError = m_particleDensity[id] - m_particleRestDensity;
                            divergence += 0.2f * densityError;
                        }

                        // 4. 计算6个邻居的流体权重和
                        //    固体邻居贡献 m_s=0, 自然不计入
                        float s = 0.0f;
                        s += fluidWeight(cell + glm::ivec3(1, 0, 0));
                        s += fluidWeight(cell + glm::ivec3(-1, 0, 0));
                        s += fluidWeight(cell + glm::ivec3(0, -1, 0));
                        s += fluidWeight(cell + glm::ivec3(0, 1, 0));
                        s += fluidWeight(cell + glm::ivec3(0, 0, 1));
                        s += fluidWeight(cell + glm::ivec3(0, 0, -1));

                        if (s < 1e-6f) continue;

                        // 5. 将散度修正按邻居权重分配到各速度面
                        //    流体邻居权重高 → 承担更多修正; 固体邻居 → 不修正
                        if (isValidVelocity(i, j, k, 0))
                            m_vel[id].x += divergence * fluidWeight(cell + glm::ivec3(-1, 0, 0)) / s;
                        if (isValidVelocity(i, j, k, 1))
                            m_vel[id].y += divergence * fluidWeight(cell + glm::ivec3(0, -1, 0)) / s;
                        if (isValidVelocity(i, j, k, 2))
                            m_vel[id].z += divergence * fluidWeight(cell + glm::ivec3(0, 0, -1)) / s;

                        if (isValidVelocity(i + 1, j, k, 0))
                            m_vel[index2GridOffset(glm::ivec3(i + 1, j, k))].x -= divergence * fluidWeight(cell + glm::ivec3(1, 0, 0)) / s;
                        if (isValidVelocity(i, j + 1, k, 1))
                            m_vel[index2GridOffset(glm::ivec3(i, j + 1, k))].y -= divergence * fluidWeight(cell + glm::ivec3(0, 1, 0)) / s;
                        if (isValidVelocity(i, j, k + 1, 2))
                            m_vel[index2GridOffset(glm::ivec3(i, j, k + 1))].z -= divergence * fluidWeight(cell + glm::ivec3(0, 0, 1)) / s;

                        // 6. 累积压力 (用于可视化和分析)
                        m_p[id] += divergence * m_h / (s * dt);
                    }
                }
            }
        }
    }

    void FluidSimulator::updateSurfaceField() {
        if (m_iNumCells <= 0 || m_iNumSpheres <= 0) return;

        buildHash();
        std::fill(m_surfaceColor.begin(), m_surfaceColor.end(), 0.0f);

        float const radius = (m_surfaceKernelRadius > 0.0f) ? m_surfaceKernelRadius : 2.0f * m_h;
        float const invR   = 1.0f / radius;
        int const   reach  = std::max(1, static_cast<int>(std::ceil(radius * m_fInvSpacing)));

        // 核函数(未归一化)
        auto kernel = [&](float r) {
            float const q = r * invR;
            if (q >= 1.0f) return 0.0f;
            float const s = 1.0f - q * q;
            return s * s * s;
        };

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int const        id      = index2GridOffset(glm::ivec3(i, j, k));
                    glm::vec3 const  cellPos = CellCenter(glm::ivec3(i, j, k));
                    glm::ivec3 const slot    = worldToCell(cellPos);

                    float raw = 0.0f;
                    for (int di = -reach; di <= reach; ++di) {
                        int const ni = clampSlot(slot.x + di, m_iCellX - 1);
                        for (int dj = -reach; dj <= reach; ++dj) {
                            int const nj = clampSlot(slot.y + dj, m_iCellY - 1);
                            for (int dk = -reach; dk <= reach; ++dk) {
                                int const nk         = clampSlot(slot.z + dk, m_iCellZ - 1);
                                int const neighborId = index2GridOffset(glm::ivec3(ni, nj, nk));
                                for (int ptr = m_hashtableindex[neighborId];
                                     ptr < m_hashtableindex[neighborId + 1];
                                     ++ptr) {
                                    int const   p = m_hashtable[ptr];
                                    float const r = glm::length(cellPos - m_particlePos[p]);
                                    raw += kernel(r);
                                }
                            }
                        }
                    }
                    m_surfaceColor[id] = raw;
                }
            }
        }

        // 首次使用计算m_surfaceRestField
        if (m_surfaceRestField <= 1e-6f) {
            float accum = 0.0f;
            int   count = 0;
            for (int cell = 0; cell < m_iNumCells; ++cell) {
                if (m_hashtableindex[cell + 1] == m_hashtableindex[cell]) continue;
                accum += m_surfaceColor[cell];
                ++count;
            }
            m_surfaceRestField = (count > 0) ? std::max(accum / float(count), 1e-6f) : 1.0f;
        }

        // 归一化表面场
        for (float & c : m_surfaceColor)
            c = glm::clamp(c / m_surfaceRestField, 0.0f, 1.5f);

        // 3*3*3模糊减小噪声影响
        std::vector<float> blurred(m_iNumCells, 0.0f);
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    float accum = 0.0f;
                    float wsum  = 0.0f;
                    for (int di = -1; di <= 1; ++di) {
                        int const   ni = clampCoord(i + di, 0, m_iCellX - 1);
                        float const wi = (di == 0) ? 2.0f : 1.0f;
                        for (int dj = -1; dj <= 1; ++dj) {
                            int const   nj = clampCoord(j + dj, 0, m_iCellY - 1);
                            float const wj = (dj == 0) ? 2.0f : 1.0f;
                            for (int dk = -1; dk <= 1; ++dk) {
                                int const   nk = clampCoord(k + dk, 0, m_iCellZ - 1);
                                float const wk = (dk == 0) ? 2.0f : 1.0f;
                                float const w  = wi * wj * wk;
                                accum += w * m_surfaceColor[index2GridOffset(glm::ivec3(ni, nj, nk))];
                                wsum += w;
                            }
                        }
                    }
                    blurred[index2GridOffset(glm::ivec3(i, j, k))] = accum / std::max(wsum, 1e-6f);
                }
            }
        }
        m_surfaceColor.swap(blurred);
    }

    void FluidSimulator::computeSurfaceGeometry() {
        if (m_surfaceColor.empty()) return;

        auto sampleColor = [&](int i, int j, int k) {
            return m_surfaceColor[index2GridOffset(glm::ivec3(clampCoord(i, 0, m_iCellX - 1), clampCoord(j, 0, m_iCellY - 1), clampCoord(k, 0, m_iCellZ - 1)))];
        };

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int const id = index2GridOffset(glm::ivec3(i, j, k));
                    glm::vec3 gradC(
                        (sampleColor(i + 1, j, k) - sampleColor(i - 1, j, k)) * 0.5f * m_fInvSpacing,
                        (sampleColor(i, j + 1, k) - sampleColor(i, j - 1, k)) * 0.5f * m_fInvSpacing,
                        (sampleColor(i, j, k + 1) - sampleColor(i, j, k - 1)) * 0.5f * m_fInvSpacing);

                    float const gradLen = glm::length(gradC);
                    float const phi     = (m_surfaceIsoValue - m_surfaceColor[id]) / (gradLen + 1e-4f); // phi = (iso - c)/|gradC|，一阶泰勒近似
                    m_surfacePhi[id]    = glm::clamp(phi, -3.0f * m_h, 3.0f * m_h);
                }
            }
        }

        auto samplePhi = [&](int i, int j, int k) {
            return m_surfacePhi[index2GridOffset(glm::ivec3(clampCoord(i, 0, m_iCellX - 1), clampCoord(j, 0, m_iCellY - 1), clampCoord(k, 0, m_iCellZ - 1)))];
        };

        // normal = gradPhi/|gradPhi|，gradPhi通过phi的中心差分计算得到
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::vec3 gradPhi(
                        (samplePhi(i + 1, j, k) - samplePhi(i - 1, j, k)) * 0.5f * m_fInvSpacing,
                        (samplePhi(i, j + 1, k) - samplePhi(i, j - 1, k)) * 0.5f * m_fInvSpacing,
                        (samplePhi(i, j, k + 1) - samplePhi(i, j, k - 1)) * 0.5f * m_fInvSpacing);

                    int const   id      = index2GridOffset(glm::ivec3(i, j, k));
                    float const len     = glm::length(gradPhi);
                    m_surfaceNormal[id] = (len > 1e-5f) ? gradPhi / len : glm::vec3(0.0f);
                }
            }
        }

        auto sampleNormal = [&](int i, int j, int k) {
            return m_surfaceNormal[index2GridOffset(glm::ivec3(clampCoord(i, 0, m_iCellX - 1), clampCoord(j, 0, m_iCellY - 1), clampCoord(k, 0, m_iCellZ - 1)))];
        };

        // div(normal)计算曲率，div(gradPhi/|gradPhi|)的离散形式
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::vec3 const nxp = sampleNormal(i + 1, j, k);
                    glm::vec3 const nxm = sampleNormal(i - 1, j, k);
                    glm::vec3 const nyp = sampleNormal(i, j + 1, k);
                    glm::vec3 const nym = sampleNormal(i, j - 1, k);
                    glm::vec3 const nzp = sampleNormal(i, j, k + 1);
                    glm::vec3 const nzm = sampleNormal(i, j, k - 1);

                    float curvature = ((nxp.x - nxm.x) + (nyp.y - nym.y) + (nzp.z - nzm.z))
                        * 0.5f * m_fInvSpacing;
                    int const id           = index2GridOffset(glm::ivec3(i, j, k));
                    m_surfaceCurvature[id] = glm::clamp(curvature, -m_surfaceCurvatureMax, m_surfaceCurvatureMax);
                }
            }
        }

        // 窄带内的双边滤波平滑曲率，减小噪声影响同时尽量保持边缘特征
        std::vector<float> filtered = m_surfaceCurvature;
        float const        band     = (m_surfaceBandWidth > 0.0f) ? m_surfaceBandWidth : 2.0f * m_h;
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int const id = index2GridOffset(glm::ivec3(i, j, k));
                    if (std::abs(m_surfacePhi[id]) > band) continue; // 只计算窄带内的点

                    float accum = 0.0f;
                    float wsum  = 0.0f;
                    for (int di = -1; di <= 1; ++di) {
                        int const ni = clampCoord(i + di, 0, m_iCellX - 1);
                        for (int dj = -1; dj <= 1; ++dj) {
                            int const nj = clampCoord(j + dj, 0, m_iCellY - 1);
                            for (int dk = -1; dk <= 1; ++dk) {
                                int const   nk   = clampCoord(k + dk, 0, m_iCellZ - 1);
                                int const   nid  = index2GridOffset(glm::ivec3(ni, nj, nk));
                                float const dPhi = m_surfacePhi[nid] - m_surfacePhi[id];
                                float const w    = std::exp(-(dPhi * dPhi) / (band * band + 1e-6f)); // w=e^(-ΔPhi^2 / band^2)
                                accum += w * m_surfaceCurvature[nid];
                                wsum += w;
                            }
                        }
                    }
                    filtered[id] = glm::clamp(accum / std::max(wsum, 1e-6f), -m_surfaceCurvatureMax, m_surfaceCurvatureMax);
                }
            }
        }
        m_surfaceCurvature.swap(filtered);
    }

    void FluidSimulator::applySurfaceTension(float dt) {
        if (m_surfaceTension <= 0.0f || m_iNumSpheres <= 0 || m_surfaceCurvature.empty()) return;

        // 在field中位置p采样标量 (如曲率)  的三线性插值函数
        auto sampleFloat = [&](std::vector<float> const & field, glm::vec3 const & p) {
            glm::vec3 gRaw = (p + glm::vec3(0.5f)) * m_fInvSpacing - glm::vec3(0.5f);
            glm::vec3 g(
                glm::clamp(gRaw.x, 0.0f, float(m_iCellX - 1)),
                glm::clamp(gRaw.y, 0.0f, float(m_iCellY - 1)),
                glm::clamp(gRaw.z, 0.0f, float(m_iCellZ - 1)));

            int const   ix0 = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 1);
            int const   iy0 = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 1);
            int const   iz0 = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 1);
            int const   ix1 = std::min(ix0 + 1, m_iCellX - 1);
            int const   iy1 = std::min(iy0 + 1, m_iCellY - 1);
            int const   iz1 = std::min(iz0 + 1, m_iCellZ - 1);
            float const fx  = glm::clamp(g.x - float(ix0), 0.0f, 1.0f);
            float const fy  = glm::clamp(g.y - float(iy0), 0.0f, 1.0f);
            float const fz  = glm::clamp(g.z - float(iz0), 0.0f, 1.0f);

            auto        at  = [&](int x, int y, int z) { return field[index2GridOffset(glm::ivec3(x, y, z))]; };
            float const c00 = glm::mix(at(ix0, iy0, iz0), at(ix1, iy0, iz0), fx);
            float const c10 = glm::mix(at(ix0, iy1, iz0), at(ix1, iy1, iz0), fx);
            float const c01 = glm::mix(at(ix0, iy0, iz1), at(ix1, iy0, iz1), fx);
            float const c11 = glm::mix(at(ix0, iy1, iz1), at(ix1, iy1, iz1), fx);
            return glm::mix(glm::mix(c00, c10, fy), glm::mix(c01, c11, fy), fz);
        };

        // 在field中位置p采样向量 (如法线)  的三线性插值函数
        auto sampleVec = [&](std::vector<glm::vec3> const & field, glm::vec3 const & p) {
            glm::vec3 gRaw = (p + glm::vec3(0.5f)) * m_fInvSpacing - glm::vec3(0.5f);
            glm::vec3 g(
                glm::clamp(gRaw.x, 0.0f, float(m_iCellX - 1)),
                glm::clamp(gRaw.y, 0.0f, float(m_iCellY - 1)),
                glm::clamp(gRaw.z, 0.0f, float(m_iCellZ - 1)));

            int const   ix0 = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 1);
            int const   iy0 = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 1);
            int const   iz0 = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 1);
            int const   ix1 = std::min(ix0 + 1, m_iCellX - 1);
            int const   iy1 = std::min(iy0 + 1, m_iCellY - 1);
            int const   iz1 = std::min(iz0 + 1, m_iCellZ - 1);
            float const fx  = glm::clamp(g.x - float(ix0), 0.0f, 1.0f);
            float const fy  = glm::clamp(g.y - float(iy0), 0.0f, 1.0f);
            float const fz  = glm::clamp(g.z - float(iz0), 0.0f, 1.0f);

            auto            at  = [&](int x, int y, int z) { return field[index2GridOffset(glm::ivec3(x, y, z))]; };
            glm::vec3 const c00 = glm::mix(at(ix0, iy0, iz0), at(ix1, iy0, iz0), fx);
            glm::vec3 const c10 = glm::mix(at(ix0, iy1, iz0), at(ix1, iy1, iz0), fx);
            glm::vec3 const c01 = glm::mix(at(ix0, iy0, iz1), at(ix1, iy0, iz1), fx);
            glm::vec3 const c11 = glm::mix(at(ix0, iy1, iz1), at(ix1, iy1, iz1), fx);
            return glm::mix(glm::mix(c00, c10, fy), glm::mix(c01, c11, fy), fz);
        };

        float const band = (m_surfaceBandWidth > 0.0f) ? m_surfaceBandWidth : 2.0f * m_h;
        for (int p = 0; p < m_iNumSpheres; ++p) {
            float const phi          = sampleFloat(m_surfacePhi, m_particlePos[p]);
            float const surfaceDelta = glm::clamp(1.0f - std::abs(phi) / band, 0.0f, 1.0f);
            if (surfaceDelta <= 0.0f) continue; // 只对接近界面(窄带内)的粒子施加表面张力

            glm::vec3 const n    = sampleVec(m_surfaceNormal, m_particlePos[p]);
            float const     nLen = glm::length(n);
            if (nLen <= 1e-5f) continue;

            float const     kappa = sampleFloat(m_surfaceCurvature, m_particlePos[p]);     // 曲率
            glm::vec3 const accel = -m_surfaceTension * kappa * (n / nLen) * surfaceDelta; // 张力体现为沿法向指向液体内部的加速度，曲率越大(界面越弯曲)张力越强；surfaceDelta随到表面距离线性变化，在窄带外为0
            m_particleVel[p] += accel * dt;
        }
    }

    void FluidSimulator::EnsureSimulationSDFFields() {
        if (m_iCellX <= 0 || m_iCellY <= 0 || m_iCellZ <= 0 || m_h <= 0.0f) return;

        glm::vec3 const origin(-0.5f);
        int const cellCount = m_iCellX * m_iCellY * m_iCellZ;

        bool const fluidNeedsResize = ! m_phiFluidSim.IsValid()
            || m_phiFluidSim.nx != m_iCellX
            || m_phiFluidSim.ny != m_iCellY
            || m_phiFluidSim.nz != m_iCellZ
            || std::abs(m_phiFluidSim.dx - m_h) > 1e-7f
            || m_phiFluidSim.phi.size() != std::size_t(cellCount);
        if (fluidNeedsResize) {
            m_phiFluidSim.Resize(m_iCellX, m_iCellY, m_iCellZ, m_h, origin, 4.0f * m_h);
        }

        bool const solidNeedsResize = ! m_phiSolid.IsValid()
            || m_phiSolid.nx != m_iCellX
            || m_phiSolid.ny != m_iCellY
            || m_phiSolid.nz != m_iCellZ
            || std::abs(m_phiSolid.dx - m_h) > 1e-7f
            || m_phiSolid.phi.size() != std::size_t(cellCount);
        if (solidNeedsResize) {
            m_phiSolid.Resize(m_iCellX, m_iCellY, m_iCellZ, m_h, origin, 4.0f * m_h);
        }
    }

    void FluidSimulator::ClearSimulationSDFFields() {
        m_phiFluidSim.Clear();
        m_phiSolid.Clear();
        m_simSdfParticleRadius = 0.0f;
        m_simSdfNarrowBand     = 0.0f;
        m_solidSdfMaxDistance  = 0.0f;
        ClearSimulationFractionFields();
    }

    void FluidSimulator::BuildFluidSimulationSDF() {
        EnsureSimulationSDFFields();
        if (! m_phiFluidSim.IsValid()) return;

        float const radius = (m_simSdfParticleRadius > 0.0f)
            ? m_simSdfParticleRadius
            : m_particleRadius;
        BuildFluidSDFFromParticles(m_particlePos, m_phiFluidSim, radius, m_simSdfNarrowBand);
    }

    void FluidSimulator::BuildSolidSimulationSDF(RigidBodySystem const & rigid) {
        EnsureSimulationSDFFields();
        if (! m_phiSolid.IsValid()) return;

        // 水槽静态墙已经由 m_s / IsCellSolid / SolidBoundaryVelocity 这一套
        // 规则网格边界条件稳定处理；再把它们重复塞进 SDF cut-cell 会让
        // 贴墙第一层和角落处同时吃到两套几何离散，容易出现“贴墙先掉”。
        // 这里把仿真 SDF 只留给真正的刚体几何（船体/箱体/球等）。
        BuildSolidSDFFromRigidBodies(rigid, m_phiSolid, m_solidSdfMaxDistance, false);
    }

    void FluidSimulator::BuildSimulationSDFFields(RigidBodySystem const & rigid) {
        EnsureSimulationSDFFields();
        BuildFluidSimulationSDF();
        BuildSolidSimulationSDF(rigid);
        ComputeSimulationFractions();
        PromoteBoundaryLiquidCellsToFluid();
    }

    void FluidSimulator::EnsureSimulationFractionFields() {
        if (m_iNumCells <= 0) return;

        if (m_liquidCellFraction.size() != std::size_t(m_iNumCells)) {
            m_liquidCellFraction.assign(m_iNumCells, 0.0f);
        }
        if (m_solidCellFraction.size() != std::size_t(m_iNumCells)) {
            m_solidCellFraction.assign(m_iNumCells, 0.0f);
        }
        if (m_faceOpenFraction.size() != std::size_t(m_iNumCells)) {
            m_faceOpenFraction.assign(m_iNumCells, glm::vec3(0.0f));
        }
    }

    void FluidSimulator::ClearSimulationFractionFields() {
        m_liquidCellFraction.clear();
        m_solidCellFraction.clear();
        m_faceOpenFraction.clear();
    }

    void FluidSimulator::ComputeLiquidCellFractions() {
        EnsureSimulationFractionFields();
        if (! m_phiFluidSim.IsValid() || m_liquidCellFraction.empty()) return;

        // 当前网格的 cell idx 对应 [idx, idx+1] 这一个 h^3 体积。
        // 第一版使用 8 个角点判断 phi<0 的比例；后续可替换为更密的 supersampling。
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int inside = 0;
                    for (int sx = 0; sx <= 1; ++sx) {
                        for (int sy = 0; sy <= 1; ++sy) {
                            for (int sz = 0; sz <= 1; ++sz) {
                                glm::vec3 const p = (glm::vec3(i + sx, j + sy, k + sz) * m_h) - glm::vec3(0.5f);
                                if (SampleSDF(m_phiFluidSim, p) < 0.0f) ++inside;
                            }
                        }
                    }
                    m_liquidCellFraction[index2GridOffset(glm::ivec3(i, j, k))] = float(inside) / 8.0f;
                }
            }
        }
    }

    void FluidSimulator::ComputeSolidCellFractions() {
        EnsureSimulationFractionFields();
        if (! m_phiSolid.IsValid() || m_solidCellFraction.empty()) return;

        // solid fraction 与 liquid fraction 使用同一套 8-corner 近似，便于后续比较和调试。
        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int inside = 0;
                    for (int sx = 0; sx <= 1; ++sx) {
                        for (int sy = 0; sy <= 1; ++sy) {
                            for (int sz = 0; sz <= 1; ++sz) {
                                glm::vec3 const p = (glm::vec3(i + sx, j + sy, k + sz) * m_h) - glm::vec3(0.5f);
                                if (SampleSDF(m_phiSolid, p) < 0.0f) ++inside;
                            }
                        }
                    }
                    m_solidCellFraction[index2GridOffset(glm::ivec3(i, j, k))] = float(inside) / 8.0f;
                }
            }
        }
    }

    void FluidSimulator::ComputeFaceOpenFractions() {
        EnsureSimulationFractionFields();
        if (! m_phiSolid.IsValid() || m_faceOpenFraction.empty()) return;

        std::fill(m_faceOpenFraction.begin(), m_faceOpenFraction.end(), glm::vec3(0.0f));

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::ivec3 const face(i, j, k);
                    int const faceId = index2GridOffset(face);

                    for (int dir = 0; dir < 3; ++dir) {
                        if (! isVelocityFaceInRange(i, j, k, dir)) continue;

                        int const uAxis = (dir + 1) % 3;
                        int const vAxis = (dir + 2) % 3;
                        int open = 0;

                        // 在 face 平面上做 2x2 子采样，采样点位于四个象限中心，避开边缘的符号抖动。
                        for (int su = 0; su < 2; ++su) {
                            for (int sv = 0; sv < 2; ++sv) {
                                glm::vec3 p = FaceCenter(face, dir);
                                p[uAxis] += (float(su) - 0.5f) * 0.5f * m_h;
                                p[vAxis] += (float(sv) - 0.5f) * 0.5f * m_h;
                                if (SampleSDF(m_phiSolid, p) > 0.0f) ++open;
                            }
                        }

                        m_faceOpenFraction[faceId][dir] = float(open) / 4.0f;
                    }
                }
            }
        }
    }

    void FluidSimulator::ComputeSimulationFractions() {
        EnsureSimulationFractionFields();
        ComputeLiquidCellFractions();
        ComputeSolidCellFractions();
        ComputeFaceOpenFractions();
    }

    void FluidSimulator::PromoteBoundaryLiquidCellsToFluid() {
        if (m_type.size() != std::size_t(m_iNumCells)) return;

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::ivec3 const cell(i, j, k);
                    int const        id = index2GridOffset(cell);
                    if (m_type[id] == SOLID_CELL || IsCellSolid(cell)) continue;
                    if (m_type[id] == FLUID_CELL) continue;
                    if (IsBoundaryLiquidCell(cell)) {
                        m_type[id] = FLUID_CELL;
                    }
                }
            }
        }
    }

    float FluidSimulator::FaceOpenFraction(glm::ivec3 face, int dir) const {
        if (dir < 0 || dir >= 3 || ! IsVelocityFaceInRange(face, dir) || m_faceOpenFraction.empty()) {
            return 0.0f;
        }

        int const id = index2GridOffset(face);
        if (id < 0 || id >= static_cast<int>(m_faceOpenFraction.size())) return 0.0f;
        return glm::clamp(m_faceOpenFraction[id][dir], 0.0f, 1.0f);
    }

    void FluidSimulator::EnsureSurfaceFields() {
        if (m_iNumCells <= 0) return;

        bool const hasValidSize = m_surfaceColor.size() == static_cast<std::size_t>(m_iNumCells)
            && m_surfacePhi.size() == static_cast<std::size_t>(m_iNumCells)
            && m_surfaceNormal.size() == static_cast<std::size_t>(m_iNumCells)
            && m_surfaceCurvature.size() == static_cast<std::size_t>(m_iNumCells);

        if (! hasValidSize) {
            m_surfaceColor.assign(m_iNumCells, 0.0f);
            m_surfacePhi.assign(m_iNumCells, 0.0f);
            m_surfaceNormal.assign(m_iNumCells, glm::vec3(0.0f));
            m_surfaceCurvature.assign(m_iNumCells, 0.0f);
            m_surfaceRestField = 0.0f;
        }

        if (m_surfaceKernelRadius <= 0.0f) m_surfaceKernelRadius = 2.0f * m_h;
        if (m_surfaceBandWidth <= 0.0f) m_surfaceBandWidth = 2.0f * m_h;
        if (m_surfaceCurvatureMax <= 0.0f) m_surfaceCurvatureMax = 1.0f / m_h;
    }

    void FluidSimulator::ClearSurfaceFields() {
        m_surfaceColor.clear();
        m_surfacePhi.clear();
        m_surfaceNormal.clear();
        m_surfaceCurvature.clear();
        m_surfaceKernelRadius = 0.0f;
        m_surfaceBandWidth    = 0.0f;
        m_surfaceCurvatureMax = 0.0f;
        m_surfaceRestField    = 0.0f;
    }

    void FluidSimulator::EnsureRenderableSurfaceFields() {
        if (m_iCellX <= 1 || m_iCellY <= 1 || m_iCellZ <= 1 || m_h <= 0.0f) return;
        float const renderScale = std::max(1.0f, m_renderSurfaceResolutionScale);
        int const   cellX       = std::max(2, int(std::round(float(m_iCellX - 1) * renderScale)) + 1);
        int const   cellY       = std::max(2, int(std::round(float(m_iCellY - 1) * renderScale)) + 1);
        int const   cellZ       = std::max(2, int(std::round(float(m_iCellZ - 1) * renderScale)) + 1);
        float const renderH     = 1.0f / float(cellX - 1);
        int const   cellCount   = cellX * cellY * cellZ;

        bool const needsResize = m_renderSurfaceCellX != cellX
            || m_renderSurfaceCellY != cellY
            || m_renderSurfaceCellZ != cellZ
            || m_renderSurfaceColor.size() != std::size_t(cellCount)
            || m_renderSurfacePhi.size() != std::size_t(cellCount);

        m_renderSurfaceCellX = cellX;
        m_renderSurfaceCellY = cellY;
        m_renderSurfaceCellZ = cellZ;
        m_renderSurfaceH     = renderH;
        m_renderSurfaceInvH  = 1.0f / m_renderSurfaceH;
        if (m_renderSurfaceKernelRadius <= 0.0f)
            m_renderSurfaceKernelRadius = 1.8f * m_h;

        if (needsResize) {
            m_renderSurfaceColor.assign(cellCount, 0.0f);
            m_renderSurfacePhi.assign(cellCount, 0.0f);
            m_renderSurfaceNormal.assign(cellCount, glm::vec3(0.0f, 1.0f, 0.0f));
            m_renderSurfaceMesh.positions.clear();
            m_renderSurfaceMesh.normals.clear();
            m_renderSurfaceMesh.indices.clear();
            m_renderSurfaceFrameCounter = 0;
        }
    }

    void FluidSimulator::SetSurfaceModelingEnabled(bool enabled) {
        if (enableSurfaceModeling == enabled) return;

        enableSurfaceModeling = enabled;
        if (enableSurfaceModeling)
            EnsureSurfaceFields();
        else
            ClearSurfaceFields();
    }

    void FluidSimulator::updateParticleColors() {
        if (m_iNumSpheres <= 0) return;

        // 颜色渐变: 低速/低压(蓝) → 中速/中压(青) → 高速/高压(红)
        auto const ramp = [](float t) {
            t = glm::clamp(t, 0.0f, 1.0f);
            glm::vec3 const cLow(0.10f, 0.25f, 0.95f); // 蓝色 (低)
            glm::vec3 const cMid(0.10f, 0.90f, 0.85f); // 青色 (中)
            glm::vec3 const cHi(1.00f, 0.30f, 0.05f);  // 红色 (高)
            if (t < 0.5f)
                return glm::mix(cLow, cMid, t * 2.0f);
            return glm::mix(cMid, cHi, (t - 0.5f) * 2.0f);
        };

        // 使用三线性插值从网格压力场中采样每个粒子所在位置的压力
        // 用于视觉反馈 (检查压力求解是否正常工作)
        float maxPressure = 1e-6f;
        for (int id = 0; id < m_iNumCells; ++id)
            maxPressure = std::max(maxPressure, std::abs(m_p[id]));

        for (int i = 0; i < m_iNumSpheres; ++i) {
            // 仅从有效流体单元采样压力，避免把 solid/empty 的 0 压力混入边界粒子着色。
            glm::vec3 g  = (m_particlePos[i] + glm::vec3(0.5f)) * m_fInvSpacing;
            int       ix = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 2);
            int       iy = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 2);
            int       iz = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 2);
            float     fx = g.x - ix;
            float     fy = g.y - iy;
            float     fz = g.z - iz;

            int ix0 = ix, ix1 = std::min(ix + 1, m_iCellX - 1);
            int iy0 = iy, iy1 = std::min(iy + 1, m_iCellY - 1);
            int iz0 = iz, iz1 = std::min(iz + 1, m_iCellZ - 1);

            auto accumulatePressure = [&](int x, int y, int z, float w, float & sum, float & wsum) {
                int const id = index2GridOffset(glm::ivec3(x, y, z));
                if (m_type[id] != FLUID_CELL || w <= 0.0f) return;
                sum += w * m_p[id];
                wsum += w;
            };

            float pressureSum    = 0.0f;
            float pressureWeight = 0.0f;

            for (int dx = 0; dx <= 1; ++dx) {
                float wx = dx ? fx : (1.0f - fx);
                int   sx = dx ? ix1 : ix0;
                for (int dy = 0; dy <= 1; ++dy) {
                    float wy = dy ? fy : (1.0f - fy);
                    int   sy = dy ? iy1 : iy0;
                    for (int dz = 0; dz <= 1; ++dz) {
                        float wz = dz ? fz : (1.0f - fz);
                        int   sz = dz ? iz1 : iz0;
                        accumulatePressure(sx, sy, sz, wx * wy * wz, pressureSum, pressureWeight);
                    }
                }
            }

            float p = 0.0f;
            if (pressureWeight > 1e-6f) {
                p = pressureSum / pressureWeight;
            } else {
                int const centerId = index2GridOffset(worldToCell(m_particlePos[i]));
                if (m_type[centerId] == FLUID_CELL) {
                    p = m_p[centerId];
                } else {
                    float nearestDist2 = std::numeric_limits<float>::max();
                    for (int dx = 0; dx <= 1; ++dx) {
                        int sx = dx ? ix1 : ix0;
                        for (int dy = 0; dy <= 1; ++dy) {
                            int sy = dy ? iy1 : iy0;
                            for (int dz = 0; dz <= 1; ++dz) {
                                int const sz = dz ? iz1 : iz0;
                                int const id = index2GridOffset(glm::ivec3(sx, sy, sz));
                                if (m_type[id] != FLUID_CELL) continue;

                                glm::vec3 const cellPos = glm::vec3(sx, sy, sz) * m_h + glm::vec3(-0.5f);
                                float const     dist2   = glm::dot(cellPos - m_particlePos[i], cellPos - m_particlePos[i]);
                                if (dist2 < nearestDist2) {
                                    nearestDist2 = dist2;
                                    p            = m_p[id];
                                }
                            }
                        }
                    }
                }
            }

            float t            = std::abs(p) / maxPressure;
            m_particleColor[i] = ramp(t);
        }
    }

    void FluidSimulator::setupScene(int res) {
        glm::vec3 tank(1.0f);
        glm::vec3 relWater = {0.7f, 0.8f, 0.7f};

        float _h      = tank.y / res;
        float point_r = 0.3f * _h;
        float dx      = 2.0f * point_r;
        float dy      = sqrt(3.0f) / 2.0f * dx;
        float dz      = dx;

        int numX = floor((relWater.x * tank.x - 2.0f * _h - 2.0f * point_r) / dx);
        int numY = floor((relWater.y * tank.y - 2.0f * _h - 2.0f * point_r) / dy);
        int numZ = floor((relWater.z * tank.z - 2.0f * _h - 2.0f * point_r) / dz);

        // 更新网格几何参数
        m_iNumSpheres         = numX * numY * numZ;
        m_iCellX              = res + 1;
        m_iCellY              = res + 1;
        m_iCellZ              = res + 1;
        m_h                   = 1.0f / float(res);
        m_fInvSpacing         = float(res);
        m_iNumCells           = m_iCellX * m_iCellY * m_iCellZ;
        m_particleRadius      = point_r;
        m_particleRestDensity = 1.0f;

        // 分配粒子数组
        m_particlePos.clear();
        m_particlePos.resize(m_iNumSpheres, glm::vec3(0.0f));
        m_particleVel.clear();
        m_particleVel.resize(m_iNumSpheres, glm::vec3(0.0f));
        m_particleColor.clear();
        m_particleColor.resize(m_iNumSpheres, glm::vec3(1.0f));
        m_hashtable.clear();
        m_hashtable.resize(m_iNumSpheres, 0);
        m_hashtableindex.clear();
        m_hashtableindex.resize(m_iNumCells + 1, 0);
        m_particleSlot.clear();
        m_particleSlot.resize(m_iNumSpheres, glm::ivec3(0));

        // 分配网格数组
        m_vel.clear();
        m_vel.resize(m_iNumCells, glm::vec3(0.0f));
        m_pre_vel.clear();
        m_pre_vel.resize(m_iNumCells, glm::vec3(0.0f));
        m_solidVel.clear();
        m_solidVel.resize(m_iNumCells, glm::vec3(0.0f));
        m_solidVelMask.clear();
        m_solidVelMask.resize(m_iNumCells, glm::ivec3(0));
        for (int i = 0; i < 3; ++i) {
            m_near_num[i].clear();
            m_near_num[i].resize(m_iNumCells, 0.0f);
        }

        m_p.clear();
        m_p.resize(m_iNumCells, 0.0);
        m_s.clear();
        m_s.resize(m_iNumCells, 0.0);
        m_type.clear();
        m_type.resize(m_iNumCells, 0);
        m_particleDensity.clear();
        m_particleDensity.resize(m_iNumCells, 0.0f);

        ClearSimulationSDFFields();
        EnsureSimulationSDFFields();

        // 建模流体表面
        if (enableSurfaceModeling) {
            ClearSurfaceFields();
            EnsureSurfaceFields();
        } else {
            ClearSurfaceFields();
        }

        // 生成HCP排列的粒子
        int p = 0;
        for (int i = 0; i < numX; i++) {
            for (int j = 0; j < numY; j++) {
                for (int k = 0; k < numZ; k++) {
                    m_particlePos[p++] = glm::vec3(
                                             m_h + point_r + dx * i + (j % 2 == 0 ? 0.0f : point_r),
                                             m_h + point_r + dy * j,
                                             m_h + point_r + dz * k + (j % 2 == 0 ? 0.0f : point_r))
                        + glm::vec3(-0.5f);
                }
            }
        }

        // 设置网格固体标记: 水槽壁厚为1个格点
        for (int i = 0; i < m_iCellX; i++) {
            for (int j = 0; j < m_iCellY; j++) {
                for (int k = 0; k < m_iCellZ; k++) {
                    float s = 1.0f; // 默认非固体
                    if (i == 0 || i == m_iCellX - 2
                        || j == 0 || j == m_iCellY - 2
                        || k == 0 || k == m_iCellZ - 2)
                        s = 0.0f; // 边界固体
                    m_s[index2GridOffset(glm::ivec3(i, j, k))] = s;
                }
            }
        }

        // 构建初始哈希表并更新单元格类型
        buildHash();
        for (int cell = 0; cell < m_iNumCells; ++cell) {
            if (m_s[cell] <= 0.0f) {
                m_type[cell] = SOLID_CELL;
            } else {
                int const cnt = m_hashtableindex[cell + 1] - m_hashtableindex[cell];
                m_type[cell]  = (cnt > 0) ? FLUID_CELL : EMPTY_CELL;
            }
        }
    }

    void FluidSimulator::FinalizeManualParticleSetup(std::vector<glm::vec3> positions, glm::vec3 initialVelocity) {
        m_iNumSpheres = static_cast<int>(positions.size());

        m_particlePos = std::move(positions);
        m_particleVel.assign(m_iNumSpheres, initialVelocity);
        m_particleColor.assign(m_iNumSpheres, glm::vec3(1.0f));
        m_hashtable.assign(m_iNumSpheres, 0);
        m_particleSlot.assign(m_iNumSpheres, glm::ivec3(0));
        m_hashtableindex.assign(m_iNumCells + 1, 0);

        std::fill(m_vel.begin(), m_vel.end(), glm::vec3(0.0f));
        std::fill(m_pre_vel.begin(), m_pre_vel.end(), glm::vec3(0.0f));
        ResetSolidBoundaryVelocity();
        for (int dir = 0; dir < 3; ++dir) {
            std::fill(m_near_num[dir].begin(), m_near_num[dir].end(), 0.0f);
        }
        std::fill(m_p.begin(), m_p.end(), 0.0f);
        std::fill(m_particleDensity.begin(), m_particleDensity.end(), 0.0f);

        ResetSolidMaskToTank();
        m_particleRestDensity = 1.0f;

        if (enableSurfaceModeling) {
            ClearSurfaceFields();
            EnsureSurfaceFields();
        }
        m_renderSurfaceFrameCounter = 0;

        buildHash();
        for (int cell = 0; cell < m_iNumCells; ++cell) {
            if (m_s[cell] <= 0.0f) {
                m_type[cell] = SOLID_CELL;
            } else {
                int const cnt = m_hashtableindex[cell + 1] - m_hashtableindex[cell];
                m_type[cell]  = (cnt > 0) ? FLUID_CELL : EMPTY_CELL;
            }
        }
    }

    void FluidSimulator::ResetParticlesToBoxRegion(glm::vec3 minCorner, glm::vec3 maxCorner, glm::vec3 initialVelocity) {
        minCorner = glm::min(minCorner, maxCorner);
        maxCorner = glm::max(minCorner, maxCorner);

        float const point_r = m_particleRadius;
        float const dx      = 2.0f * point_r;
        float const dy      = std::sqrt(3.0f) * 0.5f * dx;
        float const dz      = dx;

        float const startX = -0.5f + m_h + point_r;
        float const startY = -0.5f + m_h + point_r;
        float const startZ = -0.5f + m_h + point_r;
        float const endX   =  0.5f - m_h - point_r;
        float const endY   =  0.5f - m_h - point_r;
        float const endZ   =  0.5f - m_h - point_r;

        std::vector<glm::vec3> positions;
        for (int j = 0;; ++j) {
            float const y = startY + dy * float(j);
            if (y > endY + 1e-6f) break;

            float const offset = (j % 2 == 0) ? 0.0f : point_r;
            for (int i = 0;; ++i) {
                float const x = startX + dx * float(i) + offset;
                if (x > endX + 1e-6f) break;
                for (int k = 0;; ++k) {
                    float const z = startZ + dz * float(k) + offset;
                    if (z > endZ + 1e-6f) break;

                    glm::vec3 const p(x, y, z);
                    if (p.x < minCorner.x || p.x > maxCorner.x) continue;
                    if (p.y < minCorner.y || p.y > maxCorner.y) continue;
                    if (p.z < minCorner.z || p.z > maxCorner.z) continue;
                    positions.push_back(p);
                }
            }
        }

        FinalizeManualParticleSetup(std::move(positions), initialVelocity);
    }

    void FluidSimulator::ResetParticlesToSphereRegion(glm::vec3 center, float radius, glm::vec3 initialVelocity) {
        float const point_r = m_particleRadius;
        float const dx      = 2.0f * point_r;
        float const dy      = std::sqrt(3.0f) * 0.5f * dx;
        float const dz      = dx;

        float const startX = std::max(-0.5f + m_h + point_r, center.x - radius);
        float const startY = std::max(-0.5f + m_h + point_r, center.y - radius);
        float const startZ = std::max(-0.5f + m_h + point_r, center.z - radius);
        float const endX   = std::min( 0.5f - m_h - point_r, center.x + radius);
        float const endY   = std::min( 0.5f - m_h - point_r, center.y + radius);
        float const endZ   = std::min( 0.5f - m_h - point_r, center.z + radius);

        std::vector<glm::vec3> positions;
        for (int j = 0;; ++j) {
            float const y = startY + dy * float(j);
            if (y > endY + 1e-6f) break;

            float const offset = (j % 2 == 0) ? 0.0f : point_r;
            for (int i = 0;; ++i) {
                float const x = startX + dx * float(i) + offset;
                if (x > endX + 1e-6f) break;
                for (int k = 0;; ++k) {
                    float const z = startZ + dz * float(k) + offset;
                    if (z > endZ + 1e-6f) break;

                    glm::vec3 const p(x, y, z);
                    glm::vec3 const d = p - center;
                    if (glm::dot(d, d) <= radius * radius) {
                        positions.push_back(p);
                    }
                }
            }
        }

        FinalizeManualParticleSetup(std::move(positions), initialVelocity);
    }

    int FluidSimulator::GridIndex(glm::ivec3 idx) const {
        return index2GridOffset(idx);
    }

    bool FluidSimulator::IsInsideGrid(glm::ivec3 idx) const {
        return (idx.x >= 0 && idx.x < m_iCellX) && (idx.y >= 0 && idx.y < m_iCellY) && (idx.z >= 0 && idx.z < m_iCellZ);
    }

    glm::ivec3 FluidSimulator::worldToCell(glm::vec3 const & p) const {
        glm::vec3 g = (p + glm::vec3(0.5f)) * m_fInvSpacing;
        return glm::ivec3(
            clampSlot(static_cast<int>(std::floor(g.x)), m_iCellX - 1),
            clampSlot(static_cast<int>(std::floor(g.y)), m_iCellY - 1),
            clampSlot(static_cast<int>(std::floor(g.z)), m_iCellZ - 1));
    }

    glm::vec3 FluidSimulator::CellCenter(glm::ivec3 idx) const {
        return (glm::vec3(idx) + glm::vec3(0.5f)) * m_h - glm::vec3(0.5f);
    }

    void FluidSimulator::ResetSolidMaskToTank() {
        for (int i = 0; i < m_iCellX; i++) {
            for (int j = 0; j < m_iCellY; j++) {
                for (int k = 0; k < m_iCellZ; k++) {
                    float s = 1.0f;
                    if (i == 0 || i == m_iCellX - 2
                        || j == 0 || j == m_iCellY - 2
                        || k == 0 || k == m_iCellZ - 2)
                        s = 0.0f;
                    m_s[index2GridOffset(glm::ivec3(i, j, k))] = s;
                }
            }
        }
    }

    void FluidSimulator::SetCellSolid(glm::ivec3 idx, bool solid) {
        if (IsInsideGrid(idx)) {
            m_s[index2GridOffset(idx)] = solid ? 0.0f : 1.0f;
        }
    }

    void FluidSimulator::ResetSolidBoundaryVelocity() {
        std::fill(m_solidVel.begin(), m_solidVel.end(), glm::vec3(0.0f));
        std::fill(m_solidVelMask.begin(), m_solidVelMask.end(), glm::ivec3(0));
    }

    void FluidSimulator::SetSolidBoundaryVelocity(glm::ivec3 idx, int dir, float velocity, bool updatePreVel) {
        if (! isVelocityFaceInRange(idx.x, idx.y, idx.z, dir)) return;

        int const id            = index2GridOffset(idx);
        m_solidVel[id][dir]     = velocity;
        m_solidVelMask[id][dir] = 1;
        m_vel[id][dir]          = velocity;
        if (updatePreVel) {
            m_pre_vel[id][dir] = velocity;
        }
    }

    bool FluidSimulator::HasSolidBoundaryVelocity(int i, int j, int k, int dir) const {
        if (! isVelocityFaceInRange(i, j, k, dir)) return false;
        if (m_solidVelMask.empty()) return false;
        return m_solidVelMask[index2GridOffset(glm::ivec3(i, j, k))][dir] != 0;
    }

    float FluidSimulator::SolidBoundaryVelocity(int i, int j, int k, int dir) const {
        if (! HasSolidBoundaryVelocity(i, j, k, dir)) return 0.0f;
        return m_solidVel[index2GridOffset(glm::ivec3(i, j, k))][dir];
    }

    bool FluidSimulator::IsVelocityFaceSolidBoundary(glm::ivec3 face, int dir) const {
        if (! isVelocityFaceInRange(face.x, face.y, face.z, dir)) return false;

        auto const [lower, upper] = FaceNeighborCells(face, dir);
        bool const lowerSolid = ! IsInsideGrid(lower) || IsCellSolid(lower);
        bool const upperSolid = ! IsInsideGrid(upper) || IsCellSolid(upper);
        bool const lowerFluidDomain = IsInsideGrid(lower) && ! IsCellSolid(lower);
        bool const upperFluidDomain = IsInsideGrid(upper) && ! IsCellSolid(upper);

        // 只处理“固体-非固体”的界面；固体内部的 face 不参与流体采样。
        return (lowerSolid || upperSolid) && (lowerFluidDomain || upperFluidDomain);
    }

    void FluidSimulator::EnforceSolidBoundaryVelocities() {
        if (m_vel.empty()) return;

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::ivec3 const face(i, j, k);
                    int const        id = index2GridOffset(face);
                    for (int dir = 0; dir < 3; ++dir) {
                        if (! IsVelocityFaceSolidBoundary(face, dir)) continue;

                        // 没有刚体写入速度时，玻璃/水槽边界就是静止 no-penetration。
                        // 有运动刚体时，SolidBoundaryVelocity 返回 rigid.VelocityAtPoint 的法向分量。
                        float const v = SolidBoundaryVelocity(i, j, k, dir);
                        m_vel[id][dir] = v;
                        if (id < static_cast<int>(m_pre_vel.size())) {
                            m_pre_vel[id][dir] = v;
                        }
                    }
                }
            }
        }
    }

    bool FluidSimulator::IsVelocityFaceInRange(glm::ivec3 face, int dir) const {
        // 复用原来的 MAC face 有效性判断,为Coupler提供一个接口
        return isVelocityFaceInRange(face.x, face.y, face.z, dir);
    }

    std::pair<glm::ivec3, glm::ivec3> FluidSimulator::FaceNeighborCells(glm::ivec3 face, int dir) const {
        // 速度面 face 位于 lower 和 upper 两个压力 cell 中间。
        glm::ivec3 lower = face;
        lower[dir] -= 1;
        return { lower, face };
    }

    glm::vec3 FluidSimulator::FaceCenter(glm::ivec3 face, int dir) const {
        // MAC 网格中，dir 方向速度存在该方向的 cell 边界上，
        // 另外两个方向则位于半格偏移处。
        glm::vec3 pos(float(face.x), float(face.y), float(face.z));
        if (dir != 0) pos.x += 0.5f;
        if (dir != 1) pos.y += 0.5f;
        if (dir != 2) pos.z += 0.5f;
        return pos * m_h + glm::vec3(-0.5f);
    }

    glm::vec3 FluidSimulator::FaceAxis(int dir) const {
        glm::vec3 axis(0.0f);
        if (dir >= 0 && dir < 3) axis[dir] = 1.0f;
        return axis;
    }

    float FluidSimulator::FaceVelocity(std::vector<glm::vec3> const & velocities, glm::ivec3 face, int dir) const {
        if (dir < 0 || dir >= 3 || ! IsVelocityFaceInRange(face, dir)) return 0.0f;
        int const id = index2GridOffset(face);
        if (id < 0 || id >= static_cast<int>(velocities.size())) return 0.0f;
        return velocities[id][dir];
    }

    bool FluidSimulator::IsNearSolidBoundary(glm::ivec3 idx) const {
        if (! IsInsideGrid(idx) || IsCellSolid(idx)) return false;

        if (m_solidCellFraction.size() == std::size_t(m_iNumCells)) {
            int const id = index2GridOffset(idx);
            if (id >= 0 && id < static_cast<int>(m_solidCellFraction.size())) {
                float const solidFraction = m_solidCellFraction[id];
                if (solidFraction > 1e-4f && solidFraction < 1.0f - 1e-4f) {
                    return true;
                }
            }
        }

        // 对斜切/局部 SDF 边界，partial open face 仍然视作 cut-cell。
        if (m_faceOpenFraction.size() == std::size_t(m_iNumCells)) {
            for (int dir = 0; dir < 3; ++dir) {
                glm::ivec3 axis(0);
                axis[dir]                  = 1;
                glm::ivec3 const lowerFace = idx;
                glm::ivec3 const upperFace = idx + axis;

                if (IsVelocityFaceInRange(lowerFace, dir)) {
                    float const open = FaceOpenFraction(lowerFace, dir);
                    if (open > 1e-4f && open < 1.0f - 1e-4f) return true;
                }
                if (IsVelocityFaceInRange(upperFace, dir)) {
                    float const open = FaceOpenFraction(upperFace, dir);
                    if (open > 1e-4f && open < 1.0f - 1e-4f) return true;
                }
            }
        }

        return false;
    }

    bool FluidSimulator::IsBoundaryLiquidCell(glm::ivec3 idx) const {
        if (! IsInsideGrid(idx) || IsCellSolid(idx)) return false;
        if (m_liquidCellFraction.size() != std::size_t(m_iNumCells)) return false;

        int const id = index2GridOffset(idx);
        if (id < 0 || id >= static_cast<int>(m_liquidCellFraction.size())) return false;

        float const liquidFraction = m_liquidCellFraction[id];
        if (liquidFraction < 0.125f) return false;
        return IsNearSolidBoundary(idx);
    }

    bool FluidSimulator::IsFaceCutBySolidNormal(glm::ivec3 face, int dir) const {
        if (dir < 0 || dir >= 3 || ! IsVelocityFaceInRange(face, dir) || ! m_phiSolid.IsValid()) {
            return false;
        }

        glm::vec3 const center = FaceCenter(face, dir);
        glm::vec3 const axis   = FaceAxis(dir);
        float const     offset = 0.5f * m_h;
        float const     phi0   = SampleSDF(m_phiSolid, center - offset * axis);
        float const     phi1   = SampleSDF(m_phiSolid, center + offset * axis);

        if (phi0 <= 0.0f && phi1 >= 0.0f) return true;
        if (phi1 <= 0.0f && phi0 >= 0.0f) return true;

        glm::vec3 const grad = GradSDF(m_phiSolid, center);
        float const     len  = glm::length(grad);
        if (len <= 1e-5f) return false;

        float const normalAlignment = std::abs(glm::dot(grad / len, axis));
        float const nearBand        = 0.25f * m_h;
        bool const  nearSurface     = std::min(std::abs(phi0), std::abs(phi1)) <= nearBand;
        return nearSurface && normalAlignment >= 0.5f;
    }

    FluidPressureDofs FluidSimulator::BuildPressureDofs() const {
        // Ax=b 的未知量个数不是总 cell 数。
        // 旧逻辑只给“粒子落入的 FLUID_CELL”建 dof；接入 SDF 后，液体体积分数>0的 cut-cell 也可成为压力未知量。
        // 近似完全固体的 cell 仍然排除，避免在刚体内部建立压力自由度。
        FluidPressureDofs result;
        result.pressureDof.assign(m_iNumCells, -1);

        if (m_type.empty()) return result;

        bool const hasSolidFraction  = m_solidCellFraction.size() == std::size_t(m_iNumCells);

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    glm::ivec3 const cell(i, j, k);
                    int const        id = index2GridOffset(cell);
                    float const solidFraction  = hasSolidFraction ? m_solidCellFraction[id] : 0.0f;

                    bool const hasLiquid      = m_type[id] == FLUID_CELL || IsBoundaryLiquidCell(cell);
                    bool const nearlySolid = m_type[id] == SOLID_CELL || IsCellSolid(cell);
                    bool const fullSolidBySdf = hasSolidFraction && solidFraction >= 1.0f - 1e-4f;
                    if (! hasLiquid || nearlySolid || fullSolidBySdf) continue;

                    result.pressureDof[id] = static_cast<int>(result.dofCell.size());
                    result.dofCell.push_back(cell);
                }
            }
        }

        return result;
    }

    int FluidSimulator::PressureDofForCell(FluidPressureDofs const & dofs, glm::ivec3 cell) const {
        if (! IsInsideGrid(cell)) return -1;
        int const id = index2GridOffset(cell);
        if (id < 0 || id >= static_cast<int>(dofs.pressureDof.size())) return -1;
        return dofs.pressureDof[id];
    }

    bool FluidSimulator::IsCellSolid(glm::ivec3 idx) const {
        if (IsInsideGrid(idx)) {
            return m_s[index2GridOffset(idx)] <= 0.5f;
        }
        return true; // 网格外视为固体
    }

    float FluidSimulator::SamplePressure(glm::vec3 const & p) const {
        if (m_iNumCells <= 0 || m_p.empty() || m_type.empty()) return 0.0f;

        glm::vec3 const gRaw = (p + glm::vec3(0.5f)) * m_fInvSpacing - glm::vec3(0.5f); // m_p以cell中心为采样点，所以需要偏移0.5个格点
        glm::vec3 const g(
            glm::clamp(gRaw.x, 0.0f, float(m_iCellX - 1)),
            glm::clamp(gRaw.y, 0.0f, float(m_iCellY - 1)),
            glm::clamp(gRaw.z, 0.0f, float(m_iCellZ - 1)));

        int const ix0 = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 1);
        int const iy0 = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 1);
        int const iz0 = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 1);
        int const ix1 = std::min(ix0 + 1, m_iCellX - 1);
        int const iy1 = std::min(iy0 + 1, m_iCellY - 1);
        int const iz1 = std::min(iz0 + 1, m_iCellZ - 1);

        float const fx = glm::clamp(g.x - float(ix0), 0.0f, 1.0f);
        float const fy = glm::clamp(g.y - float(iy0), 0.0f, 1.0f);
        float const fz = glm::clamp(g.z - float(iz0), 0.0f, 1.0f);

        auto sampleCellPressure = [&](glm::ivec3 cell) -> float {
            if (! IsInsideGrid(cell)) return 0.0f;

            int const id = index2GridOffset(cell);
            if (m_type[id] == FLUID_CELL) return m_p[id];
            if (m_type[id] == EMPTY_CELL) return 0.0f;

            // 对固体单元格使用Neumann条件dp/dn=0的局部压力外推，借用最近的流体单元压力作为该单元压力
            float bestPressure = 0.0f;
            float bestDist2    = std::numeric_limits<float>::max();
            bool  foundFluid   = false;

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        glm::ivec3 const neighbor = cell + glm::ivec3(dx, dy, dz);
                        if (! IsInsideGrid(neighbor)) continue;

                        int const neighborId = index2GridOffset(neighbor);
                        if (m_type[neighborId] != FLUID_CELL) continue;

                        glm::vec3 const neighborCenter = CellCenter(neighbor);
                        float const     dist2          = glm::dot(neighborCenter - p, neighborCenter - p);
                        if (dist2 < bestDist2) {
                            bestDist2    = dist2;
                            bestPressure = m_p[neighborId];
                            foundFluid   = true;
                        }
                    }
                }
            }

            return foundFluid ? bestPressure : 0.0f;
        };

        float const c000 = sampleCellPressure(glm::ivec3(ix0, iy0, iz0));
        float const c100 = sampleCellPressure(glm::ivec3(ix1, iy0, iz0));
        float const c010 = sampleCellPressure(glm::ivec3(ix0, iy1, iz0));
        float const c110 = sampleCellPressure(glm::ivec3(ix1, iy1, iz0));
        float const c001 = sampleCellPressure(glm::ivec3(ix0, iy0, iz1));
        float const c101 = sampleCellPressure(glm::ivec3(ix1, iy0, iz1));
        float const c011 = sampleCellPressure(glm::ivec3(ix0, iy1, iz1));
        float const c111 = sampleCellPressure(glm::ivec3(ix1, iy1, iz1));

        float const c00 = glm::mix(c000, c100, fx);
        float const c10 = glm::mix(c010, c110, fx);
        float const c01 = glm::mix(c001, c101, fx);
        float const c11 = glm::mix(c011, c111, fx);
        float const c0  = glm::mix(c00, c10, fy);
        float const c1  = glm::mix(c01, c11, fy);
        return glm::mix(c0, c1, fz);
    }

    glm::vec3 FluidSimulator::SampleVelocityPIC(glm::vec3 const & p) const {
        auto sampleVelocity = [&](glm::vec3 samplePos, int dir) -> float {
            // 直接转换到网格坐标 (调用者已预处理偏移)
            glm::vec3 gridPos = (samplePos + glm::vec3(0.5f)) * m_fInvSpacing;

            // 钳制到有效插值范围
            gridPos.x = std::max(0.0f, std::min(gridPos.x, float(m_iCellX - 1) - 1e-4f));
            gridPos.y = std::max(0.0f, std::min(gridPos.y, float(m_iCellY - 1) - 1e-4f));
            gridPos.z = std::max(0.0f, std::min(gridPos.z, float(m_iCellZ - 1) - 1e-4f));

            int   i0 = static_cast<int>(std::floor(gridPos.x));
            int   j0 = static_cast<int>(std::floor(gridPos.y));
            int   k0 = static_cast<int>(std::floor(gridPos.z));
            float fx = gridPos.x - i0;
            float fy = gridPos.y - j0;
            float fz = gridPos.z - k0;

            float accum = 0.0f;
            float wsum  = 0.0f;
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? fx : (1.0f - fx);
                for (int dj = 0; dj <= 1; ++dj) {
                    float wy = dj ? fy : (1.0f - fy);
                    for (int dk = 0; dk <= 1; ++dk) {
                        float      wz = dk ? fz : (1.0f - fz);
                        glm::ivec3 idx(i0 + di, j0 + dj, k0 + dk);
                        bool const regularFace = isValidVelocity(idx.x, idx.y, idx.z, dir);
                        bool const boundaryFace = IsVelocityFaceSolidBoundary(idx, dir);
                        if (! regularFace && ! boundaryFace) continue;

                        int   id  = index2GridOffset(idx);
                        float w   = wx * wy * wz;
                        float val = m_vel[id][dir];
                        accum += w * val;
                        wsum += w;
                    }
                }
            }
            return (wsum > 1e-6f) ? (accum / wsum) : accum;
        };

        // 将粒子位置偏移到交错面上的对应位置:
        // x-速度面 u 位于 (i, j+1/2, k+1/2) → 偏移 (0, -0.5h, -0.5h)
        // y-速度面 v 位于 (i+1/2, j, k+1/2) → 偏移 (-0.5h, 0, -0.5h)
        // z-速度面 w 位于 (i+1/2, j+1/2, k) → 偏移 (-0.5h, -0.5h, 0)
        glm::vec3 const px = p + glm::vec3(0.0f, -0.5f, -0.5f) * m_h;
        glm::vec3 const py = p + glm::vec3(-0.5f, 0.0f, -0.5f) * m_h;
        glm::vec3 const pz = p + glm::vec3(-0.5f, -0.5f, 0.0f) * m_h;

        // PIC: 直接从网格插值得到新速度 (耗散但稳定)
        glm::vec3 pic_vel;
        pic_vel.x = sampleVelocity(px, 0);
        pic_vel.y = sampleVelocity(py, 1);
        pic_vel.z = sampleVelocity(pz, 2);

        return pic_vel;
    }

    void FluidSimulator::SimulateTimestep(float dt, FluidStepConfig const & cfg) {
        float flipRatio = m_fRatio;

        float sdt = dt / cfg.numSubSteps;

        for (int step = 0; step < cfg.numSubSteps; step++) {
            integrateParticles(sdt);
            handleParticleCollisions();
            if (cfg.separateParticles)
                pushParticlesApart(cfg.numParticleIters);
            handleParticleCollisions();
            if (enableSurfaceModeling) {
                EnsureSurfaceFields();
                updateSurfaceField();
                computeSurfaceGeometry();
                applySurfaceTension(sdt);
            }
            transferVelocities(true, flipRatio);
            EnforceSolidBoundaryVelocities();
            updateParticleDensity();
            BuildFluidSimulationSDF();
            ComputeLiquidCellFractions();
            solveIncompressibility(cfg.numPressureIters, sdt, cfg.overRelaxation, cfg.compensateDrift);
            EnforceSolidBoundaryVelocities();
            transferVelocities(false, flipRatio);
        }
        updateParticleColors();
        updateRenderableSurface();
    }

    void FluidSimulator::updateRenderableSurface() {
        EnsureRenderableSurfaceFields();
        if (m_iNumSpheres <= 0 || m_renderSurfaceCellX <= 1 || m_renderSurfaceCellY <= 1 || m_renderSurfaceCellZ <= 1)
            return;

        int const updateInterval    = std::max(1, m_renderSurfaceUpdateInterval);
        m_renderSurfaceFrameCounter = (m_renderSurfaceFrameCounter + 1) % updateInterval;
        if (m_renderSurfaceFrameCounter != 0) return;

        m_renderSurfaceMesh.positions.clear();
        m_renderSurfaceMesh.normals.clear();
        m_renderSurfaceMesh.indices.clear();

        int const renderCellCount = m_renderSurfaceCellX * m_renderSurfaceCellY * m_renderSurfaceCellZ;
        if (static_cast<int>(m_renderSurfaceColor.size()) != renderCellCount)
            m_renderSurfaceColor.assign(renderCellCount, 0.0f);
        if (static_cast<int>(m_renderSurfacePhi.size()) != renderCellCount)
            m_renderSurfacePhi.assign(renderCellCount, 0.0f);

        auto renderIndex = [&](int i, int j, int k) {
            return i * (m_renderSurfaceCellY * m_renderSurfaceCellZ) + j * m_renderSurfaceCellZ + k;
        };

        auto renderPosition = [&](int i, int j, int k) {
            return glm::vec3(i, j, k) * m_renderSurfaceH - glm::vec3(0.5f);
        };

        std::fill(m_renderSurfaceColor.begin(), m_renderSurfaceColor.end(), 0.0f);
        std::vector<unsigned char> activePoint(renderCellCount, 0); // 记录参与表面建模的格点
        std::vector<unsigned char> activeWork(renderCellCount, 0);

        // 两种离线表面重建模式：
        // 1) density：原来的核密度标量场，画面连续但容易过度圆润，像果冻/软胶。
        // 2) sdf：每个粒子作为一个小球，取 min(||x-xp||-r)，更接近 FluidRigidCoupling3D 的 SDF 导出方法。
        if (m_renderSurfaceUseParticleSdf) {
            float const sdfRadius = (m_renderSurfaceSdfParticleRadius > 0.0f)
                ? m_renderSurfaceSdfParticleRadius
                : 1.55f * m_particleRadius;
            float const farPhi = std::max(4.0f * sdfRadius, 4.0f * m_renderSurfaceH);
            float const activeBand = std::max(2.5f * m_renderSurfaceH, 0.75f * sdfRadius);
            int const reach = std::max(1, static_cast<int>(std::ceil((sdfRadius + activeBand) * m_renderSurfaceInvH)));

            std::fill(m_renderSurfacePhi.begin(), m_renderSurfacePhi.end(), farPhi);

            for (glm::vec3 const & particlePos : m_particlePos) {
                glm::vec3 const grid = (particlePos + glm::vec3(0.5f)) * m_renderSurfaceInvH;
                int const       i0   = clampCoord(static_cast<int>(std::floor(grid.x)) - reach, 0, m_renderSurfaceCellX - 1);
                int const       j0   = clampCoord(static_cast<int>(std::floor(grid.y)) - reach, 0, m_renderSurfaceCellY - 1);
                int const       k0   = clampCoord(static_cast<int>(std::floor(grid.z)) - reach, 0, m_renderSurfaceCellZ - 1);
                int const       i1   = clampCoord(static_cast<int>(std::floor(grid.x)) + reach, 0, m_renderSurfaceCellX - 1);
                int const       j1   = clampCoord(static_cast<int>(std::floor(grid.y)) + reach, 0, m_renderSurfaceCellY - 1);
                int const       k1   = clampCoord(static_cast<int>(std::floor(grid.z)) + reach, 0, m_renderSurfaceCellZ - 1);

                for (int i = i0; i <= i1; ++i) {
                    for (int j = j0; j <= j1; ++j) {
                        for (int k = k0; k <= k1; ++k) {
                            glm::vec3 const p = renderPosition(i, j, k);
                            float const phi = glm::length(p - particlePos) - sdfRadius;
                            if (phi >= m_renderSurfacePhi[renderIndex(i, j, k)] && std::abs(phi) > activeBand)
                                continue;
                            int const id = renderIndex(i, j, k);
                            if (phi < m_renderSurfacePhi[id])
                                m_renderSurfacePhi[id] = phi;
                            if (phi <= activeBand)
                                activePoint[id] = 1;
                        }
                    }
                }
            }

            // 只扩展一小圈 active band，Marching Tetrahedra 只遍历界面附近，避免整场景误出面。
            activeWork = activePoint;
            for (int expand = 0; expand < 2; ++expand) {
                std::vector<unsigned char> expanded = activeWork;
                for (int i = 0; i < m_renderSurfaceCellX; ++i) {
                    for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                        for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                            int const id = renderIndex(i, j, k);
                            if (! activeWork[id]) continue;
                            for (int di = -1; di <= 1; ++di) {
                                int const ni = clampCoord(i + di, 0, m_renderSurfaceCellX - 1);
                                for (int dj = -1; dj <= 1; ++dj) {
                                    int const nj = clampCoord(j + dj, 0, m_renderSurfaceCellY - 1);
                                    for (int dk = -1; dk <= 1; ++dk) {
                                        int const nk = clampCoord(k + dk, 0, m_renderSurfaceCellZ - 1);
                                        expanded[renderIndex(ni, nj, nk)] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
                activeWork.swap(expanded);
            }

            // 直接 union-of-particles SDF 会保留每个粒子的球形颗粒感。
            // 这里做少量窄带平滑，只用于离线渲染，不参与压力投影/耦合求解。
            // 目的不是把水变成软胶，而是补掉粒子间的小洞并让法线更连续。
            int const sdfSmoothIters = std::max(0, m_renderSurfaceSdfSmoothIters);
            std::vector<float> scratch(renderCellCount, farPhi);
            for (int iter = 0; iter < sdfSmoothIters; ++iter) {
                for (int i = 0; i < m_renderSurfaceCellX; ++i) {
                    for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                        for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                            int const id = renderIndex(i, j, k);
                            if (! activeWork[id]) {
                                scratch[id] = m_renderSurfacePhi[id];
                                continue;
                            }

                            float accum = 4.0f * m_renderSurfacePhi[id];
                            float wsum  = 4.0f;
                            for (int di = -1; di <= 1; ++di) {
                                int const ni = clampCoord(i + di, 0, m_renderSurfaceCellX - 1);
                                for (int dj = -1; dj <= 1; ++dj) {
                                    int const nj = clampCoord(j + dj, 0, m_renderSurfaceCellY - 1);
                                    for (int dk = -1; dk <= 1; ++dk) {
                                        if (di == 0 && dj == 0 && dk == 0) continue;
                                        int const nk  = clampCoord(k + dk, 0, m_renderSurfaceCellZ - 1);
                                        int const nid = renderIndex(ni, nj, nk);
                                        if (! activeWork[nid]) continue;
                                        int const manhattan = std::abs(di) + std::abs(dj) + std::abs(dk);
                                        float const w = (manhattan == 1) ? 2.0f : 1.0f;
                                        accum += w * m_renderSurfacePhi[nid];
                                        wsum += w;
                                    }
                                }
                            }
                            scratch[id] = accum / std::max(wsum, 1e-6f);
                        }
                    }
                }
                m_renderSurfacePhi.swap(scratch);
            }

            // SDF 模式的小洞修补：如果一个窄带格点被足够多内部邻居包围，
            // 说明它多半是粒子采样造成的孔洞，而不是真正的空气区域。
            std::vector<float> closedPhi = m_renderSurfacePhi;
            float const closeInsidePhi = -0.20f * m_renderSurfaceH;
            for (int i = 1; i < m_renderSurfaceCellX - 1; ++i) {
                for (int j = 1; j < m_renderSurfaceCellY - 1; ++j) {
                    for (int k = 1; k < m_renderSurfaceCellZ - 1; ++k) {
                        int const id = renderIndex(i, j, k);
                        if (! activeWork[id] || m_renderSurfacePhi[id] < 0.0f) continue;

                        int activeNeighbors = 0;
                        int insideNeighbors = 0;
                        for (int di = -1; di <= 1; ++di) {
                            for (int dj = -1; dj <= 1; ++dj) {
                                for (int dk = -1; dk <= 1; ++dk) {
                                    if (di == 0 && dj == 0 && dk == 0) continue;
                                    int const nid = renderIndex(i + di, j + dj, k + dk);
                                    if (! activeWork[nid]) continue;
                                    ++activeNeighbors;
                                    if (m_renderSurfacePhi[nid] < 0.0f) ++insideNeighbors;
                                }
                            }
                        }
                        if (activeNeighbors >= 18 && insideNeighbors >= 14)
                            closedPhi[id] = closeInsidePhi;
                    }
                }
            }
            m_renderSurfacePhi.swap(closedPhi);
        } else {
            float const radius = (m_renderSurfaceKernelRadius > 0.0f) ? m_renderSurfaceKernelRadius : 1.8f * m_h;
            float const invR   = 1.0f / radius;
            int const   reach  = std::max(1, static_cast<int>(std::ceil(radius * m_renderSurfaceInvH)));

            // 核函数
            auto kernel = [&](float r) {
                float const q = r * invR;
                if (q >= 1.0f) return 0.0f;
                float const s = 1.0f - q * q;
                return s * s * s;
            };

            std::vector<float> restSamples;
            restSamples.reserve(static_cast<std::size_t>(m_iNumSpheres));

            // 对每个粒子的近邻格子计算核函数累加到color上
            // 计算复杂度和粒子数成正比，而和场景大小无关
            for (glm::vec3 const & particlePos : m_particlePos) {
                glm::vec3 const grid = (particlePos + glm::vec3(0.5f)) * m_renderSurfaceInvH;
                int const       i0   = clampCoord(static_cast<int>(std::floor(grid.x)) - reach, 0, m_renderSurfaceCellX - 1);
                int const       j0   = clampCoord(static_cast<int>(std::floor(grid.y)) - reach, 0, m_renderSurfaceCellY - 1);
                int const       k0   = clampCoord(static_cast<int>(std::floor(grid.z)) - reach, 0, m_renderSurfaceCellZ - 1);
                int const       i1   = clampCoord(static_cast<int>(std::floor(grid.x)) + reach, 0, m_renderSurfaceCellX - 1);
                int const       j1   = clampCoord(static_cast<int>(std::floor(grid.y)) + reach, 0, m_renderSurfaceCellY - 1);
                int const       k1   = clampCoord(static_cast<int>(std::floor(grid.z)) + reach, 0, m_renderSurfaceCellZ - 1);

                for (int i = i0; i <= i1; ++i) {
                    for (int j = j0; j <= j1; ++j) {
                        for (int k = k0; k <= k1; ++k) {
                            glm::vec3 const p = renderPosition(i, j, k);
                            float const     w = kernel(glm::length(p - particlePos));
                            if (w <= 0.0f) continue;

                            int const id = renderIndex(i, j, k);
                            m_renderSurfaceColor[id] += w;
                            activePoint[id] = 1;
                        }
                    }
                }
            }

            for (int id = 0; id < renderCellCount; ++id) {
                float const value = m_renderSurfaceColor[id];
                if (value > 1e-6f) restSamples.push_back(value);
            }

            // percentile找到80%分位数作为restField
            float restField = 1.0f;
            if (! restSamples.empty()) {
                std::size_t const percentileIndex = std::min(
                    restSamples.size() - 1,
                    static_cast<std::size_t>(0.80f * float(restSamples.size() - 1)));
                std::nth_element(restSamples.begin(), restSamples.begin() + percentileIndex, restSamples.end());
                restField = std::max(restSamples[percentileIndex], 1e-6f);
            }
            for (float & value : m_renderSurfaceColor)
                value = glm::clamp(value / restField, 0.0f, 1.5f);

            // 先将参与建模的格点扩展到邻域内，避免模糊时遗漏边界点导致表面破洞
            int const blurIters = std::max(0, m_renderSurfaceBlurIters);
            activeWork = activePoint;
            for (int expand = 0; expand < blurIters + 1; ++expand) {
                std::vector<unsigned char> expanded = activeWork;
                for (int i = 0; i < m_renderSurfaceCellX; ++i) {
                    for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                        for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                            int const id = renderIndex(i, j, k);
                            if (! activeWork[id]) continue;
                            for (int di = -1; di <= 1; ++di) {
                                int const ni = clampCoord(i + di, 0, m_renderSurfaceCellX - 1);
                                for (int dj = -1; dj <= 1; ++dj) {
                                    int const nj = clampCoord(j + dj, 0, m_renderSurfaceCellY - 1);
                                    for (int dk = -1; dk <= 1; ++dk) {
                                        int const nk = clampCoord(k + dk, 0, m_renderSurfaceCellZ - 1);
                                        expanded[renderIndex(ni, nj, nk)] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
                activeWork.swap(expanded);
            }

            // 模糊处理
            std::vector<float> scratch(renderCellCount, 0.0f);
            for (int iter = 0; iter < blurIters; ++iter) {
                for (int i = 0; i < m_renderSurfaceCellX; ++i) {
                    for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                        for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                            int const id = renderIndex(i, j, k);
                            if (! activeWork[id]) {
                                scratch[id] = 0.0f;
                                continue;
                            }
                            float accum = 0.0f;
                            float wsum  = 0.0f;
                            for (int di = -1; di <= 1; ++di) {
                                int const   ni = clampCoord(i + di, 0, m_renderSurfaceCellX - 1);
                                float const wi = (di == 0) ? 2.0f : 1.0f;
                                for (int dj = -1; dj <= 1; ++dj) {
                                    int const   nj = clampCoord(j + dj, 0, m_renderSurfaceCellY - 1);
                                    float const wj = (dj == 0) ? 2.0f : 1.0f;
                                    for (int dk = -1; dk <= 1; ++dk) {
                                        int const   nk = clampCoord(k + dk, 0, m_renderSurfaceCellZ - 1);
                                        float const wk = (dk == 0) ? 2.0f : 1.0f;
                                        float const w  = wi * wj * wk;
                                        accum += w * m_renderSurfaceColor[renderIndex(ni, nj, nk)];
                                        wsum += w;
                                    }
                                }
                            }
                            scratch[id] = accum / std::max(wsum, 1e-6f);
                        }
                    }
                }
                m_renderSurfaceColor.swap(scratch);
            }

            for (int id = 0; id < renderCellCount; ++id)
                m_renderSurfacePhi[id] = m_renderSurfaceIsoValue - m_renderSurfaceColor[id];

            // 修补表面渲染中出现的破洞：如果一个格点的active邻居数和inside邻居数都超过一定阈值，则认为该格点也在内部
            std::vector<float> filledPhi = m_renderSurfacePhi;
            float const fillInsidePhi = -0.25f * m_renderSurfaceH;
            for (int i = 1; i < m_renderSurfaceCellX - 1; ++i) {
                for (int j = 1; j < m_renderSurfaceCellY - 1; ++j) {
                    for (int k = 1; k < m_renderSurfaceCellZ - 1; ++k) {
                        int const id = renderIndex(i, j, k);
                        if (! activeWork[id] || m_renderSurfacePhi[id] < 0.0f) continue;

                        int insideNeighbors = 0;
                        int activeNeighbors = 0;
                        for (int di = -1; di <= 1; ++di) {
                            for (int dj = -1; dj <= 1; ++dj) {
                                for (int dk = -1; dk <= 1; ++dk) {
                                    if (di == 0 && dj == 0 && dk == 0) continue;
                                    int const nid = renderIndex(i + di, j + dj, k + dk);
                                    if (! activeWork[nid]) continue;
                                    ++activeNeighbors;
                                    if (m_renderSurfacePhi[nid] < 0.0f) ++insideNeighbors;
                                }
                            }
                        }

                        if (activeNeighbors >= 18 && insideNeighbors >= 18)
                            filledPhi[id] = fillInsidePhi;
                    }
                }
            }
            m_renderSurfacePhi.swap(filledPhi);
        }

        auto samplePhi = [&](int i, int j, int k) {
            return m_renderSurfacePhi[renderIndex(
                clampCoord(i, 0, m_renderSurfaceCellX - 1),
                clampCoord(j, 0, m_renderSurfaceCellY - 1),
                clampCoord(k, 0, m_renderSurfaceCellZ - 1))];
        };

        if (m_renderSurfaceNormal.size() != std::size_t(renderCellCount))
            m_renderSurfaceNormal.assign(renderCellCount, glm::vec3(0.0f, 1.0f, 0.0f));

        for (int i = 0; i < m_renderSurfaceCellX; ++i) {
            for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                    int const id = renderIndex(i, j, k);
                    if (! activeWork[id]) {
                        m_renderSurfaceNormal[id] = glm::vec3(0.0f, 1.0f, 0.0f);
                        continue;
                    }

                    glm::vec3 grad(
                        (samplePhi(i + 1, j, k) - samplePhi(i - 1, j, k)) * 0.5f * m_renderSurfaceInvH,
                        (samplePhi(i, j + 1, k) - samplePhi(i, j - 1, k)) * 0.5f * m_renderSurfaceInvH,
                        (samplePhi(i, j, k + 1) - samplePhi(i, j, k - 1)) * 0.5f * m_renderSurfaceInvH);
                    float const len = glm::length(grad);
                    m_renderSurfaceNormal[id] = (len > 1e-6f) ? grad / len : glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        auto normalAt = [&](glm::vec3 const & p) {
            glm::vec3 g = (p + glm::vec3(0.5f)) * m_renderSurfaceInvH;
            g.x = glm::clamp(g.x, 0.0f, float(m_renderSurfaceCellX - 1));
            g.y = glm::clamp(g.y, 0.0f, float(m_renderSurfaceCellY - 1));
            g.z = glm::clamp(g.z, 0.0f, float(m_renderSurfaceCellZ - 1));

            int const i0 = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_renderSurfaceCellX - 1);
            int const j0 = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_renderSurfaceCellY - 1);
            int const k0 = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_renderSurfaceCellZ - 1);
            int const i1 = std::min(i0 + 1, m_renderSurfaceCellX - 1);
            int const j1 = std::min(j0 + 1, m_renderSurfaceCellY - 1);
            int const k1 = std::min(k0 + 1, m_renderSurfaceCellZ - 1);
            float const fx = glm::clamp(g.x - float(i0), 0.0f, 1.0f);
            float const fy = glm::clamp(g.y - float(j0), 0.0f, 1.0f);
            float const fz = glm::clamp(g.z - float(k0), 0.0f, 1.0f);

            auto at = [&](int i, int j, int k) {
                return m_renderSurfaceNormal[renderIndex(i, j, k)];
            };

            glm::vec3 const c00 = glm::mix(at(i0, j0, k0), at(i1, j0, k0), fx);
            glm::vec3 const c10 = glm::mix(at(i0, j1, k0), at(i1, j1, k0), fx);
            glm::vec3 const c01 = glm::mix(at(i0, j0, k1), at(i1, j0, k1), fx);
            glm::vec3 const c11 = glm::mix(at(i0, j1, k1), at(i1, j1, k1), fx);
            glm::vec3 const n = glm::mix(glm::mix(c00, c10, fy), glm::mix(c01, c11, fy), fz);
            float const len = glm::length(n);
            return (len > 1e-6f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        };

        auto emitTriangle = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
            glm::vec3 normalA = normalAt(a);
            glm::vec3 normalB = normalAt(b);
            glm::vec3 normalC = normalAt(c);
            glm::vec3 const faceNormal = glm::cross(b - a, c - a);
            glm::vec3 const averageNormal = normalA + normalB + normalC;
            if (glm::dot(faceNormal, averageNormal) < 0.0f) {
                std::swap(b, c);
                std::swap(normalB, normalC);
            }

            std::uint32_t const base = static_cast<std::uint32_t>(m_renderSurfaceMesh.positions.size());
            m_renderSurfaceMesh.positions.push_back(a);
            m_renderSurfaceMesh.positions.push_back(b);
            m_renderSurfaceMesh.positions.push_back(c);
            m_renderSurfaceMesh.normals.push_back(normalA);
            m_renderSurfaceMesh.normals.push_back(normalB);
            m_renderSurfaceMesh.normals.push_back(normalC);
            m_renderSurfaceMesh.indices.push_back(base);
            m_renderSurfaceMesh.indices.push_back(base + 1);
            m_renderSurfaceMesh.indices.push_back(base + 2);
        };

        auto interpolate = [&](glm::vec3 const & a, glm::vec3 const & b, float phiA, float phiB) {
            float const denom = phiA - phiB;
            float const t     = (std::abs(denom) > 1e-8f) ? glm::clamp(phiA / denom, 0.0f, 1.0f) : 0.5f;
            return glm::mix(a, b, t);
        };

        // 用 Marching Tetrahedra 对界面处四面体进行分割
        auto polygonizeTetra = [&](std::array<glm::vec3, 4> const & p, std::array<float, 4> const & phi) {
            std::array<int, 4> inside {};
            std::array<int, 4> outside {};
            int                insideCount  = 0;
            int                outsideCount = 0;
            for (int i = 0; i < 4; ++i) {
                if (phi[i] < 0.0f)
                    inside[insideCount++] = i;
                else
                    outside[outsideCount++] = i;
            }
            if (insideCount == 0 || insideCount == 4) return; // 整个四面体都在流体内部或者都在流体外部

            if (insideCount == 1 || insideCount == 3) {
                bool const      flip = (insideCount == 3);
                int const       a    = flip ? outside[0] : inside[0];
                int const       b    = flip ? inside[0] : outside[0];
                int const       c    = flip ? inside[1] : outside[1];
                int const       d    = flip ? inside[2] : outside[2];
                glm::vec3 const p0   = interpolate(p[a], p[b], phi[a], phi[b]);
                glm::vec3 const p1   = interpolate(p[a], p[c], phi[a], phi[c]);
                glm::vec3 const p2   = interpolate(p[a], p[d], phi[a], phi[d]);
                if (flip) emitTriangle(p0, p2, p1);
                else emitTriangle(p0, p1, p2);
                return;
            }

            int const       a  = inside[0];
            int const       b  = inside[1];
            int const       c  = outside[0];
            int const       d  = outside[1];
            glm::vec3 const p0 = interpolate(p[a], p[c], phi[a], phi[c]);
            glm::vec3 const p1 = interpolate(p[a], p[d], phi[a], phi[d]);
            glm::vec3 const p2 = interpolate(p[b], p[c], phi[b], phi[c]);
            glm::vec3 const p3 = interpolate(p[b], p[d], phi[b], phi[d]);
            emitTriangle(p0, p1, p2);
            emitTriangle(p2, p1, p3);
        };

        static constexpr int cornerOffset[8][3] {
            { 0, 0, 0 },
            { 1, 0, 0 },
            { 1, 1, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 1, 0, 1 },
            { 1, 1, 1 },
            { 0, 1, 1 }
        };
        static constexpr int tetrahedra[6][4] {
            { 0, 5, 1, 6 },
            { 0, 1, 2, 6 },
            { 0, 2, 3, 6 },
            { 0, 3, 7, 6 },
            { 0, 7, 4, 6 },
            { 0, 4, 5, 6 }
        };

        // 每个立方体网格分解成6个四面体，分别进行Marching Tetrahedra处理，生成三角形网格
        for (int i = 0; i < m_renderSurfaceCellX - 1; ++i) {
            for (int j = 0; j < m_renderSurfaceCellY - 1; ++j) {
                for (int k = 0; k < m_renderSurfaceCellZ - 1; ++k) {
                    std::array<glm::vec3, 8> cubePos;
                    std::array<float, 8>     cubePhi;
                    int                      insideMask = 0;
                    bool                     activeCube = false;
                    for (int c = 0; c < 8; ++c) {
                        int const x = i + cornerOffset[c][0];
                        int const y = j + cornerOffset[c][1];
                        int const z = k + cornerOffset[c][2];
                        cubePos[c]  = renderPosition(x, y, z);
                        cubePhi[c]  = m_renderSurfacePhi[renderIndex(x, y, z)];
                        activeCube  = activeCube || activeWork[renderIndex(x, y, z)] != 0;
                        if (cubePhi[c] < 0.0f) insideMask |= (1 << c);
                    }
                    if (! activeCube || insideMask == 0 || insideMask == 255) continue;

                    for (auto const & tet : tetrahedra) {
                        std::array<glm::vec3, 4> tetPos {
                            cubePos[tet[0]], cubePos[tet[1]], cubePos[tet[2]], cubePos[tet[3]]
                        };
                        std::array<float, 4> tetPhi {
                            cubePhi[tet[0]], cubePhi[tet[1]], cubePhi[tet[2]], cubePhi[tet[3]]
                        };
                        polygonizeTetra(tetPos, tetPhi);
                    }
                }
            }
        }
    }

} // namespace VCX::MainScene
