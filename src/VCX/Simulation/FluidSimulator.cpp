#include "Simulation/FluidSimulator.h"

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
        return m_s[index2GridOffset(glm::ivec3(i, j, k))] > 0.5f;
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

                                glm::vec3 dir = m_particlePos[i] - m_particlePos[j];
                                float     len = glm::length(dir);
                                if (len < minDist && len > 1e-8f) {
                                    glm::vec3 corr = 0.5f * (minDist - len) * dir / len;
                                    m_particlePos[i] += corr;
                                    m_particlePos[j] -= corr;
                                }
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
                            if (! isValidVelocity(idx.x, idx.y, idx.z, dir)) continue;
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
        int const renderScale       = std::max(1, m_renderSurfaceResolutionScale);
        m_renderSurfaceCellX        = (m_iCellX - 1) * renderScale + 1;
        m_renderSurfaceCellY        = (m_iCellY - 1) * renderScale + 1;
        m_renderSurfaceCellZ        = (m_iCellZ - 1) * renderScale + 1;
        m_renderSurfaceH            = m_h / float(renderScale);
        m_renderSurfaceInvH         = 1.0f / m_renderSurfaceH;
        m_renderSurfaceKernelRadius = 2.5f * m_h;

        int const renderCellCount = m_renderSurfaceCellX * m_renderSurfaceCellY * m_renderSurfaceCellZ;
        m_renderSurfaceColor.clear();
        m_renderSurfaceColor.resize(renderCellCount, 0.0f);
        m_renderSurfacePhi.clear();
        m_renderSurfacePhi.resize(renderCellCount, 0.0f);
        m_renderSurfaceMesh.positions.clear();
        m_renderSurfaceMesh.normals.clear();
        m_renderSurfaceMesh.indices.clear();
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
        glm::vec3 relWater = { 0.6f, 0.5f, 0.6f };

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
                    if (i == 0 || i == m_iCellX - 1
                        || j == 0 || j == m_iCellY - 1
                        || k == 0 || k == m_iCellZ - 1)
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
                    if (i == 0 || i == m_iCellX - 1
                        || j == 0 || j == m_iCellY - 1
                        || k == 0 || k == m_iCellZ - 1)
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

    void FluidSimulator::SetSolidBoundaryVelocity(glm::ivec3 idx, int dir, float velocity) {
        if (! isVelocityFaceInRange(idx.x, idx.y, idx.z, dir)) return;

        int const id            = index2GridOffset(idx);
        m_solidVel[id][dir]     = velocity;
        m_solidVelMask[id][dir] = 1;
        m_vel[id][dir]          = velocity;
        m_pre_vel[id][dir]      = velocity;
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
                        if (! isValidVelocity(idx.x, idx.y, idx.z, dir)) continue;
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
            updateParticleDensity();
            solveIncompressibility(cfg.numPressureIters, sdt, cfg.overRelaxation, cfg.compensateDrift);
            transferVelocities(false, flipRatio);
        }
        updateParticleColors();
        updateRenderableSurface();
    }

    void FluidSimulator::updateRenderableSurface() {
        m_renderSurfaceMesh.positions.clear();
        m_renderSurfaceMesh.normals.clear();
        m_renderSurfaceMesh.indices.clear();

        EnsureRenderableSurfaceFields();
        if (m_iNumSpheres <= 0 || m_renderSurfaceCellX <= 1 || m_renderSurfaceCellY <= 1 || m_renderSurfaceCellZ <= 1)
            return;

        buildHash();

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

        float const radius = (m_renderSurfaceKernelRadius > 0.0f) ? m_renderSurfaceKernelRadius : 2.5f * m_h;
        float const invR   = 1.0f / radius;
        int const   reach  = std::max(1, static_cast<int>(std::ceil(radius * m_fInvSpacing)));

        // 核函数
        auto kernel = [&](float r) {
            float const q = r * invR;
            if (q >= 1.0f) return 0.0f;
            float const s = 1.0f - q * q;
            return s * s * s;
        };

        float restFieldAccum = 0.0f;
        int   restFieldCount = 0;

        // 对每个网格计算Color Field值: 遍历附近粒子并累加核函数权重
        for (int i = 0; i < m_renderSurfaceCellX; ++i) {
            for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
                    glm::vec3 const  p    = renderPosition(i, j, k);
                    glm::ivec3 const slot = worldToCell(p);

                    float value = 0.0f;
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
                                    int const particle = m_hashtable[ptr];
                                    value += kernel(glm::length(p - m_particlePos[particle]));
                                }
                            }
                        }
                    }

                    int const id             = renderIndex(i, j, k);
                    m_renderSurfaceColor[id] = value;
                    if (value > 1e-6f) {
                        restFieldAccum += value;
                        ++restFieldCount;
                    }
                }
            }
        }

        // 用均值进行归一化
        float const restField = (restFieldCount > 0) ? std::max(restFieldAccum / float(restFieldCount), 1e-6f) : 1.0f;
        for (float & value : m_renderSurfaceColor)
            value = glm::clamp(value / restField, 0.0f, 1.5f);

        // 模糊处理，平滑表面
        int const          blurIters = std::max(0, m_renderSurfaceBlurIters);
        std::vector<float> scratch(renderCellCount, 0.0f);
        for (int iter = 0; iter < blurIters; ++iter) {
            for (int i = 0; i < m_renderSurfaceCellX; ++i) {
                for (int j = 0; j < m_renderSurfaceCellY; ++j) {
                    for (int k = 0; k < m_renderSurfaceCellZ; ++k) {
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
                        scratch[renderIndex(i, j, k)] = accum / std::max(wsum, 1e-6f);
                    }
                }
            }
            m_renderSurfaceColor.swap(scratch);
        }

        for (int id = 0; id < renderCellCount; ++id)
            m_renderSurfacePhi[id] = m_renderSurfaceIsoValue - m_renderSurfaceColor[id];

        auto samplePhi = [&](int i, int j, int k) {
            return m_renderSurfacePhi[renderIndex(
                clampCoord(i, 0, m_renderSurfaceCellX - 1),
                clampCoord(j, 0, m_renderSurfaceCellY - 1),
                clampCoord(k, 0, m_renderSurfaceCellZ - 1))];
        };

        auto normalAt = [&](glm::vec3 const & p) {
            glm::vec3 g = (p + glm::vec3(0.5f)) * m_renderSurfaceInvH;
            int const i = clampCoord(static_cast<int>(std::round(g.x)), 0, m_renderSurfaceCellX - 1);
            int const j = clampCoord(static_cast<int>(std::round(g.y)), 0, m_renderSurfaceCellY - 1);
            int const k = clampCoord(static_cast<int>(std::round(g.z)), 0, m_renderSurfaceCellZ - 1);

            glm::vec3 grad(
                (samplePhi(i + 1, j, k) - samplePhi(i - 1, j, k)) * 0.5f * m_renderSurfaceInvH,
                (samplePhi(i, j + 1, k) - samplePhi(i, j - 1, k)) * 0.5f * m_renderSurfaceInvH,
                (samplePhi(i, j, k + 1) - samplePhi(i, j, k - 1)) * 0.5f * m_renderSurfaceInvH);
            float const len = glm::length(grad);
            return (len > 1e-6f) ? grad / len : glm::vec3(0.0f, 1.0f, 0.0f);
        };

        auto emitTriangle = [&](glm::vec3 const & a, glm::vec3 const & b, glm::vec3 const & c) {
            std::uint32_t const base = static_cast<std::uint32_t>(m_renderSurfaceMesh.positions.size());
            m_renderSurfaceMesh.positions.push_back(a);
            m_renderSurfaceMesh.positions.push_back(b);
            m_renderSurfaceMesh.positions.push_back(c);
            m_renderSurfaceMesh.normals.push_back(normalAt(a));
            m_renderSurfaceMesh.normals.push_back(normalAt(b));
            m_renderSurfaceMesh.normals.push_back(normalAt(c));
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
                    for (int c = 0; c < 8; ++c) {
                        int const x = i + cornerOffset[c][0];
                        int const y = j + cornerOffset[c][1];
                        int const z = k + cornerOffset[c][2];
                        cubePos[c]  = renderPosition(x, y, z);
                        cubePhi[c]  = m_renderSurfacePhi[renderIndex(x, y, z)];
                        if (cubePhi[c] < 0.0f) insideMask |= (1 << c);
                    }
                    if (insideMask == 0 || insideMask == 255) continue;

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
