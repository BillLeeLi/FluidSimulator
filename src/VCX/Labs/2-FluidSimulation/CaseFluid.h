#pragma once

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/UniformBlock.hpp"
#include "Engine/Sphere.h"
#include "Labs/2-FluidSimulation/FluidSimulator.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "Labs/Scene/Content.h"
#include "Labs/Scene/SceneObject.h"

namespace VCX::Labs::Fluid {

    class CaseFluid : public Common::ICase {
    public:
        CaseFluid(std::initializer_list<Assets::ExampleScene> && scenes);

        virtual std::string_view const GetName() override { return "Fluid Simulation"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        std::vector<Assets::ExampleScene> const _scenes;
    
        // 流体粒子渲染程序 (fluid.vert + fluid.frag)
        Engine::GL::UniqueProgram             _program;
        // 边界框线框程序 (flat.vert + flat.frag)
        Engine::GL::UniqueProgram             _lineProgram;
        Engine::GL::UniqueRenderFrame         _frame;

        // 使用 SceneObject 来管理 PassConstants UBO (与lab0相同的方式)
        Rendering::SceneObject                _sceneObject;

        // 水槽边界框
        Engine::GL::UniqueIndexedRenderItem   _boundaryItem;
        
        // 相机控制
        Common::OrbitCameraManager            _cameraManager;
        
        // 球体网格 (每个粒子绘制为一个小球)
        Engine::Model                         _sphere;
        
        // 模拟参数
        int   _res        = 16;
        bool  _stopped    = false;
        bool  _uniformDirty = true;
        float _lineWidth  = 2.0f;
        
        // 流体模拟器
        Simulator _simulation;

        void ResetSystem();
    };

} // namespace VCX::Labs::Fluid
