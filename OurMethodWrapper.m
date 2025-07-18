%% ==========================================================================
%% 3. 我们方法的包装器：OurMethodWrapper.m
%% ==========================================================================
classdef OurMethodWrapper < handle
    properties
        behavior_models
        trust_region_builder
    end
    
    methods
        function obj = OurMethodWrapper(behavior_models)
            obj.behavior_models = behavior_models;
            obj.trust_region_builder = TrustRegionBuilder(behavior_models);
        end
        
        function result = build_trust_region(obj, scenario)
            % 使用完整的我们的方法
            full_result = obj.trust_region_builder.build_trust_region(scenario);
            
            % 提取关键信息用于对比
            result = struct(...
                'method_name', 'Our Method (Full)', ...
                'grid_outdoor', full_result.gpr_result.grid_outdoor, ...
                'grid_indoor', full_result.gpr_result.grid_indoor, ...
                'trust_scores', full_result.gpr_result.smooth_surface, ...
                'adjustable_area', sum(full_result.gpr_result.smooth_surface(:) > 0.4), ...
                'max_confidence', max(full_result.gpr_result.smooth_surface(:)), ...
                'mean_confidence', mean(full_result.gpr_result.smooth_surface(:)), ...
                'trust_boundaries', full_result.trust_boundaries);
        end
        
        function result = build_trust_region_no_gpr(obj, scenario)
            % 消融实验: 移除GPR平滑的版本
            
            % 创建采样网格
            [grid_outdoor, grid_indoor] = obj.trust_region_builder.create_sampling_grid();
            
            % 查询置信度 (不使用GPR平滑)
            raw_trust_scores = obj.trust_region_builder.query_trust_scores(grid_outdoor, grid_indoor, scenario);
            
            result = struct(...
                'method_name', 'Our Method (No GPR)', ...
                'grid_outdoor', grid_outdoor, ...
                'grid_indoor', grid_indoor, ...
                'trust_scores', raw_trust_scores, ...
                'adjustable_area', sum(raw_trust_scores(:) > 0.4), ...
                'max_confidence', max(raw_trust_scores(:)), ...
                'mean_confidence', mean(raw_trust_scores(:)));
        end
    end
end
