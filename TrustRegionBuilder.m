%% ==========================================================================
%% 5. 可信域构建器：TrustRegionBuilder.m（包含增量式GPR）
%% ==========================================================================
classdef TrustRegionBuilder < handle
    properties
        behavior_models
        gpr_engine
        sampling_grid
        trust_scores
    end
    
    methods
        function obj = TrustRegionBuilder(behavior_models)
            obj.behavior_models = behavior_models;
            obj.gpr_engine = IncrementalGPREngine();
        end
        
        function result = build_trust_region(obj, scenario)
            fprintf('开始构建可信调控域...\n');
            
            % 第一步：智能采样
            [grid_outdoor, grid_indoor] = obj.create_sampling_grid();
            obj.sampling_grid = struct('outdoor', grid_outdoor, 'indoor', grid_indoor);
            
            % 第二步：查询置信度
            trust_scores = obj.query_trust_scores(grid_outdoor, grid_indoor, scenario);
            obj.trust_scores = trust_scores;
            
            % 第三步：GPR拟合
            gpr_result = obj.fit_gpr_surface(grid_outdoor, grid_indoor, trust_scores);
            
            % 第四步：提取边界
            boundaries = obj.extract_trust_boundaries(gpr_result, [0.6, 0.8]);
            
            result = struct(...
                'scenario', scenario, ...
                'sampling_grid', obj.sampling_grid, ...
                'raw_trust_scores', trust_scores, ...
                'gpr_result', gpr_result, ...
                'trust_boundaries', boundaries);
            
            fprintf('可信调控域构建完成。\n');
        end
        
        function [grid_outdoor, grid_indoor] = create_sampling_grid(obj)
            outdoor_temps = linspace(15, 35, 80);
            indoor_temps = linspace(21, 26, 60);
            [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
        end

        % 在 TrustRegionBuilder.m 中
        function trust_scores = query_trust_scores(obj, grid_outdoor, grid_indoor, scenario)
            num_points = numel(grid_outdoor);
            fprintf('正在向量化查询 %d 个网格点的置信度...\n', num_points);

            % 【新方法：向量化】
            % 1. 将网格数据转换为列向量
            outdoor_temps_vec = reshape(grid_outdoor, num_points, 1);
            indoor_temps_vec = reshape(grid_indoor, num_points, 1);

            % 2. 构建一个包含所有点上下文的大table
            num_points = numel(grid_outdoor);
            context_table = table(...
                repmat(scenario.HourOfDay, num_points, 1), ...
                repmat(scenario.isWeekday, num_points, 1), ...
                outdoor_temps_vec, ...
                indoor_temps_vec, ...
                repmat(scenario.Occupant_Count, num_points, 1), ...
                'VariableNames', {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});

            % 3. 对每个动作模型执行一次批量预测
            model_keys = keys(obj.behavior_models);
            all_probs = zeros(num_points, length(model_keys)); % 预分配矩阵

            for k = 1:length(model_keys)
                model = obj.behavior_models(model_keys{k});
                all_probs(:, k) = predict(model, context_table);
            end

            % 4. 找到每个点的最大接受概率
            max_probs_vec = max(all_probs, [], 2);

            % 5. 将结果向量重新整形为网格矩阵
            trust_scores = reshape(max_probs_vec, size(grid_outdoor));

            fprintf('置信度查询完成。\n');
        end
        
        function gpr_result = fit_gpr_surface(obj, grid_outdoor, grid_indoor, trust_scores)
            % 准备GPR训练数据
            X1_vec = reshape(grid_outdoor, [], 1);
            X2_vec = reshape(grid_indoor, [], 1);
            gpr_X = [X1_vec, X2_vec];
            gpr_y = reshape(trust_scores, [], 1);
            
            % 训练GPR模型
            fprintf('正在训练GPR模型进行曲面平滑...\n');
            gpr_result = obj.gpr_engine.fit(gpr_X, gpr_y);
            
            % 生成平滑的预测曲面
            [Z_smooth, ~] = obj.gpr_engine.predict(gpr_X);
            Z_smooth_grid = reshape(Z_smooth, size(grid_outdoor));
            
            gpr_result.smooth_surface = Z_smooth_grid;
            gpr_result.grid_outdoor = grid_outdoor;
            gpr_result.grid_indoor = grid_indoor;
            
            fprintf('GPR曲面拟合完成。\n');
        end
        
        function boundaries = extract_trust_boundaries(obj, gpr_result, thresholds)
            boundaries = struct();
            
            for i = 1:length(thresholds)
                threshold = thresholds(i);
                [C, ~] = contour(gpr_result.grid_outdoor(1,:), gpr_result.grid_indoor(:,1), ...
                               gpr_result.smooth_surface, [threshold, threshold]);
                
                if ~isempty(C)
                    boundaries.(sprintf('threshold_%.0f', threshold*100)) = C;
                end
            end
        end
    end
end