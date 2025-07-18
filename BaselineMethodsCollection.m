%% ==========================================================================
%% 2. 基线方法集合：BaselineMethodsCollection.m
%% ==========================================================================
classdef BaselineMethodsCollection < handle
    properties
        outdoor_temp_range = [15, 35]
        indoor_temp_range = [21, 26]
        grid_resolution = [50, 40]
    end
    
    methods
        function result = static_comfort_zone_method(obj, scenario)
            % 基线方法1: 传统静态舒适区
            % 基于ASHRAE标准定义固定的可调控域
            
            [grid_outdoor, grid_indoor] = obj.create_standard_grid();
            
            % 静态舒适区规则:
            % 1. 室内温度在22-25°C之间
            % 2. 室外温度小于30°C
            % 3. 温差不超过5°C
            comfort_mask = (grid_indoor >= 22) & (grid_indoor <= 25) & ...
                          (grid_outdoor <= 30) & ...
                          (abs(grid_outdoor - grid_indoor) <= 5);
            
            % 转换为置信度分数 (二值化)
            trust_scores = double(comfort_mask) * 0.8; % 假设80%置信度
            
            result = struct(...
                'method_name', 'Static Comfort Zone', ...
                'grid_outdoor', grid_outdoor, ...
                'grid_indoor', grid_indoor, ...
                'trust_scores', trust_scores, ...
                'adjustable_area', sum(comfort_mask(:)), ...
                'max_confidence', max(trust_scores(:)), ...
                'mean_confidence', mean(trust_scores(:)));
        end
        
        function result = simple_rule_based_method(obj, scenario)
            % 基线方法2: 基于温度的简单IF-THEN规则
            
            [grid_outdoor, grid_indoor] = obj.create_standard_grid();
            
            % 简单规则:
            % IF 室外温度 < 25°C THEN 高置信度
            % IF 25°C <= 室外温度 < 30°C THEN 中置信度  
            % IF 室外温度 >= 30°C THEN 低置信度
            % 同时考虑室内温度的影响
            
            trust_scores = zeros(size(grid_outdoor));
            
            for i = 1:size(grid_outdoor, 1)
                for j = 1:size(grid_outdoor, 2)
                    outdoor_t = grid_outdoor(i, j);
                    indoor_t = grid_indoor(i, j);
                    
                    if outdoor_t < 25
                        base_confidence = 0.7;
                    elseif outdoor_t < 30
                        base_confidence = 0.5;
                    else
                        base_confidence = 0.2;
                    end
                    
                    % 室内温度修正
                    if indoor_t < 22 || indoor_t > 25
                        base_confidence = base_confidence * 0.7;
                    end
                    
                    % 工作时间修正
                    if scenario.isWeekday && (scenario.HourOfDay >= 9) && (scenario.HourOfDay <= 17)
                        base_confidence = base_confidence * 0.9; % 工作时间更保守
                    end
                    
                    trust_scores(i, j) = base_confidence;
                end
            end
            
            result = struct(...
                'method_name', 'Simple Rule-Based', ...
                'grid_outdoor', grid_outdoor, ...
                'grid_indoor', grid_indoor, ...
                'trust_scores', trust_scores, ...
                'adjustable_area', sum(trust_scores(:) > 0.4), ...
                'max_confidence', max(trust_scores(:)), ...
                'mean_confidence', mean(trust_scores(:)));
        end
        
        function result = linear_regression_method(obj, scenario, context_features, processed_data)
            % 基线方法3: 简单线性回归方法
            
            % 准备训练数据 (使用历史接受率作为目标)
            X_train = table2array(context_features);
            
            % 模拟历史接受率 (简化版本)
            y_simulated = obj.simulate_acceptance_rate(X_train);
            
            % 训练线性回归模型
            mdl = fitlm(X_train, y_simulated);
            
            % 在网格上进行预测
            [grid_outdoor, grid_indoor] = obj.create_standard_grid();
            trust_scores = zeros(size(grid_outdoor));
            
            for i = 1:size(grid_outdoor, 1)
                for j = 1:size(grid_outdoor, 2)
                    context_point = [scenario.HourOfDay, double(scenario.isWeekday), ...
                                   grid_outdoor(i,j), grid_indoor(i,j), scenario.Occupant_Count];
                    predicted_acceptance = predict(mdl, context_point);
                    trust_scores(i, j) = max(0, min(1, predicted_acceptance)); % 限制在[0,1]
                end
            end
            
            result = struct(...
                'method_name', 'Linear Regression', ...
                'model', mdl, ...
                'grid_outdoor', grid_outdoor, ...
                'grid_indoor', grid_indoor, ...
                'trust_scores', trust_scores, ...
                'adjustable_area', sum(trust_scores(:) > 0.4), ...
                'max_confidence', max(trust_scores(:)), ...
                'mean_confidence', mean(trust_scores(:)));
        end
        
        function [grid_outdoor, grid_indoor] = create_standard_grid(obj)
            outdoor_temps = linspace(obj.outdoor_temp_range(1), obj.outdoor_temp_range(2), obj.grid_resolution(1));
            indoor_temps = linspace(obj.indoor_temp_range(1), obj.indoor_temp_range(2), obj.grid_resolution(2));
            [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
        end
        
        function acceptance_rate = simulate_acceptance_rate(obj, context_data)
            % 基于简单假设模拟历史接受率
            % 这里使用线性组合作为基线
            
            % 归一化特征
            normalized_data = zscore(context_data);
            
            % 简单线性组合 + 噪声
            weights = [0.1, 0.3, -0.4, -0.2, 0.1]; % 手工设定的权重
            linear_combination = normalized_data * weights';
            
            % 通过sigmoid转换为概率
            acceptance_rate = 1 ./ (1 + exp(-linear_combination));
            
            % 添加噪声
            acceptance_rate = acceptance_rate + 0.1 * randn(size(acceptance_rate));
            acceptance_rate = max(0, min(1, acceptance_rate));
        end
    end
end