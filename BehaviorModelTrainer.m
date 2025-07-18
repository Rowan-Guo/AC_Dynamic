%% ==========================================================================
%% 4. 行为模型训练器：BehaviorModelTrainer.m
%% ==========================================================================
classdef BehaviorModelTrainer < handle
    properties
        actions = [0, 0.5, 1.0];  % 动作空间：设定点调整值
        context_features = {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'};
        models
        performance
    end
    
    methods
        function obj = BehaviorModelTrainer()
            obj.models = containers.Map();
        end
        
        function [models, performance] = train_contextual_bandits(obj, context_data, full_data)
            % 生成概率化奖励
            rewards = obj.generate_probabilistic_rewards(context_data, full_data);
            
            % 为每个动作训练独立的模型
            performance = struct();
            
            for action_idx = 2:length(obj.actions) % 跳过动作1（无调整）
                action_name = sprintf('Action_%d', action_idx);
                fprintf('正在训练%s模型 (设定点+%.1f°C)...\n', action_name, obj.actions(action_idx));
                
                % 训练模型
                model = obj.train_single_action_model(context_data, rewards(:, action_idx));
                obj.models(action_name) = model;
                
                % 评估模型
                perf = obj.evaluate_single_action_model(model, context_data, rewards(:, action_idx));
                performance.(action_name) = perf;
                
                % 【修改】打印新的回归指标
                fprintf('%s模型 R-squared: %.3f, RMSE: %.4f\n', action_name, perf.r_squared, perf.rmse);
            end
            
            models = obj.models;
        end
        
        function model = train_single_action_model(obj, context_data, rewards)
            % 数据划分
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);

            X_train = context_data(training(cv), :);
            Y_train = rewards(training(cv));

            % 【原代码 - 删除】
            % model = fitctree(X_train, Y_train, ...
            %     'PredictorNames', obj.context_features, ...
            %     'MaxNumSplits', 50);

            % 【新代码 - 回归模型】
            model = fitrensemble(X_train, Y_train, ...
                'Method', 'Bag', ...
                'NumLearningCycles', 20, ...
                'Learners', 'tree');
        end

        % 在 BehaviorModelTrainer.m 中
        function performance = evaluate_single_action_model(obj, model, context_data, rewards)
            % 数据划分（与训练保持一致）
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);

            X_test = context_data(test(cv), :);
            Y_test = rewards(test(cv));

            % 进行预测
            Y_pred = predict(model, X_test);

            % 【修改】计算回归性能指标，而不是分类准确率
            ss_res = sum((Y_test - Y_pred).^2);
            ss_tot = sum((Y_test - mean(Y_test)).^2);
            r_squared = 1 - ss_res / ss_tot; % R-squared
            rmse = sqrt(mean((Y_test - Y_pred).^2)); % RMSE

            performance = struct('r_squared', r_squared, 'rmse', rmse);
        end

        function rewards = generate_probabilistic_rewards(obj, context_data, full_data)
            num_samples = height(context_data);
            num_actions = length(obj.actions);
            rewards = zeros(num_samples, num_actions);

            % 动作1（无调整）总是获得奖励
            rewards(:, 1) = 1;

            % 【关键修改】为动作2和3生成确定性奖励
            for action_idx = 2:num_actions
                p_accept = obj.compute_acceptance_probability(context_data, obj.actions(action_idx));

                % 【原代码 - 删除】
                % rewards(:, action_idx) = rand(num_samples, 1) < p_accept;

                % 【新代码 - 确定性奖励】
                rewards(:, action_idx) = p_accept;  % 直接使用概率作为连续标签
            end

            fprintf('确定性奖励生成完成。奖励统计:\n');
            for i = 1:num_actions
                fprintf('  动作%d (+%.1f°C): 平均概率 %.3f\n', i, obj.actions(i), mean(rewards(:, i)));
            end
        end
        
        function p_accept = compute_acceptance_probability(obj, context_data, action_magnitude)
            % 【最终优化版本】基于验证结果的精准调优

            % 进一步降低基础不适度，提升整体接受概率
            base_discomfort = 0.1 + 0.15 * action_magnitude;  % 从0.2+0.2进一步降低到0.1+0.15

            % 更温和的温度惩罚
            temp_penalty = 0.08 * max(0, context_data.Outdoor_Temp_C - 27) + ...  % 阈值从25提升到27
                0.12 * max(0, context_data.Avg_Zone_Temp_C - 24.5);     % 阈值从24提升到24.5

            % 添加时间因子奖励（工作时间外更宽松）
            % 添加时间因子奖励（工作时间外更宽松）
            time_bonus = zeros(height(context_data), 1); % 初始化为一个全零列向量

            % 创建一个逻辑掩码（logical mask），找出所有满足条件的行
            off_hours_mask = (~context_data.isWeekday | context_data.HourOfDay < 8 | context_data.HourOfDay > 18);

            % 使用逻辑掩码为所有满足条件的行赋值
            time_bonus(off_hours_mask) = -0.1; % 非工作时间降低不适度

            % 添加占用人数因子（人少时更宽松）
            % 添加占用人数因子（人少时更宽松）
            occupancy_bonus = zeros(height(context_data), 1); % 初始化为一个全零列向量
            low_occupancy_mask = (context_data.Occupant_Count < 50);
            occupancy_bonus(low_occupancy_mask) = -0.05; % 人少时降低不适度

            discomfort = base_discomfort + temp_penalty + time_bonus + occupancy_bonus;
            p_accept = 1 ./ (1 + exp(discomfort));

            % 确保概率在合理范围内
            p_accept = max(0.05, min(0.95, p_accept));  % 限制在[0.05, 0.95]范围
        end
    end
end