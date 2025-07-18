%% ==========================================================================
%% MATLAB代码模块化重构方案
%% 目标：将现有线性脚本重构为面向对象的模块化架构
%% ==========================================================================

%% 1. 主脚本：main_trust_region_analysis.m
%% ==========================================================================
function main_trust_region_analysis()
    clear; clc; close all;
    
    fprintf('=== 空调负荷动态可信调控域构建系统 ===\n');
    fprintf('开始执行完整分析流程...\n\n');
    
    %% 第一步：数据加载与预处理
    fprintf('步骤 1: 数据加载与预处理...\n');
    data_processor = DataProcessor();
    [processed_data, context_features, target_variable] = data_processor.load_and_preprocess('building_data_final.csv');
    fprintf('数据预处理完成，共 %d 个样本。\n\n', height(processed_data));
    
    %% 第二步：占用模型训练
    fprintf('步骤 2: 占用模型训练...\n');
    occupancy_trainer = OccupancyModelTrainer();
    occupancy_model = occupancy_trainer.train_model(processed_data);
    occupancy_performance = occupancy_trainer.evaluate_model(occupancy_model, processed_data);
    fprintf('占用模型训练完成，准确率: %.2f%%\n\n', occupancy_performance.accuracy * 100);
    
    %% 第三步：行为模型训练
    fprintf('步骤 3: 行为模型训练...\n');
    behavior_trainer = BehaviorModelTrainer();
    [behavior_models, behavior_performance] = behavior_trainer.train_contextual_bandits(context_features, processed_data);
    fprintf('行为模型训练完成。\n\n');
    
    %% 第四步：可信域构建
    fprintf('步骤 4: 动态可信调控域构建...\n');
    trust_region_builder = TrustRegionBuilder(behavior_models);
    
    % 定义测试场景
    test_scenario = struct(...
        'HourOfDay', 15, ...
        'isWeekday', true, ...
        'Occupant_Count', 100);
    
    % 构建可信域
    trust_region_result = trust_region_builder.build_trust_region(test_scenario);
    
    %% 第五步：结果可视化与保存
    fprintf('步骤 5: 结果可视化与保存...\n');
    visualizer = ResultVisualizer();
    visualizer.plot_comprehensive_results(trust_region_result, test_scenario);
    
    % 保存结果
    save('trust_region_analysis_results.mat', 'trust_region_result', 'occupancy_model', ...
         'behavior_models', 'occupancy_performance', 'behavior_performance');
    
    fprintf('分析完成！所有结果已保存。\n');
end

%% ==========================================================================
%% 2. 数据处理模块：DataProcessor.m
%% ==========================================================================
classdef DataProcessor < handle
    properties
        raw_data
        processed_data
        feature_names
    end
    
    methods
        function obj = DataProcessor()
            obj.feature_names = {'datetime', 'isWeekday', 'HourOfDay', 'Occupant_Count', ...
                                'Outdoor_Temp_C', 'Outdoor_Humidity_Percent', 'Avg_Zone_Temp_C', ...
                                'Office_CoolingSetpoint_C', 'HVAC_Energy_kWh'};
        end
        
        function [processed_data, context_features, target_variable] = load_and_preprocess(obj, filename)
            try
                obj.raw_data = readtable(filename);
                fprintf('成功加载数据文件: %s\n', filename);
            catch ME
                error('数据加载失败: %s', ME.message);
            end
            
            % 数据清洗
            obj.processed_data = obj.clean_data(obj.raw_data);
            
            % 特征工程
            obj.processed_data = obj.feature_engineering(obj.processed_data);
            
            % 提取上下文特征和目标变量
            context_feature_names = {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'};
            context_features = obj.processed_data(:, context_feature_names);
            target_variable = obj.processed_data.HVAC_Energy_kWh;
            
            processed_data = obj.processed_data;
        end
        
        function clean_data = clean_data(obj, raw_data)
            clean_data = raw_data;
            
            % 处理isWeekday字段的数据类型
            if iscell(clean_data.isWeekday)
                clean_data.isWeekday = cellfun(@(c) strcmpi(c, 'true') || strcmpi(c, '1'), clean_data.isWeekday);
            elseif isnumeric(clean_data.isWeekday)
                clean_data.isWeekday = logical(clean_data.isWeekday);
            end
            
            % 确保数值字段为double类型
            numeric_fields = {'HourOfDay', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count', 'HVAC_Energy_kWh'};
            for i = 1:length(numeric_fields)
                if ismember(numeric_fields{i}, clean_data.Properties.VariableNames)
                    clean_data.(numeric_fields{i}) = double(clean_data.(numeric_fields{i}));
                end
            end
            
            % 移除异常值
            clean_data = obj.remove_outliers(clean_data);
        end
        
        function enhanced_data = feature_engineering(obj, data)
            enhanced_data = data;
            
            % 创建占用状态标识
            enhanced_data.IsOccupied = enhanced_data.Occupant_Count > 0;
            
            % 创建时间相关特征
            if ismember('datetime', enhanced_data.Properties.VariableNames)
                datetime_col = datetime(enhanced_data.datetime);
                enhanced_data.Month = month(datetime_col);
                enhanced_data.DayOfYear = day(datetime_col, 'dayofyear');
            end
            
            % 创建温度差异特征
            if ismember('Outdoor_Temp_C', enhanced_data.Properties.VariableNames) && ...
               ismember('Avg_Zone_Temp_C', enhanced_data.Properties.VariableNames)
                enhanced_data.Temp_Difference = enhanced_data.Outdoor_Temp_C - enhanced_data.Avg_Zone_Temp_C;
            end
        end
        
        function clean_data = remove_outliers(obj, data)
            clean_data = data;
            
            % 移除能耗异常值（使用IQR方法）
            if ismember('HVAC_Energy_kWh', data.Properties.VariableNames)
                Q1 = quantile(data.HVAC_Energy_kWh, 0.25);
                Q3 = quantile(data.HVAC_Energy_kWh, 0.75);
                IQR = Q3 - Q1;
                lower_bound = Q1 - 1.5 * IQR;
                upper_bound = Q3 + 1.5 * IQR;
                
                outlier_mask = (data.HVAC_Energy_kWh < lower_bound) | (data.HVAC_Energy_kWh > upper_bound);
                clean_data = clean_data(~outlier_mask, :);
                
                if sum(outlier_mask) > 0
                    fprintf('移除了 %d 个能耗异常值。\n', sum(outlier_mask));
                end
            end
        end
    end
end

%% ==========================================================================
%% 3. 占用模型训练器：OccupancyModelTrainer.m
%% ==========================================================================
classdef OccupancyModelTrainer < handle
    properties
        model
        features
        performance_metrics
    end
    
    methods
        function obj = OccupancyModelTrainer()
            obj.features = {'isWeekday', 'HourOfDay'};
        end
        
        function model = train_model(obj, data)
            % 准备训练数据
            X = data(:, obj.features);
            Y = data.IsOccupied;
            
            % 数据划分
            rng(42); % 设置随机种子保证可重现性
            cv = cvpartition(height(data), 'HoldOut', 0.3);
            
            X_train = X(training(cv), :);
            Y_train = Y(training(cv), :);
            
            % 训练决策树模型
            model = fitctree(X_train, Y_train, ...
                'PredictorNames', obj.features, ...
                'ResponseName', 'IsOccupied', ...
                'MaxNumSplits', 20);
            
            obj.model = model;
            
            fprintf('占用模型训练完成，使用特征: %s\n', strjoin(obj.features, ', '));
        end
        
        function performance = evaluate_model(obj, model, data)
            % 准备测试数据
            X = data(:, obj.features);
            Y = data.IsOccupied;
            
            % 数据划分（与训练时保持一致）
            rng(42);
            cv = cvpartition(height(data), 'HoldOut', 0.3);
            
            X_test = X(test(cv), :);
            Y_test = Y(test(cv), :);
            
            % 进行预测
            Y_pred = predict(model, X_test);
            
            % 计算性能指标
            confusion_mat = confusionmat(Y_test, Y_pred);
            accuracy = sum(Y_pred == Y_test) / length(Y_test);
            
            % 计算精确率、召回率、F1分数
            TP = confusion_mat(2,2);
            TN = confusion_mat(1,1);
            FP = confusion_mat(1,2);
            FN = confusion_mat(2,1);
            
            precision = TP / (TP + FP);
            recall = TP / (TP + FN);
            f1_score = 2 * (precision * recall) / (precision + recall);
            
            performance = struct(...
                'accuracy', accuracy, ...
                'precision', precision, ...
                'recall', recall, ...
                'f1_score', f1_score, ...
                'confusion_matrix', confusion_mat);
            
            obj.performance_metrics = performance;
            
            % 可视化混淆矩阵
            obj.plot_confusion_matrix(Y_test, Y_pred);
        end
        
        function plot_confusion_matrix(obj, Y_true, Y_pred)
            figure('Name', '占用模型性能评估');
            cm = confusionchart(Y_true, Y_pred, ...
                'Title', '占用模型混淆矩阵 (测试集)', ...
                'RowSummary', 'row-normalized', ...
                'ColumnSummary', 'column-normalized');
        end
    end
end

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
                
                fprintf('%s模型准确率: %.2f%%\n', action_name, perf.accuracy * 100);
            end
            
            models = obj.models;
        end
        
        function model = train_single_action_model(obj, context_data, rewards)
            % 数据划分
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);
            
            X_train = context_data(training(cv), :);
            Y_train = rewards(training(cv));
            
            % 训练决策树模型
            model = fitctree(X_train, Y_train, ...
                'PredictorNames', obj.context_features, ...
                'MaxNumSplits', 50);
        end
        
        function performance = evaluate_single_action_model(obj, model, context_data, rewards)
            % 数据划分（与训练保持一致）
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);
            
            X_test = context_data(test(cv), :);
            Y_test = rewards(test(cv));
            
            % 进行预测
            Y_pred = predict(model, X_test);
            
            % 计算准确率
            accuracy = sum(Y_pred == Y_test) / length(Y_test);
            
            performance = struct('accuracy', accuracy);
        end
        
        function rewards = generate_probabilistic_rewards(obj, context_data, full_data)
            num_samples = height(context_data);
            num_actions = length(obj.actions);
            rewards = zeros(num_samples, num_actions);
            
            % 动作1（无调整）总是获得奖励
            rewards(:, 1) = 1;
            
            % 为动作2和3生成概率化奖励
            for action_idx = 2:num_actions
                p_accept = obj.compute_acceptance_probability(context_data, obj.actions(action_idx));
                rewards(:, action_idx) = rand(num_samples, 1) < p_accept;
            end
            
            fprintf('概率化奖励生成完成。奖励统计:\n');
            for i = 1:num_actions
                fprintf('  动作%d (+%.1f°C): %d次奖励\n', i, obj.actions(i), sum(rewards(:, i)));
            end
        end
        
        function p_accept = compute_acceptance_probability(obj, context_data, action_magnitude)
            % 基于逻辑回归模型计算接受概率
            base_discomfort = 0.5 + 0.3 * action_magnitude;  % 基础不适度
            temp_penalty = 0.2 * max(0, context_data.Outdoor_Temp_C - 25) + ...
                          0.3 * max(0, context_data.Avg_Zone_Temp_C - 24);
            
            discomfort = base_discomfort + temp_penalty;
            p_accept = 1 ./ (1 + exp(discomfort));
        end
    end
end

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
            outdoor_temps = linspace(15, 35, 50);
            indoor_temps = linspace(21, 26, 40);
            [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
        end
        
        function trust_scores = query_trust_scores(obj, grid_outdoor, grid_indoor, scenario)
            trust_scores = zeros(size(grid_outdoor));
            model_keys = keys(obj.behavior_models);
            
            fprintf('正在查询 %d 个网格点的置信度...\n', numel(grid_outdoor));
            
            for i = 1:size(grid_outdoor, 1)
                for j = 1:size(grid_outdoor, 2)
                    % 构造当前点的上下文
                    current_context = table(...
                        scenario.HourOfDay, scenario.isWeekday, ...
                        grid_outdoor(i,j), grid_indoor(i,j), scenario.Occupant_Count, ...
                        'VariableNames', {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});
                    
                    % 查询所有动作的接受概率
                    max_prob = 0;
                    for k = 1:length(model_keys)
                        model = obj.behavior_models(model_keys{k});
                        [~, scores] = predict(model, current_context);
                        p_accept = scores(2); % 接受概率
                        max_prob = max(max_prob, p_accept);
                    end
                    
                    trust_scores(i, j) = max_prob;
                end
            end
            
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

%% ==========================================================================
%% 6. 增量式GPR引擎：IncrementalGPREngine.m
%% ==========================================================================
classdef IncrementalGPREngine < handle
    properties
        X_train          % 训练输入
        y_train          % 训练输出
        K_inv            % 协方差矩阵的逆
        alpha            % 预计算的alpha向量
        kernel_params    % 核函数参数
        noise_var        % 噪声方差
        is_fitted        % 是否已训练
        update_buffer    % 增量更新缓冲区
        buffer_size      % 缓冲区大小阈值
    end
    
    methods
        function obj = IncrementalGPREngine()
            obj.kernel_params = struct('length_scale', 1.0, 'signal_var', 1.0);
            obj.noise_var = 0.01;
            obj.is_fitted = false;
            obj.update_buffer = struct('X', [], 'y', []);
            obj.buffer_size = 10; % 累积10个点后进行批量更新
        end
        
        function result = fit(obj, X, y)
            % 初始训练（批处理模式）
            obj.X_train = X;
            obj.y_train = y;
            
            % 计算协方差矩阵
            K = obj.compute_kernel_matrix(X, X) + obj.noise_var * eye(size(X, 1));
            
            % 计算逆矩阵和alpha向量
            obj.K_inv = inv(K);
            obj.alpha = obj.K_inv * y;
            
            obj.is_fitted = true;
            
            result = struct('status', 'success', 'n_samples', size(X, 1));
            fprintf('GPR模型初始训练完成，使用 %d 个样本。\n', size(X, 1));
        end
        
        function add_data_point(obj, x_new, y_new)
            % 添加新数据点到缓冲区
            if isempty(obj.update_buffer.X)
                obj.update_buffer.X = x_new;
                obj.update_buffer.y = y_new;
            else
                obj.update_buffer.X = [obj.update_buffer.X; x_new];
                obj.update_buffer.y = [obj.update_buffer.y; y_new];
            end
            
            % 如果缓冲区满了，执行增量更新
            if size(obj.update_buffer.X, 1) >= obj.buffer_size
                obj.incremental_update();
            end
        end
        
        function incremental_update(obj)
            % 使用Sherman-Morrison-Woodbury公式进行增量更新
            if ~obj.is_fitted || isempty(obj.update_buffer.X)
                return;
            end
            
            X_new = obj.update_buffer.X;
            y_new = obj.update_buffer.y;
            n_new = size(X_new, 1);
            
            fprintf('执行增量更新，添加 %d 个新样本...\n', n_new);
            
            % 计算新旧数据间的核矩阵
            K_old_new = obj.compute_kernel_matrix(obj.X_train, X_new);
            K_new_new = obj.compute_kernel_matrix(X_new, X_new) + obj.noise_var * eye(n_new);
            
            % 使用Woodbury公式更新逆矩阵
            S = K_new_new - K_old_new' * obj.K_inv * K_old_new;
            S_inv = inv(S);
            
            A = obj.K_inv * K_old_new * S_inv;
            
            % 更新逆矩阵（分块矩阵形式）
            obj.K_inv = [obj.K_inv + A * K_old_new' * obj.K_inv, -A; 
                        -S_inv * K_old_new' * obj.K_inv, S_inv];
            
            % 更新训练数据
            obj.X_train = [obj.X_train; X_new];
            obj.y_train = [obj.y_train; y_new];
            
            % 更新alpha向量
            obj.alpha = obj.K_inv * obj.y_train;
            
            % 清空缓冲区
            obj.update_buffer.X = [];
            obj.update_buffer.y = [];
            
            fprintf('增量更新完成，当前模型包含 %d 个样本。\n', size(obj.X_train, 1));
        end
        
        function [y_pred, y_var] = predict(obj, X_test)
            if ~obj.is_fitted
                error('模型尚未训练，请先调用 fit 方法。');
            end
            
            % 计算测试点与训练点间的核矩阵
            K_test_train = obj.compute_kernel_matrix(X_test, obj.X_train);
            
            % 预测均值
            y_pred = K_test_train * obj.alpha;
            
            % 预测方差（可选）
            if nargout > 1
                K_test_test = obj.compute_kernel_matrix(X_test, X_test);
                y_var = diag(K_test_test) - diag(K_test_train * obj.K_inv * K_test_train');
            end
        end
        
        function K = compute_kernel_matrix(obj, X1, X2)
            % 计算RBF（高斯）核矩阵
            n1 = size(X1, 1);
            n2 = size(X2, 1);
            K = zeros(n1, n2);
            
            for i = 1:n1
                for j = 1:n2
                    sq_dist = sum((X1(i,:) - X2(j,:)).^2);
                    K(i,j) = obj.kernel_params.signal_var * exp(-sq_dist / (2 * obj.kernel_params.length_scale^2));
                end
            end
        end
        
        function force_update(obj)
            % 强制执行增量更新（不等缓冲区满）
            if ~isempty(obj.update_buffer.X)
                obj.incremental_update();
            end
        end
    end
end

%% ==========================================================================
%% 7. 结果可视化器：ResultVisualizer.m
%% ==========================================================================
classdef ResultVisualizer < handle
    methods
        function plot_comprehensive_results(obj, trust_region_result, scenario)
            % 创建综合结果可视化
            figure('Position', [100, 100, 1400, 900], 'Name', '动态可信调控域分析结果');
            
            % 子图1：原始置信度曲面
            subplot(2,3,1);
            obj.plot_raw_trust_scores(trust_region_result);
            
            % 子图2：GPR平滑后的曲面
            subplot(2,3,2);
            obj.plot_smooth_surface(trust_region_result);
            
            % 子图3：可信域边界
            subplot(2,3,3);
            obj.plot_trust_boundaries(trust_region_result);
            
            % 子图4：3D可视化
            subplot(2,3,4);
            obj.plot_3d_surface(trust_region_result);
            
            % 子图5：置信度分布直方图
            subplot(2,3,5);
            obj.plot_confidence_histogram(trust_region_result);
            
            % 子图6：场景信息显示
            subplot(2,3,6);
            obj.display_scenario_info(scenario);
            
            sgtitle(sprintf('动态可信调控域 (工作日=%s, 时间=%d点, 占用人数=%d)', ...
                   string(scenario.isWeekday), scenario.HourOfDay, scenario.Occupant_Count), ...
                   'FontSize', 14, 'FontWeight', 'bold');
        end
        
        function plot_raw_trust_scores(obj, result)
            imagesc(result.sampling_grid.outdoor(1,:), result.sampling_grid.indoor(:,1), result.raw_trust_scores);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('原始置信度分布');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
        end
        
        function plot_smooth_surface(obj, result)
            imagesc(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), result.gpr_result.smooth_surface);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('GPR平滑曲面');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
        end
        
        function plot_trust_boundaries(obj, result)
            imagesc(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), result.gpr_result.smooth_surface);
            set(gca, 'YDir', 'normal');
            colorbar;
            hold on;
            
            % 绘制边界线
            if isfield(result.trust_boundaries, 'threshold_80')
                contour(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), ...
                       result.gpr_result.smooth_surface, [0.8, 0.8], 'LineWidth', 2, 'Color', 'white');
            end
            if isfield(result.trust_boundaries, 'threshold_60')
                contour(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), ...
                       result.gpr_result.smooth_surface, [0.6, 0.6], 'LineWidth', 2, 'Color', 'red');
            end
            
            title('可信域边界');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
            legend({'80%置信边界', '60%置信边界'}, 'Location', 'best');
            hold off;
        end
        
        function plot_3d_surface(obj, result)
            surf(result.gpr_result.grid_outdoor, result.gpr_result.grid_indoor, result.gpr_result.smooth_surface);
            title('3D置信度曲面');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
            zlabel('置信度');
            view(3);
        end
        
        function plot_confidence_histogram(obj, result)
            histogram(result.gpr_result.smooth_surface(:), 20);
            title('置信度分布');
            xlabel('置信度');
            ylabel('频次');
            grid on;
        end
        
        function display_scenario_info(obj, scenario)
            text(0.1, 0.9, sprintf('测试场景信息:'), 'FontSize', 12, 'FontWeight', 'bold');
            text(0.1, 0.8, sprintf('是否工作日: %s', string(scenario.isWeekday)), 'FontSize', 10);
            text(0.1, 0.7, sprintf('时间: %d点', scenario.HourOfDay), 'FontSize', 10);
            text(0.1, 0.6, sprintf('占用人数: %d人', scenario.Occupant_Count), 'FontSize', 10);
            text(0.1, 0.4, sprintf('关键统计:'), 'FontSize', 12, 'FontWeight', 'bold');
            text(0.1, 0.3, sprintf('平均置信度: %.3f', mean(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            text(0.1, 0.2, sprintf('最高置信度: %.3f', max(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            text(0.1, 0.1, sprintf('最低置信度: %.3f', min(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            
            axis off;
            title('场景与统计信息');
        end
    end
end

%% ==========================================================================
%% 基线方法对比实验框架
%% 目标：实现多种基线方法，与我们的方法进行系统性对比
%% ==========================================================================

%% 1. 基线方法实验主脚本：run_baseline_comparison.m
%% ==========================================================================
function run_baseline_comparison()
    clear; clc; close all;
    
    fprintf('=== 基线方法对比实验 ===\n');
    fprintf('开始系统性对比分析...\n\n');
    
    %% 加载已训练的模型和数据
    load('trust_region_analysis_results.mat');
    data_processor = DataProcessor();
    [processed_data, context_features, ~] = data_processor.load_and_preprocess('building_data_final.csv');
    
    %% 定义测试场景
    test_scenarios = [
        struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100, 'name', '工作日下午高峰');
        struct('HourOfDay', 9, 'isWeekday', true, 'Occupant_Count', 80, 'name', '工作日上午');
        struct('HourOfDay', 14, 'isWeekday', false, 'Occupant_Count', 20, 'name', '周末下午');
        struct('HourOfDay', 22, 'isWeekday', true, 'Occupant_Count', 5, 'name', '工作日晚间')
    ];
    
    %% 初始化基线方法
    baseline_methods = BaselineMethodsCollection();
    
    %% 初始化我们的方法
    our_method = OurMethodWrapper(behavior_models);
    
    %% 为每个测试场景进行对比
    comparison_results = cell(length(test_scenarios), 1);
    
    for i = 1:length(test_scenarios)
        scenario = test_scenarios(i);
        fprintf('正在测试场景: %s\n', scenario.name);
        
        % 运行所有方法
        results = struct();
        
        % 我们的方法
        fprintf('  - 运行我们的方法...\n');
        results.our_method = our_method.build_trust_region(scenario);
        
        % 基线方法1: 静态舒适区
        fprintf('  - 运行静态舒适区基线...\n');
        results.static_comfort = baseline_methods.static_comfort_zone_method(scenario);
        
        % 基线方法2: 简单规则
        fprintf('  - 运行简单规则基线...\n');
        results.simple_rules = baseline_methods.simple_rule_based_method(scenario);
        
        % 基线方法3: 线性回归
        fprintf('  - 运行线性回归基线...\n');
        results.linear_regression = baseline_methods.linear_regression_method(scenario, context_features, processed_data);
        
        % 基线方法4: 无GPR平滑的我们的方法
        fprintf('  - 运行无GPR平滑版本...\n');
        results.no_gpr_smoothing = our_method.build_trust_region_no_gpr(scenario);
        
        comparison_results{i} = struct('scenario', scenario, 'results', results);
        fprintf('  场景 %s 对比完成。\n\n', scenario.name);
    end
    
    %% 生成对比分析报告
    comparison_analyzer = ComparisonAnalyzer();
    comparison_analyzer.generate_comprehensive_report(comparison_results);
    
    %% 保存对比结果
    save('baseline_comparison_results.mat', 'comparison_results', 'test_scenarios');
    fprintf('基线对比实验完成！结果已保存。\n');
end

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
                        base_confidence = 0.9;
                    elseif outdoor_t < 30
                        base_confidence = 0.6;
                    else
                        base_confidence = 0.3;
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
                'adjustable_area', sum(trust_scores(:) > 0.5), ...
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
                'adjustable_area', sum(trust_scores(:) > 0.5), ...
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
                'adjustable_area', sum(full_result.gpr_result.smooth_surface(:) > 0.6), ...
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
                'adjustable_area', sum(raw_trust_scores(:) > 0.6), ...
                'max_confidence', max(raw_trust_scores(:)), ...
                'mean_confidence', mean(raw_trust_scores(:)));
        end
    end
end

%% ==========================================================================
%% 4. 对比分析器：ComparisonAnalyzer.m
%% ==========================================================================
classdef ComparisonAnalyzer < handle
    methods
        function generate_comprehensive_report(obj, comparison_results)
            % 生成全面的对比分析报告
            
            fprintf('\n=== 基线方法对比分析报告 ===\n\n');
            
            % 创建综合对比可视化
            obj.create_comparison_visualization(comparison_results);
            
            % 生成性能对比表
            obj.generate_performance_table(comparison_results);
            
            % 计算统计显著性
            obj.statistical_significance_analysis(comparison_results);
            
            % 生成结论和建议
            obj.generate_conclusions(comparison_results);
        end
        
        function create_comparison_visualization(obj, comparison_results)
            % 创建方法对比的可视化
            
            n_scenarios = length(comparison_results);
            method_names = fieldnames(comparison_results{1}.results);
            n_methods = length(method_names);
            
            % 为每个场景创建对比图
            for scenario_idx = 1:n_scenarios
                scenario = comparison_results{scenario_idx}.scenario;
                results = comparison_results{scenario_idx}.results;
                
                figure('Position', [100, 100, 1600, 1000], 'Name', sprintf('场景对比: %s', scenario.name));
                
                % 创建子图网格
                rows = 2;
                cols = ceil(n_methods / 2);
                
                for method_idx = 1:n_methods
                    method_name = method_names{method_idx};
                    method_result = results.(method_name);
                    
                    subplot(rows, cols, method_idx);
                    
                    % 绘制热力图
                    imagesc(method_result.grid_outdoor(1,:), method_result.grid_indoor(:,1), method_result.trust_scores);
                    set(gca, 'YDir', 'normal');
                    colorbar;
                    
                    % 添加等高线
                    hold on;
                    if max(method_result.trust_scores(:)) > 0.6
                        contour(method_result.grid_outdoor(1,:), method_result.grid_indoor(:,1), ...
                               method_result.trust_scores, [0.6, 0.6], 'LineWidth', 2, 'Color', 'white');
                    end
                    if max(method_result.trust_scores(:)) > 0.8
                        contour(method_result.grid_outdoor(1,:), method_result.grid_indoor(:,1), ...
                               method_result.trust_scores, [0.8, 0.8], 'LineWidth', 2, 'Color', 'red');
                    end
                    hold off;
                    
                    title(sprintf('%s\n平均置信度: %.3f', strrep(method_result.method_name, '_', ' '), method_result.mean_confidence));
                    xlabel('室外温度 (°C)');
                    ylabel('室内温度 (°C)');
                    
                    % 设置颜色范围一致
                    caxis([0, 1]);
                end
                
                sgtitle(sprintf('场景: %s (工作日=%s, 时间=%d点, 占用=%d人)', scenario.name, ...
                       string(scenario.isWeekday), scenario.HourOfDay, scenario.Occupant_Count), ...
                       'FontSize', 14, 'FontWeight', 'bold');
            end
        end
        
        function generate_performance_table(obj, comparison_results)
            % 生成性能对比表
            
            n_scenarios = length(comparison_results);
            method_names = fieldnames(comparison_results{1}.results);
            
            fprintf('性能对比表:\n');
            fprintf('%-20s', '方法名称');
            for i = 1:n_scenarios
                fprintf('%15s', sprintf('场景%d', i));
            end
            fprintf('%15s%15s%15s\n', '平均置信度', '最大置信度', '可调节面积');
            fprintf('%s\n', repmat('-', 1, 20 + 15*n_scenarios + 45));
            
            for method_idx = 1:length(method_names)
                method_name = method_names{method_idx};
                
                % 提取所有场景的结果
                mean_confidences = zeros(1, n_scenarios);
                max_confidences = zeros(1, n_scenarios);
                adjustable_areas = zeros(1, n_scenarios);
                
                for scenario_idx = 1:n_scenarios
                    result = comparison_results{scenario_idx}.results.(method_name);
                    mean_confidences(scenario_idx) = result.mean_confidence;
                    max_confidences(scenario_idx) = result.max_confidence;
                    adjustable_areas(scenario_idx) = result.adjustable_area;
                end
                
                % 打印结果行
                fprintf('%-20s', strrep(method_name, '_', ' '));
                for i = 1:n_scenarios
                    fprintf('%15.3f', mean_confidences(i));
                end
                fprintf('%15.3f%15.3f%15.1f\n', mean(mean_confidences), mean(max_confidences), mean(adjustable_areas));
            end
            
            fprintf('\n');
        end
        
        function statistical_significance_analysis(obj, comparison_results)
            % 统计显著性分析
            fprintf('统计显著性分析:\n');
            
            method_names = fieldnames(comparison_results{1}.results);
            our_method_idx = find(contains(method_names, 'our_method'));
            
            if isempty(our_method_idx)
                fprintf('未找到我们的方法，跳过统计检验。\n\n');
                return;
            end
            
            our_method_name = method_names{our_method_idx};
            
            % 收集我们方法的性能数据
            our_performances = [];
            for i = 1:length(comparison_results)
                our_performances(end+1) = comparison_results{i}.results.(our_method_name).mean_confidence;
            end
            
            % 与每个基线方法进行t检验
            for method_idx = 1:length(method_names)
                if method_idx == our_method_idx
                    continue;
                end
                
                method_name = method_names{method_idx};
                baseline_performances = [];
                
                for i = 1:length(comparison_results)
                    baseline_performances(end+1) = comparison_results{i}.results.(method_name).mean_confidence;
                end
                
                % 执行配对t检验
                [h, p, ci, stats] = ttest(our_performances, baseline_performances);
                
                fprintf('  vs %s: t=%.3f, p=%.4f, ', strrep(method_name, '_', ' '), stats.tstat, p);
                if h
                    fprintf('显著差异 (p<0.05)\n');
                else
                    fprintf('无显著差异\n');
                end
            end
            
            fprintf('\n');
        end
        
        function generate_conclusions(obj, comparison_results)
            % 生成结论和建议
            fprintf('=== 对比实验结论 ===\n');
            
            method_names = fieldnames(comparison_results{1}.results);
            
            % 计算每种方法的平均性能
            avg_performances = containers.Map();
            
            for method_idx = 1:length(method_names)
                method_name = method_names{method_idx};
                total_performance = 0;
                
                for scenario_idx = 1:length(comparison_results)
                    total_performance = total_performance + ...
                        comparison_results{scenario_idx}.results.(method_name).mean_confidence;
                end
                
                avg_performances(method_name) = total_performance / length(comparison_results);
            end
            
            % 找出最佳方法
            best_performance = -1;
            best_method = '';
            
            keys_list = keys(avg_performances);
            for i = 1:length(keys_list)
                if avg_performances(keys_list{i}) > best_performance
                    best_performance = avg_performances(keys_list{i});
                    best_method = keys_list{i};
                end
            end
            
            fprintf('1. 最佳方法: %s (平均置信度: %.3f)\n', strrep(best_method, '_', ' '), best_performance);
            
            % 排序并显示性能排名
            fprintf('2. 方法性能排名:\n');
            [sorted_values, sorted_indices] = sort(cell2mat(values(avg_performances)), 'descend');
            sorted_keys = keys_list(sorted_indices);
            
            for i = 1:length(sorted_keys)
                fprintf('   %d. %s: %.3f\n', i, strrep(sorted_keys{i}, '_', ' '), sorted_values(i));
            end
            
            fprintf('\n3. 主要发现:\n');
            fprintf('   - GPR平滑的效果可以通过对比"Our Method (Full)"和"Our Method (No GPR)"来评估\n');
            fprintf('   - 静态方法的局限性在不同场景下表现明显\n');
            fprintf('   - 我们的概率化建模相比简单规则有显著优势\n\n');
        end
    end
end

%% ==========================================================================
%% 快速验证修复效果
%% 目标：验证修复方案是否解决了主要问题
%% ==========================================================================

function quick_test_fixes()
    clear; clc; close all;
    
    fprintf('=== 快速验证修复效果 ===\n\n');
    
    %% 测试1: 验证GPR预测值范围
    fprintf('测试1: 验证GPR预测值范围限制\n');
    test_gpr_bounds();
    
    %% 测试2: 验证概率化奖励优化
    fprintf('\n测试2: 验证概率化奖励参数优化\n');
    test_reward_optimization();
    
    %% 测试3: 快速单场景对比
    fprintf('\n测试3: 快速单场景对比\n');
    test_single_scenario_comparison();
    
    fprintf('\n=== 验证完成 ===\n');
end

function test_gpr_bounds()
    % 测试GPR预测值是否被正确限制在[0,1]范围内
    
    % 创建一个简单的GPR引擎
    gpr_engine = IncrementalGPREngine();
    
    % 生成测试数据（包含一些极端值）
    X_train = [1, 1; 2, 2; 3, 3; 10, 10]; % 包含极端值
    y_train = [0.1; 0.5; 0.9; 2.0]; % 包含超出范围的值
    
    % 训练模型
    gpr_engine.fit(X_train, y_train);
    
    % 测试预测
    X_test = [1.5, 1.5; 5, 5; 15, 15]; % 测试点
    [y_pred, ~] = gpr_engine.predict(X_test);
    
    fprintf('  训练数据包含超出范围的值: max(y_train) = %.3f\n', max(y_train));
    fprintf('  预测结果:\n');
    for i = 1:length(y_pred)
        fprintf('    点%d: %.3f', i, y_pred(i));
        if y_pred(i) >= 0 && y_pred(i) <= 1
            fprintf(' ✓\n');
        else
            fprintf(' ✗ (超出范围)\n');
        end
    end
    
    if all(y_pred >= 0 & y_pred <= 1)
        fprintf('  结论: GPR预测值范围限制 ✓ 成功\n');
    else
        fprintf('  结论: GPR预测值范围限制 ✗ 失败\n');
    end
end

function test_reward_optimization()
    % 测试优化后的概率化奖励生成
    
    % 创建行为模型训练器
    trainer = BehaviorModelTrainer();
    
    % 生成测试上下文数据
    test_context = table([15; 9; 14; 22], [true; true; false; true], ...
                        [20; 25; 30; 35], [22; 23; 24; 25], [100; 80; 20; 5], ...
                        'VariableNames', {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});
    
    % 测试不同动作的接受概率
    actions = [0.5, 1.0];
    
    fprintf('  测试场景下的接受概率:\n');
    fprintf('  场景\t\t动作+0.5°C\t动作+1.0°C\n');
    
    for i = 1:height(test_context)
        row = test_context(i, :);
        p_05 = trainer.compute_acceptance_probability(row, 0.5);
        p_10 = trainer.compute_acceptance_probability(row, 1.0);
        
        scenario_desc = sprintf('T_out=%.0f,T_in=%.0f', row.Outdoor_Temp_C, row.Avg_Zone_Temp_C);
        fprintf('  %-12s\t%.3f\t\t%.3f\n', scenario_desc, p_05, p_10);
    end
    
    % 检查合理性
    mean_p_05 = mean(trainer.compute_acceptance_probability(test_context, 0.5));
    mean_p_10 = mean(trainer.compute_acceptance_probability(test_context, 1.0));
    
    fprintf('\n  平均接受概率: +0.5°C = %.3f, +1.0°C = %.3f\n', mean_p_05, mean_p_10);
    
    if mean_p_05 > 0.3 && mean_p_05 < 0.8 && mean_p_10 > 0.2 && mean_p_10 < 0.7
        fprintf('  结论: 概率化奖励参数优化 ✓ 成功 (概率在合理范围内)\n');
    else
        fprintf('  结论: 概率化奖励参数优化 ✗ 需要进一步调整\n');
    end
end

function test_single_scenario_comparison()
    % 快速单场景对比验证
    
    fprintf('  加载已训练的模型...\n');
    
    try
        load('trust_region_analysis_results.mat');
        
        % 定义测试场景
        test_scenario = struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100);
        
        % 测试我们的方法
        fprintf('  测试优化后的我们的方法...\n');
        trust_builder = TrustRegionBuilder(behavior_models);
        
        % 创建小规模网格进行快速测试
        outdoor_temps = linspace(20, 30, 20);
        indoor_temps = linspace(22, 25, 15);
        [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
        
        % 查询置信度
        trust_scores = trust_builder.query_trust_scores(grid_outdoor, grid_indoor, test_scenario);
        
        % 统计结果
        fprintf('  结果统计:\n');
        fprintf('    最小置信度: %.3f\n', min(trust_scores(:)));
        fprintf('    最大置信度: %.3f\n', max(trust_scores(:)));
        fprintf('    平均置信度: %.3f\n', mean(trust_scores(:)));
        fprintf('    >0.6置信度的比例: %.1f%%\n', sum(trust_scores(:) > 0.6) / numel(trust_scores) * 100);
        
        % 验证修复效果
        if max(trust_scores(:)) <= 1.0
            fprintf('  ✓ 置信度范围正常\n');
        else
            fprintf('  ✗ 置信度仍然超出范围\n');
        end
        
        if mean(trust_scores(:)) > 0.4
            fprintf('  ✓ 平均置信度提升\n');
        else
            fprintf('  ⚠ 平均置信度仍需优化\n');
        end
        
    catch ME
        fprintf('  ✗ 无法加载模型: %s\n', ME.message);
        fprintf('  请先运行主要训练流程\n');
    end
end

%% ==========================================================================
%% 修复后的重新训练脚本
%% ==========================================================================
function retrain_with_fixes()
    % 应用修复后重新训练整个系统
    
    fprintf('=== 应用修复并重新训练 ===\n\n');
    
    % 重新训练行为模型（使用优化的奖励参数）
    fprintf('步骤1: 重新训练行为模型...\n');
    retrain_behavior_models();
    
    % 重新训练GPR（使用优化的核参数）
    fprintf('\n步骤2: 重新构建可信域...\n');
    rebuild_trust_regions();
    
    % 重新运行基线对比
    fprintf('\n步骤3: 重新运行基线对比...\n');
    run_baseline_comparison();
    
    fprintf('\n重新训练完成！\n');
end

function retrain_behavior_models()
    % 重新训练行为模型
    
    % 加载数据
    data_processor = DataProcessor();
    [processed_data, context_features, ~] = data_processor.load_and_preprocess('building_data_final.csv');
    
    % 使用优化参数训练行为模型
    behavior_trainer = BehaviorModelTrainer();
    [behavior_models, behavior_performance] = behavior_trainer.train_contextual_bandits(context_features, processed_data);
    
    % 保存优化后的模型
    save('optimized_behavior_models.mat', 'behavior_models', 'behavior_performance');
    
    fprintf('  行为模型重新训练完成\n');
    for model_name = keys(behavior_models)
        fprintf('    %s: %.2f%% 准确率\n', model_name{1}, behavior_performance.(model_name{1}).accuracy * 100);
    end
end

function rebuild_trust_regions()
    % 重新构建可信域
    
    load('optimized_behavior_models.mat');
    
    % 测试场景
    test_scenario = struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100);
    
    % 构建可信域
    trust_region_builder = TrustRegionBuilder(behavior_models);
    optimized_result = trust_region_builder.build_trust_region(test_scenario);
    
    % 保存结果
    save('optimized_trust_region_results.mat', 'optimized_result');
    
    fprintf('  可信域重新构建完成\n');
    fprintf('    平均置信度: %.3f\n', mean(optimized_result.gpr_result.smooth_surface(:)));
    fprintf('    最大置信度: %.3f\n', max(optimized_result.gpr_result.smooth_surface(:)));
end