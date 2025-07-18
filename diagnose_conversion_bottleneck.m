%% ==========================================================================
%% 突破性问题诊断与解决方案
%% 解决奖励概率与最终置信度转化效率低的问题
%% ==========================================================================

%% 1. 深度诊断脚本：diagnose_conversion_bottleneck.m
%% ==========================================================================
function diagnose_conversion_bottleneck()
    clear; clc; close all;
    
    fprintf('=== 深度诊断：奖励-置信度转化瓶颈 ===\n\n');
    
    %% 诊断1: 分析决策树模型的预测质量
    fprintf('诊断1: 分析决策树模型预测质量\n');
    analyze_decision_tree_quality();
    
    %% 诊断2: 分析概率化奖励的训练数据质量
    fprintf('\n诊断2: 分析概率化奖励训练数据质量\n');
    analyze_training_data_quality();
    
    %% 诊断3: 分析GPR输入数据的分布
    fprintf('\n诊断3: 分析GPR输入数据分布\n');
    analyze_gpr_input_distribution();
    
    %% 提供突破性解决方案
    fprintf('\n=== 突破性解决方案 ===\n');
    provide_breakthrough_solutions();
end

function analyze_decision_tree_quality()
    % 分析决策树模型的预测质量
    
    try
        load('trust_region_analysis_results.mat');
        
        model_names = keys(behavior_models);
        fprintf('  决策树模型分析:\n');
        
        for i = 1:length(model_names)
            model_name = model_names{i};
            model = behavior_models(model_name);
            
            fprintf('    %s:\n', model_name);
            
            % 分析模型复杂度
            if isprop(model, 'NumNodes')
                fprintf('      节点数: %d (过简单: <10, 过复杂: >100)\n', model.NumNodes);
            end
            
            % 分析预测概率分布
            % 创建测试数据
            test_context = create_diverse_test_contexts();
            [predictions, scores] = predict(model, test_context);
            
            acceptance_probs = scores(:, 2); % 接受概率
            
            fprintf('      预测概率统计:\n');
            fprintf('        均值: %.3f\n', mean(acceptance_probs));
            fprintf('        标准差: %.3f (太小说明预测过于保守)\n', std(acceptance_probs));
            fprintf('        范围: [%.3f, %.3f]\n', min(acceptance_probs), max(acceptance_probs));
            fprintf('        >0.5的比例: %.1f%%\n', sum(acceptance_probs > 0.5) / length(acceptance_probs) * 100);
            
            if std(acceptance_probs) < 0.1
                fprintf('      ⚠ 预测方差过小，模型可能过于保守\n');
            end
        end
        
    catch ME
        fprintf('  无法加载模型: %s\n', ME.message);
    end
end

function test_context = create_diverse_test_contexts()
    % 创建多样化的测试上下文
    
    hours = [6, 9, 12, 15, 18, 21];
    weekdays = [true, false];
    outdoor_temps = [15, 20, 25, 30, 35];
    indoor_temps = [21, 22, 23, 24, 25, 26];
    occupancies = [5, 25, 50, 75, 100];
    
    contexts = [];
    for h = hours
        for w = weekdays
            for ot = outdoor_temps
                for it = indoor_temps
                    for oc = occupancies
                        contexts(end+1, :) = [h, w, ot, it, oc];
                    end
                end
            end
        end
    end
    
    test_context = array2table(contexts, 'VariableNames', ...
        {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});
end

function analyze_training_data_quality()
    % 分析概率化奖励生成的训练数据质量
    
    fprintf('  模拟概率化奖励生成过程:\n');
    
    % 重新生成奖励数据
    trainer = BehaviorModelTrainer();
    test_context = create_diverse_test_contexts();
    
    % 生成概率化奖励
    rewards_action2 = [];
    rewards_action3 = [];
    acceptance_probs_2 = [];
    acceptance_probs_3 = [];
    
    for i = 1:height(test_context)
        context_row = test_context(i, :);
        
        % 计算接受概率
        p_2 = trainer.compute_acceptance_probability(context_row, 0.5);
        p_3 = trainer.compute_acceptance_probability(context_row, 1.0);
        
        acceptance_probs_2(end+1) = p_2;
        acceptance_probs_3(end+1) = p_3;
        
        % 模拟随机奖励生成
        rewards_action2(end+1) = rand() < p_2;
        rewards_action3(end+1) = rand() < p_3;
    end
    
    fprintf('    Action 2 (+0.5°C):\n');
    fprintf('      理论平均接受概率: %.3f\n', mean(acceptance_probs_2));
    fprintf('      实际奖励比例: %.3f\n', mean(rewards_action2));
    fprintf('      奖励方差: %.3f\n', var(rewards_action2));
    
    fprintf('    Action 3 (+1.0°C):\n');
    fprintf('      理论平均接受概率: %.3f\n', mean(acceptance_probs_3));
    fprintf('      实际奖励比例: %.3f\n', mean(rewards_action3));
    fprintf('      奖励方差: %.3f\n', var(rewards_action3));
    
    % 关键分析：随机化导致的信息损失
    info_loss_2 = abs(mean(acceptance_probs_2) - mean(rewards_action2));
    info_loss_3 = abs(mean(acceptance_probs_3) - mean(rewards_action3));
    
    fprintf('    信息损失分析:\n');
    fprintf('      Action 2 信息损失: %.3f\n', info_loss_2);
    fprintf('      Action 3 信息损失: %.3f\n', info_loss_3);
    
    if info_loss_2 > 0.05 || info_loss_3 > 0.05
        fprintf('      ⚠ 随机化过程导致显著信息损失\n');
    end
end

function analyze_gpr_input_distribution()
    % 分析GPR输入数据的分布特征
    
    fprintf('  GPR输入数据分布分析:\n');
    
    try
        load('trust_region_analysis_results.mat');
        
        % 模拟当前的置信度查询过程
        trust_builder = TrustRegionBuilder(behavior_models);
        test_scenario = struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100);
        
        % 创建小网格进行分析
        outdoor_temps = linspace(20, 30, 20);
        indoor_temps = linspace(22, 25, 15);
        [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
        
        % 查询原始置信度
        raw_scores = trust_builder.query_trust_scores(grid_outdoor, grid_indoor, test_scenario);
        
        fprintf('    原始置信度分数统计:\n');
        fprintf('      均值: %.3f\n', mean(raw_scores(:)));
        fprintf('      标准差: %.3f\n', std(raw_scores(:)));
        fprintf('      范围: [%.3f, %.3f]\n', min(raw_scores(:)), max(raw_scores(:)));
        fprintf('      非零值比例: %.1f%%\n', sum(raw_scores(:) > 0) / numel(raw_scores) * 100);
        
        % 分析数据的空间结构
        [grad_x, grad_y] = gradient(raw_scores);
        gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
        
        fprintf('    空间结构分析:\n');
        fprintf('      平均梯度幅度: %.3f (过大说明数据噪声多)\n', mean(gradient_magnitude(:)));
        fprintf('      梯度方差: %.3f\n', var(gradient_magnitude(:)));
        
        if mean(gradient_magnitude(:)) > 0.1
            fprintf('      ⚠ 原始数据空间变化过于剧烈，GPR平滑效果有限\n');
        end
        
        % 可视化原始数据
        figure('Name', 'GPR输入数据诊断');
        subplot(1,3,1);
        imagesc(raw_scores);
        title('原始置信度分布');
        colorbar;
        
        subplot(1,3,2);
        histogram(raw_scores(:), 20);
        title('置信度直方图');
        xlabel('置信度');
        ylabel('频次');
        
        subplot(1,3,3);
        imagesc(gradient_magnitude);
        title('梯度幅度');
        colorbar;
        
    catch ME
        fprintf('  无法完成GPR输入分析: %s\n', ME.message);
    end
end

function provide_breakthrough_solutions()
    % 提供突破性解决方案
    
    fprintf('基于诊断结果的突破性解决方案:\n\n');
    
    fprintf('【方案1】直接概率回归（推荐）\n');
    fprintf('  问题根源: 决策树+随机化导致信息损失过多\n');
    fprintf('  解决方案: 跳过概率化奖励，直接训练概率回归模型\n');
    fprintf('  优势: 消除随机性，直接学习连续概率分布\n');
    fprintf('  实现: 使用SVM或神经网络进行回归\n\n');
    
    fprintf('【方案2】确定性奖励+平滑（快速修复）\n');
    fprintf('  问题根源: 随机化引入过多噪声\n');
    fprintf('  解决方案: 直接使用理论概率作为目标，去除随机性\n');
    fprintf('  优势: 保持现有架构，快速改进\n');
    fprintf('  实现: 修改训练过程，使用连续概率作为标签\n\n');
    
    fprintf('【方案3】增强GPR配置（补充优化）\n');
    fprintf('  问题根源: GPR对噪声数据的平滑能力不足\n');
    fprintf('  解决方案: 使用更适合噪声数据的GPR配置\n');
    fprintf('  优势: 提升现有方法的鲁棒性\n');
    fprintf('  实现: 调整核函数类型和参数\n\n');
end

%% ==========================================================================
%% 2. 快速修复方案：确定性奖励（推荐立即实施）
%% ==========================================================================
function implement_deterministic_rewards()
    fprintf('=== 实施确定性奖励方案 ===\n\n');
    
    fprintf('修改 BehaviorModelTrainer.m 中的 generate_probabilistic_rewards 方法:\n\n');
    
    fprintf('【原方法】随机化奖励生成:\n');
    fprintf('rewards(:, action_idx) = rand(num_samples, 1) < p_accept;\n\n');
    
    fprintf('【新方法】确定性奖励生成:\n');
    fprintf('rewards(:, action_idx) = p_accept; %% 直接使用概率作为连续标签\n\n');
    
    fprintf('同时需要修改模型训练，使用回归而非分类:\n');
    fprintf('在 train_single_action_model 方法中:\n');
    fprintf('model = fitrensemble(X_train, Y_train, ...\n');
    fprintf('    ''Method'', ''Bag'', ''NumLearningCycles'', 100);\n\n');
    
    fprintf('这将消除随机性带来的信息损失，直接学习连续概率分布。\n');
end

%% ==========================================================================
%% 3. 确定性奖励的完整实现
%% ==========================================================================
classdef DeterministicBehaviorModelTrainer < BehaviorModelTrainer
    % 确定性行为模型训练器 - 继承原类并重写关键方法
    
    methods
        function [models, performance] = train_contextual_bandits(obj, context_data, full_data)
            % 使用确定性奖励训练上下文老虎机
            
            % 生成确定性奖励（连续概率值）
            deterministic_rewards = obj.generate_deterministic_rewards(context_data, full_data);
            
            % 为每个动作训练回归模型
            performance = struct();
            
            for action_idx = 2:length(obj.actions)
                action_name = sprintf('Action_%d', action_idx);
                fprintf('正在训练%s回归模型 (设定点+%.1f°C)...\n', action_name, obj.actions(action_idx));
                
                % 训练回归模型
                model = obj.train_regression_model(context_data, deterministic_rewards(:, action_idx));
                obj.models(action_name) = model;
                
                % 评估模型
                perf = obj.evaluate_regression_model(model, context_data, deterministic_rewards(:, action_idx));
                performance.(action_name) = perf;
                
                fprintf('%s模型R²: %.3f\n', action_name, perf.r_squared);
            end
            
            models = obj.models;
        end
        
        function rewards = generate_deterministic_rewards(obj, context_data, full_data)
            % 生成确定性奖励（连续概率值）
            
            num_samples = height(context_data);
            num_actions = length(obj.actions);
            rewards = zeros(num_samples, num_actions);
            
            % 动作1（无调整）总是获得奖励
            rewards(:, 1) = 1;
            
            % 为动作2和3生成确定性概率奖励
            for action_idx = 2:num_actions
                p_accept = obj.compute_acceptance_probability(context_data, obj.actions(action_idx));
                rewards(:, action_idx) = p_accept; % 直接使用概率，不随机化
            end
            
            fprintf('确定性奖励生成完成。奖励统计:\n');
            for i = 1:num_actions
                fprintf('  动作%d (+%.1f°C): 平均概率 %.3f\n', i, obj.actions(i), mean(rewards(:, i)));
            end
        end
        
        function model = train_regression_model(obj, context_data, continuous_rewards)
            % 训练回归模型而非分类模型
            
            % 数据划分
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);
            
            X_train = context_data(training(cv), :);
            Y_train = continuous_rewards(training(cv));
            
            % 训练集成回归模型（性能通常优于单一决策树）
            model = fitrensemble(X_train, Y_train, ...
                'Method', 'Bag', ...
                'NumLearningCycles', 100, ...
                'Learners', 'tree');
        end
        
        function performance = evaluate_regression_model(obj, model, context_data, continuous_rewards)
            % 评估回归模型性能
            
            % 数据划分（与训练保持一致）
            rng(42);
            cv = cvpartition(height(context_data), 'HoldOut', 0.3);
            
            X_test = context_data(test(cv), :);
            Y_test = continuous_rewards(test(cv));
            
            % 进行预测
            Y_pred = predict(model, X_test);
            
            % 计算回归性能指标
            ss_res = sum((Y_test - Y_pred).^2);
            ss_tot = sum((Y_test - mean(Y_test)).^2);
            r_squared = 1 - ss_res / ss_tot;
            
            rmse = sqrt(mean((Y_test - Y_pred).^2));
            mae = mean(abs(Y_test - Y_pred));
            
            performance = struct(...
                'r_squared', r_squared, ...
                'rmse', rmse, ...
                'mae', mae);
        end
    end
end

%% ==========================================================================
%% 4. 一键应用确定性奖励方案
%% ==========================================================================
function apply_deterministic_solution()
    fprintf('=== 应用确定性奖励解决方案 ===\n\n');
    
    fprintf('步骤1: 修改 BehaviorModelTrainer.m\n');
    fprintf('在 generate_probabilistic_rewards 方法中:\n');
    fprintf('将: rewards(:, action_idx) = rand(num_samples, 1) < p_accept;\n');
    fprintf('改为: rewards(:, action_idx) = p_accept;\n\n');
    
    fprintf('步骤2: 修改 train_single_action_model 方法\n');
    fprintf('将: model = fitctree(X_train, Y_train, ...);\n');
    fprintf('改为: model = fitrensemble(X_train, Y_train, ''Method'', ''Bag'', ''NumLearningCycles'', 100);\n\n');
    
    fprintf('步骤3: 修改查询过程\n');
    fprintf('在 query_trust_scores 方法中:\n');
    fprintf('将: [~, scores] = predict(model, current_context); p_accept = scores(2);\n');
    fprintf('改为: p_accept = predict(model, current_context);\n\n');
    
    fprintf('应用这些修改后，预期效果:\n');
    fprintf('- 消除随机性导致的信息损失\n');
    fprintf('- 平均置信度提升到0.5+\n');
    fprintf('- 与理论概率更好地对应\n\n');
end