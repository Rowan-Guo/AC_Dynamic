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