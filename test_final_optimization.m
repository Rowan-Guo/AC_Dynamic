%% ==========================================================================
%% 3. 验证最终优化效果的测试脚本
function test_final_optimization()
    clear; clc; close all;
    
    fprintf('=== 验证最终优化效果 ===\n\n');
    
    %% 测试优化后的奖励概率分布
    fprintf('测试1: 优化后的奖励概率分布\n');
    test_optimized_reward_distribution();
    
    %% 测试真实场景下的性能
    fprintf('\n测试2: 多场景性能测试\n');
    test_multi_scenario_performance();
    
    fprintf('\n=== 最终优化验证完成 ===\n');
end

function test_optimized_reward_distribution()
    % 测试优化后的奖励概率分布
    
    trainer = BehaviorModelTrainer();
    
    % 创建更全面的测试场景
    outdoor_temps = [18, 22, 26, 30, 34];
    indoor_temps = [21, 22, 23, 24, 25];
    scenarios = {
        struct('isWeekday', true, 'HourOfDay', 15, 'Occupant_Count', 100),  % 工作日高峰
        struct('isWeekday', true, 'HourOfDay', 9, 'Occupant_Count', 80),    % 工作日上午
        struct('isWeekday', false, 'HourOfDay', 14, 'Occupant_Count', 20),  % 周末下午
        struct('isWeekday', true, 'HourOfDay', 6, 'Occupant_Count', 10)     % 早晨
    };
    
    fprintf('  优化后的接受概率矩阵:\n');
    fprintf('  场景\\温度组合\t');
    for i = 1:length(outdoor_temps)
        fprintf('Out%d/In%d\t', outdoor_temps(i), indoor_temps(min(i, end)));
    end
    fprintf('\n');
    
    for s = 1:length(scenarios)
        scenario = scenarios{s};
        scenario_name = sprintf('%s_%dh_%dp', ...
            char(string(scenario.isWeekday)), scenario.HourOfDay, scenario.Occupant_Count);
        fprintf('  %-12s\t', scenario_name);
        
        for i = 1:length(outdoor_temps)
            context = table(scenario.HourOfDay, scenario.isWeekday, ...
                          outdoor_temps(i), indoor_temps(min(i, end)), scenario.Occupant_Count, ...
                          'VariableNames', {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});
            
            p_accept = trainer.compute_acceptance_probability(context, 0.5);
            fprintf('%.3f\t\t', p_accept);
        end
        fprintf('\n');
    end
    
    % 计算整体统计
    all_probs = [];
    for s = 1:length(scenarios)
        scenario = scenarios{s};
        for i = 1:length(outdoor_temps)
            context = table(scenario.HourOfDay, scenario.isWeekday, ...
                          outdoor_temps(i), indoor_temps(min(i, end)), scenario.Occupant_Count, ...
                          'VariableNames', {'HourOfDay', 'isWeekday', 'Outdoor_Temp_C', 'Avg_Zone_Temp_C', 'Occupant_Count'});
            p_accept = trainer.compute_acceptance_probability(context, 0.5);
            all_probs(end+1) = p_accept;
        end
    end
    
    fprintf('\n  整体统计:\n');
    fprintf('    平均接受概率: %.3f\n', mean(all_probs));
    fprintf('    最高接受概率: %.3f\n', max(all_probs));
    fprintf('    最低接受概率: %.3f\n', min(all_probs));
    fprintf('    >0.5概率的比例: %.1f%%\n', sum(all_probs > 0.5) / length(all_probs) * 100);
end

function test_multi_scenario_performance()
    % 在多个场景下测试性能
    
    scenarios = [
        struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100, 'name', '工作日高峰');
        struct('HourOfDay', 9, 'isWeekday', true, 'Occupant_Count', 80, 'name', '工作日上午');
        struct('HourOfDay', 14, 'isWeekday', false, 'Occupant_Count', 20, 'name', '周末下午');
        struct('HourOfDay', 22, 'isWeekday', true, 'Occupant_Count', 5, 'name', '工作日晚间')
    ];
    
    try
        load('trust_region_analysis_results.mat');
        
        fprintf('  场景\t\t\t平均置信度\t>0.6比例\t>0.8比例\n');
        fprintf('  %s\n', repmat('-', 1, 60));
        
        for i = 1:length(scenarios)
            scenario = scenarios(i);
            
            % 构建可信域
            trust_builder = TrustRegionBuilder(behavior_models);
            
            % 创建小规模网格
            outdoor_temps = linspace(18, 32, 15);
            indoor_temps = linspace(21, 26, 12);
            [grid_outdoor, grid_indoor] = meshgrid(outdoor_temps, indoor_temps);
            
            % 查询置信度
            trust_scores = trust_builder.query_trust_scores(grid_outdoor, grid_indoor, scenario);
            
            % 统计结果
            mean_conf = mean(trust_scores(:));
            ratio_60 = sum(trust_scores(:) > 0.6) / numel(trust_scores) * 100;
            ratio_80 = sum(trust_scores(:) > 0.8) / numel(trust_scores) * 100;
            
            fprintf('  %-20s\t%.3f\t\t%.1f%%\t\t%.1f%%\n', ...
                   scenario.name, mean_conf, ratio_60, ratio_80);
        end
        
    catch ME
        fprintf('  无法加载模型进行测试: %s\n', ME.message);
    end
end

%% ==========================================================================
%% 4. 一键应用所有最终优化
function apply_final_optimizations()
    fprintf('=== 应用最终优化参数 ===\n\n');
    
    fprintf('请手动应用以下优化：\n\n');
    
    fprintf('1. 在 BehaviorModelTrainer.m 的 compute_acceptance_probability 方法中：\n');
    fprintf('   base_discomfort = 0.1 + 0.15 * action_magnitude;\n');
    fprintf('   temp_penalty = 0.08 * max(0, context_data.Outdoor_Temp_C - 27) + ...\n');
    fprintf('                 0.12 * max(0, context_data.Avg_Zone_Temp_C - 24.5);\n\n');
    
    fprintf('2. 在 IncrementalGPREngine.m 构造函数中：\n');
    fprintf('   obj.kernel_params = struct(''length_scale'', 3.0, ''signal_var'', 1.2);\n');
    fprintf('   obj.noise_var = 0.0005;\n');
    fprintf('   obj.buffer_size = 5;\n\n');
    
    fprintf('3. 应用完成后，运行以下测试：\n');
    fprintf('   test_final_optimization()\n\n');
    
    fprintf('4. 然后重新运行完整实验：\n');
    fprintf('   main_trust_region_analysis()\n');
    fprintf('   run_baseline_comparison()\n\n');
end

%% ==========================================================================
%% 5. 预期效果预测
function predict_optimization_effects()
    fprintf('=== 预期优化效果 ===\n\n');
    
    fprintf('基于参数调整，预期改进效果：\n\n');
    
    fprintf('1. 平均置信度提升：\n');
    fprintf('   当前: 0.345\n');
    fprintf('   预期: 0.45-0.55 (+30-60%%)\n\n');
    
    fprintf('2. 高置信度区域扩大：\n');
    fprintf('   当前>0.6置信度: 14.3%%\n');
    fprintf('   预期>0.6置信度: 25-35%%\n\n');
    
    fprintf('3. 与基线方法对比：\n');
    fprintf('   预期超越简单规则方法\n');
    fprintf('   预期显著优于静态方法\n\n');
    
    fprintf('4. GPR平滑效果：\n');
    fprintf('   更强的平滑能力\n');
    fprintf('   更清晰的可信域边界\n\n');
    
    fprintf('5. 实用性提升：\n');
    fprintf('   更大的可调控空间\n');
    fprintf('   更合理的置信度分布\n\n');
end