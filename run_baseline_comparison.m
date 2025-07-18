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
    % test_scenarios = [
    %     struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100, 'name', '工作日下午高峰');
    %     struct('HourOfDay', 9, 'isWeekday', true, 'Occupant_Count', 80, 'name', '工作日上午');
    %     struct('HourOfDay', 14, 'isWeekday', false, 'Occupant_Count', 20, 'name', '周末下午');
    %     struct('HourOfDay', 22, 'isWeekday', true, 'Occupant_Count', 5, 'name', '工作日晚间')
    % ];
    %% 【新】定义三个“压力测试”场景
    test_scenarios = [
        struct('HourOfDay', 14, 'isWeekday', false, 'Occupant_Count', 5, ...
        'name', '酷热周末的空闲办公室 (Hot & Empty Weekend)');

        struct('HourOfDay', 15, 'isWeekday', true, 'Occupant_Count', 100, ...
        'name', '温和工作日的高峰办公室 (Mild & Crowded Weekday)');

        struct('HourOfDay', 22, 'isWeekday', true, 'Occupant_Count', 10, ...
        'name', '工作日晚间的过渡时段 (Late Evening Transition)')
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