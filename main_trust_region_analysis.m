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