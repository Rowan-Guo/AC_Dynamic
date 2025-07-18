%% “单点实时更新”效率对比实验
function run_online_validation_single_point()
    clear; clc; close all;
    
    fprintf('=== “单点实时更新”效率对比实验 ===\n\n');
    
    % ================== 1. 准备数据 ==================
    fprintf('步骤 1: 准备模拟数据...\n');
    [X_full, y_full] = generate_mock_gpr_data(1000); % 总数据量减少到1000，以控制实验时间
    
    n_initial = 100; % 初始训练集
    n_stream = 400;  % 模拟400个单点更新事件
    n_test = 500;    % 固定测试集
    
    X_initial = X_full(1:n_initial, :);
    y_initial = y_full(1:n_initial);
    
    X_stream = X_full(n_initial+1 : n_initial+n_stream, :);
    y_stream = y_full(n_initial+1 : n_initial+n_stream);
    
    X_test = X_full(n_initial+n_stream+1 : end, :);
    y_test = y_full(n_initial+n_stream+1 : end);
    
    fprintf('数据准备完成：初始集=%d, 数据流=%d个单点, 测试集=%d\n\n', ...
            n_initial, n_stream, n_test);

    % ================== 2. 初始化模型 ==================
    fprintf('步骤 2: 初始化模型...\n');
    gpr_incremental = IncrementalGPREngine();
    gpr_incremental.fit(X_initial, y_initial);
    
    gpr_batch = IncrementalGPREngine();
    gpr_batch.fit(X_initial, y_initial);
    
    X_history = X_initial;
    y_history = y_initial;
    
    results_time = zeros(n_stream, 2);
    results_rmse = zeros(n_stream, 2);
    data_point_counts = n_initial + (1:n_stream);

    % ================== 3. 模拟在线学习循环 ==================
    fprintf('步骤 3: 开始模拟逐点更新过程 (共%d步)...\n', n_stream);
    
    for i = 1:n_stream
        % 获取一个新数据点
        X_new = X_stream(i, :);
        y_new = y_stream(i);
        
        fprintf('  处理第 %d 个新数据点 (总数据量: %d)...\n', i, n_initial + i);
        
        % --- A. 增量式更新 ---
        tic;
        gpr_incremental.add_data_point(X_new, y_new); % add_data_point 会自动触发更新（因为buffer_size=1）
        time_incremental = toc;
        
        % --- B. 批量重新训练 ---
        tic;
        X_history = [X_history; X_new];
        y_history = [y_history; y_new];
        gpr_batch.fit(X_history, y_history);
        time_batch = toc;
        
        % --- C. 性能评估 ---
        y_pred_inc = gpr_incremental.predict(X_test);
        y_pred_batch = gpr_batch.predict(X_test);
        
        rmse_inc = sqrt(mean((y_test - y_pred_inc).^2));
        rmse_batch = sqrt(mean((y_test - y_pred_batch).^2));
        
        % --- D. 记录结果 ---
        results_time(i, :) = [time_incremental, time_batch];
        results_rmse(i, :) = [rmse_inc, rmse_batch];
    end
    
    fprintf('在线学习模拟完成！\n\n');

    % ================== 4. 结果可视化与分析 ==================
    fprintf('步骤 4: 生成对比分析图表...\n');
    plot_online_validation_results(data_point_counts, results_time, results_rmse);
end

% --- 辅助函数 (保持不变) ---
function [X, y] = generate_mock_gpr_data(num_points)
    X = rand(num_points, 2) * 20 - 10;
    y = sin(sqrt(X(:,1).^2 + X(:,2).^2)) ./ sqrt(X(:,1).^2 + X(:,2).^2) + 0.1 * randn(num_points, 1);
    y = (y + 1) / 2;
end

function plot_online_validation_results(data_points, time_results, rmse_results)
    figure('Position', [100, 100, 1200, 500], 'Name', '“单点实时更新”性能验证');
    
    subplot(1, 2, 1);
    plot(data_points, rmse_results(:, 1), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(data_points, rmse_results(:, 2), 'r--', 'LineWidth', 1.5);
    grid on;
    title('模型性能对比 (RMSE)', 'FontSize', 14);
    xlabel('训练样本总量', 'FontSize', 12);
    ylabel('在固定测试集上的RMSE', 'FontSize', 12);
    legend({'增量更新 (Incremental)', '批量重训练 (Batch)'}, 'Location', 'northeast');
    
    subplot(1, 2, 2);
    plot(data_points, time_results(:, 1), 'b-', 'LineWidth', 2);
    hold on;
    plot(data_points, time_results(:, 2), 'r--', 'LineWidth', 2);
    grid on;
    set(gca, 'YScale', 'log');
    title('模型更新效率对比 (对数坐标)', 'FontSize', 14);
    xlabel('训练样本总量', 'FontSize', 12);
    ylabel('单点更新/重训练所需时间 (秒)', 'FontSize', 12);
    legend({'增量更新 (Incremental)', '批量重训练 (Batch)'}, 'Location', 'northwest');
end