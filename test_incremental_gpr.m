%% 测试增量式GPR功能
function test_incremental_gpr()
    % 生成测试数据
    X_initial = rand(100, 2) * 10;
    y_initial = sin(sum(X_initial, 2)) + 0.1 * randn(100, 1);
    
    % 初始训练
    gpr_engine = IncrementalGPREngine();
    gpr_engine.fit(X_initial, y_initial);
    
    % 测试预测
    X_test = rand(20, 2) * 10;
    [y_pred_before, ~] = gpr_engine.predict(X_test);
    
    % 添加新数据点
    X_new = rand(10, 2) * 10;
    y_new = sin(sum(X_new, 2)) + 0.1 * randn(10, 1);
    
    for i = 1:size(X_new, 1)
        gpr_engine.add_data_point(X_new(i,:), y_new(i));
    end
    
    % 强制更新并测试
    gpr_engine.force_update();
    [y_pred_after, ~] = gpr_engine.predict(X_test);
    
    fprintf('增量学习前后预测差异: %.6f\n', mean(abs(y_pred_after - y_pred_before)));
    fprintf('增量GPR测试完成！\n');
end