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