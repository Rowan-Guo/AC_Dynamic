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
            obj.kernel_params = struct('length_scale', 3.0, 'signal_var', 1.2);  % 增强平滑
            obj.noise_var = 0.0005;     % 进一步降低噪声
            obj.is_fitted = false;
            obj.update_buffer = struct('X', [], 'y', []);
            obj.buffer_size = 1; % 【关键修改】从10或5修改为1，实现单点实时更新
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

            % 计算预测均值
            y_pred = K_test_train * obj.alpha;

            %% 【新增】限制预测值在[0,1]范围内
            y_pred = max(0, min(1, y_pred));
            
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