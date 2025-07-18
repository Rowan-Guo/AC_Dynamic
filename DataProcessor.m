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
