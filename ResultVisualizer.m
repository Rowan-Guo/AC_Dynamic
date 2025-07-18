%% ==========================================================================
%% 7. 结果可视化器：ResultVisualizer.m
%% ==========================================================================
classdef ResultVisualizer < handle
    methods
        function plot_comprehensive_results(obj, trust_region_result, scenario)
            % 创建综合结果可视化
            figure('Position', [100, 100, 1400, 900], 'Name', '动态可信调控域分析结果');
            
            % 子图1：原始置信度曲面
            subplot(2,3,1);
            obj.plot_raw_trust_scores(trust_region_result);
            
            % 子图2：GPR平滑后的曲面
            subplot(2,3,2);
            obj.plot_smooth_surface(trust_region_result);
            
            % 子图3：可信域边界
            subplot(2,3,3);
            obj.plot_trust_boundaries(trust_region_result);
            
            % 子图4：3D可视化
            subplot(2,3,4);
            obj.plot_3d_surface(trust_region_result);
            
            % 子图5：置信度分布直方图
            subplot(2,3,5);
            obj.plot_confidence_histogram(trust_region_result);
            
            % 子图6：场景信息显示
            subplot(2,3,6);
            obj.display_scenario_info(trust_region_result, scenario);
            
            sgtitle(sprintf('动态可信调控域 (工作日=%s, 时间=%d点, 占用人数=%d)', ...
                   string(scenario.isWeekday), scenario.HourOfDay, scenario.Occupant_Count), ...
                   'FontSize', 14, 'FontWeight', 'bold');
        end
        
        function plot_raw_trust_scores(obj, result)
            imagesc(result.sampling_grid.outdoor(1,:), result.sampling_grid.indoor(:,1), result.raw_trust_scores);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('原始置信度分布');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
        end
        
        function plot_smooth_surface(obj, result)
            imagesc(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), result.gpr_result.smooth_surface);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('GPR平滑曲面');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
        end

        % 在 ResultVisualizer.m 中
        function plot_trust_boundaries(obj, result)
            imagesc(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), result.gpr_result.smooth_surface);
            set(gca, 'YDir', 'normal');
            colorbar;
            hold on;

            % 【修改】动态创建图例
            thresholds = [0.8, 0.6]; % 您想绘制的阈值
            colors = {'white', 'red'};
            legend_handles = [];
            legend_labels = {};

            for i = 1:length(thresholds)
                threshold = thresholds(i);
                field_name = sprintf('threshold_%.0f', threshold*100);

                if isfield(result.trust_boundaries, field_name) && ~isempty(result.trust_boundaries.(field_name))
                    % 绘制等高线并保存其句柄
                    h = contour(result.gpr_result.grid_outdoor(1,:), result.gpr_result.grid_indoor(:,1), ...
                        result.gpr_result.smooth_surface, [threshold, threshold], 'LineWidth', 2, 'Color', colors{i});

                    % 如果成功绘制，则添加到图例列表
                    if ~isempty(h)
                        legend_handles(end+1) = h(1); % 只取第一个句柄
                        legend_labels{end+1} = sprintf('%.0f%%置信边界', threshold*100);
                    end
                end
            end

            if ~isempty(legend_handles)
                legend(legend_handles, legend_labels, 'Location', 'best');
            end

            title('可信域边界');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
            hold off;
        end

        function plot_3d_surface(obj, result)
            surf(result.gpr_result.grid_outdoor, result.gpr_result.grid_indoor, result.gpr_result.smooth_surface);
            title('3D置信度曲面');
            xlabel('室外温度 (°C)');
            ylabel('室内温度 (°C)');
            zlabel('置信度');
            view(3);
        end
        
        function plot_confidence_histogram(obj, result)
            histogram(result.gpr_result.smooth_surface(:), 20);
            title('置信度分布');
            xlabel('置信度');
            ylabel('频次');
            grid on;
        end
        
        function display_scenario_info(obj, result, scenario)
            text(0.1, 0.9, sprintf('测试场景信息:'), 'FontSize', 12, 'FontWeight', 'bold');
            text(0.1, 0.8, sprintf('是否工作日: %s', string(scenario.isWeekday)), 'FontSize', 10);
            text(0.1, 0.7, sprintf('时间: %d点', scenario.HourOfDay), 'FontSize', 10);
            text(0.1, 0.6, sprintf('占用人数: %d人', scenario.Occupant_Count), 'FontSize', 10);
            text(0.1, 0.4, sprintf('关键统计:'), 'FontSize', 12, 'FontWeight', 'bold');
            text(0.1, 0.3, sprintf('平均置信度: %.3f', mean(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            text(0.1, 0.2, sprintf('最高置信度: %.3f', max(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            text(0.1, 0.1, sprintf('最低置信度: %.3f', min(result.gpr_result.smooth_surface(:))), 'FontSize', 10);
            
            axis off;
            title('场景与统计信息');
        end
    end
end