function reportGPUMemory(label)
    % REPORTGPUMEMORY  Print a concise GPU memory usage summary.
    %
    %   reportGPUMemory(label)
    %
    % Inputs:
    %   label - Text scalar used as a prefix in the printed line (e.g. 'Step1').
    %
    % Outputs:
    %   None (the function prints a single-line summary to standard output).
    %
    % Example:
    %   reportGPUMemory('renderPanorama')
    %
    % Notes:
    %   - Uses `gpuDeviceCount` and `gpuDevice` to query memory. If no GPU is
    %     available the function does nothing.
    %   - Values are reported in megabytes (MB) and the percent used is shown.

    arguments
        label {mustBeTextScalar}
    end

    if gpuDeviceCount > 0
        g = gpuDevice;
        usedMB = (g.TotalMemory - g.AvailableMemory) / 1e6;
        totalMB = g.TotalMemory / 1e6;
        fprintf('[%s] GPU Memory: %.0f / %.0f MB (%.1f%% used)\n', ...
            label, usedMB, totalMB, 100 * usedMB / totalMB);
    end

end
