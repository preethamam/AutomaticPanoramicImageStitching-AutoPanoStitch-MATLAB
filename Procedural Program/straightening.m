function straightenedTforms = straightening(ccbundlerTforms)
    % STRAIGHTENING Align camera rotations so the panorama appears upright.
    %   straightenedTforms = straightening(ccbundlerTforms) computes a single
    %   global rotation per connected component of cameras/images so that the
    %   resulting panorama is visually upright. For each component, it derives
    %   an "up" direction from the distribution of camera X-axes and rotates
    %   all cameras in that component by the same world-frame transform.
    %
    %   Input
    %   - ccbundlerTforms: 1-by-K or K-by-1 cell array. Each cell contains a
    %       nonempty struct array of cameras with field:
    %         R — 3x3 numeric rotation matrix (world -> camera).
    %
    %   Output
    %   - straightenedTforms: Cell array, same size as input, where each
    %       contained struct array has updated R fields after straightening.
    %
    %   Method (high-level)
    %   - For each component, collect each camera's X-axis in world coords
    %     (row 1 of R). Find the least-variation direction via SVD; treat this
    %     as the world "up". Average Z-axes to get a dominant horizontal
    %     direction, then build an orthonormal world basis whose Y is up.
    %     The transpose of this basis is the global rotation applied to all
    %     cameras in that component.
    %
    %   Notes
    %   - If the component is degenerate (e.g., averaged Z-axis is parallel to
    %     the computed up direction), a robust fallback is used to avoid
    %     zero-norm cross products; if this also fails, the identity transform
    %     is applied for that component.
    %
    %   See also svd, cross, norm

    arguments
        ccbundlerTforms cell
    end

    % Validate cell shape and contents
    if ~(isvector(ccbundlerTforms) && ~isempty(ccbundlerTforms))
        try 
            error('straightening:InvalidInput', 'ccbundlerTforms must be a nonempty vector cell array.');
        catch
            straightenedTforms = [];
            return
        end
    end

    for cc = 1:numel(ccbundlerTforms)

        % Check for empty camera
        if isempty(ccbundlerTforms{cc}), continue, end

        cams = ccbundlerTforms{cc};

        if ~(isstruct(cams) && ~isempty(cams) && isfield(cams, 'R'))
            error('straightening:InvalidCellContent', 'Each cell must contain a nonempty struct array with field R.');
        end

        for k = 1:numel(cams)
            Rk = cams(k).R;

            if ~(isnumeric(Rk) && isequal(size(Rk), [3, 3]) && all(isfinite(Rk(:))))
                error('straightening:InvalidRotation', 'R must be a 3x3 finite numeric matrix.');
            end

        end

    end

    % Initialize output
    straightenedTforms = cell(1, numel(ccbundlerTforms));

    parfor cc = 1:numel(ccbundlerTforms)

        if isempty(ccbundlerTforms{cc}), continue, end

        cams = ccbundlerTforms{cc};

        % Collect each camera's X-axis in WORLD coords (R: world->cam ⇒ row 1)
        X = cell2mat(arrayfun(@(c) c.R(1, 1:3)', cams, 'UniformOutput', false));
        C = X * X.'; % sum_i X_i X_i^T
        [~, ~, V] = svd(C);
        up = V(:, end); % "up" direction (smallest singular)

        % FIX: Check if "up" is actually pointing down, and flip if necessary
        % Robust heuristic: "up" should be opposite to the average camera Y-axis
        % (camera Y-axis points down in the image, so world "up" should be opposite)
        Y_axes = cell2mat(arrayfun(@(c) c.R(2, 1:3)', cams, 'UniformOutput', false));
        avg_Y = mean(Y_axes, 2);
        avg_Y = avg_Y / norm(avg_Y);

        if dot(up, avg_Y) > 0
            % "up" is pointing in same direction as downward camera Y-axes, so flip it
            up = -up;
            fprintf('  Straightening component %d: flipped "up" direction\n', cc);
        end

        % Average Z (world)
        Zsum = sum(cell2mat(arrayfun(@(c) c.R(3, :)', cams, 'UniformOutput', false)), 2);

        % Build an orthonormal frame with Y = up
        xhat = cross(up, Zsum);

        if norm(xhat) < eps
            % Choose an auxiliary axis not parallel to up
            e1 = [1; 0; 0];

            if abs(dot(up, e1)) > 0.99
                e1 = [0; 0; 1];
            end

            xhat = cross(up, e1);
        end

        if norm(xhat) < eps
            % Degenerate case — fall back to identity transform
            S = eye(3);
        else
            xhat = xhat / norm(xhat);
            zhat = cross(xhat, up);

            if norm(zhat) < eps
                S = eye(3);
            else
                zhat = zhat / norm(zhat);
                % World basis whose Y is "up"
                B = [xhat, up, zhat];
                % Global rotation that maps this basis to canonical axes
                S = B; % transpose is the rotation we want
            end

        end

        % Apply the *same* world-frame change to every camera
        for k = 1:numel(cams)
            cams(k).R = cams(k).R * S;
        end

        % Store straightened cameras/images
        straightenedTforms{cc} = cams;
    end

end
