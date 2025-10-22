function straightenedTforms = straightening(ccbundlerTforms)

    straightenedTforms = cell(1,length(ccbundlerTforms));

    for cc = 1:length(ccbundlerTforms)
        cams = ccbundlerTforms{cc};

        % Collect each camera's X-axis in WORLD coords (R: world->cam ⇒ row 1)
        X = cell2mat(arrayfun(@(c) c.R(1,1:3)', cams, 'UniformOutput', false));
        C = X * X.';                          % sum_i X_i X_i^T
        [~,~,V] = svd(C);
        up = V(:,end);                         % "up" direction (smallest singular)

        % Average Z (world) and build orthonormal frame
        Zsum = sum(cell2mat(arrayfun(@(c) c.R(3,:)', cams, 'UniformOutput', false)), 2);
        xhat = cross(up, Zsum);  xhat = xhat / norm(xhat);
        zhat = cross(xhat, up);  zhat = zhat / norm(zhat);

        % World basis whose Y is "up"
        B = [xhat, up, zhat];

        % Global rotation that maps this basis to canonical axes
        S = B.';  % <— transpose is the rotation we want

        % Apply the *same* world-frame change to every camera
        for k = 1:numel(cams)
            cams(k).R = cams(k).R * S;
        end

        straightenedTforms{cc} = cams;
    end
end
