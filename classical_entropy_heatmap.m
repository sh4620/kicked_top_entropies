function classical_entropy_heatmap()
% CLASSICAL_ENTROPY_HEATMAP
% Computes and visualizes the classical phase-space entropy
% of a periodically kicked top using fixed binning and Laplace smoothing.

    %% ===== Simulation Parameters =====
    params.epsilon              = 0.5;
    params.gamma                = 0;
    params.p                    = 2;
    params.k                    = 1;       % Kicking strength
    params.tau                  = 1;       % Time between kicks
    params.tf                   = 200;     % Total simulation time
    params.num_transient_kicks = 20;      % Discard initial transient points
    
    %% ===== Phase Space Grid =====
    params.num_phi = 120;   % Resolution along φ
    params.num_sz  = 120;   % Resolution along s_z
    phi_grid = linspace(-pi, pi, params.num_phi);
    sz_grid  = linspace(-0.999, 0.999, params.num_sz);  % Avoid ±1 singularities
    [PHI, SZ] = meshgrid(phi_grid, sz_grid);
    
    %% ===== Entropy Histogram Binning =====
    params.phi_edges = linspace(-pi, pi, 20);
    params.sz_edges  = linspace(-1, 1, 20);
    params.smoothing = 1e-10;  % Laplace smoothing
    
    %% ===== ODE Solver Options =====
    params.ode_options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    
    %% ===== Entropy Calculation =====
    entropy_matrix = zeros(params.num_sz, params.num_phi);
    
    % Compute entropy at each grid point in parallel
    parfor i = 1:params.num_sz
        local_entropy = zeros(1, params.num_phi);
        for j = 1:params.num_phi
            sz   = SZ(i,j);
            phi  = PHI(i,j);
            s0   = [sqrt(1 - sz^2) * cos(phi); ...
                    sqrt(1 - sz^2) * sin(phi); ...
                    sz];

            trajectory         = simulate_trajectory(s0, params);
            local_entropy(j)   = calculate_entropy(trajectory, params);
        end
        entropy_matrix(i,:) = local_entropy;
        fprintf('Completed row %d / %d\n', i, params.num_sz);
    end
    
    %% ===== Plot Result =====
    plot_entropy_heatmap(phi_grid, sz_grid, entropy_matrix, params);
end

%% ================================
% Simulates a single classical trajectory
% ================================
function trajectory = simulate_trajectory(s0, params)
    num_kicks = floor(params.tf / params.tau);
    keep_n    = num_kicks - params.num_transient_kicks;
    trajectory_points = zeros(2, keep_n);
    s = s0;

    for kick_idx = 1:num_kicks
        % Free evolution (tau / 2)
        [~, s_traj] = ode45(@(t, s) derivatives(t, s, params), ...
                            [0, params.tau / 2], s, params.ode_options);
        s = s_traj(end, :)';
        
        % Apply instantaneous kick
        s = apply_kick(s, params.k);
        
        % Second half of free evolution
        [~, s_traj] = ode45(@(t, s) derivatives(t, s, params), ...
                            [0, params.tau / 2], s, params.ode_options);
        s = s_traj(end, :)';
        
        % Store after transient
        if kick_idx > params.num_transient_kicks
            idx = kick_idx - params.num_transient_kicks;
            trajectory_points(:, idx) = [atan2(s(2), s(1)); s(3)];
        end
    end

    trajectory = trajectory_points;
end

%% ================================
% Calculates entropy via 2D histogram
% ================================
function entropy = calculate_entropy(trajectory, params)
    if range(trajectory(1,:)) < 1e-6 && range(trajectory(2,:)) < 1e-6
        entropy = 0;  % Fixed point (no spread)
        return;
    end

    % Histogram with Laplace smoothing
    counts = histcounts2(trajectory(1,:), trajectory(2,:), ...
                         params.phi_edges, params.sz_edges);
    prob = (counts + params.smoothing) / ...
           (sum(counts(:)) + params.smoothing * numel(counts));
    
    % Remove zero entries for entropy calculation
    prob = prob(prob > 0);
    entropy = -sum(prob .* log(prob));
end

%% ================================
% Plots the interpolated entropy heatmap
% ================================
function plot_entropy_heatmap(phi_grid, sz_grid, entropy_matrix, params)
    figure('Position', [100, 100, 900, 700]);

    % Interpolate for smoother visualisation
    [phi_interp, sz_interp] = meshgrid(...
        linspace(-pi, pi, params.num_phi * 2), ...
        linspace(-1, 1,  params.num_sz  * 2));
    entropy_interp = interp2(phi_grid, sz_grid, entropy_matrix, ...
                             phi_interp, sz_interp, 'spline');

    % Display heatmap
    imagesc(phi_grid, sz_grid, entropy_interp);
    colormap(parula);
    colorbar;
    set(gca, 'YDir', 'normal', 'FontSize', 12);
    xticks([-pi, -pi/2, 0, pi/2, pi]);
    xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

    % Title and labels
    title(sprintf('Classical Entropy Heatmap (p = %.1f, k = %.1f)', ...
                  params.p, params.k), 'FontSize', 14);
    xlabel('\phi', 'FontSize', 12);
    ylabel('s_z',  'FontSize', 12);

    % Save high-resolution figure
    print('-dpdf', '-r300', '-bestfit', ...
        sprintf('entropy_p%.1f_k%.1f_g%.2f.pdf', ...
        params.p, params.k, params.gamma));
end

%% ================================
% Classical equations of motion
% ================================
function dsdt = derivatives(~, s, params)
    sx = s(1); sy = s(2); sz = s(3);
    dsx_dt = -params.epsilon * sy - params.gamma * sz * sx;
    dsy_dt =  params.epsilon * sx - params.p * sz - params.gamma * sz * sy;
    dsz_dt =  params.p * sy + params.gamma * (1 - sz^2);
    dsdt   = [dsx_dt; dsy_dt; dsz_dt];
end

%% ================================
% Kick map: rotation about z-axis
% ================================
function s_kicked = apply_kick(s, k)
    sz = s(3);
    angle = 2 * k * sz;
    Rz = [cos(angle), -sin(angle), 0;
          sin(angle),  cos(angle), 0;
               0,           0,     1];
    s_kicked = Rz * s;
end
