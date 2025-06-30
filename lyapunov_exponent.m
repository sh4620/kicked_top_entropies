%% Classical Kicked Top: Lyapunov Exponent Computation
% Computes the maximal Lyapunov exponent over phase space for the classical kicked top
% using adaptive integration (ODE45) and tangent-space renormalization.

%% Parameters
epsilon = 0;          
gamma   = 0.2;        % Dissipation strength
p       = 2;          % Precession frequency
k       = 3;          % Kick strength
tau     = 1;          % Time between kicks

%% Simulation Settings
total_kicks     = 2000;      % Total number of kicks
transient_kicks = 200;       % Transient steps before Lyapunov measurement
delta           = 1e-7;      % Initial perturbation size

%% Phase Space Grid (φ, s_z)
sz_grid  = linspace(-1, 1, 70);
phi_grid = linspace(-pi, pi, 70);
[sz_mesh, phi_mesh] = meshgrid(sz_grid, phi_grid);

% Storage for Lyapunov exponents
lyap_exp = zeros(size(sz_mesh));

% ODE45 Options
ode_options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);

%% Main Loop: Compute Lyapunov Exponent for Each Grid Point
for i = 1:numel(sz_mesh)
    fprintf('Processing point %d/%d\n', i, numel(sz_mesh));

    % Initial state from (φ, s_z) to Cartesian spin vector
    sz  = sz_mesh(i);
    phi = phi_mesh(i);
    s   = [sqrt(1 - sz^2) * cos(phi); 
           sqrt(1 - sz^2) * sin(phi); 
           sz];

    % Initial perturbation, orthogonal to s (tangent to Bloch sphere)
    random_dir = randn(3, 1);
    delta_s    = delta * (random_dir - s * dot(s, random_dir));
    s_perturbed = s + delta_s;

    % Lyapunov sum accumulator
    lyap_sum    = 0;
    valid_steps = 0;

    for kick = 1:total_kicks
        %% ===== Free Precession (Half τ) =====
        [~, s_traj] = ode45(@(t, y) derivatives(t, y, epsilon, gamma, p), ...
                            [0 tau/2], s, ode_options);
        s = s_traj(end, :)';
        s = s / norm(s);  % Normalize

        [~, s_pert_traj] = ode45(@(t, y) derivatives(t, y, epsilon, gamma, p), ...
                                 [0 tau/2], s_perturbed, ode_options);
        s_perturbed = s_pert_traj(end, :)';
        s_perturbed = s_perturbed / norm(s_perturbed);

        %% ===== Kick (z-axis rotation) =====
        angle = 2 * k * s(3);
        Rz = [cos(angle), -sin(angle); sin(angle), cos(angle)];
        s(1:2) = Rz * s(1:2);
        s = s / norm(s);

        angle_pert = 2 * k * s_perturbed(3);
        Rz_pert = [cos(angle_pert), -sin(angle_pert); sin(angle_pert), cos(angle_pert)];
        s_perturbed(1:2) = Rz_pert * s_perturbed(1:2);
        s_perturbed = s_perturbed / norm(s_perturbed);

        %% ===== Second Free Precession (Half τ) =====
        [~, s_traj] = ode45(@(t, y) derivatives(t, y, epsilon, gamma, p), ...
                            [0 tau/2], s, ode_options);
        s = s_traj(end, :)';

        [~, s_pert_traj] = ode45(@(t, y) derivatives(t, y, epsilon, gamma, p), ...
                                 [0 tau/2], s_perturbed, ode_options);
        s_perturbed = s_pert_traj(end, :)';

        %% ===== Lyapunov Measurement (after transient) =====
        if kick > transient_kicks
            displacement = s_perturbed - s;
            displacement = displacement - s * dot(s, displacement);  % Project onto tangent plane
            distance = norm(displacement);

            lyap_sum = lyap_sum + log(distance / delta);
            valid_steps = valid_steps + 1;

            % Renormalize perturbation
            s_perturbed = s + (delta / distance) * displacement;
        end
    end

    % Store the average Lyapunov exponent
    lyap_exp(i) = lyap_sum / valid_steps;
end

% Remove negative values (unphysical due to numerical errors)
lyap_exp(lyap_exp < 0) = 0;

%% Save Data for Replotting
save('lyapunov_data.mat', 'lyap_exp', 'phi_mesh', 'sz_mesh', ...
    'epsilon', 'gamma', 'p', 'k', 'total_kicks', 'transient_kicks');

%% Plot Lyapunov Exponents on (φ, s_z) Phase Space
max_lyap = max(lyap_exp(:));
caxis_max = ceil(max_lyap * 20) / 20;  % Round up to nearest 0.05

figure;
pcolor(phi_mesh, sz_mesh, lyap_exp);
shading interp;
colorbar;
caxis([0 caxis_max]);
colormap(parula);

xlabel('\phi');
ylabel('s_z');
title(sprintf('Lyapunov Exponents \\epsilon=%.1f, \\gamma=%.1f, p=%.1f, k=%.1f [0–%.2f]', ...
               epsilon, gamma, p, k, caxis_max));

%% Save Figure
set(gcf, 'Renderer', 'painters');  % Vector rendering
print('lyapunov_ode45.pdf', '-dpdf', '-bestfit');

%% ===== System Dynamics (Free Precession) =====
function dsdt = derivatives(~, s, epsilon, gamma, p)
    sx = s(1); sy = s(2); sz = s(3);
    dsx_dt = -epsilon * sy - gamma * sz * sx;
    dsy_dt =  epsilon * sx - p * sz - gamma * sz * sy;
    dsz_dt =  p * sy + gamma * (1 - sz^2);
    dsdt = [dsx_dt; dsy_dt; dsz_dt];
end
