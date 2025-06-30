%% Classical Kicked Top: Poincaré Section Simulation
% Simulates trajectories of a classical kicked top and generates a Poincaré section
% using a Runge-Kutta 4 (RK4) integrator and spin precession between delta-function kicks.

%% System Parameters
epsilon = 0;         
gamma   = 0.3;       
p       = 2;         
k       = 1;         
tau     = 1;         

%% Simulation Control
enable_backward_propagation = false;  % Toggle backward integration (for time-reversal studies)

%% Integration Parameters
t0  = 0;             % Initial time
tf  = 100;           % Final time
h   = 0.001;         % Integration time step
num_steps_per_half_tau = round((tau / 2) / h);
num_kicks = floor(tf / tau);  % Total number of kicks

%% Initial Conditions Grid in (s_z, phi) Space
num_phi = 20;        % Number of points in φ direction
num_sz  = 20;        % Number of points in s_z direction

phi_grid = linspace(-pi, pi, num_phi);
sz_grid  = linspace(-1, 1, num_sz);
[phi_mesh, sz_mesh] = meshgrid(phi_grid, sz_grid);

% Convert to Cartesian spin coordinates on the Bloch sphere
sx = sqrt(1 - sz_mesh.^2) .* cos(phi_mesh);
sy = sqrt(1 - sz_mesh.^2) .* sin(phi_mesh);
sz = sz_mesh;

initial_conditions = [sx(:)'; sy(:)'; sz(:)'];
total_trajectories = size(initial_conditions, 2);

disp(['Total trajectories: ', num2str(total_trajectories)]);

%% Setup Figure for Poincaré Plot
figure; hold on;

%% Main Trajectory Loop
for traj = 1:total_trajectories
    disp(['Trajectory: ', num2str(traj)]);
    
    directions = enable_backward_propagation * [-1, 1] + (~enable_backward_propagation) * 1;
    
    for direction = directions
        s = initial_conditions(:, traj);
        sz_vals  = zeros(1, num_kicks);
        phi_vals = zeros(1, num_kicks);
        
        for kick_idx = 1:num_kicks
            % Free evolution for τ/2 before the kick
            for step = 1:num_steps_per_half_tau
                s = rk4_step(@derivatives, 0, s, direction * h, epsilon, gamma, p);
            end
            
            % Apply the kick (z-axis rotation)
            s = apply_kick(s, direction * k);
            
            % Free evolution for τ/2 after the kick
            for step = 1:num_steps_per_half_tau
                s = rk4_step(@derivatives, 0, s, direction * h, epsilon, gamma, p);
            end
            
            % Store point in Poincaré section (φ, s_z)
            sx = s(1); sy = s(2); sz = s(3);
            phi_vals(kick_idx) = atan2(sy, sx);
            sz_vals(kick_idx)  = sz;
        end
        
        % Plot Poincaré points
        scatter(phi_vals, sz_vals, 2, 'k', 'filled');
    end
end

%% Plot Formatting
xlabel('\phi');
ylabel('s_z');

title_str = sprintf('Poincaré Section (\\epsilon = %.2f, \\gamma = %.2f, p = %.2f, k = %.2f)', ...
    epsilon, gamma, p, k);
if ~enable_backward_propagation
    title_str = [title_str, ' - Forward Only'];
end
title(title_str);

xlim([-pi, pi]);
xticks([-pi, -pi/2, 0, pi/2, pi]);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});

grid off; box on;

%% Save Output as Vector PDF
filename = sprintf('poincare_kicked_p%.2f_eps%.2f_gamma%.2f', p, epsilon, gamma);
if ~enable_backward_propagation
    filename = [filename, '_forward'];
end
filename = [filename, '.pdf'];
print(gcf, filename, '-dpdf', '-bestfit');

%% ===== Helper Functions =====

% System dynamics (free motion between kicks)
function dsdt = derivatives(~, s, epsilon, gamma, p)
    sx = s(1); sy = s(2); sz = s(3);
    dsx_dt = -epsilon * sy - gamma * sz * sx;
    dsy_dt =  epsilon * sx - p * sz - gamma * sz * sy;
    dsz_dt =  p * sy + gamma * (1 - sz^2);
    dsdt = [dsx_dt; dsy_dt; dsz_dt];
end

% Runge-Kutta 4th order integrator
function s_next = rk4_step(f, t, s, h, epsilon, gamma, p)
    k1 = h * f(t,        s,           epsilon, gamma, p);
    k2 = h * f(t + h/2,  s + k1/2,    epsilon, gamma, p);
    k3 = h * f(t + h/2,  s + k2/2,    epsilon, gamma, p);
    k4 = h * f(t + h,    s + k3,      epsilon, gamma, p);
    s_next = s + (k1 + 2*k2 + 2*k3 + k4) / 6;
end

% Kick operator: rotation about z-axis by angle 2k s_z
function s_kicked = apply_kick(s, k)
    sz = s(3);
    angle = 2 * k * sz;
    Rz = [cos(angle), -sin(angle), 0;
          sin(angle),  cos(angle), 0;
                0,           0,    1];
    s_kicked = Rz * s;
end
