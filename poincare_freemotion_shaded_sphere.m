clear all;

%% === System Parameters ===
epsilon = 0.5;  % Precession rate (e.g., magnetic field strength)
gamma   = 0;    % Dissipation parameter (set to 0 for unitary dynamics)
p       = 1;    % Interaction or driving strength

%% === System Dynamics: Bloch-like ODE ===
function dsdt = derivatives(t, s, epsilon, gamma, p)
    sx = s(1); sy = s(2); sz = s(3);
    
    dsx_dt = -epsilon * sy - gamma * sz * sx;
    dsy_dt =  epsilon * sx - p * sz - gamma * sz * sy;
    dsz_dt =  p * sy + gamma * (1 - sz^2);
    
    dsdt = [dsx_dt; dsy_dt; dsz_dt];
end

%% === RK4 Integrator ===
function s_next = rk4_step(f, t, s, h, epsilon, gamma, p)
    k1 = h * f(t,       s,        epsilon, gamma, p);
    k2 = h * f(t + h/2, s + k1/2, epsilon, gamma, p);
    k3 = h * f(t + h/2, s + k2/2, epsilon, gamma, p);
    k4 = h * f(t + h,   s + k3,   epsilon, gamma, p);
    
    s_next = s + (k1 + 2*k2 + 2*k3 + k4)/6;
end

%% === Simulation Settings ===
h  = 0.01;              % Time step
tf = 15;                % Total simulation time
t_values_forward  = 0:h:tf;
t_values_backward = 0:-h:-tf;
Nf = length(t_values_forward);
Nb = length(t_values_backward);
num_trajectories = 20;

%% === Initial Conditions (Equator: sz = 0) ===
phi_init = linspace(-pi, pi, num_trajectories);
initial_conditions = [cos(phi_init); sin(phi_init); zeros(1, num_trajectories)];

%% === Visualization Setup ===

% Bloch sphere
[X, Y, Z] = sphere(50);  % Creates mesh for unit sphere

% 3D figure
figure;
set(gcf, 'Color', 'w', 'Position', [100 100 800 600]);
ax = axes('Parent', gcf);
hold(ax, 'on');
grid(ax, 'on');
axis(ax, 'equal');

% --- Shadow plane (under the Bloch sphere) ---
[theta, rho] = meshgrid(linspace(0, 2*pi, 50), linspace(0, 1.2, 20));
X_shadow = rho .* cos(theta);
Y_shadow = rho .* sin(theta);
Z_shadow = -0.99 * ones(size(X_shadow));  % Slightly beneath z = -1

% Plot translucent circular shadow
surf(ax, X_shadow, Y_shadow, Z_shadow, ...
    'FaceColor', [0.2 0.2 0.2], ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 'interp', ...
    'AlphaData', 1 - sqrt(X_shadow.^2 + Y_shadow.^2) / max(rho(:)), ...
    'FaceLighting', 'none');

% --- Plot Bloch sphere ---
surf(ax, X, Y, Z, ...
    'FaceColor', [0.8 0.8 0.8], ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 1, ...
    'FaceLighting', 'flat');

% Lighting for 3D effect
light('Position', [1 1 1],   'Style', 'infinite');
light('Position', [-1 -1 0.5], 'Style', 'infinite');

% Labels and view
xlabel('s_x', 'FontSize', 12);
ylabel('s_y', 'FontSize', 12);
zlabel('s_z', 'FontSize', 12);
title(' ');

view(ax, 30, 20);
set(ax, 'Projection', 'perspective');
axis(ax, [-1 1 -1 1 -1 1]);

%% === Plotting Trajectories ===
for traj = 1:num_trajectories
    s0 = initial_conditions(:, traj);

    % Forward trajectory
    s_values_f = zeros(3, Nf);
    s_values_f(:, 1) = s0;
    for i = 1:Nf-1
        s_values_f(:, i+1) = rk4_step(@derivatives, t_values_forward(i), s_values_f(:, i), h, epsilon, gamma, p);
    end

    % Backward trajectory
    s_values_b = zeros(3, Nb);
    s_values_b(:, 1) = s0;
    for i = 1:Nb-1
        s_values_b(:, i+1) = rk4_step(@derivatives, t_values_backward(i), s_values_b(:, i), -h, epsilon, gamma, p);
    end

    % Combine both directions (excluding repeated point at t=0)
    s_values = [fliplr(s_values_b(:, 2:end)), s_values_f];

    % Plot in black
    plot3(ax, s_values(1,:), s_values(2,:), s_values(3,:), ...
        'Color', 'k', 'LineWidth', 1.5);
end

%% === Final Touches ===
material dull;
lighting gouraud;
set(ax, 'Color', [0.98 0.98 0.98]);
