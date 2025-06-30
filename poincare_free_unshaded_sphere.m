
%% === System Parameters ===
epsilon = 0.5;  % Energy splitting
gamma   = 0.5;  % Dissipation
p       = 1;    % Coupling strength

%% === Differential Equation (Bloch-like dynamics) ===
% Defines the time derivative of the state vector s = [s_x; s_y; s_z]
function dsdt = derivatives(t, s, epsilon, gamma, p)
    sx = s(1);
    sy = s(2);
    sz = s(3);
    
    dsx_dt = -epsilon * sy - gamma * sz * sx;
    dsy_dt = epsilon * sx - p * sz - gamma * sz * sy;
    dsz_dt = p * sy + gamma * (1 - sz^2);
    
    dsdt = [dsx_dt; dsy_dt; dsz_dt];
end

%% === Runge-Kutta 4th Order Integrator ===
% Advances one time step using RK4 for given derivative function
function s_next = rk4_step(f, t, s, h, epsilon, gamma, p)
    k1 = h * f(t, s, epsilon, gamma, p);
    k2 = h * f(t + h/2, s + k1/2, epsilon, gamma, p);
    k3 = h * f(t + h/2, s + k2/2, epsilon, gamma, p);
    k4 = h * f(t + h, s + k3, epsilon, gamma, p);
    
    s_next = s + (k1 + 2*k2 + 2*k3 + k4) / 6;
end

%% === Simulation Settings ===
h  = 0.01;               % Time step
tf = 15;                 % Final time for forward/backward integration

t_values_forward  = 0:h:tf;     % Forward time grid
t_values_backward = 0:-h:-tf;   % Backward time grid
Nf = length(t_values_forward);
Nb = length(t_values_backward);

num_trajectories = 20;

%% === Initial Conditions on the Equator (s_z = 0) ===
phi_init = linspace(-pi, pi, num_trajectories);
initial_conditions = [cos(phi_init); sin(phi_init); zeros(1, num_trajectories)];

%% === Plotting: Bloch Sphere Setup ===
[X, Y, Z] = sphere(50);  % Unit sphere surface

figure;
hold on; grid on;
axis equal;

% Draw translucent Bloch sphere
surf(X, Y, Z, 'FaceAlpha', 1, 'EdgeColor', 'none', 'FaceColor', [0.8 0.8 0.8]);
xlabel('s_x'); ylabel('s_y'); zlabel('s_z');
title(' ');
view(45, 30);  % 3D view angle

%% === Evolve and Plot Each Trajectory ===
for traj = 1:num_trajectories
    s0 = initial_conditions(:, traj);

    % --- Forward evolution ---
    s_values_f = zeros(3, Nf);
    s_values_f(:,1) = s0;
    for i = 1:Nf-1
        s_values_f(:,i+1) = rk4_step(@derivatives, t_values_forward(i), s_values_f(:,i), h, epsilon, gamma, p);
    end

    % --- Backward evolution ---
    s_values_b = zeros(3, Nb);
    s_values_b(:,1) = s0;
    for i = 1:Nb-1
        s_values_b(:,i+1) = rk4_step(@derivatives, t_values_backward(i), s_values_b(:,i), -h, epsilon, gamma, p);
    end

    % --- Combine and plot trajectory ---
    % (fliplr used to order backward points correctly)
    s_values = [fliplr(s_values_b(:,2:end)), s_values_f];
    plot3(s_values(1,:), s_values(2,:), s_values(3,:), 'k', 'LineWidth', 1.2);
end
