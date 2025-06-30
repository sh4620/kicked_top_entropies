
%% === Parameters ===
% System parameters
e = 0;       % Epsilon
g = 0.5;     % gamma
p = 1;       

% Time integration parameters
t_end = 5;             % Final time
dt = 0.01;             % Time step
tspan_forward  = 0:dt:t_end;     % Forward time span
tspan_backward = 0:-dt:-t_end;   % Backward time span

% Number of initial conditions to sample
N_initial = 150;

% Define the non-Hermitian Hamiltonian
H = [e + 1i*g,  p; 
     p,       -(e + 1i*g)];

% Initialize arrays to store all phase-space points
sz_all  = [];  % s_z values over time
phi_all = [];  % phi values over time

%% === Helper Function ===
% Enforces normalization of ψ by subtracting the projection onto ψ itself
function dpsi = normalize_psi(dpsi_raw, psi)
    dpsi = dpsi_raw - psi * (psi' * dpsi_raw);  % Ensures d/dt(|ψ|²) = 0
end

%% === Generate Random Initial Conditions ===
% sz ∈ [-1, 1], ϕ ∈ [-π, π]
sz_initial  = 2 * rand(N_initial, 1) - 1;
phi_initial = 2 * pi * rand(N_initial, 1) - pi;

%% === Time Evolution for Each Initial State ===
for i = 1:N_initial
    % Convert (sz, phi) to normalized 2D quantum state |ψ⟩
    psi1 = sqrt((1 + sz_initial(i)) / 2);
    psi2 = sqrt((1 - sz_initial(i)) / 2) * exp(1i * phi_initial(i));
    psi0 = [psi1; psi2];

    % --- Forward evolution ---
    [~, psi_forward] = ode45(@(t, psi) normalize_psi(-1i * H * psi, psi), tspan_forward, psi0);
    sz_fwd  = abs(psi_forward(:,1)).^2 - abs(psi_forward(:,2)).^2;
    phi_fwd = angle(psi_forward(:,2) ./ psi_forward(:,1));
    
    % Store results
    sz_all  = [sz_all;  sz_fwd];
    phi_all = [phi_all; phi_fwd];

    % --- Backward evolution ---
    [~, psi_backward] = ode45(@(t, psi) normalize_psi(-1i * H * psi, psi), tspan_backward, psi0);
    sz_bwd  = abs(psi_backward(:,1)).^2 - abs(psi_backward(:,2)).^2;
    phi_bwd = angle(psi_backward(:,2) ./ psi_backward(:,1));
    
    % Store results
    sz_all  = [sz_all;  sz_bwd];
    phi_all = [phi_all; phi_bwd];
end

%% === Phase-Space Plot ===
figure;
scatter(phi_all, sz_all, 1, 'k', 'filled', 'MarkerFaceAlpha', 0.3);  % Scatter plot in black
xlabel('\phi');
ylabel('s_z');
title(' ');
xlim([-pi, pi]);
ylim([-1, 1]);
grid off;

% Custom tick labels for better readability
xticks(-pi:pi/2:pi);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
