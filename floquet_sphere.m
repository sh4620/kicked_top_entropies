function floquet_su2_sphere()
% FLOQUET_SU2_SPHERE
% Computes and visualizes Husimi distributions of Floquet eigenmodes
% for a PT-symmetric kicked top on the SU(2) sphere.

    %% Parameters
    J     = 20;      % Total angular momentum quantum number
    p     = 2;       % Precession strength
    gamma = 0.0;     % Gain-loss parameter (PT-symmetry)
    k     = 4;       % Kicking strength
    tau   = 1;       % Time period
    L     = 2 * J + 1; % Hilbert space dimension

    %% Compute Floquet operator and eigenstates
    F = compute_floquet_operator_kicked_top(J, p, gamma, k, tau);
    [eigenstates, ~] = eig(F);

    %% Phase space grid for Husimi distribution
    grid_size = 140;
    sz  = linspace(-0.999, 0.999, grid_size); % Normalized angular momentum z
    phi = linspace(-pi, pi, grid_size);        % Azimuthal angle
    [SZ, PHI] = meshgrid(sz, phi);

    %% Unit sphere coordinates for 3D projection
    [sx_sphere, sy_sphere, sz_sphere] = sphere(100);

    %% Loop over Floquet modes and plot Husimi distributions
    for mode_idx = 1:L
        psi = eigenstates(:, mode_idx);

        % Compute Husimi distribution on the grid
        husimi = compute_nh_husimi_su2(psi, sz, phi, J);

        % ===== 2D Plot (φ vs s_z) =====
        figure('Position', [100 100 1200 500]);
        subplot(1, 2, 1);
        surf(PHI, SZ, husimi, 'linestyle', 'none');
        view(2);
        axis([-pi pi -1 1]);
        colormap parula;
        colorbar;
        title(['2D Husimi Distribution - Mode ', num2str(mode_idx)]);
        xlabel('\phi');
        ylabel('s_z');

        % ===== 3D Spherical Projection =====
        subplot(1, 2, 2);

        husimi_3d = zeros(size(sx_sphere));
        for i = 1:size(sx_sphere,1)
            for j = 1:size(sx_sphere,2)
                % Convert Cartesian to spherical coordinates
                [az, el, ~] = cart2sph(sx_sphere(i,j), sy_sphere(i,j), sz_sphere(i,j));
                sz_val = sin(el);
                phi_val = az;

                % Find closest grid indices
                [~, sz_idx] = min(abs(sz - sz_val));
                [~, phi_idx] = min(abs(phi - phi_val));

                % Assign Husimi value
                husimi_3d(i,j) = husimi(phi_idx, sz_idx);
            end
        end

        % Plot on unit sphere with log scaling for visibility
        surf(sx_sphere, sy_sphere, sz_sphere, log10(husimi_3d + 1e-6), ...
            'EdgeColor', 'none', 'FaceColor', 'interp');
        axis equal;
        colormap parula;
        colorbar;
        title(['3D Spherical Projection - Mode ', num2str(mode_idx)]);
        xlabel('s_x');
        ylabel('s_y');
        zlabel('s_z');
        view(40, 30);

        pause(1.5); % Pause to visualize each mode
    end
end

%% ================================
function F = compute_floquet_operator_kicked_top(J, p, gamma, k, tau)
% COMPUTE_FLOQUET_OPERATOR_KICKED_TOP
% Constructs the Floquet operator for the PT-symmetric kicked top

    [Jx, ~, Jz] = angular_momentum_matrices(J);

    % Half-period free evolution
    H_free = p * Jx + 1i * gamma * Jz;
    U_free = expm(-1i * H_free * (tau / 2));

    % Kicking term
    H_kick = (k / J) * Jz^2;
    U_kick = expm(-1i * H_kick);

    % Full Floquet operator
    F = U_free * U_kick * U_free;
end

%% ================================
function [Jx, Jy, Jz, J_plus, J_minus] = angular_momentum_matrices(J)
% ANGULAR_MOMENTUM_MATRICES
% Generates angular momentum operators for SU(2) algebra

    m = (-J:J)'; % Magnetic quantum numbers
    Jz = diag(-m);
    J_plus = diag(sqrt(J*(J+1) - m(1:end-1) .* (m(1:end-1) + 1)), 1);
    J_minus = diag(sqrt(J*(J+1) - m(2:end) .* (m(2:end) - 1)), -1);
    Jx = 0.5 * (J_plus + J_minus);
    Jy = -0.5i * (J_plus - J_minus);
end

%% ================================
function husimi = compute_nh_husimi_su2(psi, SZ, PHI, J)
% COMPUTE_NH_HUSIMI_SU2
% Computes the non-Hermitian SU(2) Husimi distribution for a state vector psi

    husimi = zeros(length(PHI), length(SZ));
    basis_vector = zeros(2 * J + 1, 1);
    basis_vector(1) = 1;  % |J, J> state

    for i = 1:length(SZ)
        for k = 1:length(PHI)
            zeta = compute_zeta(SZ(i), PHI(k), J);
            A_zeta = compute_A_zeta(zeta, J);
            coherent_state = A_zeta * basis_vector;
            husimi(k, i) = abs(coherent_state' * psi)^2;
        end
    end

    % Normalize Husimi distribution
    husimi = husimi / sum(husimi(:));
end

%% ================================
function zeta = compute_zeta(sz, phi, ~)
% COMPUTE_ZETA
% Computes the complex coordinate zeta on the Bloch sphere

    zeta = sqrt((1 - sz) / (1 + sz)) * exp(1i * phi);
end

%% ================================
function A_zeta = compute_A_zeta(zeta, J)
% COMPUTE_A_ZETA
% Constructs the coherent state displacement operator matrix A(zeta)

    [~, ~, ~, J_plus, J_minus] = angular_momentum_matrices(J);
    prefactor = (1 + abs(zeta)^2)^(-J);
    A_zeta = prefactor * expm(conj(zeta) * J_minus);
end
