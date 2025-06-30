function floquet_su2()
    % Parameters for the PT-symmetric kicked top
    J = 80;              % Total angular momentum quantum number
    p = 2;               % Precession strength
    gamma = 0.01;        % Gain-loss parameter
    k = 1;               % Kicking strength
    tau = 1;             % Time period

    % Plot options
    plot_left_eigenstates = false;
    plot_combined_husimi = true;
    plot_individual_modes = false;

    % Initialize system: get Floquet operator and eigenstates
    [F, right_eigenstates, left_eigenstates] = initialize_system(J, p, gamma, k, tau, plot_left_eigenstates);

    % Phase space grid setup: phi first (x-axis), then sz (y-axis)
    grid_size = 140;
    phi = linspace(-pi, pi, grid_size);          % Azimuthal angle (horizontal axis)
    sz = linspace(-0.999, 0.999, grid_size);     % Normalized angular momentum z (vertical axis)

    % Precompute coherent states for efficiency
    coherent_states = precompute_coherent_states(phi, sz, J);

    % Plot Husimi distributions for individual modes if enabled
    if plot_individual_modes
        plot_all_husimi(right_eigenstates, left_eigenstates, phi, sz, coherent_states, plot_left_eigenstates);
    end

    % Plot combined Husimi distribution if enabled
    if plot_combined_husimi
        plot_combined_distribution(right_eigenstates, phi, sz, coherent_states);
    end
end

function [F, right_eigenstates, left_eigenstates] = initialize_system(J, p, gamma, k, tau, get_left_states)
    % Generate angular momentum matrices
    [Jx, ~, Jz] = angular_momentum_matrices(J);

    % Free evolution operator for half the time period
    H_free = p * Jx + 1i * gamma * Jz;
    U_free = expm(-1i * H_free * (tau / 2));

    % Kick operator
    H_kick = (k / J) * Jz^2;
    U_kick = expm(-1i * H_kick);

    % Full Floquet operator
    F = U_free * U_kick * U_free;

    % Compute right eigenstates
    [right_eigenstates, ~] = eig(F);

    left_eigenstates = [];
    if get_left_states
        % Compute left eigenstates (eigenvectors of F')
        [left_eigenstates, ~] = eig(F');
        left_eigenstates = conj(left_eigenstates);

        % Normalize left and right eigenstates biorthogonally
        for i = 1:size(right_eigenstates, 2)
            norm_factor = left_eigenstates(:, i)' * right_eigenstates(:, i);
            left_eigenstates(:, i) = left_eigenstates(:, i) / norm_factor;
        end
    end
end

function coherent_states = precompute_coherent_states(phi, sz, J)
    % Precompute all coherent states for each (phi, sz) grid point
    [~, ~, ~, ~, J_minus] = angular_momentum_matrices(J);
    n_phi = length(phi);
    n_sz = length(sz);

    coherent_states = cell(n_phi, n_sz);  % Cell array indexed by (phi, sz)
    basis_vector = zeros(2 * J + 1, 1);
    basis_vector(1) = 1;  % Highest weight state |J, J>

    for i = 1:n_phi
        for j = 1:n_sz
            zeta = compute_zeta(sz(j), phi(i), J);
            A_zeta = compute_A_zeta(zeta, J_minus, J);
            coherent_states{i, j} = A_zeta * basis_vector;
        end
    end
end

function plot_all_husimi(right_eigenstates, left_eigenstates, phi, sz, coherent_states, plot_left)
    % Plot Husimi distributions for all modes
    L = size(right_eigenstates, 2);

    for mode_idx = 1:L
        figure;

        % Compute Husimi for right eigenstate
        psi_right = right_eigenstates(:, mode_idx);
        husimi_right = compute_all_husimi(psi_right, coherent_states);

        if plot_left
            % Plot right eigenstate Husimi
            subplot(1, 2, 1);
            plot_single_husimi(phi, sz, husimi_right, ['Right Mode ', num2str(mode_idx)]);

            % Compute and plot left eigenstate Husimi
            psi_left = left_eigenstates(:, mode_idx);
            husimi_left = compute_all_husimi(psi_left, coherent_states);

            subplot(1, 2, 2);
            plot_single_husimi(phi, sz, husimi_left, ['Left Mode ', num2str(mode_idx)]);
        else
            % Plot only right eigenstate Husimi
            plot_single_husimi(phi, sz, husimi_right, ['Mode ', num2str(mode_idx)]);
        end

        pause(1.5);
    end
end

function plot_combined_distribution(right_eigenstates, phi, sz, coherent_states)
    % Compute and plot the average Husimi distribution over all modes
    L = size(right_eigenstates, 2);
    combined_husimi = zeros(length(phi), length(sz));

    for mode_idx = 1:L
        psi = right_eigenstates(:, mode_idx);
        husimi = compute_all_husimi(psi, coherent_states);
        combined_husimi = combined_husimi + husimi;
    end

    combined_husimi = combined_husimi / L;  % Average over modes

    figure;
    plot_single_husimi(phi, sz, combined_husimi, 'Combined Husimi Distribution');
end

function husimi = compute_all_husimi(psi, coherent_states)
    % Compute Husimi distribution for given eigenstate psi over the grid of coherent states
    n_phi = size(coherent_states, 1);
    n_sz = size(coherent_states, 2);
    husimi = zeros(n_sz, n_phi);  % Note switched dimensions for correct plotting

    for j = 1:n_sz  % sz index (rows)
        for i = 1:n_phi  % phi index (columns)
            husimi(j, i) = abs(coherent_states{i, j}' * psi)^2;
        end
    end

    husimi = husimi / sum(husimi(:));  % Normalize
end

function plot_single_husimi(phi, sz, husimi, title_str)
    % Plot Husimi distribution with correct axis orientation
    imagesc(phi, sz, husimi);
    axis xy;               % Flip y-axis to match coordinate system
    colorbar;
    title(title_str);
    xlabel('\phi');
    ylabel('s_z');
    axis([-pi pi -1 1]);
end

function [Jx, Jy, Jz, J_plus, J_minus] = angular_momentum_matrices(J)
    % Construct angular momentum operators for SU(2) in J basis
    m = (-J:J)';
    Jz = diag(-m);
    J_plus = diag(sqrt(J * (J + 1) - m(1:end-1) .* (m(1:end-1) + 1)), 1);
    J_minus = diag(sqrt(J * (J + 1) - m(2:end) .* (m(2:end) - 1)), -1);
    Jx = 0.5 * (J_plus + J_minus);
    Jy = -0.5i * (J_plus - J_minus);
end

function zeta = compute_zeta(sz, phi, J)
    % Calculate the complex coordinate zeta for coherent states parameterization
    zeta = sqrt((1 - sz) / (1 + sz)) * exp(1i * phi);
end

function A_zeta = compute_A_zeta(zeta, J_minus, J)
    % Compute coherent state transformation matrix A(zeta)
    prefactor = (1 + abs(zeta)^2)^(-J);
    A_zeta = prefactor * expm(conj(zeta) * J_minus);
end
