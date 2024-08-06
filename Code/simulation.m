clc
clearvars

% Constants
EA = 1e-6;
mu = 3.7e-3; 
dt = 1e-14; 
total_time = 1; 
T = 273; 
k_B = 1.380649e-23; 
A_values = 4;%[5,4.8, 4.5, 4.2, 4, 3.8, 3.5, 3.2, 3 ,2.8,2.6,2.4];
B_values = 2;%;[4.2, 4, 3.8, 3.5, 3.2, 3 ,2.8,2.6,2.4, 2.1,1.8, 1.5];
%EA_mem = [A;B;B;B;B;A;A;B;B;B;B;A;A;A;A;A;A;A;A;A;B;B;B;B]*EA;

% Number of A values
numA = length(A_values);
% Number of B values
numB = length(B_values);
% Total number of pairs
numPairs = numA * numB;


% Defining truss geometry and member properties
node = [0 0 0; 0 2 0; 2 0 0; 2 2 0; 0 0 2; 0 2 2; 2 0 2; 2 2 2]*1e-9;
member = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4;5 6;5 7;5 8;6 7;6 8;7 8;1 5;2 6;1 6;2 5;4 8;3 7;4 7;3 8;2 8;4 6;1 7;3 5];
nel = size(member, 1);
nnodes = size(node, 1);


% Rotation matrix for 45 degrees around Z-axis
theta = pi / 4; % 45 degrees in radians
Rz = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];

% Apply rotation to each node
node = (Rz * node')'; % Transpose to match dimensions and then transpose back

% Initializing displacements and internal loads
u = zeros(nnodes, 3);
F = zeros(nnodes, 3);
zeta_vec = zeros(nnodes,1); % zetas for each node

current_positions = node; % Current positions
next_positions = node;


%% Bin initialization

% Example estimated range for eigenvalues
estimated_min_eigenvalueX = -2e-17; % Adjust based on your knowledge
estimated_max_eigenvalueX = -1e-17; % Adjust based on your knowledge
estimated_min_eigenvalueY = 3e-17; % Adjust based on your knowledge
estimated_max_eigenvalueY = 5e-17; % Adjust based on your knowledge

% Define fixed edges based on estimated range
num_bins = 500; % Number of bins you want
edgesX = linspace(estimated_min_eigenvalueX, estimated_max_eigenvalueX, num_bins + 1);
edgesY = linspace(estimated_min_eigenvalueY, estimated_max_eigenvalueY, num_bins + 1);

results(numPairs) = struct('A', [], 'B', [], 'BinCounts', [], 'XEdges', [], 'YEdges', []);

%% Calculation of zeta(i) out of time step loop

for i = 1:nnodes % Loop over each node

    x0 = current_positions(i,:); % current_positions is initialized with the node matrix
    P_m = 0;
    %Finding nodes connected to the current node
    connectedMembers = any(member == i, 2);
    connectedNodesIndices = member(connectedMembers, :);
    connectedNodesIndices(connectedNodesIndices == i) = [];
    uniqueConnectedNodes = unique(connectedNodesIndices);

    for j = 1:length(uniqueConnectedNodes) %Looping over connected nodes
        
        x1 = current_positions(uniqueConnectedNodes(j),:);
        elementVector = x1 - x0;
        P_m = P_m + norm(elementVector);
    end
    
    % Calculate zeta
    zeta_vec(i) = 6 * pi * mu * P_m / 2;

end
%% Loop through time steps

num_steps = round(total_time / dt);

pairIndex = 0;

for A_index = 1:length(A_values)

    u = zeros(nnodes, 3);
    F = zeros(nnodes, 3);
    pairIndex = pairIndex + 1; % Increment the pair index
    display(pairIndex)
    eigenvalues_list = {};

    for k = 1:5e6
        % Update nodal positions 
        for i = 1:nnodes
            FF = F(i,:);
            
            % thermal fluctuation (external) force (f)
            f = sqrt(24 * k_B * T * zeta_vec(i) / dt) * (randn(1, 3)); % Maybe changes in the later
            
            %displacement of the node
            delta = (FF + f) * (1 / zeta_vec(i)) * dt;
            %updating position for node i at the next time step
            x0 = current_positions(i,:);
            next_positions(i,:) = x0 + delta;
            
        end
    
        %calculate pel and n 
        F = zeros(nnodes,3);
        for el=1:nel
    
            a = member(el, 1);
            b = member(el, 2);
    
            x0a = node (a,:); % current_positions(a,:); %%%% this could change alot
            x0b = node (b,:); % current_positions(b,:);
    
            l0 = norm(x0b-x0a);
    
            xa = next_positions(a,:); % new segment lengths
            xb = next_positions(b,:); 
    
            xdiff = xb - xa;
            l = norm(xdiff);
            n = xdiff / l;

            A = A_values(A_index);
            B = B_values(A_index);
    
            EA_mem = construct_EA(A,B, EA);
            
            Pel = EA_mem(el) * (l - l0)/l0; % Internal force in the member
    
            F(a, :) = F(a, :) + Pel * n;
            F(b, :) = F(b, :) - Pel * n;
    
        end
       
        % Every 1000 iterations, update the animation and calculate eigenvalues
        if mod(k, 1e3) == 0
            % Calculate distance matrix
            distance_matrix = pdist2(current_positions(:,1:2), current_positions(:,1:2));
            % Calculate eigenvalues of the square of distance matrices
            eigenvalues = eig(distance_matrix.^2);
            % Store eigenvalues
            eigenvalues_list{end+1} = eigenvalues;

        end

        if mod(k, 5e5) == 0

            display(k/ 5e5 *10);

        end

    
        % Update positions for the next iteration
        current_positions = next_positions;
    end


    all_eigenvalues = cell2mat(eigenvalues_list);

    num_eigenvalues_per_set = size(eigenvalues_list{1}, 1);
    
    
    first_eigenvalues = all_eigenvalues(1:num_eigenvalues_per_set:end);
    eighth_eigenvalues = all_eigenvalues(8:num_eigenvalues_per_set:end);

    [N, XEdges, YEdges] = histcounts2(first_eigenvalues, eighth_eigenvalues, edgesX, edgesY);

    % Store the results for this pair
    results(pairIndex).A = A_values(A_index); % Adjust as necessary
    results(pairIndex).B = B_values(B_index); % Adjust as necessary
    results(pairIndex).BinCounts = N;
    results(pairIndex).XEdges = edgesX; % Fixed edges used for all
    results(pairIndex).YEdges = edgesY; % Fixed edges used for all

end 

%%
save('results__test');

%%
function EA_mem = construct_EA (A,B, EA)

EA_mem = [A;B;B;B;B;A;A;B;B;B;B;A;A;A;A;A;A;A;A;A;B;B;B;B]*EA;


end