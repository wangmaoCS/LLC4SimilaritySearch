% addpath yael library


%data = 'oxford5k';                          %oxford5k, oxford105k, paris6k or paris106k
%data = 'oxford105k';                          %oxford5k, oxford105k, paris6k or paris106k
data = 'paris6k';                          %oxford5k, oxford105k, paris6k or paris106k
%data = 'paris106k';                          %oxford5k, oxford105k, paris6k or paris106k

d = 512;                                    %d = 512 or 256
featdir = sprintf('rmac/%s/',data);         %path where the features are stored
groupdir = 'models/';                       %path where group representations are stored

% Load features
load(sprintf('%s/%s_rmca_query_dim%d.mat',featdir,data,d))
load(sprintf('%s/%s_rmca_data_dim%d.mat',featdir,data,d))
load(sprintf('%s/gnd_%s.mat',featdir,data));


% Set the number of groups for precomputed models
% we use the parameters from the CVPR2016 paper
switch data
    case 'oxford5k'
        M = 400;
    case 'oxford105k'
        M = 5257;
    case 'paris6k'
        M = 504;
    case 'paris106k'
        M = 10646;
    otherwise
        error('Undefined dataset. Please set data variable to one of the following: oxford5k, oxford105k, paris6k, or paris106k')
end

% Set variables
m = 10;                                     % This controls the sparsity of the encoding matrix
N = size(X,2);

X = double(X);
Q = X(:,gnd.qidx);


% Exhaustive search
tic;
sim = Q'*X;                 %Elapsed time is 1.242364 seconds.
toc;
[basesc,bids] = sort(sim','descend');

% Exhaustive search mAP
[basemap,bap] = compute_map(bids,gnd.gnd,0); %0.6686
basemap  


%learn codebook of llc
tic;    
Y_llc = yael_kmeans(single(X), M);
toc;
Y_llc = double(Y_llc);

tic;
H_llc = LLC_coding_appr_yael(Y_llc',X',m);
toc;
clear X;

H_llc = sparse( H_llc );
H_llc = H_llc';

% Estimated similarities with approximate search
tic;
simhat_llc = Q'*Y_llc*H_llc;            
toc;
[qsc_llc,qpred_llc]=sort(simhat_llc','descend');

% Approximate search mAP
[map_llc,ap_llc] = compute_map(qpred_llc,gnd.gnd,0);
map_llc

