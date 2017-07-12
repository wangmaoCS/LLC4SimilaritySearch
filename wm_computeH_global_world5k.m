addpath ('./spam/spams-matlab-v2.6_precompiled/');
addpath ('./spam/spams-matlab-v2.6_precompiled/test_release/');
addpath ('./spam/spams-matlab-v2.6_precompiled/src_release/');
addpath ('./spam/spams-matlab-v2.6_precompiled/build/');

run start_spams;

data = 'world5k';
d = 512;                                    %d = 512 or 256
featdir = sprintf('rmac/%s/',data);         %path where the features are stored
groupdir = 'models/';                       %path where group representations are stored

% Load features
load(sprintf('%s/%s_rmac.mat',featdir,data))
load(sprintf('%s/gnd_%s.mat',featdir,data));

% Load the Y matrix in the paper
X = world5k_rmac_db;
Q = world5k_rmac_q;

% Exhaustive search
tic;
sim = Q'*X;                 %Elapsed time is 1.242364 seconds.
toc;
[basesc,bids] = sort(sim','descend');

% Exhaustive search mAP
[basemap,bap] = compute_map(bids,gnd,0); %0.6686
basemap  

M = 400;
m = 10;                                     % This controls the sparsity of the encoding matrix
N = size(X,2);

param.K=M; % learns a dictionary with M elements
param.lambda=0.1; 
param.lambda2=0;
param.mode=2;                    
param.numThreads=1; % number of threads
param.iter=100;
param.batch = true;
tic
Y = mexTrainDL(X,param);
toc;

Y = double(Y);

param1.L=10;
param1.eps=0.0;
param1.numThreads=-1;
tic;
H = mexOMP(X,Y,param1);
toc;

comp_ratio = ((numel(Y)) + (nnz(H))) / (d*N);

% Estimated similarities with approximate search
tic;
simhat = Q'*Y*H;            %Elapsed time is 0.008916 seconds.
toc;
[qsc,qpred]=sort(simhat','descend');

% Approximate search mAP
[map,ap] = compute_map(qpred,gnd,0);
map

%learn codebook of llc
tic;
Y_llc = yael_kmeans(single(X), M);
toc;
Y_llc = double(Y_llc);

tic;
H_llc = LLC_coding_appr_yael(Y_llc',X',m);
toc;
clear X;

%H_llc = sparse( H_llc' );
H_llc = sparse( H_llc );
H_llc = H_llc';

% Estimated similarities with approximate search
tic;
simhat_llc = Q'*Y_llc*H_llc;            
toc;
[qsc_llc,qpred_llc]=sort(simhat_llc','descend');

% Approximate search mAP
[map_llc,ap_llc] = compute_map(qpred_llc,gnd,0);
map_llc



% mAP(baseline) = 0.874
% mAP(SparseCode) = 0.9063, time = 5s and 0.16s, without batchsize
% mAP(SparseCode) = 0.9067, time = 33s and 0.16s, with batchsize

% mAP(LLC) = 0.9209, time = 0.45s and 0.68s
% mAP(LLC) = 0.9216, time = 0.28 and 0.73s
% mAP(LLC) = 0.9182, 0.9226, 0.9197, 0.9189, 0.9254, 0.9178,0.9177, 0.9176
