% AUTHOR :  Scott Field 
%           sfield@umd.edu
%
% DATE: March 24, 2014
%
% PURPOSE: parameter file for surrogate model building.


%%% model name %%%
% modelName = 'SpEC'
modelName = 'EOBNRv2'

%%% export surrogate? %%%
export = true

%%% how many rb to use in surrogate? (more rb, more memory) %%%
surrogate_rb = 'all'
% surrogate_rb = 15

%%% Monte Carlo study of representation errors %%%
% Note: if true, MC_files must have a location
% MC_errors = true
MC_errors = false

%%% Use subset of training set (useful for convergence studies) %%%
subset = 'None'
% subset = 1:2:501;

%%%% Information about RB-Greedy %%%%
seed = 1; % First RB to be selected
tol = 1e-6 % tol^2 ~ overlap error



%%%%%%%%% TRAINING SET PROCESSING SCHEMES  %%%%%%%

%%% mode-by-mode waveform alignment in time %%%
% alignment = 'None'
% alignment = 'PeakOfLongest' %% shorter waveforms extended with zero
alignment = 'PeakOfShortest' %% longer waveforms shortened

%%% h_c + i h_c = A exp(i phi) -- how/where to align phi %%%
PhaseAlignment = 'initial'
% PhaseAlignment = 'merger' % Warning: could produce noisier waveforms
% PhaseAlignment = 'none'

%%% remove first XXX duration of waveform after alignment/chopping %%%
%%% Ex: to cut cycles or remove LIGO tappering (if applied)
remove_seconds = 0

%%% rule for removing waveform's end %%%
% FinalTime = 'NA'
% FinalTime = 'Merger' % cut waveform at merger (e.g. if phase non-smooth)
% FinalTime = 6 % remove final duration by fixed amount
FinalTime = 'NearTerminal' % revome a few waveform samples befor zero

if(strcmp(modelName,'SpEC'))
    
    %%% spline resample training set waveforms? (set common time below) %%%
    %%% Ex: raw SpEC waveforms not given on a common temporal grid
    resampleTSpline = true
    
    %%% find continuum tpeak (defined to be t=0) from spline %%%
    spline_peak = false % Note: EOB appears to be more sensitve to peak
    
    convert_time_to_M = false % times already in M (Q: which M?)
    
elseif(strcmp(modelName,'EOBNRv2'))
    
    %%%% information about HDF5 file %%%%
    %%% Assumes complex data of the form h=hp+ihc in a single hdf5 file
    DataType = 'complex'
    NumMode = 2;
    ParamDim = 1 % dimension of parameter space (length of parameter array)
    
    resampleTSpline = false
    convert_time_to_M = true % export surrogate in units of M (uses Mtot)
    
else
    
    error('model not supported')

end




%%%%%%%%% TRAINING SET LOCATION AND INFORMATION  %%%%%%%

if(strcmp(modelName,'SpEC'))
    
    %%% all training waveforms generated for rh at scri %%%
    MegaPS_TS = 'Scri'; 
    
    %%% case 1 (2,2) modes, q = [1,10] %%%
    ell_mode = 2; m_mode = 2;
    % TrainingSpaceDirectory='NR_TS/TS_JonathansRuns_withoutFirst5_run11_mislabled/' % Jonathan's runs without first 5 (run 11 mislabled q value)
    % TrainingSpaceDirectory='NR_TS/TS_JonathansRuns_withoutFirst5/' % Jonathan's runs without first 5 (run 11 mislabled q value)
    % TrainingSpaceDirectory='NR_TS/TS_Spec_NoSpin/' % Spec catalogue of all non-spinning (Using highest Lev and N=4 extrapolation)
    TrainingSpaceDirectory='NR_TS/TS_Jon_Catalogue/' % spec plus jons
    
    %%% case 2 (3,3) modes, q = [1,10] %%%
    % ell_mode = 3; m_mode = 3;
    % TrainingSpaceDirectory='/home/balzani57/GitRepos/Codes/SurrogateBuildIMR/NR_TS/TS_Jon_Catalogue_ell3_m3/' % spec plus jons
    
    TS_files = 'TSkey.txt'
    %%% conservative max/min interval, deltaT =.01 means relative inner
    %%% products about XXX error, below greedy error when basis = dim(TS)
    maxTime = 50; minTime = -2750; deltaT = .01;
    
    
    qmax_set = 9.98875; Mtot = 1;
    
    
elseif(strcmp(modelName,'EOBNRv2'))
    
    %%% all training waveforms generated at 1 mega PS %%%
    MegaPS_TS = 1; % use GetPhysicalConstants('LAL_MKS');
    ts_units = 'LAL_MKS';
    

    %%% absolute path where hdf5 file is located %%%
    TrainingSpaceDirectory='/home/balzani57/Desktop/Data_and_experiments/HDF5MatlabData/NewData/TwoMode/'
    % TrainingSpaceDirectory='/home/balzani57/Desktop/'
    % TrainingSpaceDirectory='/home/balzani57/GitRepos/Codes/RB_GW_Matlab/TrainingGreedyData/HDF5MatlabData/NewData/TwoMode/'
    
    % TS_files = {'EOB_old'}
    % TS_files = {'EOB_old_22'}
    % TS_files ={'EOBNRv2TD_Nq11_q1_2_fixedM80_fmin9_AdaptiveTrue_Sample2048_Down1_modeboth_MergerAlignTrue_HarmonicMode2'}
    % TS_files = {'EOBNRv2TD_Nq11_q1_2_fixedM80_fmin9_AdaptiveTrue_Sample2048_Down1_modeboth_MergerAlignTrue_HarmonicMode0'}
    
    %%% effective sampling rate of 2 kHz. merger resolution changes %%%
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_32768_16_modeBoth_MergerAligned'}
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_65536_32_modeBoth_MergerAligned'}
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_131072_64_modeBoth_MergerAligned'}
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_262144_128_modeBoth_MergerAligned'
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_524288_256_modeBoth_MergerAligned'
    %%%%%%%%%% case 1, see below %%%%%%%%%%%%%%
    
    
    %%% effective sampling from 512 to 65 kHz. same merger resolution %%%
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_1048576_2048_modeBoth_MergerAligned'  % 512
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_1048576_1024_modeBoth_MergerAligned'  % 1024
    %%%%%%%%%% case 1, see below %%%%%%%%%%%%%%
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_1048576_256_modeBoth_MergerAligned'   % 4096
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_1048576_128_modeBoth_MergerAligned'   % 8192
    % TrainingSpaceBaseFile='EOB_TD_501_q1_2_Mt80_fmin9_1048576_64_modeBoth_MergerAligned'    % 16384
    % memory constraints - can't load above 16384 effective sample rate)
    
    
    %%% uses adaptive fmin to maintain waveform's length of shortest %%%
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    
    
    % % %%%  q=1-10, 2048 hz, 12000M %%% (OLD - for testing. use case below)
    % % % TS_files = {'EOB_TD_1001_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % % % MC_files = {'EOB_TD_1101_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % % % MoreInfo = 'With these settings initial r < 20 M. EOB code MIGHT
    % modify waveform for small initial r.'
    
    
    
    
    %%% cases for table 1 of surroage paper. See exp script for TS density %%%
    %%% case 1 (q = 1-2, 2048 hz, 12240M) %%%
    qmax_set = 2; Mtot = 80;
    ell_mode = 2; m_mode = 2;
    TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned'}
    MC_files = {'EOB_TD_1011_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned'}
    MoreInfo = 'Case 1 A from paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 2 (q = 9-10, 2048 hz, 11103M) --> rmin < 20 M, EOB code r %%%
    % qmax_set = 10; Mtot = 80;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_501_q9_10_Mt80_fmin13_1048576_512_modeBoth_MergerAligned'}  % 2048
    % MC_files = {'EOB_TD_1011_q9_10_Mt80_fmin13_1048576_512_modeBoth_MergerAligned'}
    
    %%% case 3 (q = 1-4, 2048 hz, 12240M)  %%%  (TS build: 7.4 hours for 2001)
    % qmax_set = 4; Mtot = 80;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_2001_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_1101_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    
    %%% case 4 (q = 1-6, 2048 hz, 12240M)  %%%
    % qmax_set = 6; Mtot = 80;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_2001_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_1101_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    
    %%% case 5 (q = 1-8, 2048 hz, 12240M)  %%%
    % qmax_set = 8; Mtot = 80;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_2001_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_1101_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    
    %%% case 6 (q = 1-10, 2048 hz, 12240M)  --> rmin < 20 M, EOB code small r_0  %%%
    % qmax_set = 10; Mtot = 80;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_2001_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_1101_q1_10_Mt80_fmin9Adaptive_1048576_512_modeBoth_MergerAligned'}
    % MoreInfo = 'Case 6 from the paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 7 (q = 1-2, 2048 hz, 80750M)
    % qmax_set = 2; Mtot = 40;
    % ell_mode = 2; m_mode = 2;
    % TS_files = {'EOB_TD_501_q1_2_Mt40_fmin9Adaptive_262144_128_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_511_q1_2_Mt40_fmin9Adaptive_262144_128_modeBoth_MergerAligned'}
    
    %%% case 8 (q = 1-2, 2048 hz, 191840M)
    % qmax_set = 2; Mtot = 29;
    % ell_mode = 2; m_mode = 2;
    % subsetMC = 1:3:1001; % subset of random set for memory
    % TS_files = {'EOB_TD_501_q1_2_Mt29_fmin9Adaptive_262144_128_modeBoth_MergerAligned'}
    % MC_files = {'EOB_TD_1001_q1_2_Mt29_fmin9Adaptive_262144_128_modeBoth_MergerAligned'}
    % MoreInfo = 'Case 8 from the paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 9 (q = 1-2, 2048 hz, 12240M, (2,1))
    % qmax_set = 2; Mtot = 80;
    % ell_mode = 2; m_mode = 1;
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell2_m1'}
    % MC_files = {'EOB_TD_1011_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell2_m1'}
    % MoreInfo = 'Case 9 from the paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 10 (q = 1-2, 2048 hz, 12240M, (3,3))
    % qmax_set = 2; Mtot = 80;
    % ell_mode = 3; m_mode = 3;
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell3_m3'}
    % MC_files = {'EOB_TD_1011_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell3_m3'}
    % MoreInfo = 'Case 10 from the paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 11 (q = 1-2, 2048 hz, 12240M, (4,4))
    % qmax_set = 2; Mtot = 80;
    % ell_mode = 4; m_mode = 4;
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell4_m4'}
    % MC_files = {'EOB_TD_1011_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell4_m4'}
    % MoreInfo = 'Case 11 from the paper Fast prediction and evaluation of gravitational waveforms using surrogate models'
    
    %%% case 12 (q = 1-2, 2048 hz, 12240M, (5,5))
    % qmax_set = 2; Mtot = 80;
    % ell_mode = 5; m_mode = 5;
    % TS_files = {'EOB_TD_501_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell5_m5'}
    % MC_files = {'EOB_TD_1011_q1_2_Mt80_fmin9_1048576_512_modeBoth_MergerAligned_ell5_m5'}
    % MoreInfo = 'Case 12 from the paper Fast prediction and evaluation of
    % gravitational waveforms using surrogate models'
    
    
else
    
    error('model not supported')

end
