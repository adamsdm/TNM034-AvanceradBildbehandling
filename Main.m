%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Read image and convert to double, grayscale format 
clear;
Im = rgb2gray( im2double( imread('./Testbilder/im1s.jpg') ) );

%% Remove staff lines 

% TODO (Optimal recognition of music symbols, 3.2)
% "line track height algorithm with the stable path algorithm
% to remove staff lines"

%% Symbol segmentation (Optimal recognition of music symbols, 3.3)
% "localizing and isolating the symbols in order to identify them"

% TODO Beam Detection

% TODO Notes, notes with flags and notes open detection



%%
imshow(Im);
 