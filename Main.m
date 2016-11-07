%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Read image and convert to double, grayscale format 
% PREPROCESSING
clear;
close all;
Im = imread('./Testbilder/im3s.jpg'); % Read the image
Im = im2double( Im ); % Convert image to double

level = graythresh(Im); % Computes the global threshold level from the image
BW = im2bw(Im,level);   % Convert the image to binary image with threshold: level

%BW2 = bwmorph(BW,'close',2);

imshow(BW);


%% Find staff lines
% HISTOGRAM
% Find best angle

close all;
invBW = BW<level;

A = findBestRotAngle(invBW);
rotIm = imrotate(invBW, A);
maxVals = sum(rotIm(:,:)');

plot(maxVals);




%% HOUGH
ib = imrotate(Im,10,'bilinear','crop');
imshow(ib);

BW = im2bw(ib,level);   % Convert the image to binary image with threshold: level
ib = BW<level;
imshow(ib);
[H, T, R] = hough(ib);

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
 