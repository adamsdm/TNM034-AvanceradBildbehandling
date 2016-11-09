%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PREPROCESSING
clear;
close all;
Im = imread('./Testbilder/im1s.jpg'); % Read the image
Im = im2double( Im ); % Convert image to double

level = graythresh(Im); % Computes the global threshold level from the image
BW = im2bw(Im,level);   % Convert the image to binary image with threshold: level

%BW2 = bwmorph(BW,'close',2);

imshow(BW);


%% Find staff lines
% HISTOGRAM
% Find best angle

close all;
invBW = BW<level;   % 

A = findBestRotAngle(invBW);
%%
rotIm = imrotate(invBW, A);         % Rotate the image with the angle obtained
maxVals = sum(rotIm(:,:)');         % calculate the histogram by summing pixels horizontally
maxVals(maxVals < 300) = 0;         % Removes all noise below a certain threshold

[peaks, locs] = findpeaks(maxVals); % Find peak locations, stemlines y value = locs

plot(maxVals); hold on;
scatter(locs,peaks,'r');

%%



%% Remove staff lines 

% TODO (Optimal recognition of music symbols, 3.2)
% "line track height algorithm with the stable path algorithm
% to remove staff lines"

%% Symbol segmentation (Optimal recognition of music symbols, 3.3)
% "localizing and isolating the symbols in order to identify them"

% TODO Beam Detection

% TODO Notes, notes with flags and notes open detection



%%

 