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
%% Show peaks in plot
invBWRotated = imrotate(invBW, A);         % Rotate the image with the angle obtained
maxVals = sum(invBWRotated(:,:)');         % calculate the histogram by summing pixels horizontally
maxVals(maxVals < 300) = 0;         % Removes all noise below a certain threshold

[peaks, locs] = findpeaks(maxVals); % Find peak locations, stemlines y value = locs

plot(maxVals); hold on;
scatter(locs,peaks,'r');
xlabel('Staff line locations, Y-value');


%% Show found peak locations in image;
imshow(imrotate(Im,A)); hold on;
for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',2,'Color','red');
end


%% 
% Split the staffline positions into a matrix with [n x 5] dimensions 
% where n = number of stafflines

stafflineMatrix = vec2mat(locs,5);      % Split staffline locations into a matrix
barWidth = diff(stafflineMatrix,1,2);   % calculate difference in rows
barWidth = mean(mean(barWidth));        % Finally, mean the values in the matrix



%% Remove staff lines 

% TODO (Optimal recognition of music symbols, 3.2)
% "line track height algorithm with the stable path algorithm
% to remove staff lines"

%% Symbol segmentation (Optimal recognition of music symbols, 3.3)
% "localizing and isolating the symbols in order to identify them"

% TODO Beam Detection

% TODO Notes, notes with flags and notes open detection



%%

 