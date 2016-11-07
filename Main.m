%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Read image and convert to double, grayscale format 
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
% Histogram plot of sum of all pixels horizontally
% or hough transform

%% HISTOGRAM
close all;
invBW = BW<level;

%% Find best angle
A = -1.0;
maxValsum = 0;

for angle=-1.0:0.1:1.0
   rotIm = imrotate(invBW, angle);      % Rotate the image with angle from -1.0 - 1.0 degrees. 
   valsum = sum( sum(rotIm(:,:)' ) );   % Calculate the integral of the histogram.
   if(valsum > maxValsum )
       maxValsum = valsum;
       A = angle;
   end
end  

invBW = imrotate(invBW, A);
maxVals = sum( invBW(:,:)' );

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
 