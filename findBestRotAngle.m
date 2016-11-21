function [ A ] = findBestRotAngle( Im )
%findBestRotAngle - finds the rotation angle which gives the biggest peaks
%of the histogram
%   invBW = logical black and white image
%   A     = the angle which gives best rotation with respect to
%           max peaks in histogram


% NEEDS IMPROVEMENT, Divide and conquer? 


A = -1.0;
maxPeak = 0;

X = zeros(1,20);
count = 1;

Im = rgb2gray(Im);

[r,c] = size(Im);
Im = ones(r,c)-Im;
T=graythresh(Im);

for angle=-1.0:0.1:1.0
   
   rotIm = imrotate(Im, angle, 'bicubic');      % Rotate the image with angle from -1.0 - 1.0 degrees. 
   % Rotate Im instead and convert to invBW
%    invBW = im2bw(rotIm, graythresh(rotIm));
   invBW = rotIm > T;
   imshow(invBW);
   peak = max( sum(invBW(:,:)' ) );   % Calculate the integral of the histogram.
   X(count) = peak;
   if(peak > maxPeak)
       maxPeak = peak;
       A = angle;
   end
   count = count +1;
end  
size(X)
plot(-1.0:0.1:1.0,X); %uncomment count and X to get a plot for the peak
%values with corresponding histogram angle.

end

