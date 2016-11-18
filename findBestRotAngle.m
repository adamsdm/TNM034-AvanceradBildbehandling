function [ A ] = findBestRotAngle( invBW )
%findBestRotAngle - finds the rotation angle which gives the biggest peaks
%of the histogram
%   invBW = logical black and white image
%   A     = the angle which gives best rotation with respect to
%           max peaks in histogram


% NEEDS IMPROVEMENT, Divide and conquer? 


A = -1.0;
maxPeak = 0;
%X = zeros(1,20);
%count = 1;
for angle=-1.0:0.1:1.0
   rotIm = imrotate(invBW, angle);      % Rotate the image with angle from -1.0 - 1.0 degrees. 
   % Rotate Im instead and convert to invBW
   peak = max( sum(rotIm(:,:)' ) );   % Calculate the integral of the histogram.
   %X(count) = peak;
   if(peak > maxPeak)
       maxPeak = peak;
       A = angle;
   end
   %count = count +1;
end  
%size(X)
%plot(-1.0:0.1:1.0,X); %uncomment count and X to get a plot for the peak
%values with corresponding histogram angle.

end

