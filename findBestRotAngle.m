function [ A ] = findBestRotAngle( invBW )
%findBestRotAngle - finds the best rotation angle
%   invBW = logical image
%   A     = the angle which gives best rotation with respect to
%           max peaks in histogram


% NEEDS IMPROVEMENT, Divide and conquer? 


A = -1.0;
maxPeak = 0;

for angle=-1.0:0.1:1.0
   rotIm = imrotate(invBW, angle);      % Rotate the image with angle from -1.0 - 1.0 degrees. 
   peak = max( sum(rotIm(:,:)' ) );   % Calculate the integral of the histogram.
   if(peak > maxPeak)
       maxPeak = peak;
       A = angle;
   end
end  

end

