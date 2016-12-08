function [ A ] = findBestRotAngle( Im )
%findBestRotAngle - finds the rotation angle which gives the biggest peaks
%of the histogram
%   invBW = logical black and white image
%   A     = the angle which gives best rotation with respect to
%           max peaks in histogram


% NEEDS IMPROVEMENT, Divide and conquer? 


A = -10.0;
maxPeak = 0;

X = zeros(1,20);
count = 1;

Im = rgb2gray(Im);

[r,c] = size(Im);
Im = ones(r,c)-Im;
T=graythresh(Im);

for angle1=-20:1:20 %find in which span the peak will be in
    rotIm = imrotate(Im, angle1, 'bicubic');
    invBW = rotIm > T;
    %imshow(invBW);
    peak = max( sum(invBW(:,:)' ) ); 
    
   % X(count) = peak;
    if(peak > maxPeak)
       maxPeak = peak;
       A = angle1;
    end
   % count = count +1;
end

%plot(-10:1:10,X);
maxPeak = 0.0;

for angle2=A-1.0:0.1:A+1.0
        rotIm = imrotate(Im, angle2, 'bicubic');
    invBW = rotIm > T;
    %imshow(invBW);
    peak = max( sum(invBW(:,:)' ) ); 
    
    if(peak > maxPeak)
       maxPeak = peak;
       A = angle2;
    end
    
end

maxPeak = 0.0;
for angle3=A-0.1:0.01:A+0.1
    rotIm = imrotate(Im, angle3, 'bicubic');
    invBW = rotIm > T;
    %imshow(invBW);
    peak = max( sum(invBW(:,:)' ) ); 
    
    if(peak > maxPeak)
       maxPeak = peak;
       A = angle3
    end
    
end

end

