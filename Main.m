%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PREPROCESSING
clear all;
close all;
Im = imread('./Testbilder/im1s.jpg'); % Read the image
Im = im2double( Im ); % Convert image to double

level = graythresh(Im); % Computes the global threshold level from the image
BW = im2bw(Im,level);   % Convert the image to binary image with threshold: level

%BW2 = bwmorph(BW,'close',2);

imshow(Im);


%% Find staff lines
% HISTOGRAM
% Find best angle

close all;
invBW = BW<level;   % 

A = findBestRotAngle(Im);
%% Show peaks in plot
invBWRotated = imrotate(invBW, A,'bilinear');   % Rotate the image with the angle obtained

maxVals = sum(invBWRotated(:,:)');              % calculate the histogram by summing pixels horizontally
plot(maxVals);
%%
[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs

maxValsThresh = 0.3*max(max(peaks));
maxVals(maxVals < maxValsThresh) = 0;                     % Removes all noise below a certain threshold

[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs

plot(maxVals); hold on;
scatter(locs,peaks,'r');
xlabel('Staff line locations, Y-value');

%% Remove trash peaks from locs and peaks
[peaks, locs] = findpeaks(maxVals); % TEMP DELETE THIS LINE

medianBarWidth = median(diff(locs)); % Find median bar width
medianBarWidth = medianBarWidth+2; 

index=1;
while( index < length(locs-5)) 
    % Check if 5 next elements are withing range
    if( locs(index+1) - locs(index+0) < medianBarWidth && locs(index+2) - locs(index+1) < medianBarWidth && locs(index+3) - locs(index+2) < medianBarWidth &&  locs(index+4) - locs(index+3) < medianBarWidth ) 
        locs(index:index+4)
        index=index+5;    % check next 5 peaks
    else % peak is trash peak
        locs(index) = -1; % set peaks not accepted to negative value 
        index=index+1; % check next peak
    end
end

exclude = locs < 0; %Logicall array containing positions where locs is negative
locs(exclude) = [];
peaks(exclude) = [];

stem(locs,peaks,'r');




%% Show found peak locations in image;
imshow(imrotate(Im,A,'bilinear')); hold on;

for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
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
BWRotated = imrotate(BW,A,'bilinear');

se = zeros(3,3);
se(:,2)=1;

BWRotatedNoStaff = imclose(BWRotated,se);

imshow(BWRotated);
figure;
imshow(BWRotatedNoStaff);

%% Corrolation and template matching
template = imread('NoteheadTemplate.png');
template = im2bw(template,0.9);

imshow(template);

% resize height of image to 130% of barwidth
tempRez = imresize(template, (1/0.72)*[barWidth NaN]);
C = normxcorr2(tempRez, BWRotatedNoStaff);

%%
% Remove border from C which normxcorr2 adds
yoffset = (size(C,1)-size(BWRotatedNoStaff,1))
xoffset = (size(C,2)-size(BWRotatedNoStaff,2))

C = C(yoffset/2:size(C,1)-yoffset/2,:);
C = C(:, xoffset/2:size(C,2)-xoffset/2);



imshow(C)
for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
end
%%
close all;
CThresh = (C>0.85*max(max(C)));
imshow(CThresh);
for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
end
figure;
imshow(BWRotatedNoStaff);

%% Remove trash bounding boxes
% Remove by checking if there is a notehead inside the bounding box
invBWRotatedNoStaff = BWRotatedNoStaff<1;
st = regionprops(invBWRotatedNoStaff,'BoundingBox');
filteredSt = st;
acceptedSt = logical(zeros(1,length(st)))';

% For each bounding box
%for i = 1:length(st);
for i = 1:length(st);
    thisBB = filteredSt(i).BoundingBox;
    
    x = round(thisBB(1));
    y = round(thisBB(2));
    w = thisBB(3);
    h = thisBB(4);
    
    %   (x,y)-------(x+w,y)
    %     |            |
    %     |            |
    %     |            |
    %  (x,y+h)-----(x+w,y+h)
    
    % if bounding box in CThresh does NOT contains ones
    if( ~any(any(CThresh(y:y+h, x:x+w))))
        acceptedSt(i)=1;
    end
end

% Removes elements where acceptedSt = 1
filteredSt(acceptedSt) = [];
    

%% Plot bounding boxes in invBWRotatedNoStaff
imshow(invBWRotatedNoStaff);

for k = 1 : length(filteredSt)
    thisBB = filteredSt(k).BoundingBox;
    rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
        'EdgeColor','r','LineWidth',2  )

end




