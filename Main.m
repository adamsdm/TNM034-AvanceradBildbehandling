%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PREPROCESSING
clear all;
close all;
Im = imread('./Testbilder/im1c.jpg'); % Read the image
Im = im2double(Im);
% Normalize image background (dynamically) by dividing with local maximum
ImGrayDouble = rgb2gray(Im);
func = @(block_struct) (block_struct.data./max(max(block_struct.data)));
Im2 = blockproc(ImGrayDouble, [9 9], func);

level = graythresh(Im2); % Computes the global threshold level from the image
BW = im2bw(Im2,level);   % Convert the image to binary image with threshold: level

imshow(Im2);
figure;
imshow(BW);


%% Masking
[ySize, xSize] = size(Im);
xSize = xSize/3;
diff = 0.2;
ImGrayTest = rgb2gray(Im);

% Mask the area of the notesheet
for i = 1:xSize -1
    for j = 1:ySize -1
        if(abs(Im(j, i, 1) - Im(j, i, 2)) < diff && abs(Im(j, i, 1) - Im(j, i, 3)) < diff)
            ImGrayTest(j, i) = 0;
        end
    end
end
%%
resMask = ImGrayTest > 0;
se = strel('line',11,90);
resMask = imerode(resMask, se);
resMask = imdilate(resMask, se);

points = detectHarrisFeatures(resMask);
points = points.Location; %Extract the coordinates

% Find corner points by finding the points closest to the corners of the
% whole image

resMaskInv = abs(resMask-1);
resMaskPAD = padarray(resMaskInv,[20 20]);

se1 = strel('disk',3);

resMaskPAD = imopen(resMaskPAD, se1);
resMaskPAD = imclose(resMaskPAD, se1);

[y,x] = size(resMaskPAD);


C = corner(resMaskPAD,'Harris');
imshow(resMaskPAD); hold on;
plot(C(:,1), C(:,2), '*');

%%
for p=1:length(C)
    thisX = C(p,1);
    thisY = C(p,2);
    
    
    C(p,3)=pdist([0, 0; thisX, thisY], 'euclidean');       %top left
    C(p,4)=pdist([x, 0; thisX, thisY], 'euclidean');       %top right
    C(p,5)=pdist([0, y; thisX, thisY], 'euclidean');       %bottom left
    C(p,6)=pdist([x, y; thisX, thisY], 'euclidean');       %bottom right
end

[~, ITL] = min(C(:,3)) ;
[~, ITR] = min(C(:,4)) ;
[~, IBL] = min(C(:,5)) ;
[~, IBR] = min(C(:,6)) ;

topLeft = C(ITL, 1:2);
topRigt = C(ITR, 1:2);
botLeft = C(IBL, 1:2);
botRigt = C(IBR, 1:2);

%%

imshow(resMaskPAD);
hold on
plot(topLeft(1),topLeft(2), 'r*');
plot(topRigt(1),topRigt(2), 'r*');
plot(botLeft(1),botLeft(2), 'r*');
plot(botRigt(1),botRigt(2), 'r*');

imshow(diff);

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
xlabel('y-position');
ylabel('Peak height');
%%
[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs

maxValsThresh = 0.3*max(max(peaks));
maxVals(maxVals < maxValsThresh) = 0;           % Removes all noise below a certain threshold

[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs

plot(maxVals); hold on;
scatter(locs,peaks,'r');
xlabel('Staff line locations, Y-value');

%% If two peaks are very close together, only keep the max of the two peaks

medianBarWidth = median(diff(locs)); % Find median bar width
medianBarWidth = medianBarWidth+2; 

indToRemove = zeros(length(peaks));
% Compare peak 1 to (length-1)
for peakInd = 1:length(peaks)-1
    % if dx<medBarW/2
    if locs(peakInd+1)-locs(peakInd) < medianBarWidth/2
        
        if peaks(peakInd)<peaks(peakInd+1)
          indToRemove(peakInd) = 1;
        elseif peaks(peakInd)>peaks(peakInd+1)
          indToRemove(peakInd+1) = 1;
        end
        
    end
end

indToRemove = logical(indToRemove);
locs(indToRemove) = [];
peaks(indToRemove) = [];

stem(locs,peaks);


%% Remove trash peaks from locs and peaks

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
rotIm = imrotate(Im,A,'bilinear');
imshow(rotIm); hold on;

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
BWRotated = imrotate(BW,A,'bicubic');

se = zeros(3,3);
se(:,2)=1;

BWRotatedNoStaff = imclose(BWRotated,se);

imshow(BWRotated);
figure;
imshow(BWRotatedNoStaff);

%% Corrolation and template matching

% GClef
gClefTemplate = imread('GClefTemplate.png');
gClefTemplate = im2bw(gClefTemplate,0.9);
gClefTempRez = imresize(gClefTemplate, 8*[barWidth NaN]);

threshValue = 0.5;
threshStepSize = 0.03;

C = normxcorr2(gClefTempRez, BWRotatedNoStaff);
CThresh = (C>threshValue*max(max(C)));
gClefsCentroids= regionprops(CThresh, 'centroid');


while length(gClefsCentroids) > size(stafflineMatrix,1)
    
    threshValue = threshValue+threshStepSize;
    C = normxcorr2(gClefTempRez, BWRotatedNoStaff);
    CThresh = (C>threshValue*max(max(C)));
    gClefsCentroids= regionprops(CThresh, 'centroid');
    %imshow(CThresh);
    %pause(0.4);
end





% Find mean x position of gClefs
gClefXMean = 0;
for p = 1:length(gClefsCentroids)
    gClefXMean = gClefXMean+gClefsCentroids(p).Centroid(1);
end

gClefXMean = int16(gClefXMean/length(gClefsCentroids)+1*barWidth);


%%
noteTemplate = imread('NoteheadTemplate.png');
noteTemplate = im2bw(noteTemplate,0.9);

% resize height of image to 130% of barwidth
noteTempRez = imresize(noteTemplate, (1/0.72)*[barWidth NaN]);


imshow(BWRotatedNoStaff)
C = normxcorr2(noteTempRez, BWRotatedNoStaff);

%%
% Remove border from C which normxcorr2 adds
yoffset = int16((size(C,1)-size(BWRotatedNoStaff,1))/2)
xoffset = int16((size(C,2)-size(BWRotatedNoStaff,2))/2)



C = C(yoffset:size(C,1)-yoffset,:);
C = C(:, xoffset:size(C,2)-xoffset);


for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
end
%%
close all;
CThresh = (C>0.78*max(max(C)));

CThresh(:,1:gClefXMean) = 0;

temp = 

imshow(CThresh);
for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
end
figure;
imshow(BWRotatedNoStaff);



invBWRotatedNoStaff = BWRotatedNoStaff<1;

%%
 s= regionprops(CThresh, 'centroid');

centroids = [s.Centroid];
x = centroids(1:2:end-1);
y = centroids(2:2:end);

imshow(CThresh); hold on;
for i=1:length(locs)
    line([0,size(Im,2)],[locs(i),locs(i)],'LineWidth',1,'Color','red');
end

plot(x,y,['*', 'green']);


%% 

notes = ['F4'; 'E4'; 'D4'; 'C4'; 'B3'; 'A3'; 'G3'; 'F3'; 'E3'; 'D3'; 'C3'; 'B2'; 'A2'; 'G2'; 'F2'; 'E2'; 'D2'; 'C2'; 'B2'; 'A2'; 'G1' ];
notes = cellstr(notes);
noteSheet='';

% for each system
for n=1:size(stafflineMatrix,1)
    barWidth = diff(stafflineMatrix(n,:),1,2);   % calculate difference in rows
    barWidth = mean(mean(barWidth));
    top = stafflineMatrix(n,1)-4*barWidth;      % Get top y-value of system
    bot = stafflineMatrix(n,5)+4*barWidth;      % Get top y-value of system


    subIm = CThresh(top:bot, :);                % Select the subimage from
    subIm2 = invBWRotatedNoStaff(top:bot, :);          % Select the subimage from

    subImNoTransform = BWRotatedNoStaff(top:bot, :);

    subIm = CThresh(top:bot, :);                % Select the subimage from 
    subIm2 = invBWRotatedNoStaff(top:bot, :);                % Select the subimage from

    %% Find boundingboxes in subIm2
    st = regionprops(subIm2,'BoundingBox');
    filteredSt = st;
    acceptedSt = false(1,length(st))';
    

    %% Remove trash bounding boxes
    % For each bounding box
    for i = 1:length(st)
        thisBB = filteredSt(i).BoundingBox;

        x = round(thisBB(1));
        y = round(thisBB(2));
        w = thisBB(3)-1;
        h = thisBB(4)-1;
 
        
        %   (x,y)-------(x+w,y)
        %     |            |
        %     |            |
        %     |            |
        %  (x,y+h)-----(x+w,y+h)

        % if bounding box in CThresh does NOT contain ones
        if( ~any(any(subIm(y:y+h, x:x+w))))
            acceptedSt(i)=1;
        end
    end

    % Removes elements where acceptedSt = 1
    filteredSt(acceptedSt) = [];
    
    
    
    %%
    
    s = regionprops(subIm, 'centroid');         % Find the centroids in the subimage

    centroids = [s.Centroid];                   % Convert the centroids struct to 2 vectors
    x = centroids(1:2:end-1);
    y = centroids(2:2:end);

    centroids2 = zeros(length(x), 2);           % Merge the x,y vector into a single vector of pairs
    centroids2(:,1) = x;
    centroids2(:,2) = y;


    sortedCentroids = sortrows(centroids2, 1);  % Sort centroids by x
    
    % Add info about notes in each boundingbox
    filteredSt(1).notePos = [];
    for i=1:length(filteredSt)
        for j=1:length(centroids2)
            thisBB = filteredSt(i).BoundingBox;
            thisCT = centroids2(j, :);
            % if statement to see if centroid is inside the boundingbox
            if (thisCT(1) > thisBB(1) && thisCT(1) < thisBB(1) + thisBB(3) ...
                    && thisCT(2) > thisBB(2) && thisCT(2) < thisBB(2) + thisBB(4))
                filteredSt(i).notePos = [filteredSt(i).notePos;thisCT];
            end
        end
    end
  
    halfBWidth = barWidth/2;
    subIm3 = invBWRotated(top:bot, :);                % Select the subimage from;
    imshow(subIm3); hold on;
    plot(x,y,'*');
    
    for u = 1:11
        line([0,size(Im,2)],[u*2*halfBWidth,u*2*halfBWidth],'LineWidth',1,'Color','red');
    end
    
    line([0,size(Im,2)],[size(subIm3,1)/2,size(subIm3,1)/2],'LineWidth',2,'Color','blue');
    
    %% for each boundingbox
    for boundInd = 1:length(filteredSt)
        thisBB = filteredSt(boundInd).BoundingBox;
        x = thisBB(1);
        y = thisBB(2);
        w = thisBB(3);
        h = thisBB(4);
        noNotesInBB = size(filteredSt(boundInd).notePos,1);
        
        
        imshow(subIm3); hold on;
        rectangle('Position', [x,y,w,h],...
                'EdgeColor','r','LineWidth',2  );
        
        firstNoteHeight = y+h-filteredSt(boundInd).notePos(1,2);
        inUpper = false;
        
        y+h-filteredSt(boundInd).notePos(:,2);
        
        if firstNoteHeight >= h/2
            inUpper = true;
        end 
            
       %% for each note in curr boundingbox 
       for noteInd = 1:size(filteredSt(boundInd).notePos, 1)
            
           %%
            thisNote = filteredSt(boundInd).notePos(noteInd,:);
            thisX = thisNote(1);
            thisY = thisNote(2);
            
            % Count number of bars from B2 (middle row)
            dy = size(subIm2, 1)/2 - thisY;
            
            
            noBars = round(dy/(halfBWidth) );            
            
            currNotePitch = notes(12-round(noBars))

            %%
            if noNotesInBB == 1
                
                if inUpper
                    low = thisY+0.5*barWidth;
                    high = thisY+h-0.5*barWidth;
                    leftBound = thisX-1.4*barWidth;
                    rightBound = thisX+0.5*barWidth;                    
                else
                    low = thisY-h;
                    high = thisY-0.5*barWidth;
                    leftBound = thisX-0.4*barWidth;
                    rightBound = thisX+1.5*barWidth;
                end
                leftBound = int16(leftBound);
                rightBound = int16(rightBound);
                low = int16(clamp(low,1, size(subIm2,1)));
                high = int16(clamp(high,1, size(subIm2,1)));
                subNoteIm = subIm2(low:high, leftBound:rightBound);
                currNotePitch = noteClassificate1(currNotePitch, subNoteIm);
           
            else
                
                if inUpper
                    low = thisY+barWidth;
                    high = thisY+h-0.5*barWidth;
                    
                    if noteInd == 1
                        leftBound = thisX-0.5*barWidth;
                        rightBound = thisX+0.5*barWidth;
                                            
                    elseif noteInd == noNotesInBB           % All notes in middle
                        leftBound = thisX-1.5*barWidth;
                        rightBound = thisX-0.5*barWidth;
                    else 
                        leftBound = thisX-1.2*barWidth;
                        rightBound = thisX+0.2*barWidth;    
                    end                    
                else
                    low = thisY-h+0.5*barWidth;
                    high = thisY-barWidth;
                    
                    if noteInd == 1
                        leftBound = thisX;
                        rightBound = thisX+1.2*barWidth;
                                       
                    elseif noteInd == 2
                        leftBound = thisX-0.5*barWidth;
                        rightBound = thisX+1.5*barWidth;
                    else 
                        leftBound = thisX-0.8*barWidth;
                        rightBound = thisX+0.4*barWidth;
                    end                      
                end
                leftBound = int16(leftBound);
                rightBound = int16(rightBound);
                low = int16(clamp(low,1, size(subIm2,1)));
                high = int16(clamp(high,1, size(subIm2,1)));
                subNoteIm = subIm2(low:high, leftBound:rightBound);
                currNotePitch = noteClassificate3(currNotePitch, subNoteIm);
            end
            
            %DEBUG
            %currNotePitch
            %figure;
            %imshow(subNoteIm);
            
            noteSheet=[noteSheet, currNotePitch];      


       end
           
    end 
    if(n < size(staffLineMatrix, 1))
        noteSheet=[noteSheet, 'n'];
    end
end % for-system
    
noteSheet

