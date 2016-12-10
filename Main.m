%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PREPROCESSING
clear all;
close all;
Im = imread('./Testbilder/im13c.jpg'); % Read the image
Im = im2double(Im);

imshow(Im)
%% Masking
[ySize, xSize] = size(Im);
xSize = xSize/3;
dChannel = 0.2;
ImGrayTest = rgb2gray(Im);

%%

% Mask the area of the notesheet
for i = 1:xSize -1
    for j = 1:ySize -1
        if(abs(Im(j, i, 1) - Im(j, i, 2)) < dChannel && abs(Im(j, i, 1) - Im(j, i, 3)) < dChannel)
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

%imshow(resMaskPAD);
ImPAD = padarray(Im,[20 20]); 
imshow(resMaskPAD);
hold on
plot(topLeft(1),topLeft(2), 'r*');
plot(topRigt(1),topRigt(2), 'r*');
plot(botLeft(1),botLeft(2), 'r*');
plot(botRigt(1),botRigt(2), 'r*');


%%
close all;

xMin = topLeft(1);
xMax = botRigt(1);
yMin = topLeft(2);
yMax = botRigt(2);

movingPoints=[topLeft; topRigt; botRigt; botLeft] %(x,y) coordinate
fixedPoints= [xMin yMin; xMax yMin; xMax yMax; xMin yMax];


Tform = fitgeotrans(movingPoints,fixedPoints,'projective');
R=imref2d(size(ImPAD),[1 size(ImPAD,2)],[1 size(ImPAD,1)]);
imgTransformed=imwarp(ImPAD,R,Tform,'OutputView',R);
imshow(imgTransformed,[]); hold on;

plot(xMin,yMin, '*');
plot(xMax,yMin, '*');
plot(xMax,yMax, '*');
plot(xMin,yMax, '*');


Im = imgTransformed(yMin:yMax,xMin:xMax,:);
figure;
imshow(Im);

%% Normalize image background (dynamically) by dividing with local maximum
ImGrayDouble = rgb2gray(Im);
func = @(block_struct) (block_struct.data./max(max(block_struct.data)));
Im2 = blockproc(ImGrayDouble, [9 9], func);

%% Dynamic thresholding
imshow(Im2);
figure; 


BW = adaptivethreshold(Im2,11,0.05);
imshow(BW);
%% Find staff lines
% HISTOGRAM
% Find best angle

close all;
invBW = abs(BW-1);   % 

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


%% Remove peaks that aren't groups of five 

index=1;
while( index < length(locs-5)) 
    %%
    remainingPeaks = length(locs-5)-index;
    if remainingPeaks < 5
        locs(index:length(locs)) = -1;
        return;
    % Check if 5 next elements are withing range
    elseif( locs(index+1) - locs(index+0) < medianBarWidth && locs(index+2) - locs(index+1) < medianBarWidth && locs(index+3) - locs(index+2) < medianBarWidth &&  locs(index+4) - locs(index+3) < medianBarWidth ) 
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
    
    MAX = max([size(CThresh,1) size(stafflineMatrix,1)])
    
    top = stafflineMatrix(n,1)-4*barWidth;      % Get top y-value of system
    bot = stafflineMatrix(n,5)+4*barWidth;      % Get top y-value of system
    
    top=clamp(top,0, size(CThresh,1));
    bot=clamp(bot,0, size(CThresh,1));
    
    
    %%
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
    if(n < size(stafflineMatrix, 1))
        noteSheet=[noteSheet, 'n'];
    end
end % for-system
  
%%
im1sfacit  = 'g3e3f3e3g2e3f3a2b2C3c3g3e3f3e3g2e3nf3a2b2C3c3e2g2g2f2a2A2d3d3g2e3c3nf3d3g3a2b2C3c3C3c3';
im3sfacit  = 'G3g3a3G3E3e3f3E3D3d3e3d3b2c3d3e3f3G3nG3g3a3G3D3g3a3B3C4E3F3G3nc3c3c3d3e3d3c3d3E3C3d3d3d3e3f3e3d3e3F3D3ne3f3G3f3e3f3g3A3g3f3g3a3B3a3g3C4C4';
im5sfacit  = 'd3b2d3g3d3b2d3b2g2f2a2d2f2a2c3e3c3a2f2d2f2g2d2g2b2d3b2g3d3b2d3b2g2nf2a2d2f2a2c3e3c3a2f2d2f2G2a2d3f3c3e3a2c3e3c3d3f3a2d3f3d3ne3g3e3c3a2c3d3f3a3g3b3a3c3e3a2c3e3c3d3f3a2d3f3d3e3g3b3g3e3c3D3';
im6sfacit  = 'G2g2G2G2F2G2F2G2nG2g2G2G2F2G2nG2g2G2G2F2G2F2A2nG2g2G2G2F2G2nD2C3b2a2B2C3nA2G2F2F2nD2C3b2a2B2C3nA2G2F2G2F2A2';
im8sfacit  = 'C2F2A2F2b2a2g2f2E2G2E2nC2F2A2F2b2a2g2f2E2G2C2E2nA2a2a2A2A2B2a2a2G2g2g2G2f2f2E2F2nA2a2a2A2A2B2A2G2G2F2E2';
im9sfacit  = 'b2d3G3C4D3e3c3b2d3b3c4C4A3F3D3na3b3C4C4C4a3g3B3B3C4E3F3G3E3b2d3nG3C4D3e3c3b2d3a3c4C4A3F3D3';
im10sfacit = 'D3B3d3e3G3f3a3C4f3g3B3a3G3e3f3D3e3c3nC3f3D3f3g3C4g3f3E3g3c4C4a3g3nF3e3d3E3d3c3D3c3b2C3a2c3F3d3C3e3f3F3E3nC3A3c3e3G3f3a3C4f3g3A3b3G3e3f3D3e3c3nC3A3c3e3G3f3a3C4f3g3A3a3G3e3f3D3e3c3nf3g3f3E3g3b3C4a3g3F3e3d3E3d3c3nD3c3b2C3a2c3F3d3C3F3E3';


noteSheet = strjoin(noteSheet);
noteSheet(ismember(noteSheet,' ,.:;!')) = [];

lsteindist = strdist(noteSheet, im8sfacit)
