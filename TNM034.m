function [ noteSheet ] = TNM034( Im )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TNM034(Im) testing %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PREPROCESSING
Im = im2double( Im ); % Convert image to double

level = graythresh(Im); % Computes the global threshold level from the image
BW = im2bw(Im,level);   % Convert the image to binary image with threshold: level



%% Find staff lines
% HISTOGRAM
% Find best angle

invBW = BW<level;   

A = findBestRotAngle(Im);
%% Show peaks in plot
invBWRotated = imrotate(invBW, A,'bilinear');   % Rotate the image with the angle obtained

maxVals = sum(invBWRotated(:,:)');              % calculate the histogram by summing pixels horizontally
%%
[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs

maxValsThresh = 0.3*max(max(peaks));
maxVals(maxVals < maxValsThresh) = 0;           % Removes all noise below a certain threshold

[peaks, locs] = findpeaks(maxVals);             % Find peak locations, stemlines y value = locs


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

%% Remove trash peaks from locs and peaks

index=1;
while( index < length(locs-5)) 
    % Check if 5 next elements are withing range
    if( locs(index+1) - locs(index+0) < medianBarWidth && locs(index+2) - locs(index+1) < medianBarWidth && locs(index+3) - locs(index+2) < medianBarWidth &&  locs(index+4) - locs(index+3) < medianBarWidth ) 
        locs(index:index+4);
        index=index+5;    % check next 5 peaks
    else % peak is trash peak
        locs(index) = -1; % set peaks not accepted to negative value 
        index=index+1; % check next peak
    end
end

exclude = locs < 0; %Logicall array containing positions where locs is negative
locs(exclude) = [];
peaks(exclude) = [];


%% Show found peak locations in image;
rotIm = imrotate(Im,A,'bilinear');


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

C = normxcorr2(noteTempRez, BWRotatedNoStaff);

%%
% Remove border from C which normxcorr2 adds
yoffset = int16((size(C,1)-size(BWRotatedNoStaff,1))/2);
xoffset = int16((size(C,2)-size(BWRotatedNoStaff,2))/2);



C = C(yoffset:size(C,1)-yoffset,:);
C = C(:, xoffset:size(C,2)-xoffset);


%%

CThresh = (C>0.78*max(max(C)));

CThresh(:,1:gClefXMean) = 0;

invBWRotatedNoStaff = BWRotatedNoStaff<1;

%%
 s= regionprops(CThresh, 'centroid');

centroids = [s.Centroid];
x = centroids(1:2:end-1);
y = centroids(2:2:end);




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
    
    %% for each boundingbox
    for boundInd = 1:length(filteredSt)
        thisBB = filteredSt(boundInd).BoundingBox;
        x = thisBB(1);
        y = thisBB(2);
        w = thisBB(3);
        h = thisBB(4);
        noNotesInBB = size(filteredSt(boundInd).notePos,1);

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
            
            currNotePitch = notes(12-round(noBars));

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
                                            %
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
            
            noteSheet=[noteSheet, currNotePitch];      


       end
           
    end 
    if n < size(stafflineMatrix, 1)
        noteSheet=[noteSheet, 'n'];
    end
end % for-system

noteSheet = strjoin(noteSheet);
noteSheet(ismember(noteSheet,' ,.:;!')) = [];

end



