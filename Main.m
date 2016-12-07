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
template = imread('NoteheadTemplate.png');
template = im2bw(template,0.9);

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
CThresh = (C>0.7*max(max(C)));

%Remove G-glef 
CThresh(:,1:0.05*size(CThresh,2)) = 0;
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
    
    %% for each boundingbox
    for boundInd = 1:length(filteredSt)
        thisBB = filteredSt(boundInd).BoundingBox;
        x = thisBB(1);
        y = thisBB(2);
        w = thisBB(3);
        h = thisBB(4);
        noNotesInBB = size(filteredSt(boundInd).notePos,1);
        
        
        imshow(subIm2); hold on;
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
            noBars = thisY/halfBWidth;              
            currNotePitch = notes(round(noBars));
            
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
                
                low = clamp(low,0, size(subIm2,1));
                high = clamp(high,0, size(subIm2,1));
                subNoteIm = subIm2(low:high, leftBound:rightBound);
                currNotePitch = noteClassificate1(currNotePitch, subNoteIm);
            
            elseif noNotesInBB == 2
                if inUpper
                    low = thisY+barWidth;
                    high = thisY+h-0.5*barWidth;
                    
                    if noteInd == 1
                        leftBound = thisX-0.5*barWidth;
                        rightBound = thisX+0.5*barWidth;
                                            %
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
                                       
                    else noteInd == 2
                        leftBound = thisX-0.5*barWidth;
                        rightBound = thisX+1.5*barWidth;
                    end                      
                end
                low = clamp(low,0, size(subIm2,1));
                high = clamp(high,0, size(subIm2,1));
                subNoteIm = subIm2(low:high, leftBound:rightBound);
                currNotePitch = noteClassificate3(currNotePitch, subNoteIm);
            
            elseif noNotesInBB ==3
                
                if inUpper
                    low = thisY+barWidth;
                    high = thisY+h-0.5*barWidth;
                    
                    if noteInd == 1
                        leftBound = thisX-0.5*barWidth;
                        rightBound = thisX+0.5*barWidth;
                                            %
                    elseif noteInd == 2
                        leftBound = thisX-1.2*barWidth;
                        rightBound = thisX+0.2*barWidth;
                    else 
                        leftBound = thisX-1.5*barWidth;
                        rightBound = thisX-0.5*barWidth;
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
                low = clamp(low,0, size(subIm2,1));
                high = clamp(high,0, size(subIm2,1));
                subNoteIm = subIm2(low:high, leftBound:rightBound);
                currNotePitch = noteClassificate3(currNotePitch, subNoteIm);
            
            elseif noNotesInBB == 4                             % If 4 notes in boundingbox -> sixteenth note
                currNotePitch = '';
            else
               currNotePitch = '';                              % Must be 32th notes
            end
            
            %currNotePitch
            %figure;
            %imshow(subNoteIm);
            noteSheet=[noteSheet, currNotePitch];      


       end
           
    end 
    noteSheet=[noteSheet, 'n'];
end % for-system
    
noteSheet

