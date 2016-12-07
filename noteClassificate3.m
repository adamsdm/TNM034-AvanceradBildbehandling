function [ outNote ] = noteClassificate3(currNotePitch, subNoteIm);
    outNote = currNotePitch;
    
    % If for some reason the input image is empty
    if isempty(subNoteIm)  
        outNote = strcat(outNote,'!');
        return;
    end
    
    %% COUNT BARS
    %create histogram 
    hist = sum(subNoteIm, 2);
    [pks, locs] = findpeaks(hist);
    
    % remove peaks less than 60% of max peak
    maxPeak = max(pks);
    peakFilter = pks>4;  % OR 0.6*maxPeak;
    
    
    filteredPeaks = pks(peakFilter);
    filteredLocs = locs(peakFilter);

    
    plot(hist); hold on;
    plot(filteredLocs, filteredPeaks, '*');
    
    %Eight note
    if(length(filteredPeaks) == 1)
        outNote = lower(currNotePitch);
        return
    % Fourth note
    elseif(length(filteredPeaks) == 1)
        outNote = currNotePitch;
        return
    end
    
    outNote = '';
end

