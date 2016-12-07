function [ outNote ] = noteClassificate(currNotePitch, subNoteIm);
    outNote = currNotePitch;
    
    
    %% COUNT BARS
    %create histogram 
    hist = sum(subNoteIm, 2);
    [pks, locs] = findpeaks(hist);
    
    % remove peaks less than 60% of max peak
    maxPeak = max(pks);
    peakFilter = pks>9;  % OR 0.6*maxPeak;
    
    
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

