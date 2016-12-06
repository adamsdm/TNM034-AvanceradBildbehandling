function [ outNote ] = noteClassificate1(currNotePitch, subNoteIm);
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
    

    
    plot(hist);
    % remove peaks less than 60% of max peak
    peakFilter = pks>5;  
    
    
    filteredPeaks = pks(peakFilter);
    filteredLocs = locs(peakFilter);
    
    plot(hist); hold on;
    plot(filteredLocs, filteredPeaks, '*');
    
    %Eight note
    if(length(filteredPeaks) == 1)
        outNote = lower(currNotePitch)
        return
    % Fourth note
    elseif(isempty(filteredPeaks) )
        outNote = currNotePitch
        return
    end
    
    outNote = 'O';
end

