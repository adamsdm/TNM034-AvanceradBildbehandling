function [ outNote ] = noteClassificate(currNotePitch, subNoteIm);
    outNote = currNotePitch;
    
    % if eigth-note convert to lowercase
    outNote = lower(currNotePitch);
end

