%% Create an audio file from a string given by an OMR

% This string will be translated to an audio file. 
% A big letter is a 4th note and a small letter is a 8th note
exampleString = {'G1', 'a1', 'B1' ,'C2', 'D2', 'D2', 'G2', 'G2'};

% Frequencies of the notes can be found at:
% http://www.phy.mtu.edu/~suits/notefreqs.html
% "notes" and "frequencies" correspond to eachother
% --> ex: G1 has the frequency of 196.00Hz
notes = {'G1', 'A1', 'B1' ,'C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2',...
         'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4', 'E4'};
     
frequencies = [196.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23,...
               392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46,... 
               783.99, 880.00, 987.77, 1046.50, 1174.66, 1318.51];


% Fs: sample rate, bpm (beats per minute): the "tempo" of the song
Fs = 44100;
bpm = 120;

% The final audio vector, y
% Volume should be between 0 and 1. (0 is silence, 1 is full volume) 
y = [];
volume = 1;

% Construct the audio file from the string (ex: exampleString)
% source: https://se.mathworks.com/matlabcentral/newsreader/view_thread/136160
for i = 1:length(exampleString)

    note  = exampleString{i};
    if(isstrprop(note(1), 'alpha'))
        if(isstrprop(note(1), 'upper'))
           t = 0:(1/Fs):(60/bpm); %Seconds should be replaced with the correct length
        else
           t = 0:(1/Fs):(60/(2*bpm)); 
        end
        
        % Search the notes vector and compare to current note. set freq if found. 
        noteFound = true;
        counter = 0;
        for j = 1:length(notes)
            if(strcmp(upper(note),notes{j}))
                freq = frequencies(j);
            else
                counter = counter + 1;
            end
            
            if(counter == length(notes))
                noteFound = false;
            end
        end
        
        if(noteFound == false)
            disp('One of the notes could not be found.');
            disp('Please check your input string!')
        end
       
       % Create note to add to the final audio vector 
       yNote = volume*sin(2*pi*freq*t);
        
       % Add the note to the final audio vector
       y = [y, yNote];
    else
        disp('----- Wrong format of one or more notes in the input string! -----');
    end
end


% Normalize y if any sample has an amplitude > 1.
if(abs(max(y)) > 1)
    y = y/abs(max(y));
end

% Play the sound
sound(y, Fs);

