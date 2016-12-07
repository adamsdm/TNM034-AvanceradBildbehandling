function [ out ] = clamp( in, min, max )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    out = in;        
    if out>max
        out = max;
    elseif out<min
        out = min;
    end
    
end

