function [ outIm ] = maxOfMatrix( inIm )
    outIm = inIm;
    outIm(:,:) = max(max(inIm));
end

