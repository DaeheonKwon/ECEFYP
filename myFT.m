function freq = myFT(img)
    
    freq = fftshift(fftshift(fft2(img),1),2);
    
end