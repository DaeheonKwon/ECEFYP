function img = myIFT(freq)
    img = ifft2(ifftshift(ifftshift(freq,1),2));
end