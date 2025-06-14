# TQRNG_WebCam

I tried to make it more scientifically sound TQRNG with webcam. Deriving the quantum effect of photon arrival from expirementa measurements and theoretical estimation of photons. The success is partial - and there is a lot of junk code made for this purpose.

Eventually I recommend just to use the site and libraries as is. 

----------------
I have made several simulations and recordings, and tried to find how many photons do I have, my noise, create theoretical ground 
if i have some bias of dominant value + noise I call it white/black ratio so for 1 + a ratio for pixel A and 1 + b pixel B I get 
1 + c, c = -(ab) / (2 + a + b + ab). 

This shows reduced bias by xoring operation. My empirical data show sometimes low noise and high bias i.e. sigma = 0.33 
This brings to some reasonable amount of photons like ~100K per pixel with normal lighting. 
