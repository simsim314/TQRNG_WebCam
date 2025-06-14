# TQRNG WebCam
[Site Link](https://simsim314.github.io/TQRNG_WebCam/)

This repository provides [python](quantum_cam_rng.py) and [js](quantum_cam_rng.js) libraries for True Quantum Random Generator based on webcam streaming, xorins least significant bit of batches of rgb pixels (i.e. for several pixels r xor g xor b).  

A [site](https://simsim314.github.io/TQRNG_WebCam/) and [python test code](test_random.py) are included. 

For image of size 640x480 on 30 fps it should give ~180K random bits, or 6K random float/sec. 

----------------

I tried to make it more scientifically sound TQRNG with webcam. Deriving the quantum effect of photon arrival from expirementa measurements and theoretical estimation of photons. The success is partial - and there is a lot of junk code made for this purpose.

Eventually I recommend just to use the site and libraries as is. 

----------------
I have made several simulations and recordings, and tried to find how many photons do I have, using statistics of stability value per pixel, and fitting gasusian. I also tried to create theoretical ground for bias toward one of the values on the gaussian. 

if i have some bias of dominant value + noise I call it white/black ratio so for 1 + a ratio for pixel A and 1 + b pixel B I get 
1 + c, c = -(ab) / (2 + a + b + ab) for A xor B

This shows reduced bias by xoring operation. My empirical data show sometimes low noise and high bias i.e. sigma = 0.33 
This brings to some reasonable amount of photons like ~100K per pixel with normal lighting. Needing to use about 33 rgb pixel xoring (i.e. 33 pixel each r xor g xor b totalling in 99 xoring) to remove bias and provide trully unbiased quantum random generator. My default is set to 50. For regular lighting setting it should be enough to generate quantum dominated unbiased random with theoretical bias < 1-e12 toward one of the bits. 

