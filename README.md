**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Taylor Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor)
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

## Results
![](images/mLo_dMed_gMed.gif)
*No Grid used for implementation*
![](images/mMed_dMed_gMed.gif)
*Uniform grid used within implementation*
![](images/mHi_dMed_gMed.gif)
*Coherent grid used within implementation*

## Analysis

Unsurprisingly, the grid implementations ended up significantly more efficient than the naive implementation. For runs with 5000 boids, with a block size of 128, the FPS over a 45-ish-second run yielded the following results:

![](images/All-Grids,-Medium-Density,-Block-Size-128.png)

### Interesting Notes

During the naive implementation, I changed my `distBetween` function, which computed the distance between two vectors, 
between using the `glm::distance` function and a simple `sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff)` function.
Though I would have expected the `glm::distance` function to be highly optimized in some fashion,
I saw framerate drop from around 10fps to around 2.5fps in the simulation window.

#### NOTESPACE

##### Testing IV's

* Performance Mode (only switch for the mediums)
* Particle Density (low, medium, high)
* Grid Density (rule distances) (low, medium, high)
* Framerate at beginning, and after some convergence
