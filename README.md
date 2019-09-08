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

### Implementation Strategy

Unsurprisingly, the grid implementations ended up significantly more efficient than the naive implementation. For runs with 5000 boids, with a block size of 128, the FPS over a 45-ish-second run yielded the following results:

![](images/All&#32;Grids,&#32;Medium&#32;Density,&#32;Block&#32;Size&#32;128.png)

There are a few things to unpack here. First, the spike in the initial framerate of the naive implementation. Frankly, I have no idea why this exists; I was taking data points every 300 ticks, so I can't imagine this being some fluke of initial set-up taking less time. In all honesty, I would need to do significantly more debugging to figure it out.

Of course, the more interesting behavior lies within the meat of the simulation. The grid-based solutions performed better overall, with a slight improvement for the coherent grid over the uniform grid. Of course, algorithmically, this makes sense; the grid-based approaches cut execution time on the GPU from order `O(N)` to `O(n)`, where `n` is the number of boids that are within the grid-neighborhood of each boid. (On a CPU, the naive approach would run in time `O(N^2)`, while the grid approaches would run in time `O(Nn)`.)

For another example, let's look at each of the models with a higher density of boids; in this case, we're operating with 10,000 boids, rather than 5,000, in that same space:

![](images/All&#32;Grids,&#32;High&#32;Density,&#32;Block&#32;Size&#32;128.png)

Another notable part here is that the framerate drops off over time for the uniform grid, while the coherent grid stays relatively steady. The best I can figure for the drop is that, over time, each boid has more neighbors, and so the number of data accesses to those neighbors increases (as more of the boids are in flocks). This increases the penalties felt from the boid data being more scattered in the uniform grid implementation, as cached data accesses become less favorable.

### Number of Boids

Unsurprisingly, as the number of boids increase, the execution speed of the simulation decreases. Here are some comparisons for all the models, running with 2000 boids, 5000 boids, and 10000 boids:

![](images/No&#32;Grid,&#32;All&#32;Densities,&#32;Block&#32;Size&#32;128.png) ![](images/Uniform&#32;Grid,&#32;All&#32;Densities,&#32;Block&#32;Size&#32;128.png) ![](images/Coherent&#32;Grid,&#32;All&#32;Densities,&#32;Block&#32;Size&#32;128.png)

### Miscellaneous Notes

During the naive implementation, I changed my `distBetween` function, which computed the distance between two vectors, 
between using the `glm::distance` function and a simple `sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff)` function.
Though I would have expected the `glm::distance` function to be highly optimized in some fashion,
I saw framerate drop from around 10fps to around 2.5fps in the simulation window.
