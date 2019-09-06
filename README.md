**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Taylor Nelms
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### Interesting Notes

During the naive implementation, I changed my `distBetween` function, which computed the distance between two vectors, 
between using the `glm::distance` function and a simple `sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff)` function.
Though I would have expected the `glm::distance` function to be highly optimized in some fashion,
I saw framerate drop from around 10fps to around 2.5fps in the simulation window.

#### NOTESPACE

halfSideCount 11, gridSideCount 22, gridCellCount 10648, gridInverseCellWidth 0.1, halfGridWidth 110.0, gridmin (x, y, z) (-110.0, -110.0, -110.0)
