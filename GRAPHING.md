GRAPHING WITH PYTHON
====================

## Recording the CSV file

After collecting the data across a number of iterations, inside a `std::vector` of structs called `eventRecords`, I looped through each of them to extract the iteration number, the time between the two records, and the total simulation time up until that point:

```C++
      void writeTime(const char* fileName) {
          FILE* of = fopen(fileName, "w");

          for (auto& record : eventRecords) {
              double millisPerFrame = record.time / TIMEKEEPING_FRAMESIZE;
              millisPerFrame /= 1000.0;//seconds per frame
              double fps = 1.0 / millisPerFrame;
              double seconds = record.totalTime / 1000.0;
              fprintf(of, "%d,%0.3f,%f\n", record.frameNo, seconds, fps);
          }//for

          fclose(of);
      }//writeTime
```

After this, I wrote the records into the csv. This involves rows being separated by newlines, and columns being separated by commas.

For my particular code, the filenames were hard-coded filenames, like the following:

    const char timingFileName[] = "../outputData/CoherentGrid_HighDensity_128.csv";

## Reading the CSV file
