GRAPHING WITH PYTHON
====================

## Files

The python script may be found (here)[outputData/csvReader.py]

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

```C++
    const char timingFileName[] = "../outputData/CoherentGrid_HighDensity_128.csv";
```

## Using the Data

### Invoking the Python script

The script I wrote takes in at least two arguments; the first is the title for the graph (and, by extension, the name of the image to save), and all arguments after that are names of csv files, in the format recorded above.

The main function is as follows:

```Python

    def main():
        if len(sys.argv) < 3:
            print("Please input a title and file names")
            exit(0)

        resultSets = []

        for i, fileName in enumerate(sys.argv):
            if i == 0 or i == 1:
                continue
            resultSets.append((fileName, readCSV(fileName)))

        makeGraphs(resultSets, sys.argv[1])

```

It takes each CSV file, transforms it into a Numpy array, and binds that into a tuple along with the name of the file the data came from. For each of those sets
