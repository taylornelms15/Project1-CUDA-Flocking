GRAPHING WITH PYTHON
====================

## Files

The python script may be found [here](outputData/csvReader.py). The other relevant code is in the `main` file within `src`. For any gaps, or if you're curious as to the support structure behind some of this functionality, feel free to look there.

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

For example, a way to use this script would be to call `python csvReader.py "Coherent Grid, High Density, All Block Sizes" CoherentGrid_*.csv`

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

It takes each CSV file, transforms it into a Numpy array, and binds that into a tuple along with the name of the file the data came from. It then hands each of those sets off to a function that graphs them, and saves out the data (or, optionally, displays it onscreen).

### Reading in the CSV data

I made use of the `csv` package, but the file format is so simple, you could quickly write up your own. The function was the following:

```Python
    def readCSV(filename):
        """
        takes in a filename
        returns a numpy array, with the first row as the
            timestamp in seconds, and the second row
            as the fps across the last time block
        """
        results = []
        with open(filename) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line in reader:
                timestamp = float(line[1])
                fps = float(line[2])
                results.append([timestamp, fps])

        return np.array(results).T
```

If you're not familiar with the `numpy` library, it's a handy library for making array operations more efficient, while also making some simple things entirely too complicated. In this case, the `.T` within the return statement transposes the 2D array, making our two columns into two rows, which will be essential for displaying them nicely.

### Graphing the data

I used `matplotlib` to graph the data. It has a billion fun features, along with occasionally frustrating documentation, meaning all of my matplotlib code is a hacked-together mess of code snippets stolen from forum posts. That said, the goals here were relatively simple; I wanted to take a bunch of 2d arrays representing pairs of `(timestamp, fps)` data points and graph those series' onto a line plot.

The code was as follows:

```Python
    def makeGraphs(resultSets, title):
        """
        Displays the resultant data sets, along with a given title
        """
        fig, ax = plt.subplots(1)
        for filename, data in resultSets:
            ax.plot(data[0], data[1], label = cleanFileName(filename))

        ax.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Ticks/Frames per Second")

        fig.suptitle(title)
        fig.set_size_inches(10,6)

        #plt.show() #uncomment this to display the graph on your screen
        filePath = makeSavePath(title)
        plt.savefig(filePath)
```

The core functionality is the `plot` function, where you can provide a series of `x` data, along with a series of `y` data, along with a bunch of other optional variables, such as a label for the series. I did this for each set of data, stuck some labels onto the plot, and then saved it all to an image file.

Happy coding!
