import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv


def readCSV(filename):
    results = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for line in reader:
            timestamp = float(line[1])
            fps = float(line[2])
            results.append([timestamp, fps])

    return np.array(results).T

def makeGraphs(resultSets, title):
    fig, ax = plt.subplots(1)
    for filename, data in resultSets:
        ax.plot(data[0], data[1], label = cleanFileName(filename))

    ax.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Ticks/Frames per Second")
    
    fig.suptitle(title)
    fig.set_size_inches(10,6)
    
    #plt.show()
    filePath = makeSavePath(title)
    plt.savefig(filePath)

def makeGraph(array):
    plt.plot(array[0], array[1])
    plt.show()

def cleanFileName(fileName):
    regstr = "([A-Z][a-z]+)([A-Z][a-z]+)_([A-Z][a-z]+)([A-Z][a-z]+)_([0-9]+).csv"
    tokens = re.match(regstr, fileName).groups()

    retval = "%s %s, %s %s, BlockSize %s" % tokens
    return retval

def makeSavePath(title):
    filePath = os.path.join("..", "images", title + ".png")
    return filePath



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



if __name__ == "__main__":
    main()


