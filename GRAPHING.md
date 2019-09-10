GRAPHING WITH PYTHON
====================

## Recording the CSV file

After collecting the data across a number of iterations, inside a `std::vector` of structs called `eventRecords`, I looped through each of them to extract the iteration number, the time between the two records, and the total simulation time up until that point:

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
