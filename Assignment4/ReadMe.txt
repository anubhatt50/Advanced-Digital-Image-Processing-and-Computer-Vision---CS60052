To run the program, run:
	python assign4.py --source_img [source image name] --target_img [Target image name]

The command line arguments are optional. Default source and target images are IMG_6481.jpg and IMG_6479.jpg respectively

Sample command:
	python assign4.py
	python assign4.py --source_img IMG_6481.jpg --target_img IMG_6479.jpg

When an image pops up in a window, click on two points and close the window. Those two points determine opposite corners of an interactively selected rectangular window.

Output images will be saved in the 'outputs' folder . Sample results have been provided in the same folder (An extra set of results have also been provided in the 'outputs_2' folder). Relevant comments have been provided in code.

Implementation:

1. The source and target images are read.

2. A rectangular window is selected interactively from the first (source) image and saved.

3. The pixel values are converted to 2D CIE and a 2D chromaticity plot is made.

4. The points are clustered for finding the mode using K means clustering. The dominant cluster corresponds to the cluster containing maximum number of points and the mean of those points determine the dominant colour. In this experiment, number of clusters (K) is kept as 3.

5. Another 2D chromaticity plot is made showing the dominant points (points belonging to the dominant cluster).

6. The corresponding dominant pixels in the cropped image are made white and saved.

7. The domimant colour is saved as another image.

8. Steps 2 to 7 are repeated for the target image.

9. The dominant pixels of the source and target images are converted to l-alpha-beta space and the dominant colour of the source region is transferred to the target region using colour transfer.

10. The final result (both cropped and inside the original source image) is saved.