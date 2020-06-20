To run the program, run:
	python assign5.py --img_number [image number]

The command line arguments are optional. Default image number is 1.

Sample command:
	python assign5.py
	python assign5.py --img_number 2

Output images will be saved in the 'outputs/output<image number>' folder (e.g., outputs/output1 for image 1). Sample outputs have been provided in the same folder. Relevant comments have been provided in code.

Implementation:

1. The depth image is read.

2. The 1st and 2nd order derivatives of the image are computed (using Sobel filter) and from those the principal curvatures and mean and Gaussian curvatures are computed.

3. Local topology images are computed based on the signs of principal curvatures as well as mean and Gaussian curvatures and saved. The colour charts (colour labels) of the corresponding images are also saved.

4. NPS of each pixel is computed (with k=5).

5. The images are segmented and the results saved.

Observation:

We can see that for all the images, segmentation from NPS captures more information than the other two images. Hence for all the 4 images:

(c)>(a)=(b)