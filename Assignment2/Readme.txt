To run the program, run:
	python assign2.py -part <option number>

Option number:
	1: Draw line segment
		A window containing the original image will pop up. Click on any two points on the window and close the 		window. A new window will pop up showing the line joining the two clicked points and the P2 representation 		will be printed.

	2:  Vanishing line V:
		A window containing the original image will pop up. First click on 4 points which correspond to 2 parallel 			lines (in order) and close the window. Another window will pop up showing the lines. On that window, again 			click on 4 more points corresponding to another pair of parallel lines, and close it. A new window will pop 		up showing the line parallel to the vanishing line passing through the centre of the image. the P2 		representation of the vanishing line will be printed too.

	3: Three sets of transformed parallel lines:
		Repeat the steps of option 2. Close the final window which shows the line parallel to the vanishing line 		passing through the centre of the image. Another similar window will pop up. Click on any point on the 		image and close it. A new window will pop up showing the vertical line passing through the clicked point. 		Close the window and the final window which opens will show the final output.

	4: Affine rectification:
		Repeat the steps of option 2. Then close the image window and the new window which opens will show the 		rectified image.