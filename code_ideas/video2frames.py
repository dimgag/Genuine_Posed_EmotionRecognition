
# Function to extract frames from videos
def video2frames(dir, dirname, file):
	vidcap = cv2.VideoCapture(dir)
	'''
	Get length of the videos in frames:
		- Get frames from 50% of the video's length until 90% of the length
	'''
	length_of_video = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	success, image = vidcap.read()

	new_dir = dir.replace('patches','frames3D').split('.MP4')[0]
	# print( new_dir )

	print( os.path.basename(new_dir) )

	count = 0
	frame_counter = 0
	vid = []
	while success:
		success,image = vidcap.read()

		if frame_counter > 16:
			frame_counter = 0
			print(file)
			# np.save(, vid)
			vid = []
		
		if count > int(length_of_video*.6) and count < int(length_of_video*.9):
			vid.append(dirname)
			cv2.imwrite(os.path.join(frames_dir, file + 'frame%d.jpg' % count), image)
            # cv2.imwrite(os.path.join(dirname, file + 'frame%d.jpg' % count), image)     # save frame as JPEG file
            

		if cv2.waitKey(10) == 27:                     # exit if Escape is hit
			break

		count += 1
		frame_counter += 1
