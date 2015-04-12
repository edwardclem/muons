import numpy as np
from scipy.stats import norm
import cv2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#this is just to test muon data collection + analysis from a USB webcam using OpenCV for Python
#The overall goal of the software is simple:
# 1: constantly take images from a webcam, looking for a bright spot (required signal significance for detection is TBD)
# 2: if a flash is detected, record the image, time, location, etc
# 3: keep looking

#NOTES FROM 4/11/15: 
#We got one! Probably! There were several images captured with small streaks whose energies were
#nearly 85 percent of the maximum. This implies that even 70 times the standard deviation (over 70 R+G+B)
#is much too low of a threshold. 
#So the next step is to find a statistical distribution of high values and determine how
#improbable a high-amplitude event is. 


def search_v1(cam, thresh, datafile, path):
	capture = cv2.VideoCapture(cam)
	print "video stream started"
	with open(datafile, 'a') as outfile:
		while True:
			ret, frame = capture.read()
			#look for pixels that are significantly brighter than the mean
			#get sum of RGB 
			brightness = np.sum(frame, 2)
			x, y = np.where(brightness >= thresh) 
			if x.shape[0] > 0:
				time = datetime.now().time().isoformat()
				output = "event detected at " + time + " max value: " + str(brightness.max())
				print output
				#save image
				filename = path +  'event_' + time + '.jpg'

				cv2.imwrite(filename, frame)

				outfile.write(output + '\n')

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	capture.release()
	cv2.destroyAllWindows()

#looks for a specified amount of time, and builds a distribution of events above the threshold
#now a threshold is a percentage instead of an absolute value! 
def stats_v1(cam, thresh, hi_thresh, datafile, path, time = 12):
	start = datetime.now()
	delta_t = timedelta(hours = time)
	future = start + delta_t

	#saving max value pairs:
	#assuming there will be around 3 events every minute
	#the array will resize if anything else occurs
	length = 3*60*time
	values = np.ndarray((length))
	times = np.ndarray((length))

	events = 0;


	capture = cv2.VideoCapture(cam)
	print "video monitoring initiated"

	with open(datafile, 'a') as outfile:
		while datetime.now() < future:
			ret, frame = capture.read()

			brightness = np.sum(frame, 2)/(255.0*3.0) #normalized
			max_val = brightness.max()
			if max_val >= thresh:
				now = datetime.now()
				time_from_start = now - start
				time_seconds = time_from_start.total_seconds()
				values[events] = max_val
				times[events] = time_seconds
				events += 1
				formatted_intensity = '%.3g' % max_val

				

				#save image to file 
				if (max_val > hi_thresh):
					output = " high-energy (above " + str(hi_thresh) + ") event detected at " + now.isoformat() + " with max intensity: " + formatted_intensity
					filename = path + "event_" + now.isoformat() + "_" +  formatted_intensity + '.jpg'
					cv2.imwrite(filename, frame)
					print output
					mean = values.mean()
					stdev = values.std()
					prob = 1 - norm.cdf(max_val, mean, stdev)
					print "Event probability under current distribution: " + str(prob)
				
				
				#outfile.write(output + '\n')


			#take care of array doubling
			if events >= length:
				length *= 2
				values = np.resize(values, (length))
				times = np.resize(times, (length))	

		capture.release()
		cv2.destroyAllWindows()
		#data processing!
		#plotting things! 

		#using Law of Large Numbers, the distribution should be normal if the limit is low enough
		#calculate probability that event with given intensity will occur
		mean = values.mean()
		stdev = values.std()
		print "computing probabilities using normal distribution"
		#probability that an event greater than or equal to the intensity will occur
		prob1 = 1 - norm.cdf(0.2, mean, stdev)
		prob2 = 1 - norm.cdf(0.5, mean, stdev)
		prob3 = 1 - norm.cdf(0.8, mean, stdev)
		print "Probability that intensity greater than or equal 0.2: " + str(prob1)
		print "Probability that intensity greater than or equal 0.5: " + str(prob2)
		print "Probability that intensity greater than or equal 0.8: " + str(prob3)
		print "saving data to .npy file"
		np.save(values, 'values')
		np.save(times, 'times')

		plt.plot(times, values)
		plt.xlabel("time from start (s)")
		plt.ylabel("intensity (percentage of maximum)")
		plt.show()

if __name__ == '__main__':
	print "muon detection v.0.2"

	datafile = 'data2.txt'
	thresh = 0.05 #low to start
	path = 'captures2/'
	cam = 0

	stats_v1(cam, thresh, 0.5, datafile, path, time = 6)





	