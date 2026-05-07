#ifndef DSP_ALGORITHM_H
#define DSP_ALGORITHM_H

//Filter class

class EMAFilter {
	private:
		float alpha; // Filter Coefficient (0.0 -> 1.0)
		float current_val;
		bool is_initialized;
	public:
		// Init
		EMAFilter(float alpha = 0.274f);

		void reset();

		float filter(float new_val);
};

// Circular Buffer
const int BUFFER_SIZE = 250;

struct IMUData{
	unsigned long timestamp;
	float ax, ay, az;
	float gx, gy, gz;
};

class CircularBuffer{
	private:
		IMUData buffer[BUFFER_SIZE];
		int head;
		int count;
	
	public:
		CircularBuffer();

		void push(IMUData data);

		int getCount();

		IMUData get(int offset);
		
};

// Peak Detector
enum DetectorState{
	LOOKING_FOR_VALLEY = 0,
	LOOKING_FOR_PEAK = 1
};

class PeakDetector{
	private:
		DetectorState state;
		float threshold; 
		float temp_max, temp_min;
		unsigned long temp_max_time, temp_min_time;

		unsigned long last_valley_time;
		bool has_first_valley;
	
	public:
		PeakDetector(float threshold = 0.6f);
		void reset();

		//Process sample -> return true if finding completely action
		//Also store out_start_time and out_end_time through pointer
		bool processSample(float value, unsigned long timestamp, unsigned long &out_start_time, unsigned long &out_end_time);
};

#endif