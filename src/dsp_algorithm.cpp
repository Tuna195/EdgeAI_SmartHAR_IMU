#include "dsp_algorithm.h"

//constructor EMAFilter class
EMAFilter::EMAFilter(float alpha){
	this -> alpha = alpha;
	this -> current_val = 0.0f;
	this -> is_initialized = false;
}

void EMAFilter::reset(){
	this -> is_initialized = false;
}

//Algorithm of EMA
float EMAFilter::filter(float new_val){
	if(!is_initialized){
		current_val = new_val;
		is_initialized = true;
	}
	else{
		current_val = alpha * new_val + (1.0f - alpha)*current_val;
	}
	return current_val;
}

//constructor CircularBuffer class
CircularBuffer::CircularBuffer(){
	this -> head = 0;
	this -> count = 0;
}

int CircularBuffer::getCount(){
	return count;
}

void CircularBuffer::push(IMUData data){
	buffer[head] = data;
	head++;
	head %= BUFFER_SIZE;
	if(count < 250) count++;
}

IMUData CircularBuffer::get(int offset){
	if(offset >= count) offset = count - 1;
	if(offset < 0) offset = 0;
	int idx = head - 1 - offset;
	if(idx < 0){
		idx += BUFFER_SIZE;
	}
	return buffer[idx];
}

// Peak Detector
#define MAX_INIT -9999.0f
#define MIN_INIT 9999.0f

PeakDetector::PeakDetector(float threshold){
	this->threshold = threshold;
	reset();
}

void PeakDetector::reset(){
	state = LOOKING_FOR_VALLEY;
	temp_max = MAX_INIT;
	temp_min = MIN_INIT;
	has_first_valley = false; 
}

bool PeakDetector::processSample(float value, unsigned long timestamp, unsigned long &out_start_time, unsigned long &out_end_time){
	bool rep_found = false; //init

	if(temp_max == MAX_INIT) temp_max = value;
	if(temp_min == MIN_INIT) temp_min = value;

	if(state == LOOKING_FOR_VALLEY){
		if(value < temp_min){
			//keep finding deepest valley
			temp_min = value;
			temp_min_time = timestamp;
		}
		else if(value >= temp_min + threshold){
			// If it's already had a valley, so connect last valley with new valley to a count
			if(has_first_valley){
				out_start_time = last_valley_time;
				out_end_time = temp_min_time;
				rep_found = true;
			}

			last_valley_time = temp_min_time;
			has_first_valley = true;

			//switch to finding peak state
			state = LOOKING_FOR_PEAK;
			temp_max = value;
			temp_max_time = timestamp;
		}
	}
	else if(state == LOOKING_FOR_PEAK){
		if(value > temp_max){
			//keep finding highest peak
			temp_max = value;
			temp_max_time = timestamp;
		}
		else if(value <= temp_max - threshold){
			//Switch to finding valley state
			state = LOOKING_FOR_VALLEY;
			temp_min = value;
			temp_min_time = timestamp;
		}
	}
	return rep_found;
}