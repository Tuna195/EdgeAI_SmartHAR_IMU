#include "dsp_algorithm.h"

//1. EMA filter
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

// 2. CircularBuffer
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

// 3. Peak Detector
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
			if(has_first_valley){
				out_start_time = last_valley_time;
				out_end_time = temp_min_time;
				rep_found = true;
			} else {
				// Nhip dau tien: khong co start_time nhung van bao rep
				out_start_time = temp_min_time;
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

// 4. Resampling Algorithm
void Resampler::resample(CircularBuffer *cb, unsigned long start_time, unsigned long end_time, float (*out_matrix)[6]){
	int count = cb->getCount();
	int start_idx = -1;
	int end_idx = -1;

	for(int i = 0; i < count; i++){
		IMUData d = cb->get(i);
		if(d.timestamp == start_time) start_idx = i;
		if(d.timestamp == end_time) end_idx = i;
	}

	if(start_idx == -1 || end_idx == -1) return;

	int raw_length = start_idx - end_idx + 1;
	if(raw_length < 2) return;

	for(int step = 0; step < 100; step++){
		float t = step / 99.0;
		float virtual_idx = t*(raw_length - 1);
		int idx1 = virtual_idx;
		int idx2 = idx1 + 1;
		if(idx2 >= raw_length) idx2 = raw_length - 1;
		float fraction = virtual_idx - idx1;

		int buf_offset1 = start_idx - idx1;
		int buf_offset2 = start_idx - idx2;

		IMUData d1 = cb->get(buf_offset1);
		IMUData d2 = cb->get(buf_offset2);

		out_matrix[step][0] = d1.ax + fraction*(d2.ax - d1.ax);
		out_matrix[step][1] = d1.ay + fraction*(d2.ay - d1.ay);
		out_matrix[step][2] = d1.az + fraction*(d2.az - d1.az);
		out_matrix[step][3] = d1.gx + fraction*(d2.gx - d1.gx);
		out_matrix[step][4] = d1.gy + fraction*(d2.gy - d1.gy);
		out_matrix[step][5] = d1.gz + fraction*(d2.gz - d1.gz);

	}
}
// 5.Dominant Axis Selection
DominantAxisSelector::DominantAxisSelector(){
	for(int i = 0; i < 3; i++){
		sum[i] = 0;
		sum_sq[i] = 0;
	}
	this->head = 0;
	this->count = 0;
	this->selected_axis = 1;
}
float DominantAxisSelector::update(float ax, float ay, float az){
	float vals[3] = {ax, ay, az};

	// Buoc 1: Neu buffer day, tru mau cu nhat
	if(count >= WINDOW){
		for(int j = 0; j < 3; j++){
			sum[j] -= history[head][j];
			sum_sq[j] -= history[head][j] * history[head][j];
		}
	}
	else count++;

	// Buoc 2: Ghi mau moi va cong vao tong
	for(int j = 0; j < 3; j++){
		history[head][j] = vals[j];
		sum[j] += vals[j];
		sum_sq[j] += vals[j] * vals[j];
	}
	head = (head + 1) % WINDOW;

	// Buoc 3: Tinh variance va chon truc (chi khi du WINDOW mau)
	if(count >= WINDOW){
		float max_var = -1;
		for(int j = 0; j < 3; j++){
			float mean = sum[j] / count;
			float var = sum_sq[j] / count - mean * mean;
			if(var > max_var){
				max_var = var;
				selected_axis = j;
			}
		}
	}

	return vals[selected_axis];
}

