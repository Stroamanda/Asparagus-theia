import sys
import torch
import time

def reduce_dimension(weights1, weights2):
	reduced1 = weights1[0]
	reduced2 = weights2[0]
	for index in range(1, len(weights1)):
		reduced1 += weights1[index]
		reduced2 += weights2[index]
	return reduced1, reduced2

def main():
	start_time = time.time()

	model1 = torch.load(sys.argv[1])
	model2 = torch.load(sys.argv[2])
	all_weights1 = []
	all_weights2 = []
	for index, key in enumerate(model1.keys()):
		weights1 = model1[key].tolist()
		weights2 = model2[key].tolist()
		for _ in range(max(0, len(model1[key].size()) - 1)):
			weights1, weights2 = reduce_dimension(weights1, weights2)
		if len(model1[key].size()) - 1 >= 0:
			all_weights1 += weights1
			all_weights2 += weights2

	buckets = [0, 0, 0, 0, 0]
	other = 0
	for index in range(len(all_weights1)):
		abs_diff = abs(all_weights2[index] - all_weights1[index])
		if abs_diff < 0.00001:
			buckets[4] += 1
		elif abs_diff < 0.0001:
			buckets[3] += 1
		elif abs_diff < 0.001:
			buckets[2] += 1
		elif abs_diff < 0.01:
			buckets[1] += 1
		elif abs_diff < 0.1:
			buckets[0] += 1
		else:
			other += 1

	end_time = time.time()
	print("Buckets:", buckets)
	print("Other:", other)
	print(f'Took {end_time - start_time} seconds')
	return

	print(f'The absolute difference is {abs_diff} between the two models with a size of {len(all_weights1)} parameters')
	print(f'This is an average change of {abs_diff / len(all_weights1)} for each weight')

if __name__ == "__main__":
	main()