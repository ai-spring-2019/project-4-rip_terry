"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, itertools

def read_data(filename, delimiter=",", has_header=True):
	"""Reads datafile using given delimiter. Returns a header and a list of
	the rows of data."""
	data = []
	header = []
	with open(filename) as f:
		reader = csv.reader(f, delimiter=delimiter)
		if has_header:
			header = next(reader, None)
		for line in reader:
			example = [float(x) for x in line]
			data.append(example)

		return header, data

def convert_data_to_pairs(data, header):
	"""Turns a data list of lists into a list of (attribute, target) pairs."""
	pairs = []
	for example in data:
		x = []
		y = []
		for i, element in enumerate(example):
			if header[i].startswith("target"):
				y.append(element)
			else:
				x.append(element)
		pair = (x, y)
		pairs.append(pair)
	return pairs

def dot_product(v1, v2):
	"""Computes the dot product of v1 and v2"""
	sum = 0
	for i in range(len(v1)):
		sum += v1[i] * v2[i]
	return sum

def logistic(x):
	"""Logistic / sigmoid function"""
	try:
		denom = (1 + math.e ** -x)
	except OverflowError:
		return 0.0
	return 1.0 / denom

def accuracy(nn, pairs):
	"""Computes the accuracy of a network on given pairs. Assumes nn has a
	predict_class method, which gives the predicted class for the last run
	forward_propagate. Also assumes that the y-values only have a single
	element, which is the predicted class.

	Optionally, you can implement the get_outputs method and uncomment the code
	below, which will let you see which outputs it is getting right/wrong.

	Note: this will not work for non-classification problems like the 3-bit
	incrementer."""

	true_positives = 0
	total = len(pairs)

	for (x, y) in pairs:
		nn.forward_propagate(x)
		class_prediction = nn.predict_class()
		if class_prediction != y[0]:
			true_positives += 1

		# outputs = nn.get_outputs()
		# print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

	return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNetwork:
	"""A neural network class"""

	def __init__(self, structure):
		"""creates a representation of the neural network based on structure"""

		self.network = {}
		self.structure = structure
		#create all of the nodes
		for i in range(len(structure)-1):
			for j in range(sum(structure[:i]), sum(structure[:i+1])):
				self.network[j+1] = []
				for _ in range(structure[i+1]):
					self.network[j+1].append(random.uniform(-1, 1))

		#create dummy node
		self.network[0] = [random.uniform(-1, 1) for _ in range(sum(structure[1:]))]


	def back_propagation_learning(self, training):
		"""the neural net learns using the back propogation algorithm from the book"""

		# print(len(training))

		errors = [[1000]]
		t = 0

		#randomize the weights
		for node in self.network.keys():
			for i in range(len(self.network[node])):
				self.network[node][i] = random.uniform(-1, 1)

		#while abs(sum([sum(e) for e in errors])) >= .0001 * len(training):
		for _ in range(2):
			errors = []

			for x, y in training:
				error = [0] * (sum(self.structure) + 1)
				values = []

				#propogate input to compute outputs
				for i in range(self.structure[0] + 1):
					values.append(x[i])

				#for each layer
				for i in range(1, len(self.structure)):
					#for each node in the layer
					for j in range(self.structure[i]):
						#for the input nodes
						inputs = [values[0] * self.network[0][j + sum(self.structure[1:i])]]
						#for each nonDummy input to the node
						for n in range(self.structure[i - 1]):
							#print(n + sum(self.structure[:i-1]) + 1, j, self.network[n + sum(self.structure[:i-1]) + 1][j])
							inputs.append(values[n + sum(self.structure[:i-1]) + 1] * self.network[n + sum(self.structure[:i-1]) + 1][j])
						values.append(logistic(sum(inputs)))

				#propogate error backwards
				#output layer
				for i in range(self.structure[-1]):
					#print(-1 - i, y[-1 - i], values[-1 - i], y[-1 - i] - values[-1 - i])
					error[-1 - i] = (logistic(values[-1 - i]) * (1 - logistic(values[-1 - i]))) * (y[-1 - i] - values[-1 - i])

				#for each layer backwards style 
				for i in range(len(self.structure) - 2, -1, -1):
					#for each node in that layer
					for j in range(self.structure[i]):

						error[j + sum(self.structure[:i]) + 1] = logistic(values[j + sum(self.structure[:i]) + 1]) * \
						(1 - logistic(values[j + sum(self.structure[:i]) + 1])) * \
						sum([self.network[j + sum(self.structure[:i]) + 1][k] * error[sum(self.structure[:i + 1]) + k + 1] for k in range(self.structure[i+1])])

				#calculate error of dummy weight
				error[0] = logistic(values[0]) * (1 - logistic(values[0])) * sum([self.network[0][k] * error[k + self.structure[0] + 1] for k in range(sum(self.structure[1:]))])

				#update the weights
				for i in range(len(self.structure)-1):
					for j in range(sum(self.structure[:i]), sum(self.structure[:i+1])):
						for k in range(self.structure[i+1]):
							#print(i, j, k, k + sum(self.structure[:i+1]) + 1)
							self.network[j+1][k] = self.network[j+1][k] + (1000/(1000 + t)) * values[j+1] * error[k + sum(self.structure[:i+1]) + 1]

				#update weights of dummy variable
				for i in range(len(self.network[0])):
					self.network[0][i] = self.network[0][i] + (1000/(1000 + t)) * values[0] * error[i + self.structure[0] + 1]

				errors.append(error)
			#print("network", p, "\n", self.network, '\n')

			# print(abs(sum([sum(e) for e in errors])))

			t += 1

	def estimate(self, x):
		"""finds an estimate based on the data's input values"""

		values = []

		#propogate input to compute outputs
		for i in range(self.structure[0] + 1):
			values.append(x[i])

		#for each layer
		for i in range(1, len(self.structure)):
			#for each node in the layer
			for j in range(self.structure[i]):
				#for the input nodes

				inputs = [values[0] * self.network[0][j + sum(self.structure[1:i])]]
				#for each nonDummy input to the node
				for n in range(self.structure[i - 1]):
					#print(n + sum(self.structure[:i-1]) + 1, j, self.network[n + sum(self.structure[:i-1]) + 1][j])
					inputs.append(values[n + sum(self.structure[:i-1]) + 1] * self.network[n + sum(self.structure[:i-1]) + 1][j])
				values.append(logistic(sum(inputs)))

		return values[0 - self.structure[-1]:]

def crossValidation(nn, data):
	"""n-fold cross validation"""

	n = 5

	subsets = []
	for _ in range(n):
		subsets.append([])

	#split the data into n subsets
	for i in range(len(data)):
		subsets[i % n].append(data[i])

	errors = []

	#train and test on each combination of subsets
	for i in range(n):
		training = []
		for subset in subsets:
			if subset != subsets[i]:
				training += subset

		#train and test
		nn.back_propagation_learning(training)

		error = []
		for j in range(len(subsets[i])):
			error.append(sum([abs(nn.estimate(subsets[i][j][0])[n] - subsets[i][j][1][n]) for n in range(len(subsets[i][j][1]))]))

		errors.append(sum(error)/(len(error) * len(subsets[0][1])))

	return (sum(errors) / len(errors))

def createNets(x, y, layerNumMin, layerNumMax, minNodes, maxNodes, step=1):
	"""returns a list of every neural net within the range of layers each with nodes in the range of nodes"""

	#create hidden layer structures 
	structures = []
	for i in range(layerNumMin, layerNumMax):
		structures.append(itertools.product(range(minNodes, maxNodes, step), repeat = i))

	#create nn's
	networks = []
	for struct in structures:
		for s in struct:
			networks.append(NeuralNetwork([x] + list(s) + [y]))

	return networks



def main():
	"""main"""

	header, data = read_data(sys.argv[1], ",")

	pairs = convert_data_to_pairs(data, header)

	# Note: add 1.0 to the front of each x vector to account for the dummy input
	training = [([1.0] + x, y) for (x, y) in pairs]

	# Check out the data:
	for example in training:
		print(example)

	### I expect the running of your program will work something like this;
	### this is not mandatory and you could have something else below entirely.

	networks = createNets(len(training[0][0])-1, len(training[0][1]), 1, 4, 5, 105, step=10)

	test = training[:len(training) // 10]
	validation = training[len(training) // 10:]

	errors = []
	for nn in networks:
		errors.append((crossValidation(nn, validation), nn.structure, nn))

	print(errors)

	min(errors)[2].back_propagation_learning(validation)

	print(sum([abs(min(errors)[2].estimate(test[i][0])[n] - test[i][1][n]) for i in range(len(test)) for n in range(len(test[0][1]))]) / (len(test) * len(test[0][1])))



	# for j in range(10, 15):
	# 	nn = NeuralNetwork([3, j, 3])
	# 	nn.back_propagation_learning(training)

	# 	errors = []
	# 	for i in range(len(training)):
	# 		errors.append(sum([abs(nn.estimate(training[i][0])[n] - training[i][1][n]) for n in range(len(training[i][1]))]))

	# 	print(j, sum(errors)/(len(errors) * len(training[0][1])))



if __name__ == "__main__":
	main()
