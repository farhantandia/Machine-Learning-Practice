import sys

def factorial(n):
    fact = 1
    for num in range(2, n + 1):
        fact *= num
    return fact

def online_learning():
	file_data = sys.argv[1]
	data = open(file_data,'r').read().split('\n')
	a = int(input('alpha : '))
	b = int(input('beta : '))
	print (data)
	for case in range(len(data)):
		data_val = data[case]
		case = case + 1
		print("case ", case, ": ", data_val)
		m = 0
		N = len(data_val)
		# print(N)
		for i in range(N):
			if data_val[i] == '1':
				m += 1
		likelihood = ( factorial(N) / ( factorial(m) * factorial(N - m) ) ) * ((m / N) ** m) * ((1 - m / N) ** (N - m))
		print("Likelihood: ", likelihood)
		print("Beta prior:\ta = ", a, "\tb = ", b)
		a = a + m
		b = b+(N - m)
		print("Beta posterior:\ta = ", a, "\tb = ", b, "\n")

if __name__ == "__main__":
	online_learning()