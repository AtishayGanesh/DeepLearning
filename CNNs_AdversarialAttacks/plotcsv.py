import matplotlib.pyplot as plt
import csv
a= []
for i in range(1,9):
	f = open(f'{i}.csv',newline = '')
	a.append(csv.reader(f))
x = []
y = []
for i in a:
	xt= []
	yt = []
	for j in i:
		if str(j[1]).isalpha():
			continue
		else:
			xt.append(float(j[1]))
			yt.append(float(j[2]))
	x.append(xt)
	y.append(yt)
title = {1:'Training and Validation Accuracy',3:'Training and Validation Loss',5:'Training and Validation Accuracy with BN',7:'Training and Validation Loss with BN'}

ylabel = {1:'Accuracy',3:'Loss',5:'Accuracy',7:'Loss'}
for j in range(1,9,2):
	plt.plot(x[j-1],y[j-1])
	plt.plot(x[j],y[j])
	plt.legend(('Training','Validation'))
	plt.title(title[j])
	plt.ylabel(ylabel[j])
	plt.xlabel('Iteration')
	plt.savefig(f'{j}.png')
	plt.clf()
title2 = {5:'Accuracy Comparision with and without BN',7:'Loss Comparision with and without BN'}
for j in range(5,9,2):
	plt.plot(x[j],y[j])
	plt.plot(x[j-4],y[j-4])
	plt.legend(('With BN','Without BN'))
	plt.title(title2[j])
	plt.ylabel(ylabel[j])
	plt.xlabel('Iteration')
	plt.savefig(f'{j+4}.png')
	plt.clf()