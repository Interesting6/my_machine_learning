import numpy as np
import random
import matplotlib.pyplot as plt


class my_kmeans(object):
	"""docstring for my_kmeans"""
	def __init__(self, k, max_iter=500,tol=1e-3):
		super(my_kmeans, self).__init__()
		self.k_clusters = k
		self.max_iter = max_iter
		self.tol = tol
		self.center_labels = list(range(self.k_clusters))

	def DistP2Centers(self, x, centers):
		# 一个点到所有中心点的距离
		"""X is a sample point, centers is all centers point"""
		return np.sum((x - centers)**2, axis=1)**0.5

	def random_centers(self, X, X_num):
		random_index = random.sample(range(X_num), self.k_clusters)
		centers = X[random_index]
		return centers

	def get_labels(self, X, centers):
		X_num, X_dim = X.shape
		labels = np.zeros(X_num).astype(int)
		for i in range(X_num):
			# 计算该样本点到所有中心点的距离
			p2centers = self.DistP2Centers(X[i], centers)
			# 判断该样本点
			labels[i] = int(p2centers.tolist().index(min(p2centers)))
		return labels

	def update_centers(self, X, X_labels):
		X_num, X_dim = X.shape
		centers = np.zeros((self.k_clusters, X_dim))
		for i in range(self.k_clusters):
			index = X_labels == i
			centers[i] = np.sum(X[index], axis=0)/sum(index)
		return centers


	def train(self, X, ):
		self.X_train = X
		X_num, X_dim = X.shape
		centers = self.random_centers(X, X_num)

		iter_num = 0
		while iter_num < self.max_iter:
			iter_num += 1
			X_labels = self.get_labels(X, centers)
			new_centers = self.update_centers(X, X_labels)
			diff = new_centers - centers
			if np.sum(abs(diff)) > self.tol:
				centers = new_centers
			else:
				break
		self.cluster_centers = new_centers
		return self
		# self.X_labels = self.get_labels(X, self.cluster_centers)

	def predict_one(self, x):
		# p2centers = self.DistP2Centers(x, centers)
		# x_labels = p2centers.tolist().index(min(p2centers))
		# return x_labels
		x_labels = self.get_labels(x, self.cluster_centers)[0]
		return x_labels

	def predict_all(self, X):
		X_labels = self.get_labels(X, self.cluster_centers)
		return X_labels

	def calc_accuracy(self,X,y):
		X_labels = self.predict_all(X)
		accuracy = sum(X_labels==y)/len(y)
		return accuracy

def loadData(filePath):
    # 在这里进行数据格式预处理
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    retData = np.array(retData)
    return retData, retCityName

if __name__ == '__main__':
    data, cityName =  loadData("./city.txt")

    km = my_kmeans(4)
    km = km.train(data)
    labels = km.predict_all(data)
    print(labels)

    expenses = np.sum(km.cluster_centers, axis=1)  # 每个类的平均消费
    # print(expenses)
    # 将城市按label分成设定的簇
    CityCluster = [[], [], [], []]
    CityNum = []
    for i in range(len(cityName)):
        CityCluster[labels[i]].append(cityName[i])

    #计算城市消费排名
    expenses_ = sorted(expenses,reverse=True)
    ranking = []
    for i in range(len(CityCluster)):
        ranking.append(expenses_.index(expenses[i])+1)
    ranking = [str(i) + '线消费城市' for i in ranking]
    # print(ranking)

    for i in range(len(CityCluster)):
        j = expenses_.index(expenses[i])
        print("Expenses:%.2f" % expenses[i],end='  ')    # 将每个簇的评价消费输出
        print(str(j+1),'线消费城市')
        CityNum.append(len(CityCluster[i]))
        print(CityCluster[i])

