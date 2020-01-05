import numpy as np
import matplotlib.pyplot as plt

# Ex 1
print(list(range(10)))
print(list(reversed(range(10))))

# Ex 2-3
a = list(range(10))
even = a[::2]
odd = a[1::2]

# Ex 4
a = list(range(10))
s = 0
for i in a:
    s += i

print(s)

# Ex 5
# m represents a 4x2 matrix
m = [[1, 2], [3, 4], [5, 6], [7, 8]]
s = 0
for i in range(4):
    for j in range(2):
        s += m[i][j]
        
print(s)


# Ex 6
t = np.arange(10)
print(t)

# Ex 7-9
print("type",type(t))
print("type",type(t[0]))
print(t.shape)


# Ex 10
print(t[3:7])

# Ex 11
x = list(range(10))
t = np.array(x)
print(t, type(t))

# Ex 12
a = [[1, 2], [3, 4], [5, 6], [7, 8]]
b = np.array(a)
print(b)
print(b.shape)

# Ex 13
print(np.sum(b))
print(np.sum(b, axis=0))

# Ex 14
o = np.ones((100, 2))
print(o.shape)

# Ex 15
n = np.random.normal(2.5, 1.2, (100, 2))
print(n.shape)

# Ex 16
o[:, 0] = n[:, 1]

# Ex 17
print(np.mean(o, axis=0))
print(np.std(o, axis=0))

# download 'http://www.iro.umontreal.ca/~dift3395/files/iris.txt'
iris = np.loadtxt('iris.txt')

# Ex 18-19
print(iris[0][0])
print(iris[0, 0])

# Ex 20
print(iris[:10, :4])

# Ex 21
iris[:, -1]

# Ex Play:
np.min(iris)
np.min(iris,axis=1)
np.min(iris,axis=0)
np.max(iris,axis=0)

np.argmin(iris)
np.argmin(iris,axis=1)
np.argmin(iris,axis=0)
np.argmax(iris,axis=0)

np.abs(iris[:10,1:4])

iris[0,:-1]**4.5

seq = np.arange(10)
np.random.shuffle(seq)
print(seq)


# Ex 22
plt.hist(iris[:, 1])
plt.xlabel('x_2')
plt.ylabel('quantity')

# Ex 23
plt.scatter(iris[:,2], iris[:,3])
plt.xlabel('x_3')
plt.ylabel('x_4')

# Ex 24
plt.scatter(iris[:, 2], iris[:, 3], c=iris[:, -1])
plt.xlabel('x_3')
plt.ylabel('x_4')

# Ex 25
mean = np.mean(iris[:, :4], axis=0)
std = np.std(iris[:, :4], axis=0)
plt.errorbar([1, 2, 3, 4], mean, yerr=std)
plt.xlabel('feature')
plt.ylabel('average')


# Ex 26
x = np.linspace(-3, 3, 100) # linspace generates here 100 numbers equally spaced between -3 and 3
f = np.sin(x) 
noise = np.random.randn(100)*0.1 # Gaussian noise, mean=0, variance=sqrt(0.1)
plt.plot(x, f) 
plt.plot(x, f + noise,'--')
plt.grid(True) # add a grid
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(('f(x) without noise','f(x) with noise'))

