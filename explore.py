from sklearn.datasets import load_iris
import matplotlib.pyplot as plot
import numpy as npy
# import pandas as pds

iris = load_iris()
dat = iris.data

# Consists of Setosas (0), Versicolors (1) and Virginicas (2)
# Each entry has [Sepal Legnth (0), Sepal Width (1), Petal Length (2), Petal Width (3)]
seto = [[], [], [], []]
vers = [[], [], [], []]
virg = [[], [], [], []]

for idx, type in enumerate(iris.target):
    if type==0:
        seto[0].append(dat[idx,0]) # Sepal Length
        seto[1].append(dat[idx,1]) # Sepal Width
        seto[2].append(dat[idx,2]) # Petal Length
        seto[3].append(dat[idx,3]) # Petal Width
    if type==1:
        vers[0].append(dat[idx,0]) # Sepal Length
        vers[1].append(dat[idx,1]) # Sepal Width
        vers[2].append(dat[idx,2]) # Petal Length
        vers[3].append(dat[idx,3]) # Petal Width
    if type==2:
        virg[0].append(dat[idx,0]) # Sepal Length
        virg[1].append(dat[idx,1]) # Sepal Width
        virg[2].append(dat[idx,2]) # Petal Length
        virg[3].append(dat[idx,3]) # Petal Width

petlmin = npy.min(seto[2] + vers[2] + virg[2])
petwmin = npy.min(seto[3] + vers[3] + virg[3])
petlmax = npy.max(seto[2] + vers[2] + virg[2])
petwmax = npy.max(seto[3] + vers[3] + virg[3])

figure, subs = plot.subplots(2,2)
figure.suptitle('Comparison of features of Iris species')
subs[0,0].scatter(seto[0], seto[1], s=seto[2], c=seto[3])
subs[0,1].scatter(vers[0], vers[1], s=vers[2], c=vers[3])
subs[1,0].scatter(virg[0], virg[1], s=virg[2], c=virg[3])
x = npy.arange(petwmin,petwmax,(petwmax-petwmin)/20)
y = npy.arange(petlmin,petlmax,(petlmax-petlmin)/20)
subs[1,1].scatter(x, y, s=x, c=y)
plot.show()