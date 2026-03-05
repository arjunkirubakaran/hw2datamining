import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from student_hw2 import perceptron_gradient_descent

# -----------------------------
# Part 1: Construct dataset
# -----------------------------
np.random.seed(0)

class_pos = np.random.randn(25, 2) + np.array([2, 2])
class_neg = np.random.randn(25, 2) + np.array([-2, -2])

X = np.vstack([class_pos, class_neg])
y = np.array([1]*25 + [-1]*25)

# -----------------------------
# Part 2: Perceptron experiments
# -----------------------------
plt.figure(figsize=(8, 6))

# plot dataset
plt.scatter(class_pos[:,0], class_pos[:,1], color='blue', label='+1')
plt.scatter(class_neg[:,0], class_neg[:,1], color='green', label='-1')

# run perceptron 10 times
for i in range(10):
    w_init = np.random.randn(2)
    b_init = np.random.randn()

    w, b = perceptron_gradient_descent(X, y, w_init, b_init)

    # plot line: w1*x + w2*y + b = 0
    x_vals = np.linspace(-6, 6, 200)
    y_vals = -(w[0]*x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, color='black', linewidth=1)

# -----------------------------
# Part 3: SVM comparison
# -----------------------------
clf = SVC(kernel='linear', C=1000)
clf.fit(X, y)

w_svm = clf.coef_[0]
b_svm = clf.intercept_[0]

x_vals = np.linspace(-6, 6, 200)
y_vals = -(w_svm[0]*x_vals + b_svm) / w_svm[1]
plt.plot(x_vals, y_vals, color='red', linewidth=2, label='SVM')

plt.legend()
plt.title("Perceptron (10 runs) vs SVM")
plt.savefig("task3_main_plot.png")
plt.close()

# -----------------------------
# Part 5: Add label noise
# -----------------------------
y_noisy = y.copy()
flip_pos = np.random.choice(np.where(y_noisy==1)[0], 5, replace=False)
flip_neg = np.random.choice(np.where(y_noisy==-1)[0], 5, replace=False)
y_noisy[flip_pos] = -1
y_noisy[flip_neg] = 1

# -----------------------------
# Part 6: Soft-margin SVM study
# -----------------------------
C_values = [0.01, 0.1, 1, 10, 100]

for C in C_values:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X, y_noisy)

    w_svm = clf.coef_[0]
    b_svm = clf.intercept_[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(class_pos[:,0], class_pos[:,1], color='blue')
    plt.scatter(class_neg[:,0], class_neg[:,1], color='green')

    x_vals = np.linspace(-6, 6, 200)
    y_vals = -(w_svm[0]*x_vals + b_svm) / w_svm[1]
    plt.plot(x_vals, y_vals, color='red')

    plt.title(f"SVM with C={C}")
    plt.savefig(f"svm_C_{C}.png")
    plt.close()
