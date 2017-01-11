import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

all_samples = np.concatenate((class1_sample,class2_sample),axis=1)
assert all_samples.shape == (3,40),"The matrix has not the dimensions 3x40"

mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1)-mean_vector).dot((all_samples[:,i].reshape(3,1)-mean_vector).T)


cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])

eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(),"Eigenvectors are not indentical"

    print("Eigenvectors: {}: \n {}".format(i+1,eigvec_sc))
    print("Eigenvalue {} from scatter matrix:{}".format(i+1,eig_val_sc[i]))
    print("Eigenvalue {} from covariance matrix:{}".format(i+1,eig_val_cov[i]))
    print("Scaling factor:",eig_val_sc[i]/eig_val_cov[i])
    print(40*'-')

#check that the eigenvector-eigenvalue calculation is correct and satisfy the equation
for i in range(len(eig_val_sc)):
    eigv = eig_vec_sc[:,i].reshape(1,3).T
    np.testing.assert_almost_equal(scatter_matrix.dot(eigv),eig_val_sc[i]*eigv,
                                   decimal=6,err_msg="",verbose=True)


class Arrow3D(FancyArrowPatch):
    def __init__(self,xs,ys,zs,*args,**kwargs):
        FancyArrowPatch.__init__(self,(0,0),(0,0),*args,**kwargs)
        self._verts3d = xs,ys,zs

    def draw(self,renderer):
        xs3d,ys3d,zs3d = self._verts3d
        xs,ys,zs = proj3d.proj_transform(xs3d,ys3d,zs3d,renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self,renderer)


for ev in eig_vec_sc:
    np.testing.assert_almost_equal(1.0,np.linalg.norm(ev))


eig_pairs = [(np.abs(eig_val_sc[i]),eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

eig_pairs.sort(key=lambda x:x[0],reverse=True)
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),eig_pairs[1][1].reshape(3,1)))

transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2,40) ,"The matrix is not 2*40 dimensional"

plt.plot(transformed[0,0:20],transformed[1,0:20],'o',markersize=7,color='blue',alpha=0.5,label='class1')
plt.plot(transformed[0,20:40],transformed[1,20:40],'^',markersize=7,color='red',alpha=0.5,label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel("x_label")
plt.ylabel("y_label")
plt.legend("upper left")
plt.show()


"""
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111,projection="3d")

ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)



for v in eig_vec_sc.T:
    a = Arrow3D([mean_x,v[0]],[mean_y,v[1]],[mean_z,v[2]],mutation_scale=20,lw=3,arrowstyle="-|>",color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('EigenVectors')

plt.show()
"""



"""
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection = '3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:],class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:],class2_sample[1,:],class2_sample[2,:],'^',markersize=8,color='red',alpha=0.5,label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()
"""