import numpy as np
import time


# generate tr core tensors randomly
def init_tr_cores(tensor_size, tr_rank, value = 'random'):
    tr_cores = []
    ndims = len(tensor_size)
    # print(ndims)
    tr_rank.append(tr_rank[0])
  #  print(tr_rank)
    if value == 'random':
            for n in range(0, ndims):
                tr_cores.append(0.1 * np.random.rand(tr_rank[n], tensor_size[n], tr_rank[n+1]))
    elif value == 'zeros':
            for n in range(0, ndims):
                tr_cores.append( np.zeros((tr_rank[n], tensor_size[n], tr_rank[n+1])))
    elif value == 'large_value':
            for n in range(0, ndims):
                tr_cores.append(0.0005 * np.random.rand(tr_rank[n], tensor_size[n], tr_rank[n+1]))
        # print(len(tr_cores))
    return tr_cores


# reshape tensor to matrix
'''
input_tensor size:  I_1 x I_2 x ... I_N 
mat_type=1: kolda matricization → I_n x I_1I_2...I_n-1I_n+1...I_N
mat_type=2: tensor train matricization → I_1I_2...I_n x I_n+1...I_N
mat_type=3: tensor ring matricization → I_n x I_n+1...I_NI_1...I_n-1
'''


def tensor2mat(input_tensor, n, mat_type=1):
    tensor_size = input_tensor.shape
    num = input_tensor.size
    dim = len(tensor_size)
    if mat_type == 1:
        arr = np.append(n - 1, np.arange(0, n - 1))
        arr = np.append(arr, np.arange(n, dim))
        temp = input_tensor.transpose(arr)
        mat = temp.reshape(tensor_size[n-1], int(num/tensor_size[n-1]), order = 'F').copy()
    elif mat_type ==2:
        arr = np.append(np.arange(0, n), np.arange(n, dim))
        temp = input_tensor
        mat = temp.reshape(np.prod(tensor_size[0:n]), np.prod(tensor_size[n:dim]), order = 'F').copy()
    elif mat_type ==3:
        arr = np.append(np.arange(n - 1, dim), np.arange(0, n - 1))
        temp = input_tensor.transpose(arr)
        mat = temp.reshape(tensor_size[n-1], int(num/tensor_size[n-1]), order = 'F').copy()
    # print("Type: %d" %(mat_type), ", Reshape at mode-%d" %(n), ", Transpose index:", arr, ", Matrix size: %u x %u" %(mat.shape[0], mat.shape[1]))
    return mat


# merge tr_cores EXCEPT the nth core
# important operatition of TRD
def core_merge(tr_cores, n):
    dim = len(tr_cores)
   # print(dim)
    tr_cores_shift = tr_cores[n:dim] + tr_cores[0:n] # shift the nth core to the last
    #for i in range(3):
       # print(tr_cores_shift[i].shape)
    tr_mul = np.copy(tr_cores_shift[0])
    for i in range(dim-2):
       # print(i)
        temp_core = np.copy(tr_cores_shift[i+1])
        zl = tr_mul.reshape(int(tr_mul.size/temp_core.shape[0]), temp_core.shape[0],  order = 'F').copy()
        zr = temp_core.reshape(temp_core.shape[0], temp_core.shape[1] * temp_core.shape[2],  order = 'F').copy()
        tr_mul = np.dot(zl, zr)
    s1 = tr_cores_shift[0].shape[0]
    s2 = tr_cores_shift[dim-2].shape[2]
    merge_neq_out = tr_mul.reshape(s1, int(tr_mul.size/(s1 * s2)), s2,  order = 'F').copy()
    return merge_neq_out


# reshape the "matricized tensor" to tensor
def mat2tensor(input_matrix, n, tensor_size, mat_type=1):
    dim = len(tensor_size)
    if mat_type == 1:
        arr = np.append(tensor_size[n-1], int(np.prod(tensor_size[0:n-1])))
        arr = np.append(arr, int(np.prod(tensor_size[n: dim])))
        temp = input_matrix.reshape(arr, order = 'F').transpose(1, 0, 2).copy()
        output_tensor = temp.reshape(tensor_size, order = 'F').copy()
    elif mat_type == 2:
        output_tensor = input_matrix.reshape(tensor_size, order = 'F').copy()
    elif mat_type == 3:
        arr = np.append(int(np.prod(tensor_size[n-1:dim])), int(np.prod(tensor_size[0:n - 1])))
        temp = input_matrix.reshape(arr, order = 'F').transpose(1, 0).copy()
        output_tensor = temp.reshape(tensor_size, order = 'F').copy()
    # print("The size of tensor is", output_tensor.shape)
    return output_tensor


# tensor ring alternating least square
def TR_ALS(input_tensor, tr_rank, maxiter=10):
    tensor_size = input_tensor.shape
    dim = len(tensor_size)
    tr_cores = init_tr_cores(tensor_size, tr_rank)
    print('Converging TR-ALS')
    for i in range(maxiter):
        for n in range(1, dim+1):
           # print('n=', n)
            core_merge_flatten_trans = np.transpose(tensor2mat(core_merge(tr_cores, n), 2, mat_type=3))
            print("Left shape: ", core_merge_flatten_trans.shape)
            G_neq_pinv = np.linalg.pinv(core_merge_flatten_trans)  #求伪逆矩阵
            print("pinv shape: ", G_neq_pinv.shape)
            print("Right shape: ", (tensor2mat(input_tensor, n, mat_type = 3)).shape)
            tr_cores[n-1] = mat2tensor(np.dot(tensor2mat(input_tensor, n, mat_type = 3), G_neq_pinv), 2, tr_cores[n-1].shape, mat_type=1)
            print("TR-", n-1, " shape: ", tr_cores[n-1].shape)
        print('.', end='')
    print('Finished!')
    return tr_cores


if __name__=='__main__':
    f = open("./test.txt", "w")
    for i in range(100, 200, 100):
        f.write(str(i)+" ")
        print(i)
        r = [1, int(i*0.1), int(i*0.1), 1]
        tensor = np.random.random((i, i, i))
        time_start = time.time()
        TR_ALS(tensor, r, 2)
        timeall = time.time()-time_start
        print('time cost ', timeall, 's')
        f.write(str(timeall)+" \n")
    f.close()
