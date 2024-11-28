import numpy as np
import scipy as sp
import itertools

eps = np.finfo(float).eps

def colour_transfer_mkl(x0, x1):
    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None))) 

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None))) 

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T) 

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)
    return np.dot(x0 - mx0, t) + mx1

def colour_transfer_mkl_onechannel(x0, x1):
    a = np.cov(x0.T)
    b = np.cov(x1.T)

    # a = np.diag([a])

    a = np.diag([a])
    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None))) 

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None))) 

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T) 

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)
    params = [mx0,mx1,t]
    return np.dot(x0-mx0, t) + mx1,params

def colour_transfer_idt(i0, i1, bins=300, n_rot=10, relaxation=1):
    n_dims = i0.shape[1]
    
    d0 = i0.T
    d1 = i1.T
    
    for i in range(n_rot):
        
        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)
        
        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)
        
        for j in range(n_dims):
            
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())
            
            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _     = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]
            
            f = np.interp(cp0r, cp1r, edges[1:])
            
            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)
        
        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0
    
    return d0.T

def pad_matrix_1d(matrix, N):
    if N == 0:
        return matrix
    
    return np.pad(matrix, pad_width=((0, 0), (N, N), (0, 0)), mode='constant', constant_values=0)

def pad_matrix(matrix, N):
    if N == 0:
        return matrix
    
    return np.pad(matrix, pad_width=((0, 0), (N, N), (N, N)), mode='constant', constant_values=0)

def create_weight_matrix_1d(size, edge_width,orientation):
    weight_matrix = np.ones((size[0], size[1]))
    if orientation == 0:
        for i in range(edge_width):
            weight_matrix[i, :] = i / edge_width
            weight_matrix[size[0] - 1 - i, :] = i / edge_width
    else:
        for i in range(edge_width):
            weight_matrix[:, i] = i / edge_width
            weight_matrix[:,size[1] - 1 - i] = i / edge_width
    return weight_matrix 

def MKL(im_orig_0,im_target_0,bandnum = 5):
    pass


def match_brightness(aster_brightness, modis_brightness):
    # 使用直方图匹配技术
    from skimage import exposure
    matched_brightness = exposure.match_histograms(aster_brightness, modis_brightness)
    return matched_brightness

def brightnessBalance(aster_data,modis_data,bands = 9,tasseled_cap=False):
    if not tasseled_cap:
        # 设置权重
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


        # 计算亮度值
        aster_brightness = np.sum(aster_data[:5]*weights[:, np.newaxis, np.newaxis],axis=0)
        aster_brightness[aster_brightness==0] = 1e-5

        modis_brightness = np.sum(modis_data*weights[:, np.newaxis, np.newaxis],axis=0)
    else:
        aster=np.array(
            [
                [0.3909, -0.0318, 0.4571],
                [0.5224, -0.1031, 0.4262],
                [0.1184, 0.9422, -0.1568],
                [0.3233, 0.2512, 0.2809],
                [0.305, -0.0737, -0.2417],
                [0.3571, -0.069, -0.3269],
                [0.3347, -0.0957, -0.4077],
                [0.3169, -0.1195, -0.3731],
                [0.151, -0.0625, -0.1877],
            ],
            dtype='float64',
        )
        modis=np.array(
            [
                [0.4395, -0.4064, 0.1147],
                [0.5945, 0.5129, 0.2489],
                [0.2460, -0.2744, 0.2408],
                [0.3918, -0.2893, 0.3132],
                [0.3506, 0.4882, -0.3122],
                [0.2136, -0.0036, -0.6416],
                [0.2678, -0.4169, -0.5087],
            ],
            dtype='float64',
        )
        aster_weights = aster[:,0]
        modis_weights = modis[:,0]
        # 计算亮度值
        aster_brightness = np.sum(aster_data*aster_weights[:, np.newaxis, np.newaxis],axis=0)
        aster_brightness[aster_brightness==0] = 1e-5

        modis_brightness = np.sum(modis_data*modis_weights[:, np.newaxis, np.newaxis],axis=0)

    matched_brightness = match_brightness(aster_brightness, modis_brightness)
    aster_data_adjust = np.zeros_like(aster_data)
    for i in range(bands):
        aster_data_adjust[i] = aster_data[i] * (matched_brightness / aster_brightness)
    return aster_data_adjust

def color_transfer(merged_matrix,reference_matrix,size=256,overlap = 100,auto = True,bandMatch=True,modisRange=[0,10000],tasseled_cap=False):
    data = color_transfer_block_dealingseam(merged_matrix,reference_matrix,size=256,overlap = 100,auto = True,bandMatch=True,modisRange=[0,10000],tasseled_cap=False)
    return data

"""
均色和接缝处理；

Args:
    merged_matrix (float32,0~1,[channel,width,height]): tiles矩阵，需进行均色处理，假定width==height，且channel>=5
    reference_matrix(int16,-100~16000,[channel,width,height]): modis对应位置的参考矩阵,如果modis波段位置没有调整过，则bandMatch=False
    size (int): tile内block的尺寸大小，必须能被size整除，不能太小。
    overlap(int)：block之间的重叠区域大小，一般不要多于size的1/2大小
    auto(bool):设置未True表示size和overlap会自动调节，前面的设置无效
    bandMatch(bool): True表示modis矩阵波段经过调整；False表示未经过调整
Returns:
    np.ndarray(float32,0~1,[channel,width,height])
"""
def color_transfer_block_dealingseam(merged_matrix,reference_matrix,size=256,overlap = 100,auto = True,bandMatch=True,modisRange=[-100,16000],tasseled_cap=False):
    
    reference_matrix = (reference_matrix-modisRange[0])/(modisRange[1]-modisRange[0])
    reference_matrix[reference_matrix>1] = 1
    reference_matrix[reference_matrix<0] = 0

    channelref,widthref,heightref = np.shape(reference_matrix)
    channel,width,height = np.shape(merged_matrix)
    if auto:
        size = width//4
        overlap = size//2
    else:
        if np.mod(width,size)!=0 or width<2*size or overlap>=size:
            return merged_matrix
    
    if not tasseled_cap:
        #------根据bandMatch构建参考矩阵----------
        if bandMatch:
            correspondingImg2 = reference_matrix
        else:
            correspondingImg2 = np.zeros([5,widthref,heightref])
            correspondingImg2[0,:,:] = reference_matrix[3,:,:]
            correspondingImg2[1,:,:] = reference_matrix[0,:,:]
            correspondingImg2[2,:,:] = reference_matrix[1,:,:]            
            correspondingImg2[3,:,:] = reference_matrix[5,:,:]
            correspondingImg2[4,:,:] = reference_matrix[6,:,:]
        #边界填充
        merged_matrix_padding = pad_matrix(merged_matrix,overlap//2)
        correspondingImg2_padding = pad_matrix(correspondingImg2,overlap//2)
        current_data_xy = np.zeros_like(merged_matrix_padding)
        
        tiles = np.zeros([width//size,height//size,channel,size+overlap,size+overlap])
        tiles_modify = np.zeros([width//size,height//size,channel,size+overlap,size+overlap])
        step = width//size
        current_data_x = np.zeros([channel,width+overlap,(size+overlap)*step])

        extend_size = size+overlap
        #开始均色处理;目前除MKL的其他方式可暂时替入channel<=5之后
        if channel>5:
            for i,j in itertools.product(range(step), range(step)):
                startx = i * (extend_size - overlap)
                endx = startx + extend_size
                starty = j * (extend_size - overlap)
                endy = starty + extend_size

                tiles[i,j,:] = brightnessBalance(merged_matrix_padding[:,startx:endx,starty:endy],correspondingImg2_padding[:5,startx:endx,starty:endy],bands=channel)
        else:
            for i,j in itertools.product(range(step), range(step)):
                startx = i * (extend_size - overlap)
                endx = startx + extend_size
                starty = j * (extend_size - overlap)
                endy = starty + extend_size
                #这里可以替入其他方法
                #tiles[i,j,:],_ = MKL(merged_matrix_padding[:5,startx:endx,starty:endy],correspondingImg2_padding[:5,startx:endx,starty:endy],bandnum=channel)
                tiles[i,j,:] = brightnessBalance(merged_matrix_padding[:5,startx:endx,starty:endy],correspondingImg2_padding[:5,startx:endx,starty:endy],bands=channel)
    else:
        if bandMatch:
            return merged_matrix
        #边界填充
        merged_matrix_padding = pad_matrix(merged_matrix,overlap//2)
        correspondingImg2_padding = pad_matrix(reference_matrix,overlap//2)
        current_data_xy = np.zeros_like(merged_matrix_padding)
        
        tiles = np.zeros([width//size,height//size,channel,size+overlap,size+overlap])
        tiles_modify = np.zeros([width//size,height//size,channel,size+overlap,size+overlap])
        step = width//size
        current_data_x = np.zeros([channel,width+overlap,(size+overlap)*step])

        extend_size = size+overlap
        for i,j in itertools.product(range(step), range(step)):
            startx = i * (extend_size - overlap)
            endx = startx + extend_size
            starty = j * (extend_size - overlap)
            endy = starty + extend_size

            tiles[i,j,:] = brightnessBalance(merged_matrix_padding[:,startx:endx,starty:endy],correspondingImg2_padding[:,startx:endx,starty:endy],bands=channel,tasseled_cap=True)
        
    #首先对x方向进行接缝处理
    weight_matrix = create_weight_matrix_1d([extend_size,extend_size], overlap,orientation=0)
    weight_matrix_3d = np.stack([weight_matrix] * channel, axis=0)

    for i,j in itertools.product(range(step), range(step)):
        tiles_modify[i,j] = tiles[i,j]*weight_matrix_3d
        if i ==step-1:
            tiles_modify[i,j,:,extend_size-overlap:extend_size-overlap//2,:]= tiles[i,j,:,extend_size-overlap:extend_size-overlap//2,:]
        elif i==0:
            tiles_modify[i,j,:,overlap//2:overlap,:]= tiles[i,j,:,overlap//2:overlap,:]

        x_start = i * (extend_size - overlap)
        x_end = x_start + extend_size
        current_data_x[:,x_start:x_end, extend_size*j:extend_size*(j+1)] += tiles_modify[i, j, :, :, :]
        #current_data_xy[:5,x_start:x_end, size*j:size*(j+1)] += tiles_modify[i, j, :, :, :]
    
    #然后对y方向进行接缝处理

    weight_matrix = create_weight_matrix_1d([width+overlap,extend_size], overlap,orientation=1)
    weight_matrix_3d_y = np.stack([weight_matrix] * channel, axis=0)

    for k in range(step):
 
        y_start = k * (extend_size - overlap)
        y_end = y_start + extend_size             

        tmp =  current_data_x[:,:, extend_size*k:extend_size*(k+1)]*weight_matrix_3d_y
        if k ==step-1:
            tmp[:,:,extend_size-overlap:extend_size-overlap//2]= current_data_x[:,:,extend_size*k+extend_size-overlap:extend_size*k+extend_size-overlap//2]
        elif k==0:
            tmp[:,:,overlap//2:overlap]= current_data_x[:,:,extend_size*k+overlap//2:extend_size*k+overlap]

        current_data_xy[:,:, y_start:y_end] += tmp  
    
    return  current_data_xy[:,overlap//2:-overlap//2,overlap//2:-overlap//2]