"""

This is my implementation of StyleGAN2 + ADA which is based off the official pytorch repository https://github.com/NVlabs/stylegan2-ada-pytorch
I reimplement everything from the training folder. I copy the code directly two times and I have highlighted where this is and explained why I have done this 


I also looked at https://nn.labml.ai/gan/stylegan/index.html 

"""




import numpy as np
import torch
import mathutils

def upsample(x, up):
     # insert zeros
     w = x.new_zeros(up, up)
     w[0, 0] = 1

     return torch.nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, up, up), stride=up, groups=x.size(1))
    

def pad_crop(x, padding_list):

    padding_list_pad = [max(x, 0) for x in padding_list]
    
    x = torch.nn.functional.pad(x, padding_list_pad)
    
    padding_list_crop = [max(-x, 0) for x in padding_list]

    crop_x0 = padding_list_crop[0]
    crop_x1 = x.shape[3] - padding_list_crop[1]
    crop_y0 = padding_list_crop[2]
    crop_y1 = x.shape[2] - padding_list_crop[3]
    x = x[:, :, crop_y0 : crop_y1 , crop_x0 : crop_x1]
    
    return x 


def downsample(x, down):
    if isinstance(down, int):
        downx, downy = down, down
    
    elif isinstance(down, list) and len(down) == 2:
        downx, downy = down
        
    x = x[:, :, ::downy, ::downx]
    
    return x
    

def convolve(x, f, num_channels):
    
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)

    f.to(x.device)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels).to(x.device)
    else:
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    

    
    return x
    

def upfirdn2d_(x, f = None, up = 1, padding = None, down = 1, adjust_padding_= False):

    padding = return_padding(padding)

    
    if adjust_padding_ == True: 
        if f == None:
            fw, fh = 1, 1
        else:
            fw, fh = int(f.shape[-1]), int(f.shape[0])

        padding = adjust_padding(fw, fh, padding, up, down)
    
    _, num_channels, _, _ = x.shape
    if up > 1:
        x = upsample(x, up)
        
    if padding != None: 
        x = pad_crop(x, padding)
    
    if f != None:
        x = convolve(x, f, num_channels)
        
    if down > 1:
        x = downsample(x, down)
        
    return x
    

def return_padding(padding):
    if padding == None:
        return [0, 0, 0, 0]
    elif isinstance(padding, int):
        return [padding, padding, padding, padding]
    
    elif len(padding) == 4:
        return padding 
    
    else:
        print('something wrong with padding')
    
    
def adjust_padding(fw, fh, padding, up, down):

    px0, px1, py0, py1 = padding[0], padding[1], padding[2], padding[3]
    
    #Adjust padding to account for up and down sampling 
    if up > 1:
        x0, x1, y0, y1 = (fw + up - 1) // 2, (fw - up) // 2, (fh + up - 1) // 2, (fh - up) // 2
        px0 = px0 + x0 
        px1 = px1 + x1
        py0 = py0 + y0
        py1 = py1 + y1
        
    if down > 1:
        x0, x1, y0, y1 = (fw - down + 1) // 2, (fw - down) // 2, (fh - down + 1) // 2, (fh - down) // 2
        px0 = px0 + x0 
        px1 = px1 + x1
        py0 = py0 + y0
        py1 = py1 + y1
        
    
    return [px0, px1, py0, py1]



# Helpers for constructing transformation matrices.
def create_identity_matrix(size, device):
    t = torch.zeros(size, size, device =device).fill_diagonal_(1)
    return t 

def rotation_2d_matrix(theta, device):
  #rotation = torch.tensor(mathutils.Matrix.Rotation(theta, 3, 'X'))
  #rotation = rotation.flip([0,1])
  #I found that mathutils.Matrix.Rotation would sometimes return the wrong sign so I created a rotation matrix manually
  rotation = torch.tensor([[torch.cos(theta), torch.sin(-theta), 0], [torch.sin(theta), torch.cos(theta),  0], [0, 0, 1]], device = device)
  return rotation 


def rotation_3d_matrix(theta, device, dimension_of_rotation = ([1, 1, 1])/np.sqrt(3)):
  rotation = torch.tensor(mathutils.Matrix.Rotation(theta, 4, dimension_of_rotation) , device = device)
  return rotation


def scale_2d_matrix(X_scale, Y_scale, device):
  scale_matrix = torch.tensor(mathutils.Matrix.Scale(X_scale, 3, (1,0,0))) + torch.tensor(mathutils.Matrix.Scale(Y_scale, 3, (0,1,0))) - create_identity_matrix(3, device = 'cpu')
  scale_matrix = scale_matrix.to(device = device)
  return scale_matrix

def scale_3d_matrix(X_scale, Y_scale, Z_scale, device): 
  scale_matrix = torch.tensor(mathutils.Matrix.Scale(X_scale, 4, (1,0,0))) + torch.tensor(mathutils.Matrix.Scale(Y_scale, 4, (0,1,0))) + torch.tensor(mathutils.Matrix.Scale(Z_scale, 4, (0,0,1))) - 2 * create_identity_matrix(4, device = 'cpu')
  scale_matrix = scale_matrix.to(device = device)
  return scale_matrix


def translation_3d_matrix(X_translation, Y_translation, Z_translation, device):
  translation_matrix = torch.tensor(mathutils.Matrix.Translation((X_translation, Y_translation, Z_translation)), device = device)
  return translation_matrix

def translation_2d_matrix(X_translation, Y_translation, device):
  #mathutils does not have a class supporting 2D translations 
  translation_matrix = torch.tensor([[1,0,X_translation], [0,1, Y_translation], [0, 0, 1]], device = device)
  return translation_matrix



#Function to apply the 2d transformation to a batch
def return_2d_transformation(tx, ty, transformation_function, device):
  if isinstance(tx, torch.Tensor) and isinstance(ty, torch.Tensor):
    #Apply translation 
    lst_of_matrix = []
    for (x,y) in zip(tx, ty): 
      lst_of_matrix.append(transformation_function(x,y, device =device))
    t = torch.stack(lst_of_matrix)
    return t
    
  #Apply scale
  elif ty != None: 
    lst_of_matrix = []
    for x in tx: 
      lst_of_matrix.append(transformation_function(x,ty, device = device))
    t = torch.stack(lst_of_matrix)
    return t

  else: 
    #Apply rotation
    lst_of_matrix = []
    for x in tx:
      lst_of_matrix.append(transformation_function(x, device = device))
    t = torch.stack(lst_of_matrix)
    return t 


#Function to apply transformation to the three colour channels to a batch 
def return_3d_transformation(tx, ty, tz, transformation_function, device):
  if isinstance(tx, torch.Tensor) and isinstance(ty, torch.Tensor) and isinstance(tz, torch.Tensor) :
    #Apply translation or scale
    lst_of_matrix = []
    for (x,y,z) in zip(tx, ty, tz): 
      lst_of_matrix.append(transformation_function(x,y,z, device= device))
    t = torch.stack(lst_of_matrix)
    return t

  else: 
    #Apply rotation
    lst_of_matrix = []
    for x in tx:
      lst_of_matrix.append(transformation_function(x, device = device))
    t = torch.stack(lst_of_matrix)
    return t 

#Return tensor with probability of transformation
def create_magnitude_of_transformation(shape, intialisation_function, scaling_factor, p, mode, device, torch_like ='zeros'):
  batch_size = shape[0]
  if mode == 'rotation':
     t = (torch.rand(shape, device=device) * 2 - 1) * np.pi 
  elif mode =='translation':
     t = (torch.rand([batch_size, 2], device=device) * 2 - 1)  * scaling_factor
  elif mode == 'fractional translation':
    t = torch.randn([batch_size, 2], device=device) * scaling_factor
  elif mode == '3d translation':
    t = torch.randn([batch_size], device=device) * scaling_factor
  else: 
    t = intialisation_function(torch.rand(shape, device=device) * scaling_factor)
    
  if torch_like == 'ones':
    t = torch.where(torch.rand(shape, device=device) < p, t, torch.ones_like(t))
  
  elif torch_like == 'zeros':
    t = torch.where(torch.rand(shape, device=device) < p, t, torch.zeros_like(t))

  return t




#Custom replacement for torch.nn.functional.grid_sample - taken from stylegan2-ada-pytorch repository. I tried using torch.nn.functional.grid_sample instead however the derivative for grid_sampler_2d_backward is not implemented
class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid


class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)

        return grad2_grad_output, grad2_input, grad2_grid


class AugmentPipe(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))      
    
        self.register_buffer('Hz_geom', torch.tensor([ 0.0109,  0.0025, -0.0834, -0.0342,  0.3472,  0.5569,  0.2390, -0.0514,-0.0149,  0.0316,  0.0012, -0.0055]))

        

    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device


        
        p = self.p 
        p_rot = 1 - torch.sqrt((1 - 1 * p).clamp(0, 1))
        #2D TRANSFORMATIONS
        #xflip 
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = torch.floor, scaling_factor = 2, p = p, mode = None, device = device)
        xflip_transformation = return_2d_transformation(t, 1, scale_2d_matrix, device = device)
        
        #rotate
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = torch.floor, scaling_factor = 4, p = p, mode = None, device= device)
        rotate_transformation = return_2d_transformation(np.pi/2 * t, None, rotation_2d_matrix, device = device)
        
        #xint
        t = create_magnitude_of_transformation(shape = [batch_size, 1], intialisation_function = None, scaling_factor = 0.125, p = p, mode = 'translation', device = device)
        first_term, second_term = torch.round(t[:,0] * width), torch.round(t[:,1] * height)
        xint_transformation = return_2d_transformation(-first_term, -second_term, translation_2d_matrix, device = device)
        
        #scale 
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = torch.exp2, scaling_factor = 0.2, p = p, mode = None, torch_like = 'ones', device = device)
        scale_transformation = return_2d_transformation(1/t, 1/t, scale_2d_matrix, device = device)
        
        #rotate
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = None, scaling_factor = None, p = p_rot, mode = 'rotation', device = device)
        rotate_pre_aniso_transformation = return_2d_transformation(t, None, rotation_2d_matrix, device = device)
        
        #aniso
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = torch.exp2, scaling_factor = 0.2, p =p, mode  = None, torch_like = 'ones', device = device)
        aniso_transformation = return_2d_transformation(1/t, t, scale_2d_matrix, device = device) 
        
        #rotate
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = None, scaling_factor = None, p = p_rot, mode = 'rotation', device = device)
        rotate_post_aniso_transformation = return_2d_transformation(t, None, rotation_2d_matrix, device = device)
        
        #xfrac
        t = create_magnitude_of_transformation(shape = [batch_size, 1], intialisation_function = None, scaling_factor = 0.125, p = p, mode = 'fractional translation', device = device)
        xfrac_transformation = return_2d_transformation(-t[:,0] * width, -t[:,1] * height, translation_2d_matrix, device = device)  
        
        #Create 2D transformation
        I = create_identity_matrix(3, device = device)
        I = torch.matmul(I , xflip_transformation)
        I = torch.matmul(I , rotate_transformation)
        I = torch.matmul(I , xint_transformation)
        I = torch.matmul(I , scale_transformation)
        I = torch.matmul(I , rotate_pre_aniso_transformation)
        I = torch.matmul(I , aniso_transformation)
        I = torch.matmul(I , rotate_post_aniso_transformation)
        I = torch.matmul(I , xfrac_transformation)
        

        
        #cx = (width - 1) / 2
        #cy = (height - 1) / 2
        if height != width: 
          print('height and width not equal')
        
        scale = (width - 1) / 2
        cp = torch.tensor([[-scale, -scale, 1], [scale, -scale, 1], [scale, scale, 1], [-scale, scale, 1]], device=device) # [idx, xyz]
        cp = torch.matmul(I,  cp.t()) # [batch, xyz, idx]
        
        margin = cp[:, :2, :].transpose(1, 0).reshape(2, -1) # [xy, batch * idx]
        margin = torch.cat([-margin, margin]).amax(dim=1) # [x0, y0, x1, y1]
        margin = margin + torch.tensor([6 - scale] * 4, device=device)
        margin = torch.clamp(margin, min=0, max=scale*2)
        mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)
        
        # Pad image and adjust origin.
        images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
        #G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
        translation_transformation = translation_2d_matrix((mx0 - mx1) / 2, (my0 - my1) / 2,  device = device) 
        I = torch.matmul(translation_transformation, I)
        
        # Upsample.
        images = upfirdn2d_(x= images, f = self.Hz_geom, up = 2, padding = 0, down = 1, adjust_padding_= True)
        #images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
        #G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
        scale_transformation = scale_2d_matrix(2, 2, device = device) 
        inv_scale_transformation = scale_2d_matrix(1/2, 1/2, device = device) 
        I = torch.matmul(scale_transformation, I)
        I = torch.matmul(I, inv_scale_transformation  )
        
        #G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)
        translation_transformation = translation_2d_matrix(-1/2 , -1/2,  device = device)
        inverse_translation_transformation = translation_2d_matrix(1/2, 1/2,  device = device)
        I = torch.matmul(translation_transformation, I)
        I = torch.matmul(I, inverse_translation_transformation)
        
        # Execute transformation.
        shape = [batch_size, num_channels, (height + 6) * 2, (width + 6) * 2]
        #G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
        scale_transformation = scale_2d_matrix(2 / width, 2 / height,  device = device) 
        inverse_scale_transformation = scale_2d_matrix(2 / shape[3], 2 / shape[2], device = device) 
        I = torch.matmul(scale_transformation, I)
        I = torch.matmul(I, inverse_scale_transformation)
        grid = torch.nn.functional.affine_grid(theta=I[:,:2,:], size=shape, align_corners=False)
        #images = torch.nn.functional.grid_sample(input = images, grid = grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        images = _GridSample2dForward.apply(images, grid)
        
        # Downsample and crop.
        images = upfirdn2d_(x=images, f=self.Hz_geom, down=2, padding=-6, adjust_padding_ = True)
        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        
        axis = torch.tensor([1, 1, 1, 0], device = device) / np.sqrt(3)
        axis_outer = axis.outer(axis)
        
        C_inv = create_identity_matrix(4, device = device)
        I_4 =  create_identity_matrix(4, device = device)
        
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = None, scaling_factor = 0.2, p = p, mode = '3d translation', device = device)
        brightness_transformation = return_3d_transformation(t, t, t, translation_3d_matrix, device = device) 
        
        #contrast
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = torch.exp2, scaling_factor = 0.5, p = p, mode = None, torch_like = 'ones', device = device)
        contrast_transformation = return_3d_transformation(t, t, t, scale_3d_matrix, device = device)
        
        #luma
        t = create_magnitude_of_transformation(shape = [batch_size, 1, 1], intialisation_function = torch.floor, scaling_factor = 2, p = p, mode = None, device = device)
        luma_transformation = (I_4 - 2 * axis_outer * t) 
        
        #Hue
        t = create_magnitude_of_transformation(shape = [batch_size], intialisation_function = None, scaling_factor = None, p = p, mode = 'rotation', device = device)
        hue_transformation = return_3d_transformation(t, None, None, rotation_3d_matrix, device = device) 
        
        #Saturation
        t = create_magnitude_of_transformation(shape = [batch_size, 1, 1], intialisation_function = torch.exp2, scaling_factor = 1, p = p, mode = None, torch_like = 'ones', device = device)
        saturation_transformation = (axis_outer + (I_4 - axis_outer) * t) 
        
        C_inv = torch.matmul(brightness_transformation, C_inv)
        C_inv = torch.matmul( contrast_transformation, C_inv)
        C_inv = torch.matmul(luma_transformation, C_inv)
        C_inv = torch.matmul(hue_transformation, C_inv)
        C_inv = torch.matmul(saturation_transformation, C_inv)
        
        images = images.reshape([batch_size, num_channels, height * width])
        images = torch.matmul(C_inv[:, :3, :3], images) 
        images += C_inv[:, :3, 3:]

        images = images.reshape([batch_size, num_channels, height, width])

        return images


import numpy as np
import torch




TRAINING = True 
#DEVICE = 'cuda'


def get_weight(w, lrmul, shape, gain = 1, return_multipled_weight = True):
  if w == None:
    #Need to intialise weights
    if lrmul != None: 
        init_std = 1.0 / lrmul
        w = torch.empty(shape).normal_(mean=0,std= init_std)

        return w
    
    elif lrmul == None: 
        return torch.randn(shape)
    
  fan_in = np.prod(shape[1:])
  he_std = gain / np.sqrt(fan_in)
  
  if lrmul == None: 
      lrmul = 1
  runtime_coef = he_std * lrmul
  
  if return_multipled_weight == True: 
      return w * runtime_coef
  
  else: 
      return runtime_coef
  


def normalise_filter(f):
    
    f = torch.tensor(f, dtype=torch.float32)
    f = f.outer(f)
    f /= f.sum()
    
    return f

def cast_to_data_type(parameter, data_type):
    
    try:
        return parameter.to(data_type)
    
    except:
        return None

def return_gain(activation):
    if activation == 'relu' or activation == 'lrelu':
        return np.sqrt(2)  
    return 1

def return_activation_gain(activation):
    if activation == 'relu' or activation == 'lrelu':
        return np.sqrt(2)
    
    return 1 

def return_activation(activation, x):

  if activation =='lrelu':
    #alpha = torch.tensor(0.2)
    return torch.nn.functional.leaky_relu(x)

  elif activation == 'relu':
    return torch.nn.functional.relu(x)

  elif activation == 'linear':
    return x


def return_reshape(ndim, dim = 1): 
    dimension_of_x = [1] * ndim
    dimension_of_x[dim] = -1 
    
    return dimension_of_x


def matmul_x_and_w_and_add_bias(x, b, w, activation, conv_clamp = None):
    if w != None: 
        if activation == 'linear' and b != None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
            return x
        w = w.to(device = x.device)
        x = x@(w.T)
        
    if b != None: 
        x = x + b.reshape(return_reshape(x.ndim, dim = 1))
    x = return_activation(activation, x)
    gain = return_gain(activation)
    x = x * gain
    
    if conv_clamp != None: 
       x = torch.clamp(x, min=-256, max=265)
     
    x = x.to()
    return x 

def truncation(apply_truncation, x, w_avg, num_ws, truncation_cutoff):
    if apply_truncation == 1: 
        return x
    
    if num_ws is None or truncation_cutoff is None:
        return w_avg.lerp(x, apply_truncation)
    
    else: 
        x[:, :truncation_cutoff] = w_avg.lerp(x[:, :truncation_cutoff], apply_truncation)
        return x 



def normalize_2nd_moment(x, dim=1, eps=1e-8):
    intermediate = (torch.square(x).mean(dim=1, keepdim = True) +  1e-8)
    reciprocal = torch.reciprocal(intermediate)
    sqrtr = torch.sqrt(reciprocal)
    
    x *= sqrtr
    return x




def conv2d_resample_(x, w, f=None, up=1, down=1, padding=None, groups=1): 
    
    _, num_channels, _, _ = x.shape
    

    padding = return_padding(padding)

    
    if f == None:
        fw, fh = 1, 1
    else:
        fw, fh = int(f.shape[-1]), int(f.shape[0])
    
    padding = adjust_padding(fw, fh, padding, up, down)
    
    if up > 1:
        x = upfirdn2d_(x=x, f=f, up=up, padding=padding)
    else:
        x = upfirdn2d_(x=x, f=None, up=up, padding=padding)
    

    x = x.to(device = w.device)
    x = torch.nn.functional.conv2d(x=x, weight=w, groups=groups)
    
    if down > 1:
        x = upfirdn2d_(x=x, f=f, down=down)
        
    return x 


def broadcast(x, number_of_channels_to_broadcast):
    if number_of_channels_to_broadcast == None: 
        return x
    
    return x.unsqueeze(1).repeat([1, number_of_channels_to_broadcast, 1])

def wrapper_function(function, argument1, argument2):
    
    try:
        result = function(argument1, argument2)
        return result
    except: 
        return argument1

def to_dtype(x, dtype, memory_format):
    
    if x == None:
        return None
    
    if memory_format == None:
        x = x.to(dtype = dtype)
    else:
        x = x.to(dtype = dtype, memory_format=memory_format)
        return x

# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def modulated_conv2d(
    x, weight,styles, noise = None, up = 1,down = 1, padding= 0,resample_filter = None,demodulate = True,flip_weight = True,fused_modconv = True):
    batch_size = int(x.shape[0])
    out_channels, in_channels, in_height, in_width = weight.shape
    styles = styles.to(device = x.device)
    weight = weight.to(device = x.device)


    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * in_height * in_width) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    
    s = styles[:, np.newaxis, :, np.newaxis, np.newaxis]
    weights = weight[np.newaxis]
    weights = weights * s
    if demodulate:
      sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-08)
      #Multiple kernel weights by demodulation coefficient 
      weights = weights * sigma_inv
 
    weights = weights.reshape(-1, in_channels, in_height, in_width)
    x = x.reshape(1, -1, *x.shape[2:])


    x = conv2d_resample_(x=x, w=weights.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
        
    return x




class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,in_features,out_features, bias = True,activation = 'linear', lr_multiplier = 1, bias_init= 0):
        super().__init__()
        self.activation = activation
        self.shape = [out_features, in_features]

        self.weight = torch.nn.Parameter(get_weight(w= None, lrmul = lr_multiplier, shape = self.shape))
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) 
        self.lr_multiplier = lr_multiplier

    def forward(self, x):
        
        w = get_weight(w =self.weight.to(x.dtype), lrmul = self.lr_multiplier, shape = self.shape)
        b = self.bias * self.lr_multiplier
        
        b = to_dtype(b, dtype = x.dtype, memory_format = None)
        
        
        x = matmul_x_and_w_and_add_bias(x= x, b = b, w= w, activation = self.activation)
        return x




class Conv2dLayer(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,bias= True,activation= 'linear',up = 1,down= 1,resample_filter = [1,3,3,1], conv_clamp = None, channels_last = False, trainable= True):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', normalise_filter(resample_filter))

        
        self.shape = [out_channels, in_channels, kernel_size, kernel_size]

        weight = get_weight(w=None, lrmul = None, shape = self.shape)
        
        if bias != None:
            bias = torch.zeros([out_channels]) 
        else:
            bias = None
        
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            if bias != None:
                self.bias = torch.nn.Parameter(torch.zeros([out_channels])) 
            else:
                self.bias = None 
        else:
            self.register_buffer('weight', weight)
            if bias != None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = get_weight(w = self.weight, lrmul = 1, shape = self.shape)
        padding = self.shape[2] // 2
        

        if self.bias != None: 
            b = (self.bias).to(x.dtype)
        
        x = conv2d_resample_(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=padding)

        if self.conv_clamp != None: 
            clamp = self.conv_clamp * gain 
            x = matmul_x_and_w_and_add_bias(x, b = b, w = None, activation =self.activation, conv_clamp = clamp)

            return x
        
        else: 
            x = matmul_x_and_w_and_add_bias(x, b = b, w = None, activation =self.activation, conv_clamp = None)
            
        return x




class MappingNetwork(torch.nn.Module):
    def __init__(self,z_dim, c_dim, w_dim,num_ws,num_layers= 8, embed_features  = None, layer_features  = None, activation = 'lrelu', lr_multiplier   = 0.01, w_avg_beta= 0.995):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        features_list = [z_dim] + [512]*7 + [w_dim]
        layers = [FullyConnectedLayer(features_list[i], features_list[i+1], activation=activation, lr_multiplier=lr_multiplier) for i in range(len(features_list) - 1)]
        self.network = torch.nn.Sequential(*layers)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):

        x = None
        
        if self.z_dim > 0:
            x = z.to(torch.float32)
            intermediate = (torch.square(x).mean(dim=1, keepdim = True) +  1e-8)
            reciprocal = torch.reciprocal(intermediate)
            sqrtr = torch.sqrt(reciprocal)
            x *= sqrtr
            
                
        x = self.network(x)        
           
        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        x = broadcast(x, self.num_ws)

        # Apply truncation.
        x = truncation(truncation_psi, x, self.w_avg, self.num_ws, truncation_cutoff)

        return x
    

    




class SynthesisLayer(torch.nn.Module):
    def __init__(self,in_channels, out_channels,w_dim,resolution, kernel_size = 3, up= 1,use_noise= True, activation= 'lrelu', resample_filter = [1,3,3,1], conv_clamp= None, channels_last = False):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.register_buffer('resample_filter', normalise_filter(resample_filter))
        self.act_gain = return_gain(activation) 
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(get_weight(w= None, lrmul = None, shape = [out_channels, in_channels, kernel_size, kernel_size]))
        self.register_buffer('noise_const', torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        styles = self.affine(w)

        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        
        elif noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
        else:
            noise = None

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=1, resample_filter=self.resample_filter, fused_modconv=fused_modconv)

        x = matmul_x_and_w_and_add_bias(x = x, w=None, b = self.bias.to(x.dtype), activation=self.activation, conv_clamp=True)
        return x




class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(get_weight(w = None, lrmul = None, shape = [out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.shape = [out_channels, in_channels, kernel_size, kernel_size]


    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * get_weight(w = w, lrmul = None, shape = self.shape, return_multipled_weight=False)
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = matmul_x_and_w_and_add_bias(x = x, w= None, b = self.bias.to(x.dtype), activation='linear', conv_clamp=True)

        return x




class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,out_channels,w_dim,resolution, img_channels,is_last,architecture= 'skip',resample_filter = [1,3,3,1], conv_clamp= None,use_fp16 = False, fp16_channels_last  = False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', normalise_filter([1,3,3,1]))
        self.num_conv, self.num_torgb = 0, 0

        #Using skip architecture 
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
            self.num_conv = self.num_conv - 1
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
            conv_clamp=conv_clamp, channels_last=self.channels_last)
        
        self.num_conv = self.num_conv + 2
        self.num_torgb = self.num_torgb + 1
       
     

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):

        w_iter = iter(ws.unbind(dim=1))
        if self.in_channels == 0:
            x = self.const
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        
        x = to_dtype(x, dtype=torch.float32, memory_format=torch.contiguous_format)


        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=True, **layer_kwargs)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=True, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=True, **layer_kwargs)

        # ToRGB.
        if img != None:
            #upsample
            img = upfirdn2d_(img, f =self.resample_filter, up = 2, padding = 0, down = 1, adjust_padding_ = True)
        
        y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
        img = wrapper_function(torch.add, y, img)
    

        return x, img
    
    



class SynthesisNetwork(torch.nn.Module):
    def __init__(self, w_dim,img_resolution,img_channels, channel_base = 32768, channel_max= 512, num_fp16_res= 0, **block_kwargs):
        
        super().__init__()
        self.list_of_synthesis_blocks = []
        self.w_dim, self.img_resolution, self.img_resolution_log2, self.img_channels = w_dim, img_resolution, int(np.log2(img_resolution)), img_channels
        self.block_resolutions = [4, 8, 16, 32, 64, 128, 256]
        channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        in_ = [channels_dict[res // 2] if res > 4 else 0 for res in self.block_resolutions]
        out_ = [channels_dict[res] for res in self.block_resolutions]
        use_fp16 = [res >= fp16_resolution for res in self.block_resolutions]
        is_last = [(res == img_resolution) for res in self.block_resolutions]

        for i in range(len(self.block_resolutions)):
            res = self.block_resolutions[i]
            num_in_channels, num_out_channels, use_fp16, last = in_[i], out_[i], True, is_last[i]
            block = SynthesisBlock(num_in_channels, num_out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            
            if last == True: 
                self.num_ws += block.num_torgb + block.num_conv
                
            else: 
                self.num_ws +=  block.num_conv
                
        
            self.list_of_synthesis_blocks.append(block)
            
    def forward(self, ws, **block_kwargs):
        block_ws = []
    
        x, img = None, None 
        
        ws = ws.to(torch.float32)
        w_idx = 0
            
            
        for i in range(len(self.list_of_synthesis_blocks)):
            block = self.list_of_synthesis_blocks[i]
            sum_ = block.num_conv + block.num_torgb
            block_ws.append(ws.narrow(1, w_idx, sum_))
            w_idx = w_idx + block.num_conv

        
        for i in range(len(self.list_of_synthesis_blocks)):
           block, ws_current = self.list_of_synthesis_blocks[i], block_ws[i]
           x, img = block(x, img, ws_current, **block_kwargs)
            
            
        return img




class Generator(torch.nn.Module):
    def __init__(self, z_dim, c_dim, w_dim,img_resolution, img_channels, mapping_kwargs = {},synthesis_kwargs    = {}):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim, self.w_dim, self.img_resolution, self.img_channels = z_dim, w_dim, img_resolution, img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img



class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, tmp_channels, out_channels,resolution,img_channels,first_layer_idx, architecture = 'resnet',activation = 'lrelu', resample_filter = [1,3,3,1],conv_clamp = None, use_fp16= False, fp16_channels_last  = False,freeze_layers= 0):

        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', normalise_filter([1,3,3,1]))
        self.architecture = architecture
        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
                
                
        trainable_iter = trainable_gen()
        layer1, layer2, layer3, layer4 = None, None, None, None
        
        
        #Use resnet architecture
        if in_channels == 0:
           layer1 = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation='lrelu',
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        layer2 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation='lrelu',
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        layer3 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation='lrelu', down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        layer4 = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

        if layer1 != None:
            self.fromrgb = layer1
            
        self.conv0 = layer2 
        self.conv1 = layer3 
        self.skip = layer4

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        x = to_dtype(x, dtype, memory_format)
        

        # FromRGB.
        if self.in_channels == 0:

            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = wrapper_function(torch.add, y, x)
            img = None

            
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = torch.add(y, x)

        return x, img






class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):

        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(8), torch.as_tensor(N)) 
        c = C // self.num_channels
        
        y = x.reshape(G, -1, self.num_channels, c, H, W)   
        std = torch.sqrt(y.var(dim=0) + 1e-8)
        std = std.mean(dim=[2,3,4])           
        std = std.reshape(-1, self.num_channels, 1, 1)         
        std = std.repeat(G, 1, H, W)           
        return torch.cat([x, std], dim=1)       

class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,in_channels, cmap_dim, resolution, img_channels,  architecture = 'resnet', mbstd_group_size= 4, mbstd_num_channels  = 1, activation= 'lrelu',  conv_clamp= None,     
    ):

        super().__init__()
        self.in_channels, self.resolution, img_channels  = in_channels, resolution, img_channels

        self.mbstd = MinibatchStdLayer(group_size=8, num_channels=1)
        self.conv = Conv2dLayer(in_channels + 1, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, x, img, cmap, force_fp32=False):

        x = to_dtype(x, dtype= torch.float32, memory_format = torch.contiguous_format)

        x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)


        return x




class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          
        img_resolution,                 
        img_channels,                   
        architecture        = 'resnet', 
        channel_base        = 32768,    
        channel_max         = 512,      
        num_fp16_res        = 0,        
        conv_clamp          = None,    
        cmap_dim            = None,    
        block_kwargs        = {},       
        mapping_kwargs      = {},       
        epilogue_kwargs     = {},       
    ):
        super().__init__()
        self.img_resolution, self.img_resolution_log2, self.img_channels = img_resolution, int(np.log2(img_resolution)), img_channels

        self.block_resolutions = [256, 128, 64, 32, 16, 8]
        channels_dict = {256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512}
        fp16_resolution = 32


        common_kwargs = dict(img_channels=img_channels,  conv_clamp=256)
        cur_layer_idx = 0
        self.list_of_discriminator_blocks = []
        
        in_channels = [channels_dict[res] if res < img_resolution else 0 for res in self.block_resolutions]
        tmp_channels = [channels_dict[res] for res in self.block_resolutions]
        out_channels = [channels_dict[res // 2] for res in self.block_resolutions]
        use_fp16 = [(res >= fp16_resolution) for res in self.block_resolutions]
        
        for i in range(len(self.block_resolutions)):
            res = self.block_resolutions[i]
            in_, tmp_, out_, use_ = in_channels[i], tmp_channels[i], out_channels[i], use_fp16[i]
            block = DiscriminatorBlock(in_, tmp_, out_, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_, **block_kwargs, **common_kwargs)
            self.list_of_discriminator_blocks.append(block)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
            

        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        
        for i in range(len(self.list_of_discriminator_blocks)):
            block = self.list_of_discriminator_blocks[i]
            x, img = block(x, img, **block_kwargs)
 
        x = self.b4(x, img, None)
        return x

import os
import json
import torch

import copy
import PIL.Image
import numpy as np
import torch



import zipfile


import numpy as np



class StyleGAN2Loss():
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)



    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):



        if phase == 'Gmain':
                z, c = gen_z, gen_c
  
                _gen_ws = self.G_mapping(z, c)
                random_number = np.random.normal()
                if random_number < self.style_mixing_prob:
                  cutoff = np.random.uniform(low=1, high= _gen_ws.shape[1]) 
                else: 
                  cutoff = _gen_ws.shape[1]
                _gen_ws[:, cutoff:] = self.G_mapping(torch.randn_like(gen_z), c, skip_w_avg_update=True)[:, cutoff:]
                gen_img = self.G_synthesis(_gen_ws)

                img = self.augment_pipe(gen_img)
                gen_logits = self.D(img, c)

                loss_Gmain = torch.log(1 + torch.exp(-gen_logits))
                loss_Gmain.mean().mul(gain).backward()


        if phase == 'Greg':
                batch_size = gen_z.shape[0] // self.pl_batch_shrink

                z, c  = gen_z[:batch_size], gen_c[:batch_size]
                
                gen_ws = self.G_mapping(z, c)
                random_number = np.random.normal()
                if random_number < self.style_mixing_prob:
                  cutoff = np.random.uniform(low=1, high= _gen_ws.shape[1]) 
                else: 
                  cutoff = gen_ws.shape[1]
                gen_ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
                gen_img = self.G_synthesis(gen_ws)
                shape = gen_img.shape

                pl_noise = torch.randn(shape) / np.sqrt(np.prod(shape[2:]))
                output = torch.sum(gen_img * pl_noise)
                pl_grads, *_ = torch.autograd.grad(outputs=[output], inputs=[gen_ws], create_graph=True)
                pl_lengths = torch.sqrt((pl_grads ** 2).sum(dim=2).mean(dim=1))

                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                loss_Gpl = ((pl_lengths - pl_mean) ** 2 ) * self.pl_weight
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()




        loss_Dgen = 0
        if phase == 'Dmain':
                _gen_ws = self.G_mapping(z, c)
                random_number = np.random.normal()
                if random_number < self.style_mixing_prob:
                  cutoff = np.random.uniform(low=1, high= _gen_ws.shape[1]) 
                else: 
                  cutoff = _gen_ws.shape[1]
                _gen_ws[:, cutoff:] = self.G_mapping(torch.randn_like(gen_z), c, skip_w_avg_update=True)[:, cutoff:]
                gen_img = self.G_synthesis(_gen_ws)
                
                img = self.augment_pipe(gen_img)
                gen_logits = self.D(img, c)
                
                loss_Dgen = torch.log(1 + torch.exp(gen_logits))
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen.mean().mul(gain).backward()


        if phase == 'Dmain' or 'Dreg':
            
   
            if phase == 'Dreg': 
                grad_ = True
            else:
                grad_ = False
            real_img_tmp = real_img.detach().requires_grad_(grad_)
            img = self.augment_pipe(gen_img)
            real_logits = self.D(img, real_c)
            real_logits_sum = real_logits.sum()


            loss_Dreal = 0
            if phase == 'Dmain':
                loss_Dreal  = torch.log(1 + torch.exp(-real_logits))


            loss_Dr1 = 0
            if phase == 'Dreg':
                r1_grads, *_ = torch.autograd.grad(outputs=[real_logits_sum], inputs=[real_img_tmp], create_graph=True, retain_graph=True)
                r1_penalty = (r1_grads**2).sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


try:
    import pyspng
except ImportError:
    pyspng = None




class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


#I have used the Image folder dataset from the styleGAN2 +ada github file dataset.py to load in the images in the right format (removing redundant methods)

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self,path, resolution = None,max_size  = None, use_labels  = False, xflip = False, random_seed = 0):
        
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())


        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)


        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

            
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels


    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()



    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):

        return self.image_shape[0]

    @property
    def resolution(self):

        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


    


def image_grid_(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    
    else:
        print('Problems')

    
    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)



def image_grid_save_function(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    if C == 1:
        print('C = 1')
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#Interpolate: pass in w-vector that I saved during training
def return_linearly_interpolated_vectors(filename1, filename2):
    a = np.load(filename1)
    b = np.load(filename2)
    
    vector_between = (a-b)/7
    
    for i in range(1, 8): 
       interpolated_vector = b + i*vector_between
       np.savez(outdir + str(i), w=interpolated_vector)



def training_loop(
    run_dir  = '.',  training_set_kwargs= {}, data_loader_kwargs = {},G_kwargs  = {},D_kwargs = {},G_opt_kwargs = {}, D_opt_kwargs = {}, augment_kwargs = None,     
    loss_kwargs = {}, metrics = [], random_seed = 0,num_gpus= 1, rank = 0,batch_size= 4,batch_gpu= 4, ema_kimg= 10,       
    ema_rampup= None, G_reg_interval = 4,  D_reg_interval = 16, augment_p= 0,ada_target= None, ada_interval= 4,        
    ada_kimg= 500,   total_kimg= 25000, kimg_per_tick= 4, image_snapshot_ticks    = 50, network_snapshot_ticks  = 50,       
    resume_pkl= None,  cudnn_benchmark  = True, allow_tf32= False,  abort_fn= None, progress_fn= None):

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * 1 + rank)
    torch.manual_seed(random_seed * 1 + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  
    torch.backends.cudnn.allow_tf32 = allow_tf32        

    
    training_set = ImageFolderDataset(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))


    G = Generator(z_dim = 512, w_dim = 512,  img_resolution = 256, img_channels = 3, c_dim = 0).requires_grad_(False).to(device) 
    D = Discriminator(c_dim = 0, img_resolution = 256, img_channels = 3, architecture = 'resnet', channel_base = 16384, channel_max = 512, num_fp16_res = 4, conv_clamp = 256, epilogue_kwargs = {"mbstd_group_size": 8}).requires_grad_(False).to(device) 

    G_ema = copy.deepcopy(G).eval()
    augment_pipe = None

    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = AugmentPipe().train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))

    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if name is not None:
            ddp_modules[name] = module
    

    loss_ = StyleGAN2Loss(device = device,  **ddp_modules, r1_gamma = 1)
    phases = []
    #Lazy regularisation
    G_mb_ratio = 4/5
    D_mb_ratio = 16/17
    #0.0025 initial learning rate
    G_lr = 0.0025 * G_mb_ratio
    D_lr = 0.0025 * D_mb_ratio
    G_beta = [0, 0.99 * G_mb_ratio]
    D_beta = [0, 0.99 * D_mb_ratio]
    opt_G = torch.optim.Adam(params=G.parameters(), lr=G_lr, betas = G_beta,eps= 1e-08)
    opt_D = torch.optim.Adam(params=D.parameters(), lr=D_lr, betas = D_beta,eps= 1e-08)
    
    phases += [dict(name = 'Gmain', module = G, opt = opt_G, interval = 1)]
    phases += [dict(sname ='Greg', module = G, opt = opt_G, interval = 4)]
    
    phases += [dict(name ='Dmain', module = D, opt = opt_D, interval = 1)]
    phases += [dict(name = 'Dreg', module = D, opt = opt_D, interval = 16)]
    
    for phase in phases:

        phase['start_event'] = None
        phase['end_event'] = None
        if rank == 0:
            phase['start_event'] = torch.cuda.Event(enable_timing=True)
            phase['end_event'] = torch.cuda.Event(enable_timing=True)


    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        grid_size, images, labels = image_grid_(training_set=training_set)
        image_grid_save_function(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        image_grid_save_function(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        print('here')
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    batch_idx = 0

    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        
        phase_real_img, phase_real_c = next(training_set_iterator)

        data_type = torch.float32
        phase_real_img = (phase_real_img.to(device).to(data_type) / 127.5 - 1).split(batch_gpu)
        phase_real_c = phase_real_c.to(device).split(batch_gpu)
        all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
        all_gen_z = []
        for phase_gen_z in all_gen_z.split(batch_size):
            all_gen_z.append(phase_gen_z.split(batch_gpu))
            
        
        all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
        all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
        

        all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)

        all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase['interval'] != 0:
                continue

            # Initialize gradient accumulation.
            if phase['start_event'] is not None:
                phase['start_event'].record(torch.cuda.current_stream(device))
            phase['opt'].zero_grad(set_to_none=True)
            phase['module'].requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                #sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase['interval']
                loss_.accumulate_gradients(phase=phase['name'], real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=gain) #sync=sync,

            # Update weights.
            phase['module'].requires_grad_(False)
            with torch.autograd.profiler.record_function(phase['name'] + '_opt'):
                for param in phase['module'].parameters():
                    if param.grad is not None:
                        print('entered nan to num' )
                        torch.clamp(param.grad.unsqueeze(0).nansum(0), min==-1e5, max=1e5, out=param.grad)
                phase['opt'].step()
            if phase['end_event'] is not None:
                phase['end_event'].record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
   
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        torch.cuda.reset_peak_memory_stats()

    
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg

        if done:
            break


def main():



    #Same parameters used in StyleGAN2 + ADA 
    args = {
      "num_gpus": 1, "image_snapshot_ticks": 50, "network_snapshot_ticks": 50,"metrics": ["fid50k_full"],"random_seed": 0,
      "training_set_kwargs": {"path": "/home2/mxzq47/dataset/ffhq","use_labels": False,"max_size": 70000,"xflip": False,"resolution": 256},
      "data_loader_kwargs": {"pin_memory": True,"num_workers": 3,"prefetch_factor": 2},
      "G_kwargs": {"class_name": "training.networks.Generator","z_dim": 512, "w_dim": 512, "mapping_kwargs": {
          "num_layers": 8}, "synthesis_kwargs": {"channel_base": 16384,"channel_max": 512,"num_fp16_res": 4,"conv_clamp": 256}},
      "D_kwargs": {"class_name": "training.networks.Discriminator","block_kwargs": {},"mapping_kwargs": {},"epilogue_kwargs": {"mbstd_group_size": 8},"channel_base": 16384,"channel_max": 512,"num_fp16_res": 4,"conv_clamp": 256
      },
      "G_opt_kwargs": {"class_name": "torch.optim.Adam","lr": 0.0025,"betas": [0,0.99],"eps": 1e-08},
       "D_opt_kwargs": {"class_name": "torch.optim.Adam","lr": 0.0025,"betas": [0,0.99],"eps": 1e-08
      },
      "loss_kwargs": {"class_name": "training.loss.StyleGAN2Loss","r1_gamma": 1},
      "total_kimg": 25000,"batch_size": 64,"batch_gpu": 8, "ema_kimg": 20, "ema_rampup": None,"ada_target": 0.6,
      "augment_kwargs": {"class_name": "training.augment.AugmentPipe","xflip": 1,"rotate90": 1,"xint": 1,"scale": 1,"rotate": 1,"aniso": 1, "xfrac": 1, "brightness": 1, "contrast": 1, "lumaflip": 1, "hue": 1, "saturation": 1
      },
      "run_dir": "/home2/mxzq47/training-print/00008-ffhq-paper256"
    }


    os.makedirs(args['run_dir'])
    with open(os.path.join(args['run_dir'], 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    
    

    torch.multiprocessing.set_start_method('spawn')
    training_loop(rank=0, **args) 



if __name__ == "__main__":
    main() 
    
    




    
    
    
    
    
    
    
    
    
    
    
    